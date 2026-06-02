#!/usr/bin/env python3
"""Generate NTT twiddle + CDT tables for lemur-rs.

Usage
-----
    python gen_tables.py > src/tables_d256_k4.rs

KOTS multiplication routes through the CRT backend (`aux_ntt.rs`) for
the shipped parameter set because q' ≡ 17 (mod 32) has no native length-d
negacyclic NTT — only HVC NTT + CDT tables are emitted.  HVC uses the
u64 Montgomery stack since the shipped HVC modulus exceeds 2³².

Do not edit the output files by hand.
"""
import argparse
import bisect
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lemur-py'))
from sample import build_cdt  # noqa: E402


def _sigma_from_alpha(alpha):
    return float(alpha) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Named parameter set — match `lemur-py/profiles.py`.
# ---------------------------------------------------------------------------

PROFILES = {
    'd256_k4': {
        'name': 'd256_k4',
        'prefix': 'D256_K4_',
        'module_doc': (
            'd=256, k=4 (tau=20, N=1024) HVC u64 NTT twiddles and CDT '
            "table.  KOTS uses the CRT backend (aux_ntt.rs) because q' "
            "≡ 17 (mod 32) has no native length-d NTT."
        ),
        'q_kots': 827_463_089,
        'q_hvc':  4_398_046_523_393,
        'd':      256,
        'sigma':  _sigma_from_alpha(83),
        'cdt_bits': 32,
        'tailcut':  5,
        'cdt_prefix_bits': 9,
    },
}

_WIDTH_BITS = {'u32': 32, 'u64': 64}


def find_prim_root(q, d):
    """Smallest x whose (q-1)/(2d)-th power has order exactly 2d."""
    assert (q - 1) % (2 * d) == 0, f"q={q} not NTT-friendly for d={d}"
    exp = (q - 1) // (2 * d)
    for x in range(2, q):
        z = pow(x, exp, q)
        if pow(z, d, q) == q - 1:
            return z
    raise ValueError(f"No primitive root for q={q}, d={d}")


def bitrev(k, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


def mont_params(q, d, width):
    """Return `(R, q_inv, r2, inv_d_mont)` for a Montgomery stack of the
    given width (`'u32'` → R = 2^32, `'u64'` → R = 2^64).  `q_inv` is the
    width-sized `-q^{-1}` value that the Montgomery reduction expects.
    """
    bits = _WIDTH_BITS[width]
    r_big = 1 << bits
    q_inv = (r_big - pow(q, -1, r_big)) % r_big
    r2 = (r_big * r_big) % q
    inv_d_mont_val = (pow(d, -1, q) * r_big) % q
    return r_big, q_inv, r2, inv_d_mont_val


def zeta_table_mont(q, d, r_big):
    """Twiddle-factor table in Montgomery form for a caller-supplied
    R-value, bit-reversed order.  Works for any width since R is explicit.
    """
    zeta = find_prim_root(q, d)
    bits = d.bit_length() - 1
    table = [pow(zeta, bitrev(k, bits), q) for k in range(d)]
    return [(z * r_big) % q for z in table]


def cdt_prefix_bounds(cdt, prefix_bits):
    """Build a coarse bucket index for a 32-bit CDT.

    For bucket j, bounds[j] is the first index i with cdt[i] > threshold_j,
    where threshold_j is the top of the previous bucket.  For any sampled
    u whose top `prefix_bits` equal j, the exact answer lies in
    [bounds[j], bounds[j + 1]] (inclusive), so the runtime path can finish
    with only a tiny in-bucket search.
    """
    assert prefix_bits > 0
    assert len(cdt) <= 0xFFFF
    bucket_count = 1 << prefix_bits
    step = 1 << (32 - prefix_bits)
    bounds = []
    for j in range(bucket_count):
        threshold = -1 if j == 0 else j * step - 1
        idx = bisect.bisect_right(cdt, threshold)
        bounds.append(min(idx, len(cdt) - 1))
    bounds.append(len(cdt) - 1)
    return bounds


def emit(profile, out):
    """Write a profile's tables.rs-style module to `out`."""
    p = profile
    prefix = p['prefix']
    q_hvc, d = p['q_hvc'], p['d']

    cdt = build_cdt(p['sigma'], p['cdt_bits'], p['tailcut'])
    max_word = (1 << 32) - 1
    cdt = [min(v, max_word) for v in cdt]
    cdt_hi = cdt_prefix_bounds(cdt, p['cdt_prefix_bits'])

    def _emit_header():
        print("// GENERATED FILE — do not edit by hand. "
              "Run: python gen_tables.py > src/tables_d256_k4.rs", file=out)
        print(f"// Profile: {p['name']}", file=out)
        print(f"// {p['module_doc']}", file=out)
        print("// Generated from lemur-rs/gen_tables.py", file=out)
        print(file=out)

    def _emit_const_array(name, ty, values, cols):
        print(f"pub const {name}: [{ty}; {len(values)}] = [", file=out)
        for i in range(0, len(values), cols):
            chunk = values[i:i + cols]
            row = ", ".join(f"{v}" for v in chunk)
            print(f"    {row},", file=out)
        print("];", file=out)
        print(file=out)

    def _emit_const(name, ty, value):
        print(f"pub const {name}: {ty} = {value};", file=out)

    def _emit_ring_block(label, q, width):
        """Emit zetas + Montgomery constants for one modulus at a chosen width."""
        ty = width                         # 'u32' or 'u64'
        cols = 8 if width == 'u32' else 4  # u64 values print wider
        bits = _WIDTH_BITS[width]
        r_big, q_inv_v, r2_v, inv_d_v = mont_params(q, d, width)
        zetas = zeta_table_mont(q, d, r_big)

        print(
            f"/// NTT twiddle factors for q_{label}={q} in Montgomery form (R=2^{bits}).",
            file=out,
        )
        _emit_const_array(f"{prefix}{label.upper()}_ZETAS", ty, zetas, cols=cols)

        up = label.upper()
        print(f"/// -q_{label}^{{-1}} mod 2^{bits} (q_{label}={q}).", file=out)
        _emit_const(f"{prefix}{up}_Q_INV", ty, q_inv_v)
        print(f"/// R^2 mod q_{label}  (R = 2^{bits}).", file=out)
        _emit_const(f"{prefix}{up}_R2", ty, r2_v)
        print(f"/// d^{{-1}} * R mod q_{label}  (Montgomery form of d^{{-1}}).", file=out)
        _emit_const(f"{prefix}{up}_INV_D_MONT", ty, inv_d_v)
        print(file=out)

    _emit_header()
    print(f"// KOTS NTT tables intentionally omitted: profile {p['name']} "
          f"uses the CRT backend (aux_ntt.rs) at q_kots={p['q_kots']}.", file=out)
    print(file=out)
    _emit_ring_block('hvc', q_hvc, 'u64')

    sigma = p['sigma']
    print(f"/// CDT table for discrete Gaussian with sigma={sigma:.6f}, "
          f"cdt_bits={p['cdt_bits']}, tailcut={p['tailcut']}.", file=out)
    print(f"/// cdt[k] = floor(Pr[|X| <= k] * 2^{{cdt_bits}}) for k=0..{len(cdt)-1}.",
          file=out)
    _emit_const_array(f"{prefix}CDT", 'u32', cdt, cols=4)
    print(f"/// Bucket index over the top {p['cdt_prefix_bits']} bits of a 32-bit CDT sample.",
          file=out)
    print("/// For bucket j, the exact CDT answer lies in "
          f"[{prefix}CDT_HI[j], {prefix}CDT_HI[j + 1]] (inclusive).", file=out)
    _emit_const(f"{prefix}CDT_PREFIX_BITS", 'u32', p['cdt_prefix_bits'])
    _emit_const_array(f"{prefix}CDT_HI", 'u16', cdt_hi, cols=12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    emit(PROFILES['d256_k4'], sys.stdout)


if __name__ == '__main__':
    main()
