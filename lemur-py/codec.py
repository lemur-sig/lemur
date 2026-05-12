"""
codec.py — Compact serialization for LEMUR keys, parameters, and signatures.

Encoding strategy
-----------------
Public parameters (pp)
    `kots_seed(32) || hvc_seed(32) || tau_u8` — 65 B total.  The
    profile is not embedded; both sides must agree out-of-band.
    `pp_peek_tau(data)` extracts τ without instantiating a scheme.
    Matrices are re-expanded on load.

Secret key (sk)
    32-byte master seed.

Public key (pk)
    HVC commitment c in R_q^omega: packed at `logq_hvc` bits per
    coefficient (profile-dependent).

Stateful secret key (sk.state)
    Full BDS08 traversal cache, no magic bytes:
        master_seed(32) || phi_u32 || tau_u32 || k_u32 ||
        auth[tau] || keep[tau] || retain[tau] || treehash[tau-k]
    The "current slot" is the single counter `phi` — there is no separate
    next-slot field.  Labels are bit-packed at
    `dx_dig = ceil(log2(2*eta+1))` bits per coefficient with offset
    `+eta`.  Reuses the sibling-label encoding from individual
    signatures.

Individual signature
    Z (fixed-width) | sibling labels (fixed-width) | u (fixed-width).
    Path labels are omitted; the verifier reconstructs them from siblings + u
    (Babai reconstruction).

Aggregated signature
    attempt (1 byte) | Z_agg (fixed-width, N-dependent bound)
    | Babai path data (Rice or fixed-width) | sibling labels (Rice or fixed-width)
    | u (Rice or fixed-width).
    Encoding parameters are determined by compute_agg_encoding() from public
    scheme parameters and signer count N.  No magic bytes; the format is
    entirely determined by (pp, N).

Each polynomial is independently serialized and padded to a byte boundary.
Signed values use offset encoding: store (val + offset) as unsigned.
Rice coding uses: low_bits verbatim | unary high | stop bit 0 | sign bit.
All encodings are canonical: decoders reject nonzero padding, out-of-range
coefficients, and any representation that differs from what the encoder
would produce.
"""

import math
import secrets
import numpy as np
from Crypto.Hash import SHAKE256

from kots import KOTS
from hvc import HVC
from lemur import LEMUR
from profiles import LemurProfile, DEFAULT


# ---------------------------------------------------------------------------
# Fixed-width bit-packing
# ---------------------------------------------------------------------------

def poly_serial(poly, dx: int, offset: int = 0) -> bytes:
    """Pack len(poly) coefficients at dx bits each, byte-aligned."""
    mask = (1 << dx) - 1
    buf = 0
    bits = 0
    out = bytearray()
    for x in poly:
        buf |= ((int(x) + offset) & mask) << bits
        bits += dx
        while bits >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            bits -= 8
    if bits > 0:
        out.append(buf & 0xFF)
    return bytes(out)


def poly_deserial(data: bytes, dx: int, d: int, offset: int = 0) -> tuple:
    """Unpack d coefficients. Returns (list, bytes_consumed).

    Raises ValueError on nonzero padding, out-of-range values, or
    truncated input.
    """
    max_unsigned = 2 * offset if offset > 0 else (1 << dx) - 1
    mask = (1 << dx) - 1
    expected_bytes = _poly_bytes(d, dx)
    if len(data) < expected_bytes:
        raise ValueError(
            f"truncated: need {expected_bytes} bytes, got {len(data)}"
        )
    p = []
    buf = 0
    bits = 0
    i = 0
    while len(p) < d:
        while bits < dx:
            buf |= data[i] << bits
            bits += 8
            i += 1
        raw = buf & mask
        if raw > max_unsigned:
            raise ValueError(
                f"coefficient {raw - offset} out of range [-{offset}, {offset}]"
            )
        p.append(raw - offset)
        buf >>= dx
        bits -= dx
    if bits > 0 and (buf & ((1 << bits) - 1)) != 0:
        raise ValueError("nonzero padding bits in fixed-width polynomial")
    return p, i


def vec_serial(v, dx: int, offset: int = 0) -> bytes:
    """Serialize a sequence of polynomials at dx bits per coefficient."""
    return b''.join(poly_serial(p, dx, offset) for p in v)


def vec_deserial(
    data: bytes, dx: int, n: int, d: int, offset: int = 0
) -> tuple:
    """Deserialize n polynomials of degree d."""
    v = []
    i = 0
    for _ in range(n):
        p, consumed = poly_deserial(data[i:], dx, d, offset)
        v.append(p)
        i += consumed
    return v, i


def _poly_bytes(d: int, dx: int) -> int:
    """Byte count for a fixed-width polynomial."""
    return (d * dx + 7) // 8


def _fmt(n: int) -> str:
    """Human-readable byte size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.2f} MB"


# ---------------------------------------------------------------------------
# Golomb-Rice bit-packing
# ---------------------------------------------------------------------------

def poly_serial_rice(poly, rice_k: int, bound: int) -> bytes:
    """Rice-encode a polynomial, byte-aligned.

    Each coefficient x is encoded as:
      x == 0 : rice_k zero bits + stop bit (0)       [rice_k + 1 bits]
      x != 0 : low rice_k bits of |x|                [rice_k bits]
               + unary(|x| >> rice_k) ones            [hi bits]
               + stop bit (0)                         [1 bit]
               + sign bit (0 = positive, 1 = negative)[1 bit]
    Result is padded to byte boundary with zero bits.
    """
    low_mask = (1 << rice_k) - 1
    buf = 0
    bits = 0
    out = bytearray()
    for x in poly:
        x = int(x)
        ax = abs(x)
        low = ax & low_mask
        hi = ax >> rice_k
        buf |= low << bits
        bits += rice_k
        buf |= ((1 << hi) - 1) << bits
        bits += hi + 1
        if x != 0:
            if x < 0:
                buf |= 1 << bits
            bits += 1
        while bits >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            bits -= 8
    if bits > 0:
        out.append(buf & 0xFF)
    return bytes(out)


def poly_deserial_rice(
    data: bytes, d: int, rice_k: int, bound: int
) -> tuple:
    """Decode a Rice-coded polynomial. Returns (list, bytes_consumed).

    Raises ValueError on nonzero padding, out-of-range, truncated input,
    or non-canonical encoding.
    """
    max_hi = (bound >> rice_k) + 1
    p = []
    buf = 0
    bits = 0
    i = 0
    dlen = len(data)

    def _pull(need):
        nonlocal buf, bits, i
        while bits < need:
            if i >= dlen:
                raise ValueError("truncated Rice-coded polynomial")
            buf |= data[i] << bits
            bits += 8
            i += 1

    for _ in range(d):
        _pull(rice_k)
        low = buf & ((1 << rice_k) - 1)
        buf >>= rice_k
        bits -= rice_k
        hi = 0
        while True:
            _pull(1)
            bit = buf & 1
            buf >>= 1
            bits -= 1
            if bit == 0:
                break
            hi += 1
            if hi > max_hi:
                raise ValueError(
                    f"Rice unary run {hi} exceeds max {max_hi}"
                )
        ax = low | (hi << rice_k)
        if ax == 0:
            p.append(0)
        else:
            if ax > bound:
                raise ValueError(
                    f"Rice-decoded |coeff| {ax} exceeds bound {bound}"
                )
            _pull(1)
            sign = buf & 1
            buf >>= 1
            bits -= 1
            p.append(-ax if sign else ax)

    if bits > 0 and (buf & ((1 << bits) - 1)) != 0:
        raise ValueError("nonzero padding bits in Rice-coded polynomial")
    return p, i


def vec_serial_rice(v, rice_k: int, bound: int) -> bytes:
    """Rice-serialize a sequence of polynomials."""
    return b''.join(poly_serial_rice(p, rice_k, bound) for p in v)


def vec_deserial_rice(
    data: bytes, n: int, d: int, rice_k: int, bound: int
) -> tuple:
    """Rice-deserialize n polynomials of degree d."""
    v = []
    i = 0
    for _ in range(n):
        p, consumed = poly_deserial_rice(data[i:], d, rice_k, bound)
        v.append(p)
        i += consumed
    return v, i


# ---------------------------------------------------------------------------
# Tau-aware loading helpers
# ---------------------------------------------------------------------------

PP_BYTES = 65


def pp_peek_tau(data: bytes) -> int:
    """Extract tau from a pp byte string without building a scheme."""
    if len(data) != PP_BYTES:
        raise ValueError(f"pp must be {PP_BYTES} bytes, got {len(data)}")
    return int(data[64])


def make_codec(tau: int, profile: LemurProfile | None = None) -> "LemurCodec":
    """Build a fresh LemurCodec (and underlying scheme) at a custom tree
    depth.  Used by the CLI to match the codec's tau to a pp file."""
    if profile is None:
        profile = DEFAULT
    kots = KOTS(profile=profile)
    hvc = HVC(kots, tau=tau)
    scheme = LEMUR(kots, hvc)
    return LemurCodec(scheme)


# ---------------------------------------------------------------------------
# Encoding parameter computation
# ---------------------------------------------------------------------------

def compute_agg_encoding(
    kots: KOTS, hvc: HVC, alpha_w: int, n_signers: int
) -> dict:
    """Compute optimal encoding parameters for aggregated signatures.

    Returns a dict consumed by LemurCodec.agg_sig_encode / agg_sig_decode.
    All parameters are derived from public scheme constants and n_signers,
    so both encoder and decoder produce identical results.
    """
    eta = hvc.eta
    d = kots.d

    # --- sigma estimates (CLT over ring convolutions) ---
    # Aggregated label/u: sum of N ternary(alpha_w) * decomposition([-eta,eta])
    var_digit = eta * (eta + 1) / 3
    sigma_label = math.sqrt(n_signers * alpha_w * var_digit)

    # Z individual: H*S where H ternary weight alpha_h, S Gaussian(sigma).
    # Use the sampler's sigma (actual per-coefficient stddev of S), not
    # the paper's width parameter kots.alpha — they differ by a factor
    # of sqrt(2*pi) and conflating them overestimates the Z bound by
    # ~2.5x, which drives the Rice-vs-fixed choice the wrong way.
    sigma_z_ind = kots.sampler.sigma * math.sqrt(
        1 + (kots.k - kots.ell) * kots.alpha_h
    )
    # Z_agg: sum of N ternary(alpha_w) * Z^i
    sigma_zagg = sigma_z_ind * math.sqrt(n_signers * alpha_w)

    # Babai-encoded path label coefficients: ~sigma_label / (2*eta)
    sigma_babai = sigma_label / (2 * eta)

    def _rice_params(sigma, max_bound):
        """Pick the encoding that minimises expected bits per coefficient.

        Models each scheme's per-coefficient cost and returns whichever
        is cheaper:
          * fixed-width: `dx_fixed = bit_length(2*max_bound)` bits/coef
          * Rice at parameter k: `k + mu/2^k + 2` bits/coef
            (folded-Gaussian mean `mu = 0.7979*sigma`, plus stop+sign bits)

        Searches `k` over `[0, dx_fixed]`, which brackets the minimum
        because the cost function is unimodal in k.  Returns
        `(mode, rice_k, bound, dx_fixed)`.  No fixed `sigma<2` cutoff
        and no Rice/fixed hysteresis — the formulas decide.
        """
        dx_fixed = (2 * max_bound).bit_length()
        mu = 0.7979 * max(float(sigma), 0.0)

        best_k = 0
        best_bits = mu + 2.0  # k=0 cost
        for k in range(1, dx_fixed + 1):
            bits = k + 2.0 + mu / float(1 << k)
            if bits < best_bits:
                best_bits = bits
                best_k = k

        if best_bits < dx_fixed:
            return 'rice', best_k, max_bound, dx_fixed
        return 'fixed', None, max_bound, dx_fixed

    # Z_agg: N-dependent fixed-width (small component, simple encoding)
    n_zagg_coeffs = kots.ell * kots.m * d
    c_zagg = math.sqrt(2 * math.log(2 * n_zagg_coeffs * 256))
    zagg_bound = min(int(math.ceil(c_zagg * sigma_zagg)), kots.beta_sigma)
    zagg_dx = max(1, (2 * zagg_bound).bit_length())

    # Babai path labels
    bm, brk, bb, bdx = _rice_params(sigma_babai, hvc.beta_encode)

    # Aggregated sibling labels and u
    am, ark, ab, adx = _rice_params(sigma_label, hvc.beta_agg)

    return {
        'n_signers': n_signers,
        'zagg_bound': zagg_bound,
        'zagg_dx': zagg_dx,
        'babai_mode': bm,
        'babai_rice_k': brk,
        'babai_bound': bb,
        'babai_dx': bdx,
        'agg_mode': am,
        'agg_rice_k': ark,
        'agg_bound': ab,
        'agg_dx': adx,
    }


# ---------------------------------------------------------------------------
# LemurCodec
# ---------------------------------------------------------------------------

class LemurCodec:
    """Serialization for Lemur keys, parameters, and signatures.

    Encoding format is canonical and unambiguous: no magic bytes, no
    encoding options.  Individual signatures always use Babai (path labels
    omitted).  Aggregated signature encoding is determined by the signer
    count N via compute_agg_encoding().
    """

    def __init__(self, scheme: LEMUR):
        self.scheme = scheme
        kots = scheme.kots
        hvc = scheme.hvc
        d = kots.d

        self.dx_pk = hvc.logq
        self.dx_z = kots.bits_z
        self.dx_dig = hvc.bits_dig
        self.off_z = kots.beta_z
        self.off_dig = hvc.eta

        self.PP_BYTES = 65
        self.PK_BYTES = hvc.omega * _poly_bytes(d, self.dx_pk)
        self.SK_BYTES = 32

        # Stateful secret key format (no magic bytes): the .state file is
        # always a BDS08 traversal cache, written as
        #   master_seed(32) || phi_u32 || tau_u32 || k_u32 || body
        # where body = auth[tau] + keep[tau] + retain[tau] + treehash[tau-k].
        # The "current slot" is the single counter `bds["phi"]`; there is
        # no separate next_slot field.

        # One HVC label is omega*kappa polynomials at dx_dig bits/coeff,
        # each padded to a byte boundary.
        self.LABEL_BYTES = hvc.omega * hvc.kappa * _poly_bytes(d, self.dx_dig)
        self.SK_STATE_HEADER_BYTES = 32 + 4 + 4 + 4

    # -----------------------------------------------------------------------
    # Setup / keygen / sign
    # -----------------------------------------------------------------------

    @staticmethod
    def _slot_seed(master_seed: bytes, t: int) -> bytes:
        return SHAKE256.new(
            master_seed + b'slot' + t.to_bytes(4, 'little')
        ).read(32)

    def setup(
        self,
        kots_seed: bytes | None = None,
        hvc_seed: bytes | None = None,
    ) -> tuple:
        if kots_seed is None:
            kots_seed = secrets.token_bytes(32)
        if hvc_seed is None:
            hvc_seed = secrets.token_bytes(32)
        pp = self.scheme.setup(kots_seed, hvc_seed)
        return pp, kots_seed, hvc_seed

    def keygen(self, pp: tuple, master_seed: bytes | None = None) -> tuple:
        if master_seed is None:
            master_seed = secrets.token_bytes(32)
        sk_seed, sk_state, pk = self.scheme.keygen(pp, master_seed)
        return sk_seed, sk_state, pk

    def sign_seed(self, pp: tuple, sk_seed: bytes, t: int, m: bytes) -> tuple:
        return self.scheme.sign_seed(pp, sk_seed, t, m)

    def sign_stateful(
        self, pp: tuple, sk_state: dict, m: bytes, t: int | None = None
    ) -> tuple:
        return self.scheme.sign_stateful(pp, sk_state, m, t)

    # -----------------------------------------------------------------------
    # Public parameters
    # -----------------------------------------------------------------------

    def pp_encode(self, kots_seed: bytes, hvc_seed: bytes) -> bytes:
        """Encode pp as `kots_seed(32) || hvc_seed(32) || tau_u8` (65 bytes)."""
        assert len(kots_seed) == 32 and len(hvc_seed) == 32
        tau = int(self.scheme.hvc.tau)
        if not (0 < tau <= 0xFF):
            raise ValueError(f"tau={tau} does not fit in u8")
        return bytes(kots_seed) + bytes(hvc_seed) + bytes([tau])

    def pp_decode(self, data: bytes) -> tuple:
        """Decode pp.  Rejects files whose tau does not match this codec's
        scheme.  Use `pp_peek_tau` + `make_codec` to build a correctly-
        configured codec before calling this if the tree depth is not
        known ahead of time."""
        if len(data) != self.PP_BYTES:
            raise ValueError(
                f"pp must be {self.PP_BYTES} bytes, got {len(data)}"
            )
        tau_in = data[64]
        scheme_tau = int(self.scheme.hvc.tau)
        if tau_in != scheme_tau:
            raise ValueError(
                f"pp tau={tau_in} does not match scheme tau={scheme_tau}"
            )
        return self.setup(bytes(data[:32]), bytes(data[32:64]))

    # -----------------------------------------------------------------------
    # Secret key
    # -----------------------------------------------------------------------

    def sk_encode(self, sk: bytes) -> bytes:
        assert len(sk) == 32
        return sk

    def sk_decode(self, data: bytes) -> bytes:
        if len(data) != self.SK_BYTES:
            raise ValueError(
                f"seed secret key must be {self.SK_BYTES} bytes, got {len(data)}"
            )
        return bytes(data)

    # ------ Internal label helpers for the stateful sk (BDS) format ------

    def _label_encode(self, label: np.ndarray) -> bytes:
        """Serialize one HVC label at dx_dig bits per coeff (= sibling-label format)."""
        return vec_serial(label, self.dx_dig, self.off_dig)

    def _label_decode_at(self, data: bytes, pos: int) -> tuple:
        """Decode one label starting at `pos`.  Returns (label_array, new_pos)."""
        hvc = self.scheme.hvc
        n_rows = hvc.omega * hvc.kappa
        v, consumed = vec_deserial(
            data[pos:], self.dx_dig, n_rows, hvc.d, self.off_dig
        )
        return np.array(v, dtype=np.int64), pos + consumed

    # ------ Stateful secret key (BDS state file, no magic bytes) ------

    def sk_state_encode(self, sk_state: dict) -> bytes:
        """Serialize a stateful sk as a BDS08 traversal cache.

        Wire layout (no magic bytes):
            master_seed(32) || phi_u32 || tau_u32 || k_u32 ||
            auth[tau] || keep[tau] || retain[tau] || treehash[tau-k]

        `auth` is a dense sequence of `tau` labels.  `keep` is per-level
        `u8 present [+ label]`.  `retain` is per-level `u16 count` followed
        by `count` labels in pop-front order.  `treehash` is one record per
        level in `[0, tau-k)`: `u8 finished || u32 leaf_index ||
        u32 leaves_remaining || u8 node_present [+ label] || u16 stack_count
        || stack_count × (u8 height + label)`.  Each label is bit-packed at
        `dx_dig = 6` bits per coefficient with offset `+eta`, reusing the
        sibling-label encoding from individual signatures.
        """
        master_seed = sk_state["master_seed"]
        if len(master_seed) != 32:
            raise ValueError("stateful secret key master_seed must be 32 bytes")
        bds = sk_state.get("bds")
        if bds is None:
            raise ValueError("stateful sk missing BDS state")

        h = int(bds["H"])
        k = int(bds["K"])
        phi = int(bds["phi"])
        if not (0 <= phi <= 0xFFFFFFFF):
            raise ValueError("stateful secret key phi out of range")
        if not (0 <= h <= 0xFFFFFFFF and 0 <= k <= 0xFFFFFFFF):
            raise ValueError("stateful secret key tau/k out of range")

        out = bytearray()
        out += bytes(master_seed)
        out += phi.to_bytes(4, "little")
        out += h.to_bytes(4, "little")
        out += k.to_bytes(4, "little")

        # auth: h labels, always dense.
        auth = bds["auth"]
        for level in range(h):
            out += self._label_encode(auth[level])

        # keep: sparse per-level, iterate all h levels with presence flag.
        keep_map = bds["keep"]
        for level in range(h):
            entry = keep_map.get(level)
            if entry is None:
                out += b"\x00"
            else:
                out += b"\x01"
                out += self._label_encode(entry)

        # retain: iterate all h levels with u16 count; non-retain levels
        # have count=0, retain levels have count >= 0 up to the FIFO size.
        retain_map = bds["retain"]
        for level in range(h):
            labels = retain_map.get(level, [])
            count = len(labels)
            if count > 0xFFFF:
                raise ValueError(f"retain[{level}] count {count} > 65535")
            out += count.to_bytes(2, "little")
            for lab in labels:
                out += self._label_encode(lab)

        # treehash: (h - k) records at levels 0..h-k.
        treehash_map = bds["treehash"]
        th_count = max(h - k, 0)
        for level in range(th_count):
            th = treehash_map[level]
            out += b"\x01" if th.finished else b"\x00"
            li = int(th.leaf_index)
            if li < 0:
                li = 0
            out += li.to_bytes(4, "little")
            out += int(th.leaves_remaining).to_bytes(4, "little")
            if th.node is not None:
                out += b"\x01"
                out += self._label_encode(th.node)
            else:
                out += b"\x00"
            stack_count = len(th.stack)
            if stack_count > 0xFFFF:
                raise ValueError(
                    f"treehash[{level}] stack too large ({stack_count})"
                )
            out += stack_count.to_bytes(2, "little")
            for (stack_h, stack_lab) in th.stack:
                if not (0 <= int(stack_h) <= 0xFF):
                    raise ValueError(
                        f"treehash[{level}] stack entry height out of range"
                    )
                out += int(stack_h).to_bytes(1, "little")
                out += self._label_encode(stack_lab)

        return bytes(out)

    def sk_state_decode(self, data: bytes) -> dict:
        """Decode a stateful sk from the on-disk BDS-state format."""
        if len(data) < self.SK_STATE_HEADER_BYTES:
            raise ValueError("stateful secret key header truncated")
        master_seed = bytes(data[0:32])
        phi = int.from_bytes(data[32:36], "little")
        h = int.from_bytes(data[36:40], "little")
        k = int.from_bytes(data[40:44], "little")

        if h > 32:
            raise ValueError(f"stateful sk: tau={h} out of supported range")
        if k > h:
            raise ValueError(f"stateful sk: invalid k={k} (tau={h})")
        if phi > (1 << h):
            raise ValueError(
                f"stateful sk: phi={phi} out of range for tau={h}"
            )

        hvc_tau = self.scheme.hvc.tau
        if h != hvc_tau:
            raise ValueError(
                f"stateful sk: state tau={h} does not match scheme tau={hvc_tau}"
            )

        pos = self.SK_STATE_HEADER_BYTES

        # auth
        auth = [None] * h
        for level in range(h):
            lab, pos = self._label_decode_at(data, pos)
            auth[level] = lab

        # keep
        keep: dict = {}
        for level in range(h):
            if pos >= len(data):
                raise ValueError("stateful sk: truncated (keep section)")
            flag = data[pos]
            pos += 1
            if flag == 1:
                if level >= h - 1:
                    raise ValueError(
                        f"stateful sk: keep entry at level {level} >= h-1={h - 1}"
                    )
                lab, pos = self._label_decode_at(data, pos)
                keep[level] = lab
            elif flag != 0:
                raise ValueError(
                    f"stateful sk: invalid keep flag {flag} at level {level}"
                )

        # retain
        retain: dict = {
            level: [] for level in range(max(h - k, 0), max(h - 1, 0))
        }
        for level in range(h):
            if pos + 2 > len(data):
                raise ValueError("stateful sk: truncated (retain section)")
            count = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
            if count == 0:
                continue
            in_range = (h - k) <= level < (h - 1)
            if not in_range:
                raise ValueError(
                    f"stateful sk: retain entries at non-retain level {level}"
                )
            for _ in range(count):
                lab, pos = self._label_decode_at(data, pos)
                retain[level].append(lab)

        # treehash
        from hvc import TreehashInst  # local import to avoid cycles at module load
        treehash: dict = {}
        th_count = max(h - k, 0)
        for level in range(th_count):
            th = TreehashInst(level)
            if pos >= len(data):
                raise ValueError("stateful sk: truncated (treehash section)")
            finished = data[pos]
            pos += 1
            if finished not in (0, 1):
                raise ValueError(
                    f"stateful sk: invalid treehash finished flag {finished}"
                )
            if pos + 8 > len(data):
                raise ValueError("stateful sk: truncated (treehash counters)")
            leaf_index = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            leaves_remaining = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            if pos >= len(data):
                raise ValueError("stateful sk: truncated (treehash node flag)")
            node_flag = data[pos]
            pos += 1
            if node_flag == 1:
                lab, pos = self._label_decode_at(data, pos)
                th.node = lab
            elif node_flag != 0:
                raise ValueError(
                    f"stateful sk: invalid treehash node flag {node_flag}"
                )
            if pos + 2 > len(data):
                raise ValueError("stateful sk: truncated (treehash stack count)")
            stack_count = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
            stack = []
            for _ in range(stack_count):
                if pos >= len(data):
                    raise ValueError(
                        "stateful sk: truncated (treehash stack entry height)"
                    )
                stack_h = data[pos]
                pos += 1
                lab, pos = self._label_decode_at(data, pos)
                stack.append((int(stack_h), lab))
            th.stack = stack
            th.leaf_index = leaf_index
            th.leaves_remaining = leaves_remaining
            th.finished = bool(finished)
            treehash[level] = th

        if pos != len(data):
            raise ValueError(
                f"stateful sk: trailing bytes (consumed {pos}, total {len(data)})"
            )

        bds = {
            "H": h,
            "K": k,
            "phi": phi,
            "auth": auth,
            "keep": keep,
            "retain": retain,
            "treehash": treehash,
        }
        return self.scheme.make_stateful_sk(master_seed, bds)

    # -----------------------------------------------------------------------
    # Public key
    # -----------------------------------------------------------------------

    def pk_encode(self, pk: np.ndarray) -> bytes:
        return vec_serial(pk, self.dx_pk)

    def pk_decode(self, data: bytes) -> np.ndarray:
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        v, _ = vec_deserial(data, self.dx_pk, hvc.omega, kots.d)
        return np.array(v, dtype=np.int64)

    # -----------------------------------------------------------------------
    # Individual signature (always Babai: path labels omitted)
    # -----------------------------------------------------------------------

    def sig_encode(self, sigma: tuple) -> bytes:
        """Encode individual signature.  Path labels omitted (Babai)."""
        Z, d_open = sigma
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        _, sibling_labels, u = d_open
        out = vec_serial(Z.reshape(-1, kots.d), self.dx_z, self.off_z)
        for s in sibling_labels:
            out += vec_serial(s, self.dx_dig, self.off_dig)
        out += vec_serial(u, self.dx_dig, self.off_dig)
        return out

    def sig_decode(self, data: bytes, pp: tuple, t: int) -> tuple:
        """Decode individual signature, reconstructing path labels."""
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        d = kots.d
        ell, m = kots.ell, kots.m
        v, l = vec_deserial(data, self.dx_z, ell * m, d, self.off_z)
        Z = np.array(v, dtype=np.int64).reshape(ell, m, d)
        off = l
        n_label = hvc.omega * hvc.kappa
        sibling_labels = []
        for _ in range(hvc.tau):
            v, l = vec_deserial(
                data[off:], self.dx_dig, n_label, d, self.off_dig
            )
            sibling_labels.append(np.array(v, dtype=np.int64))
            off += l
        n_u = hvc.rho * hvc.nu * hvc.kappa_prime
        v, l = vec_deserial(data[off:], self.dx_dig, n_u, d, self.off_dig)
        u = np.array(v, dtype=np.int64)
        off += l
        if off != len(data):
            raise ValueError(
                f"trailing bytes: consumed {off}, total {len(data)}"
            )
        _, hvc_pp = pp
        path_labels = hvc.reconstruct_path_labels_ind(
            hvc_pp, t, sibling_labels, u
        )
        return Z, (path_labels, sibling_labels, u)

    def sig_bytes(self) -> int:
        """Byte count for an individual signature."""
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        d = kots.d
        pb_z = _poly_bytes(d, self.dx_z)
        pb_dig = _poly_bytes(d, self.dx_dig)
        n_label = hvc.omega * hvc.kappa
        n_u = hvc.rho * hvc.nu * hvc.kappa_prime
        return (
            kots.ell * kots.m * pb_z
            + hvc.tau * n_label * pb_dig
            + n_u * pb_dig
        )

    # -----------------------------------------------------------------------
    # Aggregated signature (Babai path labels + Rice/fixed-width components)
    # -----------------------------------------------------------------------

    def _agg_enc(self, n_signers: int) -> dict:
        """Compute encoding params for n_signers."""
        return compute_agg_encoding(
            self.scheme.kots, self.scheme.hvc,
            self.scheme.alpha_w, n_signers,
        )

    def agg_sig_encode(self, sigma_agg: tuple, n_signers: int) -> bytes:
        """Encode aggregated signature with optimal encoding for n_signers."""
        Z_agg, d_agg, attempt = sigma_agg
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        d = kots.d
        path_labels, sibling_labels, u = d_agg
        enc = self._agg_enc(n_signers)

        out = bytes([attempt])

        # Z_agg: N-dependent fixed-width
        out += vec_serial(
            Z_agg.reshape(-1, d), enc['zagg_dx'], enc['zagg_bound']
        )

        # Babai-encoded path labels
        for label in path_labels:
            hint = hvc._proj_label(label)
            encoded = hvc.babai_encode_label(label, hint)
            for a_star, alphas in encoded:
                if enc['babai_mode'] == 'rice':
                    out += poly_serial_rice(
                        a_star, enc['babai_rice_k'], enc['babai_bound']
                    )
                    out += vec_serial_rice(
                        alphas, enc['babai_rice_k'], enc['babai_bound']
                    )
                else:
                    out += poly_serial(
                        a_star, enc['babai_dx'], enc['babai_bound']
                    )
                    out += vec_serial(
                        alphas, enc['babai_dx'], enc['babai_bound']
                    )

        # Sibling labels
        for s in sibling_labels:
            if enc['agg_mode'] == 'rice':
                out += vec_serial_rice(
                    s, enc['agg_rice_k'], enc['agg_bound']
                )
            else:
                out += vec_serial(s, enc['agg_dx'], enc['agg_bound'])

        # u
        if enc['agg_mode'] == 'rice':
            out += vec_serial_rice(
                u, enc['agg_rice_k'], enc['agg_bound']
            )
        else:
            out += vec_serial(u, enc['agg_dx'], enc['agg_bound'])

        return out

    def agg_sig_decode(
        self, data: bytes, pp: tuple, t: int, n_signers: int
    ) -> tuple:
        """Decode aggregated signature."""
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        d = kots.d
        enc = self._agg_enc(n_signers)

        attempt = data[0]
        off = 1

        # Z_agg
        ell, m = kots.ell, kots.m
        v, l = vec_deserial(
            data[off:], enc['zagg_dx'], ell * m, d, enc['zagg_bound']
        )
        Z_agg = np.array(v, dtype=np.int64).reshape(ell, m, d)
        off += l

        # Babai path labels
        path_encoded = []
        for _ in range(hvc.tau):
            level = []
            for _ in range(hvc.omega):
                if enc['babai_mode'] == 'rice':
                    a_list, la = poly_deserial_rice(
                        data[off:], d,
                        enc['babai_rice_k'], enc['babai_bound'],
                    )
                    off += la
                    al_list, la = vec_deserial_rice(
                        data[off:], hvc.kappa - 1, d,
                        enc['babai_rice_k'], enc['babai_bound'],
                    )
                    off += la
                else:
                    a_list, la = poly_deserial(
                        data[off:], enc['babai_dx'], d, enc['babai_bound']
                    )
                    off += la
                    al_list, la = vec_deserial(
                        data[off:], enc['babai_dx'],
                        hvc.kappa - 1, d, enc['babai_bound'],
                    )
                    off += la
                a_star = np.array(a_list, dtype=np.int64)
                alphas = np.array(al_list, dtype=np.int64)
                level.append((a_star, alphas))
            path_encoded.append(level)

        # Sibling labels
        n_label = hvc.omega * hvc.kappa
        sibling_labels = []
        for _ in range(hvc.tau):
            if enc['agg_mode'] == 'rice':
                v, l = vec_deserial_rice(
                    data[off:], n_label, d,
                    enc['agg_rice_k'], enc['agg_bound'],
                )
            else:
                v, l = vec_deserial(
                    data[off:], enc['agg_dx'], n_label, d, enc['agg_bound']
                )
            sibling_labels.append(np.array(v, dtype=np.int64))
            off += l

        # u
        n_u = hvc.rho * hvc.nu * hvc.kappa_prime
        if enc['agg_mode'] == 'rice':
            v, l = vec_deserial_rice(
                data[off:], n_u, d,
                enc['agg_rice_k'], enc['agg_bound'],
            )
        else:
            v, l = vec_deserial(
                data[off:], enc['agg_dx'], n_u, d, enc['agg_bound']
            )
        u = np.array(v, dtype=np.int64)
        off += l

        if off != len(data):
            raise ValueError(
                f"trailing bytes: consumed {off}, total {len(data)}"
            )

        # Reconstruct path labels from Babai data + hints
        _, hvc_pp = pp
        path_labels = hvc.reconstruct_path_labels_agg(
            hvc_pp, t, path_encoded, sibling_labels, u
        )
        return Z_agg, (path_labels, sibling_labels, u), attempt

    # -----------------------------------------------------------------------
    # Size summary
    # -----------------------------------------------------------------------

    def sizes(self, n_signers: int = 1024) -> dict[str, int]:
        kots = self.scheme.kots
        hvc = self.scheme.hvc
        d = kots.d
        ell, m = kots.ell, kots.m
        tau = hvc.tau
        pb_z = _poly_bytes(d, self.dx_z)
        pb_dig = _poly_bytes(d, self.dx_dig)
        n_label = hvc.omega * hvc.kappa
        n_u = hvc.rho * hvc.nu * hvc.kappa_prime

        enc = self._agg_enc(n_signers)
        pb_zagg = _poly_bytes(d, enc['zagg_dx'])

        eta = hvc.eta
        alpha_w = self.scheme.alpha_w
        var_digit = eta * (eta + 1) / 3
        sigma_label = math.sqrt(n_signers * alpha_w * var_digit)
        sigma_babai = sigma_label / (2 * eta)

        def _rice_poly_bytes(sigma, rice_k):
            """Expected bytes per polynomial under Rice coding."""
            mean_hi = 0.7979 * sigma / (1 << rice_k)
            bits_per_coeff = rice_k + mean_hi + 1 + 1
            return int(math.ceil(d * bits_per_coeff / 8))

        if enc['babai_mode'] == 'rice':
            babai_desc = f"Rice k={enc['babai_rice_k']}"
            pb_babai = _rice_poly_bytes(sigma_babai, enc['babai_rice_k'])
        else:
            babai_desc = f"fixed {enc['babai_dx']}b"
            pb_babai = _poly_bytes(d, enc['babai_dx'])
        babai_total = tau * hvc.omega * hvc.kappa * pb_babai

        if enc['agg_mode'] == 'rice':
            agg_desc = f"Rice k={enc['agg_rice_k']}"
            pb_agg = _rice_poly_bytes(sigma_label, enc['agg_rice_k'])
        else:
            agg_desc = f"fixed {enc['agg_dx']}b"
            pb_agg = _poly_bytes(d, enc['agg_dx'])
        sib_total = tau * n_label * pb_agg
        u_total = n_u * pb_agg

        agg_total = 1 + ell * m * pb_zagg + babai_total + sib_total + u_total
        label = "~" if enc['babai_mode'] == 'rice' or enc['agg_mode'] == 'rice' else ""

        # Fresh BDS state (phi=0) size: header + tau auth labels + tau keep
        # presence bytes (all 0) + tau retain count headers + (2^k - k - 1)
        # retain labels (pre-computed right-sibling queue) + (tau - k)
        # treehash records each holding one completed node.
        from hvc import bds_choose_k
        bds_k = bds_choose_k(tau)
        n_retain_labels = max((1 << bds_k) - bds_k - 1, 0) if tau >= 2 else 0
        n_treehash = max(tau - bds_k, 0)
        lb = self.LABEL_BYTES
        fresh_state_bytes = (
            self.SK_STATE_HEADER_BYTES
            + tau * lb                  # auth[tau]
            + tau                       # keep: tau presence flags, all 0
            + 2 * tau                   # retain: tau u16 count headers
            + n_retain_labels * lb      # retain labels
            + n_treehash * (12 + lb)    # treehash fixed fields + completed node
        )

        return {
            "pp (seeds + tau)": self.PP_BYTES,
            "sk (master seed)": self.SK_BYTES,
            "sk.state (fresh BDS)": fresh_state_bytes,
            "pk (HVC commitment)": self.PK_BYTES,
            f"individual sig": self.sig_bytes(),
            "  Z (KOTS sig)": ell * m * pb_z,
            "  sibling labels": tau * n_label * pb_dig,
            "  u": n_u * pb_dig,
            f"aggregated sig (N={n_signers}, {label}{_fmt(agg_total)})": agg_total,
            f"  Z_agg ({enc['zagg_dx']}b, bound={enc['zagg_bound']})":
                1 + ell * m * pb_zagg,
            f"  Babai path ({babai_desc})": babai_total,
            f"  sibling labels ({agg_desc})": sib_total,
            f"  u ({agg_desc})": u_total,
        }
