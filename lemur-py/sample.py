"""
sample.py — Deterministic polynomial samplers for Lemur / KOTS / HVC.

All samplers consume from a caller-supplied SHAKE XOF so that the
calling code controls domain separation and seed derivation.

Functions
---------
GaussianSampler                  discrete Gaussian sampler profile
build_cdt(sigma, cdt_bits, tailcut)   precompute CDT table for discrete Gaussian
xof_uniform_poly(xof, q, d)      uniform in Z_q^d via rejection sampling
xof_gauss_poly(xof, cdt, cdt_bits, d) discrete Gaussian D_sigma via CDT lookup
xof_ternary_poly(xof, weight, d) fixed-weight ternary in {-1,0,1}^d
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianSampler:
    """Discrete-Gaussian sampler profile.

    `sigma` is the *sampling standard deviation* of D_sigma over Z.  The
    Lemur spec's formal Gaussian definition uses the paper symbol alpha
    as a width parameter with sigma = alpha/sqrt(2*pi); callers (e.g.
    profiles.py) convert and pass sigma here.  The sampler itself is
    agnostic to the convention.

    cdt_bits=32 is a byte-aligned implementation choice for the
    discrete-Gaussian Renyi-divergence bound: the 32-bit read supplies
    31 bits of CDF comparison plus one sign bit (LSB).  For the shipped
    D256_K4 profile, parameter/Lemur-DGS-Prec_TailCut.py computes
    prec_re=28, so the implementation keeps a 3-bit comparison margin.
    tailcut=5 sigma is sufficient by the same analysis
    (`tc_re = 5`).
    """

    sigma: float
    cdt_bits: int = 32
    tailcut: int = 5

    @property
    def cdt_bytes(self) -> int:
        if self.cdt_bits % 8 != 0:
            raise ValueError("cdt_bits must be byte-aligned")
        return self.cdt_bits // 8


def build_cdt(sigma: float, cdt_bits: int, tailcut: int = 14) -> list[int]:
    """Build a CDT table for the absolute value of a discrete Gaussian.

    Returns a list of (cdt_bits+1)-bit integers:
        cdt[k] = floor(Pr[|X| <= k] * 2^cdt_bits)   for k = 0, 1, ..., floor(tailcut*sigma)+1
    with the last entry clamped to 2^cdt_bits so that binary search always terminates.

    Uses mpmath when available; otherwise falls back to decimal arithmetic from
    the standard library. The tailcut gives Pr[|X| > tailcut*sigma] <
    2*exp(-tailcut^2/2), which is < 2^{-140} at tailcut=14.
    """
    mp_prec = max(2 * cdt_bits + 64, 256)
    kmax = int(tailcut * sigma) + 1
    try:
        from mpmath import mp, mpf, exp, fsum  # type: ignore[import]

        mp.prec = mp_prec
        sigma2 = mpf(sigma) ** 2
        weights = [exp(-mpf(k) ** 2 / (2 * sigma2)) for k in range(kmax + 1)]
        total = weights[0] + 2 * fsum(weights[1:])
        scale = mpf(1 << cdt_bits)
        cdt = []
        acc = mpf(0)
        for k, w in enumerate(weights):
            acc += (w if k == 0 else 2 * w) / total
            cdt.append(int(acc * scale))
        cdt[-1] = 1 << cdt_bits
        return cdt
    except ModuleNotFoundError:
        from decimal import Decimal, ROUND_FLOOR, localcontext

        # Convert bit precision to a conservative number of decimal digits.
        dec_prec = max(int(mp_prec * 0.31) + 20, 100)
        with localcontext() as ctx:
            ctx.prec = dec_prec
            sigma_dec = Decimal(str(sigma))
            sigma2 = sigma_dec * sigma_dec
            weights = [
                (-(Decimal(k * k) / (Decimal(2) * sigma2))).exp()
                for k in range(kmax + 1)
            ]
            total = weights[0] + Decimal(2) * sum(weights[1:], Decimal(0))
            scale = Decimal(1 << cdt_bits)
            cdt = []
            acc = Decimal(0)
            for k, w in enumerate(weights):
                acc += (w if k == 0 else Decimal(2) * w) / total
                cdt.append(
                    int((acc * scale).to_integral_value(rounding=ROUND_FLOOR))
                )
            cdt[-1] = 1 << cdt_bits
            return cdt


def xof_uniform_poly(xof, q: int, d: int) -> np.ndarray:
    """Sample a uniform polynomial in [0, q)^d from a SHAKE XOF stream.

    Uses rejection sampling: draws ceil(log2(q))-bit integers and discards
    those >= q.  Matches the ML-KEM / ML-DSA matrix-expansion convention.

    Reads are batched: we request a single large block from the XOF up
    front (enough for ~2x the expected number of draws) and service
    rejections out of the buffer; a small fallback read handles the
    tail.  SHAKE buffers one rate block internally (168 B for SHAKE128,
    136 B for SHAKE256), so a single `.read(n)` still triggers exactly
    `ceil(n / rate)` Keccak-f permutations — batching saves the
    per-`read()` Python call overhead (dispatch, small allocations,
    memoryview creation), not Keccak work.  Because each caller hands
    in a fresh ephemeral XOF per poly, over-reading is harmless: the
    XOF is discarded once the poly is full.
    """
    bits_q = (q - 1).bit_length()
    byte_q = (bits_q + 7) // 8
    mask_q = (1 << bits_q) - 1
    # Pull enough bytes up front for ~2 * d draws (handles the 23 %
    # rejection rate of Q_KOTS and the 3 % rejection rate of Q_HVC
    # with ample margin).
    buf = xof.read(2 * d * byte_q)
    i = 0
    r = []
    while len(r) < d:
        if i + byte_q > len(buf):
            buf = buf[i:] + xof.read(d * byte_q)
            i = 0
        x = int.from_bytes(buf[i:i + byte_q], 'little') & mask_q
        i += byte_q
        if x < q:
            r.append(x)
    return np.array(r, dtype=np.int64)


def xof_gauss_poly(xof, cdt: list[int], cdt_bits: int, d: int) -> np.ndarray:
    """Sample a discrete Gaussian polynomial using CDT lookup.

    For each coefficient:
      1. Read cdt_bits bits (cdt_bits//8 bytes) from xof as a little-endian integer u.
      2. Extract sign bit, remove it from u.
      3. Binary-search cdt for the smallest k with cdt[k] > u.
      4. return +k or -k, depending on the sign bit.

    cdt must be a table built by build_cdt(sigma, cdt_bits), with
    cdt[-1] = 2^cdt_bits.

    The XOF read is batched: a single `.read(d * cdt_bytes)` call pulls
    the entire per-poly budget in one go, amortising per-call
    XofReader overhead.  Byte consumption (and therefore the produced
    coefficient stream) is identical to the unbatched form.
    """
    if cdt_bits % 8 != 0:
        raise ValueError("cdt_bits must be byte-aligned")
    lam_bytes = cdt_bits // 8
    buf = xof.read(d * lam_bytes)
    r = []
    for c in range(d):
        u = int.from_bytes(buf[c * lam_bytes:(c + 1) * lam_bytes], 'little')
        sign = u & 1
        u ^= sign
        # binary search: smallest k with cdt[k] > u
        (lo, hi) = (0, len(cdt) - 1)
        while lo < hi:
            mid = (lo + hi) // 2
            if cdt[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        k = lo
        r.append(-k if sign else k)
    return np.array(r, dtype=np.int64)


def xof_ternary_poly(xof, weight: int, d: int) -> np.ndarray:
    """Sample a fixed-weight ternary polynomial from a SHAKE XOF.

    Reads 9 bytes per batch: 8 position bytes + 1 sign-bit byte (bit per
    candidate).  Retries on collision or out-of-range position.
    Works correctly for d <= 256.
    """
    poly = [0] * d
    i = 8
    while weight > 0:
        if i == 8:
            i = 0
            raw = xof.read(9)
        pos = raw[i]
        if pos < d and poly[pos] == 0:
            poly[pos] = 1 - 2 * ((raw[8] >> i) & 1)
            weight -= 1
        i += 1
    return np.array(poly, dtype=np.int64)
