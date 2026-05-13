#!/usr/bin/env python3
"""
Lemur-DGS-Prec_TailCut.py -- precision and tailcut computation for the
CDT-based Lemur discrete Gaussian base sampler over Z.

Given security parameter lambda, sampling standard deviation sigma, and
Q_bs = k*m*d base-sampler queries per KOTS key generation, this script
computes:

* tailcut tc such that RD_{2*lambda+1}(D^{tc}_{Z+,sigma}, D_{Z+,sigma})
  <= 1/(8*Q_bs), and
* CDT comparison precision prec such that
  RD_{2*lambda+1}(D^{tc,prec}_{Z+,sigma}, D_{Z+,sigma}) <= 1/(4*Q_bs).

As in the proof of Theorem 7 in HPRR19 (ePrint 2019/1411), this bounds
the unforgeability loss from replacing the ideal sampler by the truncated,
finite-precision CDT sampler by at most one bit.

Q_bs is deliberately per key generation, not multiplied by the number of
Lemur time slots.  The slot/time dimension is handled at the reduction
level; using 2^tau*k*m*d here would double-count that hybrid step.

Implementation convention:

The Rust/Python implementations read one 32-bit word per coefficient.  The
least significant bit is used as the sign, leaving 31 bits for the CDT
comparison.  In general, an implementation with `cdt_bits = b` provides
`b-1` CDT comparison bits.  The shipped implementation uses `cdt_bits = 32`,
which is conservative for the current D256_K4 profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import argparse
import math


@dataclass(frozen=True)
class DgsProfile:
    name: str
    lam_act: int
    alpha: float
    d: int
    k: int
    m: int
    tc_id: int = 20
    prec_id: int = 380
    prec_comp: int = 508
    rust_cdt_bits: int = 32
    rust_tailcut: int = 5

    @property
    def sigma(self) -> float:
        # The implementation stores sigma = alpha / sqrt(2*pi).
        return self.alpha / math.sqrt(2.0 * math.pi)

    @property
    def q_bs(self) -> int:
        return self.k * self.m * self.d


# Shipped Rust/Python profile:
#   lemur-rs/src/profile.rs::D256_K4
#   lemur-py/profiles.py::D256_K4
D256_K4 = DgsProfile(
    name="d256_k4",
    lam_act=128,
    alpha=87.0,
    d=256,
    k=4,
    m=9,
)


def _dec_prec(bits: int) -> int:
    """Convert a bit-precision request into conservative decimal digits."""
    return math.ceil(bits * math.log10(2.0)) + 30


def _dec(x: float | int | str | Decimal) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def compute_cdt(
    sigma: float,
    lam: int,
    tailcut: int = 14,
    mp_prec: int = 320,
) -> list[int]:
    """Build a CDT table for the absolute value of a discrete Gaussian.

    Returns a list of integers:
        cdt[k] = floor(Pr[|X| <= k] * 2^lam)
    for k = 0, ..., floor(tailcut*sigma)+1, with the final entry clamped
    to 2^lam so binary search always terminates.

    The arithmetic uses Python's Decimal module, so the script does not
    require mpmath at runtime.
    """
    with localcontext() as ctx:
        ctx.prec = _dec_prec(mp_prec)
        sigma_dec = _dec(sigma)
        sigma2 = sigma_dec * sigma_dec
        kmax = int(tailcut * sigma) + 1

        weights = [
            (-(Decimal(k * k) / (Decimal(2) * sigma2))).exp()
            for k in range(kmax + 1)
        ]
        total = weights[0] + Decimal(2) * sum(weights[1:], Decimal(0))
        scale = Decimal(2) ** lam
        cdt: list[int] = []
        acc = Decimal(0)
        for k, w in enumerate(weights):
            acc += (w if k == 0 else Decimal(2) * w) / total
            cdt.append(int((acc * scale).to_integral_value(rounding=ROUND_FLOOR)))
        cdt[-1] = 1 << lam
        return cdt


def compute_Rdiv(
    a: int,
    cdt_re: list[int],
    prec_re: int,
    cdt_id: list[int],
    prec_id: int,
    prec_comp: int = 640,
) -> Decimal:
    """Compute the Renyi divergence ratio between two CDT distributions."""
    with localcontext() as ctx:
        ctx.prec = _dec_prec(prec_comp)
        cdt_re_prev = 0
        cdt_id_prev = 0
        scale_re = Decimal(2) ** prec_re
        scale_id = Decimal(2) ** prec_id
        rd_sum = Decimal(0)
        am1 = Decimal(a - 1)

        for x in range(len(cdt_re)):
            p_re = Decimal(cdt_re[x] - cdt_re_prev) / scale_re
            p_id = Decimal(cdt_id[x] - cdt_id_prev) / scale_id
            if p_re:
                rd_sum += p_re * ((p_re / p_id) ** (a - 1))
            cdt_re_prev = cdt_re[x]
            cdt_id_prev = cdt_id[x]

        # Return rd_sum^(1/(a-1)) without relying on fractional Decimal powers.
        return (rd_sum.ln() / am1).exp()


def log2_decimal(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = max(x.adjusted() + 50, 80)
        return x.ln() / Decimal(2).ln()


def compute_tc_prec_dgs(
    sigma: float,
    lam_act: int,
    k: int,
    m: int,
    d: int,
    tc_id: int = 20,
    prec_id: int = 380,
    prec_comp: int = 508,
) -> tuple[int, int, Decimal, Decimal, list[int], list[int]]:
    """Compute Lemur DGS precision/tailcut parameters."""
    q_bs = k * m * d
    a = 2 * lam_act + 1

    with localcontext() as ctx:
        ctx.prec = _dec_prec(prec_comp)
        eps_1 = Decimal(1) / Decimal(8 * q_bs)
        eps_tot = Decimal(1) / Decimal(4 * q_bs)
        sigma_dec = _dec(sigma)

        # Tail bound: Pr[|X| > kmax] < 2*exp(-(kmax/sigma)^2/2).
        target = sigma_dec * (Decimal(2) * (Decimal(2) / eps_1).ln()).sqrt()
        kmax_bound = int(target.to_integral_value(rounding=ROUND_CEILING))
        tc_re = int((Decimal(kmax_bound - 1) / sigma_dec).to_integral_value(rounding=ROUND_FLOOR)) + 1

        # Start with a conservative upper bound and binary search down.
        prec_re_lb = 1
        prec_re_ub = int(
            (Decimal(10) * (Decimal(1) / eps_tot).ln() / Decimal(2).ln())
            .to_integral_value(rounding=ROUND_CEILING)
        )

    cdt_id = compute_cdt(sigma, prec_id, tc_id, prec_comp)

    while prec_re_ub - prec_re_lb > 1:
        prec_re_cur = (prec_re_lb + prec_re_ub) // 2
        cdt_re_cur = compute_cdt(sigma, prec_re_cur, tc_re, prec_comp)
        rdiv_cur = compute_Rdiv(
            a, cdt_re_cur, prec_re_cur, cdt_id, prec_id, prec_comp
        )
        if rdiv_cur <= Decimal(1) + eps_tot:
            prec_re_ub = prec_re_cur
        else:
            prec_re_lb = prec_re_cur

    prec_re = prec_re_ub
    cdt_re = compute_cdt(sigma, prec_re, tc_re, prec_comp)
    rd = compute_Rdiv(a, cdt_re, prec_re, cdt_id, prec_id, prec_comp)
    return prec_re, tc_re, rd, eps_tot, cdt_re, cdt_id


def compute_DGSPars(profile: DgsProfile = D256_K4, check: bool = False) -> None:
    """Compute and print DGS parameters for a Lemur implementation profile."""
    sigma = profile.sigma
    prec_re, tc_re, rd, eps, cdt_re, _cdt_id = compute_tc_prec_dgs(
        sigma,
        profile.lam_act,
        profile.k,
        profile.m,
        profile.d,
        profile.tc_id,
        profile.prec_id,
        profile.prec_comp,
    )

    min_cdt_bits = prec_re + 1  # one sign bit + prec_re comparison bits
    impl_comparison_bits = profile.rust_cdt_bits - 1
    kmax_re = int(tc_re * sigma) + 1
    impl_kmax = int(profile.rust_tailcut * sigma) + 1
    lg2rd = log2_decimal(rd)
    lg2rd_des = log2_decimal(Decimal(1) + eps)

    print("*** Inputs ***")
    print("profile =", profile.name)
    print("lam_act =", profile.lam_act)
    print("alpha =", profile.alpha)
    print("sigma =", sigma)
    print("d =", profile.d)
    print("k =", profile.k)
    print("m =", profile.m)
    print("Q_bs = k*m*d per keygen =", profile.q_bs)
    print("tc_id =", profile.tc_id)
    print("prec_id =", profile.prec_id)
    print("prec_comp =", profile.prec_comp)
    print()
    print("*** Outputs ***")
    print("minimum prec_re (CDT comparison precision) =", prec_re)
    print("minimum cdt_bits including sign bit =", min_cdt_bits)
    print("implementation comparison bits =", impl_comparison_bits)
    print("implementation cdt_bits =", profile.rust_cdt_bits)
    print("tc_re (minimum tailcut parameter rel. to sigma) =", tc_re)
    print("implementation tailcut =", profile.rust_tailcut)
    print("kmax_re (minimum table max index) =", kmax_re)
    print("implementation table max index =", impl_kmax)
    print("CDT entries =", len(cdt_re))
    print("lg2rd (log2(RD)) =", float(lg2rd))
    print("lg2rd_des (desired log2(RD) upper bound) =", float(lg2rd_des))

    if check:
        assert impl_comparison_bits >= prec_re, (
            f"implementation comparison bits={impl_comparison_bits} "
            f"is below required prec_re={prec_re}"
        )
        assert profile.rust_tailcut >= tc_re, (
            f"implementation tailcut={profile.rust_tailcut} "
            f"is below required tc_re={tc_re}"
        )
        assert len(cdt_re) == kmax_re + 1
        assert rd <= Decimal(1) + eps
        print()
        print("*** Checks passed ***")
        print("Implementation cdt_bits/tailcut are conservative for this profile.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert that the shipped implementation covers the computed bounds",
    )
    args = parser.parse_args()
    compute_DGSPars(check=args.check)


if __name__ == "__main__":
    main()
