"""
profiles.py — Named Lemur parameter sets.

Each LemurProfile bundles every scalar a KOTS / HVC / LEMUR instance needs.
Values come from the authoritative spreadsheet
`doc/Lemur Parameter Setting.ods`, sheet `IMPLEMENTATION - d=256_k=4`
(representative cell τ=20, N=1024).

KOTS proof condition
--------------------
The KOTS unforgeability reduction needs invertibility of
differences of T_{alpha_H} ring elements in R_{q'}.  The sufficient
condition is that R_{q'} splits into at most 8 residue fields, i.e.

    q' ≡ 2t + 1 (mod 4t)     with  t <= 8  a power of two

so with t = 8 the constraint is  q' ≡ 17 (mod 32).

Under this condition q' is NOT NTT-friendly, so KOTS ring multiplication
cannot use a native length-d negacyclic NTT.  The Python reference uses
schoolbook multiplication when the modulus fails the native NTT condition;
HVC q is unchanged and keeps the native NTT path.

Sampler convention (discrete-Gaussian precision analysis):
    sigma = alpha / sqrt(2 * pi)          # actual sampling stddev
    cdt_bits = 32  (32-bit read, LSB-as-sign, 31-bit CDF precision)
    tailcut  = 5 * sigma
"""

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class LemurProfile:
    """One concrete Lemur parameter set.

    `alpha` is the paper's Gaussian *width parameter* (appears in norm-bound
    derivations).  `sigma` is the actual sampling stddev used by the CDT
    sampler, related by sigma = alpha / sqrt(2*pi).
    """

    name: str
    # Common
    d: int
    tau: int
    n_signers: int
    # KOTS
    k: int
    m: int
    n: int
    q_prime: int
    alpha: float
    sigma: float
    alpha_h: int
    beta_z: int
    beta_sigma: int
    # HVC
    q: int
    omega: int
    eta: int
    beta_agg: int
    # Aggregation
    alpha_w: int
    gamma: int = 10
    # Sampler precision
    cdt_bits: int = 32
    tailcut: int = 5


def _profile(*, name, d, tau, n_signers, k, m, n, q_prime,
             alpha, alpha_h, beta_z, beta_sigma, q, omega, eta, beta_agg,
             alpha_w, gamma=10):
    """Build a LemurProfile with sigma = alpha / sqrt(2*pi).

    Asserts the KOTS proof condition `q' ≡ 17 (mod 32)` and the HVC
    native-NTT condition `q ≡ 1 (mod 2d)` at construction time so that
    parameter copy-paste mistakes fail at import.
    """
    assert q_prime % 32 == 17, (
        f"profile {name}: q_prime={q_prime} violates "
        f"the KOTS proof condition q' ≡ 17 (mod 32)"
    )
    assert (q - 1) % (2 * d) == 0, (
        f"profile {name}: HVC q={q} violates "
        f"the native NTT condition q ≡ 1 (mod 2d)"
    )
    return LemurProfile(
        name=name,
        d=d, tau=tau, n_signers=n_signers,
        k=k, m=m, n=n, q_prime=q_prime,
        alpha=float(alpha), sigma=float(alpha) / sqrt(2.0 * pi),
        alpha_h=alpha_h, beta_z=beta_z, beta_sigma=beta_sigma,
        q=q, omega=omega, eta=eta, beta_agg=beta_agg,
        alpha_w=alpha_w, gamma=gamma,
        cdt_bits=32, tailcut=5,
    )


# ---------------------------------------------------------------------------
# Lemur parameter set — doc/Lemur Parameter Setting.ods, representative cell
# (tau=20, N=1024).
# ---------------------------------------------------------------------------

D256_K4 = _profile(
    name="d256_k4",
    d=256, tau=20, n_signers=1024,
    k=4, m=9, n=4, q_prime=3_469_416_721,
    alpha=87, alpha_h=60, beta_z=7_023, beta_sigma=6_614_676,
    q=9_007_199_254_746_113, omega=2, eta=776, beta_agg=919_945,
    alpha_w=23,
)

PROFILES = {
    "d256_k4": D256_K4,
}


# Default profile used wherever a function takes a `profile=` kwarg.
DEFAULT = D256_K4


def get(name: str) -> LemurProfile:
    """Look up a profile by name (case-insensitive)."""
    try:
        return PROFILES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; known: {sorted(PROFILES)}"
        ) from None
