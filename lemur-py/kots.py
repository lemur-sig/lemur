"""
KOTS: Key-Homomorphic One-Time Signature.

All randomness is derived from SHAKE XOFs:
  - Public matrix A has the structured form A = [I; A2] and only A2 is
    expanded from SHAKE128(seed || [i,j] || b'A').
  - Secret keys are sampled per-entry from SHAKE256(seed || [i,j] || b'S').
  - Message hash H(mu) uses SHAKE256(mu || index || b'H').
"""

import numpy as np
from Crypto.Hash import SHAKE128, SHAKE256

from profiles import LemurProfile, DEFAULT
from ring import Ring
from sample import (
    GaussianSampler,
    build_cdt,
    xof_gauss_poly,
    xof_ternary_poly,
    xof_uniform_poly,
)


class KOTS:
    """Key-Homomorphic One-Time Signature scheme.

    Default profile: profiles.DEFAULT (d256_k4 — d=256, tau=20, N=1024).
    Individual keyword overrides take precedence over profile fields,
    which is how `make_codec(tau=...)` keeps working.
    """

    def __init__(self, profile: LemurProfile | None = None, *,
                 d=None, q=None, k=None, ell=None, m=None, n=None,
                 alpha=None, alpha_h=None, beta_z=None, beta_sigma=None,
                 lam=128, sigma: float | None = None,
        sampler: GaussianSampler | None = None):
        if profile is None:
            profile = DEFAULT
        self.profile    = profile
        self.d          = d          if d          is not None else profile.d
        self.q          = q          if q          is not None else profile.q_prime
        self.k          = k          if k          is not None else profile.k
        self.ell        = ell        if ell        is not None else profile.ell
        self.m          = m          if m          is not None else profile.m
        self.n          = n          if n          is not None else profile.n
        self.alpha      = alpha      if alpha      is not None else profile.alpha
        self.alpha_h    = alpha_h    if alpha_h    is not None else profile.alpha_h
        self.beta_z     = beta_z     if beta_z     is not None else profile.beta_z
        self.beta_sigma = beta_sigma if beta_sigma is not None else profile.beta_sigma
        self.lam        = lam
        # `alpha` is the paper symbol (Gaussian width parameter; appears in
        # norm-bound derivations).  The sampler's sigma is the actual CDT
        # stddev: sigma = alpha / sqrt(2*pi).  Both come from the profile
        # unless the caller overrides.
        if sampler is None:
            sigma_val = profile.sigma if sigma is None else sigma
            sampler = GaussianSampler(
                sigma=sigma_val,
                cdt_bits=profile.cdt_bits,
                tailcut=profile.tailcut,
            )
        self.sampler    = sampler
        # Derived bit widths (used by codec)
        self.logq       = (self.q - 1).bit_length()
        self.bits_z     = (2 * self.beta_z).bit_length()
        self.bits_zagg  = (2 * self.beta_sigma).bit_length()
        # Ring with NTT
        self.ring       = Ring(self.q, self.d)
        # CDT table for discrete Gaussian keygen (built once at init)
        self.cdt        = build_cdt(
            self.sampler.sigma,
            self.sampler.cdt_bits,
            self.sampler.tailcut,
        )

    @staticmethod
    def _inf_norm(Z: np.ndarray) -> int:
        """Max absolute coefficient of a polynomial or polynomial matrix."""
        return int(np.max(np.abs(Z)))

    # -----------------------------------------------------------------------
    # Hash: mu -> H(mu) in T_{alpha_h}
    # -----------------------------------------------------------------------

    def _hash_to_ternary(self, mu, index=0):
        xof = SHAKE256.new(mu + index.to_bytes(4, 'little') + b'H')
        return xof_ternary_poly(xof, self.alpha_h, self.d)

    def _build_H(self, mu):
        """Construct H = [I_ell | H'] in R^{ell x k}.  Returns shape (ell, k, d)."""
        k, ell, d = self.k, self.ell, self.d
        H = np.zeros((ell, k, d), dtype=np.int64)
        for i in range(ell):
            H[i, i, 0] = 1
        for i in range(ell):
            for j in range(k - ell):
                H[i, ell + j] = self._hash_to_ternary(mu, index=i * (k - ell) + j)
        return H

    # -----------------------------------------------------------------------
    # KOTS algorithms
    # -----------------------------------------------------------------------

    def setup(self, seed):
        """Setup(seed) -> A2 in R_q^{(m-n) x n}.

        The effective public matrix is the structured matrix
            A = [I_n; A2] in R_q^{m x n},
        but only the lower block A2 is materialized and stored.
        """
        rows = self.m - self.n
        A2 = np.zeros((rows, self.n, self.d), dtype=np.int64)
        for i in range(rows):
            for j in range(self.n):
                xof = SHAKE128.new(seed + bytes([i, j]) + b'A')
                A2[i, j] = xof_uniform_poly(xof, self.q, self.d)
        return A2

    def _matmul_with_A(self, X, A2):
        """Multiply X by the implicit structured matrix A = [I; A2] mod q."""
        top = X[:, :self.n].copy() % self.q
        bottom = self.ring.mat_mul(X[:, self.n:], A2)
        return (top + bottom) % self.q

    def keygen(self, A2, seed):
        """KGen(A2, seed) -> (S, T).

        S[i][j] from SHAKE256(seed || [i,j] || b'S'), T = S * A mod q where
        A is the implicit structured matrix [I; A2].
        """
        S = np.zeros((self.k, self.m, self.d), dtype=np.int64)
        for i in range(self.k):
            for j in range(self.m):
                xof = SHAKE256.new(seed + bytes([i, j]) + b'S')
                S[i, j] = xof_gauss_poly(xof, self.cdt, self.sampler.cdt_bits, self.d)
        T = self._matmul_with_A(S, A2)
        return S, T

    def sign(self, A2, S, mu):
        """Sign(A2, S, mu) -> Z = H * S in R^{ell x m}.

        Each Z[i,j] = sum_l H[i,l] * S[l,j] (over Z, exact signed arithmetic).
        The coefficient bound k * alpha_h * max|S| << q/2 so mul_signed is valid.
        """
        H = self._build_H(mu)
        Z = np.zeros((self.ell, self.m, self.d), dtype=np.int64)
        for i in range(self.ell):
            for j in range(self.m):
                for l in range(self.k):
                    Z[i, j] = Z[i, j] + self.ring.mul_signed(H[i, l], S[l, j])
        return Z

    def vrfy(self, A2, T, mu, Z, beta):
        """Vrfy: return True iff ||Z||_inf <= beta and Z*A == H*T (mod q)."""
        if self._inf_norm(Z) > beta:
            return False
        H  = self._build_H(mu)
        ZA = self._matmul_with_A(Z, A2)
        HT = self.ring.mat_mul(H, T)
        return bool(np.all(ZA == HT))

    def ivrfy(self, A2, T, mu, Z):
        return self.vrfy(A2, T, mu, Z, self.beta_z)

    def svrfy(self, A2, T, mu, Z):
        return self.vrfy(A2, T, mu, Z, self.beta_sigma)

    def wvrfy(self, A2, T, mu, Z):
        return self.vrfy(A2, T, mu, Z, 2 * self.beta_sigma)


# ---------------------------------------------------------------------------
# Quick correctness test
# Module-level alias so hvc / lemur / cli can keep `from kots import inf_norm`.
inf_norm = KOTS._inf_norm


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kots = KOTS()
    print(f"KOTS: d={kots.d}, q={kots.q}, k={kots.k}, ell={kots.ell}, "
          f"m={kots.m}, n={kots.n}")
    print(f"  alpha={kots.alpha}, alpha_h={kots.alpha_h}, "
          f"beta_z={kots.beta_z}, beta_sigma={kots.beta_sigma}")
    print(f"  logq={kots.logq}, bits_z={kots.bits_z}, bits_zagg={kots.bits_zagg}")
    print(f"  ring zeta={kots.ring.zeta}")
    print()

    A2   = kots.setup(bytes(range(32)))
    S, T = kots.keygen(A2, bytes(range(32, 64)))
    mu   = b"test message"
    Z    = kots.sign(A2, S, mu)

    print(f"  implicit A shape: ({kots.m}, {kots.n}), stored A2 shape: {A2.shape[:2]}")
    print(f"  ||Z||_inf = {kots._inf_norm(Z)}  (beta_z={kots.beta_z})")
    print(f"  ivrfy: {'PASS' if kots.ivrfy(A2, T, mu, Z) else 'FAIL'}")
    print(f"  svrfy: {'PASS' if kots.svrfy(A2, T, mu, Z) else 'FAIL'}")
    Z2 = kots.sign(A2, S, b"wrong")
    print(f"  wrong-msg ivrfy: "
          f"{'FAIL (expected)' if not kots.ivrfy(A2, T, mu, Z2) else 'PASS (unexpected!)'}")
