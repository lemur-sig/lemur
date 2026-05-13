"""
KOTS: Key-Homomorphic One-Time Signature.

All randomness is derived from SHAKE128 XOFs:
  - Public matrix A has the structured form A = [I; A2] and only A2 is
    expanded from SHAKE128(seed || [i,j] || b'A').
  - Secret keys are sampled per-entry from SHAKE128(seed || [i,j] || b'S').
  - Message hash H(mu) uses SHAKE128(mu || index || b'H').
"""

import numpy as np
from Crypto.Hash import SHAKE128

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
                 d=None, q=None, k=None, m=None, n=None,
                 alpha=None, alpha_h=None, beta_z=None, beta_sigma=None,
                 lam=128, sigma: float | None = None,
        sampler: GaussianSampler | None = None):
        if profile is None:
            profile = DEFAULT
        self.profile    = profile
        self.d          = d          if d          is not None else profile.d
        self.q          = q          if q          is not None else profile.q_prime
        self.k          = k          if k          is not None else profile.k
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
        xof = SHAKE128.new(mu + index.to_bytes(4, 'little') + b'H')
        return xof_ternary_poly(xof, self.alpha_h, self.d)

    def _build_h(self, mu):
        """Construct h = [1 | h'] in R^k.  Returns shape (k, d)."""
        k, d = self.k, self.d
        h = np.zeros((k, d), dtype=np.int64)
        h[0, 0] = 1
        for j in range(k - 1):
            h[1 + j] = self._hash_to_ternary(mu, index=j)
        return h

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

        S[i][j] from SHAKE128(seed || [i,j] || b'S'), T = S * A mod q where
        A is the implicit structured matrix [I; A2].
        """
        S = np.zeros((self.k, self.m, self.d), dtype=np.int64)
        for i in range(self.k):
            for j in range(self.m):
                xof = SHAKE128.new(seed + bytes([i, j]) + b'S')
                S[i, j] = xof_gauss_poly(xof, self.cdt, self.sampler.cdt_bits, self.d)
        T = self._matmul_with_A(S, A2)
        return S, T

    def sign(self, A2, S, mu):
        """Sign(A2, S, mu) -> z = h * S in R^m, returned shape (m, d).

        z[j] = sum_l h[l] * S[l, j] over Z (exact signed arithmetic).
        The coefficient bound k * alpha_h * max|S| << q/2 so mul_signed is valid.
        """
        h = self._build_h(mu)
        z = np.zeros((self.m, self.d), dtype=np.int64)
        for j in range(self.m):
            for l in range(self.k):
                z[j] = z[j] + self.ring.mul_signed(h[l], S[l, j])
        return z

    def vrfy(self, A2, T, mu, z, beta):
        """Vrfy: return True iff ||z||_inf <= beta and z*A == h*T (mod q)."""
        if self._inf_norm(z) > beta:
            return False
        h  = self._build_h(mu)
        # Treat z as a 1-row matrix for the structured-A multiply.
        zA = self._matmul_with_A(z[np.newaxis, :, :], A2)[0]
        hT = self.ring.mat_mul(h[np.newaxis, :, :], T)[0]
        return bool(np.all(zA == hT))

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
    print(f"KOTS: d={kots.d}, q={kots.q}, k={kots.k}, m={kots.m}, n={kots.n}")
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
