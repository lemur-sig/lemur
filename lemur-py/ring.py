"""
ring.py — Negacyclic polynomial arithmetic for R_q = Z_q[X]/(X^d + 1).

Supports power-of-two ring dimensions d ∈ {128, 256}.

Two multiplication backends, selected automatically from `q` in `__init__`:

  * Native NTT — for q that satisfies the full NTT condition
      q ≡ 1 (mod 2d).
    This is the case for every shipped HVC q.

  * Schoolbook fallback — for q that is NOT NTT-friendly.  Every KOTS
    prime satisfies `q' ≡ 17 (mod 32)` and takes this path.  This keeps
    the Python implementation readable as a reference.

Signed exact arithmetic (`scale_vec`, `scale_mat`, `mul_signed`) uses
canonical lifting from [0, q) to [-q//2, q//2] after multiplication.
This recovers the true integer result whenever it fits in (-q/2, q/2),
which the caller is responsible for verifying via parameter analysis.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Primitive root-of-unity search
# ---------------------------------------------------------------------------

def find_primitive_root(q: int, d: int) -> int:
    """Find the smallest primitive 2d-th root of unity in Z_q.

    Tries x = 2, 3, 4, ... in order.  For each x computes
        zeta = x^((q-1) // (2d))  mod q
    and returns the first zeta whose order is exactly 2d, verified by
        zeta^d ≡ -1 (mod q).
    """
    assert (q - 1) % (2 * d) == 0, f"q-1 must be divisible by 2d (q={q}, d={d})"
    exp = (q - 1) // (2 * d)
    for x in range(2, q):
        zeta = pow(x, exp, q)
        if pow(zeta, d, q) == q - 1:
            return zeta
    raise ValueError(f"No primitive 2d-th root found for q={q}, d={d}")


# ---------------------------------------------------------------------------
# Twiddle-factor table
# ---------------------------------------------------------------------------

def _bitrev(k: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (k & 1)
        k >>= 1
    return result


def make_zeta_table(q: int, d: int, zeta: int) -> list[int]:
    """Build the NTT twiddle-factor table (length d, bit-reversed order).

    table[k] = zeta^bitrev(k, log2(d))  mod q.
    Forward NTT uses table[1..d-1]; inverse uses them in reverse.
    Matches the ML-DSA twiddle convention.
    """
    bits = d.bit_length() - 1
    return [pow(zeta, _bitrev(k, bits), q) for k in range(d)]


# ---------------------------------------------------------------------------
# Forward NTT
# ---------------------------------------------------------------------------

def ntt(f: list[int], zetas: list[int], q: int) -> list[int]:
    """Forward NTT in R_q = Z_q[X]/(X^d + 1).

    Input:  polynomial in normal order, values in [0, q)  (negatives accepted).
    Output: NTT representation in bit-reversed order, values in [0, q).
    """
    f = f.copy()
    d = len(f)
    m = 0
    le = d // 2
    while le >= 1:
        st = 0
        while st < d:
            m += 1
            z = zetas[m]
            for j in range(st, st + le):
                t         = z * f[j + le] % q
                f[j + le] = (f[j] - t) % q
                f[j]      = (f[j] + t) % q
            st += 2 * le
        le //= 2
    return f


# ---------------------------------------------------------------------------
# Inverse NTT
# ---------------------------------------------------------------------------

def intt(f: list[int], zetas: list[int], q: int) -> list[int]:
    """Inverse NTT in R_q = Z_q[X]/(X^d + 1).

    Input:  NTT representation in bit-reversed order, values in [0, q).
    Output: polynomial in normal order, values in [0, q).
    """
    f = f.copy()
    d = len(f)
    m = d
    le = 1
    while le < d:
        st = 0
        while st < d:
            m -= 1
            z = (-zetas[m]) % q
            for j in range(st, st + le):
                t         = f[j]
                f[j]      = (t + f[j + le]) % q
                f[j + le] = z * (t - f[j + le]) % q
            st += 2 * le
        le *= 2
    inv_d = pow(d, -1, q)
    return [inv_d * x % q for x in f]


# ---------------------------------------------------------------------------
# Schoolbook multiplication
# ---------------------------------------------------------------------------

def poly_mul_schoolbook(a: list[int], b: list[int], q: int) -> list[int]:
    """Schoolbook negacyclic multiplication in Z_q[X]/(X^d + 1)."""
    d = len(a)
    c = [0] * d
    for i in range(d):
        for j in range(d):
            idx = i + j
            if idx < d:
                c[idx] = (c[idx] + a[i] * b[j]) % q
            else:
                c[idx - d] = (c[idx - d] - a[i] * b[j]) % q
    return c


# ---------------------------------------------------------------------------
# Ring class
# ---------------------------------------------------------------------------

class Ring:
    """Negacyclic polynomial ring Z_q[X]/(X^d + 1) with NTT-backed arithmetic.

    All mod-q operations return values in [0, q).  The *_signed variants
    (mul_signed, scale_vec, scale_mat) additionally lift the result to the
    canonical signed range [-q//2, q//2], recovering the exact integer
    product.  This is valid whenever the true product coefficients fit in
    (-q/2, q/2); the caller is responsible for verifying the norm bound via
    parameter analysis.

    The schoolbook fallback is intentionally simple and slow; it is used
    only for non-NTT-friendly KOTS moduli in the Python reference.
    """

    def __init__(self, q: int, d: int):
        self.q     = q
        self.d     = d
        # Pick backend based on whether q is NTT-friendly.  KOTS rings
        # (q' ≡ 17 mod 32) fail the native condition and use schoolbook
        # multiplication; HVC rings satisfy q ≡ 1 (mod 2d) and keep the
        # native path.
        self._native_ntt = (q - 1) % (2 * d) == 0
        if self._native_ntt:
            self.zeta  = find_primitive_root(q, d)
            self.zetas = make_zeta_table(q, d, self.zeta)
            # Use dtype=object for native NTT intermediates when q^2 does
            # not fit in int64, falling back to int64 for speed otherwise.
            # int64.max is about 2^63 - 1, so cutoff is q > 2^31.
            self._ntt_dtype = object if q > (1 << 31) else np.int64
        else:
            self.zeta = None
            self.zetas = None
            self._ntt_dtype = np.int64

    # -----------------------------------------------------------------------
    # Internal NTT helpers (native path)
    # -----------------------------------------------------------------------

    def _ntt(self, a) -> np.ndarray:
        """Forward NTT (native).  Accepts array-like; signed values reduced mod q."""
        assert self._native_ntt, "native NTT helper called on non-NTT-friendly Ring"
        return np.array(
            ntt([int(x) % self.q for x in a], self.zetas, self.q),
            dtype=self._ntt_dtype,
        )

    def _intt(self, a) -> np.ndarray:
        """Inverse NTT (native).  Input in [0, q); output in [0, q)."""
        assert self._native_ntt, "native NTT helper called on non-NTT-friendly Ring"
        return np.array(
            intt(list(map(int, a)), self.zetas, self.q),
            dtype=self._ntt_dtype,
        )

    def _lift(self, a: np.ndarray) -> np.ndarray:
        """Lift from [0, q) to canonical signed [-q//2, q//2]."""
        a = a.copy()
        a[a > self.q // 2] -= self.q
        return a

    # -----------------------------------------------------------------------
    # Single polynomial operations
    # -----------------------------------------------------------------------

    def mul(self, a, b) -> np.ndarray:
        """Negacyclic poly multiplication mod q.  Returns (d,) in [0, q)."""
        if not self._native_ntt:
            a_list = [int(x) % self.q for x in a]
            b_list = [int(x) % self.q for x in b]
            return np.array(poly_mul_schoolbook(a_list, b_list, self.q), dtype=np.int64)
        a_hat = self._ntt(a)
        b_hat = self._ntt(b)
        return self._intt(a_hat * b_hat % self.q).astype(np.int64)

    def mul_signed(self, a, b) -> np.ndarray:
        """Negacyclic poly multiplication; lifts result to signed [-q//2, q//2]."""
        return self._lift(self.mul(a, b))

    # -----------------------------------------------------------------------
    # Matrix / vector operations (mod q)
    # -----------------------------------------------------------------------

    def mat_vec(self, A: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Polynomial matrix–vector product mod q.

        A: (r, c, d),  v: (c, d)  →  result: (r, d) in [0, q).

        Native path: pre-transforms v once per column; accumulates each
        output row in the NTT domain before a single inverse transform.
        Schoolbook path: straightforward mul + add accumulation per output row.
        """
        r, c = A.shape[:2]
        if not self._native_ntt:
            result = np.zeros((r, self.d), dtype=np.int64)
            for i in range(r):
                acc = np.zeros(self.d, dtype=np.int64)
                for j in range(c):
                    acc = (acc + self.mul(A[i, j], v[j])) % self.q
                result[i] = acc
            return result
        v_hat = [self._ntt(v[j]) for j in range(c)]
        result = np.zeros((r, self.d), dtype=np.int64)
        for i in range(r):
            acc = np.zeros(self.d, dtype=self._ntt_dtype)
            for j in range(c):
                acc = (acc + self._ntt(A[i, j]) * v_hat[j]) % self.q
            result[i] = self._intt(acc).astype(np.int64)
        return result

    def mat_mul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Polynomial matrix–matrix product mod q.

        A: (r, s, d),  B: (s, t, d)  →  result: (r, t, d) in [0, q).
        """
        r, s = A.shape[:2]
        t = B.shape[1]
        if not self._native_ntt:
            result = np.zeros((r, t, self.d), dtype=np.int64)
            for i in range(r):
                for j in range(t):
                    acc = np.zeros(self.d, dtype=np.int64)
                    for l in range(s):
                        acc = (acc + self.mul(A[i, l], B[l, j])) % self.q
                    result[i, j] = acc
            return result
        B_hat = [[self._ntt(B[l, j]) for j in range(t)] for l in range(s)]
        result = np.zeros((r, t, self.d), dtype=np.int64)
        for i in range(r):
            A_hat_row = [self._ntt(A[i, l]) for l in range(s)]
            for j in range(t):
                acc = np.zeros(self.d, dtype=self._ntt_dtype)
                for l in range(s):
                    acc = (acc + A_hat_row[l] * B_hat[l][j]) % self.q
                result[i, j] = self._intt(acc).astype(np.int64)
        return result

    # -----------------------------------------------------------------------
    # Signed scaling  (for LEMUR aggregation)
    # -----------------------------------------------------------------------

    def scale_vec(self, w, v: np.ndarray) -> np.ndarray:
        """Multiply scalar poly w by each row of v: (n, d) → (n, d) signed exact.

        Valid when each product coefficient is < q/2.
        """
        if not self._native_ntt:
            result = np.zeros_like(v, dtype=np.int64)
            for i in range(v.shape[0]):
                result[i] = self._lift(self.mul(w, v[i]))
            return result
        w_hat = self._ntt(w)
        result = np.zeros_like(v, dtype=np.int64)
        for i in range(v.shape[0]):
            prod = (w_hat * self._ntt(v[i])) % self.q
            result[i] = self._lift(self._intt(prod).astype(np.int64))
        return result

    def scale_mat(self, w, M: np.ndarray) -> np.ndarray:
        """Multiply scalar poly w by each entry of M: (r,c,d) → (r,c,d) signed exact.

        Valid when each product coefficient is < q/2.
        """
        r, c = M.shape[:2]
        if not self._native_ntt:
            result = np.zeros_like(M, dtype=np.int64)
            for i in range(r):
                for j in range(c):
                    result[i, j] = self._lift(self.mul(w, M[i, j]))
            return result
        w_hat = self._ntt(w)
        result = np.zeros_like(M, dtype=np.int64)
        for i in range(r):
            for j in range(c):
                prod = (w_hat * self._ntt(M[i, j])) % self.q
                result[i, j] = self._lift(self._intt(prod).astype(np.int64))
        return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    # HVC modulus (NTT-friendly: q ≡ 1 mod 2d).
    Q_D256_K4   = 1_125_899_906_856_961    # ~2^50
    # KOTS modulus (q' ≡ 17 mod 32, schoolbook path).
    QP_D256_K4  = 867_354_289

    random.seed(42)

    native_cases = [
        (Q_D256_K4, "Q_D256_K4 (native)", 256),
    ]
    schoolbook_cases = [
        (QP_D256_K4, "KOTS q' d256_k4 (schoolbook)", 256),
    ]

    for q, q_name, d in native_cases + schoolbook_cases:
        print(f"--- q = {q} ({q_name}),  d = {d} ---")
        ring = Ring(q, d)

        backend = "native NTT" if ring._native_ntt else "schoolbook"
        print(f"  backend: {backend}")
        if ring._native_ntt:
            assert pow(ring.zeta, 2 * d, q) == 1
            assert pow(ring.zeta, d, q) == q - 1

        a = [random.randrange(q) for _ in range(d)]
        b = [random.randrange(q) for _ in range(d)]

        c_ref = poly_mul_schoolbook(a, b, q)
        c_got = list(ring.mul(a, b))
        print(f"  ring.mul == schoolbook:    {'PASS' if c_ref == c_got else 'FAIL'}")

        # Signed: small inputs, check exact round-trip
        a_s = [random.randint(-200, 200) for _ in range(d)]
        b_s = [random.randint(-200, 200) for _ in range(d)]
        c_exact = poly_mul_schoolbook([x % q for x in a_s], [x % q for x in b_s], q)
        c_exact = [x - q if x > q // 2 else x for x in c_exact]
        c_signed = list(ring.mul_signed(a_s, b_s))
        print(f"  ring.mul_signed exact:     {'PASS' if c_exact == c_signed else 'FAIL'}")
        print()
