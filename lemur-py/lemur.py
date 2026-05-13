"""
LEMUR: Synchronized N-wise multi-signature scheme.

Composes KOTS (kots.py) and HVC (hvc.py).  Parameters alpha_w and gamma live
at this level; all sub-scheme parameters are accessed via the injected instances.
"""

import numpy as np
from Crypto.Hash import SHAKE128

from kots import KOTS, inf_norm
from hvc import HVC, bds_copy_state
from profiles import LemurProfile
from sample import xof_ternary_poly


# ---------------------------------------------------------------------------
# LEMUR class
# ---------------------------------------------------------------------------

class LEMUR:
    """Top-level Lemur multi-signature scheme.

    Composes KOTS and HVC sub-schemes.  Parameters alpha_w (randomizer weight)
    and gamma (max aggregation attempts) live here.
    """

    def __init__(self, kots: KOTS, hvc: HVC, *,
                 profile: LemurProfile | None = None,
                 alpha_w: int | None = None, gamma: int | None = None):
        if profile is None:
            profile = kots.profile
        self.profile = profile
        self.kots    = kots
        self.hvc     = hvc
        self.alpha_w = alpha_w if alpha_w is not None else profile.alpha_w
        self.gamma   = gamma   if gamma   is not None else profile.gamma
        self.n_slots = hvc.n_slots

    # -----------------------------------------------------------------------
    # Private: randomizer oracle and commitment aggregation
    # -----------------------------------------------------------------------

    def _hash_to_randomizers(self, t: int, m: bytes, P: list, attempt: int) -> list:
        """Sample N ternary randomizers w^i ∈ T_{alpha_w} via one SHAKE128 XOF.

        All N randomizers are streamed from a single XOF keyed on (t, m, P, attempt).
        """
        pk_bytes = b"".join(c.tobytes() for c in P)
        domain = (
            t.to_bytes(4, "little")
            + len(m).to_bytes(4, "little")
            + m
            + pk_bytes
            + attempt.to_bytes(4, "little")
        )
        xof = SHAKE128.new(domain)
        return [xof_ternary_poly(xof, self.alpha_w, self.kots.d) for _ in range(len(P))]

    @staticmethod
    def _add_openings(d1: tuple, d2: tuple) -> tuple:
        """Add two HVC openings componentwise over Z."""
        p1, s1, u1 = d1
        p2, s2, u2 = d2
        return (
            [p1[j] + p2[j] for j in range(len(p1))],
            [s1[j] + s2[j] for j in range(len(s1))],
            u1 + u2,
        )

    def _scale_opening(self, w: np.ndarray, opening: tuple) -> tuple:
        """Scale HVC opening (path_labels, sibling_labels, u) by w, signed exact."""
        path_labels, sibling_labels, u = opening
        ring = self.hvc.ring
        return (
            [ring.scale_vec(w, p) for p in path_labels],
            [ring.scale_vec(w, s) for s in sibling_labels],
            ring.scale_vec(w, u),
        )

    def _weighted_sum_commitments(self, ws: list, pks: list) -> np.ndarray:
        """c_agg = sum_i w^i * c_i  in R_q^omega."""
        result = np.zeros((self.hvc.omega, self.kots.d), dtype=np.int64)
        ring = self.hvc.ring
        for w, c in zip(ws, pks):
            for r in range(self.hvc.omega):
                result[r] = (result[r] + ring.mul(w, c[r])) % self.hvc.q
        return result

    # -----------------------------------------------------------------------
    # Public algorithms
    # -----------------------------------------------------------------------

    def setup(self, kots_seed: bytes, hvc_seed: bytes) -> tuple:
        """Setup(kots_seed, hvc_seed) -> pp = (kots_pp, hvc_pp)."""
        return self.kots.setup(kots_seed), self.hvc.setup(hvc_seed)

    @staticmethod
    def _slot_seed(master_seed: bytes, t: int) -> bytes:
        """Derive per-slot seed from master seed."""
        return SHAKE128.new(
            master_seed + b'slot' + t.to_bytes(4, 'little')
        ).read(32)

    @staticmethod
    def make_stateful_sk(master_seed: bytes, bds: dict) -> dict:
        """Construct a stateful signer key from a master seed and BDS state.

        The stateful form always carries a populated BDS08 traversal cache
        (built by `keygen`/`bds_init`).  The "current slot" is the single
        counter `bds["phi"]` — there is no separate `next_slot` field.
        Callers that only have a raw master seed should use `sign_seed`
        (which passes the slot explicitly) or run `keygen` to materialise
        a stateful sk with its BDS cache.
        """
        if len(master_seed) != 32:
            raise ValueError("master_seed must be 32 bytes")
        if bds is None:
            raise ValueError("stateful sk must carry a BDS state")
        return {
            "master_seed": bytes(master_seed),
            "bds":         bds,
        }

    def _build_leaf_fn(self, kots_pp, master_seed: bytes):
        kots = self.kots
        def leaf_fn(t):
            ss = self._slot_seed(master_seed, t)
            _, opk = kots.keygen(kots_pp, ss)
            return opk
        return leaf_fn

    def keygen(self, pp: tuple, seed: bytes) -> tuple:
        """KGen(pp, seed) -> (sk_seed, sk_state, pk).

        sk_seed = master_seed (32 bytes).
        sk_state = stateful signer key with BDS08 cache at phi=0.
        pk = c: HVC commitment in R_q^omega, shape (omega, d).

        Builds the HVC commitment and the initial BDS08 traversal state
        in a single tree walk.  Per-slot KOTS keys are re-derived from
        the master seed on demand.
        """
        kots_pp, hvc_pp = pp
        leaf_fn = self._build_leaf_fn(kots_pp, seed)
        c, bds = self.hvc.bds_init(hvc_pp, leaf_fn)
        return seed, self.make_stateful_sk(seed, bds), c

    def sign_seed(self, pp: tuple, sk: bytes, t: int, m: bytes) -> tuple:
        """Sign(pp, sk_seed, t, m) -> (Z, d_open).

        sk_seed is the 32-byte master seed.
        Z:      KOTS signature, shape (m_dim, d).
        d_open: HVC opening for slot t.
        """
        kots_pp, hvc_pp = pp
        kots = self.kots
        master_seed = sk

        ss = self._slot_seed(master_seed, t)
        osk, _ = kots.keygen(kots_pp, ss)
        Z = kots.sign(kots_pp, osk, m)

        def leaf_fn(slot):
            s = self._slot_seed(master_seed, slot)
            _, opk = kots.keygen(kots_pp, s)
            return opk

        d_open = self.hvc.open(hvc_pp, t, leaf_fn)
        return Z, d_open

    def sign_stateful(
        self, pp: tuple, sk_state: dict, m: bytes, t: int | None = None
    ) -> tuple:
        """Sign using a stateful signer key via the BDS08 auth-path traversal.

        The sk_state always carries a populated BDS traversal cache.  The
        "current slot" is derived from `sk_state["bds"]["phi"]` — there is
        no separate `next_slot` counter.  The optional `t` argument is a
        defensive check: if supplied, it must match `bds.phi`.

        Amortised cost per call: `(tau - K) / 2` leaf evaluations plus one
        KOTS sign, versus 2^tau for `sign_seed`.  The caller's `sk_state`
        is NOT mutated: the BDS cache is deep-copied before being
        advanced, so the input remains a valid pre-sign snapshot for
        rollback or reproduction.  Use the returned new state for the
        next call.

        Args:
            pp: scheme public parameters `(kots_pp, hvc_pp)`.
            sk_state: stateful sk dict from `make_stateful_sk` or a
                previous `sign_stateful` call.  Not modified.
            m: message bytes.
            t: optional slot override; must equal `sk_state["bds"]["phi"]`
                if given.

        Returns:
            `(sigma, new_sk_state, t_used)` where `sigma = (Z, d_open)`,
            `new_sk_state` is an independent dict with the BDS cache
            advanced to slot `t_used + 1`, and `t_used` is the slot that
            was signed.

        Raises:
            ValueError: slot out of range, mismatched against `bds.phi`,
                or the BDS state's internal phi does not match the slot
                being signed (raised by `HVC.bds_opening`).
        """
        master_seed = sk_state["master_seed"]
        bds_in = sk_state["bds"]
        if bds_in is None:
            raise ValueError("sk_state missing BDS cache; use sign_seed with an explicit slot")
        phi = int(bds_in["phi"])
        if t is None:
            t = phi
        if t != phi:
            raise ValueError(
                f"stateful signer is at slot {phi}, got {t}"
            )
        if not (0 <= t < self.n_slots):
            raise ValueError(f"slot {t} out of range [0, {self.n_slots - 1}]")

        kots_pp, hvc_pp = pp
        kots = self.kots
        leaf_fn = self._build_leaf_fn(kots_pp, master_seed)

        # Work on a deep-copied BDS state so the caller's input stays valid
        # as a pre-sign snapshot.
        bds = bds_copy_state(bds_in)

        # KOTS sign at slot t.
        ss = self._slot_seed(master_seed, t)
        osk, _ = kots.keygen(kots_pp, ss)
        Z = kots.sign(kots_pp, osk, m)

        # HVC opening assembled from the BDS auth path.  Any slot/phi
        # inconsistency is raised here as ValueError (survives python -O).
        d_open = self.hvc.bds_opening(bds, hvc_pp, t, leaf_fn)

        # Advance traversal state (unless we just signed the last leaf).
        if t + 1 < self.n_slots:
            self.hvc.bds_advance(bds, hvc_pp, leaf_fn)

        return (Z, d_open), self.make_stateful_sk(master_seed, bds), t

    def ivrfy(self, pp: tuple, pk: np.ndarray, t: int, m: bytes, sigma: tuple) -> bool:
        """iVrfy(pp, pk, t, m, sigma) -> bool.

        1. Recover KOTS opk from the HVC opening (iVrfy bound).
        2. Verify the KOTS signature against opk (iVrfy bound).
        Returns False on any error (malformed input, bad shapes, etc.).
        """
        try:
            kots_pp, hvc_pp = pp
            Z, d_open = sigma
            if t >= self.n_slots:
                return False
            opk = self.hvc.ivrfy(hvc_pp, pk, t, d_open)
            if opk is None:
                return False
            return self.kots.ivrfy(kots_pp, opk, m, Z)
        except Exception:
            return False

    def aggregate(
        self, pp: tuple, pks: list, t: int, m: bytes, sigs: list
    ) -> tuple | None:
        """Aggregate(pp, pks, t, m, sigs) -> (Z_agg, d_agg, attempt) or None.

        Rejects if any individual signature is invalid.
        Retries up to gamma times with different randomizer draws until avrfy passes.
        """
        if len(sigs) != len(pks):
            return None
        for pk, sig in zip(pks, sigs):
            if not self.ivrfy(pp, pk, t, m, sig):
                return None

        n    = len(pks)
        m_   = self.kots.m
        d    = self.kots.d

        for attempt in range(1, self.gamma + 1):
            ws = self._hash_to_randomizers(t, m, pks, attempt)

            Z_agg = np.zeros((m_, d), dtype=np.int64)
            for i in range(n):
                Z_agg += self.kots.ring.scale_vec(ws[i], sigs[i][0])

            d_agg = self._scale_opening(ws[0], sigs[0][1])
            for i in range(1, n):
                d_agg = self._add_openings(d_agg, self._scale_opening(ws[i], sigs[i][1]))

            if self.avrfy(pp, pks, t, m, (Z_agg, d_agg, attempt)):
                return Z_agg, d_agg, attempt

        return None

    def avrfy(
        self, pp: tuple, pks: list, t: int, m: bytes, sigma_agg: tuple
    ) -> bool:
        """aVrfy(pp, pks, t, m, sigma_agg) -> bool.

        1. Recompute randomizers from (t, m, pks, attempt).
        2. Aggregate commitments: c_agg = sum_i w^i * c_i.
        3. Recover aggregated KOTS opk via HVC.svrfy on c_agg.
        4. Verify the aggregated KOTS signature.
        Returns False on any error (malformed input, bad shapes, etc.).
        """
        try:
            kots_pp, hvc_pp = pp
            Z_agg, d_agg, attempt = sigma_agg
            ws      = self._hash_to_randomizers(t, m, pks, attempt)
            c_agg   = self._weighted_sum_commitments(ws, pks)
            opk_agg = self.hvc.svrfy(hvc_pp, c_agg, t, d_agg)
            if opk_agg is None:
                return False
            return self.kots.svrfy(kots_pp, opk_agg, m, Z_agg)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kots = KOTS()
    hvc  = HVC(kots, tau=3)  # small tau for quick self-test
    scheme = LEMUR(kots, hvc)

    N = hvc.n_slots   # use n_slots as signer count for quick test
    t = 2
    m = b"the committee approves"

    print("LEMUR parameters:")
    print(f"  d={kots.d}, tau={hvc.tau} ({hvc.n_slots} slots), N_signers={N}")
    print(f"  alpha_w={scheme.alpha_w}, gamma={scheme.gamma}")
    print(f"  beta_z={kots.beta_z}, beta_sigma={kots.beta_sigma}, beta_agg={hvc.beta_agg}")
    print()

    print("Setup ... ", end="", flush=True)
    pp = scheme.setup(b'\x01' * 32, b'\x02' * 32)
    print("done")

    print(f"KeyGen x{N} ... ", end="", flush=True)
    sks, pks = [], []
    for i in range(N):
        kgen_seed = bytes([i + 3]) + b'\x00' * 31
        sk_seed, _, pk = scheme.keygen(pp, kgen_seed)
        sks.append(sk_seed)
        pks.append(pk)
    print("done")

    print(f"Sign + iVrfy at slot t={t} ... ", end="", flush=True)
    sigs = []
    for i in range(N):
        sig = scheme.sign_seed(pp, sks[i], t, m)
        assert scheme.ivrfy(pp, pks[i], t, m, sig), f"iVrfy failed for signer {i}"
        sigs.append(sig)
    print("PASS")

    print(f"Aggregate ({N} signers) ... ", end="", flush=True)
    sigma_agg = scheme.aggregate(pp, pks, t, m, sigs)
    assert sigma_agg is not None, "Aggregate returned None (all gamma attempts failed)"
    _, _, attempt = sigma_agg
    print(f"done (attempt {attempt}/{scheme.gamma})")
    print(f"  Aggregated Z norm: {inf_norm(sigma_agg[0])}  (beta_sigma={kots.beta_sigma})")

    print("aVrfy ... ", end="", flush=True)
    ok = scheme.avrfy(pp, pks, t, m, sigma_agg)
    print(f"{'PASS' if ok else 'FAIL'}")

    print("\nNegative tests:")
    bad_msg  = scheme.avrfy(pp, pks, t, b"wrong message", sigma_agg)
    bad_slot = scheme.ivrfy(pp, pks[0], t + 1, m, sigs[0])
    bad_pks  = scheme.avrfy(pp, [pks[-1]] + pks[1:], t, m, sigma_agg)
    print(f"  Wrong message aVrfy: {'FAIL (expected)' if not bad_msg  else 'PASS (unexpected!)'}")
    print(f"  Wrong slot  iVrfy:   {'FAIL (expected)' if not bad_slot else 'PASS (unexpected!)'}")
    print(f"  Wrong pk    aVrfy:   {'FAIL (expected)' if not bad_pks  else 'PASS (unexpected!)'}")
