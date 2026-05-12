"""Regression tests for the proj_{eta,kappa} bound check in HVC vrfy.

Paper Fig. HVC step 4c bounds the ZZ-valued projection proj_{eta,kappa}
against q*beta/(2*eta) (see sections 6.2 and 6.7).  An earlier version
of this code reduced mod q before bounding, which silently capped the
inf-norm at (q-1)/2 and made the threshold vacuous in sVrfy/wVrfy where
q*beta/(2*eta) >> q/2.

These tests cover:
  1. _proj_label_zz returns the unreduced ZZ value.
  2. Honest decompositions agree on both projections.
  3. An all-eta synthetic label demonstrates the divergence: the ZZ
     projection exceeds q/2 (paper's iVrfy threshold), while the
     mod-q-centered projection does not.
  4. Honest sign + aggregate + sVrfy still passes after the fix.
"""

import numpy as np

from hvc import HVC
from kots import KOTS, inf_norm
from lemur import LEMUR


def test_proj_label_zz_matches_centered_modq_on_honest_decomp():
    hvc = HVC(kots=KOTS(), tau=3)
    rng = np.random.default_rng(0xC0FFEE)
    y = rng.integers(0, hvc.q, size=(hvc.omega, hvc.d), dtype=np.int64)
    digits = hvc._dec_vec(y, hvc.q, hvc.kappa)
    proj_zz = hvc._proj_label_zz(digits)
    proj_q = hvc._proj_label(digits)
    centered = proj_q.astype(np.int64).copy()
    centered[centered > hvc.q // 2] -= hvc.q
    for r in range(hvc.omega):
        for i in range(hvc.d):
            assert int(proj_zz[r, i]) == int(centered[r, i]), (
                f"honest decomp mismatch at (r={r}, i={i}): "
                f"zz={proj_zz[r, i]} centered={centered[r, i]}"
            )


def test_proj_label_zz_exceeds_q_over_2_for_max_eta_digits():
    """All digits at +eta saturate the (2eta+1)-ary representation past q/2."""
    hvc = HVC(kots=KOTS(), tau=3)
    label = np.full((hvc.omega * hvc.kappa, hvc.d), hvc.eta, dtype=np.int64)

    proj_zz = hvc._proj_label_zz(label)
    proj_centered = hvc._proj_label(label).astype(np.int64).copy()
    proj_centered[proj_centered > hvc.q // 2] -= hvc.q

    max_zz = max(abs(int(x)) for x in proj_zz.ravel())
    max_centered = int(np.max(np.abs(proj_centered)))

    expected_zz = ((2 * hvc.eta + 1) ** hvc.kappa - 1) // 2
    assert max_zz == expected_zz

    # Old check (mod-q centered) is bounded by q/2 — vacuously accepts.
    assert max_centered <= hvc.q // 2
    # New check (unreduced ZZ) exceeds q/2 — correctly rejects at iVrfy.
    assert max_zz > hvc.q // 2

    # At iVrfy, threshold = q*eta/(2*eta) = q/2 exactly.
    threshold_ivrfy = hvc.q * hvc.eta // (2 * hvc.eta)
    assert max_centered <= threshold_ivrfy   # what the old code did: ACCEPT
    assert max_zz > threshold_ivrfy          # what the paper requires: REJECT

    # |digits| == eta passes the inf_norm bound under iVrfy.
    assert inf_norm(label) == hvc.eta


def test_honest_aggregate_still_passes_svrfy():
    """End-to-end: an honest 4-signer aggregate at tau=3 still verifies."""
    kots = KOTS()
    hvc = HVC(kots=kots, tau=3)
    lemur = LEMUR(kots, hvc)
    pp = lemur.setup(b"proj-norm-regression-k" * 2, b"proj-norm-regression-h" * 2)

    keys = [lemur.keygen(pp, bytes([i + 1]) * 32) for i in range(4)]
    sk_seeds = [k[0] for k in keys]
    pks = [k[2] for k in keys]

    msg = b"unreduced projection regression"
    slot = 1

    sigs = [lemur.sign_seed(pp, sk, slot, msg) for sk in sk_seeds]
    for (sk_seed, _, pk), sig in zip(keys, sigs):
        assert lemur.ivrfy(pp, pk, slot, msg, sig), "ivrfy on honest sig must succeed"

    agg = lemur.aggregate(pp, pks, slot, msg, sigs)
    assert agg is not None, "honest aggregation must succeed"
    assert lemur.avrfy(pp, pks, slot, msg, agg), "honest aggregate must verify"


if __name__ == "__main__":
    print("test_proj_label_zz_matches_centered_modq_on_honest_decomp ...", end=" ")
    test_proj_label_zz_matches_centered_modq_on_honest_decomp()
    print("PASS")

    print("test_proj_label_zz_exceeds_q_over_2_for_max_eta_digits ...", end=" ")
    test_proj_label_zz_exceeds_q_over_2_for_max_eta_digits()
    print("PASS")

    print("test_honest_aggregate_still_passes_svrfy ...", end=" ")
    test_honest_aggregate_still_passes_svrfy()
    print("PASS")
