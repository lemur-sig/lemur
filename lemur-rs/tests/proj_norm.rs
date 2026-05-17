//! Regression tests for the proj_{eta,kappa} bound check in HVC vrfy.
//!
//! Paper Fig. HVC step 4c bounds the ZZ-valued projection proj_{eta,kappa}
//! against q*beta/(2*eta) (see Section 6.2).  An earlier version reduced
//! mod q before bounding, which silently capped the inf-norm at (q-1)/2
//! and made the threshold vacuous in sVrfy/wVrfy where q*beta/(2*eta) >> q/2.
//!
//! Mirrors lemur-py/test_proj_norm.py.

use lemur_rs::hvc::{
    proj_label_with_profile, proj_label_zz_with_profile,
};
use lemur_rs::lemur::{
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau,
    lemur_sign_seed,
};
use lemur_rs::profile::{Profile, D256_K4};

fn center(x: i64, q: i64) -> i64 {
    if x > q / 2 {
        x - q
    } else {
        x
    }
}

#[test]
fn proj_label_zz_matches_centered_modq_on_arbitrary_label() {
    // For any label (digit polynomial), the centered representative of
    // proj_q is congruent to proj_zz mod q.  They are equal whenever
    // proj_zz lands inside [-(q-1)/2, (q-1)/2].
    let profile: &'static Profile = &D256_K4;
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let eta = profile.eta;
    let q = profile.q_hvc() as i64;

    // Construct a small synthetic label with |digits| <= eta (honest range).
    // Use a deterministic pattern so the test is reproducible.
    let mut label = vec![0i64; omega * kappa * d];
    for k in 0..(omega * kappa) {
        for i in 0..d {
            let v = ((k * d + i) as i64) % (2 * eta + 1) - eta;
            label[k * d + i] = v;
        }
    }

    let proj_q = proj_label_with_profile(&label, profile);
    let proj_zz = proj_label_zz_with_profile(&label, profile);
    assert_eq!(proj_q.len(), omega * d);
    assert_eq!(proj_zz.len(), omega * d);

    let half_q = (q as i128 - 1) / 2;
    for i in 0..(omega * d) {
        let centered = center(proj_q[i], q) as i128;
        // Congruence mod q always holds.
        assert_eq!(
            (proj_zz[i] - centered).rem_euclid(q as i128),
            0,
            "centered mod q must be congruent to ZZ-proj at i={i}"
        );
        // Equality holds whenever proj_zz lands inside [-(q-1)/2, (q-1)/2].
        if proj_zz[i].abs() <= half_q {
            assert_eq!(
                centered, proj_zz[i],
                "centered must equal ZZ-proj in the central band at i={i}"
            );
        }
    }
}

#[test]
fn proj_label_zz_exceeds_q_over_2_for_max_eta_digits() {
    // All-eta digits push ZZ-projection past q/2 = iVrfy threshold.
    // This is the regime where the old mod-q-centered check silently
    // capped to q/2 and accepted.
    let profile: &'static Profile = &D256_K4;
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let eta = profile.eta;
    let q = profile.q_hvc() as i64;

    let label = vec![eta; omega * kappa * d];

    let proj_zz = proj_label_zz_with_profile(&label, profile);
    let proj_q = proj_label_with_profile(&label, profile);
    let centered: Vec<i64> = proj_q.iter().map(|&x| center(x, q)).collect();

    let max_zz = proj_zz.iter().map(|x| x.abs()).max().unwrap();
    let max_centered = centered.iter().map(|x| x.abs()).max().unwrap();

    // Closed form: sum_{k=0}^{kappa-1} eta * (2*eta+1)^k = ((2*eta+1)^kappa - 1)/2
    let b_val = (2 * eta + 1) as i128;
    let mut expected: i128 = 0;
    let mut base: i128 = 1;
    for _ in 0..kappa {
        expected += eta as i128 * base;
        base *= b_val;
    }
    assert_eq!(max_zz, expected);

    // Old mod-q-centered check: bounded by q/2 — vacuously accepts.
    assert!(max_centered <= q / 2);
    // New ZZ check: exceeds q/2 — correctly rejects at iVrfy.
    assert!(max_zz > q as i128 / 2);

    // At iVrfy, threshold = q*eta/(2*eta) = q/2 exactly.
    let threshold_ivrfy = (q as i128 * eta as i128) / (2 * eta) as i128;
    assert!((max_centered as i128) <= threshold_ivrfy); // old code: ACCEPT
    assert!(max_zz > threshold_ivrfy); // new code: REJECT (paper's intent)
}

#[test]
fn honest_aggregate_still_passes_avrfy() {
    // End-to-end: an honest 2-signer aggregate at tau=3 still verifies
    // after the proj-norm check is tightened.
    let profile: &'static Profile = &D256_K4;
    let kots_seed = [0x5au8; 32];
    let hvc_seed = [0xa5u8; 32];
    let pp = lemur_setup_with_profile_and_tau(&kots_seed, &hvc_seed, profile, 3);

    let (sk0, _state0, pk0) = lemur_keygen(&pp, &[1u8; 32]);
    let (sk1, _state1, pk1) = lemur_keygen(&pp, &[2u8; 32]);

    let msg = b"proj-norm regression after fix";
    let slot = 1usize;

    let sig0 = lemur_sign_seed(&pp, &sk0, slot, msg);
    let sig1 = lemur_sign_seed(&pp, &sk1, slot, msg);
    lemur_ivrfy(&pp, &pk0, slot, msg, &sig0).expect("ivrfy 0 must accept honest sig");
    lemur_ivrfy(&pp, &pk1, slot, msg, &sig1).expect("ivrfy 1 must accept honest sig");

    let pks = vec![pk0, pk1];
    let sigs = vec![sig0, sig1];
    let agg = lemur_aggregate(&pp, &pks, slot, msg, &sigs).expect("honest aggregate must succeed");
    lemur_avrfy(&pp, &pks, slot, msg, &agg).expect("honest aggregate must verify");
}
