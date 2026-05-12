//! Regression tests for the proj_{eta,kappa} bound check in HVC vrfy.
//!
//! Paper Fig. HVC step 4c bounds the ZZ-valued projection proj_{eta,kappa}
//! against q*beta/(2*eta) (see Section 6.2).  An earlier version reduced
//! mod q before bounding, which silently capped the inf-norm at (q-1)/2
//! and made the threshold vacuous in sVrfy/wVrfy where q*beta/(2*eta) >> q/2.
//!
//! Mirrors lemur-py/test_proj_norm.py.
//!
//! Also covers an i64-overflow check: wVrfy's threshold q*2*beta_agg/(2*eta)
//! is ~2^63.2 — the i128 type is mandatory for the unreduced projection
//! arithmetic used by the bound check.

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

    for i in 0..(omega * d) {
        let centered = center(proj_q[i], q) as i128;
        assert_eq!(
            centered, proj_zz[i],
            "centered mod q must equal ZZ-proj at i={i}"
        );
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
fn wvrfy_threshold_overflows_i64() {
    // wVrfy uses beta = 2*beta_agg.  The threshold q*beta/(2*eta) and the
    // numerator q*beta must fit in the arithmetic type.  Demonstrates why
    // the projection helper must be i128, not i64.
    let profile: &'static Profile = &D256_K4;
    let q = profile.q_hvc() as i128;
    let eta = profile.eta as i128;
    let tau: usize = 20; // shipped depth
    let beta_agg = lemur_rs::hvc::compute_beta_agg_with_profile(tau, profile) as i128;

    let threshold = q * (2 * beta_agg) / (2 * eta);
    let numerator = q * (2 * beta_agg);

    assert!(
        threshold > i64::MAX as i128,
        "wVrfy threshold {threshold} fits in i64 — i128 is unneeded?"
    );
    assert!(
        numerator > i64::MAX as i128,
        "q * 2*beta_agg {numerator} fits in i64 — i128 is unneeded?"
    );
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
