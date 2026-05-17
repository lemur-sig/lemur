//! End-to-end coverage for the shipped parameter set.

use lemur_rs::codec::{
    agg_sig_decode, agg_sig_encode, compute_agg_encoding, pk_decode_with_profile,
    pk_encode_with_profile, sig_decode, sig_encode, sk_state_decode_with_profile,
    sk_state_encode_with_profile, EncMode,
};
use lemur_rs::lemur::{
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau,
    lemur_sign_seed, lemur_sign_stateful_mut,
};
use lemur_rs::profile::{Profile, D256_K4};

const TAU: usize = 3;

fn full_pipeline(profile: &'static Profile, msg: &[u8]) {
    let kots_seed = [0x5au8; 32];
    let hvc_seed = [0xa5u8; 32];

    let pp = lemur_setup_with_profile_and_tau(&kots_seed, &hvc_seed, profile, TAU);
    assert!(std::ptr::eq(pp.profile, profile));

    let (sk0, mut state0, pk0) = lemur_keygen(&pp, &[1u8; 32]);
    let (sk1, _state1, pk1) = lemur_keygen(&pp, &[2u8; 32]);

    let pk0_bytes = pk_encode_with_profile(&pk0, profile);
    let pk0_back = pk_decode_with_profile(&pk0_bytes, profile).expect("pk0 decode");
    assert_eq!(pk0_back.0 .0, pk0.0 .0);

    let state0_bytes = sk_state_encode_with_profile(&state0, profile).expect("state0 encode");
    let _state0_back =
        sk_state_decode_with_profile(&state0_bytes, profile).expect("state0 decode");

    let sig0 = lemur_sign_seed(&pp, &sk0, 0, msg);
    let sig1 = lemur_sign_seed(&pp, &sk1, 0, msg);
    lemur_ivrfy(&pp, &pk0, 0, msg, &sig0).expect("ivrfy 0 ok");
    lemur_ivrfy(&pp, &pk1, 0, msg, &sig1).expect("ivrfy 1 ok");

    let (sig0_stateful, slot_used) =
        lemur_sign_stateful_mut(&pp, &mut state0, msg, Some(0)).expect("stateful sign");
    assert_eq!(slot_used, 0);
    assert_eq!(sig0_stateful.z.0, sig0.z.0, "stateful/seed KOTS sig drift");
    lemur_ivrfy(&pp, &pk0, 0, msg, &sig0_stateful).expect("stateful ivrfy");

    let sig0_bytes = sig_encode(&sig0, &pp.hvc_pp);
    let sig0_back = sig_decode(&sig0_bytes, &pp, 0).expect("sig decode");
    lemur_ivrfy(&pp, &pk0, 0, msg, &sig0_back).expect("decoded ivrfy");

    let pks = vec![pk0.clone(), pk1.clone()];
    let sigs = vec![sig0, sig1];
    let agg = lemur_aggregate(&pp, &pks, 0, msg, &sigs).expect("aggregate");
    lemur_avrfy(&pp, &pks, 0, msg, &agg).expect("avrfy");

    let agg_bytes = agg_sig_encode(&agg, pks.len(), &pp);
    let agg_back = agg_sig_decode(&agg_bytes, &pp, 0, pks.len()).expect("agg decode");
    lemur_avrfy(&pp, &pks, 0, msg, &agg_back).expect("decoded avrfy");
}

#[test]
fn d256_k4_full_pipeline() {
    full_pipeline(&D256_K4, b"d256_k4 pipeline");
}

#[test]
fn default_cell_bounds_are_cell_scoped() {
    D256_K4.validate();
    assert_eq!(
        D256_K4.beta_encode,
        (D256_K4.beta_agg + 2 * D256_K4.eta - 1) / (2 * D256_K4.eta),
        "beta_encode drifted from the representative ODS cell",
    );
}

#[test]
fn tau_extremes_recompute_runtime_hvc_bounds() {
    for tau in [12usize, 24usize] {
        let p = &D256_K4;
        let pp = lemur_setup_with_profile_and_tau(&[0x31u8; 32], &[0x32u8; 32], p, tau);
        assert_eq!(pp.hvc_pp.tau, tau);
        assert_eq!(
            pp.hvc_pp.beta_encode,
            (pp.hvc_pp.beta_agg + 2 * p.eta - 1) / (2 * p.eta),
            "{} tau={tau}: runtime beta_encode must follow runtime beta_agg",
            p.name
        );
    }
}

#[test]
fn agg_encoding_snapshots_match_python_formula() {
    let cases = [
        (&D256_K4, 1024usize, 378_933i64, 20usize, 5usize, 15usize),
        (&D256_K4, 2usize, 16_747i64, 16usize, 0usize, 10usize),
    ];

    for (profile, n_signers, zagg_bound, zagg_dx, babai_k, agg_k) in cases {
        let pp = lemur_setup_with_profile_and_tau(
            &[0x41u8; 32],
            &[0x42u8; 32],
            profile,
            profile.tau,
        );
        let enc = compute_agg_encoding(&pp, n_signers);
        assert_eq!(enc.zagg_bound, zagg_bound, "{} N={n_signers}", profile.name);
        assert_eq!(enc.zagg_dx, zagg_dx, "{} N={n_signers}", profile.name);
        assert!(matches!(enc.babai_mode, EncMode::Rice { rice_k } if rice_k == babai_k));
        assert!(matches!(enc.agg_mode, EncMode::Rice { rice_k } if rice_k == agg_k));
        assert_eq!(enc.babai_bound, pp.hvc_pp.beta_encode);
        assert_eq!(enc.agg_bound, pp.hvc_pp.beta_agg);
    }
}
