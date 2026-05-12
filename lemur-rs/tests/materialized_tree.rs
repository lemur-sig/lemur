//! Integration tests for the optional in-memory materialized HVC tree.
//!
//! Covers:
//!   - `MaterializedHvcTree::build` produces the same root as `lemur_keygen`
//!   - `lemur_sign_tree` is byte-equal to `lemur_sign_seed` on every slot
//!   - signatures produced via the tree path verify under `lemur_ivrfy`
//!   - byte-size formula matches the concrete buffer size
//!   - τ mismatch returns a clean `Err`, not a panic
//!
//! These run at τ = 3 so they complete in seconds.

use lemur_rs::kots::kots_pk_from_seed;
use lemur_rs::lemur::{
    lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau, lemur_sign_seed, LemurSk,
};
use lemur_rs::materialized::{lemur_sign_tree, MaterializedHvcTree};
use lemur_rs::profile::DEFAULT;
use lemur_rs::sample::slot_seed;

const TAU: usize = 3;
const MSG: &[u8] = b"materialized-tree test";

fn sigs_byte_equal(a: &lemur_rs::lemur::LemurSig, b: &lemur_rs::lemur::LemurSig) -> bool {
    a.z.0 == b.z.0
        && a.opening.u == b.opening.u
        && a.opening.path_labels == b.opening.path_labels
        && a.opening.sibling_labels == b.opening.sibling_labels
}

#[test]
fn build_commits_to_same_root_as_keygen() {
    let pp = lemur_setup_with_profile_and_tau(&[0u8; 32], &[1u8; 32], DEFAULT, TAU);
    let seed = [2u8; 32];
    let (_sk, _sk_state, pk) = lemur_keygen(&pp, &seed);

    let master_seed = seed;
    let leaf_fn = |t: usize| {
        let ss = slot_seed(&master_seed, t);
        kots_pk_from_seed(&pp.kots_a, &ss, pp.profile).0
    };
    let tree = MaterializedHvcTree::build(&pp.hvc_pp, leaf_fn);

    assert_eq!(
        tree.commitment().0,
        pk.0 .0,
        "tree root must match keygen commitment"
    );
}

#[test]
fn tree_sign_matches_seed_path_every_slot() {
    let pp = lemur_setup_with_profile_and_tau(&[3u8; 32], &[4u8; 32], DEFAULT, TAU);
    let seed = [5u8; 32];
    let (sk, _sk_state, pk) = lemur_keygen(&pp, &seed);

    let leaf_fn = {
        let ms = seed;
        let kots_a = pp.kots_a.clone();
        let profile = pp.profile;
        move |t: usize| {
            let ss = slot_seed(&ms, t);
            kots_pk_from_seed(&kots_a, &ss, profile).0
        }
    };
    let tree = MaterializedHvcTree::build(&pp.hvc_pp, leaf_fn);

    for t in 0..(1usize << TAU) {
        let seed_sig = lemur_sign_seed(&pp, &sk, t, MSG);
        let tree_sig = lemur_sign_tree(&pp, &sk, &tree, t, MSG).expect("tree sign");

        assert!(
            sigs_byte_equal(&seed_sig, &tree_sig),
            "sig mismatch at t={t}"
        );

        // Both must verify.
        lemur_ivrfy(&pp, &pk, t, MSG, &tree_sig).expect("tree-sig ivrfy");
        lemur_ivrfy(&pp, &pk, t, MSG, &seed_sig).expect("seed-sig ivrfy");
    }
}

#[test]
fn byte_size_formula_matches_buffer_size() {
    // Projected labels are stored as i64 (8 B each).  Per-node stride is
    // `omega * d` coefficients; the default profile carries omega=2, d=256.
    let bytes_per_coeff = std::mem::size_of::<i64>();
    let stride = DEFAULT.omega * DEFAULT.d;
    let n_nodes = (1usize << (TAU + 1)) - 1;
    let expected = n_nodes * stride * bytes_per_coeff;
    assert_eq!(
        MaterializedHvcTree::byte_size_with_profile(TAU, DEFAULT),
        expected
    );
}

#[test]
fn tau_mismatch_returns_error() {
    let pp_small = lemur_setup_with_profile_and_tau(&[6u8; 32], &[7u8; 32], DEFAULT, 3);
    let seed = [8u8; 32];
    let leaf_fn = {
        let ms = seed;
        let kots_a = pp_small.kots_a.clone();
        let profile = pp_small.profile;
        move |t: usize| {
            let ss = slot_seed(&ms, t);
            kots_pk_from_seed(&kots_a, &ss, profile).0
        }
    };
    let tree = MaterializedHvcTree::build(&pp_small.hvc_pp, leaf_fn);

    // Same seeds, different τ.
    let pp_big = lemur_setup_with_profile_and_tau(&[6u8; 32], &[7u8; 32], DEFAULT, 4);

    let sk = LemurSk { master_seed: seed };
    let result = lemur_sign_tree(&pp_big, &sk, &tree, 0, MSG);
    assert!(
        result.is_err(),
        "tree tau mismatch must return Err, not panic"
    );
}

#[test]
fn slot_out_of_range_returns_error() {
    let pp = lemur_setup_with_profile_and_tau(&[9u8; 32], &[10u8; 32], DEFAULT, TAU);
    let seed = [11u8; 32];
    let (sk, _sk_state, _pk) = lemur_keygen(&pp, &seed);

    let leaf_fn = {
        let ms = seed;
        let kots_a = pp.kots_a.clone();
        let profile = pp.profile;
        move |t: usize| {
            let ss = slot_seed(&ms, t);
            kots_pk_from_seed(&kots_a, &ss, profile).0
        }
    };
    let tree = MaterializedHvcTree::build(&pp.hvc_pp, leaf_fn);

    let result = lemur_sign_tree(&pp, &sk, &tree, 1 << TAU, MSG);
    assert!(
        result.is_err(),
        "out-of-range slot must return Err, not panic"
    );
}
