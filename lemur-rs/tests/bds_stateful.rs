//! BDS08 stateful-signing integration tests.
//!
//! Verify that `lemur_sign_stateful` (BDS-backed) produces byte-identical
//! output to `lemur_sign_seed` (targeted-traversal open) on every slot,
//! plus the on-disk BDS state format properties:
//!   - snapshot preservation: signing from a shared `sk_state` twice
//!     produces the same signature; the input state stays at its old phi
//!   - encode/decode round-trip preserves behaviour on every slot
//!   - mid-state round-trip with a non-trivial populated keep/treehash
//!   - no magic bytes: encoded blob starts with the master seed
//!   - frozen byte-offset header layout (cross-implementation contract)
//!   - slot mismatch returns a `LemurError`, never a panic

use lemur_rs::codec::{sk_state_decode_with_profile, sk_state_encode_with_profile};
use lemur_rs::lemur::{
    lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau, lemur_sign_seed,
    lemur_sign_stateful, LemurSk,
};
use lemur_rs::profile::DEFAULT;

const TAU: usize = 3;
const MSG: &[u8] = b"bds stateful";

fn openings_byte_equal(a: &lemur_rs::hvc::HvcOpening, b: &lemur_rs::hvc::HvcOpening) -> bool {
    a.u == b.u && a.path_labels == b.path_labels && a.sibling_labels == b.sibling_labels
}

fn sigs_byte_equal(a: &lemur_rs::lemur::LemurSig, b: &lemur_rs::lemur::LemurSig) -> bool {
    a.z.0 == b.z.0 && openings_byte_equal(&a.opening, &b.opening)
}

#[test]
fn stateful_equals_seed_every_slot() {
    let pp = lemur_setup_with_profile_and_tau(&[0u8; 32], &[1u8; 32], DEFAULT, TAU);
    let (_sk_seed, sk_state0, pk) = lemur_keygen(&pp, &[2u8; 32]);
    let n_slots = 1usize << TAU;

    let mut state = sk_state0;
    for t in 0..n_slots {
        let seed_sk = LemurSk {
            master_seed: state.master_seed,
        };
        let seed_sig = lemur_sign_seed(&pp, &seed_sk, t, MSG);

        let (st_sig, next_state, t_used) =
            lemur_sign_stateful(&pp, &state, MSG, None).expect("sign_stateful");
        assert_eq!(t_used, t);

        assert!(sigs_byte_equal(&seed_sig, &st_sig), "sig mismatch at t={t}");

        lemur_ivrfy(&pp, &pk, t, MSG, &seed_sig).expect("seed ivrfy");
        lemur_ivrfy(&pp, &pk, t, MSG, &st_sig).expect("stateful ivrfy");

        state = next_state;
    }
}

#[test]
fn snapshot_preservation() {
    let pp = lemur_setup_with_profile_and_tau(&[4u8; 32], &[5u8; 32], DEFAULT, TAU);
    let (_sk_seed, sk_state, _pk) = lemur_keygen(&pp, &[6u8; 32]);

    // Sign from sk_state.
    let (sig_a, new_state, t_a) = lemur_sign_stateful(&pp, &sk_state, MSG, None).expect("sign a");
    assert_eq!(t_a, 0);

    // sk_state must be untouched — sign slot 0 again from the same input.
    let (sig_b, _new_state_b, t_b) =
        lemur_sign_stateful(&pp, &sk_state, MSG, None).expect("sign b");
    assert_eq!(t_b, 0);

    assert!(
        sigs_byte_equal(&sig_a, &sig_b),
        "snapshot-preserved second sign didn't match the first"
    );

    // Internal invariant: the old state's BDS phi stays at 0, the new
    // state's phi is 1.
    assert_eq!(sk_state.bds.phi, 0);
    assert_eq!(new_state.bds.phi, 1);
}

#[test]
fn slot_mismatch_returns_error() {
    let pp = lemur_setup_with_profile_and_tau(&[10u8; 32], &[11u8; 32], DEFAULT, TAU);
    let (_sk_seed, sk_state, _pk) = lemur_keygen(&pp, &[12u8; 32]);

    // sk_state is at phi=0; asking for slot 2 is a user error.
    let result = lemur_sign_stateful(&pp, &sk_state, MSG, Some(2));
    assert!(result.is_err(), "slot mismatch must return Err");
}

#[test]
fn tau_mismatch_returns_error() {
    // Key a stateful signer under tau=3, then try to sign against a
    // pp whose tau=4 (say, from a differently-configured installation).
    // Historically this panicked deep inside `bds_opening` when the auth
    // path was shorter than the verifier's tree depth.
    let pp_small = lemur_setup_with_profile_and_tau(&[20u8; 32], &[21u8; 32], DEFAULT, 3);
    let (_sk, sk_state_small, _pk) = lemur_keygen(&pp_small, &[22u8; 32]);

    let pp_big = lemur_setup_with_profile_and_tau(&[20u8; 32], &[21u8; 32], DEFAULT, 4);

    let result = lemur_sign_stateful(&pp_big, &sk_state_small, MSG, None);
    assert!(result.is_err(), "tau mismatch must return Err, not panic");
}

// ---------------------------------------------------------------------------
// On-disk BDS state file (no magic bytes)
// ---------------------------------------------------------------------------

#[test]
fn state_file_round_trip_matches_seed_path() {
    // Encode a keygen-produced state, decode it back, sign every slot via
    // the decoded state, and confirm signatures match the seed path.  This
    // is the core persistence-across-processes property.
    let pp = lemur_setup_with_profile_and_tau(&[0u8; 32], &[1u8; 32], DEFAULT, TAU);
    let seed = [2u8; 32];
    let (_sk_seed, sk_state0, pk) = lemur_keygen(&pp, &seed);

    let enc = sk_state_encode_with_profile(&sk_state0, DEFAULT).expect("encode");
    // No magic bytes: the blob starts with the master seed verbatim.
    assert_eq!(&enc[..32], &seed);
    // phi / tau / k header.
    assert_eq!(
        u32::from_le_bytes(enc[32..36].try_into().unwrap()),
        0,
        "phi must be 0 right after keygen"
    );
    assert_eq!(
        u32::from_le_bytes(enc[36..40].try_into().unwrap()) as usize,
        TAU
    );

    let decoded = sk_state_decode_with_profile(&enc, DEFAULT).expect("decode");
    assert_eq!(decoded.bds.phi, 0);

    let seed_sk = LemurSk { master_seed: seed };
    let mut state = decoded;
    for t in 0..(1usize << TAU) {
        let seed_sig = lemur_sign_seed(&pp, &seed_sk, t, MSG);
        let (st_sig, next_state, t_used) =
            lemur_sign_stateful(&pp, &state, MSG, None).expect("stateful sign");
        assert_eq!(t_used, t);
        assert!(sigs_byte_equal(&seed_sig, &st_sig), "sig mismatch at t={t}");
        lemur_ivrfy(&pp, &pk, t, MSG, &st_sig).expect("ivrfy");
        state = next_state;
    }
}

#[test]
fn state_file_mid_round_trip() {
    // Advance the state a few slots, encode/decode, then sign at the current
    // slot and confirm the result matches the seed path.  Exercises a
    // non-trivial populated keep/treehash/retain layout.
    let pp = lemur_setup_with_profile_and_tau(&[3u8; 32], &[4u8; 32], DEFAULT, TAU);
    let seed = [5u8; 32];
    let (_sk_seed, sk_state0, _pk) = lemur_keygen(&pp, &seed);

    // Advance to slot 4 of 8.
    let mut state = sk_state0;
    for _ in 0..4 {
        let (_, next_state, _) = lemur_sign_stateful(&pp, &state, MSG, None).expect("advance");
        state = next_state;
    }
    assert_eq!(state.bds.phi, 4);

    let enc = sk_state_encode_with_profile(&state, DEFAULT).expect("encode mid");
    assert_eq!(
        u32::from_le_bytes(enc[32..36].try_into().unwrap()),
        4,
        "encoded phi must be 4"
    );
    let decoded = sk_state_decode_with_profile(&enc, DEFAULT).expect("decode mid");
    assert_eq!(decoded.bds.phi, 4);

    // Continue signing from the decoded mid-state.
    let seed_sk = LemurSk { master_seed: seed };
    let (st_sig, _next, t_used) =
        lemur_sign_stateful(&pp, &decoded, MSG, None).expect("post-decode sign");
    assert_eq!(t_used, 4);
    let seed_sig = lemur_sign_seed(&pp, &seed_sk, 4, MSG);
    assert!(
        sigs_byte_equal(&seed_sig, &st_sig),
        "post-decode sign mismatch at t=4"
    );
}

#[test]
fn state_file_header_layout_is_frozen() {
    // Hand-verify header byte offsets so the cross-implementation format
    // stays frozen: master_seed(32) || phi_u32 || tau_u32 || k_u32.  No
    // magic bytes.
    let pp = lemur_setup_with_profile_and_tau(&[9u8; 32], &[10u8; 32], DEFAULT, TAU);
    let seed = [11u8; 32];
    let (_sk_seed, state0, _pk) = lemur_keygen(&pp, &seed);

    let enc = sk_state_encode_with_profile(&state0, DEFAULT).expect("encode");
    assert_eq!(&enc[0..32], &seed);
    assert_eq!(&enc[32..36], &0u32.to_le_bytes()); // phi
    assert_eq!(&enc[36..40], &(TAU as u32).to_le_bytes()); // tau
                                                           // K = 3 for odd TAU = 3
    assert_eq!(&enc[40..44], &3u32.to_le_bytes());
}

#[test]
fn state_file_rejects_truncation_and_trailing_bytes() {
    let pp = lemur_setup_with_profile_and_tau(&[12u8; 32], &[13u8; 32], DEFAULT, TAU);
    let (_sk_seed, state0, _pk) = lemur_keygen(&pp, &[14u8; 32]);
    let enc = sk_state_encode_with_profile(&state0, DEFAULT).expect("encode");

    // Empty.
    assert!(sk_state_decode_with_profile(&[], DEFAULT).is_err());

    // Truncated header (less than 32 + 4 + 4 + 4 = 44 bytes).
    assert!(sk_state_decode_with_profile(&enc[..20], DEFAULT).is_err());
    assert!(sk_state_decode_with_profile(&enc[..43], DEFAULT).is_err());

    // Truncated body (cuts into auth section).
    assert!(sk_state_decode_with_profile(&enc[..100], DEFAULT).is_err());

    // Trailing bytes.
    let mut trailing = enc.clone();
    trailing.push(0);
    assert!(sk_state_decode_with_profile(&trailing, DEFAULT).is_err());
}
