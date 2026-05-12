//! Robustness tests: verify that corrupted and truncated signatures
//! produce clean verification failures (no panics, no false accepts).

use lemur_rs::codec::{agg_sig_decode, agg_sig_encode, sig_decode, sig_encode};
use lemur_rs::lemur::{
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile_and_tau,
    lemur_sign_seed,
};
use lemur_rs::profile::DEFAULT;

const TAU: usize = 3;
const SLOT: usize = 5;
const MSG: &[u8] = b"robustness test";

/// Build a small test fixture: pp, 2 signers, individual + aggregated sigs.
fn fixture() -> Fixture {
    let pp = lemur_setup_with_profile_and_tau(&[0u8; 32], &[1u8; 32], DEFAULT, TAU);

    let (sk0, _sk0_state, pk0) = lemur_keygen(&pp, &[2u8; 32]);
    let (sk1, _sk1_state, pk1) = lemur_keygen(&pp, &[3u8; 32]);

    let sig0 = lemur_sign_seed(&pp, &sk0, SLOT, MSG);
    let sig1 = lemur_sign_seed(&pp, &sk1, SLOT, MSG);

    let sig0_bytes = sig_encode(&sig0, &pp.hvc_pp);
    let pks = vec![pk0, pk1];
    let sigs = vec![sig0, sig1];

    let agg = lemur_aggregate(&pp, &pks, SLOT, MSG, &sigs).expect("aggregation failed");
    let agg_bytes = agg_sig_encode(&agg, 2, &pp);

    Fixture {
        pp,
        pks,
        sig0_bytes,
        agg_bytes,
    }
}

struct Fixture {
    pp: lemur_rs::lemur::LemurPp,
    pks: Vec<lemur_rs::lemur::LemurPk>,
    sig0_bytes: Vec<u8>,
    agg_bytes: Vec<u8>,
}

// -----------------------------------------------------------------------
// Individual signature: bit flips
// -----------------------------------------------------------------------

#[test]
fn ind_sig_bitflip_rejected() {
    let f = fixture();
    let mut rejected = 0usize;
    let mut errors = Vec::new();

    for byte_idx in (0..f.sig0_bytes.len()).step_by(f.sig0_bytes.len().max(1) / 64 + 1) {
        for bit in 0..8 {
            let mut corrupted = f.sig0_bytes.clone();
            corrupted[byte_idx] ^= 1 << bit;

            let result = std::panic::catch_unwind(|| {
                let decoded = sig_decode(&corrupted, &f.pp, SLOT);
                match decoded {
                    Err(_) => true, // decode rejected — good
                    Ok(sig) => {
                        // decoded OK — verify must reject
                        lemur_ivrfy(&f.pp, &f.pks[0], SLOT, MSG, &sig).is_err()
                    }
                }
            });

            match result {
                Ok(true) => rejected += 1,
                Ok(false) => errors.push(format!(
                    "ACCEPTED corrupted sig (byte {byte_idx}, bit {bit})"
                )),
                Err(panic) => errors.push(format!(
                    "PANIC on corrupted sig (byte {byte_idx}, bit {bit}): {panic:?}"
                )),
            }
        }
    }

    let total = rejected + errors.len();
    eprintln!(
        "ind bitflip: {rejected}/{total} cleanly rejected, {} errors",
        errors.len()
    );
    for e in &errors {
        eprintln!("  {e}");
    }
    assert!(errors.is_empty(), "individual sig bitflip failures");
}

// -----------------------------------------------------------------------
// Individual signature: truncation
// -----------------------------------------------------------------------

#[test]
fn ind_sig_truncation_rejected() {
    let f = fixture();
    let mut rejected = 0usize;
    let mut errors = Vec::new();
    let len = f.sig0_bytes.len();

    // Test truncation at various points including 0, 1, 50%, 99%
    let positions: Vec<usize> = {
        let mut v: Vec<usize> = (0..len).step_by(len.max(1) / 32 + 1).collect();
        v.push(0);
        v.push(1);
        v.push(len.saturating_sub(1));
        v.sort();
        v.dedup();
        v
    };

    for &trunc_at in &positions {
        let truncated = &f.sig0_bytes[..trunc_at];

        let result = std::panic::catch_unwind(|| {
            let decoded = sig_decode(truncated, &f.pp, SLOT);
            match decoded {
                Err(_) => true,
                Ok(sig) => lemur_ivrfy(&f.pp, &f.pks[0], SLOT, MSG, &sig).is_err(),
            }
        });

        match result {
            Ok(true) => rejected += 1,
            Ok(false) => errors.push(format!("ACCEPTED truncated sig at {trunc_at}/{len}")),
            Err(panic) => errors.push(format!(
                "PANIC on truncated sig at {trunc_at}/{len}: {panic:?}"
            )),
        }
    }

    let total = rejected + errors.len();
    eprintln!(
        "ind truncation: {rejected}/{total} cleanly rejected, {} errors",
        errors.len()
    );
    for e in &errors {
        eprintln!("  {e}");
    }
    assert!(errors.is_empty(), "individual sig truncation failures");
}

// -----------------------------------------------------------------------
// Aggregated signature: bit flips
// -----------------------------------------------------------------------

#[test]
fn agg_sig_bitflip_rejected() {
    let f = fixture();
    let mut rejected = 0usize;
    let mut errors = Vec::new();

    for byte_idx in (0..f.agg_bytes.len()).step_by(f.agg_bytes.len().max(1) / 64 + 1) {
        for bit in 0..8 {
            let mut corrupted = f.agg_bytes.clone();
            corrupted[byte_idx] ^= 1 << bit;

            let result = std::panic::catch_unwind(|| {
                let decoded = agg_sig_decode(&corrupted, &f.pp, SLOT, 2);
                match decoded {
                    Err(_) => true,
                    Ok(agg) => lemur_avrfy(&f.pp, &f.pks, SLOT, MSG, &agg).is_err(),
                }
            });

            match result {
                Ok(true) => rejected += 1,
                Ok(false) => errors.push(format!(
                    "ACCEPTED corrupted agg sig (byte {byte_idx}, bit {bit})"
                )),
                Err(panic) => errors.push(format!(
                    "PANIC on corrupted agg sig (byte {byte_idx}, bit {bit}): {panic:?}"
                )),
            }
        }
    }

    let total = rejected + errors.len();
    eprintln!(
        "agg bitflip: {rejected}/{total} cleanly rejected, {} errors",
        errors.len()
    );
    for e in &errors {
        eprintln!("  {e}");
    }
    assert!(errors.is_empty(), "aggregated sig bitflip failures");
}

// -----------------------------------------------------------------------
// Aggregated signature: truncation
// -----------------------------------------------------------------------

#[test]
fn agg_sig_truncation_rejected() {
    let f = fixture();
    let mut rejected = 0usize;
    let mut errors = Vec::new();
    let len = f.agg_bytes.len();

    let positions: Vec<usize> = {
        let mut v: Vec<usize> = (0..len).step_by(len.max(1) / 32 + 1).collect();
        v.push(0);
        v.push(1);
        v.push(len.saturating_sub(1));
        v.sort();
        v.dedup();
        v
    };

    for &trunc_at in &positions {
        let truncated = &f.agg_bytes[..trunc_at];

        let result = std::panic::catch_unwind(|| {
            let decoded = agg_sig_decode(truncated, &f.pp, SLOT, 2);
            match decoded {
                Err(_) => true,
                Ok(agg) => lemur_avrfy(&f.pp, &f.pks, SLOT, MSG, &agg).is_err(),
            }
        });

        match result {
            Ok(true) => rejected += 1,
            Ok(false) => errors.push(format!("ACCEPTED truncated agg sig at {trunc_at}/{len}")),
            Err(panic) => errors.push(format!(
                "PANIC on truncated agg sig at {trunc_at}/{len}: {panic:?}"
            )),
        }
    }

    let total = rejected + errors.len();
    eprintln!(
        "agg truncation: {rejected}/{total} cleanly rejected, {} errors",
        errors.len()
    );
    for e in &errors {
        eprintln!("  {e}");
    }
    assert!(errors.is_empty(), "aggregated sig truncation failures");
}

// -----------------------------------------------------------------------
// Wrong message / wrong slot / wrong pk
// -----------------------------------------------------------------------

#[test]
fn wrong_context_rejected() {
    let f = fixture();

    // Decode a valid sig
    let sig0 = sig_decode(&f.sig0_bytes, &f.pp, SLOT).expect("valid sig decode");

    // Wrong message
    assert!(
        lemur_ivrfy(&f.pp, &f.pks[0], SLOT, b"wrong message", &sig0).is_err(),
        "wrong message should fail"
    );

    // Wrong slot
    assert!(
        lemur_ivrfy(&f.pp, &f.pks[0], SLOT + 1, MSG, &sig0).is_err(),
        "wrong slot should fail"
    );

    // Wrong public key (pk1 instead of pk0)
    assert!(
        lemur_ivrfy(&f.pp, &f.pks[1], SLOT, MSG, &sig0).is_err(),
        "wrong pk should fail"
    );

    // Correct context should pass
    assert!(
        lemur_ivrfy(&f.pp, &f.pks[0], SLOT, MSG, &sig0).is_ok(),
        "correct context should pass"
    );
}
