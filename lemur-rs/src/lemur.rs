//! Top-level Lemur multi-signature scheme.

use sha3::digest::XofReader;

use crate::aux_ntt::CrtBackend;
use crate::error::LemurError;
use crate::hvc::{
    add_openings, bds_advance, bds_init, bds_opening, hvc_ivrfy, hvc_open_with_known_leaf,
    hvc_setup_with_profile, hvc_setup_with_profile_and_tau, hvc_svrfy, scale_opening_any, BdsState,
    HvcCom, HvcOpening, HvcPp,
};
use crate::kots::{
    kots_ivrfy, kots_keygen, kots_pk_from_seed, kots_setup, kots_sign, kots_svrfy, KotsA, KotsSig,
};
use crate::profile::{HvcRing, Profile};
use crate::sample::{agg_randomizer_xof, slot_seed, xof_ternary_poly};
use rayon::prelude::*;

/// Multiply each polynomial of `v` (flat `n × d` i64) by scalar poly `w`
/// in the CRT-backed KOTS ring; result is in the signed canonical range
/// `[-q'/2, q'/2]`.  Analogous to `poly::scale_vec` but routes through
/// the CRT backend because q' is not natively NTT-friendly.
pub fn scale_vec_crt(w: &[i64], v: &[i64], n: usize, backend: &CrtBackend) -> Vec<i64> {
    let d = backend.d();
    let q = backend.q() as i64;
    let half = q / 2;

    // Pre-NTT w once (reused across all n rows).
    let (w_p1, w_p2) = backend.forward_pair_i64(w);

    let mut v_p1 = vec![0u64; d];
    let mut v_p2 = vec![0u64; d];
    let mut acc_p1 = vec![0u64; d];
    let mut acc_p2 = vec![0u64; d];
    let mut result = vec![0i64; n * d];

    for i in 0..n {
        backend.forward_into_i64(&v[i * d..(i + 1) * d], &mut v_p1, &mut v_p2);
        acc_p1.iter_mut().for_each(|x| *x = 0);
        acc_p2.iter_mut().for_each(|x| *x = 0);
        backend.accum_mul_slices(&mut acc_p1, &mut acc_p2, &w_p1, &w_p2, &v_p1, &v_p2);
        let prod = backend.finalize_accum_slices(&mut acc_p1, &mut acc_p2);
        for k in 0..d {
            let x = prod[k] as i64;
            result[i * d + k] = if x > half { x - q } else { x };
        }
    }
    result
}

/// Scale each entry of M (shape `r × c × d`) by scalar poly `w` in the
/// CRT-backed KOTS ring.  Result is signed.
pub fn scale_mat_crt(
    w: &[i64],
    m_mat: &[i64],
    r: usize,
    c: usize,
    backend: &CrtBackend,
) -> Vec<i64> {
    scale_vec_crt(w, m_mat, r * c, backend)
}

// ---------------------------------------------------------------------------
// Scheme public parameters
// ---------------------------------------------------------------------------

/// Lemur public parameters.
///
/// Carries a `profile: &'static Profile` reference so every scheme
/// function that takes `&LemurPp` is automatically profile-aware.
/// The `profile` must match the one used to expand `kots_a` and
/// `hvc_pp`; `lemur_setup_with_profile` enforces this.
#[derive(Clone)]
pub struct LemurPp {
    pub kots_a: KotsA,
    pub hvc_pp: HvcPp,
    pub kots_seed: [u8; 32],
    pub hvc_seed: [u8; 32],
    pub profile: &'static Profile,
}

/// Lemur secret key: just the master seed.
/// OPK is re-derived on demand; the HVC tree is rebuilt when opening.
#[derive(Clone)]
pub struct LemurSk {
    pub master_seed: [u8; 32],
}

/// Lemur stateful signer key: master seed plus the BDS08 traversal cache.
///
/// The "current slot" is the single counter `bds.phi` — there is no
/// separate `next_slot` field.  `lemur_sign_stateful` advances the
/// cache incrementally and costs amortised `(tau - K) / 2` leaf
/// evaluations per call instead of `2^tau`.  The on-disk format (see
/// `codec::sk_state_encode`) serialises the full cache with no magic
/// bytes.  For a single-shot signature at a specific slot, use the
/// raw `LemurSk` with `lemur_sign_seed` and pass the slot explicitly.
#[derive(Clone)]
pub struct LemurStateSk {
    pub master_seed: [u8; 32],
    pub bds: BdsState,
}

/// Lemur public key: HVC commitment in R_q^omega.
#[derive(Clone)]
pub struct LemurPk(pub HvcCom);

/// Lemur individual signature.
pub struct LemurSig {
    pub z: KotsSig,
    pub opening: HvcOpening,
}

/// Lemur aggregated signature.
pub struct LemurAggSig {
    pub z_agg: KotsSig,
    pub d_agg: HvcOpening,
    pub attempt: usize,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Generate Lemur public parameters from KOTS and HVC seeds.
pub fn lemur_setup_with_profile(
    kots_seed: &[u8; 32],
    hvc_seed: &[u8; 32],
    profile: &'static Profile,
) -> LemurPp {
    profile.validate();
    let kots_a = kots_setup(kots_seed, profile);
    let hvc_pp = hvc_setup_with_profile(hvc_seed, profile);
    LemurPp {
        kots_a,
        hvc_pp,
        kots_seed: *kots_seed,
        hvc_seed: *hvc_seed,
        profile,
    }
}

/// Generate Lemur public parameters with a custom tree depth.
pub fn lemur_setup_with_profile_and_tau(
    kots_seed: &[u8; 32],
    hvc_seed: &[u8; 32],
    profile: &'static Profile,
    tau: usize,
) -> LemurPp {
    profile.validate();
    let kots_a = kots_setup(kots_seed, profile);
    let hvc_pp = hvc_setup_with_profile_and_tau(hvc_seed, profile, tau);
    LemurPp {
        kots_a,
        hvc_pp,
        kots_seed: *kots_seed,
        hvc_seed: *hvc_seed,
        profile,
    }
}

// ---------------------------------------------------------------------------
// Key generation
// ---------------------------------------------------------------------------

/// Construct a stateful signer key from a master seed and BDS state.
///
/// The stateful form always carries a populated BDS08 traversal cache
/// (built by `lemur_keygen`/`bds_init`).  The "current slot" is the
/// single counter `bds.phi` — there is no separate `next_slot`.
pub fn lemur_make_stateful_sk(master_seed: &[u8; 32], bds: BdsState) -> LemurStateSk {
    LemurStateSk {
        master_seed: *master_seed,
        bds,
    }
}

/// Build a leaf-opk closure for a given master seed.
fn make_leaf_fn(
    pp: &LemurPp,
    master_seed: [u8; 32],
) -> impl Fn(usize) -> Vec<i64> + Send + Sync + '_ {
    let profile = pp.profile;
    move |t: usize| {
        let ss = slot_seed(&master_seed, t);
        kots_pk_from_seed(&pp.kots_a, &ss, profile).0
    }
}

/// Generate a Lemur key pair from a master seed.
///
/// Uses `bds_init` so the HVC commitment and the initial BDS08
/// traversal state are produced in a single fused tree walk — no extra
/// cost beyond the plain commitment. Memory: O(tau^2) labels for the
/// cache (≈ a few MB at tau=21).
pub fn lemur_keygen(pp: &LemurPp, master_seed: &[u8; 32]) -> (LemurSk, LemurStateSk, LemurPk) {
    let leaf_fn = make_leaf_fn(pp, *master_seed);
    let (c, bds_state) = bds_init(&pp.hvc_pp, leaf_fn);
    let sk = LemurSk {
        master_seed: *master_seed,
    };
    let sk_state = lemur_make_stateful_sk(master_seed, bds_state);
    let pk = LemurPk(c);
    (sk, sk_state, pk)
}

// ---------------------------------------------------------------------------
// Sign
// ---------------------------------------------------------------------------

/// Sign at slot t: re-derive per-slot KOTS key, compute HVC opening.
///
/// The HVC opening requires O(2^τ) leaf evaluations (offline cost).
pub fn lemur_sign_seed(pp: &LemurPp, sk: &LemurSk, t: usize, msg: &[u8]) -> LemurSig {
    let profile = pp.profile;
    let ms = sk.master_seed;
    let ss = slot_seed(&ms, t);
    let (osk, opk) = kots_keygen(&pp.kots_a, &ss, profile);
    let z = kots_sign(&osk, msg, profile);
    let leaf_fn = move |slot: usize| {
        let s = slot_seed(&ms, slot);
        let pk = kots_pk_from_seed(&pp.kots_a, &s, profile);
        pk.0
    };
    let opening = hvc_open_with_known_leaf(&pp.hvc_pp, t, Some(&opk.0), leaf_fn);
    LemurSig { z, opening }
}

/// Sign via the BDS08 auth-path traversal, advancing `sk_state` in place.
///
/// The "current slot" is derived from `sk_state.bds.phi` — there is no
/// separate `next_slot` counter.  Optional `t` is a defensive check:
/// if supplied, it must match `bds.phi`.
///
/// Amortised cost per call: `(tau - K) / 2` leaf evaluations plus one
/// KOTS sign, versus `2^tau` for `lemur_sign_seed`.  This is the
/// efficient path: it mutates `sk_state.bds` directly, so there is no
/// deep copy of the (tens-to-hundreds-of-KB) traversal cache.  On
/// error, `sk_state` is left unchanged — the validity checks run
/// before any mutation.
///
/// Use `lemur_sign_stateful` instead if you need the pre-sign state to
/// remain valid for rollback or reproduction.
///
/// Returns `Err(LemurError::InvalidEncoding)` on tau/slot mismatch or
/// out-of-range slot.
pub fn lemur_sign_stateful_mut(
    pp: &LemurPp,
    sk_state: &mut LemurStateSk,
    msg: &[u8],
    t: Option<usize>,
) -> Result<(LemurSig, usize), LemurError> {
    if sk_state.bds.h != pp.hvc_pp.tau {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful sk tau={} does not match pp tau={}",
            sk_state.bds.h, pp.hvc_pp.tau
        )));
    }
    let phi = sk_state.bds.phi;
    let slot = t.unwrap_or(phi);
    if slot != phi {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful signer is at slot {phi}, got {slot}"
        )));
    }
    let n_slots = 1usize << pp.hvc_pp.tau;
    if slot >= n_slots {
        return Err(LemurError::InvalidEncoding(format!(
            "slot {slot} out of range [0, {}]",
            n_slots - 1
        )));
    }

    let master_seed = sk_state.master_seed;
    let profile = pp.profile;
    let leaf_fn = make_leaf_fn(pp, master_seed);

    // KOTS sign at slot t.
    let ss = slot_seed(&master_seed, slot);
    let (osk, _opk) = kots_keygen(&pp.kots_a, &ss, profile);
    let z = kots_sign(&osk, msg, profile);

    // HVC opening assembled from the BDS auth path.
    let opening = bds_opening(&sk_state.bds, &pp.hvc_pp, slot, &leaf_fn)?;

    // Advance traversal state (unless we just signed the last leaf).
    if slot + 1 < n_slots {
        bds_advance(&mut sk_state.bds, &pp.hvc_pp, &leaf_fn);
    }

    Ok((LemurSig { z, opening }, slot))
}

/// Snapshot-preserving variant of `lemur_sign_stateful_mut`.
///
/// Clones the BDS cache before advancing so the caller's `sk_state`
/// stays a valid pre-sign snapshot.  Callers that always persist the
/// returned `next_state` should prefer `lemur_sign_stateful_mut` to
/// avoid the deep copy.
pub fn lemur_sign_stateful(
    pp: &LemurPp,
    sk_state: &LemurStateSk,
    msg: &[u8],
    t: Option<usize>,
) -> Result<(LemurSig, LemurStateSk, usize), LemurError> {
    let mut next_state = sk_state.clone();
    let (sig, slot) = lemur_sign_stateful_mut(pp, &mut next_state, msg, t)?;
    Ok((sig, next_state, slot))
}

pub fn lemur_sign(pp: &LemurPp, sk: &LemurSk, t: usize, msg: &[u8]) -> LemurSig {
    lemur_sign_seed(pp, sk, t, msg)
}

// ---------------------------------------------------------------------------
// Individual verify
// ---------------------------------------------------------------------------

/// Individually verify a signature (iVrfy).
///
/// Never panics: catches any internal error (including malformed array
/// shapes) and returns `Err(VerifyFailed)`.
pub fn lemur_ivrfy(
    pp: &LemurPp,
    pk: &LemurPk,
    t: usize,
    msg: &[u8],
    sig: &LemurSig,
) -> Result<(), LemurError> {
    lemur_ivrfy_inner(pp, pk, t, msg, sig).map_err(|_| LemurError::VerifyFailed)
}

fn lemur_ivrfy_inner(
    pp: &LemurPp,
    pk: &LemurPk,
    t: usize,
    msg: &[u8],
    sig: &LemurSig,
) -> Result<(), LemurError> {
    let n_slots = 1usize << pp.hvc_pp.tau;
    if t >= n_slots {
        return Err(LemurError::VerifyFailed);
    }
    let opk = hvc_ivrfy(&pp.hvc_pp, &pk.0, t, &sig.opening)?;
    kots_ivrfy(&pp.kots_a, &opk, msg, &sig.z, pp.profile)
}

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

/// Sample N ternary randomizers from a single XOF keyed on (t, msg, pks, attempt).
fn hash_to_randomizers(
    t: usize,
    msg: &[u8],
    pks_bytes: &[u8],
    attempt: usize,
    n: usize,
    profile: &Profile,
) -> Vec<Vec<i64>> {
    let mut xof = agg_randomizer_xof(t, msg, pks_bytes, attempt);
    (0..n)
        .map(|_| xof_ternary_poly(&mut xof as &mut dyn XofReader, profile.alpha_w, profile.d))
        .collect()
}

/// Concatenate all pk bytes in order for the randomizer oracle.
fn concat_pk_bytes(pks: &[LemurPk]) -> Vec<u8> {
    pks.iter()
        .flat_map(|pk| pk.0 .0.iter().flat_map(|&x| x.to_le_bytes()))
        .collect()
}

/// Compute weighted sum of commitments: c_agg = sum_i w^i * c_i in R_q^omega.
fn weighted_sum_commitments(ws: &[Vec<i64>], pks: &[LemurPk], profile: &Profile) -> HvcCom {
    let omega = profile.omega;
    let d = profile.d;
    let q = profile.q_hvc() as i64;
    let mut result = vec![0i64; omega * d];
    match &profile.hvc_ring {
        HvcRing::U32(rp) => {
            for (w, pk) in ws.iter().zip(pks.iter()) {
                for r in 0..omega {
                    let w_hat_prod = crate::poly::poly_mul(w, &pk.0 .0[r * d..(r + 1) * d], rp);
                    for i in 0..d {
                        result[r * d + i] = (result[r * d + i] + w_hat_prod[i]).rem_euclid(q);
                    }
                }
            }
        }
        HvcRing::U64(rp) => {
            for (w, pk) in ws.iter().zip(pks.iter()) {
                for r in 0..omega {
                    let w_hat_prod = crate::poly::poly_mul_u64(w, &pk.0 .0[r * d..(r + 1) * d], rp);
                    for i in 0..d {
                        result[r * d + i] = (result[r * d + i] + w_hat_prod[i]).rem_euclid(q);
                    }
                }
            }
        }
    }
    HvcCom(result)
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

/// Aggregate N individual signatures into one aggregated signature.
///
/// Rejects if any individual signature is invalid.
/// Retries up to gamma times until avrfy passes.
pub fn lemur_aggregate(
    pp: &LemurPp,
    pks: &[LemurPk],
    t: usize,
    msg: &[u8],
    sigs: &[LemurSig],
) -> Result<LemurAggSig, LemurError> {
    if sigs.len() != pks.len() {
        return Err(LemurError::VerifyFailed);
    }
    pks.par_iter()
        .zip(sigs.par_iter())
        .try_for_each(|(pk, sig)| lemur_ivrfy(pp, pk, t, msg, sig))?;

    let profile = pp.profile;
    let ell = profile.ell;
    let m = profile.m;
    let gamma = profile.gamma;
    let n = pks.len();
    let pks_bytes = concat_pk_bytes(pks);
    let kots_rp = profile.kots_ring.clone();
    let d = profile.d;

    for attempt in 1..=gamma {
        let ws = hash_to_randomizers(t, msg, &pks_bytes, attempt, n, profile);

        // Aggregate Z = sum_i w_i * sig_i.z.  Route through the backend
        // that the profile's KOTS ring uses:
        //   * CRT       (every shipped profile; q' not natively NTT-friendly)
        //   * u64 NTT   (reserved for future natively NTT-friendly q' >= 2^32)
        //   * u32 NTT   (reserved for future natively NTT-friendly q' < 2^32).
        let z_agg = if let Some(cfg) = profile.kots_crt() {
            let backend = cfg.backend();
            sigs.par_iter()
                .zip(ws.par_iter())
                .map(|(sig, w)| scale_mat_crt(w, &sig.z.0, ell, m, &backend))
                .reduce(
                    || vec![0i64; ell * m * d],
                    |mut acc, scaled| {
                        for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                            *a += *b;
                        }
                        acc
                    },
                )
        } else if let Some(kots_rp64) = profile.kots_ring64.as_ref() {
            sigs.par_iter()
                .zip(ws.par_iter())
                .map(|(sig, w)| crate::poly::scale_mat_u64(w, &sig.z.0, ell, m, kots_rp64))
                .reduce(
                    || vec![0i64; ell * m * d],
                    |mut acc, scaled| {
                        for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                            *a += *b;
                        }
                        acc
                    },
                )
        } else {
            let kots_rp = kots_rp.as_ref().expect("u32 KOTS ring for u32 profile");
            sigs.par_iter()
                .zip(ws.par_iter())
                .map(|(sig, w)| crate::poly::scale_mat(w, &sig.z.0, ell, m, kots_rp))
                .reduce(
                    || vec![0i64; ell * m * d],
                    |mut acc, scaled| {
                        for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                            *a += *b;
                        }
                        acc
                    },
                )
        };

        // Aggregate openings
        let d_agg = sigs
            .par_iter()
            .zip(ws.par_iter())
            .map(|(sig, w)| scale_opening_any(w, &sig.opening, profile))
            .reduce_with(|a, b| add_openings(&a, &b))
            .expect("non-empty signature list");

        let sigma_agg = LemurAggSig {
            z_agg: KotsSig(z_agg),
            d_agg,
            attempt,
        };

        if lemur_avrfy(pp, pks, t, msg, &sigma_agg).is_ok() {
            return Ok(sigma_agg);
        }
    }

    Err(LemurError::AggregationFailed(profile.gamma))
}

// ---------------------------------------------------------------------------
// Aggregated verify
// ---------------------------------------------------------------------------

/// Verify an aggregated signature (aVrfy).
///
/// Never panics: catches any internal error (including malformed array
/// shapes) and returns `Err(VerifyFailed)`.
pub fn lemur_avrfy(
    pp: &LemurPp,
    pks: &[LemurPk],
    t: usize,
    msg: &[u8],
    sigma_agg: &LemurAggSig,
) -> Result<(), LemurError> {
    lemur_avrfy_inner(pp, pks, t, msg, sigma_agg).map_err(|_| LemurError::VerifyFailed)
}

fn lemur_avrfy_inner(
    pp: &LemurPp,
    pks: &[LemurPk],
    t: usize,
    msg: &[u8],
    sigma_agg: &LemurAggSig,
) -> Result<(), LemurError> {
    let profile = pp.profile;
    let pks_bytes = concat_pk_bytes(pks);
    let ws = hash_to_randomizers(t, msg, &pks_bytes, sigma_agg.attempt, pks.len(), profile);
    let c_agg = weighted_sum_commitments(&ws, pks, profile);
    let opk_agg = hvc_svrfy(&pp.hvc_pp, &c_agg, t, &sigma_agg.d_agg)?;
    kots_svrfy(&pp.kots_a, &opk_agg, msg, &sigma_agg.z_agg, profile)
}
