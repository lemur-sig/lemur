//! Top-level Lemur multi-signature scheme.

use sha3::digest::XofReader;

use crate::aux_ntt::CrtBackend;
use crate::error::LemurError;
use crate::hvc::{
    aggregate_openings_any, bds_advance, bds_init, bds_opening, hvc_ivrfy, hvc_open_with_known_leaf,
    hvc_setup_with_profile, hvc_setup_with_profile_and_tau, hvc_svrfy, BdsState, HvcCom, HvcOpening,
    HvcPp,
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

const NTT_AGG_CHUNKS_PER_THREAD: usize = 4;
const NTT_AGG_MAX_CHUNK: usize = 64;

/// Aggregate `Σ_i w_i * v_i` for a flat `n_polys × d` CRT-backed vector.
///
/// This is the KOTS-side analogue of HVC opening aggregation: accumulate in
/// the two auxiliary-prime NTT domains, then pay one inverse NTT/CRT combine
/// per output polynomial instead of one per signer per output polynomial.
fn aggregate_vec_crt_ntt(
    ws: &[Vec<i64>],
    vectors: &[&[i64]],
    n_polys: usize,
    backend: &CrtBackend,
) -> Vec<i64> {
    assert_eq!(ws.len(), vectors.len(), "ws/vectors length mismatch");
    assert!(!vectors.is_empty(), "aggregate_vec_crt_ntt: empty input");

    let d = backend.d();
    let q = backend.q() as i64;
    let half = q / 2;
    let acc_len = n_polys * d;
    debug_assert!(ws.iter().all(|w| w.len() == d));
    debug_assert!(vectors.iter().all(|v| v.len() == acc_len));

    let target_chunks = rayon::current_num_threads()
        .saturating_mul(NTT_AGG_CHUNKS_PER_THREAD)
        .max(1);
    let chunk_size = ws.len().div_ceil(target_chunks).clamp(1, NTT_AGG_MAX_CHUNK);

    let (mut acc_p1, mut acc_p2) = ws
        .par_chunks(chunk_size)
        .zip(vectors.par_chunks(chunk_size))
        .map(|(w_chunk, v_chunk)| {
            let mut local_p1 = vec![0u64; acc_len];
            let mut local_p2 = vec![0u64; acc_len];
            let mut w_p1 = vec![0u64; d];
            let mut w_p2 = vec![0u64; d];
            let mut v_p1 = vec![0u64; d];
            let mut v_p2 = vec![0u64; d];

            for (w, vector) in w_chunk.iter().zip(v_chunk.iter()) {
                backend.forward_into_i64(w, &mut w_p1, &mut w_p2);
                for poly in 0..n_polys {
                    let off = poly * d;
                    backend.forward_into_i64(&vector[off..off + d], &mut v_p1, &mut v_p2);
                    backend.accum_mul_slices(
                        &mut local_p1[off..off + d],
                        &mut local_p2[off..off + d],
                        &w_p1,
                        &w_p2,
                        &v_p1,
                        &v_p2,
                    );
                }
            }

            (local_p1, local_p2)
        })
        .reduce(
            || (vec![0u64; acc_len], vec![0u64; acc_len]),
            |mut a, b| {
                let (b_p1, b_p2) = b;
                for (x, y) in a.0.iter_mut().zip(b_p1) {
                    *x = (*x)
                        .checked_add(y)
                        .expect("CRT z aggregate accumulator overflow");
                }
                for (x, y) in a.1.iter_mut().zip(b_p2) {
                    *x = (*x)
                        .checked_add(y)
                        .expect("CRT z aggregate accumulator overflow");
                }
                a
            },
        );

    let mut result = vec![0i64; acc_len];
    for poly in 0..n_polys {
        let off = poly * d;
        let prod =
            backend.finalize_accum_slices(&mut acc_p1[off..off + d], &mut acc_p2[off..off + d]);
        for k in 0..d {
            let x = prod[k] as i64;
            result[off + k] = if x > half { x - q } else { x };
        }
    }
    result
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

/// Modular addition for inputs already known to be in `[0, q)`.
///
/// Caller must ensure `q < 2^63` so that the plain `a + b` cannot
/// overflow u64.  Shipped HVC modulus is `q ≈ 2^53`, leaving wide
/// margin; the `debug_assert!` trips a future profile that pushes
/// `q` toward `2^63`.
#[inline(always)]
fn add_mod_q_u64(a: u64, b: u64, q: u64) -> u64 {
    debug_assert!(q < (1u64 << 63), "add_mod_q_u64: q must fit in i64");
    let s = a + b;
    if s >= q {
        s - q
    } else {
        s
    }
}

fn weighted_sum_commitments_u64_ntt(
    ws: &[Vec<i64>],
    pks: &[LemurPk],
    omega: usize,
    rp: &crate::poly::RingParams64,
) -> HvcCom {
    let d = rp.d;
    let q = rp.q;
    let acc_len = omega * d;
    let target_chunks = rayon::current_num_threads()
        .saturating_mul(NTT_AGG_CHUNKS_PER_THREAD)
        .max(1);
    let chunk_size = ws.len().div_ceil(target_chunks).clamp(1, NTT_AGG_MAX_CHUNK);

    let mut acc = ws
        .par_chunks(chunk_size)
        .zip(pks.par_chunks(chunk_size))
        .map(|(w_chunk, pk_chunk)| {
            let mut local = vec![0u64; acc_len];
            let mut w_hat = vec![0u64; d];
            let mut pk_hat = vec![0u64; d];

            for (w, pk) in w_chunk.iter().zip(pk_chunk.iter()) {
                crate::poly::poly_to_ntt_buf_u64(w, rp, &mut w_hat);
                for r in 0..omega {
                    let off = r * d;
                    crate::poly::poly_to_ntt_buf_u64(&pk.0 .0[off..off + d], rp, &mut pk_hat);
                    crate::poly::mac_ntt_u64(
                        &mut local[off..off + d],
                        &w_hat,
                        &pk_hat,
                        rp.q,
                        rp.q_inv,
                    );
                }
            }

            local
        })
        .reduce(
            || vec![0u64; acc_len],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x = add_mod_q_u64(*x, y, q);
                }
                a
            },
        );

    let mut result = vec![0i64; acc_len];
    for r in 0..omega {
        let off = r * d;
        crate::poly::ntt_to_poly_buf_u64(&mut acc[off..off + d], rp, &mut result[off..off + d]);
    }
    HvcCom(result)
}

/// KOTS aggregate Z = Σᵢ wᵢ · sigᵢ.z routing.
///
/// Applies the no-wrap gate (`N·αw·βz ≤ q'/2`) and chooses between the
/// in-CRT-domain fast path and the per-signer slow fallback.  This is the
/// single source of truth used by both production `lemur_aggregate` and
/// the `bench_internals::aggregate_kots_z` wrapper, so the bench cannot
/// drift from production over time.
fn aggregate_kots_z_inner(
    ws: &[Vec<i64>],
    sigs: &[LemurSig],
    profile: &Profile,
) -> Vec<i64> {
    let m = profile.m;
    let d = profile.d;
    let n = sigs.len();
    let kots_rp = profile.kots_ring.clone();
    let z_sum_no_wrap = (n as i128) * (profile.alpha_w as i128) * (profile.beta_z as i128)
        <= (profile.q_kots() as i128) / 2;
    let kots_crt_acc_backend = profile.kots_crt().and_then(|cfg| {
        if z_sum_no_wrap {
            CrtBackend::new_for_accum(cfg.q, cfg.d, n)
        } else {
            None
        }
    });
    let kots_crt_single_backend =
        if profile.kots_crt().is_some() && kots_crt_acc_backend.is_none() {
            profile.kots_crt().map(|cfg| cfg.backend())
        } else {
            None
        };

    if profile.kots_crt().is_some() {
        if let Some(backend) = kots_crt_acc_backend.as_ref() {
            let z_refs: Vec<&[i64]> = sigs.iter().map(|sig| sig.z.0.as_slice()).collect();
            aggregate_vec_crt_ntt(ws, &z_refs, m, backend)
        } else {
            let backend = kots_crt_single_backend
                .as_ref()
                .expect("single-product CRT backend must exist");
            sigs.par_iter()
                .zip(ws.par_iter())
                .map(|(sig, w)| scale_mat_crt(w, &sig.z.0, 1, m, backend))
                .reduce(
                    || vec![0i64; m * d],
                    |mut acc, scaled| {
                        for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                            *a += *b;
                        }
                        acc
                    },
                )
        }
    } else if let Some(kots_rp64) = profile.kots_ring64.as_ref() {
        sigs.par_iter()
            .zip(ws.par_iter())
            .map(|(sig, w)| crate::poly::scale_mat_u64(w, &sig.z.0, 1, m, kots_rp64))
            .reduce(
                || vec![0i64; m * d],
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
            .map(|(sig, w)| crate::poly::scale_mat(w, &sig.z.0, 1, m, kots_rp))
            .reduce(
                || vec![0i64; m * d],
                |mut acc, scaled| {
                    for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                        *a += *b;
                    }
                    acc
                },
            )
    }
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
            return weighted_sum_commitments_u64_ntt(ws, pks, omega, rp);
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
    if sigs.is_empty() {
        return Err(LemurError::VerifyFailed);
    }
    pks.par_iter()
        .zip(sigs.par_iter())
        .try_for_each(|(pk, sig)| lemur_ivrfy(pp, pk, t, msg, sig))?;

    let profile = pp.profile;
    let gamma = profile.gamma;
    let n = pks.len();
    let pks_bytes = concat_pk_bytes(pks);

    // The CRT NTT aggregate canonicalizes once at the end.  The routing
    // (in-CRT-domain fast path vs. per-signer slow fallback) is shared with
    // `bench_internals::aggregate_kots_z` via `aggregate_kots_z_inner`.
    //
    // The fast path activates iff `N·alpha_w·beta_z <= q'/2`, since each
    // coefficient of `Σᵢ wᵢ·zᵢ` is bounded by `N·alpha_w·beta_z` (ternary
    // randomizers have `||wᵢ||₁ ≤ alpha_w`; iVrfy on each partial signature
    // guarantees `||zᵢ||∞ ≤ beta_z`), and the centred representative in
    // `(-q'/2, q'/2]` equals the unreduced signed sum only when the
    // magnitude stays under `q'/2`.  For the shipped profile `D256_K4`
    // (`alpha_w=23, beta_z=7023, q' ≈ 3.47·10⁹`) the crossover is at
    // `N ≈ 10739`, so the fast path is used at `N ∈ {2^10, 8192}` and the
    // slow per-signer fallback kicks in at `N ∈ {2^15, 2^17, 2^20}`.  Both
    // paths produce identical centred outputs; the crossover is a
    // performance-only routing decision.

    for attempt in 1..=gamma {
        let ws = hash_to_randomizers(t, msg, &pks_bytes, attempt, n, profile);

        // Aggregate Z = sum_i w_i * sig_i.z (see helper for the routing).
        let z_agg = aggregate_kots_z_inner(&ws, sigs, profile);

        // Aggregate openings: NTT-domain accumulation across signers.  See
        // `aggregate_openings_any` in `hvc.rs` for the algebra and overflow
        // analysis.  Equivalent to (and bit-identical with) the reference
        // pattern `Σᵢ scale_opening_any(wᵢ, sigᵢ.opening) + add_openings(...)`,
        // but pays only `(2τ+1)·ωκ` inverse NTTs instead of `N·(2τ+1)·ωκ`,
        // shaving ~15 % off the full `lemur_aggregate` call at the deployed
        // parameter cell (`τ=20`, `N=1024`).
        let opening_refs: Vec<&HvcOpening> = sigs.iter().map(|s| &s.opening).collect();
        let d_agg = aggregate_openings_any(&ws, &opening_refs, profile);

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

// ---------------------------------------------------------------------------
// Bench-only helpers: thin re-exports of internal aggregation sub-steps.
// `#[doc(hidden)]` keeps them off the public surface in rustdoc.  Used by
// `src/bin/bench_aggregate.rs` to time each sub-step of `lemur_aggregate`
// and `lemur_avrfy` in isolation.
// ---------------------------------------------------------------------------
#[doc(hidden)]
pub mod bench_internals {
    use super::*;

    /// Concatenate all pk bytes for the randomizer oracle.
    pub fn concat_pk_bytes_pub(pks: &[LemurPk]) -> Vec<u8> {
        concat_pk_bytes(pks)
    }

    /// Derive the `N` ternary aggregation weights `wᵢ` from the random
    /// oracle inputs.
    pub fn hash_to_randomizers_pub(
        t: usize,
        msg: &[u8],
        pks_bytes: &[u8],
        attempt: usize,
        n: usize,
        profile: &'static Profile,
    ) -> Vec<Vec<i64>> {
        hash_to_randomizers(t, msg, pks_bytes, attempt, n, profile)
    }

    /// HVC commitment aggregation `c_agg = Σᵢ wᵢ · cᵢ` (the avrfy step).
    pub fn weighted_sum_commitments_pub(
        ws: &[Vec<i64>],
        pks: &[LemurPk],
        profile: &'static Profile,
    ) -> HvcCom {
        weighted_sum_commitments(ws, pks, profile)
    }

    /// KOTS Z aggregation `z_agg = Σᵢ wᵢ · zᵢ`.  Thin wrapper around the
    /// shared production routing helper so bench and production cannot
    /// drift.
    pub fn aggregate_kots_z(
        ws: &[Vec<i64>],
        sigs: &[LemurSig],
        profile: &'static Profile,
    ) -> Vec<i64> {
        aggregate_kots_z_inner(ws, sigs, profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::D256_K4;

    #[test]
    fn aggregate_vec_crt_ntt_matches_reference_scaling() {
        let profile = &D256_K4;
        let cfg = profile.kots_crt().expect("D256_K4 uses CRT KOTS");
        let backend_acc = cfg.backend_for_accum(5);
        let backend_single = cfg.backend();
        let d = profile.d;
        let m = profile.m;

        let ws: Vec<Vec<i64>> = (0..5)
            .map(|signer| {
                let mut w = vec![0i64; d];
                for j in 0..profile.alpha_w {
                    let idx = (17 * signer + 31 * j) % d;
                    w[idx] = if (signer + j) & 1 == 0 { 1 } else { -1 };
                }
                w
            })
            .collect();
        let vectors: Vec<Vec<i64>> = (0..5)
            .map(|signer| {
                (0..m * d)
                    .map(|i| ((signer as i64 * 19 + i as i64 * 7) % 101) - 50)
                    .collect()
            })
            .collect();
        let vector_refs: Vec<&[i64]> = vectors.iter().map(|v| v.as_slice()).collect();

        let fast = aggregate_vec_crt_ntt(&ws, &vector_refs, m, &backend_acc);
        let reference = ws
            .iter()
            .zip(vectors.iter())
            .map(|(w, v)| scale_mat_crt(w, v, 1, m, &backend_single))
            .reduce(|mut acc, scaled| {
                for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                    *a += *b;
                }
                acc
            })
            .expect("nonempty vector set");

        assert_eq!(fast, reference);
    }

    /// At `N` past the algebraic no-wrap crossover for `D256_K4`
    /// (`N·alpha_w·beta_z > q'/2`, i.e. `N > q'/(2·alpha_w·beta_z) ≈ 10739`),
    /// `lemur_aggregate` routes to the per-signer slow path.  Synthetic
    /// `z`-vectors with `||z||∞ ≤ beta_z` and ternary `w`-vectors of
    /// weight `alpha_w` are constructed at `N = n_crossover + 1` (the
    /// smallest integer past the crossover, computed at runtime from the
    /// profile constants); the per-signer `scale_mat_crt`+sum result is
    /// compared against the in-NTT-domain `aggregate_vec_crt_ntt` to
    /// confirm the two paths still agree past the gate.  Both paths
    /// produce the centred signed sum mod `q'`, so equivalence must hold
    /// for any `N` inside the CRT backend's headroom.
    #[test]
    fn aggregate_vec_paths_agree_past_no_wrap_crossover() {
        let profile = &D256_K4;
        let cfg = profile.kots_crt().expect("D256_K4 uses CRT KOTS");
        let d = profile.d;
        let m = profile.m;
        let alpha_w = profile.alpha_w;
        let beta_z = profile.beta_z;
        let q_prime = profile.q_kots() as i128;
        let n_crossover = (q_prime / (2 * alpha_w as i128 * beta_z as i128)) as usize;
        // `n` must be past the crossover for the routing claim to be
        // load-bearing, and inside the fast backend's `terms` headroom.
        let n: usize = n_crossover + 1;
        // Sanity: the algebraic gate must in fact be false at this `n`.
        let z_sum_no_wrap =
            (n as i128) * (alpha_w as i128) * (beta_z as i128) <= q_prime / 2;
        assert!(!z_sum_no_wrap, "test setup failed: still inside fast-path range");
        let backend_acc = cfg.backend_for_accum(n);
        let backend_single = cfg.backend();

        let ws: Vec<Vec<i64>> = (0..n)
            .map(|signer| {
                let mut w = vec![0i64; d];
                for j in 0..alpha_w {
                    let idx = (17 * signer + 31 * j) % d;
                    w[idx] = if (signer + j) & 1 == 0 { 1 } else { -1 };
                }
                w
            })
            .collect();
        // Synthetic `z`-vectors: pick coefficients deterministically in
        // `[-beta_z, beta_z]` so they satisfy the iVrfy bound the slow
        // path's no-wrap analysis relies on.
        let vectors: Vec<Vec<i64>> = (0..n)
            .map(|signer| {
                (0..m * d)
                    .map(|i| {
                        let raw = (signer as i64 * 1009 + i as i64 * 1597) % (2 * beta_z + 1);
                        raw - beta_z
                    })
                    .collect()
            })
            .collect();
        let vector_refs: Vec<&[i64]> = vectors.iter().map(|v| v.as_slice()).collect();

        let fast = aggregate_vec_crt_ntt(&ws, &vector_refs, m, &backend_acc);
        let reference = ws
            .iter()
            .zip(vectors.iter())
            .map(|(w, v)| scale_mat_crt(w, v, 1, m, &backend_single))
            .reduce(|mut acc, scaled| {
                for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                    *a += *b;
                }
                acc
            })
            .expect("nonempty vector set");

        assert_eq!(fast, reference);
    }

    #[test]
    fn weighted_sum_commitments_u64_ntt_matches_reference() {
        let profile = &D256_K4;
        let rp = profile.hvc_ring.as_u64().expect("D256_K4 uses u64 HVC");
        let d = profile.d;
        let omega = profile.omega;
        let q = profile.q_hvc() as i64;

        let ws: Vec<Vec<i64>> = (0..5)
            .map(|signer| {
                let mut w = vec![0i64; d];
                for j in 0..profile.alpha_w {
                    let idx = (11 * signer + 37 * j) % d;
                    w[idx] = if (signer + j) & 1 == 0 { 1 } else { -1 };
                }
                w
            })
            .collect();
        let pks: Vec<LemurPk> = (0..5)
            .map(|signer| {
                let c = (0..omega * d)
                    .map(|i| ((signer as i64 * 23 + i as i64 * 41) % 10_000).rem_euclid(q))
                    .collect();
                LemurPk(HvcCom(c))
            })
            .collect();

        let fast = weighted_sum_commitments(&ws, &pks, profile);
        let mut reference = vec![0i64; omega * d];
        for (w, pk) in ws.iter().zip(pks.iter()) {
            for r in 0..omega {
                let off = r * d;
                let prod = crate::poly::poly_mul_u64(w, &pk.0 .0[off..off + d], rp);
                for i in 0..d {
                    reference[off + i] = (reference[off + i] + prod[i]).rem_euclid(q);
                }
            }
        }

        assert_eq!(fast.0, reference);
    }
}
