//! HVC: Homomorphic Vector Commitment.
//!
//! Stateful signers maintain an authentication path via the BDS08
//! tree-traversal algorithm (Buchmann-Dahmen-Schneider, PQCrypto 2008);
//! see `TreehashInst`, `BdsState`, `bds_init`, `bds_advance`, and
//! `bds_opening` below.
//!
//! # API layering
//!
//! * `HvcPp` carries a `profile: &'static Profile` reference.  Every
//!   function that takes `&HvcPp` is automatically profile-aware: the
//!   implementation reads `omega`, `kappa`, `kappa_prime`, `rho`, `nu`,
//!   `eta`, `b_val = 2*eta + 1`, `q_hvc`, `q_kots`, `d`, and `tau`
//!   from `pp.profile` (plus the already-runtime `pp.tau` /
//!   `pp.beta_agg` / `pp.beta_encode`).
//! * Utility functions that do not take `&HvcPp` but need profile
//!   constants (`proj_label_with_profile`, `leaf_label_with_profile`,
//!   `internal_label_with_profile`, `dec_vec_with_profile`,
//!   `babai_encode_label_with_profile`, `babai_decode_label_with_profile`)
//!   take a `&Profile` parameter.
//!
//! # Ring dimension
//!
//! Every buffer and slice index reads the ring dimension from
//! `profile.d` (or `pp.profile.d` / `rp.d`), so the HVC code follows the
//! active parameter set instead of a compile-time ring degree.

use std::collections::VecDeque;

use sha3::digest::XofReader;

use crate::error::LemurError;
use crate::kots::{inf_norm, KotsPk};
use crate::ntt::{mont_mul_u64, ntt_inverse_u64};
use crate::poly::{
    mac_ntt, mac_ntt_u64, mat_to_ntt, mat_to_ntt_u64, mat_vec_prentt, mat_vec_prentt_pair,
    mat_vec_prentt_pair_u64, mat_vec_prentt_u64, ntt_to_poly_buf, ntt_to_poly_buf_u64,
    poly_to_ntt, poly_to_ntt_buf, poly_to_ntt_buf_u64, scale_vec_with_ntt_w,
    scale_vec_with_ntt_w_u64, RingParams, RingParams64,
};
use rayon::prelude::*;
use crate::profile::{HvcRing, Profile};
use crate::sample::{hvc_setup_xof, xof_uniform_poly};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// HVC public parameters: (B, A0, A1), each a flat Vec<i64>.
/// B: (omega x rho * nu * kappa_prime x d)
/// A0, A1: (omega x omega * kappa x d)
///
/// Pre-NTT'd copies are stored for fast repeated mat_vec calls.  The
/// `profile` reference carries every other scalar (`omega`, `kappa`,
/// `kappa_prime`, `rho`, `nu`, `eta`, `q_hvc`, `q_kots`, the matching
/// `RingParams`, and the profile's default `tau`), so downstream
/// functions that take `&HvcPp` are automatically profile-aware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HvcNttVec {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Clone)]
pub struct HvcPp {
    pub b_mat: Vec<i64>,
    pub a0: Vec<i64>,
    pub a1: Vec<i64>,
    pub b_mat_ntt: HvcNttVec,
    pub a0_ntt: HvcNttVec,
    pub a1_ntt: HvcNttVec,
    /// Active Lemur profile (provides `omega`, `kappa`, `q_hvc`, the
    /// HVC `RingParams`, and every other profile-scoped scalar).
    pub profile: &'static Profile,
    /// Tree depth (may differ from `profile.tau` when `hvc_setup_with_tau`
    /// is used for fast test-vector generation at small τ).
    pub tau: usize,
    /// Aggregated opening norm bound (depends on tau).
    pub beta_agg: i64,
    /// Babai encoding bound (derived from beta_agg).
    pub beta_encode: i64,
}

/// HVC commitment: shape (omega x d), flat.
#[derive(Clone)]
pub struct HvcCom(pub Vec<i64>);

/// HVC opening: (path_labels, sibling_labels, u).
/// Each label: flat (omega*kappa x d).
/// u: flat (rho*nu*kappa_prime x d).
#[derive(Clone)]
pub struct HvcOpening {
    pub path_labels: Vec<Vec<i64>>,
    pub sibling_labels: Vec<Vec<i64>>,
    pub u: Vec<i64>,
}

// ---------------------------------------------------------------------------
// BDS08 tree-traversal data structures
// ---------------------------------------------------------------------------

/// One BDS08 treehash instance for computing a single height-h node.
///
/// Each instance owns its own tail-node stack.  Sharing a single stack
/// across instances is a memory optimisation we do not use here; total
/// state is still O(tau^2) labels.
#[derive(Clone)]
pub struct TreehashInst {
    /// Target height (one instance per height in `[0, tau-K)`).
    pub h: usize,
    /// Per-instance tail-node stack: `(height, label)`.
    pub stack: Vec<(usize, Vec<i64>)>,
    /// Finished height-`h` output, or `None` if not yet produced.
    pub node: Option<Vec<i64>>,
    /// Index of the next leaf to process.
    pub leaf_index: usize,
    /// Number of leaves still to consume in this traversal.
    pub leaves_remaining: usize,
    /// `true` once this instance has produced its output.
    pub finished: bool,
}

impl TreehashInst {
    pub fn new(h: usize) -> Self {
        Self {
            h,
            stack: Vec::new(),
            node: None,
            leaf_index: 0,
            leaves_remaining: 0,
            finished: false,
        }
    }

    /// Begin a new traversal rooted at `leaf_index`, covering `2^h` leaves.
    pub fn initialize(&mut self, leaf_index: usize) {
        self.stack.clear();
        self.node = None;
        self.leaf_index = leaf_index;
        self.leaves_remaining = 1usize << self.h;
        self.finished = false;
    }

    /// Inject a precomputed height-`h` node (used by `bds_init`).
    pub fn set_ready(&mut self, node: Vec<i64>) {
        self.stack.clear();
        self.node = Some(node);
        self.leaf_index = 0;
        self.leaves_remaining = 0;
        self.finished = true;
    }

    /// Clear the finished output; subsequent `initialize` will start afresh.
    pub fn take_ready(&mut self) -> Option<Vec<i64>> {
        let out = self.node.take();
        self.finished = false;
        out
    }

    /// Lowest stored tail-node height; `h` if stack empty, `usize::MAX` if done.
    pub fn height_metric(&self) -> usize {
        if self.finished || self.leaves_remaining == 0 {
            return usize::MAX;
        }
        if self.stack.is_empty() {
            return self.h;
        }
        self.stack.iter().map(|(h, _)| *h).min().unwrap()
    }

    /// Execute one treehash step: consume one leaf, hash up while possible.
    pub fn update<F>(&mut self, pp: &HvcPp, leaf_fn: &F)
    where
        F: Fn(usize) -> Vec<i64>,
    {
        if self.finished || self.leaves_remaining == 0 {
            return;
        }
        let leaf = leaf_label_with_profile_any(&leaf_fn(self.leaf_index), &pp.b_mat_ntt, pp.profile);
        let mut cur_h: usize = 0;
        let mut cur_lab = leaf;
        while matches!(self.stack.last(), Some((top_h, _)) if *top_h == cur_h) {
            let (_, top_lab) = self.stack.pop().unwrap();
            cur_lab = internal_label_with_profile_any(
                &top_lab,
                &cur_lab,
                &pp.a0_ntt,
                &pp.a1_ntt,
                pp.profile,
            );
            cur_h += 1;
        }
        self.stack.push((cur_h, cur_lab));
        self.leaf_index = self.leaf_index.wrapping_add(1);
        self.leaves_remaining -= 1;
        if self.leaves_remaining == 0 {
            debug_assert_eq!(self.stack.len(), 1);
            debug_assert_eq!(self.stack[0].0, self.h);
            let (_, out) = self.stack.pop().unwrap();
            self.node = Some(out);
            self.finished = true;
        }
    }
}

/// BDS08 authentication-path traversal state.
///
/// `Clone` is deliberate: `sign_stateful` clones the state before advancing,
/// so the caller's pre-sign snapshot remains a valid standalone state.
/// This is the Rust analogue of Python's `bds_copy_state` — `Vec` and
/// `VecDeque` deep-copy by default.
#[derive(Clone)]
pub struct BdsState {
    /// Tree height (= tau).
    pub h: usize,
    /// BDS08 traversal parameter: `h - k` is even, `k >= 2`.
    pub k: usize,
    /// Current leaf index — the leaf whose auth path `auth` represents.
    pub phi: usize,
    /// Authentication path: `auth[level]` is the sibling at that level.
    pub auth: Vec<Vec<i64>>,
    /// Keep slots: `keep[level]` holds the previous auth node when the
    /// algorithm saves it for a future height-`tau_local` merge.
    pub keep: Vec<Option<Vec<i64>>>,
    /// Pre-computed right-node queues for heights `[h - k, h - 1)`.
    /// Each queue is consumed in FIFO order via `pop_front`.
    pub retain: Vec<VecDeque<Vec<i64>>>,
    /// Treehash instances, one per height in `[0, h - k)`.
    pub treehash: Vec<TreehashInst>,
}

/// Pick the BDS08 general-version parameter K.
///
/// The BDS08 general traversal requires `K >= 2` with `(tau - K)` even.  We take the
/// smallest valid K: 2 for even tau, 3 for odd tau.  For `tau < 2` the
/// algorithm degenerates — we return `tau`, which yields an empty
/// treehash/retain range and the walk reduces to a plain auth-path init.
pub fn bds_choose_k(tau: usize) -> usize {
    if tau < 2 {
        return tau;
    }
    if tau.is_multiple_of(2) {
        2
    } else {
        3
    }
}

/// Predicate: does BDS init want to save the node at `(level, idx)`?
fn bds_wants(level: usize, idx: usize, h: usize, k: usize) -> bool {
    if level >= h {
        return false;
    }
    // auth[level] ← y_level[1]
    if idx == 1 {
        return true;
    }
    // treehash[level].node ← y_level[3] for level in [0, h-k)
    if k < h && level < h - k && idx == 3 {
        return true;
    }
    // retain[level] ← y_level[{3, 5, 7, ...}] for level in [h-k, h-1)
    if level >= h.saturating_sub(k) && level < h.saturating_sub(1) && idx >= 3 && (idx & 1) == 1 {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// (2η+1)-ary decomposition helpers
// ---------------------------------------------------------------------------

/// Decompose a polynomial (d,) into kappa digit polynomials (kappa x d),
/// writing into a caller-provided flat buffer of length `kappa * d`.
///
/// Inlines the per-coefficient decomposition to avoid d small Vec
/// allocations per call (critical for the 2M+ leaf evaluations).
fn dec_poly_into(a: &[i64], out: &mut [i64], modulus: u64, kappa: usize, profile: &Profile) {
    let d = profile.d;
    let eta = profile.eta;
    let b_val = 2 * eta + 1;
    let m = modulus as i64;
    let half_m = m / 2;
    debug_assert_eq!(out.len(), kappa * d);
    for i in 0..d {
        let mut c = a[i].rem_euclid(m);
        if c > half_m {
            c -= m;
        }
        for ki in 0..kappa {
            let mut r = c.rem_euclid(b_val);
            if r > eta {
                r -= b_val;
            }
            out[ki * d + i] = r;
            c = (c - r) / b_val;
        }
    }
}

/// Allocating wrapper — kept for callers that still want a fresh `Vec`.
fn dec_poly(a: &[i64], modulus: u64, kappa: usize, profile: &Profile) -> Vec<i64> {
    let mut out = vec![0i64; kappa * profile.d];
    dec_poly_into(a, &mut out, modulus, kappa, profile);
    out
}

/// Reconstruct polynomial from kappa digit polys (kappa x d) -> (d,).
fn proj_poly(digits: &[i64], kappa: usize, profile: &Profile) -> Vec<i64> {
    let d = profile.d;
    let b_val = 2 * profile.eta + 1;
    let mut result = vec![0i64; d];
    let mut base: i64 = 1;
    for k in 0..kappa {
        for i in 0..d {
            result[i] += digits[k * d + i] * base;
        }
        base *= b_val;
    }
    result
}

/// Reconstruct polynomial from kappa digit polys, in i128, no mod q.
///
/// Implements proj_{eta,kappa} from Sec 6.2 (codomain R). Used for the
/// proj-norm bound check in vrfy. i128 is needed because at wVrfy
/// (beta = 2*beta_agg) the worst-case sum overflows i64.
fn proj_poly_zz(digits: &[i64], kappa: usize, profile: &Profile) -> Vec<i128> {
    let d = profile.d;
    let b_val = (2 * profile.eta + 1) as i128;
    let mut result = vec![0i128; d];
    let mut base: i128 = 1;
    for k in 0..kappa {
        for i in 0..d {
            result[i] += digits[k * d + i] as i128 * base;
        }
        base *= b_val;
    }
    result
}

/// Decompose poly vector (n x d) into (n*kappa x d) — profile-aware real impl.
pub(crate) fn dec_vec_with_profile(
    v: &[i64],
    n: usize,
    modulus: u64,
    kappa: usize,
    profile: &Profile,
) -> Vec<i64> {
    let d = profile.d;
    let mut result = vec![0i64; n * kappa * d];
    for i in 0..n {
        let dec = dec_poly(&v[i * d..(i + 1) * d], modulus, kappa, profile);
        result[i * kappa * d..(i + 1) * kappa * d].copy_from_slice(&dec);
    }
    result
}

// Legacy wrapper `dec_vec(v, n, modulus, kappa)` removed:
// the only crate-internal caller (`materialized.rs`) was migrated to
// `dec_vec_with_profile` directly.

/// Reconstruct poly vector from (n*kappa x d) -> (n x d).
fn proj_vec(digits: &[i64], n: usize, kappa: usize, profile: &Profile) -> Vec<i64> {
    let d = profile.d;
    let mut result = vec![0i64; n * d];
    for i in 0..n {
        let p = proj_poly(&digits[i * kappa * d..(i + 1) * kappa * d], kappa, profile);
        result[i * d..(i + 1) * d].copy_from_slice(&p);
    }
    result
}

// ---------------------------------------------------------------------------
// Label functions
// ---------------------------------------------------------------------------

/// Project (omega*kappa x d) label to R_q^omega — profile-aware real impl.
pub fn proj_label_with_profile(label: &[i64], profile: &Profile) -> Vec<i64> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let q_hvc = profile.q_hvc() as i64;
    let mut result = vec![0i64; omega * d];
    for r in 0..omega {
        let p = proj_poly(&label[r * kappa * d..(r + 1) * kappa * d], kappa, profile);
        for i in 0..d {
            result[r * d + i] = p[i].rem_euclid(q_hvc);
        }
    }
    result
}

/// Unreduced i128 projection of an (omega*kappa x d) label, no mod q.
///
/// Returns (omega x d) of i128. Used in vrfy for the proj-norm bound
/// check (paper Fig. HVC step 4c uses proj_{eta,kappa}, not proj_q).
pub fn proj_label_zz_with_profile(label: &[i64], profile: &Profile) -> Vec<i128> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let mut result = vec![0i128; omega * d];
    for r in 0..omega {
        let p = proj_poly_zz(&label[r * kappa * d..(r + 1) * kappa * d], kappa, profile);
        result[r * d..(r + 1) * d].copy_from_slice(&p);
    }
    result
}


/// Compute leaf label for slot t given public key T_t — profile-aware real impl.
///
/// Fused version: decomposes each original polynomial and immediately
/// accumulates into NTT-domain row accumulators, avoiding the full
/// intermediate decomposed vector.
pub fn leaf_label_with_profile(
    t_flat: &[i64],
    b_mat_ntt: &[u32],
    rp: &RingParams,
    profile: &Profile,
) -> Vec<i64> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let kappa_prime = profile.kappa_prime;
    let rho = profile.k; // rho = K
    let nu = profile.n; // nu = N (KOTS)
    let q_kots = profile.q_kots();
    let q_hvc = profile.q_hvc();
    let b_rows = omega;
    let b_cols = rho * nu * kappa_prime;

    let mut row_acc = vec![0u32; b_rows * d];
    let mut digit_ntt = vec![0u32; d];
    // Per-p scratch for the kappa_prime digits.  Reused across all
    // rho*nu iterations to avoid (kappa_prime * d) i64 alloc/free per p.
    let mut digits = vec![0i64; kappa_prime * d];

    // Columns-outer / rows-inner, fusing (forward NTT of digit) with
    // the mac_ntt row sweep so digit_ntt stays hot in L1.  A "batched
    // forward NTTs + row-major B" rewrite via mat_vec_prentt was tried
    // and cost ~10% more: it introduces a (b_cols * d * 4)-byte vec_hat
    // round-trip through L2 for no cache-locality gain, because B's
    // total footprint (omega * b_cols * d * 4 ≈ 360 KB at d = 256)
    // already fits in L2 and strided access is cheap after warm-up.
    for p in 0..(rho * nu) {
        dec_poly_into(
            &t_flat[p * d..(p + 1) * d],
            &mut digits,
            q_kots,
            kappa_prime,
            profile,
        );
        for ki in 0..kappa_prime {
            poly_to_ntt_buf(&digits[ki * d..(ki + 1) * d], rp, &mut digit_ntt);
            let col = p * kappa_prime + ki;
            for i in 0..b_rows {
                mac_ntt(
                    &mut row_acc[i * d..(i + 1) * d],
                    &b_mat_ntt[(i * b_cols + col) * d..(i * b_cols + col + 1) * d],
                    &digit_ntt,
                    rp.q32,
                    rp.q_inv,
                );
            }
        }
    }

    // Fused tail: inverse-NTT each row straight into a small scratch and
    // decompose directly into the correct window of the output buffer.
    // Avoids the intermediate `raw` allocation of size omega * d.
    let mut result = vec![0i64; omega * kappa * d];
    let mut row_raw = vec![0i64; d];
    for i in 0..b_rows {
        ntt_to_poly_buf(&mut row_acc[i * d..(i + 1) * d], rp, &mut row_raw);
        dec_poly_into(
            &row_raw,
            &mut result[i * kappa * d..(i + 1) * kappa * d],
            q_hvc,
            kappa,
            profile,
        );
    }
    result
}

fn leaf_label_with_profile_u64(
    t_flat: &[i64],
    b_mat_ntt: &[u64],
    rp: &RingParams64,
    profile: &Profile,
) -> Vec<i64> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let kappa_prime = profile.kappa_prime;
    let rho = profile.k;
    let nu = profile.n;
    let q_kots = profile.q_kots();
    let q_hvc = profile.q_hvc();
    let b_rows = omega;
    let b_cols = rho * nu * kappa_prime;

    let mut row_acc = vec![0u64; b_rows * d];
    let mut digit_ntt = vec![0u64; d];
    let mut digits = vec![0i64; kappa_prime * d];

    for p in 0..(rho * nu) {
        dec_poly_into(
            &t_flat[p * d..(p + 1) * d],
            &mut digits,
            q_kots,
            kappa_prime,
            profile,
        );
        for ki in 0..kappa_prime {
            poly_to_ntt_buf_u64(&digits[ki * d..(ki + 1) * d], rp, &mut digit_ntt);
            let col = p * kappa_prime + ki;
            for i in 0..b_rows {
                mac_ntt_u64(
                    &mut row_acc[i * d..(i + 1) * d],
                    &b_mat_ntt[(i * b_cols + col) * d..(i * b_cols + col + 1) * d],
                    &digit_ntt,
                    rp.q,
                    rp.q_inv,
                );
            }
        }
    }

    let mut result = vec![0i64; omega * kappa * d];
    let mut row_raw = vec![0i64; d];
    for i in 0..b_rows {
        ntt_to_poly_buf_u64(&mut row_acc[i * d..(i + 1) * d], rp, &mut row_raw);
        dec_poly_into(
            &row_raw,
            &mut result[i * kappa * d..(i + 1) * kappa * d],
            q_hvc,
            kappa,
            profile,
        );
    }
    result
}

pub fn leaf_label_with_profile_any(
    t_flat: &[i64],
    b_mat_ntt: &HvcNttVec,
    profile: &Profile,
) -> Vec<i64> {
    match (&profile.hvc_ring, b_mat_ntt) {
        (HvcRing::U32(rp), HvcNttVec::U32(ntt)) => leaf_label_with_profile(t_flat, ntt, rp, profile),
        (HvcRing::U64(rp), HvcNttVec::U64(ntt)) => {
            leaf_label_with_profile_u64(t_flat, ntt, rp, profile)
        }
        _ => panic!(
            "profile {:?}: HVC ring backend does not match pre-NTT matrix backend",
            profile.name
        ),
    }
}


/// Compute internal node label from left and right child labels — profile-aware real impl.
///
/// Uses fused pair multiplication: both products are accumulated in a
/// single NTT-domain accumulator, saving one set of inverse NTTs and
/// the coefficient-domain addition.
pub fn internal_label_with_profile(
    left: &[i64],
    right: &[i64],
    a0_ntt: &[u32],
    a1_ntt: &[u32],
    rp: &RingParams,
    profile: &Profile,
) -> Vec<i64> {
    let omega = profile.omega;
    let kappa = profile.kappa;
    let q_hvc = profile.q_hvc();
    let combined = mat_vec_prentt_pair(a0_ntt, left, a1_ntt, right, omega, omega * kappa, rp);
    dec_vec_with_profile(&combined, omega, q_hvc, kappa, profile)
}

fn internal_label_with_profile_u64(
    left: &[i64],
    right: &[i64],
    a0_ntt: &[u64],
    a1_ntt: &[u64],
    rp: &RingParams64,
    profile: &Profile,
) -> Vec<i64> {
    let omega = profile.omega;
    let kappa = profile.kappa;
    let q_hvc = profile.q_hvc();
    let combined = mat_vec_prentt_pair_u64(a0_ntt, left, a1_ntt, right, omega, omega * kappa, rp);
    dec_vec_with_profile(&combined, omega, q_hvc, kappa, profile)
}

pub fn internal_label_with_profile_any(
    left: &[i64],
    right: &[i64],
    a0_ntt: &HvcNttVec,
    a1_ntt: &HvcNttVec,
    profile: &Profile,
) -> Vec<i64> {
    match (&profile.hvc_ring, a0_ntt, a1_ntt) {
        (HvcRing::U32(rp), HvcNttVec::U32(a0), HvcNttVec::U32(a1)) => {
            internal_label_with_profile(left, right, a0, a1, rp, profile)
        }
        (HvcRing::U64(rp), HvcNttVec::U64(a0), HvcNttVec::U64(a1)) => {
            internal_label_with_profile_u64(left, right, a0, a1, rp, profile)
        }
        _ => panic!(
            "profile {:?}: HVC ring backend does not match pre-NTT matrix backend",
            profile.name
        ),
    }
}

pub fn hvc_mat_vec_prentt(
    mat_ntt: &HvcNttVec,
    vec: &[i64],
    rows: usize,
    cols: usize,
    profile: &Profile,
) -> Vec<i64> {
    match (&profile.hvc_ring, mat_ntt) {
        (HvcRing::U32(rp), HvcNttVec::U32(mat)) => mat_vec_prentt(mat, vec, rows, cols, rp),
        (HvcRing::U64(rp), HvcNttVec::U64(mat)) => mat_vec_prentt_u64(mat, vec, rows, cols, rp),
        _ => panic!(
            "profile {:?}: HVC ring backend does not match pre-NTT matrix backend",
            profile.name
        ),
    }
}

pub fn hvc_mat_vec_prentt_pair(
    mat0_ntt: &HvcNttVec,
    vec0: &[i64],
    mat1_ntt: &HvcNttVec,
    vec1: &[i64],
    rows: usize,
    cols: usize,
    profile: &Profile,
) -> Vec<i64> {
    match (&profile.hvc_ring, mat0_ntt, mat1_ntt) {
        (HvcRing::U32(rp), HvcNttVec::U32(mat0), HvcNttVec::U32(mat1)) => {
            mat_vec_prentt_pair(mat0, vec0, mat1, vec1, rows, cols, rp)
        }
        (HvcRing::U64(rp), HvcNttVec::U64(mat0), HvcNttVec::U64(mat1)) => {
            mat_vec_prentt_pair_u64(mat0, vec0, mat1, vec1, rows, cols, rp)
        }
        _ => panic!(
            "profile {:?}: HVC ring backend does not match pre-NTT matrix backend",
            profile.name
        ),
    }
}

fn slot_to_addr(t: usize, tau: usize) -> Vec<u8> {
    (0..tau).map(|j| ((t >> (tau - 1 - j)) & 1) as u8).collect()
}

// ---------------------------------------------------------------------------
// Streaming tree algorithms (O(τ) space)
// ---------------------------------------------------------------------------

/// Sequential stack-based subtree root computation.
/// Processes [leaf_start, leaf_start + leaf_count) in order; leaf_count must be a power of 2.
/// Memory: O(log(leaf_count)) labels on the stack.
fn hvc_subtree_root_seq<F>(
    pp: &HvcPp,
    leaf_start: usize,
    leaf_count: usize,
    leaf_fn: &F,
) -> Vec<i64>
where
    F: Fn(usize) -> Vec<i64>,
{
    debug_assert!(leaf_count.is_power_of_two());
    let mut stack: Vec<(usize, Vec<i64>)> = Vec::new();
    for i in 0..leaf_count {
        let opk = leaf_fn(leaf_start + i);
        let leaf = leaf_label_with_profile_any(&opk, &pp.b_mat_ntt, pp.profile);
        let mut node = (0usize, leaf);
        while stack.last().is_some_and(|(h, _)| *h == node.0) {
            let (_, left) = stack.pop().unwrap();
            let merged = internal_label_with_profile_any(
                &left,
                &node.1,
                &pp.a0_ntt,
                &pp.a1_ntt,
                pp.profile,
            );
            node = (node.0 + 1, merged);
        }
        stack.push(node);
    }
    debug_assert_eq!(stack.len(), 1);
    stack.pop().unwrap().1
}

/// Divide-and-conquer subtree root with parallel splits via rayon.
/// Falls back to the sequential algorithm for small subtrees.
fn hvc_subtree_root<F>(
    pp: &HvcPp,
    leaf_start: usize,
    leaf_count: usize,
    leaf_fn: &F,
) -> Vec<i64>
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    const PAR_THRESHOLD: usize = 256;
    if leaf_count <= PAR_THRESHOLD {
        hvc_subtree_root_seq(pp, leaf_start, leaf_count, leaf_fn)
    } else {
        let half = leaf_count / 2;
        let (left, right) = rayon::join(
            || hvc_subtree_root(pp, leaf_start, half, leaf_fn),
            || hvc_subtree_root(pp, leaf_start + half, half, leaf_fn),
        );
        internal_label_with_profile_any(&left, &right, &pp.a0_ntt, &pp.a1_ntt, pp.profile)
    }
}

/// Compute the authentication data for a single target leaf in one targeted traversal.
///
/// Returns:
/// - subtree root
/// - path nodes bottom-up including the leaf and subtree root
/// - sibling labels bottom-up
fn hvc_open_target<F>(
    pp: &HvcPp,
    leaf_start: usize,
    leaf_count: usize,
    t: usize,
    leaf_fn: &F,
    leaf_opk_known: Option<&[i64]>,
) -> (Vec<i64>, Vec<Vec<i64>>, Vec<Vec<i64>>)
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    debug_assert!(leaf_count.is_power_of_two());
    if leaf_count == 1 {
        let leaf_opk_buf;
        let leaf_opk = if let Some(opk) = leaf_opk_known {
            opk
        } else {
            leaf_opk_buf = leaf_fn(leaf_start);
            &leaf_opk_buf
        };
        let leaf = leaf_label_with_profile_any(leaf_opk, &pp.b_mat_ntt, pp.profile);
        return (leaf.clone(), vec![leaf], Vec::new());
    }

    let half = leaf_count / 2;
    let mid = leaf_start + half;
    if t < mid {
        let ((left_root, mut path_nodes, mut siblings), right_root) = rayon::join(
            || hvc_open_target(pp, leaf_start, half, t, leaf_fn, leaf_opk_known),
            || hvc_subtree_root(pp, mid, half, leaf_fn),
        );
        let parent = internal_label_with_profile_any(
            &left_root,
            &right_root,
            &pp.a0_ntt,
            &pp.a1_ntt,
            pp.profile,
        );
        siblings.push(right_root);
        path_nodes.push(parent.clone());
        (parent, path_nodes, siblings)
    } else {
        let (left_root, (right_root, mut path_nodes, mut siblings)) = rayon::join(
            || hvc_subtree_root(pp, leaf_start, half, leaf_fn),
            || hvc_open_target(pp, mid, half, t, leaf_fn, leaf_opk_known),
        );
        let parent = internal_label_with_profile_any(
            &left_root,
            &right_root,
            &pp.a0_ntt,
            &pp.a1_ntt,
            pp.profile,
        );
        siblings.push(left_root);
        path_nodes.push(parent.clone());
        (parent, path_nodes, siblings)
    }
}

// ---------------------------------------------------------------------------
// BDS08 fused init + advance + opening
// ---------------------------------------------------------------------------

/// `(level, idx, label)` collected during the BDS-init walk.
type SavedNode = (usize, usize, Vec<i64>);

/// Parallel divide-and-conquer subtree walk that also collects every
/// `(level, idx)` node needed by BDS initialisation.
///
/// Returns `(root_label, saved_nodes)` where `saved_nodes` is a list
/// in left-to-right, bottom-up encounter order.  Within any single
/// level the `idx` values are monotonically increasing, so the retain
/// queues end up in the correct consumption order for `pop_front`.
fn bds_fused_walk<F>(
    pp: &HvcPp,
    leaf_start: usize,
    leaf_count: usize,
    leaf_fn: &F,
    h: usize,
    k: usize,
) -> (Vec<i64>, Vec<SavedNode>)
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    const PAR_THRESHOLD: usize = 256;
    debug_assert!(leaf_count.is_power_of_two());

    if leaf_count == 1 {
        let leaf = leaf_label_with_profile_any(&leaf_fn(leaf_start), &pp.b_mat_ntt, pp.profile);
        let mut saved = Vec::new();
        if bds_wants(0, leaf_start, h, k) {
            saved.push((0usize, leaf_start, leaf.clone()));
        }
        return (leaf, saved);
    }

    let half = leaf_count / 2;
    let (left, right) = if leaf_count > PAR_THRESHOLD {
        rayon::join(
            || bds_fused_walk(pp, leaf_start, half, leaf_fn, h, k),
            || bds_fused_walk(pp, leaf_start + half, half, leaf_fn, h, k),
        )
    } else {
        (
            bds_fused_walk(pp, leaf_start, half, leaf_fn, h, k),
            bds_fused_walk(pp, leaf_start + half, half, leaf_fn, h, k),
        )
    };
    let (left_root, mut left_saved) = left;
    let (right_root, right_saved) = right;

    let merged = internal_label_with_profile_any(
        &left_root,
        &right_root,
        &pp.a0_ntt,
        &pp.a1_ntt,
        pp.profile,
    );
    left_saved.extend(right_saved);

    let merged_level = leaf_count.trailing_zeros() as usize;
    let merged_idx = leaf_start / leaf_count;
    if bds_wants(merged_level, merged_idx, h, k) {
        left_saved.push((merged_level, merged_idx, merged.clone()));
    }

    (merged, left_saved)
}

/// Fused initialisation: produce the HVC commitment AND the initial BDS
/// traversal state (at phi=0) in a single parallel tree walk.  Replaces
/// `hvc_com` in stateful keygen flows to avoid an extra O(2^tau) pass.
pub fn bds_init<F>(pp: &HvcPp, leaf_fn: F) -> (HvcCom, BdsState)
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    let h = pp.tau;
    let k = bds_choose_k(h);
    let n_slots = 1usize << h;

    let mut auth: Vec<Vec<i64>> = vec![Vec::new(); h];
    let keep: Vec<Option<Vec<i64>>> = vec![None; h];
    let mut retain: Vec<VecDeque<Vec<i64>>> = (0..h).map(|_| VecDeque::new()).collect();
    let treehash_len = h.saturating_sub(k);
    let mut treehash: Vec<TreehashInst> = (0..treehash_len).map(TreehashInst::new).collect();

    let (root_label, saved) = bds_fused_walk(pp, 0, n_slots, &leaf_fn, h, k);

    for (level, idx, lab) in saved {
        if idx == 1 && level < h {
            auth[level] = lab;
        } else if k < h && level < h - k && idx == 3 {
            treehash[level].set_ready(lab);
        } else if level >= h.saturating_sub(k)
            && level < h.saturating_sub(1)
            && idx >= 3
            && (idx & 1) == 1
        {
            retain[level].push_back(lab);
        }
    }

    // Sanity: auth[level] should be populated for every level < h.
    debug_assert!(auth.iter().all(|a| !a.is_empty() || h == 0));

    let state = BdsState {
        h,
        k,
        phi: 0,
        auth,
        keep,
        retain,
        treehash,
    };

    (
        HvcCom(proj_label_with_profile(&root_label, pp.profile)),
        state,
    )
}

/// Advance BDS state from phi to phi+1 using the BDS08 general traversal.
///
/// Preconditions: `state.phi < 2^state.h - 1`.  Panics (internal-invariant
/// check) if the treehash/retain consumption order is inconsistent with the
/// algorithm — those would indicate a bug in `bds_init`, not user input.
pub fn bds_advance<F>(state: &mut BdsState, pp: &HvcPp, leaf_fn: &F)
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    let h = state.h;
    let k = state.k;
    let phi = state.phi;
    debug_assert!(phi + 1 < (1usize << h), "bds_advance past last leaf");

    // 1. tau_local: height of the first parent of leaf phi that is a left node.
    let tau_l = if phi & 1 == 0 {
        0
    } else {
        let mut t = 0usize;
        let mut m = phi + 1;
        while m & 1 == 0 {
            t += 1;
            m >>= 1;
        }
        t
    };

    // 2. Save current Auth[tau_l] into Keep[tau_l] if it will be needed later.
    if tau_l + 1 < h && ((phi >> (tau_l + 1)) & 1) == 0 {
        state.keep[tau_l] = Some(state.auth[tau_l].clone());
    }

    if tau_l == 0 {
        // 3. New leaf-level auth is the left sibling of phi+1, i.e. leaf(phi).
        state.auth[0] = leaf_label_with_profile_any(&leaf_fn(phi), &pp.b_mat_ntt, pp.profile);
    } else {
        // 4(a) New right auth node on height tau_l.
        let prev_keep = state.keep[tau_l - 1]
            .take()
            .expect("bds_advance: keep[tau_l - 1] missing");
        let new_parent = internal_label_with_profile_any(
            &state.auth[tau_l - 1],
            &prev_keep,
            &pp.a0_ntt,
            &pp.a1_ntt,
            pp.profile,
        );
        state.auth[tau_l] = new_parent;

        // 4(b) Fetch new right auth nodes on heights 0..tau_l-1.
        for level in 0..tau_l {
            if level + k < h {
                // Treehash range: consume the completed output.
                let th = &mut state.treehash[level];
                debug_assert!(th.finished, "treehash[{level}] not ready at phi={phi}");
                let node = th
                    .take_ready()
                    .expect("bds_advance: treehash output missing");
                state.auth[level] = node;
            } else {
                // Retain range: consume the next queued right node.
                state.auth[level] = state.retain[level]
                    .pop_front()
                    .expect("bds_advance: retain queue empty");
            }
        }

        // 4(c) Re-initialise treehash instances for heights that need a
        // new node computed for a future consumption.
        let reinit_upper = tau_l.min(h.saturating_sub(k));
        for level in 0..reinit_upper {
            let new_start = phi + 1 + 3 * (1usize << level);
            if new_start < (1usize << h) {
                state.treehash[level].initialize(new_start);
            }
        }
    }

    // 5. Spend the budget of (h - k) / 2 treehash updates.
    let budget = if h > k { (h - k) / 2 } else { 0 };
    for _ in 0..budget {
        let mut best: Option<usize> = None;
        let mut best_metric = usize::MAX;
        for (i, th) in state.treehash.iter().enumerate() {
            let metric = th.height_metric();
            if metric < best_metric {
                best_metric = metric;
                best = Some(i);
            }
        }
        match best {
            Some(i) if best_metric != usize::MAX => {
                state.treehash[i].update(pp, leaf_fn);
            }
            _ => break,
        }
    }

    state.phi = phi + 1;
}

/// Assemble an HVC opening for slot `t` from a BDS state.
///
/// The state must hold the auth path for leaf `t`.  Returns an
/// `HvcOpening` byte-identical to what `hvc_open` would produce for the
/// same slot, same leaf_fn.
///
/// Returns `Err(LemurError::VerifyFailed)` if the state's internal `phi`
/// disagrees with `t` — this guards against user-side state-machine bugs
/// and is a real error (not a `debug_assert`) so it survives release
/// builds without assertions.
pub fn bds_opening<F>(
    state: &BdsState,
    pp: &HvcPp,
    t: usize,
    leaf_fn: &F,
) -> Result<HvcOpening, LemurError>
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    if state.phi != t {
        return Err(LemurError::VerifyFailed);
    }

    let leaf_opk = leaf_fn(t);
    let u = dec_vec_with_profile(
        &leaf_opk,
        pp.profile.k * pp.profile.n,
        pp.profile.q_kots(),
        pp.profile.kappa_prime,
        pp.profile,
    );

    // sibling_labels[j-1] corresponds to the sibling at depth j (root-relative),
    // which is at BDS height tau - j.  Reverse `auth` to match.
    let sibling_labels: Vec<Vec<i64>> = state.auth.iter().rev().cloned().collect();

    let path_labels = reconstruct_path_labels_ind(pp, t, &sibling_labels, &u);

    Ok(HvcOpening {
        path_labels,
        sibling_labels,
        u,
    })
}

// ---------------------------------------------------------------------------
// HVC algorithms
// ---------------------------------------------------------------------------

/// Compute the aggregation-opening norm bound for a profile + tau.
///
/// Reads omega, kappa, kappa_prime, rho, nu, alpha_w, eta, n_signers
/// from the profile.  The tau override lets `hvc_setup_with_profile_and_tau`
/// switch tree depth for fast test vectors without constructing a
/// different profile.
pub fn compute_beta_agg_with_profile(tau: usize, profile: &Profile) -> i64 {
    let d = profile.d;
    let n_signers = profile.n_signers as f64;
    let omega = profile.omega as f64;
    let kappa = profile.kappa as f64;
    let kappa_prime = profile.kappa_prime as f64;
    let rho = profile.k as f64; // rho = K
    let nu = profile.n as f64; // nu = N (kots)
    let alpha_w = profile.alpha_w as f64;
    let eta = profile.eta as f64;
    let eps: f64 = 1.0 / 32768.0;
    let n_coeffs = n_signers
        * (2.0 * tau as f64 * omega * kappa + rho * nu * kappa_prime + 2.0 * tau as f64 * omega);
    let inner = 2.0 * alpha_w * n_signers * ((2.0 * d as f64 / eps).ln() + n_coeffs.ln());
    (eta * inner.sqrt()).ceil() as i64
}

fn compute_beta_encode_with_profile(beta_agg: i64, profile: &Profile) -> i64 {
    // ceil(beta_agg / (2 * eta))
    (beta_agg + 2 * profile.eta - 1) / (2 * profile.eta)
}

/// HVC Setup (profile-aware, real impl): expand (B, A0, A1) from seed.
///
/// Uses `profile.tau` as the tree depth (call `hvc_setup_with_profile_and_tau`
/// to override, e.g. for fast test vectors).
pub fn hvc_setup_with_profile(seed: &[u8], profile: &'static Profile) -> HvcPp {
    hvc_setup_with_profile_and_tau(seed, profile, profile.tau)
}

/// HVC Setup (profile-aware with custom tau, real impl).
pub fn hvc_setup_with_profile_and_tau(seed: &[u8], profile: &'static Profile, tau: usize) -> HvcPp {
    let d = profile.d;
    let q = profile.q_hvc();
    let omega = profile.omega;
    let kappa = profile.kappa;
    let kappa_prime = profile.kappa_prime;
    let rho = profile.k; // rho = K
    let nu = profile.n; // nu = N (kots)

    let b_rows = omega;
    let b_cols = rho * nu * kappa_prime;
    let a_cols = omega * kappa;

    let mut b_mat = vec![0i64; b_rows * b_cols * d];
    for i in 0..b_rows {
        for j in 0..b_cols {
            let mut xof = hvc_setup_xof(seed, i, j, b"B");
            let poly = xof_uniform_poly(&mut xof as &mut dyn XofReader, q, d);
            b_mat[(i * b_cols + j) * d..(i * b_cols + j + 1) * d].copy_from_slice(&poly);
        }
    }

    let mut a0 = vec![0i64; omega * a_cols * d];
    for i in 0..omega {
        for j in 0..a_cols {
            let mut xof = hvc_setup_xof(seed, i, j, b"A0");
            let poly = xof_uniform_poly(&mut xof as &mut dyn XofReader, q, d);
            a0[(i * a_cols + j) * d..(i * a_cols + j + 1) * d].copy_from_slice(&poly);
        }
    }

    let mut a1 = vec![0i64; omega * a_cols * d];
    for i in 0..omega {
        for j in 0..a_cols {
            let mut xof = hvc_setup_xof(seed, i, j, b"A1");
            let poly = xof_uniform_poly(&mut xof as &mut dyn XofReader, q, d);
            a1[(i * a_cols + j) * d..(i * a_cols + j + 1) * d].copy_from_slice(&poly);
        }
    }

    let (b_mat_ntt, a0_ntt, a1_ntt) = match &profile.hvc_ring {
        HvcRing::U32(rp) => (
            HvcNttVec::U32(mat_to_ntt(&b_mat, b_rows, b_cols, rp)),
            HvcNttVec::U32(mat_to_ntt(&a0, omega, a_cols, rp)),
            HvcNttVec::U32(mat_to_ntt(&a1, omega, a_cols, rp)),
        ),
        HvcRing::U64(rp) => (
            HvcNttVec::U64(mat_to_ntt_u64(&b_mat, b_rows, b_cols, rp)),
            HvcNttVec::U64(mat_to_ntt_u64(&a0, omega, a_cols, rp)),
            HvcNttVec::U64(mat_to_ntt_u64(&a1, omega, a_cols, rp)),
        ),
    };
    let beta_agg = compute_beta_agg_with_profile(tau, profile);
    let beta_encode = compute_beta_encode_with_profile(beta_agg, profile);
    HvcPp {
        b_mat,
        a0,
        a1,
        b_mat_ntt,
        a0_ntt,
        a1_ntt,
        profile,
        tau,
        beta_agg,
        beta_encode,
    }
}


/// HVC Com: compute commitment from a leaf function.
///
/// `leaf_fn(t)` must return the flat KOTS public key for slot `t` as `(rho*nu*d,)` i64.
/// Streams over all 2^tau leaves with O(tau) working memory.
pub fn hvc_com<F>(pp: &HvcPp, leaf_fn: F) -> HvcCom
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    let n_slots = 1usize << pp.tau;
    let root = hvc_subtree_root(pp, 0, n_slots, &leaf_fn);
    HvcCom(proj_label_with_profile(&root, pp.profile))
}

/// HVC Open: generate opening for slot t.
///
/// Computes τ sibling subtree roots (total O(2^τ) leaf calls) then builds
/// path labels bottom-up. All sibling subtrees are independent and may run
/// in parallel via the rayon-enabled `hvc_subtree_root`.
pub fn hvc_open<F>(pp: &HvcPp, t: usize, leaf_fn: F) -> HvcOpening
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    hvc_open_with_known_leaf(pp, t, None, leaf_fn)
}

/// HVC Open with optional precomputed leaf OPK for slot t.
pub fn hvc_open_with_known_leaf<F>(
    pp: &HvcPp,
    t: usize,
    leaf_opk_known: Option<&[i64]>,
    leaf_fn: F,
) -> HvcOpening
where
    F: Fn(usize) -> Vec<i64> + Send + Sync,
{
    let tau = pp.tau;

    // Compute leaf for slot t (called once; not in any sibling subtree).
    let leaf_opk_buf;
    let leaf_opk: &[i64] = if let Some(opk) = leaf_opk_known {
        opk
    } else {
        leaf_opk_buf = leaf_fn(t);
        &leaf_opk_buf
    };
    let u = dec_vec_with_profile(
        leaf_opk,
        pp.profile.k * pp.profile.n,
        pp.profile.q_kots(),
        pp.profile.kappa_prime,
        pp.profile,
    );
    let n_slots = 1usize << tau;
    let (_root, path_nodes, sibling_labels_bottom_up) =
        hvc_open_target(pp, 0, n_slots, t, &leaf_fn, Some(leaf_opk));
    let sibling_labels: Vec<Vec<i64>> = sibling_labels_bottom_up.into_iter().rev().collect();
    let path_labels: Vec<Vec<i64>> = path_nodes[..tau].iter().rev().cloned().collect();

    HvcOpening {
        path_labels,
        sibling_labels,
        u,
    }
}

/// HVC Verify: returns the recovered KOTS pk on success.
pub fn hvc_vrfy(
    pp: &HvcPp,
    c: &HvcCom,
    t: usize,
    opening: &HvcOpening,
    beta: i64,
) -> Result<KotsPk, LemurError> {
    let tau = pp.tau;
    let t_bar = slot_to_addr(t, tau);
    let omega = pp.profile.omega;
    let kappa = pp.profile.kappa;
    let kappa_prime = pp.profile.kappa_prime;
    let rho = pp.profile.k; // rho = K
    let nu = pp.profile.n; // nu = N (kots)
    let eta = pp.profile.eta;
    let q_hvc = pp.profile.q_hvc() as i64;
    let q_kots = pp.profile.q_kots() as i64;

    if inf_norm(&opening.u) > beta {
        return Err(LemurError::VerifyFailed);
    }

    // hint starts as B*u (the leaf projection)
    let hint_raw =
        hvc_mat_vec_prentt(&pp.b_mat_ntt, &opening.u, omega, rho * nu * kappa_prime, pp.profile);
    let mut hint = hint_raw; // (omega x d)

    for j in (1..=tau).rev() {
        let p_bar_j = &opening.path_labels[j - 1];
        let s_j = &opening.sibling_labels[j - 1];

        let p_proj = proj_label_with_profile(p_bar_j, pp.profile);

        // Check that proj(path_label) == hint
        if p_proj != hint {
            return Err(LemurError::VerifyFailed);
        }

        // Norm check on labels
        if inf_norm(p_bar_j) > beta || inf_norm(s_j) > beta {
            return Err(LemurError::VerifyFailed);
        }

        // proj_{eta,kappa} (Sec 6.2 over ZZ, not mod q) — paper Fig. step 4c.
        // (2*eta+1)^kappa > q, so reducing mod q before bounding would
        // discard the carry the threshold is meant to constrain.
        let threshold = (q_hvc as i128 * beta as i128) / (2 * eta) as i128;
        let exceeds_threshold = |v: &[i64]| -> bool {
            let proj = proj_label_zz_with_profile(v, pp.profile);
            proj.iter().any(|x| x.abs() > threshold)
        };
        if exceeds_threshold(p_bar_j) || exceeds_threshold(s_j) {
            return Err(LemurError::VerifyFailed);
        }

        let bit = t_bar[j - 1];
        let (a_path_ntt, a_sib_ntt) = if bit == 0 {
            (&pp.a0_ntt, &pp.a1_ntt)
        } else {
            (&pp.a1_ntt, &pp.a0_ntt)
        };

        hint = hvc_mat_vec_prentt_pair(
            a_path_ntt,
            p_bar_j,
            a_sib_ntt,
            s_j,
            omega,
            omega * kappa,
            pp.profile,
        );
    }

    // Check root matches commitment
    if hint != c.0 {
        return Err(LemurError::VerifyFailed);
    }

    // Recover KOTS pk from u
    let opk_flat_raw = proj_vec(&opening.u, rho * nu, kappa_prime, pp.profile);
    let opk_flat: Vec<i64> = opk_flat_raw.iter().map(|&x| x.rem_euclid(q_kots)).collect();
    Ok(KotsPk(opk_flat))
}

pub fn hvc_ivrfy(
    pp: &HvcPp,
    c: &HvcCom,
    t: usize,
    opening: &HvcOpening,
) -> Result<KotsPk, LemurError> {
    hvc_vrfy(pp, c, t, opening, pp.profile.eta)
}

pub fn hvc_svrfy(
    pp: &HvcPp,
    c: &HvcCom,
    t: usize,
    opening: &HvcOpening,
) -> Result<KotsPk, LemurError> {
    hvc_vrfy(pp, c, t, opening, pp.beta_agg)
}

pub fn hvc_wvrfy(
    pp: &HvcPp,
    c: &HvcCom,
    t: usize,
    opening: &HvcOpening,
) -> Result<KotsPk, LemurError> {
    hvc_vrfy(pp, c, t, opening, 2 * pp.beta_agg)
}

// ---------------------------------------------------------------------------
// Opening aggregation helpers (used by lemur.rs)
// ---------------------------------------------------------------------------

/// Add two HVC openings componentwise over Z.
pub fn add_openings(a: &HvcOpening, b: &HvcOpening) -> HvcOpening {
    let path_labels = a
        .path_labels
        .iter()
        .zip(b.path_labels.iter())
        .map(|(p1, p2)| p1.iter().zip(p2.iter()).map(|(&x, &y)| x + y).collect())
        .collect();
    let sibling_labels = a
        .sibling_labels
        .iter()
        .zip(b.sibling_labels.iter())
        .map(|(s1, s2)| s1.iter().zip(s2.iter()).map(|(&x, &y)| x + y).collect())
        .collect();
    let u: Vec<i64> = a.u.iter().zip(b.u.iter()).map(|(&x, &y)| x + y).collect();
    HvcOpening {
        path_labels,
        sibling_labels,
        u,
    }
}

/// Scale HVC opening by scalar poly w (signed exact).
///
/// Pre-computes the NTT of w once and reuses it across all three
/// components (path labels, sibling labels, u), avoiding redundant
/// forward NTTs.
pub fn scale_opening(w: &[i64], opening: &HvcOpening, rp: &RingParams) -> HvcOpening {
    let d = rp.d;
    let w_hat = poly_to_ntt(w, rp);
    let scale_labels = |labels: &[Vec<i64>]| -> Vec<Vec<i64>> {
        labels
            .iter()
            .map(|label| {
                let n = label.len() / d;
                scale_vec_with_ntt_w(&w_hat, label, n, rp)
            })
            .collect()
    };
    let path_labels = scale_labels(&opening.path_labels);
    let sibling_labels = scale_labels(&opening.sibling_labels);
    let n_u = opening.u.len() / d;
    let u = scale_vec_with_ntt_w(&w_hat, &opening.u, n_u, rp);
    HvcOpening {
        path_labels,
        sibling_labels,
        u,
    }
}

fn scale_opening_u64(w: &[i64], opening: &HvcOpening, rp: &RingParams64) -> HvcOpening {
    let d = rp.d;
    let mut w_hat = vec![0u64; d];
    poly_to_ntt_buf_u64(w, rp, &mut w_hat);
    let scale_labels = |labels: &[Vec<i64>]| -> Vec<Vec<i64>> {
        labels
            .iter()
            .map(|label| {
                let n = label.len() / d;
                scale_vec_with_ntt_w_u64(&w_hat, label, n, rp)
            })
            .collect()
    };
    let path_labels = scale_labels(&opening.path_labels);
    let sibling_labels = scale_labels(&opening.sibling_labels);
    let n_u = opening.u.len() / d;
    let u = scale_vec_with_ntt_w_u64(&w_hat, &opening.u, n_u, rp);
    HvcOpening {
        path_labels,
        sibling_labels,
        u,
    }
}

pub fn scale_opening_any(w: &[i64], opening: &HvcOpening, profile: &Profile) -> HvcOpening {
    match &profile.hvc_ring {
        HvcRing::U32(rp) => scale_opening(w, opening, rp),
        HvcRing::U64(rp) => scale_opening_u64(w, opening, rp),
    }
}

// ---------------------------------------------------------------------------
// NTT-domain opening aggregation
// ---------------------------------------------------------------------------
//
// `aggregate_openings_any` computes `Σᵢ wᵢ · openingᵢ` over N signers in one
// pass.  The reference path (`scale_opening_any` per signer + `add_openings`
// to reduce) performs `N × (2τ+1) · ωκ` inverse NTTs — one per poly per
// signer — and accumulates in the coefficient domain.  Because the inverse
// NTT distributes over addition (the NTT is `R_q`-linear), we can swap the
// order of operations:
//
//   Σᵢ INTT(NTT(wᵢ) ⊗ NTT(openingᵢ))  =  INTT(Σᵢ NTT(wᵢ) ⊗ NTT(openingᵢ))
//
// and pay only `(2τ+1) · ωκ` inverse NTTs total — independent of N.  The
// per-signer forward NTTs are unavoidable (the openings on the wire are
// coefficient-domain bytes), but the inverse-NTT half of the work, the
// dominant cost at large N, drops by a factor of N.  Bit-exact result.
//
// Overflow safety (u64 HVC ring).  For `D256_K4` the HVC modulus is
// slightly above `2^53` and Montgomery products live in `[0, q)`.  Two
// safe-zone strategies are combined:
//
//   1. **Bounded chunk accumulation.**  Per-chunk we add up to
//      `REDUCE_INTERVAL` Montgomery products with `wrapping_add`; the
//      running value stays in `[0, chunk_size · q)`.  The chunk size is
//      capped dynamically so `chunk_size · (q-1) ≤ u64::MAX`; for the
//      shipped profile this keeps the default `REDUCE_INTERVAL = 1024`.
//      At the end of the chunk we reduce each slot mod q before it reaches
//      any signed reducer.
//
//   2. **Reduce-on-merge.**  When rayon combines two chunk accumulators
//      we add two values in `[0, q)` to get a value in `[0, 2q) ≈ 2^54`
//      and conditionally subtract `q`, restoring `[0, q)` invariantly.
//      No further headroom is consumed across the reduction tree.
//
// With this design the implementation is correct up to any practically
// representable N (the rayon reduction tree has `O(log N)` depth but the
// per-step invariant is the same).  The final inverse NTT operates on
// already-reduced inputs.
//
// For the u32 ring (not used by the shipped profile but kept for forward
// compatibility) we fall back to the reference implementation.
// ---------------------------------------------------------------------------

/// Preferred number of contributions added with `wrapping_add` between full
/// mod-q reductions on the u64 HVC ring.  The runtime chunk size is capped by
/// `lazy_reduce_interval(q)` so future larger HVC moduli cannot overflow this
/// lazy accumulator.
const REDUCE_INTERVAL: usize = 1024;

#[inline]
fn lazy_reduce_interval(q: u64) -> usize {
    let max_terms = (u64::MAX as u128 / ((q - 1) as u128)) as usize;
    REDUCE_INTERVAL.min(max_terms.max(1))
}

#[inline(always)]
fn reduce_slot_u64(x: u64, q: u64) -> u64 {
    x % q
}

/// Conditional-subtract reduction for inputs already known to be in `[0, 2q)`.
///
/// Caller must ensure `q < 2^63` so that the two-operand `a + b` sum
/// (with both in `[0, q)`) cannot overflow u64.  All shipped HVC moduli
/// satisfy this with wide margin (`D256_K4` has `q ≈ 2^53`); the
/// `debug_assert!` trips a future profile that pushes `q` toward `2^63`.
#[inline(always)]
fn cond_sub_q(x: u64, q: u64) -> u64 {
    debug_assert!(q < (1u64 << 63), "cond_sub_q: q must fit in i64");
    if x >= q {
        x - q
    } else {
        x
    }
}

/// Flat NTT-domain accumulator mirroring an `HvcOpening`'s segment layout.
///
/// Each `Vec<u64>` inside `path_acc[i]` / `sib_acc[i]` / `u_acc` is a single
/// length-d NTT buffer for one polynomial slot of the corresponding
/// opening segment.
#[derive(Clone)]
struct NttAcc {
    path_acc: Vec<Vec<Vec<u64>>>, // [tau][n_polys_in_segment][d]
    sib_acc: Vec<Vec<Vec<u64>>>,
    u_acc: Vec<Vec<u64>>, // [n_polys_in_u][d]
}

impl NttAcc {
    fn zero(ref_open: &HvcOpening, d: usize) -> Self {
        let path_acc = ref_open
            .path_labels
            .iter()
            .map(|label| {
                let n = label.len() / d;
                (0..n).map(|_| vec![0u64; d]).collect()
            })
            .collect();
        let sib_acc = ref_open
            .sibling_labels
            .iter()
            .map(|label| {
                let n = label.len() / d;
                (0..n).map(|_| vec![0u64; d]).collect()
            })
            .collect();
        let n_u = ref_open.u.len() / d;
        let u_acc = (0..n_u).map(|_| vec![0u64; d]).collect();
        NttAcc {
            path_acc,
            sib_acc,
            u_acc,
        }
    }

    /// Reduce every slot mod q (used at the end of each accumulation chunk).
    fn reduce_mod_q(&mut self, q: u64) {
        for label in self.path_acc.iter_mut() {
            for poly in label.iter_mut() {
                for x in poly.iter_mut() {
                    *x = reduce_slot_u64(*x, q);
                }
            }
        }
        for label in self.sib_acc.iter_mut() {
            for poly in label.iter_mut() {
                for x in poly.iter_mut() {
                    *x = reduce_slot_u64(*x, q);
                }
            }
        }
        for poly in self.u_acc.iter_mut() {
            for x in poly.iter_mut() {
                *x = reduce_slot_u64(*x, q);
            }
        }
    }

    /// Merge `other` into `self`, both already reduced mod q.
    /// Result stays in `[0, q)` via per-slot conditional subtract.
    fn add_mod_q(&mut self, other: &NttAcc, q: u64) {
        for (la, lb) in self.path_acc.iter_mut().zip(other.path_acc.iter()) {
            for (pa, pb) in la.iter_mut().zip(lb.iter()) {
                for (a, &b) in pa.iter_mut().zip(pb.iter()) {
                    *a = cond_sub_q(*a + b, q);
                }
            }
        }
        for (la, lb) in self.sib_acc.iter_mut().zip(other.sib_acc.iter()) {
            for (pa, pb) in la.iter_mut().zip(lb.iter()) {
                for (a, &b) in pa.iter_mut().zip(pb.iter()) {
                    *a = cond_sub_q(*a + b, q);
                }
            }
        }
        for (pa, pb) in self.u_acc.iter_mut().zip(other.u_acc.iter()) {
            for (a, &b) in pa.iter_mut().zip(pb.iter()) {
                *a = cond_sub_q(*a + b, q);
            }
        }
    }

    /// Add `w · opening` into `self` (NTT-domain MAC, `wrapping_add`).
    /// Caller is responsible for invoking `reduce_mod_q` at most every
    /// `REDUCE_INTERVAL` such adds to prevent u64 overflow.
    fn mac(&mut self, w_hat: &[u64], opening: &HvcOpening, rp: &RingParams64, p_hat: &mut [u64]) {
        let d = rp.d;
        let q = rp.q;
        let q_inv = rp.q_inv;
        debug_assert_eq!(p_hat.len(), d);

        let mut mac_one = |target: &mut [u64], coeffs: &[i64]| {
            debug_assert_eq!(target.len(), d);
            debug_assert_eq!(coeffs.len(), d);
            poly_to_ntt_buf_u64(coeffs, rp, p_hat);
            for k in 0..d {
                let prod = mont_mul_u64(w_hat[k], p_hat[k], q, q_inv);
                target[k] = target[k].wrapping_add(prod);
            }
        };

        for (acc_label, sig_label) in self.path_acc.iter_mut().zip(opening.path_labels.iter()) {
            let n = sig_label.len() / d;
            for p in 0..n {
                mac_one(&mut acc_label[p], &sig_label[p * d..(p + 1) * d]);
            }
        }
        for (acc_label, sig_label) in self.sib_acc.iter_mut().zip(opening.sibling_labels.iter()) {
            let n = sig_label.len() / d;
            for p in 0..n {
                mac_one(&mut acc_label[p], &sig_label[p * d..(p + 1) * d]);
            }
        }
        let n_u = opening.u.len() / d;
        for p in 0..n_u {
            mac_one(&mut self.u_acc[p], &opening.u[p * d..(p + 1) * d]);
        }
    }

    /// Finalize: inverse NTT each accumulator and emit an `HvcOpening` in
    /// signed canonical form `(-q/2, q/2]`.
    fn finalize_u64(mut self, rp: &RingParams64) -> HvcOpening {
        let d = rp.d;
        let q = rp.q;
        let q_inv = rp.q_inv;
        let inv_d = rp.inv_d_mont;
        let half = (q / 2) as i64;
        let q_i64 = q as i64;

        let inv_one = |buf: &mut Vec<u64>| -> Vec<i64> {
            ntt_inverse_u64(buf, rp.zetas, q, q_inv, inv_d);
            buf.iter()
                .map(|&x| {
                    let v = rp.from_mont(x) as i64;
                    if v > half {
                        v - q_i64
                    } else {
                        v
                    }
                })
                .collect()
        };

        let path_labels: Vec<Vec<i64>> = self
            .path_acc
            .iter_mut()
            .map(|label| {
                let n = label.len();
                let mut flat = Vec::with_capacity(n * d);
                for poly in label.iter_mut() {
                    flat.extend(inv_one(poly));
                }
                flat
            })
            .collect();
        let sibling_labels: Vec<Vec<i64>> = self
            .sib_acc
            .iter_mut()
            .map(|label| {
                let n = label.len();
                let mut flat = Vec::with_capacity(n * d);
                for poly in label.iter_mut() {
                    flat.extend(inv_one(poly));
                }
                flat
            })
            .collect();
        let mut u = Vec::with_capacity(self.u_acc.len() * d);
        for poly in self.u_acc.iter_mut() {
            u.extend(inv_one(poly));
        }
        HvcOpening {
            path_labels,
            sibling_labels,
            u,
        }
    }
}

/// Aggregate N openings under N ternary scalars in one pass, accumulating
/// in the HVC ring's NTT domain.  Parallel across signers via rayon: each
/// chunk of up to `REDUCE_INTERVAL` signers builds a local NTT-domain
/// accumulator (bounded by `chunk_size · (q-1) ≤ u64::MAX`), reduces mod q,
/// and the per-chunk accumulators are tree-reduced with per-slot
/// conditional-subtract to keep the invariant `slot ∈ [0, q)`.  A single
/// inverse NTT per output poly at the end produces the final `HvcOpening`.
///
/// Bit-identical to `Σ scale_opening_any(wᵢ, sigᵢ.opening) (coefficient
/// domain)` — see commit message / paper appendix for the algebraic
/// argument (the NTT is R_q-linear).
fn aggregate_openings_u64(
    ws: &[Vec<i64>],
    openings: &[&HvcOpening],
    rp: &RingParams64,
) -> HvcOpening {
    assert_eq!(ws.len(), openings.len(), "ws/openings length mismatch");
    assert!(!openings.is_empty(), "aggregate_openings: empty input");

    let d = rp.d;
    let q = rp.q;
    let ref_open = openings[0];

    // Per-chunk worker: accumulate up to `REDUCE_INTERVAL` contributions, then
    // reduce mod q so the result is a valid `[0, q)` accumulator.
    let chunk_size = lazy_reduce_interval(q);

    let acc = ws
        .par_chunks(chunk_size)
        .zip(openings.par_chunks(chunk_size))
        .map(|(w_chunk, opening_chunk)| {
            let mut local = NttAcc::zero(ref_open, d);
            let mut w_hat = vec![0u64; d];
            let mut p_hat = vec![0u64; d];
            for (w, opening) in w_chunk.iter().zip(opening_chunk.iter()) {
                // Forward NTT of w_i (once per signer, reused for all opening segments).
                for (dst, &x) in w_hat.iter_mut().zip(w.iter()) {
                    *dst = rp.to_mont(rp.reduce_i64(x));
                }
                crate::ntt::ntt_forward_u64(&mut w_hat, rp.zetas, q, rp.q_inv);
                local.mac(&w_hat, opening, rp, &mut p_hat);
            }
            local.reduce_mod_q(q);
            local
        })
        .reduce(
            || NttAcc::zero(ref_open, d),
            |mut a, b| {
                a.add_mod_q(&b, q);
                a
            },
        );

    // One inverse NTT per output poly.  Accumulator already in [0, q).
    acc.finalize_u64(rp)
}

/// Public NTT-domain opening aggregator.  For the u64 HVC ring (the only
/// shipped backend) uses `aggregate_openings_u64`; for the u32 ring falls
/// back to the per-signer reference path (kept compile-checked for future
/// cells with natively NTT-friendly q < 2^32).
pub fn aggregate_openings_any(
    ws: &[Vec<i64>],
    openings: &[&HvcOpening],
    profile: &Profile,
) -> HvcOpening {
    match &profile.hvc_ring {
        HvcRing::U64(rp) => aggregate_openings_u64(ws, openings, rp),
        HvcRing::U32(_) => {
            // Reference fallback (u32 ring is not exercised by the shipped profile).
            let mut acc: Option<HvcOpening> = None;
            for (w, opening) in ws.iter().zip(openings.iter()) {
                let scaled = scale_opening_any(w, opening, profile);
                acc = Some(match acc {
                    None => scaled,
                    Some(prev) => add_openings(&prev, &scaled),
                });
            }
            acc.expect("aggregate_openings: empty input")
        }
    }
}

// ---------------------------------------------------------------------------
// Babai encoding helpers for compressed HVC openings
// ---------------------------------------------------------------------------

/// Balanced base-B decomposition over ZZ (no modular reduction).
fn decompose_coeff_zz(c: i64, kappa: usize, profile: &Profile) -> Vec<i64> {
    let eta = profile.eta;
    let b_val = 2 * eta + 1;
    let mut c = c;
    let mut digits = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut r = c.rem_euclid(b_val);
        if r > eta {
            r -= b_val;
        }
        digits.push(r);
        c = (c - r) / b_val;
    }
    digits
}

/// Decompose polynomial over ZZ into kappa digit polys. Returns flat (kappa * d).
fn dec_poly_zz(a: &[i64], kappa: usize, profile: &Profile) -> Vec<i64> {
    let d = profile.d;
    let mut result = vec![0i64; kappa * d];
    for i in 0..d {
        let digits = decompose_coeff_zz(a[i], kappa, profile);
        for (ki, &dv) in digits.iter().enumerate() {
            result[ki * d + i] = dv;
        }
    }
    result
}

/// Babai-encode one omega-block of kappa digit polys (profile-aware real impl).
///
/// Returns `(a_star [d], alphas [(kappa-1)*d])`.
pub fn babai_encode_block_with_profile(
    digits: &[i64],
    hint_block: &[i64],
    profile: &Profile,
) -> (Vec<i64>, Vec<i64>) {
    let d = profile.d;
    let q = profile.q_hvc() as i64;
    let kappa = profile.kappa;
    let b_val = 2 * profile.eta + 1;

    let proj_zz = proj_poly(digits, kappa, profile);

    let hint_signed: Vec<i64> = hint_block
        .iter()
        .map(|&x| if x > q / 2 { x - q } else { x })
        .collect();

    let a_star: Vec<i64> = proj_zz
        .iter()
        .zip(hint_signed.iter())
        .map(|(&p, &h)| (p - h) / q)
        .collect();

    let w = dec_poly_zz(&proj_zz, kappa, profile);

    let mut alphas = vec![0i64; (kappa - 1) * d];

    // alpha[0] = -(digits[0] - w[0]) / B
    for i in 0..d {
        let residual = digits[i] - w[i];
        alphas[i] = -(residual / b_val);
    }

    // alpha[k] = -(digits[k] - w[k] - alpha[k-1]) / B
    for k in 1..kappa - 1 {
        for i in 0..d {
            let residual = digits[k * d + i] - w[k * d + i] - alphas[(k - 1) * d + i];
            alphas[k * d + i] = -(residual / b_val);
        }
    }

    (a_star, alphas)
}


/// Babai-decode one omega-block back to kappa digit polys (profile-aware real impl).
///
/// Returns flat `(kappa * d)`.
pub fn babai_decode_block_with_profile(
    a_star: &[i64],
    alphas: &[i64],
    hint_block: &[i64],
    profile: &Profile,
) -> Vec<i64> {
    let d = profile.d;
    let q = profile.q_hvc() as i64;
    let kappa = profile.kappa;
    let b_val = 2 * profile.eta + 1;

    let hint_signed: Vec<i64> = hint_block
        .iter()
        .map(|&x| if x > q / 2 { x - q } else { x })
        .collect();

    let h_zz: Vec<i64> = hint_signed
        .iter()
        .zip(a_star.iter())
        .map(|(&h, &a)| h + q * a)
        .collect();

    let w = dec_poly_zz(&h_zz, kappa, profile);

    // Derive alpha_last from carry: carry = (h_zz - proj(w)) / B^kappa
    let w_sum = proj_poly(&w, kappa, profile);
    let b_pow_kappa = b_val.pow(kappa as u32);
    let alpha_last: Vec<i64> = h_zz
        .iter()
        .zip(w_sum.iter())
        .map(|(&h, &ws)| -((h - ws) / b_pow_kappa))
        .collect();

    let mut digits = vec![0i64; kappa * d];

    // digits[0] = w[0] - B * alphas[0]
    for i in 0..d {
        digits[i] = w[i] - b_val * alphas[i];
    }

    // digits[k] = w[k] + alphas[k-1] - B * alphas[k]
    for k in 1..kappa - 1 {
        for i in 0..d {
            digits[k * d + i] = w[k * d + i] + alphas[(k - 1) * d + i] - b_val * alphas[k * d + i];
        }
    }

    // digits[kappa-1] = w[kappa-1] + alphas[kappa-2] - B * alpha_last
    for i in 0..d {
        digits[(kappa - 1) * d + i] =
            w[(kappa - 1) * d + i] + alphas[(kappa - 2) * d + i] - b_val * alpha_last[i];
    }

    digits
}


/// Babai-encode a full label (omega*kappa*d flat) given hint (omega*d) — profile-aware real impl.
pub fn babai_encode_label_with_profile(
    label: &[i64],
    hint: &[i64],
    profile: &Profile,
) -> Vec<(Vec<i64>, Vec<i64>)> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let mut encoded = Vec::with_capacity(omega);
    for r in 0..omega {
        let block = &label[r * kappa * d..(r + 1) * kappa * d];
        let hint_block = &hint[r * d..(r + 1) * d];
        encoded.push(babai_encode_block_with_profile(block, hint_block, profile));
    }
    encoded
}


/// Decode a full label from Babai data and hint (omega*d) — profile-aware real impl.
pub fn babai_decode_label_with_profile(
    encoded: &[(Vec<i64>, Vec<i64>)],
    hint: &[i64],
    profile: &Profile,
) -> Vec<i64> {
    let d = profile.d;
    let omega = profile.omega;
    let kappa = profile.kappa;
    let mut label = vec![0i64; omega * kappa * d];
    for r in 0..omega {
        let (a_star, alphas) = &encoded[r];
        let hint_block = &hint[r * d..(r + 1) * d];
        let block = babai_decode_block_with_profile(a_star, alphas, hint_block, profile);
        label[r * kappa * d..(r + 1) * kappa * d].copy_from_slice(&block);
    }
    label
}


/// Reconstruct individual-sig path labels from sibling labels and u.
///
/// Path labels are exact decompositions of the hash-chain hints,
/// so no Babai data is needed.
pub fn reconstruct_path_labels_ind(
    pp: &HvcPp,
    t: usize,
    sibling_labels: &[Vec<i64>],
    u: &[i64],
) -> Vec<Vec<i64>> {
    let tau = pp.tau;
    let omega = pp.profile.omega;
    let kappa = pp.profile.kappa;
    let kappa_prime = pp.profile.kappa_prime;
    let rho = pp.profile.k;
    let nu = pp.profile.n;
    let q_hvc = pp.profile.q_hvc();

    let mut hint = hvc_mat_vec_prentt(&pp.b_mat_ntt, u, omega, rho * nu * kappa_prime, pp.profile);

    let mut path_labels = vec![vec![]; tau];
    for j in (1..=tau).rev() {
        path_labels[j - 1] = dec_vec_with_profile(&hint, omega, q_hvc, kappa, pp.profile);

        let s_j = &sibling_labels[j - 1];
        let bit = (t >> (tau - j)) & 1;
        let (ap, as_) = if bit == 0 {
            (&pp.a0_ntt, &pp.a1_ntt)
        } else {
            (&pp.a1_ntt, &pp.a0_ntt)
        };

        hint = hvc_mat_vec_prentt_pair(
            ap,
            &path_labels[j - 1],
            as_,
            s_j,
            omega,
            omega * kappa,
            pp.profile,
        );
    }

    path_labels
}

/// Reconstruct aggregated-sig path labels from Babai data.
pub fn reconstruct_path_labels_agg(
    pp: &HvcPp,
    t: usize,
    path_encoded: &[Vec<(Vec<i64>, Vec<i64>)>],
    sibling_labels: &[Vec<i64>],
    u: &[i64],
) -> Vec<Vec<i64>> {
    let tau = pp.tau;
    let omega = pp.profile.omega;
    let kappa = pp.profile.kappa;
    let kappa_prime = pp.profile.kappa_prime;
    let rho = pp.profile.k;
    let nu = pp.profile.n;

    let mut hint = hvc_mat_vec_prentt(&pp.b_mat_ntt, u, omega, rho * nu * kappa_prime, pp.profile);

    let mut path_labels = vec![vec![]; tau];
    for j in (1..=tau).rev() {
        path_labels[j - 1] =
            babai_decode_label_with_profile(&path_encoded[j - 1], &hint, pp.profile);

        let s_j = &sibling_labels[j - 1];
        let bit = (t >> (tau - j)) & 1;
        let (ap, as_) = if bit == 0 {
            (&pp.a0_ntt, &pp.a1_ntt)
        } else {
            (&pp.a1_ntt, &pp.a0_ntt)
        };

        hint = hvc_mat_vec_prentt_pair(
            ap,
            &path_labels[j - 1],
            as_,
            s_j,
            omega,
            omega * kappa,
            pp.profile,
        );
    }

    path_labels
}

// ---------------------------------------------------------------------------
// BDS08 smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bds_smoke {
    use super::*;
    use crate::kots::{kots_pk_from_seed, kots_setup};
    use crate::profile::DEFAULT;

    /// Build a (pp, leaf_fn-source) harness usable at any tau.
    fn setup_harness(tau: usize) -> (HvcPp, Vec<Vec<i64>>) {
        let kots_a = kots_setup(&[7u8; 32], DEFAULT);
        let hvc_pp = hvc_setup_with_profile_and_tau(&[9u8; 32], DEFAULT, tau);
        let n_slots = 1usize << tau;
        let opks: Vec<Vec<i64>> = (0..n_slots)
            .map(|i| {
                let mut seed = [0u8; 32];
                seed[0] = i as u8;
                seed[1] = (i >> 8) as u8;
                kots_pk_from_seed(&kots_a, &seed, DEFAULT).0
            })
            .collect();
        (hvc_pp, opks)
    }

    fn openings_equal(a: &HvcOpening, b: &HvcOpening) -> bool {
        a.u == b.u && a.path_labels == b.path_labels && a.sibling_labels == b.sibling_labels
    }

    #[test]
    fn bds_init_matches_hvc_com_small_tau() {
        for &tau in &[2usize, 3, 4, 5] {
            let (pp, opks) = setup_harness(tau);

            let c_ref = hvc_com(&pp, |t: usize| opks[t].clone());
            let (c_bds, _state) = bds_init(&pp, |t: usize| opks[t].clone());

            assert_eq!(c_ref.0, c_bds.0, "commitment mismatch at tau={tau}");
        }
    }

    #[test]
    fn bds_opening_matches_hvc_open_every_slot() {
        for &tau in &[2usize, 3, 4, 5] {
            let (pp, opks) = setup_harness(tau);
            let n_slots = 1usize << tau;

            let (_c, mut state) = bds_init(&pp, |t: usize| opks[t].clone());

            for t in 0..n_slots {
                let ref_open = hvc_open(&pp, t, |t: usize| opks[t].clone());
                let leaf_fn = |t: usize| opks[t].clone();
                let bds_open = bds_opening(&state, &pp, t, &leaf_fn).expect("bds_opening failed");
                assert!(
                    openings_equal(&ref_open, &bds_open),
                    "bds vs hvc_open mismatch tau={tau} t={t}"
                );
                if t + 1 < n_slots {
                    bds_advance(&mut state, &pp, &leaf_fn);
                }
            }
        }
    }

    #[test]
    fn bds_state_clone_is_independent() {
        let tau = 3;
        let (pp, opks) = setup_harness(tau);

        let (_c, state0) = bds_init(&pp, |t: usize| opks[t].clone());

        // Clone, advance the copy, original must still be at phi=0.
        let mut advanced = state0.clone();
        {
            let leaf_fn = |t: usize| opks[t].clone();
            bds_advance(&mut advanced, &pp, &leaf_fn);
        }
        assert_eq!(state0.phi, 0);
        assert_eq!(advanced.phi, 1);

        // Opening the original at slot 0 must still work (and match the
        // reference opening at slot 0).
        let ref_open = hvc_open(&pp, 0, |t: usize| opks[t].clone());
        let leaf_fn = |t: usize| opks[t].clone();
        let snap_open = bds_opening(&state0, &pp, 0, &leaf_fn).expect("snap opening");
        assert!(openings_equal(&ref_open, &snap_open));
    }

    #[test]
    fn bds_opening_wrong_slot_errors() {
        let (pp, opks) = setup_harness(3);
        let (_c, state) = bds_init(&pp, |t: usize| opks[t].clone());
        // state is at phi=0; asking for slot 2 should error, not panic.
        let leaf_fn = |t: usize| opks[t].clone();
        assert!(bds_opening(&state, &pp, 2, &leaf_fn).is_err());
    }

    /// Babai encode/decode round-trips a path label exactly under the
    /// default profile.
    #[test]
    fn babai_round_trip_default_profile() {
        let tau = 3usize;
        let (pp, opks) = setup_harness(tau);
        let open = hvc_open(&pp, 2, |t: usize| opks[t].clone());
        let hint = proj_label_with_profile(&open.path_labels[0], DEFAULT);
        let enc = babai_encode_label_with_profile(&open.path_labels[0], &hint, DEFAULT);
        let dec = babai_decode_label_with_profile(&enc, &hint, DEFAULT);
        assert_eq!(dec, open.path_labels[0], "babai round-trip drift");
    }

    #[test]
    fn aggregate_openings_ntt_matches_reference_path() {
        let tau = 3usize;
        let (pp, opks) = setup_harness(tau);
        let openings: Vec<HvcOpening> = (0..4)
            .map(|slot| hvc_open(&pp, slot, |t: usize| opks[t].clone()))
            .collect();
        let opening_refs: Vec<&HvcOpening> = openings.iter().collect();

        let ws: Vec<Vec<i64>> = (0..openings.len())
            .map(|signer| {
                let mut w = vec![0i64; DEFAULT.d];
                for j in 0..DEFAULT.alpha_w {
                    let idx = (13 * signer + 29 * j) % DEFAULT.d;
                    w[idx] = if (signer + j) & 1 == 0 { 1 } else { -1 };
                }
                w
            })
            .collect();

        let fast = aggregate_openings_any(&ws, &opening_refs, DEFAULT);
        let reference = ws
            .iter()
            .zip(openings.iter())
            .map(|(w, opening)| scale_opening_any(w, opening, DEFAULT))
            .reduce(|acc, scaled| add_openings(&acc, &scaled))
            .expect("nonempty opening set");

        assert!(openings_equal(&fast, &reference));
    }
}
