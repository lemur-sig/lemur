//! Optional in-memory materialized HVC tree for O(τ) offline signing.
//!
//! The "signer state" for this path is a flat buffer of **projected** HVC
//! labels — one `R_q^ω` vector per tree node (root at index 0, leaves at the
//! bottom of the heap layout).  Storage per node:
//!
//! ```text
//!     OMEGA * D coefficients × 8 bytes (i64)
//! ```
//!
//! Tree size: `(2^(τ+1) - 1)` nodes, so the shipped τ=20 profile needs
//! about 8 GiB (8.59 GB decimal).  This is neither part of the secret key
//! nor written to disk — it is an optional in-memory accessory, kept by
//! callers (currently only the benchmark binary) that can afford the
//! memory in exchange for O(τ) per-signature latency.  Callers that want
//! disk-persistable signer state should still use `LemurStateSk` (the
//! ~134 KB BDS08 cache).
//!
//! The decomposed form of each label (`OMEGA*KAPPA*D` i64) is never
//! stored — it would be 8× larger.  Instead, `opening` re-decomposes
//! each projected label on lookup (a linear pass over `OMEGA*D`
//! coefficients per level — microseconds).

use rayon::prelude::*;

use crate::error::LemurError;
use crate::hvc::{
    dec_vec_with_profile, internal_label_with_profile_any, leaf_label_with_profile_any,
    proj_label_with_profile, HvcCom, HvcOpening, HvcPp,
};
use crate::kots::{kots_keygen, kots_sign};
use crate::lemur::{LemurPk, LemurPp, LemurSig, LemurSk};
use crate::profile::Profile;
use crate::sample::slot_seed;

/// Per-node stride in coefficients.  `omega * D`.
#[inline]
fn stride_coeffs(profile: &Profile) -> usize {
    profile.omega * profile.d
}

/// Flat heap-layout buffer of projected HVC labels, stored at i64.
///
/// Node indexing (`2^(τ+1) - 1` total nodes):
/// - `nodes[0]` is the root (depth 0)
/// - children of node `i` live at `2i + 1` and `2i + 2`
/// - parent of node `i > 0` is `(i - 1) / 2`
/// - depth `d` occupies node indices `[2^d - 1, 2^(d+1) - 1)`
/// - leaves occupy `[2^τ - 1, 2^(τ+1) - 1)`; slot `t` is at `2^τ - 1 + t`
pub struct MaterializedHvcTree {
    /// Active Lemur profile (shared with the source `HvcPp`).
    pub profile: &'static Profile,
    /// Tree depth (matches `pp.hvc_pp.tau`).
    pub tau: usize,
    /// Flat buffer of projected labels: one `omega * D` window per node.
    nodes: Vec<i64>,
}

impl MaterializedHvcTree {
    /// Number of tree nodes at depth `tau` (leaves + internal nodes + root).
    #[inline]
    pub fn node_count(tau: usize) -> usize {
        (1usize << (tau + 1)) - 1
    }

    /// Total byte size of the materialized tree at depth `tau` (profile-aware).
    #[inline]
    pub fn byte_size_with_profile(tau: usize, profile: &Profile) -> usize {
        Self::node_count(tau) * stride_coeffs(profile) * std::mem::size_of::<i64>()
    }

    /// Build the materialized tree from a leaf-label closure.  Same cost
    /// model as `hvc_com` (one O(2^τ) parallel walk) plus the extra
    /// memory writes; correctness is verified by the
    /// `tree_sign_matches_seed_path` integration test.
    pub fn build<F>(pp: &HvcPp, leaf_fn: F) -> Self
    where
        F: Fn(usize) -> Vec<i64> + Send + Sync,
    {
        let profile = pp.profile;
        let tau = pp.tau;
        let n_leaves = 1usize << tau;
        let n_nodes = Self::node_count(tau);
        let stride = stride_coeffs(profile);
        let mut nodes = vec![0i64; n_nodes * stride];

        // ---- Leaves ----
        let leaf_base = (n_leaves - 1) * stride;
        nodes[leaf_base..]
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(t, dst)| {
                let opk = leaf_fn(t);
                let leaf_dec = leaf_label_with_profile_any(&opk, &pp.b_mat_ntt, profile);
                write_projection(&leaf_dec, dst, profile);
            });

        // ---- Internal levels, τ-1 → 0 ----
        for depth in (0..tau).rev() {
            let parent_count = 1usize << depth;
            let parent_base_node = parent_count - 1;
            let child_base_node = (parent_count << 1) - 1;

            let parent_base = parent_base_node * stride;
            let child_base = child_base_node * stride;
            let child_end = child_base + 2 * parent_count * stride;

            let (front, back) = nodes.split_at_mut(child_base);
            let parent_slab = &mut front[parent_base..child_base];
            let child_slab: &[i64] = &back[..child_end - child_base];

            parent_slab
                .par_chunks_mut(stride)
                .enumerate()
                .for_each(|(i, dst)| {
                    let l_off = (2 * i) * stride;
                    let r_off = (2 * i + 1) * stride;
                    let left_dec =
                        decompose_projected_hvc(&child_slab[l_off..l_off + stride], profile);
                    let right_dec =
                        decompose_projected_hvc(&child_slab[r_off..r_off + stride], profile);
                    let parent_dec = internal_label_with_profile_any(
                        &left_dec, &right_dec, &pp.a0_ntt, &pp.a1_ntt, profile,
                    );
                    write_projection(&parent_dec, dst, profile);
                });
        }

        Self {
            profile,
            tau,
            nodes,
        }
    }

    /// HVC commitment (root of the tree) in the standard `HvcCom` form.
    pub fn commitment(&self) -> HvcCom {
        let stride = stride_coeffs(self.profile);
        let root: Vec<i64> = self.nodes[..stride].to_vec();
        HvcCom(root)
    }

    /// Build the HVC opening for slot `t` via O(τ) cache lookups.
    ///
    /// `leaf_opk` is the KOTS public key for slot `t` — needed to
    /// reconstruct the `u` component of the opening (which we never
    /// store in the tree).  Typical callers call `kots_pk_from_seed`
    /// or `kots_keygen` immediately before invoking this function.
    pub fn opening(&self, t: usize, leaf_opk: &[i64]) -> Result<HvcOpening, LemurError> {
        let profile = self.profile;
        let tau = self.tau;
        let n_leaves = 1usize << tau;
        if t >= n_leaves {
            return Err(LemurError::InvalidEncoding(format!(
                "slot {t} out of range [0, {}]",
                n_leaves - 1
            )));
        }

        let u = dec_vec_with_profile(
            leaf_opk,
            profile.k * profile.n,
            profile.q_kots(),
            profile.kappa_prime,
            profile,
        );

        let mut path_labels: Vec<Vec<i64>> = vec![Vec::new(); tau];
        let mut sibling_labels: Vec<Vec<i64>> = vec![Vec::new(); tau];

        let mut idx = (n_leaves - 1) + t;
        for j in (1..=tau).rev() {
            path_labels[j - 1] = self.decompose_node(idx);
            sibling_labels[j - 1] = self.decompose_node(sibling_of(idx));
            idx = parent_of(idx);
        }

        Ok(HvcOpening {
            path_labels,
            sibling_labels,
            u,
        })
    }

    fn decompose_node(&self, node_idx: usize) -> Vec<i64> {
        let stride = stride_coeffs(self.profile);
        let off = node_idx * stride;
        let window = &self.nodes[off..off + stride];
        decompose_projected_hvc(window, self.profile)
    }
}

/// Decomposition from a stored projected label to the full
/// `omega * kappa * D` decomposed form.
fn decompose_projected_hvc(projected: &[i64], profile: &Profile) -> Vec<i64> {
    dec_vec_with_profile(
        projected,
        profile.omega,
        profile.q_hvc(),
        profile.kappa,
        profile,
    )
}

/// Project a decomposed label back to `R_q^omega`.
fn write_projection(decomposed_label: &[i64], dst: &mut [i64], profile: &Profile) {
    let proj = proj_label_with_profile(decomposed_label, profile);
    for (d, &s) in dst.iter_mut().zip(proj.iter()) {
        *d = s;
    }
}

#[inline]
fn sibling_of(i: usize) -> usize {
    if i & 1 == 1 {
        i + 1
    } else {
        i - 1
    }
}

#[inline]
fn parent_of(i: usize) -> usize {
    (i - 1) / 2
}

// ---------------------------------------------------------------------------
// Top-level Lemur sign wrapper using a pre-built tree
// ---------------------------------------------------------------------------

/// Sign at slot `t` using a pre-materialized HVC tree.
///
/// Cost: one KOTS sign + O(τ) flat-array lookups and decompositions —
/// essentially equivalent to a Chipmunk-style O(1) stored-tree sign at
/// per-level constant work.  Intended for benchmark comparisons; the
/// normal API path for production use is `lemur_sign_stateful_mut`,
/// which has ~134 KB state instead of ~8 GiB.
pub fn lemur_sign_tree(
    pp: &LemurPp,
    sk: &LemurSk,
    tree: &MaterializedHvcTree,
    t: usize,
    msg: &[u8],
) -> Result<LemurSig, LemurError> {
    if tree.tau != pp.hvc_pp.tau {
        return Err(LemurError::InvalidEncoding(format!(
            "materialized tree tau={} does not match pp tau={}",
            tree.tau, pp.hvc_pp.tau
        )));
    }
    let n_slots = 1usize << pp.hvc_pp.tau;
    if t >= n_slots {
        return Err(LemurError::InvalidEncoding(format!(
            "slot {t} out of range [0, {}]",
            n_slots - 1
        )));
    }

    let profile = pp.profile;
    let ss = slot_seed(&sk.master_seed, t);
    let (osk, opk) = kots_keygen(&pp.kots_a, &ss, profile);
    let z = kots_sign(&osk, msg, profile);

    let opening = tree.opening(t, &opk.0)?;
    Ok(LemurSig { z, opening })
}

/// Convenience: derive the `LemurPk` that the tree commits to.
pub fn tree_public_key(tree: &MaterializedHvcTree) -> LemurPk {
    LemurPk(tree.commitment())
}
