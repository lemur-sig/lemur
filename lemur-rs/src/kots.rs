//! KOTS: Key-Homomorphic One-Time Signature.
//!
//! The shipped parameter set satisfies the proof condition `q' ≡ 17 (mod 32)`,
//! which forbids a length-d negacyclic NTT in `R_{q'}`.  Multiplication
//! is therefore routed through the CRT-via-aux-primes backend in
//! [`crate::aux_ntt`]: two 48-bit auxiliary NTT-friendly primes whose
//! product is large enough to recover the exact signed product before
//! reducing mod q'.
//!
//! The signing path is constant-time.

use sha3::digest::XofReader;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::aux_ntt::CrtBackend;
use crate::error::LemurError;
use crate::profile::Profile;
use crate::sample::{
    kots_hash_xof, kots_keygen_xof, kots_setup_xof, xof_gauss_poly_ctx_into,
    xof_gauss_poly_with_profile_fill, xof_ternary_poly, xof_uniform_poly, GaussCtx,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// KOTS public matrix A2.
///
/// Shape: `(m-n) x n x d` coefficient-domain, flat `Vec<i64>` in `[0, q')`.
/// The pre-NTT'd companion stores, per polynomial, its forward NTT under
/// each of the two auxiliary primes — so a matmul against `A2` costs only
/// one forward NTT pair per input operand plus one inverse NTT pair and a
/// CRT combine per output polynomial (vs three per output for naive `mul`).
///
/// `ntt_p1`, `ntt_p2` are flat `(m-n) * n * d` `u64` buffers; entry `(i, j)`
/// starts at offset `(i * n + j) * d` and runs for `d` words.
#[derive(Clone)]
pub struct KotsA {
    pub coeffs: Vec<i64>,
    pub ntt_p1: Vec<u64>,
    pub ntt_p2: Vec<u64>,
}

/// KOTS secret key S: shape (k x m x d), flat i64 array.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct KotsSk(pub Vec<i64>);

/// KOTS public key T: shape (k x n x d), flat i64 array.
#[derive(Clone)]
pub struct KotsPk(pub Vec<i64>);

/// KOTS signature Z: shape (ell x m x d), flat i64 array.
#[derive(Clone)]
pub struct KotsSig(pub Vec<i64>);

/// Maximum absolute coefficient of a polynomial array.
pub fn inf_norm(v: &[i64]) -> i64 {
    v.iter().map(|&x| x.abs()).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the pre-NTT caches for an `A2` matrix already in coefficient form.
fn build_a2_ntt_cache(
    coeffs: &[i64],
    a2_rows: usize,
    n: usize,
    backend: &CrtBackend,
) -> (Vec<u64>, Vec<u64>) {
    let d = backend.d();
    let mut p1 = vec![0u64; a2_rows * n * d];
    let mut p2 = vec![0u64; a2_rows * n * d];
    for i in 0..a2_rows {
        for j in 0..n {
            let off = (i * n + j) * d;
            backend.forward_into_i64(
                &coeffs[off..off + d],
                &mut p1[off..off + d],
                &mut p2[off..off + d],
            );
        }
    }
    (p1, p2)
}

/// Multiply X (rows x m) by the structured A = [I_n; A2].
///
/// Caches X's NTT pair once per (row, left-col) pair and accumulates
/// pointwise products in each auxiliary prime's NTT domain.
fn matmul_structured_a(x: &[i64], rows: usize, a: &KotsA, profile: &Profile) -> Vec<i64> {
    let m = profile.m;
    let n = profile.n;
    let a2_rows = m - n;
    let cfg = profile.kots_crt().expect("CRT KOTS profile");
    // Each output coefficient sums `a2_rows` per-coefficient products
    // (X[:, n:][i,k] · A2[k,j] for k in 0..a2_rows) in the paired NTT
    // domain before reconstruction; size the backend for that.
    let backend = cfg.backend_for_accum(a2_rows);
    let d = cfg.d;
    let q = cfg.q as i64;

    // Forward-NTT X[:, n:] into the paired aux-prime domain once.  Layout
    // mirrors `a.ntt_p{1,2}`: (rows x a2_rows) polynomials at stride d.
    let mut xr_p1 = vec![0u64; rows * a2_rows * d];
    let mut xr_p2 = vec![0u64; rows * a2_rows * d];
    for i in 0..rows {
        for k in 0..a2_rows {
            let src_off = (i * m + n + k) * d;
            let dst_off = (i * a2_rows + k) * d;
            backend.forward_into_i64(
                &x[src_off..src_off + d],
                &mut xr_p1[dst_off..dst_off + d],
                &mut xr_p2[dst_off..dst_off + d],
            );
        }
    }

    // Accumulate T_right[i, j] = sum_k X[:, n:][i, k] * A2[k, j] in NTT domain.
    let mut t = vec![0i64; rows * n * d];
    let mut acc_p1 = vec![0u64; d];
    let mut acc_p2 = vec![0u64; d];
    for i in 0..rows {
        for j in 0..n {
            acc_p1.iter_mut().for_each(|v| *v = 0);
            acc_p2.iter_mut().for_each(|v| *v = 0);
            for k in 0..a2_rows {
                let xr_off = (i * a2_rows + k) * d;
                let a2_off = (k * n + j) * d;
                backend.accum_mul_slices(
                    &mut acc_p1,
                    &mut acc_p2,
                    &xr_p1[xr_off..xr_off + d],
                    &xr_p2[xr_off..xr_off + d],
                    &a.ntt_p1[a2_off..a2_off + d],
                    &a.ntt_p2[a2_off..a2_off + d],
                );
            }
            let prod = backend.finalize_accum_slices(&mut acc_p1, &mut acc_p2);
            // T[i, j] = X[:, :n][i, j] + T_right[i, j] (mod q').
            for l in 0..d {
                let x_val = x[(i * m + j) * d + l];
                let p_val = prod[l] as i64;
                t[(i * n + j) * d + l] = (x_val + p_val).rem_euclid(q);
            }
        }
    }
    t
}

/// Multiply the structured KOTS hash matrix `H = [I_ell | H']` by `B`.
///
/// This is the only CRT matrix multiply needed on the sign / verify paths.
/// The identity block contributes `B[i, j]` directly in coefficient form,
/// so we only forward-NTT and multiply the hashed columns `ell..k`.
fn mat_mul_h(
    h: &[i64],
    b: &[i64],
    ell: usize,
    k: usize,
    t: usize,
    backend: &CrtBackend,
) -> Vec<i64> {
    let d = backend.d();
    let q = backend.q() as i64;
    let hashed_cols = k - ell;

    debug_assert_eq!(h.len(), ell * k * d);
    debug_assert_eq!(b.len(), k * t * d);

    // Pre-NTT only H' and the matching lower rows of B.  The identity
    // columns H[i,i] = 1 are copied directly into the coefficient-domain
    // output below.
    let mut h_p1 = vec![0u64; ell * hashed_cols * d];
    let mut h_p2 = vec![0u64; ell * hashed_cols * d];
    for i in 0..ell {
        for l in 0..hashed_cols {
            let h_col = ell + l;
            let src_off = (i * k + h_col) * d;
            let dst_off = (i * hashed_cols + l) * d;
            backend.forward_into_i64(
                &h[src_off..src_off + d],
                &mut h_p1[dst_off..dst_off + d],
                &mut h_p2[dst_off..dst_off + d],
            );
        }
    }

    let mut b_p1 = vec![0u64; hashed_cols * t * d];
    let mut b_p2 = vec![0u64; hashed_cols * t * d];
    for l in 0..hashed_cols {
        let b_row = ell + l;
        for j in 0..t {
            let src_off = (b_row * t + j) * d;
            let dst_off = (l * t + j) * d;
            backend.forward_into_i64(
                &b[src_off..src_off + d],
                &mut b_p1[dst_off..dst_off + d],
                &mut b_p2[dst_off..dst_off + d],
            );
        }
    }

    let mut out = vec![0i64; ell * t * d];
    let mut acc_p1 = vec![0u64; d];
    let mut acc_p2 = vec![0u64; d];
    for i in 0..ell {
        for j in 0..t {
            acc_p1.iter_mut().for_each(|v| *v = 0);
            acc_p2.iter_mut().for_each(|v| *v = 0);
            for l in 0..hashed_cols {
                let h_off = (i * hashed_cols + l) * d;
                let b_off = (l * t + j) * d;
                backend.accum_mul_slices(
                    &mut acc_p1,
                    &mut acc_p2,
                    &h_p1[h_off..h_off + d],
                    &h_p2[h_off..h_off + d],
                    &b_p1[b_off..b_off + d],
                    &b_p2[b_off..b_off + d],
                );
            }

            let out_off = (i * t + j) * d;
            let identity_off = (i * t + j) * d;
            if hashed_cols == 0 {
                for l in 0..d {
                    out[out_off + l] = b[identity_off + l].rem_euclid(q);
                }
                continue;
            }

            let prod = backend.finalize_accum_slices(&mut acc_p1, &mut acc_p2);
            for l in 0..d {
                out[out_off + l] = (b[identity_off + l] + prod[l] as i64).rem_euclid(q);
            }
        }
    }
    out
}

/// Build H = [I_ell | H'] in R^{ell x k}, stride = profile.d.
fn build_h(mu: &[u8], profile: &Profile) -> Vec<i64> {
    let ell = profile.ell;
    let k = profile.k;
    let alpha_h = profile.alpha_h;
    let d = profile.d;

    let mut h = vec![0i64; ell * k * d];
    for i in 0..ell {
        h[(i * k + i) * d] = 1;
    }
    for i in 0..ell {
        for j in 0..(k - ell) {
            let index = (i * (k - ell) + j) as u32;
            let mut xof = kots_hash_xof(mu, index);
            let poly = xof_ternary_poly(&mut xof as &mut dyn XofReader, alpha_h, d);
            let col = ell + j;
            h[(i * k + col) * d..(i * k + col + 1) * d].copy_from_slice(&poly);
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Setup / KeyGen / Sign / Verify
// ---------------------------------------------------------------------------

/// KOTS Setup: expand A2 from seed and pre-NTT it under both aux primes.
pub fn kots_setup(seed: &[u8], profile: &'static Profile) -> KotsA {
    let cfg = profile.kots_crt().expect("CRT KOTS profile");
    let backend = cfg.backend();
    let d = cfg.d;
    let q = cfg.q;
    let m = profile.m;
    let n = profile.n;
    let a2_rows = m - n;

    let mut coeffs = vec![0i64; a2_rows * n * d];
    for i in 0..a2_rows {
        for j in 0..n {
            let mut xof = kots_setup_xof(seed, i, j);
            let poly = xof_uniform_poly(&mut xof as &mut dyn XofReader, q, d);
            coeffs[(i * n + j) * d..(i * n + j + 1) * d].copy_from_slice(&poly);
        }
    }
    let (ntt_p1, ntt_p2) = build_a2_ntt_cache(&coeffs, a2_rows, n, &backend);
    KotsA {
        coeffs,
        ntt_p1,
        ntt_p2,
    }
}

/// KOTS KeyGen using the profile's default Gaussian sampler.
pub fn kots_keygen(a: &KotsA, seed: &[u8], profile: &'static Profile) -> (KotsSk, KotsPk) {
    let d = profile.d;
    let k = profile.k;
    let m = profile.m;

    let mut s = vec![0i64; k * m * d];
    for i in 0..k {
        for j in 0..m {
            let mut xof = kots_keygen_xof(seed, i, j);
            xof_gauss_poly_with_profile_fill(
                &mut xof as &mut dyn XofReader,
                profile,
                &mut s[(i * m + j) * d..(i * m + j + 1) * d],
            );
        }
    }
    let t = matmul_structured_a(&s, k, a, profile);
    (KotsSk(s), KotsPk(t))
}

/// KOTS KeyGen with a caller-supplied Gaussian sampler context (used by the
/// bench harness to swap CDT widths / sigmas without rebuilding the baked
/// table).
pub fn kots_keygen_ctx(
    a: &KotsA,
    seed: &[u8],
    profile: &'static Profile,
    ctx: &GaussCtx,
) -> (KotsSk, KotsPk) {
    let d = profile.d;
    let k = profile.k;
    let m = profile.m;

    let mut s = vec![0i64; k * m * d];
    for i in 0..k {
        for j in 0..m {
            let mut xof = kots_keygen_xof(seed, i, j);
            xof_gauss_poly_ctx_into(
                &mut xof as &mut dyn XofReader,
                ctx,
                &mut s[(i * m + j) * d..(i * m + j + 1) * d],
            );
        }
    }
    let t = matmul_structured_a(&s, k, a, profile);
    (KotsSk(s), KotsPk(t))
}

/// KOTS pk-only KeyGen — avoids returning the full secret key when only
/// the public key is needed (used by HVC leaf expansion).
pub fn kots_pk_from_seed(a: &KotsA, seed: &[u8], profile: &'static Profile) -> KotsPk {
    let d = profile.d;
    let k = profile.k;
    let m = profile.m;

    let mut s = vec![0i64; k * m * d];
    for i in 0..k {
        for j in 0..m {
            let mut xof = kots_keygen_xof(seed, i, j);
            xof_gauss_poly_with_profile_fill(
                &mut xof as &mut dyn XofReader,
                profile,
                &mut s[(i * m + j) * d..(i * m + j + 1) * d],
            );
        }
    }
    let t = matmul_structured_a(&s, k, a, profile);
    KotsPk(t)
}

/// KOTS Sign: Z = H * S with signed lift.
pub fn kots_sign(sk: &KotsSk, mu: &[u8], profile: &'static Profile) -> KotsSig {
    let cfg = profile.kots_crt().expect("CRT KOTS profile");
    let ell = profile.ell;
    let k = profile.k;
    let m = profile.m;
    // H = [I_ell | H']; only the hashed `k - ell` columns require ring
    // products.  Size the backend for that lazy accumulation depth.
    let backend = cfg.backend_for_accum(k - ell);
    let q = cfg.q as i64;
    let h = build_h(mu, profile);

    let z_unsigned = mat_mul_h(&h, &sk.0, ell, k, m, &backend);
    let half = q / 2;
    let z: Vec<i64> = z_unsigned
        .into_iter()
        .map(|x| if x > half { x - q } else { x })
        .collect();
    KotsSig(z)
}

/// KOTS Verify: ||Z||_inf <= beta AND Z*A == H*T (mod q').
pub fn kots_vrfy(
    a: &KotsA,
    pk: &KotsPk,
    mu: &[u8],
    sig: &KotsSig,
    beta: i64,
    profile: &'static Profile,
) -> Result<(), LemurError> {
    if inf_norm(&sig.0) > beta {
        return Err(LemurError::VerifyFailed);
    }
    let cfg = profile.kots_crt().expect("CRT KOTS profile");
    let ell = profile.ell;
    let k = profile.k;
    let n = profile.n;
    let backend = cfg.backend_for_accum(k - ell);

    let h = build_h(mu, profile);
    let za = matmul_structured_a(&sig.0, ell, a, profile);
    let ht = mat_mul_h(&h, &pk.0, ell, k, n, &backend);
    if za == ht {
        Ok(())
    } else {
        Err(LemurError::VerifyFailed)
    }
}

/// Individual-verify bound β_z.
pub fn kots_ivrfy(
    a: &KotsA,
    pk: &KotsPk,
    mu: &[u8],
    sig: &KotsSig,
    profile: &'static Profile,
) -> Result<(), LemurError> {
    kots_vrfy(a, pk, mu, sig, profile.beta_z, profile)
}

/// Strong-verify bound β_σ.
pub fn kots_svrfy(
    a: &KotsA,
    pk: &KotsPk,
    mu: &[u8],
    sig: &KotsSig,
    profile: &'static Profile,
) -> Result<(), LemurError> {
    kots_vrfy(a, pk, mu, sig, profile.beta_sigma, profile)
}

/// Weak-verify bound 2·β_σ.
pub fn kots_wvrfy(
    a: &KotsA,
    pk: &KotsPk,
    mu: &[u8],
    sig: &KotsSig,
    profile: &'static Profile,
) -> Result<(), LemurError> {
    kots_vrfy(a, pk, mu, sig, 2 * profile.beta_sigma, profile)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::D256_K4;

    fn run_self_consistency(profile: &'static Profile) {
        let seed_setup = [0x42u8; 32];
        let seed_key = [0x43u8; 32];
        let mu: &[u8] = b"kots self-consistency";

        let a = kots_setup(&seed_setup, profile);
        assert_eq!(
            a.coeffs.len(),
            (profile.m - profile.n) * profile.n * profile.d
        );
        assert_eq!(a.ntt_p1.len(), a.coeffs.len());
        assert_eq!(a.ntt_p2.len(), a.coeffs.len());
        let q_kots = profile.q_kots() as i64;
        assert!(a.coeffs.iter().all(|&x| (0..q_kots).contains(&x)));

        let (sk, pk) = kots_keygen(&a, &seed_key, profile);
        let pk_only = kots_pk_from_seed(&a, &seed_key, profile);
        assert_eq!(pk.0, pk_only.0, "pk_from_seed drift vs keygen");

        assert_eq!(sk.0.len(), profile.k * profile.m * profile.d);
        assert_eq!(pk.0.len(), profile.k * profile.n * profile.d);

        let sig = kots_sign(&sk, mu, profile);
        assert_eq!(sig.0.len(), profile.ell * profile.m * profile.d);
        kots_ivrfy(&a, &pk, mu, &sig, profile).expect("ivrfy");
        kots_svrfy(&a, &pk, mu, &sig, profile).expect("svrfy");
        kots_wvrfy(&a, &pk, mu, &sig, profile).expect("wvrfy");

        assert!(kots_ivrfy(&a, &pk, b"different mu", &sig, profile).is_err());
    }

    #[test]
    fn d256_k4_kots_self_consistency() {
        run_self_consistency(&D256_K4);
    }

}
