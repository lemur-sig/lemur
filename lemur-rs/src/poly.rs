//! Polynomial ring arithmetic over Z_q[X]/(X^d + 1) with NTT backing.
//!
//! Polynomials are stored as `Vec<i64>` with coefficients in [0, q).
//! Signed variants lift to [-q/2, q/2] after inverse NTT.
//!
//! # Ring dimension `d`
//!
//! `d` is a **runtime property of the ring**, carried on
//! [`RingParams`] (u32 backend) and [`RingParams64`] (u64 backend).
//! Every helper reads `rp.d` (or `rp_u64.d`) for stride and buffer
//! sizing, so the implementation is not tied to a compile-time ring degree.

use crate::ntt::{ct_reduce, mont_mul, mont_reduce, ntt_forward, ntt_inverse, to_mont};
use crate::ntt::{
    ct_reduce_u64, mont_mul_u64, mont_reduce_u64, ntt_forward_u64, ntt_inverse_u64, to_mont_u64,
};

/// Parameters for one ring `Z_q[X]/(X^d + 1)` using the **u32 Montgomery
/// backend** (`R = 2^32`).
///
/// Valid for moduli `q < 2^32`.  Not used by any shipping profile (every
/// HVC modulus exceeds 2³² and routes through the u64 stack; KOTS is not
/// natively NTT-friendly and routes through `aux_ntt::CrtBackend`).
/// Kept compile-checked for future cells with smaller HVC moduli.
#[derive(Clone)]
pub struct RingParams {
    pub q: u64,
    pub q32: u32,
    pub q_inv: u32,
    pub r2: u32,
    pub inv_d_mont: u32,
    /// Ring dimension.  Independent of the crate-level `params::D` —
    /// this is the degree `d` of the ring `Z_q[X]/(X^d + 1)` the
    /// carrier refers to (length of `zetas`).
    pub d: usize,
    /// Twiddle-factor table in Montgomery form (bit-reversed order).
    /// Length equals `self.d`.
    pub zetas: &'static [u32],
}

/// Parameters for one ring `Z_q[X]/(X^d + 1)` using the **u64 Montgomery
/// backend** (`R = 2^64`, 128-bit intermediate products).
///
/// Used by the shipped HVC ring.
/// KOTS rings are not natively NTT-friendly and route through
/// `aux_ntt::CrtBackend` instead.
#[derive(Clone)]
pub struct RingParams64 {
    pub q: u64,
    pub q_inv: u64,
    pub r2: u64,
    pub inv_d_mont: u64,
    /// Ring dimension (= length of `zetas`).
    pub d: usize,
    /// Twiddle-factor table in Montgomery form (bit-reversed order).
    pub zetas: &'static [u64],
}

impl RingParams {
    /// Reduce an i64 value to [0, q).
    #[inline(always)]
    pub fn reduce_i64(&self, x: i64) -> u32 {
        let r = x.rem_euclid(self.q as i64) as u32;
        ct_reduce(r, self.q32)
    }

    /// Convert a u32 in [0,q) to Montgomery form.
    #[inline(always)]
    pub fn to_mont(&self, x: u32) -> u32 {
        to_mont(x, self.r2, self.q32, self.q_inv)
    }

    /// Convert from Montgomery form back to normal.
    #[inline(always)]
    pub fn from_mont(&self, x: u32) -> u32 {
        mont_reduce(x as u64, self.q32, self.q_inv)
    }
}

impl RingParams64 {
    /// Reduce an i64 value to [0, q).
    #[inline(always)]
    pub fn reduce_i64(&self, x: i64) -> u64 {
        let r = x.rem_euclid(self.q as i64) as u64;
        ct_reduce_u64(r, self.q)
    }

    /// Convert a u64 in [0,q) to Montgomery form.
    #[inline(always)]
    pub fn to_mont(&self, x: u64) -> u64 {
        to_mont_u64(x, self.r2, self.q, self.q_inv)
    }

    /// Convert from Montgomery form back to normal.
    #[inline(always)]
    pub fn from_mont(&self, x: u64) -> u64 {
        mont_reduce_u64(x as u128, self.q, self.q_inv)
    }
}

// ---------------------------------------------------------------------------
// Internal NTT helpers
// ---------------------------------------------------------------------------

/// Forward-NTT a polynomial from coefficient domain to NTT domain.
pub fn poly_to_ntt(p: &[i64], rp: &RingParams) -> Vec<u32> {
    let mut f: Vec<u32> = p.iter().map(|&x| rp.to_mont(rp.reduce_i64(x))).collect();
    ntt_forward(&mut f, rp.zetas, rp.q32, rp.q_inv);
    f
}

/// Forward-NTT into a caller-provided buffer (avoids allocation).
#[inline]
pub(crate) fn poly_to_ntt_buf(p: &[i64], rp: &RingParams, buf: &mut [u32]) {
    for (dst, &x) in buf.iter_mut().zip(p.iter()) {
        *dst = rp.to_mont(rp.reduce_i64(x));
    }
    ntt_forward(buf, rp.zetas, rp.q32, rp.q_inv);
}

/// Inverse-NTT writing directly into an i64 output slice (avoids allocation).
#[inline]
pub(crate) fn ntt_to_poly_buf(f: &mut [u32], rp: &RingParams, out: &mut [i64]) {
    ntt_inverse(f, rp.zetas, rp.q32, rp.q_inv, rp.inv_d_mont);
    for (dst, &x) in out.iter_mut().zip(f.iter()) {
        *dst = rp.from_mont(x) as i64;
    }
}

fn ntt_to_poly(f: &mut [u32], rp: &RingParams) -> Vec<i64> {
    ntt_inverse(f, rp.zetas, rp.q32, rp.q_inv, rp.inv_d_mont);
    f.iter().map(|&x| rp.from_mont(x) as i64).collect()
}

fn pointwise_mul_ntt(a: &[u32], b: &[u32], rp: &RingParams) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| mont_mul(ai, bi, rp.q32, rp.q_inv))
        .collect()
}

fn lift_signed(v: Vec<i64>, q: u64) -> Vec<i64> {
    let half = (q / 2) as i64;
    v.into_iter()
        .map(|x| if x > half { x - q as i64 } else { x })
        .collect()
}

/// Multiply-accumulate: `acc[k] += a_ntt[k] * b_hat[k]` for each k, with reduction.
///
/// # Bounds invariant (eager reduction)
///
/// `acc[k]`, `a_ntt[k]`, `b_hat[k]` are all in `[0, q)` on entry and exit.
/// Per coefficient:
///   * `prod = mont_mul(a, b)` with `a, b ∈ [0, q)` yields `prod ∈ [0, q)`.
///   * `acc[k] + prod ∈ [0, 2q)` — `ct_reduce` (signed-mask idiom) brings
///     it back to `[0, q)`.
///
/// The whole loop is a straight-line sequence of unconditional arithmetic,
/// which LLVM auto-vectorises into SIMD `vpmuludq / vpsrlq / vpsubd /
/// vpsrad / vpand / vpaddd` sequences under AVX2 (analogous under NEON).
#[inline(always)]
pub(crate) fn mac_ntt(acc: &mut [u32], a_ntt: &[u32], b_hat: &[u32], q: u32, q_inv: u32) {
    for k in 0..acc.len() {
        let prod = mont_mul(a_ntt[k], b_hat[k], q, q_inv);
        acc[k] = ct_reduce(acc[k].wrapping_add(prod), q);
    }
}

// ---------------------------------------------------------------------------
// Public polynomial operations
// ---------------------------------------------------------------------------

/// Negacyclic poly multiplication mod q. Returns (d,) in [0, q).
pub fn poly_mul(a: &[i64], b: &[i64], rp: &RingParams) -> Vec<i64> {
    let a_hat = poly_to_ntt(a, rp);
    let b_hat = poly_to_ntt(b, rp);
    let mut c_hat = pointwise_mul_ntt(&a_hat, &b_hat, rp);
    ntt_to_poly(&mut c_hat, rp)
}

/// Negacyclic poly multiplication; lifts result to signed [-q/2, q/2].
pub fn poly_mul_signed(a: &[i64], b: &[i64], rp: &RingParams) -> Vec<i64> {
    lift_signed(poly_mul(a, b, rp), rp.q)
}

/// Matrix-vector product: (rows x cols) poly matrix times (cols,) poly vector.
///
/// Accumulates in NTT domain; result in [0, q).
pub fn mat_vec(mat: &[i64], vec: &[i64], rows: usize, cols: usize, rp: &RingParams) -> Vec<i64> {
    let d = rp.d;
    // Pre-transform vector columns into flat buffer
    let mut vec_hat = vec![0u32; cols * d];
    for j in 0..cols {
        poly_to_ntt_buf(
            &vec[j * d..(j + 1) * d],
            rp,
            &mut vec_hat[j * d..(j + 1) * d],
        );
    }

    let mut result = vec![0i64; rows * d];
    let mut a_hat_buf = vec![0u32; d];
    let mut acc = vec![0u32; d];
    for i in 0..rows {
        acc.fill(0);
        for j in 0..cols {
            poly_to_ntt_buf(
                &mat[(i * cols + j) * d..(i * cols + j + 1) * d],
                rp,
                &mut a_hat_buf,
            );
            mac_ntt(
                &mut acc,
                &a_hat_buf,
                &vec_hat[j * d..(j + 1) * d],
                rp.q32,
                rp.q_inv,
            );
        }
        ntt_to_poly_buf(&mut acc, rp, &mut result[i * d..(i + 1) * d]);
    }
    result
}

/// Matrix-vector product with pre-NTT'd matrix.
///
/// `mat_ntt` is rows*cols*D u32 values already in NTT/Montgomery domain.
/// Skips the forward NTT on matrix entries — only the vector is transformed.
pub fn mat_vec_prentt(
    mat_ntt: &[u32],
    vec: &[i64],
    rows: usize,
    cols: usize,
    rp: &RingParams,
) -> Vec<i64> {
    let d = rp.d;
    // Pre-transform vector columns into flat buffer
    let mut vec_hat = vec![0u32; cols * d];
    for j in 0..cols {
        poly_to_ntt_buf(
            &vec[j * d..(j + 1) * d],
            rp,
            &mut vec_hat[j * d..(j + 1) * d],
        );
    }

    let mut result = vec![0i64; rows * d];
    let mut acc = vec![0u32; d];
    for i in 0..rows {
        acc.fill(0);
        for j in 0..cols {
            mac_ntt(
                &mut acc,
                &mat_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec_hat[j * d..(j + 1) * d],
                rp.q32,
                rp.q_inv,
            );
        }
        ntt_to_poly_buf(&mut acc, rp, &mut result[i * d..(i + 1) * d]);
    }
    result
}

/// Compute mat0 * vec0 + mat1 * vec1 with pre-NTT'd matrices.
///
/// Transforms vec0 and vec1 once, then accumulates both products into
/// a single NTT-domain accumulator per row, saving one set of inverse NTTs
/// and the coefficient-domain addition.
pub fn mat_vec_prentt_pair(
    mat0_ntt: &[u32],
    vec0: &[i64],
    mat1_ntt: &[u32],
    vec1: &[i64],
    rows: usize,
    cols: usize,
    rp: &RingParams,
) -> Vec<i64> {
    let d = rp.d;
    let mut vec0_hat = vec![0u32; cols * d];
    let mut vec1_hat = vec![0u32; cols * d];
    for j in 0..cols {
        poly_to_ntt_buf(
            &vec0[j * d..(j + 1) * d],
            rp,
            &mut vec0_hat[j * d..(j + 1) * d],
        );
        poly_to_ntt_buf(
            &vec1[j * d..(j + 1) * d],
            rp,
            &mut vec1_hat[j * d..(j + 1) * d],
        );
    }

    let mut result = vec![0i64; rows * d];
    let mut acc = vec![0u32; d];
    for i in 0..rows {
        acc.fill(0);
        for j in 0..cols {
            mac_ntt(
                &mut acc,
                &mat0_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec0_hat[j * d..(j + 1) * d],
                rp.q32,
                rp.q_inv,
            );
            mac_ntt(
                &mut acc,
                &mat1_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec1_hat[j * d..(j + 1) * d],
                rp.q32,
                rp.q_inv,
            );
        }
        ntt_to_poly_buf(&mut acc, rp, &mut result[i * d..(i + 1) * d]);
    }
    result
}

/// Pre-compute NTT form of a flat polynomial matrix (rows*cols*d i64 → rows*cols*d u32).
pub fn mat_to_ntt(mat: &[i64], rows: usize, cols: usize, rp: &RingParams) -> Vec<u32> {
    let d = rp.d;
    let mut ntt = vec![0u32; rows * cols * d];
    for idx in 0..rows * cols {
        poly_to_ntt_buf(
            &mat[idx * d..(idx + 1) * d],
            rp,
            &mut ntt[idx * d..(idx + 1) * d],
        );
    }
    ntt
}

/// Matrix-matrix product with pre-NTT'd right matrix.
pub fn mat_mul_prentt_b(
    a: &[i64],
    b_ntt: &[u32],
    r: usize,
    s: usize,
    t: usize,
    rp: &RingParams,
) -> Vec<i64> {
    let d = rp.d;
    let mut result = vec![0i64; r * t * d];
    let mut a_hat_row = vec![0u32; s * d];
    let mut acc = vec![0u32; d];
    for i in 0..r {
        for l in 0..s {
            poly_to_ntt_buf(
                &a[(i * s + l) * d..(i * s + l + 1) * d],
                rp,
                &mut a_hat_row[l * d..(l + 1) * d],
            );
        }
        for j in 0..t {
            acc.fill(0);
            for l in 0..s {
                mac_ntt(
                    &mut acc,
                    &a_hat_row[l * d..(l + 1) * d],
                    &b_ntt[(l * t + j) * d..(l * t + j + 1) * d],
                    rp.q32,
                    rp.q_inv,
                );
            }
            ntt_to_poly_buf(
                &mut acc,
                rp,
                &mut result[(i * t + j) * d..(i * t + j + 1) * d],
            );
        }
    }
    result
}

/// Matrix-matrix product: (r x s) times (s x t) poly matrices.
///
/// Result in [0, q).
pub fn mat_mul(a: &[i64], b: &[i64], r: usize, s: usize, t: usize, rp: &RingParams) -> Vec<i64> {
    let d = rp.d;
    // Pre-transform all of B
    let mut b_hat = vec![0u32; s * t * d];
    for idx in 0..s * t {
        poly_to_ntt_buf(
            &b[idx * d..(idx + 1) * d],
            rp,
            &mut b_hat[idx * d..(idx + 1) * d],
        );
    }

    let mut result = vec![0i64; r * t * d];
    let mut a_hat_row = vec![0u32; s * d];
    let mut acc = vec![0u32; d];
    for i in 0..r {
        for l in 0..s {
            poly_to_ntt_buf(
                &a[(i * s + l) * d..(i * s + l + 1) * d],
                rp,
                &mut a_hat_row[l * d..(l + 1) * d],
            );
        }
        for j in 0..t {
            acc.fill(0);
            for l in 0..s {
                mac_ntt(
                    &mut acc,
                    &a_hat_row[l * d..(l + 1) * d],
                    &b_hat[(l * t + j) * d..(l * t + j + 1) * d],
                    rp.q32,
                    rp.q_inv,
                );
            }
            ntt_to_poly_buf(
                &mut acc,
                rp,
                &mut result[(i * t + j) * d..(i * t + j + 1) * d],
            );
        }
    }
    result
}

/// Scale each row of v (shape n x d) by pre-NTT'd scalar w_hat. Result is signed.
pub fn scale_vec_with_ntt_w(w_hat: &[u32], v: &[i64], n: usize, rp: &RingParams) -> Vec<i64> {
    let d = rp.d;
    let half = (rp.q / 2) as i64;
    let q_i64 = rp.q as i64;
    let mut buf = vec![0u32; d];
    let mut result = vec![0i64; n * d];
    for i in 0..n {
        poly_to_ntt_buf(&v[i * d..(i + 1) * d], rp, &mut buf);
        for k in 0..d {
            buf[k] = mont_mul(w_hat[k], buf[k], rp.q32, rp.q_inv);
        }
        ntt_inverse(&mut buf, rp.zetas, rp.q32, rp.q_inv, rp.inv_d_mont);
        for k in 0..d {
            let x = rp.from_mont(buf[k]) as i64;
            result[i * d + k] = if x > half { x - q_i64 } else { x };
        }
    }
    result
}

/// Scale each row of v (shape n x d) by scalar poly w. Result is signed.
pub fn scale_vec(w: &[i64], v: &[i64], n: usize, rp: &RingParams) -> Vec<i64> {
    let w_hat = poly_to_ntt(w, rp);
    scale_vec_with_ntt_w(&w_hat, v, n, rp)
}

/// Scale each entry of M (shape r x c x d) by scalar poly w. Result is signed.
pub fn scale_mat(w: &[i64], m_mat: &[i64], r: usize, c: usize, rp: &RingParams) -> Vec<i64> {
    scale_vec(w, m_mat, r * c, rp)
}

// ---------------------------------------------------------------------------
// u64 backend: poly / matrix helpers for moduli that exceed 2^32.
//
// Structure mirrors the u32 side above, but every Montgomery constant
// is u64 and every intermediate product is u128.  Used by every shipping
// HVC ring (q ≈ 2³⁹ to 2⁵³).
// ---------------------------------------------------------------------------

/// Forward-NTT into a caller-provided buffer (avoids allocation). u64 backend.
#[inline]
pub(crate) fn poly_to_ntt_buf_u64(p: &[i64], rp: &RingParams64, buf: &mut [u64]) {
    for (dst, &x) in buf.iter_mut().zip(p.iter()) {
        *dst = rp.to_mont(rp.reduce_i64(x));
    }
    ntt_forward_u64(buf, rp.zetas, rp.q, rp.q_inv);
}

/// Inverse-NTT writing into an i64 output slice (avoids allocation). u64 backend.
#[inline]
pub(crate) fn ntt_to_poly_buf_u64(f: &mut [u64], rp: &RingParams64, out: &mut [i64]) {
    ntt_inverse_u64(f, rp.zetas, rp.q, rp.q_inv, rp.inv_d_mont);
    for (dst, &x) in out.iter_mut().zip(f.iter()) {
        *dst = rp.from_mont(x) as i64;
    }
}

/// Multiply-accumulate: acc[k] += a_ntt[k] * b_hat[k] for each k, with reduction.
///
/// u64 twin of [`mac_ntt`]: same bounds invariant (`acc[k] ∈ [0, q)` on
/// entry and exit, `a_ntt`/`b_hat` in NTT/Montgomery domain) with `q < 2^62`.
#[inline(always)]
pub(crate) fn mac_ntt_u64(acc: &mut [u64], a_ntt: &[u64], b_hat: &[u64], q: u64, q_inv: u64) {
    for k in 0..acc.len() {
        let prod = mont_mul_u64(a_ntt[k], b_hat[k], q, q_inv);
        acc[k] = ct_reduce_u64(acc[k].wrapping_add(prod), q);
    }
}

/// Negacyclic poly multiplication mod q. u64 backend, result in [0, q).
pub fn poly_mul_u64(a: &[i64], b: &[i64], rp: &RingParams64) -> Vec<i64> {
    let d = rp.d;
    let mut a_hat = vec![0u64; d];
    let mut b_hat = vec![0u64; d];
    poly_to_ntt_buf_u64(a, rp, &mut a_hat);
    poly_to_ntt_buf_u64(b, rp, &mut b_hat);
    for k in 0..d {
        a_hat[k] = mont_mul_u64(a_hat[k], b_hat[k], rp.q, rp.q_inv);
    }
    let mut out = vec![0i64; d];
    ntt_to_poly_buf_u64(&mut a_hat, rp, &mut out);
    out
}

/// Pre-compute NTT form of a flat polynomial matrix (rows*cols*d i64 → rows*cols*d u64).
pub fn mat_to_ntt_u64(mat: &[i64], rows: usize, cols: usize, rp: &RingParams64) -> Vec<u64> {
    let d = rp.d;
    let mut ntt = vec![0u64; rows * cols * d];
    for idx in 0..rows * cols {
        poly_to_ntt_buf_u64(
            &mat[idx * d..(idx + 1) * d],
            rp,
            &mut ntt[idx * d..(idx + 1) * d],
        );
    }
    ntt
}

/// Matrix-matrix product with pre-NTT'd right matrix. u64 backend.
pub fn mat_mul_prentt_b_u64(
    a: &[i64],
    b_ntt: &[u64],
    r: usize,
    s: usize,
    t: usize,
    rp: &RingParams64,
) -> Vec<i64> {
    let d = rp.d;
    let mut result = vec![0i64; r * t * d];
    let mut a_hat_row = vec![0u64; s * d];
    let mut acc = vec![0u64; d];
    for i in 0..r {
        for l in 0..s {
            poly_to_ntt_buf_u64(
                &a[(i * s + l) * d..(i * s + l + 1) * d],
                rp,
                &mut a_hat_row[l * d..(l + 1) * d],
            );
        }
        for j in 0..t {
            acc.fill(0);
            for l in 0..s {
                mac_ntt_u64(
                    &mut acc,
                    &a_hat_row[l * d..(l + 1) * d],
                    &b_ntt[(l * t + j) * d..(l * t + j + 1) * d],
                    rp.q,
                    rp.q_inv,
                );
            }
            ntt_to_poly_buf_u64(
                &mut acc,
                rp,
                &mut result[(i * t + j) * d..(i * t + j + 1) * d],
            );
        }
    }
    result
}

/// Matrix-vector product with pre-NTT'd matrix. u64 backend.
pub fn mat_vec_prentt_u64(
    mat_ntt: &[u64],
    vec: &[i64],
    rows: usize,
    cols: usize,
    rp: &RingParams64,
) -> Vec<i64> {
    let d = rp.d;
    let mut vec_hat = vec![0u64; cols * d];
    for j in 0..cols {
        poly_to_ntt_buf_u64(
            &vec[j * d..(j + 1) * d],
            rp,
            &mut vec_hat[j * d..(j + 1) * d],
        );
    }

    let mut result = vec![0i64; rows * d];
    let mut acc = vec![0u64; d];
    for i in 0..rows {
        acc.fill(0);
        for j in 0..cols {
            mac_ntt_u64(
                &mut acc,
                &mat_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec_hat[j * d..(j + 1) * d],
                rp.q,
                rp.q_inv,
            );
        }
        ntt_to_poly_buf_u64(&mut acc, rp, &mut result[i * d..(i + 1) * d]);
    }
    result
}

/// Compute mat0 * vec0 + mat1 * vec1 with pre-NTT'd matrices. u64 backend.
pub fn mat_vec_prentt_pair_u64(
    mat0_ntt: &[u64],
    vec0: &[i64],
    mat1_ntt: &[u64],
    vec1: &[i64],
    rows: usize,
    cols: usize,
    rp: &RingParams64,
) -> Vec<i64> {
    let d = rp.d;
    let mut vec0_hat = vec![0u64; cols * d];
    let mut vec1_hat = vec![0u64; cols * d];
    for j in 0..cols {
        poly_to_ntt_buf_u64(
            &vec0[j * d..(j + 1) * d],
            rp,
            &mut vec0_hat[j * d..(j + 1) * d],
        );
        poly_to_ntt_buf_u64(
            &vec1[j * d..(j + 1) * d],
            rp,
            &mut vec1_hat[j * d..(j + 1) * d],
        );
    }

    let mut result = vec![0i64; rows * d];
    let mut acc = vec![0u64; d];
    for i in 0..rows {
        acc.fill(0);
        for j in 0..cols {
            mac_ntt_u64(
                &mut acc,
                &mat0_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec0_hat[j * d..(j + 1) * d],
                rp.q,
                rp.q_inv,
            );
            mac_ntt_u64(
                &mut acc,
                &mat1_ntt[(i * cols + j) * d..(i * cols + j + 1) * d],
                &vec1_hat[j * d..(j + 1) * d],
                rp.q,
                rp.q_inv,
            );
        }
        ntt_to_poly_buf_u64(&mut acc, rp, &mut result[i * d..(i + 1) * d]);
    }
    result
}

/// Matrix-matrix product: (r x s) times (s x t) poly matrices. u64 backend.
///
/// Result in [0, q).
pub fn mat_mul_u64(
    a: &[i64],
    b: &[i64],
    r: usize,
    s: usize,
    t: usize,
    rp: &RingParams64,
) -> Vec<i64> {
    let d = rp.d;
    // Pre-transform all of B
    let mut b_hat = vec![0u64; s * t * d];
    for idx in 0..s * t {
        poly_to_ntt_buf_u64(
            &b[idx * d..(idx + 1) * d],
            rp,
            &mut b_hat[idx * d..(idx + 1) * d],
        );
    }

    let mut result = vec![0i64; r * t * d];
    let mut a_hat_row = vec![0u64; s * d];
    let mut acc = vec![0u64; d];
    for i in 0..r {
        for l in 0..s {
            poly_to_ntt_buf_u64(
                &a[(i * s + l) * d..(i * s + l + 1) * d],
                rp,
                &mut a_hat_row[l * d..(l + 1) * d],
            );
        }
        for j in 0..t {
            acc.fill(0);
            for l in 0..s {
                mac_ntt_u64(
                    &mut acc,
                    &a_hat_row[l * d..(l + 1) * d],
                    &b_hat[(l * t + j) * d..(l * t + j + 1) * d],
                    rp.q,
                    rp.q_inv,
                );
            }
            ntt_to_poly_buf_u64(
                &mut acc,
                rp,
                &mut result[(i * t + j) * d..(i * t + j + 1) * d],
            );
        }
    }
    result
}

/// Scale each row of v (shape n x d) by pre-NTT'd scalar w_hat. Result is signed. u64 backend.
pub fn scale_vec_with_ntt_w_u64(w_hat: &[u64], v: &[i64], n: usize, rp: &RingParams64) -> Vec<i64> {
    let d = rp.d;
    let half = (rp.q / 2) as i64;
    let q_i64 = rp.q as i64;
    let mut buf = vec![0u64; d];
    let mut result = vec![0i64; n * d];
    for i in 0..n {
        poly_to_ntt_buf_u64(&v[i * d..(i + 1) * d], rp, &mut buf);
        for k in 0..d {
            buf[k] = mont_mul_u64(w_hat[k], buf[k], rp.q, rp.q_inv);
        }
        ntt_inverse_u64(&mut buf, rp.zetas, rp.q, rp.q_inv, rp.inv_d_mont);
        for k in 0..d {
            let x = rp.from_mont(buf[k]) as i64;
            result[i * d + k] = if x > half { x - q_i64 } else { x };
        }
    }
    result
}

/// Scale each row of v (shape n x d) by scalar poly w. Result is signed. u64 backend.
pub fn scale_vec_u64(w: &[i64], v: &[i64], n: usize, rp: &RingParams64) -> Vec<i64> {
    let mut w_hat = vec![0u64; rp.d];
    poly_to_ntt_buf_u64(w, rp, &mut w_hat);
    scale_vec_with_ntt_w_u64(&w_hat, v, n, rp)
}

/// Scale each entry of M (shape r x c x d) by scalar poly w. Result is signed. u64 backend.
pub fn scale_mat_u64(w: &[i64], m_mat: &[i64], r: usize, c: usize, rp: &RingParams64) -> Vec<i64> {
    scale_vec_u64(w, m_mat, r * c, rp)
}

// u32 / u64 NTT correctness is exercised by the synthesized tests in
// `ntt.rs::tests` plus the per-profile end-to-end tests in `tests/`.
// No shipping profile uses the u32 NTT stack today (every HVC modulus
// exceeds 2³² and routes through the u64 Montgomery stack), but the
// u32 path is kept compile-checked for future cells.
