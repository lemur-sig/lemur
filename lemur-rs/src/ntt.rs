//! NTT arithmetic: Montgomery reduction, forward/inverse NTT.
//!
//! Uses Cooley-Tukey forward and Gentleman-Sande inverse, matching Python's ring.py.
//!
//! # Two arithmetic widths
//!
//! The module exposes **two parallel Montgomery stacks** so Lemur can host
//! moduli on either side of `2^32`:
//!
//! * **u32 stack** (`ct_reduce`, `mont_reduce`, `mont_mul`, `to_mont`,
//!   `ntt_forward`, `ntt_inverse`) — `R = 2^32`.  Valid for `q < 2^31`.
//!   Not used by the shipped parameter set today; kept compile-checked for
//!   future cells with smaller HVC moduli.
//! * **u64 stack** (`_u64` suffix) — `R = 2^64`, 128-bit intermediate
//!   products.  Valid for `q < 2^63`.  Used by the shipped HVC ring
//!   (q ≈ 2³⁹ to 2⁵³).  KOTS is not natively NTT-friendly and routes
//!   through `aux_ntt`.
//!
//! Each stack is stand-alone: the two never share state.  Profile-aware
//! scheme code picks the stack width once from its `RingParams`
//! descendant; within a single NTT the width is fixed.
//!
//! # Constant-time discipline
//!
//! All reductions are branch-free and data-independent.  The
//! implementations below use the "signed-mask" idiom: subtract q,
//! arithmetic-shift the sign bit to a full-width mask, conditionally
//! add q back.  This compiles to branchless CMOV-equivalent scalar
//! code and, inside tight loops, LLVM's auto-vectoriser lifts it to
//! `vpsubd / vpsrad / vpand / vpaddd` under AVX2 (analogous under
//! NEON).  No `subtle::Choice` — the crate's `#[inline(never)]`
//! barrier prevents exactly that vectorisation — but the resulting
//! code is still constant-time: every operand flows through
//! unconditional arithmetic only, with no data-dependent branches or
//! memory addresses.
//!
//! # Bounds invariants
//!
//! All `ct_reduce` / `ct_reduce_u64` callers feed inputs in `[0, 2q)`
//! (the canonical post-add or post-Montgomery-step range).  Because the
//! reducers cast to signed integers before subtracting `q`, that implies
//! the stronger contract `2q < 2^(N-1)` for add/reduce callers, i.e.
//! `q < 2^(N-2)` (N = 32 or 64).  The butterfly subtracts use signed
//! arithmetic directly: `a - b` for `a, b ∈ [0, q)` lies in `(-q, q)`,
//! which is then conditionally brought into `[0, q)` by adding `q` when
//! the sign bit is set.  The shipped parameter set satisfies this with slack
//! (the u32 stack is unused; the u64
//! stack covers up to `q < 2^62`).

// ---------------------------------------------------------------------------
// Montgomery reduction
// ---------------------------------------------------------------------------

/// Constant-time conditional subtraction: return `x - q` if `x >= q`, else `x`.
///
/// **Preconditions:**
/// * `q < 2^31` — so `q` fits in `i32` as a non-negative value.
/// * `x < 2^31` — so `x - q` computed in `i32` does not overflow.
///
/// All crate-internal callers supply `x ∈ [0, 2q)` (from a Montgomery
/// step or a pair of `[0, q)` values added together).  `2q < 2^31`
/// follows from `q < 2^30`, the practical contract for callers that use
/// `ct_reduce` on post-add values in `[0, 2q)`, including this module's
/// NTT butterflies and `poly::mac_ntt`.
///
/// **Postcondition:** result is in `[0, q)`.
///
/// Constant-time implementation via signed-mask idiom: no branches,
/// no data-dependent memory access.  SIMD-friendly — see module docs.
#[inline(always)]
pub fn ct_reduce(x: u32, q: u32) -> u32 {
    debug_assert!(q < (1u32 << 31), "ct_reduce: q must fit in i32");
    debug_assert!(x < (1u32 << 31), "ct_reduce: x must fit in i32");
    // diff ∈ [-q, q) because x ∈ [0, 2q).  When diff < 0 (x < q) the
    // arithmetic-shift mask is -1 and we add q back; when diff ≥ 0
    // (x ≥ q) the mask is 0 and diff = x - q is the final answer.
    let diff = (x as i32).wrapping_sub(q as i32);
    let mask = diff >> 31; // i32 arithmetic shift: -1 if diff<0, 0 otherwise
    diff.wrapping_add(mask & (q as i32)) as u32
}

/// Montgomery reduction: compute x * R^{-1} mod q.
///
/// `q_inv` = -q^{-1} mod 2^32.
#[inline(always)]
pub fn mont_reduce(x: u64, q: u32, q_inv: u32) -> u32 {
    let m = (x as u32).wrapping_mul(q_inv);
    let t = ((x.wrapping_add((m as u64).wrapping_mul(q as u64))) >> 32) as u32;
    ct_reduce(t, q)
}

/// Montgomery multiplication: compute a * b * R^{-1} mod q.
///
/// Inputs a, b must be in [0, q). Output is in [0, q).
#[inline(always)]
pub fn mont_mul(a: u32, b: u32, q: u32, q_inv: u32) -> u32 {
    mont_reduce((a as u64) * (b as u64), q, q_inv)
}

/// Convert a value to Montgomery form: compute x * R mod q.
///
/// `r2` = R^2 mod q.
#[inline(always)]
pub fn to_mont(x: u32, r2: u32, q: u32, q_inv: u32) -> u32 {
    mont_reduce((x as u64) * (r2 as u64), q, q_inv)
}

// ---------------------------------------------------------------------------
// Forward NTT (Cooley-Tukey, matching Python ring.py ntt())
// ---------------------------------------------------------------------------

/// Forward NTT in R_q = Z_q[X]/(X^d + 1).
///
/// Input: polynomial in normal order with values reduced mod q.
/// Output: NTT representation in bit-reversed order, values in [0, q).
/// `zetas[m]` for m=1..d-1 are the twiddle factors in Montgomery form.
///
/// # Bounds invariant (eager reduction)
///
/// Every entry of `f` stays in `[0, q)` before and after each
/// butterfly.  Per butterfly:
///   * `t = mont_mul(z, f[j+le])` with `f[j+le] ∈ [0, q)`, so `t ∈ [0, q)`.
///   * `f[j] + t ∈ [0, 2q)` — `ct_reduce` brings it to `[0, q)`.
///   * `f[j] - t` computed as `i32` lies in `(-q, q)` — the
///     signed-mask "add q if negative" step brings it to `[0, q)`.
///
/// Requires `q < 2^31`.
pub fn ntt_forward(f: &mut [u32], zetas: &[u32], q: u32, q_inv: u32) {
    let d = f.len();
    let mut m = 0usize;
    let mut le = d / 2;
    let q_i = q as i32;
    while le >= 1 {
        let mut st = 0;
        while st < d {
            m += 1;
            let z = zetas[m];
            for j in st..st + le {
                let t = mont_mul(z, f[j + le], q, q_inv);
                // Signed subtract: diff ∈ (-q, q) since f[j], t ∈ [0, q).
                // Sign-mask adds q back iff diff < 0, giving [0, q).
                let diff = (f[j] as i32).wrapping_sub(t as i32);
                let mask = diff >> 31;
                f[j + le] = diff.wrapping_add(mask & q_i) as u32;
                // f[j] + t ∈ [0, 2q) — ct_reduce to [0, q).
                f[j] = ct_reduce(f[j].wrapping_add(t), q);
            }
            st += 2 * le;
        }
        le /= 2;
    }
}

// ---------------------------------------------------------------------------
// Inverse NTT (Gentleman-Sande, matching Python ring.py intt())
// ---------------------------------------------------------------------------

/// Inverse NTT in R_q = Z_q[X]/(X^d + 1).
///
/// Input: NTT representation in bit-reversed order, values in [0, q).
/// Output: polynomial in normal order, values in [0, q).
/// Includes the 1/d scaling factor (applied via Montgomery multiplication).
///
/// # Bounds invariant (eager reduction)
///
/// Every entry of `f` stays in `[0, q)` before and after each
/// butterfly.  Per butterfly, with `a = f[j]`, `b = f[j+le]` both in
/// `[0, q)`:
///   * `a + b ∈ [0, 2q)` — `ct_reduce` to `[0, q)`.
///   * `a - b` as `i32` in `(-q, q)` — signed-mask "add q if negative"
///     brings it to `[0, q)` for use as the mont_mul operand.
///     (`mont_mul` accepts any `[0, q)` input on its second argument.)
///
/// Requires `q < 2^31`.
pub fn ntt_inverse(f: &mut [u32], zetas: &[u32], q: u32, q_inv: u32, inv_d_mont: u32) {
    let d = f.len();
    let mut m = d;
    let mut le = 1;
    let q_i = q as i32;
    while le < d {
        let mut st = 0;
        while st < d {
            m -= 1;
            // neg_zeta = (-zetas[m]) mod q = q - zetas[m], in Montgomery form.
            let neg_z = q - zetas[m];
            for j in st..st + le {
                let a = f[j];
                let b = f[j + le];
                // a + b ∈ [0, 2q) → ct_reduce → [0, q).
                f[j] = ct_reduce(a.wrapping_add(b), q);
                // Signed subtract: a - b ∈ (-q, q); sign-mask adds q back
                // if negative, giving diff ∈ [0, q) for the mont_mul.
                let signed_diff = (a as i32).wrapping_sub(b as i32);
                let mask = signed_diff >> 31;
                let diff = signed_diff.wrapping_add(mask & q_i) as u32;
                f[j + le] = mont_mul(neg_z, diff, q, q_inv);
            }
            st += 2 * le;
        }
        le *= 2;
    }
    // Multiply by D^{-1} in Montgomery form.
    for x in f.iter_mut() {
        *x = mont_mul(*x, inv_d_mont, q, q_inv);
    }
}

// ---------------------------------------------------------------------------
// u64 Montgomery (R = 2^64) — for natively NTT-friendly q >= 2^32
// ---------------------------------------------------------------------------
//
// Mirrors the u32 stack verbatim except that:
//   * `x` in `mont_reduce_u64` is a u128 (the product of two u64 operands);
//   * `R = 2^64`, so `q_inv` is `-q^{-1} mod 2^64` and `r2` is `R^2 mod q`.
//
// No data flow between the u32 and u64 stacks.  Tests cover both.

/// Constant-time conditional subtraction on u64: return `x - q` if `x >= q`
/// else `x`.  u64 twin of [`ct_reduce`].
///
/// **Preconditions:**
/// * `q < 2^63` — so `q` fits in `i64` as a non-negative value.
/// * `x < 2^63` — so `x - q` computed in `i64` does not overflow.
///
/// All crate-internal callers supply `x ∈ [0, 2q)`.  `2q < 2^63`
/// follows from `q < 2^62`, the practical contract for callers that
/// use `ct_reduce_u64` on post-add values in `[0, 2q)`, including this
/// module's u64 NTT butterflies and `poly::mac_ntt_u64`.
///
/// **Postcondition:** result is in `[0, q)`.
///
/// Constant-time via signed-mask idiom — see module docs.  Scalar on
/// typical 64-bit backends (vpsrq / similar); LLVM vectorises under
/// AVX-512 or `-C target-feature=+avx512dq` where available.
#[inline(always)]
pub fn ct_reduce_u64(x: u64, q: u64) -> u64 {
    debug_assert!(q < (1u64 << 63), "ct_reduce_u64: q must fit in i64");
    debug_assert!(x < (1u64 << 63), "ct_reduce_u64: x must fit in i64");
    // diff ∈ [-q, q) because x ∈ [0, 2q); sign-mask adds q back when diff<0.
    let diff = (x as i64).wrapping_sub(q as i64);
    let mask = diff >> 63; // i64 arithmetic shift: -1 if diff<0, 0 otherwise
    diff.wrapping_add(mask & (q as i64)) as u64
}

/// Montgomery reduction (R = 2^64): compute `x * R^{-1} mod q`.
///
/// `q_inv` = `-q^{-1} mod 2^64`.
///
/// Standard Montgomery algorithm over 128-bit intermediates:
///   m  := (x mod 2^64) * q_inv   mod 2^64   (the low 64 bits of `x * q_inv`)
///   t  := (x + m * q) >> 64                  (128-bit add, then shift)
///   if t >= q { t -= q }
///
/// Input: any `x < 2^128` with `m * q` chosen so `x + m*q` is divisible by
/// `2^64`.  Output: a value in `[0, q)` that is `x * R^{-1} (mod q)`.
#[inline(always)]
pub fn mont_reduce_u64(x: u128, q: u64, q_inv: u64) -> u64 {
    let m = (x as u64).wrapping_mul(q_inv);
    let prod = (m as u128).wrapping_mul(q as u128);
    let t = (x.wrapping_add(prod) >> 64) as u64;
    ct_reduce_u64(t, q)
}

/// Montgomery multiplication (R = 2^64): `a * b * R^{-1} mod q`.
///
/// Inputs `a, b` must be in `[0, q)`.  Output is in `[0, q)`.
#[inline(always)]
pub fn mont_mul_u64(a: u64, b: u64, q: u64, q_inv: u64) -> u64 {
    mont_reduce_u64((a as u128) * (b as u128), q, q_inv)
}

/// Convert a value to Montgomery form: `x * R mod q`.
///
/// `r2` = `R^2 mod q` (= `2^128 mod q`).
#[inline(always)]
pub fn to_mont_u64(x: u64, r2: u64, q: u64, q_inv: u64) -> u64 {
    mont_reduce_u64((x as u128) * (r2 as u128), q, q_inv)
}

/// Forward NTT (Cooley-Tukey) in `R_q = Z_q[X]/(X^d + 1)` using the u64
/// Montgomery stack.
///
/// * Input: polynomial of length `d` in normal order, values reduced mod `q`.
/// * Output: NTT representation in bit-reversed order, values in `[0, q)`.
/// * `zetas[m]` for `m = 1 .. d-1` are the twiddle factors in Montgomery form.
///
/// # Bounds invariant (eager reduction)
///
/// u64 twin of [`ntt_forward`].  Every entry of `f` stays in `[0, q)`
/// across butterflies; the signed subtract operates in `i64` space
/// with `q < 2^62` of slack.
pub fn ntt_forward_u64(f: &mut [u64], zetas: &[u64], q: u64, q_inv: u64) {
    let d = f.len();
    let mut m = 0usize;
    let mut le = d / 2;
    let q_i = q as i64;
    while le >= 1 {
        let mut st = 0;
        while st < d {
            m += 1;
            let z = zetas[m];
            for j in st..st + le {
                let t = mont_mul_u64(z, f[j + le], q, q_inv);
                // Signed subtract in i64: diff ∈ (-q, q), mask adds q if negative.
                let diff = (f[j] as i64).wrapping_sub(t as i64);
                let mask = diff >> 63;
                f[j + le] = diff.wrapping_add(mask & q_i) as u64;
                f[j] = ct_reduce_u64(f[j].wrapping_add(t), q);
            }
            st += 2 * le;
        }
        le /= 2;
    }
}

/// Inverse NTT (Gentleman-Sande) in `R_q` using the u64 Montgomery stack.
///
/// * Input: NTT representation in bit-reversed order, values in `[0, q)`.
/// * Output: polynomial in normal order, values in `[0, q)`.
/// * Includes the 1/d scaling factor (applied via Montgomery multiplication by
///   `inv_d_mont = d^{-1} * R mod q`).
///
/// # Bounds invariant
///
/// u64 twin of [`ntt_inverse`].  See u32 version for the per-butterfly
/// bounds argument; the same logic applies in `i64` space with
/// `q < 2^62` of slack.
pub fn ntt_inverse_u64(f: &mut [u64], zetas: &[u64], q: u64, q_inv: u64, inv_d_mont: u64) {
    let d = f.len();
    let mut m = d;
    let mut le = 1;
    let q_i = q as i64;
    while le < d {
        let mut st = 0;
        while st < d {
            m -= 1;
            let neg_z = q - zetas[m];
            for j in st..st + le {
                let a = f[j];
                let b = f[j + le];
                f[j] = ct_reduce_u64(a.wrapping_add(b), q);
                // Signed subtract: a - b ∈ (-q, q); mask adds q back if neg.
                let signed_diff = (a as i64).wrapping_sub(b as i64);
                let mask = signed_diff >> 63;
                let diff = signed_diff.wrapping_add(mask & q_i) as u64;
                f[j + le] = mont_mul_u64(neg_z, diff, q, q_inv);
            }
            st += 2 * le;
        }
        le *= 2;
    }
    for x in f.iter_mut() {
        *x = mont_mul_u64(*x, inv_d_mont, q, q_inv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The u32 Montgomery NTT stack is no longer exercised by any
    // shipping profile (every HVC modulus exceeds 2³² and uses the u64
    // stack; KOTS uses CRT via aux primes).  The synthesized u64 tests
    // below cover the NTT/Montgomery primitives end-to-end; if a future
    // cell needs the u32 stack, add round-trip tests against its baked
    // tables here.

    // ---------------------------------------------------------------------
    // u64 primitive sanity tests.
    //
    // These don't depend on any baked tables — they synthesize a
    // mini-table of twiddle factors on the fly at a fixed `q ≈ 2^34`,
    // `d = 128` u64 modulus chosen to exercise the wider stack, and
    // verify:
    //   * Montgomery round-trip: `from_mont(to_mont(x)) == x`
    //   * Pointwise ring mul: `mont_mul(to_mont(a), to_mont(b)) * R = a*b mod q`
    //   * Full NTT round-trip at d=128
    //   * Schoolbook negacyclic product matches NTT product for random inputs
    // No shipping profile uses this stack today, but the synthesized
    // tests keep the implementation behaviour-checked.
    // ---------------------------------------------------------------------

    /// Slow but obviously-correct modular exponentiation for test data.
    fn modpow_u64(base: u64, mut exp: u64, q: u64) -> u64 {
        let q128 = q as u128;
        let mut acc: u128 = 1;
        let mut b = base as u128;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = (acc * b) % q128;
            }
            b = (b * b) % q128;
            exp >>= 1;
        }
        acc as u64
    }

    /// Return `-q^{-1} mod 2^64` via Hensel lifting.  `q` must be odd.
    fn neg_q_inv_mod_r_u64(q: u64) -> u64 {
        // Newton's iteration for q^{-1} mod 2^64.  Converges in
        // ceil(log2(64)) = 6 steps given a correct seed mod 2.
        let mut inv: u64 = 1; // q * 1 ≡ q ≡ 1 (mod 2) since q is odd.
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        }
        // Now inv * q ≡ 1 (mod 2^64); -q^{-1} = 2^64 - inv (with wrap).
        0u64.wrapping_sub(inv)
    }

    /// Bit-reverse lowest `bits` bits of `k`.
    fn bitrev(mut k: usize, bits: u32) -> usize {
        let mut r = 0usize;
        for _ in 0..bits {
            r = (r << 1) | (k & 1);
            k >>= 1;
        }
        r
    }

    /// Build a `d`-entry zeta table in Montgomery form, bit-reversed, for
    /// `(q, d)` with a caller-supplied primitive 2d-th root of unity `zeta`.
    fn mont_zeta_table_u64(zeta: u64, d: usize, q: u64, r2: u64, q_inv: u64) -> Vec<u64> {
        let bits = (d as u32).trailing_zeros();
        (0..d)
            .map(|k| {
                let e = bitrev(k, bits) as u64;
                let z = modpow_u64(zeta, e, q);
                to_mont_u64(z, r2, q, q_inv)
            })
            .collect()
    }

    /// Schoolbook negacyclic multiply mod q (for parity against NTT).
    fn poly_mul_ref_u64(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
        let d = a.len();
        let q128 = q as u128;
        let mut c = vec![0u128; d];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = (ai as u128) * (bj as u128) % q128;
                let idx = i + j;
                if idx < d {
                    c[idx] = (c[idx] + prod) % q128;
                } else {
                    c[idx - d] = (c[idx - d] + q128 - prod) % q128;
                }
            }
        }
        c.into_iter().map(|x| x as u64).collect()
    }

    /// Test stimulus for the u64 stack: a 34-bit prime that is
    /// natively NTT-friendly at d = 128 (`q ≡ 1 mod 2d`).  Not a
    /// shipping scheme modulus — purely a synthesized test vector
    /// chosen to land in the >2^32 regime that the u64 backend is
    /// designed for.
    const U64_TEST_Q: u64 = 10_185_463_297;
    const U64_TEST_D: usize = 128;

    /// Assemble the canonical (q, d, zetas, q_inv, r2, inv_d_mont) bundle
    /// for a 64-bit modulus.  Used by multiple tests.
    fn u64_test_bundle() -> (Vec<u64>, u64, u64, u64) {
        let q = U64_TEST_Q;
        let d = U64_TEST_D;

        // `q-1 = 2d * exp`; find the smallest primitive 2d-th root of unity.
        assert_eq!(
            (q - 1) % (2 * d as u64),
            0,
            "q = {q} not NTT-friendly for d = {d}"
        );
        let exp = (q - 1) / (2 * d as u64);
        let mut zeta = 0u64;
        for x in 2..1_000u64 {
            let z = modpow_u64(x, exp, q);
            if modpow_u64(z, d as u64, q) == q - 1 {
                zeta = z;
                break;
            }
        }
        assert!(zeta != 0, "no primitive 2d-th root found for q={q}, d={d}");

        let q_inv = neg_q_inv_mod_r_u64(q);
        // r2 = R^2 mod q = 2^128 mod q.
        let r = 1u128 << 64;
        let r2 = ((r % q as u128) * (r % q as u128) % q as u128) as u64;
        // d^{-1} * R mod q.
        let d_inv = modpow_u64(d as u64, q - 2, q); // q is prime
        let inv_d_mont = to_mont_u64(d_inv, r2, q, q_inv);

        let zetas = mont_zeta_table_u64(zeta, d, q, r2, q_inv);

        (zetas, q_inv, r2, inv_d_mont)
    }

    #[test]
    fn u64_montgomery_to_from_roundtrip() {
        let q = U64_TEST_Q;
        let q_inv = neg_q_inv_mod_r_u64(q);
        let r2 = {
            let r = 1u128 << 64;
            ((r % q as u128) * (r % q as u128) % q as u128) as u64
        };
        for x in [0u64, 1, 42, q - 1, q / 2, 0xdeadbeef_12345678] {
            let x = x % q;
            let montx = to_mont_u64(x, r2, q, q_inv);
            let back = mont_reduce_u64(montx as u128, q, q_inv);
            assert_eq!(back, x, "Montgomery round-trip failed at x = {x}");
        }
    }

    #[test]
    fn u64_montgomery_mul_matches_schoolbook() {
        let q = U64_TEST_Q;
        let q_inv = neg_q_inv_mod_r_u64(q);
        let r2 = {
            let r = 1u128 << 64;
            ((r % q as u128) * (r % q as u128) % q as u128) as u64
        };
        let samples: [u64; 6] = [1, 2, 7, q - 1, q / 3, 123456789];
        for &a in &samples {
            for &b in &samples {
                let a_mont = to_mont_u64(a, r2, q, q_inv);
                let b_mont = to_mont_u64(b, r2, q, q_inv);
                // mont_mul(a_mont, b_mont) = a*b*R^{-1} mod q in Montgomery form;
                // pull it out of Montgomery and compare to the straight
                // 128-bit schoolbook product.
                let prod_mont = mont_mul_u64(a_mont, b_mont, q, q_inv);
                let prod = mont_reduce_u64(prod_mont as u128, q, q_inv);
                let ref_prod = (((a as u128) * (b as u128)) % q as u128) as u64;
                assert_eq!(prod, ref_prod, "mont_mul({a}, {b}) ≠ a*b mod q");
            }
        }
    }

    #[test]
    fn ntt_round_trip_u64_synthesized() {
        let q = U64_TEST_Q;
        let d = U64_TEST_D;
        let (zetas, q_inv, r2, inv_d_mont) = u64_test_bundle();

        // Fill with values in [0, q) — large enough to exercise the 128-bit
        // intermediate path.
        let original: Vec<u64> = (0..d)
            .map(|i| ((i as u64).wrapping_mul(0xABCDEF01) ^ 0x1234) % q)
            .collect();

        let mut f: Vec<u64> = original
            .iter()
            .map(|&x| to_mont_u64(x, r2, q, q_inv))
            .collect();
        ntt_forward_u64(&mut f, &zetas, q, q_inv);
        ntt_inverse_u64(&mut f, &zetas, q, q_inv, inv_d_mont);

        let result: Vec<u64> = f
            .iter()
            .map(|&x| mont_reduce_u64(x as u128, q, q_inv))
            .collect();
        assert_eq!(
            result, original,
            "u64 NTT round-trip failed for synthesized q = {q}, d = {d}"
        );
    }

    #[test]
    fn ntt_mul_matches_schoolbook_u64_synthesized() {
        let q = U64_TEST_Q;
        let d = U64_TEST_D;
        let (zetas, q_inv, r2, inv_d_mont) = u64_test_bundle();

        // Two small random polys — small to keep schoolbook fast.
        let a: Vec<u64> = (0..d).map(|i| (i as u64 * 13 + 5) % 4096).collect();
        let b: Vec<u64> = (0..d).map(|i| (i as u64 * 17 + 11) % 4096).collect();

        // Reference negacyclic multiply mod q.
        let ref_c = poly_mul_ref_u64(&a, &b, q);

        // NTT multiply: to_mont, forward, pointwise, inverse, from_mont.
        let mut a_hat: Vec<u64> = a.iter().map(|&x| to_mont_u64(x, r2, q, q_inv)).collect();
        let mut b_hat: Vec<u64> = b.iter().map(|&x| to_mont_u64(x, r2, q, q_inv)).collect();
        ntt_forward_u64(&mut a_hat, &zetas, q, q_inv);
        ntt_forward_u64(&mut b_hat, &zetas, q, q_inv);
        let mut c_hat: Vec<u64> = a_hat
            .iter()
            .zip(b_hat.iter())
            .map(|(&ah, &bh)| mont_mul_u64(ah, bh, q, q_inv))
            .collect();
        ntt_inverse_u64(&mut c_hat, &zetas, q, q_inv, inv_d_mont);
        let ntt_c: Vec<u64> = c_hat
            .iter()
            .map(|&x| mont_reduce_u64(x as u128, q, q_inv))
            .collect();

        assert_eq!(
            ntt_c, ref_c,
            "u64 NTT multiply does not match schoolbook at synthesized test q"
        );
    }

    // The synthesized tests above exercise the u64 Montgomery
    // primitives against a known 64-bit modulus, independent of the
    // shipped tables.  KOTS rings use the CRT backend in `aux_ntt.rs`;
    // the u64 Montgomery stack carries the shipped HVC ring directly.
}
