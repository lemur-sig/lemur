//! CRT-NTT backend.
//!
//! Fast negacyclic multiplication in `R_q = Z_q[X]/(X^d + 1)` via two
//! auxiliary NTT-friendly primes near `2^48`.  Both primes are `≡ 1 mod
//! 512`, so each supports a standard radix-2 negacyclic NTT for `d ∈
//! {128, 256}`.  The scheme modulus `q` does **not** need to be
//! NTT-friendly — it only needs to satisfy the paper-facing conditions
//! (primality + size).
//!
//! Used for the KOTS ring R_{q'}: the KOTS prime satisfies `q' ≡ 17
//! (mod 32)`, so a native length-d negacyclic NTT is not available on
//! the scheme ring.  The HVC ring is unaffected and uses the native
//! Montgomery backend in `poly.rs`.
//!
//! Pipeline for one multiplication:
//!
//! 1. Centre-lift each coefficient of `a, b` from `[0, q)` to `(-q/2, q/2]`.
//! 2. Reduce mod each auxiliary prime.
//! 3. Forward negacyclic NTT under each prime.
//! 4. Pointwise multiply.
//! 5. Inverse negacyclic NTT.
//! 6. Garner-CRT to the centred integer in `(-P/2, P/2]`.
//! 7. Reduce mod `q`.
//!
//! The auxiliary primes have the form `2^48 - c` with `c ∈ {16383, 19967}`.
//! A later C / asm port can reduce via the Mersenne-style identity
//! `x mod p ≡ (hi·c + lo) mod p` for 96-bit `x = hi·2^48 + lo`.  In Rust
//! we rely on `u128` multiplies and native `%`, which the compiler can
//! specialize when the prime is a compile-time constant.

// ---- constants -----------------------------------------------------------

/// First auxiliary prime: `2^48 - 16383`.
pub const AUX_P1: u64 = 281_474_976_694_273;
/// Second auxiliary prime: `2^48 - 19967`.
pub const AUX_P2: u64 = 281_474_976_690_689;

pub const AUX_MERSENNE_C1: u64 = 16_383;
pub const AUX_MERSENNE_C2: u64 = 19_967;

/// Supported ring dimensions for the NTT fast path.
pub const SUPPORTED_D: &[usize] = &[128, 256];

// Product of the two auxiliary primes as a `u128`.
const AUX_P: u128 = AUX_P1 as u128 * AUX_P2 as u128;

// ---- compile-time specialisation on the auxiliary prime -----------------
//
// The butterfly inner loop is the single hottest piece of code in the
// crate.  Making `P` (and the Mersenne constant `C = 2^48 − P`) compile-
// time constants lets the compiler specialise `reduce_pseudo_mersenne`
// into a pair of constant multiplies, fold all `% p` into conditional
// subtracts, and unroll / vectorise the butterfly.
//
// Concrete implementors: [`P1Marker`] and [`P2Marker`].

/// Compile-time parameters for a specialised auxiliary NTT.
pub trait AuxPrime {
    const P: u64;
    const C: u64;
}

/// Marker type representing `AUX_P1`.
#[derive(Clone, Copy)]
pub enum P1Marker {}
impl AuxPrime for P1Marker {
    const P: u64 = AUX_P1;
    const C: u64 = AUX_MERSENNE_C1;
}

/// Marker type representing `AUX_P2`.
#[derive(Clone, Copy)]
pub enum P2Marker {}
impl AuxPrime for P2Marker {
    const P: u64 = AUX_P2;
    const C: u64 = AUX_MERSENNE_C2;
}

/// Monomorphic `mul_mod` — equivalent to `mul_mod(a, b, T::P)` but with
/// the Mersenne constants folded at compile time.
#[inline(always)]
fn mul_mod_aux<T: AuxPrime>(a: u64, b: u64) -> u64 {
    debug_assert!(a < T::P && b < T::P);
    reduce_pseudo_mersenne(a as u128 * b as u128, T::C, T::P)
}

/// Constant-time conditional subtract for `x < 2p`, returning `x mod p`.
///
/// The signed-mask idiom below casts both `x` and `p` to `i64`; callers
/// must keep `x < 2^63` so the sign bit of `x` still represents a
/// non-negative integer.  For the usual post-add contract `x < 2p`,
/// `p < 2^62` is a simple sufficient bound.  The shipped auxiliary
/// primes are near `2^48`, with ample slack.
#[inline(always)]
fn ct_reduce_once(x: u64, p: u64) -> u64 {
    debug_assert!(p < (1u64 << 62));
    debug_assert!(x < (1u64 << 63));
    debug_assert!(x < 2 * p);
    let diff = (x as i64).wrapping_sub(p as i64);
    let mask = diff >> 63;
    diff.wrapping_add(mask & (p as i64)) as u64
}

#[inline(always)]
fn add_mod_aux<T: AuxPrime>(a: u64, b: u64) -> u64 {
    debug_assert!(a < T::P && b < T::P);
    let s = a + b;
    ct_reduce_once(s, T::P)
}

#[inline(always)]
fn sub_mod_aux<T: AuxPrime>(a: u64, b: u64) -> u64 {
    debug_assert!(a < T::P && b < T::P);
    let diff = (a as i64).wrapping_sub(b as i64);
    let mask = diff >> 63;
    diff.wrapping_add(mask & (T::P as i64)) as u64
}

// ---- primitive root search ----------------------------------------------

/// Return the smallest `x ∈ [2, p)` with `x^d ≡ -1 (mod p)`, i.e. exact
/// multiplicative order `2d`.  Requires `2d | p - 1`.
pub fn find_primitive_2d_root(p: u64, d: usize) -> Option<u64> {
    let two_d = (2 * d) as u64;
    if !(p - 1).is_multiple_of(two_d) {
        return None;
    }
    let exponent = (p - 1) / two_d;
    for x in 2..p {
        let psi = pow_mod(x, exponent, p);
        if pow_mod(psi, d as u64, p) == p - 1 {
            return Some(psi);
        }
    }
    None
}

/// `base^exp mod p` using 128-bit multiplies.  Constant-time in `exp`
/// only up to the bit-length; not meant for secret exponents.
#[inline]
pub fn pow_mod(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut acc: u64 = 1;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, p);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod(base, base, p);
        }
    }
    acc
}

/// Modular inverse via Fermat (p is prime).
#[inline]
pub fn inv_mod(a: u64, p: u64) -> u64 {
    pow_mod(a, p - 2, p)
}

/// `(a * b) mod p` via `u128`.  Both `a` and `b` assumed `< p < 2^63`.
#[inline]
pub fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    debug_assert!(a < p && b < p);
    match p {
        AUX_P1 => reduce_pseudo_mersenne(a as u128 * b as u128, AUX_MERSENNE_C1, AUX_P1),
        AUX_P2 => reduce_pseudo_mersenne(a as u128 * b as u128, AUX_MERSENNE_C2, AUX_P2),
        _ => ((a as u128 * b as u128) % p as u128) as u64,
    }
}

#[inline]
fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    debug_assert!(a < p && b < p);
    let s = a + b;
    ct_reduce_once(s, p)
}

#[inline]
fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    debug_assert!(a < p && b < p);
    let diff = (a as i64).wrapping_sub(b as i64);
    let mask = diff >> 63;
    diff.wrapping_add(mask & (p as i64)) as u64
}

#[inline]
fn reduce_pseudo_mersenne(x: u128, c: u64, p: u64) -> u64 {
    const MASK48: u128 = (1u128 << 48) - 1;
    let c128 = c as u128;
    let mut r = (x & MASK48) + c128 * (x >> 48);
    r = (r & MASK48) + c128 * (r >> 48);
    let out = r as u64;
    debug_assert!(out < 2 * p);
    ct_reduce_once(out, p)
}

// ---- bit-reversed twiddle table (FIPS 204 convention) ------------------

#[inline]
fn bitrev(mut k: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (k & 1);
        k >>= 1;
    }
    r
}

/// `zetas[k] = psi^bitrev(k, log2 d) mod p` for `k ∈ [0, d)`.
pub fn make_zeta_table(p: u64, d: usize, psi: u64) -> Vec<u64> {
    let bits = d.trailing_zeros();
    (0..d)
        .map(|k| pow_mod(psi, bitrev(k, bits) as u64, p))
        .collect()
}

// ---- radix-2 negacyclic NTT (FIPS 204 Algs. 41 / 42) -------------------

/// Forward negacyclic NTT.  Input in natural order; output in bit-reversed
/// order; values in `[0, p)`.
pub fn ntt(f: &mut [u64], zetas: &[u64], p: u64) {
    let d = f.len();
    let mut m = 0usize;
    let mut le = d / 2;
    while le >= 1 {
        let mut st = 0usize;
        while st < d {
            m += 1;
            let z = zetas[m];
            for j in st..st + le {
                let t = mul_mod(z, f[j + le], p);
                let fj = f[j];
                f[j + le] = sub_mod(fj, t, p);
                f[j] = add_mod(fj, t, p);
            }
            st += 2 * le;
        }
        le /= 2;
    }
}

/// Inverse negacyclic NTT.  Input bit-reversed; output natural order.
/// Includes the trailing `1/d` scale.
pub fn intt(f: &mut [u64], zetas: &[u64], p: u64) {
    let d = f.len();
    let mut m = d;
    let mut le = 1usize;
    while le < d {
        let mut st = 0usize;
        while st < d {
            m -= 1;
            let z = (p - zetas[m]) % p; // -zetas[m] mod p
            for j in st..st + le {
                let t = f[j];
                let u = f[j + le];
                f[j] = add_mod(t, u, p);
                f[j + le] = mul_mod(z, sub_mod(t, u, p), p);
            }
            st += 2 * le;
        }
        le *= 2;
    }
    let inv_d = inv_mod(d as u64, p);
    for x in f.iter_mut() {
        *x = mul_mod(*x, inv_d, p);
    }
}

// ---- monomorphic NTT butterflies ----------------------------------------

/// Forward negacyclic NTT, specialised on `T`.  Equivalent to
/// `ntt(f, zetas, T::P)` but with every `mul_mod` / `add_mod` /
/// `sub_mod` folded at compile time.
pub fn ntt_aux<T: AuxPrime>(f: &mut [u64], zetas: &[u64]) {
    let d = f.len();
    let mut m = 0usize;
    let mut le = d / 2;
    while le >= 1 {
        let mut st = 0usize;
        while st < d {
            m += 1;
            let z = zetas[m];
            for j in st..st + le {
                let t = mul_mod_aux::<T>(z, f[j + le]);
                let fj = f[j];
                f[j + le] = sub_mod_aux::<T>(fj, t);
                f[j] = add_mod_aux::<T>(fj, t);
            }
            st += 2 * le;
        }
        le /= 2;
    }
}

/// Inverse negacyclic NTT, specialised on `T`.  Includes the trailing
/// `1/d` scale.
fn intt_aux_scaled<T: AuxPrime>(f: &mut [u64], zetas: &[u64], inv_d: u64) {
    debug_assert!(inv_d < T::P);
    let d = f.len();
    let mut m = d;
    let mut le = 1usize;
    while le < d {
        let mut st = 0usize;
        while st < d {
            m -= 1;
            // -zetas[m] mod P.  zetas[m] is always in (0, T::P) for a
            // valid twiddle table, but we guard the 0 case defensively.
            let z = if zetas[m] == 0 { 0 } else { T::P - zetas[m] };
            for j in st..st + le {
                let t = f[j];
                let u = f[j + le];
                f[j] = add_mod_aux::<T>(t, u);
                f[j + le] = mul_mod_aux::<T>(z, sub_mod_aux::<T>(t, u));
            }
            st += 2 * le;
        }
        le *= 2;
    }
    for x in f.iter_mut() {
        *x = mul_mod_aux::<T>(*x, inv_d);
    }
}

pub fn intt_aux<T: AuxPrime>(f: &mut [u64], zetas: &[u64]) {
    let inv_d = inv_mod(f.len() as u64, T::P);
    intt_aux_scaled::<T>(f, zetas, inv_d);
}

// ---- per-prime context --------------------------------------------------

#[derive(Clone)]
struct AuxContext<T: AuxPrime> {
    zetas: Vec<u64>,
    inv_d: u64,
    _phantom: core::marker::PhantomData<fn() -> T>,
}

impl<T: AuxPrime> AuxContext<T> {
    fn new(d: usize) -> Self {
        let psi = find_primitive_2d_root(T::P, d).expect("aux prime does not support this d");
        Self {
            zetas: make_zeta_table(T::P, d, psi),
            inv_d: inv_mod(d as u64, T::P),
            _phantom: core::marker::PhantomData,
        }
    }

    fn forward(&self, input: &[u64]) -> Vec<u64> {
        let mut out = input.to_vec();
        ntt_aux::<T>(&mut out, &self.zetas);
        out
    }

    fn inverse_in_place(&self, f: &mut [u64]) {
        intt_aux_scaled::<T>(f, &self.zetas, self.inv_d);
    }
}

// ---- CRT backend --------------------------------------------------------

/// Exact negacyclic multiplication in `R_q = Z_q[X]/(X^d + 1)`.
#[derive(Clone)]
pub struct CrtBackend {
    q: u64,
    d: usize,
    half_q: u64,
    ctx1: AuxContext<P1Marker>,
    ctx2: AuxContext<P2Marker>,
    p1_inv_mod_p2: u64,
    max_accum_terms: usize,
}

impl CrtBackend {
    /// Build a backend for `R_q` at dimension `d`, sized for a single
    /// negacyclic product (no accumulation beyond the convolution sum).
    ///
    /// Equivalent to `Self::new_for_accum(q, d, 1)`; see that function
    /// for the full bound.  Use [`Self::new_for_accum`] directly when
    /// you intend to sum multiple products in the paired NTT domain
    /// before [`Self::finalize_accum_slices`] (typical for matrix
    /// multiplications: each output coefficient sums `s` product
    /// coefficients, where `s` is the inner dimension).
    pub fn new(q: u64, d: usize) -> Option<Self> {
        Self::new_for_accum(q, d, 1)
    }

    /// Build a backend for `R_q` at dimension `d`, sized to support
    /// CRT-exact reconstruction after accumulating up to `terms`
    /// per-coefficient products in each auxiliary prime's NTT domain.
    ///
    /// Returns `None` if
    ///
    /// * `d` is not in [`SUPPORTED_D`], or
    /// * the worst-case post-accumulation coefficient
    ///   `terms · d · ((q-1)/2)^2` would equal or exceed `P/2` (where
    ///   `P = p1 · p2 ≈ 2^96`).  In that case CRT cannot recover the
    ///   exact signed integer coefficient and the backend would
    ///   silently produce wrong results for worst-case operands.
    /// * `terms == 0` (no work to do is treated as a misuse).
    ///
    /// The bound is the natural extension of the single-product check:
    /// each negacyclic product has `|c_k| <= d · ((q-1)/2)^2`, and
    /// summing `terms` such products preserves the centred sign so
    /// `|sum| <= terms · d · ((q-1)/2)^2`.  We require
    /// `2 · |sum| < P` so that the centred reconstruction in
    /// `(-P/2, P/2]` is unambiguous.
    pub fn new_for_accum(q: u64, d: usize, terms: usize) -> Option<Self> {
        if terms == 0 {
            return None;
        }
        if !SUPPORTED_D.contains(&d) {
            return None;
        }
        let half_q: u128 = ((q - 1) / 2) as u128;
        let per_product = (d as u128).checked_mul(half_q.checked_mul(half_q)?)?;
        let bound = per_product.checked_mul(terms as u128)?;
        if bound >= AUX_P / 2 {
            return None;
        }
        // `accum_mul_slices` keeps small matrix-product accumulations as
        // raw sums in `u64` and normalizes once before the inverse NTT.
        // Each added product is `< p`; refuse term counts that could
        // overflow that lazy accumulator even if the CRT exactness bound
        // itself would be satisfied for a tiny scheme modulus.  Checking
        // `AUX_P1 - 1` covers both accumulators because `AUX_P1 > AUX_P2`.
        let lazy_bound = (terms as u128).checked_mul((AUX_P1 - 1) as u128)?;
        if lazy_bound >= u64::MAX as u128 {
            return None;
        }
        Some(Self {
            q,
            d,
            half_q: q / 2,
            ctx1: AuxContext::<P1Marker>::new(d),
            ctx2: AuxContext::<P2Marker>::new(d),
            p1_inv_mod_p2: inv_mod(AUX_P1, AUX_P2),
            max_accum_terms: terms,
        })
    }

    /// Ring modulus q.
    #[inline]
    pub fn q(&self) -> u64 {
        self.q
    }

    /// Ring dimension d.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// `a * b` mod `X^d + 1`, mod `q`.  Inputs and output in `[0, q)`.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        assert_eq!(a.len(), self.d);
        assert_eq!(b.len(), self.d);

        // centre-lift and reduce mod each auxiliary prime
        let (a1, a2) = self.split(a);
        let (b1, b2) = self.split(b);

        let mut c1 =
            mul_pointwise_aux::<P1Marker>(&self.ctx1.forward(&a1), &self.ctx1.forward(&b1));
        self.ctx1.inverse_in_place(&mut c1);

        let mut c2 =
            mul_pointwise_aux::<P2Marker>(&self.ctx2.forward(&a2), &self.ctx2.forward(&b2));
        self.ctx2.inverse_in_place(&mut c2);

        self.crt_combine(&c1, &c2)
    }

    // ---- Pre-NTT API: cache forward transforms for reuse ---------------
    //
    // For matrix multiplications like `T = X * A2` we forward-transform each
    // operand once across both aux primes, accumulate pointwise products in
    // each aux prime's NTT domain, then single-shot the two inverse NTTs and
    // the CRT combine per output polynomial.  Cuts ~2 NTTs off the per-mul
    // cost vs [`Self::mul`], which matters when `A2` is reused across all
    // HVC leaves.  The API is slice-based so callers can pack many cached
    // NTTs into flat `Vec<u64>` buffers.

    /// Forward-transform a signed coefficient-form polynomial (values in i64,
    /// centred or unsigned) into the aux-prime NTT pair, written into caller-
    /// supplied slices of length `d`.
    pub fn forward_into_i64(&self, a: &[i64], out_p1: &mut [u64], out_p2: &mut [u64]) {
        debug_assert_eq!(a.len(), self.d);
        debug_assert_eq!(out_p1.len(), self.d);
        debug_assert_eq!(out_p2.len(), self.d);
        let q = self.q as i128;
        for i in 0..self.d {
            // bring into canonical unsigned [0, q), then centre-lift
            let u = (a[i] as i128).rem_euclid(q);
            let cc: i128 = if u > (self.half_q as i128) { u - q } else { u };
            out_p1[i] = reduce_i128(cc, AUX_P1);
            out_p2[i] = reduce_i128(cc, AUX_P2);
        }
        ntt_aux::<P1Marker>(out_p1, &self.ctx1.zetas);
        ntt_aux::<P2Marker>(out_p2, &self.ctx2.zetas);
    }

    /// Convenience: forward-transform an i64 polynomial, allocating a fresh
    /// pair.  Used by tests; production hot paths use
    /// [`Self::forward_into_i64`] against a preallocated flat buffer.
    pub fn forward_pair_i64(&self, a: &[i64]) -> (Vec<u64>, Vec<u64>) {
        let mut p1 = vec![0u64; self.d];
        let mut p2 = vec![0u64; self.d];
        self.forward_into_i64(a, &mut p1, &mut p2);
        (p1, p2)
    }

    /// MAC into paired NTT-domain accumulators.  Each argument is a length-d
    /// slice in the corresponding aux prime's NTT domain.  The accumulators
    /// are lazy raw sums, not reduced after each term; call
    /// [`Self::finalize_accum_slices`] after at most the `terms` count used
    /// to construct this backend via [`Self::new_for_accum`].
    ///
    /// Callers must initialize both accumulators to zero before the first
    /// call for an output polynomial.  Continuing a partial sum is also
    /// valid, but the total number of accumulated product terms since the
    /// zeroing point must not exceed the constructor's `terms` bound.
    #[inline]
    pub fn accum_mul_slices(
        &self,
        acc_p1: &mut [u64],
        acc_p2: &mut [u64],
        a_p1: &[u64],
        a_p2: &[u64],
        b_p1: &[u64],
        b_p2: &[u64],
    ) {
        let d = self.d;
        debug_assert_eq!(acc_p1.len(), d);
        debug_assert_eq!(acc_p2.len(), d);
        debug_assert_eq!(a_p1.len(), d);
        debug_assert_eq!(a_p2.len(), d);
        debug_assert_eq!(b_p1.len(), d);
        debug_assert_eq!(b_p2.len(), d);
        let max_p1 = (self.max_accum_terms as u128) * ((AUX_P1 - 1) as u128);
        let max_p2 = (self.max_accum_terms as u128) * ((AUX_P2 - 1) as u128);
        for i in 0..d {
            let prod1 = mul_mod_aux::<P1Marker>(a_p1[i], b_p1[i]);
            let prod2 = mul_mod_aux::<P2Marker>(a_p2[i], b_p2[i]);
            debug_assert!((acc_p1[i] as u128) + (prod1 as u128) <= max_p1);
            debug_assert!((acc_p2[i] as u128) + (prod2 as u128) <= max_p2);
            acc_p1[i] += prod1;
            acc_p2[i] += prod2;
        }
    }

    /// Inverse-transform a paired NTT-domain accumulator and CRT-combine
    /// into a coefficient-form polynomial in [0, q).  The accumulator
    /// buffers are modified in place during the inverse transform.
    pub fn finalize_accum_slices(&self, acc_p1: &mut [u64], acc_p2: &mut [u64]) -> Vec<u64> {
        debug_assert_eq!(acc_p1.len(), self.d);
        debug_assert_eq!(acc_p2.len(), self.d);
        for x in acc_p1.iter_mut() {
            *x = reduce_pseudo_mersenne(*x as u128, AUX_MERSENNE_C1, AUX_P1);
        }
        for x in acc_p2.iter_mut() {
            *x = reduce_pseudo_mersenne(*x as u128, AUX_MERSENNE_C2, AUX_P2);
        }
        self.ctx1.inverse_in_place(acc_p1);
        self.ctx2.inverse_in_place(acc_p2);
        self.crt_combine(acc_p1, acc_p2)
    }

    /// Inverse-transform a paired NTT-domain accumulator and CRT-combine
    /// into the exact centred signed integer coefficients, without reducing
    /// them modulo `q`.
    ///
    /// This is for aggregating KOTS signatures over `ZZ`: individual
    /// `w_i * z_i` products are small, but the final aggregate may exceed
    /// `q'/2` and must remain the exact signed sum for the `beta_sigma`
    /// verification bound.  `new_for_accum(q, d, terms)` has already checked
    /// that the worst-case coefficient is below `P/2`, so the centred CRT
    /// lift is unambiguous.
    pub fn finalize_accum_slices_signed_i64(
        &self,
        acc_p1: &mut [u64],
        acc_p2: &mut [u64],
    ) -> Vec<i64> {
        debug_assert_eq!(acc_p1.len(), self.d);
        debug_assert_eq!(acc_p2.len(), self.d);
        for x in acc_p1.iter_mut() {
            *x = reduce_pseudo_mersenne(*x as u128, AUX_MERSENNE_C1, AUX_P1);
        }
        for x in acc_p2.iter_mut() {
            *x = reduce_pseudo_mersenne(*x as u128, AUX_MERSENNE_C2, AUX_P2);
        }
        self.ctx1.inverse_in_place(acc_p1);
        self.ctx2.inverse_in_place(acc_p2);
        self.crt_combine_signed_i64(acc_p1, acc_p2)
    }

    fn split(&self, a: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let (q, hq) = (self.q, self.half_q);
        let mut a1 = Vec::with_capacity(self.d);
        let mut a2 = Vec::with_capacity(self.d);
        for &c in a {
            // centred integer in  [ -(q-1)/2 , (q-1)/2 ]  stored as i128
            let cc: i128 = if c > hq {
                c as i128 - q as i128
            } else {
                c as i128
            };
            a1.push(reduce_i128(cc, AUX_P1));
            a2.push(reduce_i128(cc, AUX_P2));
        }
        (a1, a2)
    }

    #[inline]
    fn crt_centered_coeff(&self, r1: u64, r2: u64) -> i128 {
        let p1 = AUX_P1;
        let p2 = AUX_P2;
        let inv = self.p1_inv_mod_p2;
        let half_p: i128 = (AUX_P / 2) as i128;
        let big_p: i128 = AUX_P as i128;

        // t = (r2 - r1) * inv mod p2
        let diff: u64 = sub_mod_aux::<P2Marker>(r2, r1 % p2);
        let t = mul_mod_aux::<P2Marker>(diff, inv);
        // x = r1 + t * p1  in  [0, P)
        let x: u128 = r1 as u128 + t as u128 * p1 as u128;
        let mut xi: i128 = x as i128;
        if xi > half_p {
            xi -= big_p;
        }
        xi
    }

    fn crt_combine(&self, c1: &[u64], c2: &[u64]) -> Vec<u64> {
        let q_i128 = self.q as i128;
        let mut out = Vec::with_capacity(self.d);
        for i in 0..self.d {
            let xi = self.crt_centered_coeff(c1[i], c2[i]);
            let mut y = xi % q_i128;
            if y < 0 {
                y += q_i128;
            }
            out.push(y as u64);
        }
        out
    }

    fn crt_combine_signed_i64(&self, c1: &[u64], c2: &[u64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.d);
        for i in 0..self.d {
            let xi = self.crt_centered_coeff(c1[i], c2[i]);
            out.push(i64::try_from(xi).expect("centred CRT coefficient exceeds i64"));
        }
        out
    }
}

/// Reduce a centred integer `x ∈ i128` mod `p` (returns `[0, p)`).
#[inline]
fn reduce_i128(x: i128, p: u64) -> u64 {
    let p_i = p as i128;
    let r = x % p_i;
    if r < 0 {
        (r + p_i) as u64
    } else {
        r as u64
    }
}

#[inline]
fn mul_pointwise_aux<T: AuxPrime>(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| mul_mod_aux::<T>(x, y))
        .collect()
}

// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn schoolbook(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
        let d = a.len();
        let mut c = vec![0i128; d];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = (ai as i128) * (bj as i128);
                let k = i + j;
                if k < d {
                    c[k] += prod;
                } else {
                    c[k - d] -= prod;
                }
            }
        }
        c.into_iter()
            .map(|v| {
                let m = v.rem_euclid(q as i128);
                m as u64
            })
            .collect()
    }

    fn rng(seed: u64) -> impl FnMut(u64) -> u64 {
        // Linear-congruential — fine for test fixtures, not crypto.
        let mut s = seed;
        move |bound: u64| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 16) % bound
        }
    }

    #[test]
    fn aux_primes_support_d256() {
        for &p in &[AUX_P1, AUX_P2] {
            assert_eq!((p - 1) % 512, 0);
        }
    }

    #[test]
    fn primitive_root_has_exact_order_2d() {
        for &p in &[AUX_P1, AUX_P2] {
            for &d in &[128, 256] {
                let psi = find_primitive_2d_root(p, d).unwrap();
                assert_eq!(pow_mod(psi, d as u64, p), p - 1);
                assert_eq!(pow_mod(psi, 2 * d as u64, p), 1);
            }
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        for &p in &[AUX_P1, AUX_P2] {
            let d = 128usize;
            let psi = find_primitive_2d_root(p, d).unwrap();
            let zetas = make_zeta_table(p, d, psi);
            let mut r = rng(0xABCDEF01);
            let original: Vec<u64> = (0..d).map(|_| r(p)).collect();
            let mut f = original.clone();
            ntt(&mut f, &zetas, p);
            intt(&mut f, &zetas, p);
            assert_eq!(f, original);
        }
    }

    // Shipped KOTS modulus (q' ≡ 17 mod 32) is not NTT-friendly on
    // its own ring, so it routes through this CRT backend.
    const KOTS_PARAMS: &[(u64, usize)] = &[(867_354_289, 256)];

    #[test]
    fn crt_mul_matches_schoolbook() {
        for &(q, d) in KOTS_PARAMS {
            let backend = CrtBackend::new(q, d).unwrap();
            let mut r = rng(0x1234_5678);
            for _ in 0..5 {
                let a: Vec<u64> = (0..d).map(|_| r(q)).collect();
                let b: Vec<u64> = (0..d).map(|_| r(q)).collect();
                let want = schoolbook(&a, &b, q);
                let got = backend.mul(&a, &b);
                assert_eq!(got, want, "d={}, q={}", d, q);
            }
        }
    }

    #[test]
    fn crt_backend_rejects_q_beyond_reconstruction_bound() {
        // q near 2^48 at d = 256 blows the CRT bound: 256 * (2^47)^2 = 2^102
        // vastly exceeds P/2 ≈ 2^95.  Construction must refuse.
        assert!(CrtBackend::new(281_474_976_710_597, 256).is_none());
        // Same q at d = 128 also fails: 128 * (2^47)^2 = 2^101 > 2^95.
        assert!(CrtBackend::new(281_474_976_710_597, 128).is_none());
    }

    #[test]
    fn crt_backend_accepts_kots_moduli() {
        for &(q, d) in KOTS_PARAMS {
            assert!(
                CrtBackend::new(q, d).is_some(),
                "KOTS q={q}, d={d} must satisfy the CRT bound",
            );
        }
    }

    /// KOTS accumulation depth check: every backend constructor used
    /// inside the KOTS hot paths must satisfy the post-accumulation bound
    /// `terms · d · ((q-1)/2)^2 < P/2`.  Encodes the largest term counts
    /// that the shipped parameter set drives into the CRT backend:
    ///
    /// * `matmul_structured_a`: `terms = m - n = 9 - 4 = 5`.
    /// * `mat_mul_h` for sign / verify: `terms = k - 1 = 4 - 1 = 3`.
    ///
    /// Exercised at `terms = 16` as a comfortable envelope.
    #[test]
    fn crt_backend_accepts_kots_accumulation_depth() {
        for &(q, d) in KOTS_PARAMS {
            for &terms in &[1usize, 4, 8, 16] {
                assert!(
                    CrtBackend::new_for_accum(q, d, terms).is_some(),
                    "KOTS q={q}, d={d}, terms={terms} must satisfy the CRT bound",
                );
            }
        }
    }

    /// `terms == 0` is treated as misuse — there is no work to size for.
    #[test]
    fn crt_backend_rejects_zero_terms() {
        let (q, d) = KOTS_PARAMS[0];
        assert!(CrtBackend::new_for_accum(q, d, 0).is_none());
    }

    /// The accumulation constructor must reject configurations that would
    /// either silently overflow the CRT reconstruction or overflow the raw
    /// `u64` lazy accumulator used before final normalization.
    #[test]
    fn crt_backend_rejects_terms_that_blow_the_bound() {
        for &(q, d) in KOTS_PARAMS {
            let half_q: u128 = ((q - 1) / 2) as u128;
            let per_product = (d as u128) * half_q * half_q;
            // Largest accepted `terms` satisfies terms * per_product < AUX_P / 2.
            let max_crt_terms = ((AUX_P / 2 - 1) / per_product) as usize;
            let max_lazy_terms = ((u64::MAX as u128 - 1) / ((AUX_P1 - 1) as u128)) as usize;
            let max_terms = max_crt_terms.min(max_lazy_terms);
            assert!(
                CrtBackend::new_for_accum(q, d, max_terms).is_some(),
                "max_terms={max_terms} should still fit for q={q}, d={d}",
            );
            assert!(
                CrtBackend::new_for_accum(q, d, max_terms + 1).is_none(),
                "max_terms+1={} should be refused for q={q}, d={d}",
                max_terms + 1,
            );
        }
    }

    #[test]
    fn crt_mul_edge_cases() {
        // X * X^{d-1} = -1 in R_q = Z_q[X]/(X^d + 1).
        for &(q, d) in KOTS_PARAMS {
            let backend = CrtBackend::new(q, d).unwrap();
            let mut x = vec![0u64; d];
            x[1] = 1; // X
            let mut x_last = vec![0u64; d];
            x_last[d - 1] = 1; // X^{d-1}
            let prod = backend.mul(&x, &x_last);
            let mut expected = vec![0u64; d];
            expected[0] = q - 1;
            assert_eq!(prod, expected, "d={}, q={}", d, q);
        }
    }

    #[test]
    fn kots_primes_satisfy_proof_condition() {
        // KOTS proof condition: q' ≡ 2t+1 (mod 4t) with t = 8,
        // i.e. q' ≡ 17 (mod 32).
        for &(q, _d) in KOTS_PARAMS {
            assert_eq!(q % 32, 17, "q={q} violates q' ≡ 17 mod 32");
        }
    }
}
