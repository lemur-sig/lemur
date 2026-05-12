//! Runtime `GaussCtx` / `build_cdt` integration tests.

use lemur_rs::kots::{kots_ivrfy, kots_keygen, kots_keygen_ctx, kots_setup, kots_sign};
use lemur_rs::params::{GAUSS_CDT_BITS, GAUSS_TAILCUT};
use lemur_rs::profile::DEFAULT;
use lemur_rs::sample::{
    build_cdt, kots_keygen_xof, xof_gauss_poly_ctx, xof_gauss_poly_ctx_into, GaussCtx,
};

/// Runtime-built CDT at the default profile's params must agree with the
/// baked table closely enough that no binary-search boundary flips for
/// any u.
///
/// Using an f64 accumulator vs mpmath for the baked table can differ by
/// a small ULP count at the low-probability tail.  We accept the runtime
/// table iff every threshold lies in the same "cell" as the baked one,
/// i.e. the max absolute difference is bounded.
#[test]
fn runtime_cdt_matches_baked_default() {
    let baked = DEFAULT.cdt;
    let built = build_cdt(DEFAULT.sigma, DEFAULT.cdt_bits, DEFAULT.tailcut);
    assert_eq!(
        built.len(),
        baked.len(),
        "runtime CDT length {} does not match baked {}",
        built.len(),
        baked.len()
    );
    let max_diff: u64 = built
        .iter()
        .zip(baked.iter())
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .max()
        .unwrap_or(0);
    // f64 mantissa has 53 bits, so at cdt_bits=32 an ULP is ~2^-21 of the
    // target.  Empirically the drift is within a handful of units.
    assert!(
        max_diff < 1 << 10,
        "runtime CDT drifts by {max_diff} vs baked table (budget 1024)"
    );
}

/// KOTS keygen via `_ctx` with the profile's CDT must match plain `kots_keygen`.
#[test]
fn kots_keygen_ctx_default_matches_kots_keygen() {
    let a = kots_setup(&[7u8; 32], DEFAULT);
    let (sk_a, pk_a) = kots_keygen(&a, &[9u8; 32], DEFAULT);
    let ctx = GaussCtx::from_profile(DEFAULT);
    let (sk_b, pk_b) = kots_keygen_ctx(&a, &[9u8; 32], DEFAULT, &ctx);
    assert_eq!(sk_a.0, sk_b.0, "sk differs between kots_keygen and _ctx");
    assert_eq!(pk_a.0, pk_b.0, "pk differs between kots_keygen and _ctx");
}

/// A custom narrower CDT must still produce a keygen that passes KOTS
/// verification (the sampled S is smaller-variance, but the same
/// matrix-multiply + norm-bound machinery applies).
#[test]
fn kots_keygen_ctx_narrower_cdt_still_verifies() {
    let a = kots_setup(&[11u8; 32], DEFAULT);
    let cdt = build_cdt(DEFAULT.sigma / 2.0, 16, GAUSS_TAILCUT);
    let ctx = GaussCtx {
        cdt: &cdt,
        cdt_bytes: 2,
    };
    let (sk, pk) = kots_keygen_ctx(&a, &[13u8; 32], DEFAULT, &ctx);
    let msg = b"narrower cdt test";
    let sig = kots_sign(&sk, msg, DEFAULT);
    kots_ivrfy(&a, &pk, msg, &sig, DEFAULT).expect("narrower-CDT sig must verify");
}

#[test]
fn gauss_poly_ctx_into_matches_allocating_variant() {
    let ctx = GaussCtx::from_profile(DEFAULT);
    let mut xof_a = kots_keygen_xof(&[31u8; 32], 0, 0);
    let mut xof_b = kots_keygen_xof(&[31u8; 32], 0, 0);
    let poly_a = xof_gauss_poly_ctx(&mut xof_a, &ctx, DEFAULT.d);
    let mut poly_b = vec![0i64; DEFAULT.d];
    xof_gauss_poly_ctx_into(&mut xof_b, &ctx, &mut poly_b);
    assert_eq!(poly_a, poly_b, "_into sampler drifted from allocating path");
}

/// Sanity: the `GAUSS_CDT_BITS` module constant matches the parameter set.
#[test]
fn profile_cdt_bits_match_module_constant() {
    for p in lemur_rs::profile::all() {
        assert_eq!(p.cdt_bits, GAUSS_CDT_BITS);
    }
}
