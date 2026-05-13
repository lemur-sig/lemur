//! Sampler constants.
//!
//! Parameter-set values live in `src/profile.rs`.  This module only
//! holds the discrete-Gaussian sampler precision constants, which are
//! pinned by the Rényi-divergence analysis at λ=128.

/// CDT precision for the discrete Gaussian sampler.
///
/// 32 bits byte-aligned = 31-bit CDF comparison + 1 sign bit (LSB).
/// For the shipped `D256_K4` profile,
/// `parameter/Lemur-DGS-Prec_TailCut.py` computes `prec_re = 28`, so
/// the implementation keeps a 3-bit comparison margin from the
/// discrete-Gaussian Rényi-divergence analysis at λ = 128.
pub const GAUSS_CDT_BITS: usize = 32;

/// Gaussian sampler precision in bytes.
pub const GAUSS_CDT_BYTES: usize = GAUSS_CDT_BITS / 8;

/// Tailcut parameter as a multiple of sigma (`tc_re = 5`).
pub const GAUSS_TAILCUT: usize = 5;

/// Lambda (security parameter) in bits.
pub const SECURITY_BITS: usize = 128;
