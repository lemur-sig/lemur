//! Named Lemur parameter sets.
//!
//! Each [`Profile`] bundles every scalar and every table reference that
//! the scheme / sampler / codec layers need at runtime.  The fields
//! mirror `lemur-py/profiles.py` byte-for-byte; values originate from
//! `doc/Lemur Parameter Setting.ods` (representative cell τ=20, N=1024).
//!
//! # KOTS backend split
//!
//! The KOTS modulus is selected to satisfy the proof condition
//! `q' ≡ 17 (mod 32)`, which forbids a length-d negacyclic NTT.  All
//! shipped parameter set therefore routes KOTS multiplication through the
//! CRT-via-aux-primes backend in [`crate::aux_ntt`].  The struct still
//! carries `kots_ring`/`kots_ring64` slots so a future natively
//! NTT-friendly modulus can be wired in without changing every call
//! site, but every static here sets only `kots_crt`.
//!
//! # HVC backend
//!
//! HVC moduli always satisfy `q ≡ 1 (mod 2d)` and use the native
//! Montgomery NTT.  The shipped parameter set chooses q above 2³², so
//! [`HvcRing::U64`] is the active variant; the `HvcRing::U32` slot is
//! retained for future cells.

use crate::aux_ntt::CrtBackend;
use crate::poly::{RingParams, RingParams64};
use crate::tables_d256_k4 as t_d256_k4;

/// Minimal carrier for a CRT-backed KOTS ring.
///
/// The heavy state (twiddle tables, CRT constants) lives in the
/// [`CrtBackend`] that `kots.rs` builds on first use.  The profile only
/// needs to advertise the modulus and dimension so that dispatch sites
/// know to go through `aux_ntt` rather than the native Montgomery path.
#[derive(Clone, Copy)]
pub struct KotsCrtCfg {
    pub q: u64,
    pub d: usize,
}

/// HVC ring backend.
///
/// The shipped HVC q exceeds 2³² and uses the u64 Montgomery stack;
/// the `U32` slot exists so a future cell with a smaller HVC modulus
/// can be wired in without changing call sites.
#[derive(Clone)]
pub enum HvcRing {
    U32(RingParams),
    U64(RingParams64),
}

impl HvcRing {
    #[inline]
    pub fn q(&self) -> u64 {
        match self {
            Self::U32(rp) => rp.q,
            Self::U64(rp) => rp.q,
        }
    }

    #[inline]
    pub fn d(&self) -> usize {
        match self {
            Self::U32(rp) => rp.d,
            Self::U64(rp) => rp.d,
        }
    }

    #[inline]
    pub fn as_u32(&self) -> Option<&RingParams> {
        match self {
            Self::U32(rp) => Some(rp),
            Self::U64(_) => None,
        }
    }

    #[inline]
    pub fn as_u64(&self) -> Option<&RingParams64> {
        match self {
            Self::U32(_) => None,
            Self::U64(rp) => Some(rp),
        }
    }

    #[inline]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::U32(_) => "u32 Montgomery NTT",
            Self::U64(_) => "u64 Montgomery NTT",
        }
    }
}

impl KotsCrtCfg {
    /// Construct a `CrtBackend` sized for a single negacyclic product
    /// (no external accumulation).
    #[inline]
    pub fn backend(&self) -> CrtBackend {
        self.backend_for_accum(1)
    }

    /// Construct a `CrtBackend` sized to support summing up to `terms`
    /// per-coefficient products in the paired NTT domain before CRT
    /// reconstruction.  Panics on capacity violation — the profile
    /// declaration is expected to satisfy the bound.
    #[inline]
    pub fn backend_for_accum(&self, terms: usize) -> CrtBackend {
        CrtBackend::new_for_accum(self.q, self.d, terms).unwrap_or_else(|| {
            panic!(
                "KOTS CRT backend: q={}, d={}, terms={} exceeds CRT capacity (P/2)",
                self.q, self.d, terms,
            )
        })
    }
}

/// One concrete Lemur parameter set.
///
/// Mirrors `lemur-py/profiles.py::LemurProfile` field-by-field.
pub struct Profile {
    /// Human-readable parameter-set name.
    pub name: &'static str,

    // --- Common ------------------------------------------------------
    /// Ring dimension d.
    pub d: usize,
    /// Merkle-tree height tau (`n_slots = 1 << tau`).
    pub tau: usize,
    /// Number of signers the profile is sized for (capital N in the
    /// paper, `n_signers` in the ODS).
    pub n_signers: usize,

    // --- KOTS --------------------------------------------------------
    pub k: usize,
    pub ell: usize,
    pub m: usize,
    pub n: usize,
    /// Gaussian width parameter alpha (paper symbol; appears in
    /// codec bound derivations).
    pub alpha: f64,
    /// Sampling stddev sigma = alpha / sqrt(2*pi).
    pub sigma: f64,
    pub alpha_h: usize,
    pub beta_z: i64,
    pub beta_sigma: i64,
    /// KOTS ring (u32 Montgomery backend).  Unused by the shipped
    /// parameter set; reserved for future natively NTT-friendly moduli.
    pub kots_ring: Option<RingParams>,
    /// KOTS ring (u64 Montgomery backend).  Unused by the shipped
    /// parameter set; reserved for future natively NTT-friendly moduli.
    pub kots_ring64: Option<RingParams64>,
    /// KOTS ring (CRT-via-aux-primes backend).  Active under the proof
    /// condition `q' ≡ 17 (mod 32)`.
    pub kots_crt: Option<KotsCrtCfg>,

    // --- HVC ---------------------------------------------------------
    pub omega: usize,
    pub eta: i64,
    pub kappa: usize,
    pub kappa_prime: usize,
    pub beta_agg: i64,
    pub beta_encode: i64,
    /// Ring tables for the HVC modulus q.
    pub hvc_ring: HvcRing,

    // --- Aggregation -------------------------------------------------
    pub alpha_w: usize,
    pub gamma: usize,

    // --- Sampler -----------------------------------------------------
    /// CDT precision in bits (32: 31-bit CDF + 1-bit sign).
    pub cdt_bits: usize,
    /// Tailcut in multiples of sigma.
    pub tailcut: usize,
    /// CDT table for the Gaussian sampler at this profile's sigma.
    /// Length = `floor(tailcut * sigma) + 2`.
    pub cdt: &'static [u32],
    /// Optional bucket index over the high `cdt_prefix_bits` of a 32-bit
    /// CDT sample, used by the indexed-CDT sampler fast path.  When
    /// `None`, the sampler falls back to the generic CDT binary search.
    pub cdt_hi: Option<&'static [u16]>,
    /// Number of high bits of each 32-bit sample word that key into
    /// `cdt_hi`.  Ignored when `cdt_hi` is `None`.
    pub cdt_prefix_bits: u32,
}

impl Profile {
    /// Validate cross-field invariants that are easy to break when copying a
    /// spreadsheet cell into a static profile.
    ///
    /// Includes a CRT capacity check for every KOTS accumulation shape used
    /// by the implementation, so unsupported q'/shape combinations fail
    /// during setup/profile tests rather than deep inside a signing path.
    pub fn validate(&self) {
        assert_eq!(
            self.beta_encode,
            (self.beta_agg + 2 * self.eta - 1) / (2 * self.eta),
            "profile {:?}: beta_encode must be ceil(beta_agg / (2*eta))",
            self.name
        );
        assert_eq!(
            (self.q_hvc() - 1) % (2 * self.d as u64),
            0,
            "profile {:?}: HVC q must satisfy q ≡ 1 mod 2d",
            self.name
        );
        if let Some(cfg) = self.kots_crt() {
            assert_eq!(
                cfg.q % 32,
                17,
                "profile {:?}: CRT KOTS q' must satisfy q' ≡ 17 mod 32",
                self.name
            );
            assert_eq!(
                cfg.d, self.d,
                "profile {:?}: CRT KOTS d must match profile d",
                self.name
            );
            let max_terms = 1usize.max(self.m - self.n).max(self.k - self.ell);
            assert!(
                CrtBackend::new_for_accum(cfg.q, cfg.d, max_terms).is_some(),
                "profile {:?}: CRT KOTS q={}, d={}, terms={} exceeds capacity",
                self.name,
                cfg.q,
                cfg.d,
                max_terms
            );
        }
    }

    /// Borrow the u32-backed KOTS ring.  Panics with a descriptive
    /// message if the profile does not use that backend.
    #[inline]
    pub fn kots_ring_u32(&self) -> &RingParams {
        self.kots_ring.as_ref().unwrap_or_else(|| {
            panic!(
                "profile {:?} has no u32 KOTS ring (uses {} backend)",
                self.name,
                self.kots_backend_name()
            )
        })
    }

    /// Borrow the u64-backed KOTS ring.  Panics with a descriptive
    /// message if the profile does not use that backend.
    #[inline]
    pub fn kots_ring_u64(&self) -> &RingParams64 {
        self.kots_ring64.as_ref().unwrap_or_else(|| {
            panic!(
                "profile {:?} has no u64 KOTS ring (uses {} backend)",
                self.name,
                self.kots_backend_name()
            )
        })
    }

    /// Borrow the CRT-backed KOTS ring config.  Returns `None` only when
    /// a future profile wires up a native NTT backend instead.
    #[inline]
    pub fn kots_crt(&self) -> Option<&KotsCrtCfg> {
        self.kots_crt.as_ref()
    }

    /// Descriptive name of the active KOTS backend, used in panic
    /// messages.
    fn kots_backend_name(&self) -> &'static str {
        match (
            self.kots_ring.as_ref(),
            self.kots_ring64.as_ref(),
            self.kots_crt.as_ref(),
        ) {
            (Some(_), None, None) => "u32 Montgomery NTT",
            (None, Some(_), None) => "u64 Montgomery NTT",
            (None, None, Some(_)) => "CRT-via-aux-primes",
            _ => "(invalid: multiple backends set)",
        }
    }

    /// Return `q_kots` regardless of which backend the profile uses.
    #[inline]
    pub fn q_kots(&self) -> u64 {
        match (&self.kots_ring, &self.kots_ring64, &self.kots_crt) {
            (Some(rp), _, _) => rp.q,
            (_, Some(rp), _) => rp.q,
            (_, _, Some(cfg)) => cfg.q,
            _ => panic!("profile {:?} has no KOTS ring backend", self.name),
        }
    }

    #[inline]
    pub fn q_hvc(&self) -> u64 {
        self.hvc_ring.q()
    }
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

pub static D256_K4: Profile = Profile {
    name: "d256_k4",
    d: 256,
    tau: 20,
    n_signers: 1024,
    k: 4,
    ell: 1,
    m: 9,
    n: 4,
    alpha: 87.0,
    sigma: 34.707_978_394_924_645,
    alpha_h: 60,
    beta_z: 14_046,
    beta_sigma: 13_229_351,
    kots_ring: None,
    kots_ring64: None,
    kots_crt: Some(KotsCrtCfg {
        q: 3_469_416_721,
        d: 256,
    }),
    omega: 2,
    eta: 776,
    kappa: 5,
    kappa_prime: 3,
    beta_agg: 919_945,
    beta_encode: 593,
    hvc_ring: HvcRing::U64(RingParams64 {
        q: 9_007_199_254_746_113,
        q_inv: t_d256_k4::D256_K4_HVC_Q_INV,
        r2: t_d256_k4::D256_K4_HVC_R2,
        inv_d_mont: t_d256_k4::D256_K4_HVC_INV_D_MONT,
        d: 256,
        zetas: &t_d256_k4::D256_K4_HVC_ZETAS,
    }),
    alpha_w: 23,
    gamma: 10,
    cdt_bits: 32,
    tailcut: 5,
    cdt: &t_d256_k4::D256_K4_CDT,
    cdt_hi: Some(&t_d256_k4::D256_K4_CDT_HI),
    cdt_prefix_bits: t_d256_k4::D256_K4_CDT_PREFIX_BITS,
};

/// Default profile used wherever a function takes `&'static Profile` and
/// needs a sensible fallback.
pub static DEFAULT: &Profile = &D256_K4;

/// All shipped parameter sets, in canonical order.
pub fn all() -> [&'static Profile; 1] {
    [&D256_K4]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_set_validates() {
        for p in all() {
            p.validate();
        }
    }

    #[test]
    fn parameter_set_uses_crt_backend_and_satisfies_proof_condition() {
        for p in all() {
            assert!(
                p.kots_ring.is_none(),
                "{} must not carry u32 KOTS ring",
                p.name
            );
            assert!(
                p.kots_ring64.is_none(),
                "{} must not carry u64 KOTS ring",
                p.name
            );
            let cfg = p
                .kots_crt()
                .expect("shipped parameter set must carry CRT KOTS cfg");
            assert_eq!(cfg.d, p.d);
            assert_eq!(cfg.q, p.q_kots());
            assert_eq!(
                cfg.q % 32,
                17,
                "KOTS q' must satisfy the proof condition"
            );
            // Building the backend must succeed (CRT bound holds).
            let _backend = cfg.backend();
            assert_eq!(p.hvc_ring.d(), p.d);
            assert_eq!((p.q_hvc() - 1) % (2 * p.d as u64), 0);
        }
    }

    #[test]
    fn q_kots_reports_modulus() {
        assert_eq!(D256_K4.q_kots(), 3_469_416_721);
    }
}
