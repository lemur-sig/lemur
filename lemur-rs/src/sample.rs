//! XOF-based polynomial samplers matching Python's sample.py.

use crate::profile::Profile;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Upper bound on the ring dimension `d` supported by the sampler.
///
/// Drives stack-buffer sizing in the hot samplers so the d=256 path stays
/// zero-alloc.  Every profile must satisfy `profile.d <= MAX_D`; samplers
/// assert this at call time via `out.len()` bounds checks.
pub const MAX_D: usize = 256;

// ---------------------------------------------------------------------------
// Gaussian sampler context
// ---------------------------------------------------------------------------

/// Runtime-configurable Gaussian sampler parameters.
///
/// Holds a reference to a CDT table and the XOF word width used to
/// sample from it.  The production path uses each profile's baked
/// `cdt` and `cdt_bits` via `xof_gauss_poly_with_profile_fill`;
/// benchmarks build alternative tables via `build_cdt` and pass them
/// to `xof_gauss_poly_ctx` / `kots_keygen_ctx`.
///
/// `cdt_bytes` must satisfy `cdt_bytes * 8 >= ceil(log2(cdt.last()))` and
/// must be <= 4 (CDT entries are stored as `u32`).
pub struct GaussCtx<'a> {
    pub cdt: &'a [u32],
    pub cdt_bytes: usize,
}

impl<'a> GaussCtx<'a> {
    /// Wrap a profile's baked CDT in a `GaussCtx`.
    #[inline]
    pub fn from_profile(profile: &'a Profile) -> Self {
        GaussCtx {
            cdt: profile.cdt,
            cdt_bytes: profile.cdt_bits / 8,
        }
    }
}

#[inline(always)]
fn sample_cdt_indexed(u: u32, cdt: &[u32], cdt_hi: &[u16], prefix_bits: usize) -> i64 {
    let sign = u & 1;
    let cdf_u = u ^ sign;
    let bucket = (cdf_u >> (32 - prefix_bits)) as usize;
    let mut lo = cdt_hi[bucket] as usize;
    let mut hi = cdt_hi[bucket + 1] as usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdt[mid] > cdf_u {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let k = lo as i64;
    if sign == 0 {
        k
    } else {
        -k
    }
}

/// Build a CDT table for the absolute value of a discrete Gaussian at
/// runtime.  Matches Python `sample.build_cdt` byte-for-byte.
///
/// Returns a table of length `floor(tailcut*sigma) + 2` where
/// `cdt[k] = floor(Pr[|X| <= k] * 2^cdt_bits)` for `k = 0..kmax`, with
/// `cdt[kmax] = 2^cdt_bits` (or `u32::MAX` when `cdt_bits == 32`) clamped
/// so the binary search always terminates.
///
/// Uses `f64` internally; adequate for cdt_bits <= 32.  For higher
/// precision use the `gen_tables.py` script (mpmath) and bake the table.
///
/// Requires `cdt_bits <= 32` and `cdt_bits % 8 == 0`.
pub fn build_cdt(sigma: f64, cdt_bits: usize, tailcut: usize) -> Vec<u32> {
    assert!(cdt_bits <= 32, "build_cdt: cdt_bits must be <= 32");
    assert!(
        cdt_bits.is_multiple_of(8),
        "build_cdt: cdt_bits must be byte-aligned"
    );
    let kmax = (tailcut as f64 * sigma) as usize + 1;
    let sigma2 = sigma * sigma;
    let weights: Vec<f64> = (0..=kmax)
        .map(|k| (-((k * k) as f64) / (2.0 * sigma2)).exp())
        .collect();
    let total: f64 = weights[0] + 2.0 * weights[1..].iter().sum::<f64>();
    let scale = if cdt_bits == 32 {
        (1u64 << 32) as f64
    } else {
        (1u64 << cdt_bits) as f64
    };
    let mut cdt = Vec::with_capacity(kmax + 1);
    let mut acc = 0.0f64;
    let clamp = if cdt_bits == 32 {
        u32::MAX
    } else {
        ((1u64 << cdt_bits) - 1) as u32
    };
    for (k, w) in weights.iter().enumerate() {
        acc += if k == 0 { *w } else { 2.0 * w } / total;
        let v = (acc * scale).floor();
        let v_u32 = if v >= scale { clamp } else { v as u32 };
        cdt.push(v_u32);
    }
    // Clamp the final entry so that any u < 2^cdt_bits finds a match.
    let last = cdt.len() - 1;
    cdt[last] = clamp;
    cdt
}

// ---------------------------------------------------------------------------
// Uniform sampling
// ---------------------------------------------------------------------------

/// Sample a uniform polynomial in [0, q)^d from a SHAKE XOF stream.
///
/// Uses rejection sampling with ceil(log2(q))-bit integers.  The XOF
/// read is batched: we pull `2 * d * byte_q` bytes in a single call
/// (enough for ~2x the expected number of draws after rejection) and
/// service rejections out of the stack buffer; a small fallback read
/// handles the rare tail case.
///
/// SHAKE already buffers one rate block internally (168 B for
/// SHAKE128, 136 B for SHAKE256), so the number of Keccak-f
/// permutations is the same as the unbatched form — batching
/// eliminates `XofReader::read()` per-call dispatch overhead
/// (~d calls → ~1 call for the common case).
///
/// Each caller passes an ephemeral XOF seeded per matrix element
/// (see `kots_setup_xof` / `hvc_setup_xof`), so any over-read from
/// the batch is discarded with the XOF — no downstream consumer is
/// affected and the accepted-sample stream is byte-identical to the
/// unbatched form.
pub fn xof_uniform_poly(xof: &mut dyn XofReader, q: u64, d: usize) -> Vec<i64> {
    assert!(
        d <= MAX_D,
        "xof_uniform_poly: d={} exceeds MAX_D={}",
        d,
        MAX_D
    );
    let bits_q = 64 - (q - 1).leading_zeros() as usize;
    let byte_q = bits_q.div_ceil(8);
    let mask_q = if bits_q == 64 {
        u64::MAX
    } else {
        (1u64 << bits_q) - 1
    };
    // Stack-allocated batch buffer sized for 2 * d draws at byte_q <= 8.
    let batch_draws = 2 * d;
    let batch_bytes = batch_draws * byte_q;
    let mut buf = [0u8; 2 * MAX_D * 8];
    xof.read(&mut buf[..batch_bytes]);
    let mut r = Vec::with_capacity(d);
    let mut idx = 0usize;
    while r.len() < d {
        if idx + byte_q > batch_bytes {
            // Rare: refill by reading another d draws' worth.
            let carry_len = batch_bytes - idx;
            buf.copy_within(idx..batch_bytes, 0);
            xof.read(&mut buf[carry_len..carry_len + d * byte_q]);
            idx = 0;
        }
        let mut word = [0u8; 8];
        word[..byte_q].copy_from_slice(&buf[idx..idx + byte_q]);
        let x = u64::from_le_bytes(word) & mask_q;
        idx += byte_q;
        if x < q {
            r.push(x as i64);
        }
    }
    r
}

// ---------------------------------------------------------------------------
// CDT Gaussian sampling
// ---------------------------------------------------------------------------

/// Sample a discrete Gaussian polynomial using a caller-provided ctx.
///
/// For each coefficient we read `ctx.cdt_bytes` bytes, extract the
/// sign bit (LSB), and binary-search `ctx.cdt` for the smallest k
/// with `ctx.cdt[k] > u`.  Every coefficient consumes exactly
/// `ctx.cdt_bytes` of XOF output regardless of the sampled value,
/// keeping a constant-time implementation feasible.
///
/// The XOF read is batched: a single `xof.read(d * cdt_bytes)` call
/// pulls the entire per-poly budget (≤ 1024 B at cdt_bits=32, d=256)
/// into a stack buffer.  SHAKE already buffers one rate block
/// internally (168 B for SHAKE128, 136 B for SHAKE256), so permutation
/// count is unchanged — batching just amortises `XofReader::read()`
/// per-call dispatch overhead (hundreds of calls collapse to one).
/// The byte stream consumed from the XOF is identical to the
/// unbatched form, so test vectors remain byte-for-byte equal.
pub fn xof_gauss_poly_ctx_into(xof: &mut dyn XofReader, ctx: &GaussCtx, out: &mut [i64]) {
    let d = out.len();
    assert!(
        d <= MAX_D,
        "xof_gauss_poly_ctx_into: d={} exceeds MAX_D={}",
        d,
        MAX_D
    );
    let mut buf = [0u8; MAX_D * 4];
    let cdt_bytes = ctx.cdt_bytes;
    let total = d * cdt_bytes;
    xof.read(&mut buf[..total]);
    let cdt = ctx.cdt;
    for (c, dst) in out.iter_mut().enumerate() {
        let mut u_bytes = [0u8; 4];
        u_bytes[..cdt_bytes].copy_from_slice(&buf[c * cdt_bytes..(c + 1) * cdt_bytes]);
        let mut u = u32::from_le_bytes(u_bytes);
        let sign = u & 1;
        u ^= sign;
        let mut lo = 0usize;
        let mut hi = cdt.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdt[mid] > u {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let k = lo as i64;
        *dst = if sign == 0 { k } else { -k };
    }
}

pub fn xof_gauss_poly_ctx(xof: &mut dyn XofReader, ctx: &GaussCtx, d: usize) -> Vec<i64> {
    let mut out = vec![0i64; d];
    xof_gauss_poly_ctx_into(xof, ctx, &mut out);
    out
}

/// Sample a discrete Gaussian polynomial using the baked indexed CDT
/// for a known profile.  Falls back to the profile's generic `GaussCtx`
/// path for profiles without a dedicated bucket index.
pub fn xof_gauss_poly_with_profile_into(
    xof: &mut dyn XofReader,
    profile: &'static Profile,
) -> Vec<i64> {
    let mut out = vec![0i64; profile.d];
    xof_gauss_poly_with_profile_fill(xof, profile, &mut out);
    out
}

pub fn xof_gauss_poly_with_profile_fill(
    xof: &mut dyn XofReader,
    profile: &'static Profile,
    out: &mut [i64],
) {
    let d = out.len();
    assert_eq!(
        d, profile.d,
        "xof_gauss_poly_with_profile_fill: out.len() must match profile.d"
    );
    assert!(
        d <= MAX_D,
        "xof_gauss_poly_with_profile_fill: d={} exceeds MAX_D={}",
        d,
        MAX_D
    );
    if let Some(cdt_hi) = profile.cdt_hi {
        let mut buf = [0u8; MAX_D * 4];
        let total = d * 4;
        xof.read(&mut buf[..total]);
        for (dst, word_bytes) in out.iter_mut().zip(buf[..total].chunks_exact(4)) {
            let u = u32::from_le_bytes(word_bytes.try_into().unwrap());
            *dst = sample_cdt_indexed(u, profile.cdt, cdt_hi, profile.cdt_prefix_bits as usize);
        }
    } else {
        xof_gauss_poly_ctx_into(
            xof,
            &GaussCtx {
                cdt: profile.cdt,
                cdt_bytes: profile.cdt_bits / 8,
            },
            out,
        );
    }
}

pub fn xof_gauss_poly_with_profile(xof: &mut dyn XofReader, profile: &'static Profile) -> Vec<i64> {
    xof_gauss_poly_with_profile_into(xof, profile)
}

// ---------------------------------------------------------------------------
// Ternary polynomial sampling
// ---------------------------------------------------------------------------

/// Sample a fixed-weight ternary polynomial from a SHAKE XOF.
///
/// Reads 9 bytes per batch: 8 position bytes + 1 sign-bit byte (bit per
/// candidate). Retries on collision or out-of-range position.
pub fn xof_ternary_poly(xof: &mut dyn XofReader, weight: usize, d: usize) -> Vec<i64> {
    assert!(
        d <= 256,
        "xof_ternary_poly: d={} exceeds byte-indexable range",
        d
    );
    let mut poly = vec![0i64; d];
    let mut remaining = weight;
    let mut i = 8usize;
    let mut raw = [0u8; 9];
    while remaining > 0 {
        if i == 8 {
            i = 0;
            xof.read(&mut raw);
        }
        let pos = raw[i] as usize;
        if pos < d && poly[pos] == 0 {
            poly[pos] = if (raw[8] >> i) & 1 == 0 { 1 } else { -1 };
            remaining -= 1;
        }
        i += 1;
    }
    poly
}

// ---------------------------------------------------------------------------
// Domain-separated XOF constructors matching Python exactly
// ---------------------------------------------------------------------------

/// KOTS setup: expand A2[i][j] from SHAKE128(seed || [i,j] || b"A"),
/// where A = [I_n; A2] and only the lower block A2 is materialized.
pub fn kots_setup_xof(seed: &[u8], i: usize, j: usize) -> impl XofReader {
    use sha3::Shake128;
    let mut h = Shake128::default();
    h.update(seed);
    h.update(&[i as u8, j as u8]);
    h.update(b"A");
    h.finalize_xof()
}

/// KOTS keygen: expand S[i][j] from SHAKE256(seed || [i,j] || b"S").
pub fn kots_keygen_xof(seed: &[u8], i: usize, j: usize) -> impl XofReader {
    let mut h = Shake256::default();
    h.update(seed);
    h.update(&[i as u8, j as u8]);
    h.update(b"S");
    h.finalize_xof()
}

/// KOTS sign hash H: SHAKE256(mu || j.to_le_bytes(4) || b"H").
pub fn kots_hash_xof(mu: &[u8], j: u32) -> impl XofReader {
    let mut h = Shake256::default();
    h.update(mu);
    h.update(&j.to_le_bytes());
    h.update(b"H");
    h.finalize_xof()
}

/// HVC setup: expand matrix entry from SHAKE128(seed || [i,j] || tag).
pub fn hvc_setup_xof(seed: &[u8], i: usize, j: usize, tag: &[u8]) -> impl XofReader {
    use sha3::Shake128;
    let mut h = Shake128::default();
    h.update(seed);
    h.update(&[i as u8, j as u8]);
    h.update(tag);
    h.finalize_xof()
}

/// Aggregation randomizers XOF.
///
/// Domain: t.to_bytes(4, 'little') || len(m).to_bytes(4, 'little') || m || pk_bytes || attempt.to_bytes(4, 'little').
/// Matches Python lemur.py `_hash_to_randomizers`.
pub fn agg_randomizer_xof(
    t: usize,
    msg: &[u8],
    pk_bytes_concat: &[u8],
    attempt: usize,
) -> impl XofReader {
    let mut h = Shake256::default();
    h.update(&(t as u32).to_le_bytes());
    h.update(&(msg.len() as u32).to_le_bytes());
    h.update(msg);
    h.update(pk_bytes_concat);
    h.update(&(attempt as u32).to_le_bytes());
    h.finalize_xof()
}

/// Per-slot seed derivation: SHAKE256(master_seed || b"slot" || t.to_le_bytes(4)).read(32).
pub fn slot_seed(master_seed: &[u8], t: usize) -> [u8; 32] {
    let mut h = Shake256::default();
    h.update(master_seed);
    h.update(b"slot");
    h.update(&(t as u32).to_le_bytes());
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    out
}
