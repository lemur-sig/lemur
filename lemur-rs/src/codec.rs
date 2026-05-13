//! Bit-packed serialization matching Python's codec.py exactly.
//!
//! Individual signatures always use Babai format
//! (path labels omitted, reconstructed from siblings + u).
//! Aggregated signatures use N-dependent encoding with optional
//! Golomb-Rice coding.

use crate::error::LemurError;
use std::collections::VecDeque;

use crate::hvc::{
    babai_encode_label_with_profile, proj_label_with_profile, reconstruct_path_labels_agg,
    reconstruct_path_labels_ind, BdsState, HvcOpening, HvcPp, TreehashInst,
};
use crate::kots::KotsSig;
use crate::lemur::{lemur_make_stateful_sk, LemurAggSig, LemurPk, LemurPp, LemurSig, LemurStateSk};
use crate::profile::Profile;

// ---------------------------------------------------------------------------
// Bit widths — all profile-aware.
// ---------------------------------------------------------------------------

fn logq_hvc(profile: &Profile) -> usize {
    64 - (profile.q_hvc() - 1).leading_zeros() as usize
}

fn bits_z(profile: &Profile) -> usize {
    64 - (2 * profile.beta_z as u64).leading_zeros() as usize
}

fn bits_dig(profile: &Profile) -> usize {
    64 - (2 * profile.eta as u64).leading_zeros() as usize
}

fn poly_bytes(d: usize, dx: usize) -> usize {
    (d * dx).div_ceil(8)
}

/// Expected Rice codeword bits per coefficient under a continuous-Gaussian
/// model `X ~ N(0, sigma^2)`:
///   L = k+1                       if X = 0
///   L = k+2+j  (j >= 0)            if j*2^k <= |X| < (j+1)*2^k
/// so  E[L] = k + 2 - P(X=0) + sum_{j>=1} 2*(1 - Phi(j*2^k/sigma))
///         = k + 2 - P(X=0) + sum_{j>=1} erfc(j*2^k/(sigma*sqrt 2)).
///
/// Used only by `sizes()` to print a documentation estimate; the on-wire
/// `rice_k` selection in `compute_agg_encoding` still uses the older,
/// looser cost model (kept stable for byte-level interop).
fn rice_bits_per_coef(sigma: f64, rice_k: usize) -> f64 {
    let sigma = sigma.max(0.0);
    if sigma == 0.0 {
        return (rice_k + 1) as f64;
    }
    let p_zero = libm::erf(0.5 / (sigma * std::f64::consts::SQRT_2));
    let step = (1u64 << rice_k) as f64;
    let inv = 1.0 / (sigma * std::f64::consts::SQRT_2);
    let mut e_floor = 0.0;
    let mut j = 1usize;
    loop {
        let p = libm::erfc(j as f64 * step * inv);
        e_floor += p;
        if p < 1e-15 && j > 5 {
            break;
        }
        j += 1;
        if j > 10_000 {
            break;
        }
    }
    rice_k as f64 + 2.0 - p_zero + e_floor
}

// ---------------------------------------------------------------------------
// Fixed-width bit-packing
// ---------------------------------------------------------------------------

/// Pack coefficients at dx bits each (unsigned after adding offset).
pub fn poly_serial(poly: &[i64], dx: usize, offset: i64) -> Vec<u8> {
    let mask = if dx == 64 { u64::MAX } else { (1u64 << dx) - 1 };
    let mut buf: u64 = 0;
    let mut bits: usize = 0;
    let mut out = Vec::with_capacity(poly_bytes(poly.len(), dx));
    for &x in poly {
        let v = ((x + offset) as u64) & mask;
        buf |= v << bits;
        bits += dx;
        while bits >= 8 {
            out.push((buf & 0xFF) as u8);
            buf >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((buf & 0xFF) as u8);
    }
    out
}

/// Unpack d coefficients. Returns (values, bytes_consumed).
/// Checks: coefficient range, nonzero padding, truncated input.
pub fn poly_deserial(
    data: &[u8],
    dx: usize,
    d: usize,
    offset: i64,
) -> Result<(Vec<i64>, usize), LemurError> {
    let max_unsigned = if offset > 0 {
        2 * offset as u64
    } else if dx == 64 {
        u64::MAX
    } else {
        (1u64 << dx) - 1
    };
    let mask = if dx == 64 { u64::MAX } else { (1u64 << dx) - 1 };
    let expected = poly_bytes(d, dx);
    if data.len() < expected {
        return Err(LemurError::InvalidEncoding(format!(
            "truncated: need {expected} bytes, got {}",
            data.len()
        )));
    }
    let mut p = Vec::with_capacity(d);
    let mut buf: u64 = 0;
    let mut bits: usize = 0;
    let mut i = 0;
    while p.len() < d {
        while bits < dx {
            buf |= (data[i] as u64) << bits;
            bits += 8;
            i += 1;
        }
        let raw = buf & mask;
        if raw > max_unsigned {
            return Err(LemurError::InvalidEncoding(format!(
                "coefficient {} out of range [-{}, {}]",
                raw as i64 - offset,
                offset,
                offset
            )));
        }
        p.push(raw as i64 - offset);
        buf >>= dx;
        bits -= dx;
    }
    if bits > 0 && (buf & ((1u64 << bits) - 1)) != 0 {
        return Err(LemurError::InvalidEncoding(
            "nonzero padding bits in fixed-width polynomial".into(),
        ));
    }
    Ok((p, i))
}

/// Serialize n polynomials at dx bits per coefficient.
pub fn vec_serial(v: &[Vec<i64>], dx: usize, offset: i64) -> Vec<u8> {
    v.iter().flat_map(|p| poly_serial(p, dx, offset)).collect()
}

/// Deserialize n polynomials of degree d.
pub fn vec_deserial(
    data: &[u8],
    dx: usize,
    n: usize,
    d: usize,
    offset: i64,
) -> Result<(Vec<Vec<i64>>, usize), LemurError> {
    let mut result = Vec::with_capacity(n);
    let mut consumed = 0;
    for _ in 0..n {
        let (poly, l) = poly_deserial(&data[consumed..], dx, d, offset)?;
        result.push(poly);
        consumed += l;
    }
    Ok((result, consumed))
}

// ---------------------------------------------------------------------------
// Golomb-Rice bit-packing
// ---------------------------------------------------------------------------

/// Rice-encode a polynomial, byte-aligned.
///
/// Per coefficient x:
///   x == 0: rice_k zero bits + stop bit (0)
///   x != 0: low rice_k bits of |x| + unary(|x|>>rice_k) ones
///            + stop bit (0) + sign bit (0=pos, 1=neg)
pub fn poly_serial_rice(poly: &[i64], rice_k: usize, _bound: i64) -> Vec<u8> {
    let low_mask = if rice_k == 0 {
        0u64
    } else {
        (1u64 << rice_k) - 1
    };
    let mut buf: u64 = 0;
    let mut bits: usize = 0;
    let mut out = Vec::new();
    for &x in poly {
        let ax = x.unsigned_abs();
        let low = ax & low_mask;
        let hi = ax >> rice_k;
        buf |= low << bits;
        bits += rice_k;
        // unary: hi ones then a zero stop bit
        if hi > 0 {
            buf |= ((1u64 << hi) - 1) << bits;
        }
        bits += hi as usize + 1;
        if x != 0 {
            if x < 0 {
                buf |= 1u64 << bits;
            }
            bits += 1;
        }
        while bits >= 8 {
            out.push((buf & 0xFF) as u8);
            buf >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((buf & 0xFF) as u8);
    }
    out
}

/// Decode a Rice-coded polynomial. Returns (values, bytes_consumed).
pub fn poly_deserial_rice(
    data: &[u8],
    d: usize,
    rice_k: usize,
    bound: i64,
) -> Result<(Vec<i64>, usize), LemurError> {
    let max_hi = ((bound as u64) >> rice_k) + 1;
    let low_mask = if rice_k == 0 {
        0u64
    } else {
        (1u64 << rice_k) - 1
    };
    let mut p = Vec::with_capacity(d);
    let mut buf: u64 = 0;
    let mut bits: usize = 0;
    let mut i: usize = 0;
    let dlen = data.len();

    let pull =
        |buf: &mut u64, bits: &mut usize, i: &mut usize, need: usize| -> Result<(), LemurError> {
            while *bits < need {
                if *i >= dlen {
                    return Err(LemurError::InvalidEncoding(
                        "truncated Rice-coded polynomial".into(),
                    ));
                }
                *buf |= (data[*i] as u64) << *bits;
                *bits += 8;
                *i += 1;
            }
            Ok(())
        };

    for _ in 0..d {
        pull(&mut buf, &mut bits, &mut i, rice_k)?;
        let low = buf & low_mask;
        buf >>= rice_k;
        bits -= rice_k;

        let mut hi: u64 = 0;
        loop {
            pull(&mut buf, &mut bits, &mut i, 1)?;
            let bit = buf & 1;
            buf >>= 1;
            bits -= 1;
            if bit == 0 {
                break;
            }
            hi += 1;
            if hi > max_hi {
                return Err(LemurError::InvalidEncoding(format!(
                    "Rice unary run {hi} exceeds max {max_hi}"
                )));
            }
        }

        let ax = low | (hi << rice_k);
        if ax == 0 {
            p.push(0);
        } else {
            if ax > bound as u64 {
                return Err(LemurError::InvalidEncoding(format!(
                    "Rice-decoded |coeff| {ax} exceeds bound {bound}"
                )));
            }
            pull(&mut buf, &mut bits, &mut i, 1)?;
            let sign = buf & 1;
            buf >>= 1;
            bits -= 1;
            p.push(if sign != 0 { -(ax as i64) } else { ax as i64 });
        }
    }

    if bits > 0 && (buf & ((1u64 << bits) - 1)) != 0 {
        return Err(LemurError::InvalidEncoding(
            "nonzero padding bits in Rice-coded polynomial".into(),
        ));
    }
    Ok((p, i))
}

/// Rice-serialize a sequence of polynomials.
pub fn vec_serial_rice(v: &[Vec<i64>], rice_k: usize, bound: i64) -> Vec<u8> {
    v.iter()
        .flat_map(|p| poly_serial_rice(p, rice_k, bound))
        .collect()
}

/// Rice-deserialize n polynomials of degree d.
pub fn vec_deserial_rice(
    data: &[u8],
    n: usize,
    d: usize,
    rice_k: usize,
    bound: i64,
) -> Result<(Vec<Vec<i64>>, usize), LemurError> {
    let mut result = Vec::with_capacity(n);
    let mut consumed = 0;
    for _ in 0..n {
        let (poly, l) = poly_deserial_rice(&data[consumed..], d, rice_k, bound)?;
        result.push(poly);
        consumed += l;
    }
    Ok((result, consumed))
}

// ---------------------------------------------------------------------------
// Flat array helpers
// ---------------------------------------------------------------------------

fn flat_to_polys(flat: &[i64], n: usize, d: usize) -> Vec<Vec<i64>> {
    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
}

fn polys_to_flat(polys: &[Vec<i64>]) -> Vec<i64> {
    polys.iter().flat_map(|p| p.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// Encoding mode
// ---------------------------------------------------------------------------

/// Rice or fixed-width encoding mode.
#[derive(Debug, Clone)]
pub enum EncMode {
    Rice { rice_k: usize },
    Fixed { dx: usize },
}

/// Encoding parameters for aggregated signatures.
#[derive(Debug, Clone)]
pub struct AggEncoding {
    pub n_signers: usize,
    pub zagg_bound: i64,
    pub zagg_dx: usize,
    pub babai_mode: EncMode,
    pub babai_bound: i64,
    pub agg_mode: EncMode,
    pub agg_bound: i64,
}

/// Compute encoding parameters matching Python's compute_agg_encoding.
pub fn compute_agg_encoding(pp: &LemurPp, n_signers: usize) -> AggEncoding {
    let profile = pp.profile;
    let eta = profile.eta as f64;
    let alpha_w = profile.alpha_w as f64;

    let var_digit = eta * (eta + 1.0) / 3.0;
    let sigma_label = (n_signers as f64 * alpha_w * var_digit).sqrt();

    // KOTS sampler stddev (actual standard deviation of S entries).
    // sigma = alpha / sqrt(2*pi).  Using `profile.sigma` rather than
    // `profile.alpha` is required — conflating the two mis-sizes the
    // Rice vs fixed-width choice by a factor of sqrt(2*pi).
    let sigma = profile.sigma;
    let alpha_h = profile.alpha_h as f64;
    let k = profile.k as f64;
    let sigma_z_ind = sigma * (1.0 + (k - 1.0) * alpha_h).sqrt();
    let sigma_zagg = sigma_z_ind * (n_signers as f64 * alpha_w).sqrt();

    let sigma_babai = sigma_label / (2.0 * eta);

    // Z_agg: N-dependent fixed-width
    let n_zagg_coeffs = (profile.m * profile.d) as f64;
    let c_zagg = (2.0 * (2.0 * n_zagg_coeffs * 256.0).ln()).sqrt();
    let zagg_bound_raw = (c_zagg * sigma_zagg).ceil() as i64;
    let zagg_bound = zagg_bound_raw.min(profile.beta_sigma);
    // bit_length of 2*zagg_bound, minimum 1
    let zagg_dx = {
        let v = 2 * zagg_bound;
        if v <= 0 {
            1
        } else {
            64 - (v as u64).leading_zeros() as usize
        }
    };

    let babai_mode = rice_params(sigma_babai, pp.hvc_pp.beta_encode);
    let agg_mode = rice_params(sigma_label, pp.hvc_pp.beta_agg);

    AggEncoding {
        n_signers,
        zagg_bound,
        zagg_dx,
        babai_mode,
        babai_bound: pp.hvc_pp.beta_encode,
        agg_mode,
        agg_bound: pp.hvc_pp.beta_agg,
    }
}

/// Pick the encoding that minimises expected bits per coefficient.
///
/// Models each scheme's per-coefficient cost and returns whichever is
/// cheaper:
///   - fixed-width: `dx_fixed = bit_length(2*max_bound)` bits/coef
///   - Rice at parameter k: `k + mu/2^k + 2` bits/coef
///     (folded-Gaussian mean `mu = 0.7979*sigma`, plus stop+sign bits)
///
/// Searches `k` over `[0, dx_fixed]`, which brackets the minimum (the
/// cost function is unimodal in k).  No `sigma<2` cutoff and no
/// Rice/fixed hysteresis — the formulas decide.
fn rice_params(sigma: f64, max_bound: i64) -> EncMode {
    let dx_fixed = 64 - (2 * max_bound as u64).leading_zeros() as usize;
    let mu = 0.7979 * sigma.max(0.0);

    let mut best_k = 0usize;
    let mut best_bits = mu + 2.0;
    for k in 1..=dx_fixed {
        let bits = k as f64 + 2.0 + mu / ((1u64 << k) as f64);
        if bits < best_bits {
            best_bits = bits;
            best_k = k;
        }
    }

    if best_bits < dx_fixed as f64 {
        EncMode::Rice { rice_k: best_k }
    } else {
        EncMode::Fixed { dx: dx_fixed }
    }
}

// ---------------------------------------------------------------------------
// Public parameters
// ---------------------------------------------------------------------------

/// pp wire format: `kots_seed(32) || hvc_seed(32) || tau_u8` = 65 bytes.
///
/// The profile is not embedded; both sides must agree out-of-band on
/// which `&'static Profile` to construct the scheme under.  Both
/// Python (`lemur-py/codec.py`) and Rust emit the same layout.
pub const PP_BYTES: usize = 65;

/// Encode pp as `kots_seed || hvc_seed || tau_u8`.
pub fn pp_encode(kots_seed: &[u8; 32], hvc_seed: &[u8; 32], tau: usize) -> Vec<u8> {
    let tau_u8 = u8::try_from(tau).expect("tau must fit in u8");
    let mut out = Vec::with_capacity(PP_BYTES);
    out.extend_from_slice(kots_seed);
    out.extend_from_slice(hvc_seed);
    out.push(tau_u8);
    out
}

/// Tuple returned by `pp_decode`: `(kots_seed, hvc_seed, tau)`.
pub type PpDecoded = ([u8; 32], [u8; 32], usize);

/// Decode pp.  Returns `(kots_seed, hvc_seed, tau)`.  The profile is
/// not part of the wire format — callers thread their `&'static Profile`
/// of choice into the scheme they build from these seeds.
pub fn pp_decode(data: &[u8]) -> Result<PpDecoded, LemurError> {
    if data.len() != PP_BYTES {
        return Err(LemurError::InvalidEncoding(format!(
            "pp must be {PP_BYTES} bytes, got {}",
            data.len()
        )));
    }
    let mut ks = [0u8; 32];
    let mut hs = [0u8; 32];
    ks.copy_from_slice(&data[..32]);
    hs.copy_from_slice(&data[32..64]);
    let tau = data[64] as usize;
    if tau == 0 || tau > 32 {
        return Err(LemurError::InvalidEncoding(format!(
            "pp: tau={tau} out of supported range [1, 32]"
        )));
    }
    Ok((ks, hs, tau))
}

// ---------------------------------------------------------------------------
// Secret key
// ---------------------------------------------------------------------------

/// Encode sk: 32-byte master seed.
pub fn sk_encode(master_seed: &[u8; 32]) -> Vec<u8> {
    master_seed.to_vec()
}

/// Decode sk from 32 bytes.
pub fn sk_decode(data: &[u8]) -> Result<[u8; 32], LemurError> {
    if data.len() != 32 {
        return Err(LemurError::InvalidEncoding(format!(
            "seed secret key must be 32 bytes, got {}",
            data.len()
        )));
    }
    let mut master_seed = [0u8; 32];
    master_seed.copy_from_slice(data);
    Ok(master_seed)
}

// ---------------------------------------------------------------------------
// Stateful secret key: BDS08 traversal cache, no magic bytes
// ---------------------------------------------------------------------------
//
// Wire layout:
//
//   +0   32 bytes   master_seed
//   +32   4 bytes   u32 LE  phi
//   +36   4 bytes   u32 LE  tau  (= state.h)
//   +40   4 bytes   u32 LE  k    (= state.k)
//   +44           auth: tau labels, each LABEL_BYTES
//                 keep: tau × (u8 presence [+ LABEL_BYTES])
//               retain: tau × (u16 LE count, count × LABEL_BYTES)
//             treehash: (tau - k) × treehash_record
//     treehash_record = u8 finished
//                     | u32 LE leaf_index
//                     | u32 LE leaves_remaining
//                     | u8 node_present [+ LABEL_BYTES]
//                     | u16 LE stack_count, stack_count × (u8 level + LABEL_BYTES)
//
// Each label is encoded with `vec_serial` at dx=bits_dig (=6), offset=ETA,
// so LABEL_BYTES = poly_bytes(D, bits_dig) * OMEGA * KAPPA = 192 * 12 = 2304.
// Byte-for-byte interoperable with the Python `lemur-py` codec.

const SK_STATE_HEADER_SIZE: usize = 32 + 4 + 4 + 4;

fn label_bytes(profile: &Profile) -> usize {
    poly_bytes(profile.d, bits_dig(profile)) * profile.omega * profile.kappa
}

fn label_encode(label: &[i64], profile: &Profile) -> Vec<u8> {
    let polys = flat_to_polys(label, profile.omega * profile.kappa, profile.d);
    vec_serial(&polys, bits_dig(profile), profile.eta)
}

fn label_decode_at(
    data: &[u8],
    pos: usize,
    profile: &Profile,
) -> Result<(Vec<i64>, usize), LemurError> {
    let dx = bits_dig(profile);
    let n_rows = profile.omega * profile.kappa;
    let (polys, consumed) = vec_deserial(&data[pos..], dx, n_rows, profile.d, profile.eta)?;
    Ok((polys_to_flat(&polys), pos + consumed))
}

/// Encode a stateful signer key (profile-aware real impl).
pub fn sk_state_encode_with_profile(
    sk_state: &LemurStateSk,
    profile: &Profile,
) -> Result<Vec<u8>, LemurError> {
    let bds = &sk_state.bds;

    let h = u32::try_from(bds.h)
        .map_err(|_| LemurError::InvalidEncoding("stateful secret key tau out of range".into()))?;
    let k = u32::try_from(bds.k)
        .map_err(|_| LemurError::InvalidEncoding("stateful secret key k out of range".into()))?;
    let phi = u32::try_from(bds.phi)
        .map_err(|_| LemurError::InvalidEncoding("stateful secret key phi out of range".into()))?;

    let lb = label_bytes(profile);
    let expected_label_len = profile.omega * profile.kappa * profile.d;
    let mut out = Vec::with_capacity(SK_STATE_HEADER_SIZE + bds.h * lb * 4);
    out.extend_from_slice(&sk_state.master_seed);
    out.extend_from_slice(&phi.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&k.to_le_bytes());

    // auth: dense, h labels.
    for level in 0..bds.h {
        let lab = &bds.auth[level];
        if lab.len() != expected_label_len {
            return Err(LemurError::InvalidEncoding(format!(
                "auth[{level}] wrong length"
            )));
        }
        out.extend_from_slice(&label_encode(lab, profile));
    }

    // keep: sparse per level, iterate all h levels.
    for level in 0..bds.h {
        match bds.keep[level].as_ref() {
            None => out.push(0),
            Some(lab) => {
                out.push(1);
                out.extend_from_slice(&label_encode(lab, profile));
            }
        }
    }

    // retain: iterate all h levels with u16 count.
    for level in 0..bds.h {
        let queue = &bds.retain[level];
        let count = u16::try_from(queue.len()).map_err(|_| {
            LemurError::InvalidEncoding(format!("retain[{level}] count {} > 65535", queue.len()))
        })?;
        out.extend_from_slice(&count.to_le_bytes());
        for lab in queue {
            out.extend_from_slice(&label_encode(lab, profile));
        }
    }

    // treehash: h - k records.
    let th_count = bds.h.saturating_sub(bds.k);
    for level in 0..th_count {
        let th = &bds.treehash[level];
        out.push(if th.finished { 1 } else { 0 });
        let li = u32::try_from(th.leaf_index).map_err(|_| {
            LemurError::InvalidEncoding(format!(
                "treehash[{level}] leaf_index {} > u32::MAX",
                th.leaf_index
            ))
        })?;
        out.extend_from_slice(&li.to_le_bytes());
        let lr = u32::try_from(th.leaves_remaining).map_err(|_| {
            LemurError::InvalidEncoding(format!(
                "treehash[{level}] leaves_remaining {} > u32::MAX",
                th.leaves_remaining
            ))
        })?;
        out.extend_from_slice(&lr.to_le_bytes());
        match th.node.as_ref() {
            None => out.push(0),
            Some(lab) => {
                out.push(1);
                out.extend_from_slice(&label_encode(lab, profile));
            }
        }
        let stack_count = u16::try_from(th.stack.len()).map_err(|_| {
            LemurError::InvalidEncoding(format!(
                "treehash[{level}] stack too large ({})",
                th.stack.len()
            ))
        })?;
        out.extend_from_slice(&stack_count.to_le_bytes());
        for (stack_h, stack_lab) in &th.stack {
            let sh = u8::try_from(*stack_h).map_err(|_| {
                LemurError::InvalidEncoding(format!(
                    "treehash[{level}] stack height {stack_h} > 255"
                ))
            })?;
            out.push(sh);
            out.extend_from_slice(&label_encode(stack_lab, profile));
        }
    }

    Ok(out)
}

/// Decode a stateful signer key from the on-disk BDS-state format (profile-aware real impl).
pub fn sk_state_decode_with_profile(
    data: &[u8],
    profile: &Profile,
) -> Result<LemurStateSk, LemurError> {
    if data.len() < SK_STATE_HEADER_SIZE {
        return Err(LemurError::InvalidEncoding(
            "stateful secret key header truncated".into(),
        ));
    }
    let mut master_seed = [0u8; 32];
    master_seed.copy_from_slice(&data[0..32]);
    let phi = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(data[36..40].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;

    if h > 32 {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful sk: tau={h} out of supported range"
        )));
    }
    if k > h {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful sk: invalid k={k} (tau={h})"
        )));
    }
    if phi > (1usize << h) {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful sk: phi={phi} out of range for tau={h}"
        )));
    }

    let mut pos = SK_STATE_HEADER_SIZE;

    // auth
    let mut auth: Vec<Vec<i64>> = Vec::with_capacity(h);
    for _ in 0..h {
        let (lab, new_pos) = label_decode_at(data, pos, profile)?;
        auth.push(lab);
        pos = new_pos;
    }

    // keep
    let mut keep: Vec<Option<Vec<i64>>> = vec![None; h];
    for (level, slot) in keep.iter_mut().enumerate() {
        if pos >= data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (keep section)".into(),
            ));
        }
        let flag = data[pos];
        pos += 1;
        match flag {
            0 => {}
            1 => {
                if level + 1 >= h {
                    return Err(LemurError::InvalidEncoding(format!(
                        "stateful sk: keep entry at level {level} >= h-1={}",
                        h - 1
                    )));
                }
                let (lab, new_pos) = label_decode_at(data, pos, profile)?;
                *slot = Some(lab);
                pos = new_pos;
            }
            _ => {
                return Err(LemurError::InvalidEncoding(format!(
                    "stateful sk: invalid keep flag {flag} at level {level}"
                )));
            }
        }
    }

    // retain
    let mut retain: Vec<VecDeque<Vec<i64>>> = (0..h).map(|_| VecDeque::new()).collect();
    for (level, queue) in retain.iter_mut().enumerate() {
        if pos + 2 > data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (retain section)".into(),
            ));
        }
        let count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if count > 0 {
            let in_range = level + 1 < h && level + k >= h;
            if !in_range {
                return Err(LemurError::InvalidEncoding(format!(
                    "stateful sk: retain entries at non-retain level {level}"
                )));
            }
            for _ in 0..count {
                let (lab, new_pos) = label_decode_at(data, pos, profile)?;
                queue.push_back(lab);
                pos = new_pos;
            }
        }
    }

    // treehash
    let th_count = h.saturating_sub(k);
    let mut treehash: Vec<TreehashInst> = (0..th_count).map(TreehashInst::new).collect();
    for th in treehash.iter_mut() {
        if pos >= data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (treehash section)".into(),
            ));
        }
        let finished_flag = data[pos];
        pos += 1;
        if finished_flag > 1 {
            return Err(LemurError::InvalidEncoding(format!(
                "stateful sk: invalid treehash finished flag {finished_flag}"
            )));
        }
        if pos + 8 > data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (treehash counters)".into(),
            ));
        }
        let leaf_index = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let leaves_remaining = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos >= data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (treehash node flag)".into(),
            ));
        }
        let node_flag = data[pos];
        pos += 1;
        let mut node_opt: Option<Vec<i64>> = None;
        match node_flag {
            0 => {}
            1 => {
                let (lab, new_pos) = label_decode_at(data, pos, profile)?;
                node_opt = Some(lab);
                pos = new_pos;
            }
            _ => {
                return Err(LemurError::InvalidEncoding(format!(
                    "stateful sk: invalid treehash node flag {node_flag}"
                )));
            }
        }
        if pos + 2 > data.len() {
            return Err(LemurError::InvalidEncoding(
                "stateful sk: truncated (treehash stack count)".into(),
            ));
        }
        let stack_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let mut stack: Vec<(usize, Vec<i64>)> = Vec::with_capacity(stack_count);
        for _ in 0..stack_count {
            if pos >= data.len() {
                return Err(LemurError::InvalidEncoding(
                    "stateful sk: truncated (treehash stack entry)".into(),
                ));
            }
            let stack_h = data[pos] as usize;
            pos += 1;
            let (lab, new_pos) = label_decode_at(data, pos, profile)?;
            stack.push((stack_h, lab));
            pos = new_pos;
        }
        th.stack = stack;
        th.node = node_opt;
        th.leaf_index = leaf_index;
        th.leaves_remaining = leaves_remaining;
        th.finished = finished_flag == 1;
    }

    if pos != data.len() {
        return Err(LemurError::InvalidEncoding(format!(
            "stateful sk: trailing bytes (consumed {pos}, total {})",
            data.len()
        )));
    }

    let bds = BdsState {
        h,
        k,
        phi,
        auth,
        keep,
        retain,
        treehash,
    };
    Ok(lemur_make_stateful_sk(&master_seed, bds))
}

// ---------------------------------------------------------------------------
// Public key
// ---------------------------------------------------------------------------

/// Encode pk: omega polys at logq bits (unsigned) — profile-aware.
pub fn pk_encode_with_profile(pk: &LemurPk, profile: &Profile) -> Vec<u8> {
    let dx = logq_hvc(profile);
    let polys = flat_to_polys(&pk.0 .0, profile.omega, profile.d);
    vec_serial(&polys, dx, 0)
}

/// Decode pk from bytes — profile-aware.
pub fn pk_decode_with_profile(data: &[u8], profile: &Profile) -> Result<LemurPk, LemurError> {
    let dx = logq_hvc(profile);
    let (polys, _) = vec_deserial(data, dx, profile.omega, profile.d, 0)?;
    let flat = polys_to_flat(&polys);
    Ok(LemurPk(crate::hvc::HvcCom(flat)))
}

// ---------------------------------------------------------------------------
// Individual signature (always Babai: path labels omitted)
// ---------------------------------------------------------------------------

/// Encode individual signature. Path labels omitted (Babai).
/// Format: Z (fixed-width) | sibling labels (fixed-width) | u (fixed-width).
/// The HvcPp reference supplies the profile used for bit widths.
pub fn sig_encode(sig: &LemurSig, hvc_pp: &HvcPp) -> Vec<u8> {
    let profile = hvc_pp.profile;
    let dx_z = bits_z(profile);
    let off_z = profile.beta_z;
    let dx_d = bits_dig(profile);
    let off_d = profile.eta;
    let n_z = profile.m;
    let n_label = profile.omega * profile.kappa;
    let n_u = profile.k * profile.n * profile.kappa_prime;

    let z_polys = flat_to_polys(&sig.z.0, n_z, profile.d);
    let mut out = vec_serial(&z_polys, dx_z, off_z);
    for label in &sig.opening.sibling_labels {
        let polys = flat_to_polys(label, n_label, profile.d);
        out.extend(vec_serial(&polys, dx_d, off_d));
    }
    let u_polys = flat_to_polys(&sig.opening.u, n_u, profile.d);
    out.extend(vec_serial(&u_polys, dx_d, off_d));
    out
}

/// Decode individual signature, reconstructing path labels.
pub fn sig_decode(data: &[u8], pp: &LemurPp, t: usize) -> Result<LemurSig, LemurError> {
    let profile = pp.profile;
    let dx_z = bits_z(profile);
    let off_z = profile.beta_z;
    let dx_d = bits_dig(profile);
    let off_d = profile.eta;
    let n_z = profile.m;
    let n_label = profile.omega * profile.kappa;
    let n_u = profile.k * profile.n * profile.kappa_prime;
    let tau = pp.hvc_pp.tau;

    let (z_polys, l) = vec_deserial(data, dx_z, n_z, profile.d, off_z)?;
    let z = KotsSig(polys_to_flat(&z_polys));
    let mut pos = l;

    let mut sibling_labels = Vec::with_capacity(tau);
    for _ in 0..tau {
        let (polys, l) = vec_deserial(&data[pos..], dx_d, n_label, profile.d, off_d)?;
        sibling_labels.push(polys_to_flat(&polys));
        pos += l;
    }

    let (u_polys, l) = vec_deserial(&data[pos..], dx_d, n_u, profile.d, off_d)?;
    let u = polys_to_flat(&u_polys);
    pos += l;

    if pos != data.len() {
        return Err(LemurError::InvalidEncoding(format!(
            "trailing bytes: consumed {pos}, total {}",
            data.len()
        )));
    }

    let path_labels = reconstruct_path_labels_ind(&pp.hvc_pp, t, &sibling_labels, &u);

    Ok(LemurSig {
        z,
        opening: HvcOpening {
            path_labels,
            sibling_labels,
            u,
        },
    })
}

/// Byte count for an individual signature — profile-aware real impl.
pub fn sig_bytes_with_profile(tau: usize, profile: &Profile) -> usize {
    let dx_z = bits_z(profile);
    let dx_d = bits_dig(profile);
    let pb_z = poly_bytes(profile.d, dx_z);
    let pb_dig = poly_bytes(profile.d, dx_d);
    let n_label = profile.omega * profile.kappa;
    let n_u = profile.k * profile.n * profile.kappa_prime;
    profile.m * pb_z + tau * n_label * pb_dig + n_u * pb_dig
}


// ---------------------------------------------------------------------------
// Aggregated signature
// ---------------------------------------------------------------------------

/// Encode aggregated signature.
/// Format: attempt (1B) | Z_agg (N-dep FW) | Babai path (Rice/FW)
///         | sibling labels (Rice/FW) | u (Rice/FW)
pub fn agg_sig_encode(sigma_agg: &LemurAggSig, n_signers: usize, pp: &LemurPp) -> Vec<u8> {
    let profile = pp.profile;
    let enc = compute_agg_encoding(pp, n_signers);

    let mut out = vec![sigma_agg.attempt as u8];

    // Z_agg: N-dependent fixed-width
    let z_polys = flat_to_polys(&sigma_agg.z_agg.0, profile.m, profile.d);
    out.extend(vec_serial(&z_polys, enc.zagg_dx, enc.zagg_bound));

    // Babai-encoded path labels
    for label in &sigma_agg.d_agg.path_labels {
        let hint = proj_label_with_profile(label, profile);
        let encoded = babai_encode_label_with_profile(label, &hint, profile);
        for (a_star, alphas) in &encoded {
            let alpha_polys = flat_to_polys(alphas, profile.kappa - 1, profile.d);
            match &enc.babai_mode {
                EncMode::Rice { rice_k } => {
                    out.extend(poly_serial_rice(a_star, *rice_k, enc.babai_bound));
                    out.extend(vec_serial_rice(&alpha_polys, *rice_k, enc.babai_bound));
                }
                EncMode::Fixed { dx } => {
                    out.extend(poly_serial(a_star, *dx, enc.babai_bound));
                    out.extend(vec_serial(&alpha_polys, *dx, enc.babai_bound));
                }
            }
        }
    }

    // Sibling labels
    let n_label = profile.omega * profile.kappa;
    for label in &sigma_agg.d_agg.sibling_labels {
        let polys = flat_to_polys(label, n_label, profile.d);
        match &enc.agg_mode {
            EncMode::Rice { rice_k } => {
                out.extend(vec_serial_rice(&polys, *rice_k, enc.agg_bound));
            }
            EncMode::Fixed { dx } => {
                out.extend(vec_serial(&polys, *dx, enc.agg_bound));
            }
        }
    }

    // u
    let n_u = profile.k * profile.n * profile.kappa_prime;
    let u_polys = flat_to_polys(&sigma_agg.d_agg.u, n_u, profile.d);
    match &enc.agg_mode {
        EncMode::Rice { rice_k } => {
            out.extend(vec_serial_rice(&u_polys, *rice_k, enc.agg_bound));
        }
        EncMode::Fixed { dx } => {
            out.extend(vec_serial(&u_polys, *dx, enc.agg_bound));
        }
    }

    out
}

/// Decode aggregated signature.
pub fn agg_sig_decode(
    data: &[u8],
    pp: &LemurPp,
    t: usize,
    n_signers: usize,
) -> Result<LemurAggSig, LemurError> {
    if data.is_empty() {
        return Err(LemurError::InvalidEncoding("empty data".into()));
    }
    let profile = pp.profile;
    let tau = pp.hvc_pp.tau;
    let enc = compute_agg_encoding(pp, n_signers);

    let attempt = data[0] as usize;
    let mut off = 1;

    // Z_agg
    let (z_polys, l) = vec_deserial(
        &data[off..],
        enc.zagg_dx,
        profile.m,
        profile.d,
        enc.zagg_bound,
    )?;
    let z_agg = KotsSig(polys_to_flat(&z_polys));
    off += l;

    // Babai path labels
    let mut path_encoded = Vec::with_capacity(tau);
    for _ in 0..tau {
        let mut level = Vec::with_capacity(profile.omega);
        for _ in 0..profile.omega {
            let (a_star, alphas_flat) = match &enc.babai_mode {
                EncMode::Rice { rice_k } => {
                    let (a_list, la) =
                        poly_deserial_rice(&data[off..], profile.d, *rice_k, enc.babai_bound)?;
                    off += la;
                    let (al_list, la) = vec_deserial_rice(
                        &data[off..],
                        profile.kappa - 1,
                        profile.d,
                        *rice_k,
                        enc.babai_bound,
                    )?;
                    off += la;
                    (a_list, polys_to_flat(&al_list))
                }
                EncMode::Fixed { dx } => {
                    let (a_list, la) =
                        poly_deserial(&data[off..], *dx, profile.d, enc.babai_bound)?;
                    off += la;
                    let (al_list, la) = vec_deserial(
                        &data[off..],
                        *dx,
                        profile.kappa - 1,
                        profile.d,
                        enc.babai_bound,
                    )?;
                    off += la;
                    (a_list, polys_to_flat(&al_list))
                }
            };
            level.push((a_star, alphas_flat));
        }
        path_encoded.push(level);
    }

    // Sibling labels
    let n_label = profile.omega * profile.kappa;
    let mut sibling_labels = Vec::with_capacity(tau);
    for _ in 0..tau {
        let (polys, l) = match &enc.agg_mode {
            EncMode::Rice { rice_k } => {
                vec_deserial_rice(&data[off..], n_label, profile.d, *rice_k, enc.agg_bound)?
            }
            EncMode::Fixed { dx } => {
                vec_deserial(&data[off..], *dx, n_label, profile.d, enc.agg_bound)?
            }
        };
        sibling_labels.push(polys_to_flat(&polys));
        off += l;
    }

    // u
    let n_u = profile.k * profile.n * profile.kappa_prime;
    let (u_polys, l) = match &enc.agg_mode {
        EncMode::Rice { rice_k } => {
            vec_deserial_rice(&data[off..], n_u, profile.d, *rice_k, enc.agg_bound)?
        }
        EncMode::Fixed { dx } => vec_deserial(&data[off..], *dx, n_u, profile.d, enc.agg_bound)?,
    };
    let u = polys_to_flat(&u_polys);
    off += l;

    if off != data.len() {
        return Err(LemurError::InvalidEncoding(format!(
            "trailing bytes: consumed {off}, total {}",
            data.len()
        )));
    }

    let path_labels =
        reconstruct_path_labels_agg(&pp.hvc_pp, t, &path_encoded, &sibling_labels, &u);

    Ok(LemurAggSig {
        z_agg,
        d_agg: HvcOpening {
            path_labels,
            sibling_labels,
            u,
        },
        attempt,
    })
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

/// Return serialized byte counts for all object types.
pub fn sizes(pp: &LemurPp, n_signers: usize) -> Vec<(String, usize)> {
    let profile = pp.profile;
    let tau = pp.hvc_pp.tau;
    let dx_pk = logq_hvc(profile);
    let dx_z = bits_z(profile);
    let dx_d = bits_dig(profile);
    let pb_pk = poly_bytes(profile.d, dx_pk);
    let pb_z = poly_bytes(profile.d, dx_z);
    let pb_dig = poly_bytes(profile.d, dx_d);
    let n_label = profile.omega * profile.kappa;
    let n_u = profile.k * profile.n * profile.kappa_prime;

    let enc = compute_agg_encoding(pp, n_signers);
    let pb_zagg = poly_bytes(profile.d, enc.zagg_dx);

    let sig_ind = sig_bytes_with_profile(tau, profile);

    let eta = profile.eta as f64;
    let alpha_w = profile.alpha_w as f64;
    let var_digit = eta * (eta + 1.0) / 3.0;
    let sigma_label = (n_signers as f64 * alpha_w * var_digit).sqrt();
    let sigma_babai = sigma_label / (2.0 * eta);

    let rice_poly_bytes_est = |sigma: f64, rice_k: usize| -> usize {
        let bits_per_coeff = rice_bits_per_coef(sigma, rice_k);
        (profile.d as f64 * bits_per_coeff / 8.0).ceil() as usize
    };

    let (babai_desc, pb_babai) = match &enc.babai_mode {
        EncMode::Rice { rice_k } => (
            format!("Rice k={rice_k}"),
            rice_poly_bytes_est(sigma_babai, *rice_k),
        ),
        EncMode::Fixed { dx } => (format!("fixed {dx}b"), poly_bytes(profile.d, *dx)),
    };
    let babai_total = tau * profile.omega * profile.kappa * pb_babai;

    let (agg_desc, pb_agg) = match &enc.agg_mode {
        EncMode::Rice { rice_k } => (
            format!("Rice k={rice_k}"),
            rice_poly_bytes_est(sigma_label, *rice_k),
        ),
        EncMode::Fixed { dx } => (format!("fixed {dx}b"), poly_bytes(profile.d, *dx)),
    };
    let sib_total = tau * n_label * pb_agg;
    let u_total = n_u * pb_agg;

    let agg_total = 1 + profile.m * pb_zagg + babai_total + sib_total + u_total;
    let label = match (&enc.babai_mode, &enc.agg_mode) {
        (EncMode::Rice { .. }, _) | (_, EncMode::Rice { .. }) => "~",
        _ => "",
    };

    fn fmt(n: usize) -> String {
        if n < 1024 {
            format!("{n} B")
        } else if n < 1024 * 1024 {
            format!("{:.1} KB", n as f64 / 1024.0)
        } else {
            format!("{:.2} MB", n as f64 / (1024.0 * 1024.0))
        }
    }

    // Fresh BDS state (phi=0) size: header + tau auth labels + tau keep
    // presence bytes (all 0) + tau retain count headers + (2^k - k - 1)
    // retain labels (pre-computed right-sibling queue) + (tau - k)
    // treehash records each holding one completed node.
    let bds_k = crate::hvc::bds_choose_k(tau);
    let n_retain_labels = if tau >= 2 {
        (1usize << bds_k).saturating_sub(bds_k + 1)
    } else {
        0
    };
    let n_treehash = tau.saturating_sub(bds_k);
    let lb = label_bytes(profile);
    let fresh_state_bytes = SK_STATE_HEADER_SIZE
        + tau * lb
        + tau
        + 2 * tau
        + n_retain_labels * lb
        + n_treehash * (12 + lb);

    vec![
        ("pp (seeds + tau)".into(), PP_BYTES),
        ("sk (master seed)".into(), 32),
        ("sk.state (fresh BDS)".into(), fresh_state_bytes),
        ("pk (HVC commitment)".into(), profile.omega * pb_pk),
        ("individual sig".into(), sig_ind),
        ("  Z (KOTS sig)".into(), profile.m * pb_z),
        ("  sibling labels".into(), tau * n_label * pb_dig),
        ("  u".into(), n_u * pb_dig),
        (
            format!("aggregated sig (N={n_signers}, {label}{})", fmt(agg_total)),
            agg_total,
        ),
        (
            format!("  Z_agg ({}b, bound={})", enc.zagg_dx, enc.zagg_bound),
            1 + profile.m * pb_zagg,
        ),
        (format!("  Babai path ({babai_desc})"), babai_total),
        (format!("  sibling labels ({agg_desc})"), sib_total),
        (format!("  u ({agg_desc})"), u_total),
    ]
}

#[cfg(test)]
mod tests {
    use super::{sk_decode, sk_encode};

    #[test]
    fn seed_sk_roundtrip_is_exact_32_bytes() {
        let seed = [0x11u8; 32];
        let enc = sk_encode(&seed);
        assert_eq!(enc.len(), 32);
        assert_eq!(sk_decode(&enc).unwrap(), seed);
    }
}
