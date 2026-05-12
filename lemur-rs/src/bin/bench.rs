//! Benchmark binary for Lemur.
//!
//! Measures key generation, signing (online + offline), aggregation,
//! and batch verification.
//!
//! Usage:
//!   cargo run --release --bin bench
//!   cargo run --release --bin bench -- --fast   # fewer signers

use std::time::{Duration, Instant};

use lemur_rs::codec::{agg_sig_encode, sig_decode, sig_encode, sizes};
use lemur_rs::hvc::{add_openings, scale_opening_any};
use lemur_rs::kots::{kots_keygen, kots_keygen_ctx, kots_pk_from_seed, kots_sign, KotsSig};
use lemur_rs::lemur::{
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile,
    lemur_sign_seed, lemur_sign_stateful_mut, scale_mat_crt, LemurAggSig,
};
use lemur_rs::materialized::{lemur_sign_tree, MaterializedHvcTree};
use lemur_rs::params::{GAUSS_CDT_BITS, GAUSS_TAILCUT};
use lemur_rs::profile::{Profile, D256_K4};
use lemur_rs::sample::{
    agg_randomizer_xof, build_cdt, slot_seed, xof_gauss_poly_ctx, xof_gauss_poly_with_profile,
    xof_ternary_poly, GaussCtx,
};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

/// Const-generic version of `sample_one_gauss_ctx`: the byte width is a
/// compile-time constant so the compiler constant-folds the slice bound
/// and monomorphises the XOF read.  Used only by the sampler-experiment
/// microbench to compare apples-to-apples "what would we gain if
/// `GAUSS_CDT_BYTES` were set to N at compile time".
#[inline(always)]
fn sample_one_gauss_const<const CDT_BYTES: usize>(xof: &mut dyn XofReader, cdt: &[u32]) -> i64 {
    let mut u_bytes = [0u8; 4];
    xof.read(&mut u_bytes[..CDT_BYTES]);
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
    if sign == 0 {
        k
    } else {
        -k
    }
}

/// Sampler-microbench dimension: matches the largest shipped d (256).
/// Bench-only constant — the scheme uses `profile.d`.
const BENCH_D: usize = 256;

#[inline(never)]
fn bench_gauss_poly_const<const CDT_BYTES: usize>(
    xof: &mut dyn XofReader,
    cdt: &[u32],
) -> Vec<i64> {
    (0..BENCH_D)
        .map(|_| sample_one_gauss_const::<CDT_BYTES>(xof, cdt))
        .collect()
}

/// Batched variant: pulls all `BENCH_D * CDT_BYTES` bytes in one XOF
/// read, then runs the sampler off the buffer.  Amortises per-call
/// XofReader overhead, which otherwise dominates for small CDT widths.
#[inline(never)]
fn bench_gauss_poly_batched_const<const CDT_BYTES: usize>(
    xof: &mut dyn XofReader,
    cdt: &[u32],
) -> Vec<i64> {
    let mut buf = vec![0u8; BENCH_D * CDT_BYTES];
    xof.read(&mut buf);
    let mut out = Vec::with_capacity(BENCH_D);
    for chunk in buf.chunks_exact(CDT_BYTES) {
        let mut u_bytes = [0u8; 4];
        u_bytes[..CDT_BYTES].copy_from_slice(chunk);
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
        out.push(if sign == 0 { k } else { -k });
    }
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 60.0 {
        format!("{:.1} min", secs / 60.0)
    } else if secs >= 1.0 {
        format!("{:.2} s", secs)
    } else if secs >= 1e-3 {
        format!("{:.2} ms", secs * 1e3)
    } else {
        format!("{:.1} us", secs * 1e6)
    }
}

fn fmt_bytes(n: usize) -> String {
    if n >= 1_048_576 {
        format!("{:.2} MB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{} B", n)
    }
}

fn concat_pk_bytes(pks: &[lemur_rs::lemur::LemurPk]) -> Vec<u8> {
    pks.iter()
        .flat_map(|pk| pk.0 .0.iter().flat_map(|&x| x.to_le_bytes()))
        .collect()
}

fn aggregate_verified_only(
    pp: &lemur_rs::lemur::LemurPp,
    pks: &[lemur_rs::lemur::LemurPk],
    t: usize,
    msg: &[u8],
    sigs: &[lemur_rs::lemur::LemurSig],
) -> LemurAggSig {
    let profile = pp.profile;
    let n = pks.len();
    let pks_bytes = concat_pk_bytes(pks);
    let ell = profile.ell;
    let m = profile.m;
    let gamma = profile.gamma;
    let d = profile.d;

    for attempt in 1..=gamma {
        let mut xof = agg_randomizer_xof(t, msg, &pks_bytes, attempt);
        let ws: Vec<_> = (0..n)
            .map(|_| xof_ternary_poly(&mut xof, profile.alpha_w, profile.d))
            .collect();

        let z_agg = if let Some(cfg) = profile.kots_crt() {
            let backend = cfg.backend();
            sigs.iter()
                .zip(ws.iter())
                .map(|(sig, w)| scale_mat_crt(w, &sig.z.0, ell, m, &backend))
                .fold(vec![0i64; ell * m * d], |mut acc, scaled| {
                    for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                        *a += *b;
                    }
                    acc
                })
        } else if let Some(kots_rp64) = profile.kots_ring64.as_ref() {
            sigs.iter()
                .zip(ws.iter())
                .map(|(sig, w)| lemur_rs::poly::scale_mat_u64(w, &sig.z.0, ell, m, kots_rp64))
                .fold(vec![0i64; ell * m * d], |mut acc, scaled| {
                    for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                        *a += *b;
                    }
                    acc
                })
        } else {
            let kots_rp = profile.kots_ring_u32();
            sigs.iter()
                .zip(ws.iter())
                .map(|(sig, w)| lemur_rs::poly::scale_mat(w, &sig.z.0, ell, m, kots_rp))
                .fold(vec![0i64; ell * m * d], |mut acc, scaled| {
                    for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                        *a += *b;
                    }
                    acc
                })
        };

        let mut d_agg = scale_opening_any(&ws[0], &sigs[0].opening, profile);
        for i in 1..n {
            let scaled = scale_opening_any(&ws[i], &sigs[i].opening, profile);
            d_agg = add_openings(&d_agg, &scaled);
        }

        let sigma_agg = LemurAggSig {
            z_agg: KotsSig(z_agg),
            d_agg,
            attempt,
        };
        if lemur_avrfy(pp, pks, t, msg, &sigma_agg).is_ok() {
            return sigma_agg;
        }
    }

    panic!("aggregation failed after {gamma} attempts")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn parse_opt_usize(args: &[String], key: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
}

fn parse_opt_f64(args: &[String], key: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fast = args.iter().any(|a| a == "--fast");
    let with_tree = args.iter().any(|a| a == "--with-tree");

    let profile: &'static Profile = &D256_K4;

    // Optional Gaussian-sampler experiment: swap in a runtime-built CDT
    // with a caller-chosen word width (and optional sigma) and time KOTS
    // keygen side-by-side against the default CDT.  Does not affect the
    // main Lemur benchmark below, which always uses the baked-in table.
    let cdt_bits_override = parse_opt_usize(&args, "--cdt-bits");
    let cdt_sigma_override = parse_opt_f64(&args, "--cdt-sigma");
    // Run only the sampler experiment and exit; skips the O(2^tau) tree
    // keygen / aggregation benchmarks for fast iteration.
    let sampler_only = args.iter().any(|a| a == "--sampler-only");

    let kots_seed = [0u8; 32];
    let hvc_seed = [1u8; 32];
    let slot = 0usize;
    let msg: &[u8] = b"benchmark message";

    let tau = profile.tau;
    let n_slots: usize = 1 << tau;
    let threads = rayon::current_num_threads();

    println!();
    println!(
        "=== Lemur Benchmark (profile={}, d={}, tau={tau}, \
         N_SLOTS={n_slots}, lambda=128) ===",
        profile.name, profile.d,
    );
    println!("  KOTS: k={}, m={}, n={}", profile.k, profile.m, profile.n);
    println!("  Threads: {threads}");
    println!("  Mode: {}", if fast { "fast (reduced N)" } else { "full" });

    // Setup
    let pp = lemur_setup_with_profile(&kots_seed, &hvc_seed, profile);

    // -----------------------------------------------------------------
    // Sampler experiment: KOTS keygen with a custom CDT (optional).
    // Runs up-front so it completes even if the user Ctrl-Cs out of
    // the long tau=21 keygen that follows; exits early if
    // --sampler-only was passed.
    // -----------------------------------------------------------------
    if cdt_bits_override.is_some() || cdt_sigma_override.is_some() || sampler_only {
        let bits = cdt_bits_override.unwrap_or(GAUSS_CDT_BITS);
        let sigma = cdt_sigma_override.unwrap_or(profile.sigma);
        let cdt_vec = build_cdt(sigma, bits, GAUSS_TAILCUT);
        let custom_ctx = GaussCtx {
            cdt: &cdt_vec,
            cdt_bytes: bits / 8,
        };
        let exp_seed = [4u8; 32];

        println!();
        println!(
            "--- Sampler experiment \
             (custom: sigma={sigma:.3}, cdt_bits={bits}, tailcut={GAUSS_TAILCUT}, \
             entries={}) ---",
            cdt_vec.len()
        );

        // Pure sampler microbenchmark: just xof_gauss_poly_ctx, no KOTS
        // keygen overhead.  Measures the sampler in isolation.
        let poly_reps: u32 = 200_000;
        let make_xof = || {
            let mut h = Shake256::default();
            h.update(&exp_seed);
            h.update(b"sampler-microbench");
            h.finalize_xof()
        };

        let profile_ctx = GaussCtx::from_profile(profile);

        // Warmup
        for _ in 0..1000 {
            let mut x = make_xof();
            let _ = xof_gauss_poly_with_profile(&mut x, profile);
            let mut x = make_xof();
            let _ = xof_gauss_poly_ctx(&mut x, &profile_ctx, profile.d);
            let mut x = make_xof();
            let _ = xof_gauss_poly_ctx(&mut x, &custom_ctx, profile.d);
        }

        // Production specialised path (indexed CDT via profile.cdt_hi).
        let wall = Instant::now();
        let mut acc: i64 = 0;
        for _ in 0..poly_reps {
            let mut x = make_xof();
            let p = xof_gauss_poly_with_profile(&mut x, profile);
            acc = acc.wrapping_add(p[0]);
        }
        let prod_per_poly = wall.elapsed() / poly_reps;

        let wall = Instant::now();
        for _ in 0..poly_reps {
            let mut x = make_xof();
            let p = xof_gauss_poly_ctx(&mut x, &profile_ctx, profile.d);
            acc = acc.wrapping_add(p[0]);
        }
        let default_per_poly = wall.elapsed() / poly_reps;

        let wall = Instant::now();
        for _ in 0..poly_reps {
            let mut x = make_xof();
            let p = xof_gauss_poly_ctx(&mut x, &custom_ctx, profile.d);
            acc = acc.wrapping_add(p[0]);
        }
        let custom_per_poly = wall.elapsed() / poly_reps;
        std::hint::black_box(acc);

        println!(
            "Pure sampler / poly of {} coeffs ({poly_reps} reps):",
            profile.d
        );
        println!(
            "  Production xof_gauss_poly_with_profile (profile-specialised): {} per poly",
            fmt_duration(prod_per_poly)
        );
        println!(
            "  xof_gauss_poly_ctx @ profile CDT (runtime-dynamic): {} per poly",
            fmt_duration(default_per_poly)
        );
        println!(
            "  xof_gauss_poly_ctx @ custom (cdt_bits={bits}, runtime-dynamic): {} per poly",
            fmt_duration(custom_per_poly)
        );

        // Const-folded reference — what a production build would look like
        // if GAUSS_CDT_BYTES were hard-coded to each size.  Strips the
        // runtime-dynamic overhead so this is the real "if we commit" number.
        macro_rules! bench_const {
            ($bytes:literal) => {{
                let cdt_local = build_cdt(sigma, $bytes * 8, GAUSS_TAILCUT);
                for _ in 0..1000 {
                    let mut x = make_xof();
                    let _ = bench_gauss_poly_const::<$bytes>(&mut x, &cdt_local);
                }
                let wall = Instant::now();
                let mut acc_c: i64 = 0;
                for _ in 0..poly_reps {
                    let mut x = make_xof();
                    let p = bench_gauss_poly_const::<$bytes>(&mut x, &cdt_local);
                    acc_c = acc_c.wrapping_add(p[0]);
                }
                let t = wall.elapsed() / poly_reps;
                std::hint::black_box(acc_c);
                (t, cdt_local.len())
            }};
        }

        println!("Const-folded per-sample-read reference (256 XofReader::read calls):");
        let (t32, n32) = bench_const!(4);
        let (t24, n24) = bench_const!(3);
        let (t16, n16) = bench_const!(2);
        let (t8, n8) = bench_const!(1);
        let t_default = t32;
        for (b, t, n) in [
            (32usize, t32, n32),
            (24, t24, n24),
            (16, t16, n16),
            (8, t8, n8),
        ] {
            let sp = t_default.as_secs_f64() / t.as_secs_f64().max(1e-12);
            println!(
                "  cdt_bits={b:<2} ({n:>3} entries): {} per poly  ({sp:.3}x vs 32-bit)",
                fmt_duration(t)
            );
        }

        // Batched-read variant: amortises per-call XofReader overhead.
        macro_rules! bench_batched {
            ($bytes:literal) => {{
                let cdt_local = build_cdt(sigma, $bytes * 8, GAUSS_TAILCUT);
                for _ in 0..1000 {
                    let mut x = make_xof();
                    let _ = bench_gauss_poly_batched_const::<$bytes>(&mut x, &cdt_local);
                }
                let wall = Instant::now();
                let mut acc_c: i64 = 0;
                for _ in 0..poly_reps {
                    let mut x = make_xof();
                    let p = bench_gauss_poly_batched_const::<$bytes>(&mut x, &cdt_local);
                    acc_c = acc_c.wrapping_add(p[0]);
                }
                let t = wall.elapsed() / poly_reps;
                std::hint::black_box(acc_c);
                (t, cdt_local.len())
            }};
        }

        println!("Const-folded batched-read reference (1 XofReader::read(D * cdt_bytes) call):");
        let (b32, _) = bench_batched!(4);
        let (b24, _) = bench_batched!(3);
        let (b16, _) = bench_batched!(2);
        let (b8, _) = bench_batched!(1);
        let b_default = b32;
        for (b, t) in [(32usize, b32), (24, b24), (16, b16), (8, b8)] {
            let sp = b_default.as_secs_f64() / t.as_secs_f64().max(1e-12);
            println!(
                "  cdt_bits={b:<2}: {} per poly  ({sp:.3}x vs batched-32)",
                fmt_duration(t)
            );
        }

        // Whole-KOTS-keygen view: how much of a full keygen is sampler?
        // Uses the fixed parameter set just like the rest of the run.
        let keygen_reps = 32u32;
        // Warm up
        let _ = kots_keygen(&pp.kots_a, &exp_seed, profile);
        let _ = kots_keygen_ctx(&pp.kots_a, &exp_seed, profile, &custom_ctx);

        let wall = Instant::now();
        for _ in 0..keygen_reps {
            let _ = kots_keygen(&pp.kots_a, &exp_seed, profile);
        }
        let default_keygen = wall.elapsed() / keygen_reps;
        let wall = Instant::now();
        for _ in 0..keygen_reps {
            let _ = kots_keygen_ctx(&pp.kots_a, &exp_seed, profile, &custom_ctx);
        }
        let custom_keygen = wall.elapsed() / keygen_reps;
        println!("KOTS keygen ({keygen_reps} reps, sampler + NTT + mat-mul):");
        println!(
            "  Default: {}  Custom: {}  ratio: {:.3}x",
            fmt_duration(default_keygen),
            fmt_duration(custom_keygen),
            default_keygen.as_secs_f64() / custom_keygen.as_secs_f64().max(1e-12)
        );
    }

    if sampler_only {
        return;
    }

    // -----------------------------------------------------------------
    // Key generation (1 rep — O(2^tau) leaf computations)
    // -----------------------------------------------------------------
    println!();
    println!("--- Key Generation (1 rep) ---");
    let wall = Instant::now();
    let (ref_sk, ref_sk_state, _ref_pk) = lemur_keygen(&pp, &[2u8; 32]);
    let kg_elapsed = wall.elapsed();
    println!("Key Generation: {}", fmt_duration(kg_elapsed));

    // -----------------------------------------------------------------
    // Online signing (KOTS sign only, message-dependent, fast)
    // -----------------------------------------------------------------
    let online_reps = 20u32;
    println!();
    println!("--- Online Signing (KOTS-only, {online_reps} reps) ---");
    let ss = slot_seed(&ref_sk.master_seed, slot);
    let (osk, _) = kots_keygen(&pp.kots_a, &ss, profile);
    let wall = Instant::now();
    for _ in 0..online_reps {
        let _ = kots_sign(&osk, msg, profile);
    }
    let online_mean = wall.elapsed() / online_reps;
    println!("Online Sign (mean): {}", fmt_duration(online_mean));

    // -----------------------------------------------------------------
    // Full signing (online + HVC open, 1 rep — O(2^tau))
    // -----------------------------------------------------------------
    println!();
    println!("--- Full Signing (online + HVC open, 1 rep) ---");
    let wall = Instant::now();
    let ref_sig = lemur_sign_seed(&pp, &ref_sk, slot, msg);
    let sign_elapsed = wall.elapsed();
    println!("Full Sign: {}", fmt_duration(sign_elapsed));
    println!(
        "  (offline HVC open ~{}, precomputable)",
        fmt_duration(sign_elapsed.saturating_sub(online_mean))
    );

    // -----------------------------------------------------------------
    // Stateful (BDS08) signing — amortised (tau - K) / 2 leaves per call
    // -----------------------------------------------------------------
    let stateful_reps = if fast { 8u32 } else { 32u32 };
    println!();
    println!("--- Stateful Signing (BDS08, mean over {stateful_reps} reps) ---");
    let mut state = ref_sk_state;
    let wall = Instant::now();
    for _ in 0..stateful_reps {
        let (_sig, _slot_used) =
            lemur_sign_stateful_mut(&pp, &mut state, msg, None).expect("stateful sign");
    }
    let stateful_total = wall.elapsed();
    let stateful_mean = stateful_total / stateful_reps;
    println!("Stateful Sign (mean): {}", fmt_duration(stateful_mean));
    let speedup = sign_elapsed.as_secs_f64() / stateful_mean.as_secs_f64().max(1e-9);
    println!("  vs Full Sign: {speedup:.0}x faster");

    // -----------------------------------------------------------------
    // Optional: materialised HVC tree — O(τ) offline signing at the cost
    // of about 8 GiB in-memory state for the shipped τ=20 profile.  Opt in
    // with `--with-tree`.
    //
    // Matches Chipmunk's memory-for-speed tradeoff (Chipmunk's analogous
    // tree at d=1024, the parameter point used in the paper's λ=128
    // Chipmunk comparison, is roughly 16 GB; this is the analogous Lemur
    // option).
    // Neither part of the secret key nor persisted to disk.
    // -----------------------------------------------------------------
    if with_tree {
        let tree_bytes = MaterializedHvcTree::byte_size_with_profile(tau, profile);
        println!();
        println!("--- Materialised HVC Tree (optional, opt-in via --with-tree) ---");
        println!(
            "Tree memory: {} ({:.2} GB)",
            fmt_bytes(tree_bytes),
            tree_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let leaf_fn = {
            let ms = ref_sk.master_seed;
            let kots_a = pp.kots_a.clone();
            move |t: usize| {
                let ss = slot_seed(&ms, t);
                kots_pk_from_seed(&kots_a, &ss, profile).0
            }
        };

        let wall = Instant::now();
        let tree = MaterializedHvcTree::build(&pp.hvc_pp, leaf_fn);
        let build_elapsed = wall.elapsed();
        println!("Tree Build: {}", fmt_duration(build_elapsed));

        // Sanity check: root must match the reference keygen commitment.
        assert_eq!(
            tree.commitment().0,
            _ref_pk.0 .0,
            "materialised tree root does not match keygen commitment"
        );

        let tree_reps = if fast { 32u32 } else { 128u32 };
        let wall = Instant::now();
        for i in 0..tree_reps {
            let t = (i as usize) & ((1 << tau) - 1);
            let _sig = lemur_sign_tree(&pp, &ref_sk, &tree, t, msg).expect("tree sign");
        }
        let tree_total = wall.elapsed();
        let tree_mean = tree_total / tree_reps;
        println!(
            "Tree Sign (mean over {tree_reps} reps): {}",
            fmt_duration(tree_mean)
        );
        let speedup_vs_full = sign_elapsed.as_secs_f64() / tree_mean.as_secs_f64().max(1e-9);
        let speedup_vs_stateful = stateful_mean.as_secs_f64() / tree_mean.as_secs_f64().max(1e-9);
        println!("  vs Full Sign:     {speedup_vs_full:.0}x faster");
        println!("  vs Stateful Sign: {speedup_vs_stateful:.1}x faster");

        // Drop the tree explicitly so its multi-GB allocation doesn't coexist with the
        // aggregation workload that follows.
        drop(tree);
    } else {
        println!();
        println!(
            "(Tree-backed signing benchmark skipped; pass --with-tree \
             to opt in — requires ~{} of free RAM.)",
            fmt_bytes(MaterializedHvcTree::byte_size_with_profile(tau, profile))
        );
    }

    // -----------------------------------------------------------------
    // Serialized sizes
    // -----------------------------------------------------------------
    println!();
    println!("--- Serialized Sizes ---");
    for (label, bytes) in sizes(&pp, 1024) {
        if label.starts_with("  ") {
            println!("  {:38} {}", label, fmt_bytes(bytes));
        } else {
            println!("  {:38} {}", format!("[{label}]"), fmt_bytes(bytes));
        }
    }

    // -----------------------------------------------------------------
    // Pre-generate signers for aggregation
    // -----------------------------------------------------------------
    let unique_signers: usize = if fast { 2 } else { 8 };
    // `--fast` still sweeps both N=1024 and N=8192 so size/timing rows are
    // directly comparable against Chipmunk's benchmark (which also covers
    // both). The extra cost beyond N=1024 is O(N) pre-verify + combine,
    // both linear in N, so this adds ~1–2 minutes.
    let ns: &[usize] = &[1024, 8192];
    let max_n = *ns.iter().max().unwrap_or(&1024);

    println!();
    println!(
        "--- Pre-generating {unique_signers} unique signers \
         (keygen+sign each) ---"
    );
    let prep_start = Instant::now();

    // Encode the first sig we already have
    let mut base_pks = Vec::with_capacity(unique_signers);
    let mut base_sig_bytes = Vec::with_capacity(unique_signers);

    // Signer 0: reuse ref_sk/ref_sig
    base_pks.push(_ref_pk);
    base_sig_bytes.push(sig_encode(&ref_sig, &pp.hvc_pp));
    eprintln!("  signer 1/{unique_signers} done (reused keygen+sign)");

    // Remaining signers
    for i in 1..unique_signers {
        let mut seed = [3u8; 32];
        seed[0] = (i & 0xFF) as u8;
        seed[1] = ((i >> 8) & 0xFF) as u8;
        let (sk_i, _sk_state_i, pk_i) = lemur_keygen(&pp, &seed);
        let sig_i = lemur_sign_seed(&pp, &sk_i, slot, msg);
        base_pks.push(pk_i);
        base_sig_bytes.push(sig_encode(&sig_i, &pp.hvc_pp));
        eprintln!(
            "  signer {}/{unique_signers} done ({:.0} s elapsed)",
            i + 1,
            prep_start.elapsed().as_secs_f64()
        );
    }
    println!(
        "  Setup time: {:.1} s ({unique_signers} signers)",
        prep_start.elapsed().as_secs_f64()
    );

    // Replicate to max_n via codec round-trip (LemurSig is not Clone)
    let mut all_pks = Vec::with_capacity(max_n);
    let mut all_sigs = Vec::with_capacity(max_n);
    for i in 0..max_n {
        let idx = i % unique_signers;
        all_pks.push(base_pks[idx].clone());
        all_sigs.push(sig_decode(&base_sig_bytes[idx], &pp, slot).expect("sig round-trip failed"));
    }
    println!("  Replicated {unique_signers} -> {max_n} signers");

    // -----------------------------------------------------------------
    // Aggregation & Batch Verification
    // -----------------------------------------------------------------
    for &n in ns {
        let pks = &all_pks[..n];
        let sigs = &all_sigs[..n];

        println!();
        println!("--- Aggregation (N={n}) ---");

        let wall = Instant::now();
        for (pk, sig) in pks.iter().zip(sigs.iter()) {
            assert!(
                lemur_ivrfy(&pp, pk, slot, msg, sig).is_ok(),
                "ivrfy failed for n={n}"
            );
        }
        let preverify_elapsed = wall.elapsed();
        println!("Individual Pre-Verify: {}", fmt_duration(preverify_elapsed));

        let wall = Instant::now();
        let agg_sig_verified = aggregate_verified_only(&pp, pks, slot, msg, sigs);
        let combine_elapsed = wall.elapsed();
        println!(
            "Aggregate After Verified Inputs: {}",
            fmt_duration(combine_elapsed)
        );

        let wall = Instant::now();
        let agg_sig = lemur_aggregate(&pp, pks, slot, msg, sigs).expect("aggregation failed");
        let agg_elapsed = wall.elapsed();
        println!("Secure Aggregation: {}", fmt_duration(agg_elapsed));
        assert_eq!(agg_sig.attempt, agg_sig_verified.attempt);
        assert_eq!(agg_sig.z_agg.0, agg_sig_verified.z_agg.0);
        assert_eq!(agg_sig.d_agg.u, agg_sig_verified.d_agg.u);
        assert_eq!(
            agg_sig.d_agg.path_labels,
            agg_sig_verified.d_agg.path_labels
        );
        assert_eq!(
            agg_sig.d_agg.sibling_labels,
            agg_sig_verified.d_agg.sibling_labels
        );
        let overhead_pct = if agg_elapsed.is_zero() {
            0.0
        } else {
            100.0 * preverify_elapsed.as_secs_f64() / agg_elapsed.as_secs_f64()
        };
        println!(
            "  Pre-Verify Share Of Secure Aggregation: {:.1}%",
            overhead_pct
        );

        let agg_bytes = agg_sig_encode(&agg_sig, n, &pp).len();
        println!(
            "Agg Sig Size: {} bytes ({})",
            agg_bytes,
            fmt_bytes(agg_bytes)
        );

        let wall = Instant::now();
        let ok = lemur_avrfy(&pp, pks, slot, msg, &agg_sig).is_ok();
        let vrfy_elapsed = wall.elapsed();
        assert!(ok, "avrfy failed for n={n}");
        println!("Batch Verify: {}", fmt_duration(vrfy_elapsed));
    }

    println!();
    println!("Notes:");
    println!(
        "  - Key gen and full sign are O(2^tau) = O({n_slots}) \
         leaf computations."
    );
    println!(
        "  - Online signing (KOTS-only) is O(1), \
         independent of tau."
    );
    println!(
        "  - Aggregation includes O(N) individual \
         pre-verification."
    );
    println!(
        "  - Signers replicated from {unique_signers} unique \
         keypairs for aggregation timing."
    );

    let _: Option<LemurAggSig> = None;
}
