//! Batch-verification-only benchmark for large signer counts.
//!
//! This binary avoids the main benchmark's individual preverification and
//! one-signature-per-signer materialization.  It still constructs a valid
//! aggregate signature for the requested public-key list before timing
//! `lemur_avrfy`.
//!
//! Memory budget at `--n 1048576` under `D256_K4` (peaks):
//!
//! - replicated `pks` Vec        ≈ 4 GiB  (N · ω · d · 8 B)
//! - `concat_pk_bytes(pks)`      ≈ 4 GiB  (same shape, byte-flat for the XOF absorb)
//! - one chunk of randomizers    ≈ 8 MiB  (CHUNK_SIZE · d · 8 B, see `AGG_CHUNK`)
//! - per-thread scaled scratch   ≈ a few MiB
//!
//! Peak working set ≈ 9 GiB; budget for ~12 GiB free RAM.
//!
//! Usage:
//!   cargo run --release --bin bench_verify
//!   cargo run --release --bin bench_verify -- --n 1048576 --reps 1
//!   cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1

use std::env;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use lemur_rs::hvc::{add_openings, scale_opening_any, HvcCom, HvcOpening};
use lemur_rs::kots::KotsSig;
use lemur_rs::lemur::{
    lemur_avrfy, lemur_keygen, lemur_setup_with_profile, lemur_sign_seed, scale_vec_crt,
    LemurAggSig, LemurPk, LemurPp, LemurSig,
};
use lemur_rs::profile::{Profile, DEFAULT};
use lemur_rs::sample::{agg_randomizer_xof, xof_ternary_poly};

/// Rayon chunk size for the aggregation pass.  Bounds the per-chunk
/// randomizer buffer to `AGG_CHUNK · d · 8 B` (≈ 8 MiB at d = 256) and
/// caps the per-thread reduce tree depth, while leaving each chunk
/// large enough to keep all cores busy.
const AGG_CHUNK: usize = 4096;

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
    if n >= 1_073_741_824 {
        format!("{:.2} GiB", n as f64 / 1_073_741_824.0)
    } else if n >= 1_048_576 {
        format!("{:.2} MiB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn usage() -> ! {
    eprintln!(
        "Usage: bench_verify [--n N] [--tau TAU] [--unique U] [--slot T] [--reps R] [--zero-fixture]\n\
         Defaults: --n 1024 --tau 20 --unique 2 --slot 0 --reps 3\n\
         Paper-scale example: cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1"
    );
    std::process::exit(2);
}

fn parse_usize(args: &[String], i: &mut usize, flag: &str) -> usize {
    *i += 1;
    if *i >= args.len() {
        eprintln!("{flag} requires a value");
        usage();
    }
    args[*i].parse::<usize>().unwrap_or_else(|_| {
        eprintln!("{flag} requires a positive integer");
        usage();
    })
}

#[derive(Clone, Copy)]
struct Args {
    n: usize,
    tau: usize,
    unique: usize,
    slot: usize,
    reps: usize,
    zero_fixture: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let mut out = Args {
        n: 1024,
        tau: DEFAULT.tau,
        unique: 2,
        slot: 0,
        reps: 3,
        zero_fixture: false,
    };
    let mut unique_set = false;
    let mut slot_set = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => out.n = parse_usize(&args, &mut i, "--n"),
            "--tau" => out.tau = parse_usize(&args, &mut i, "--tau"),
            "--unique" => {
                unique_set = true;
                out.unique = parse_usize(&args, &mut i, "--unique");
            }
            "--slot" => {
                slot_set = true;
                out.slot = parse_usize(&args, &mut i, "--slot");
            }
            "--reps" => out.reps = parse_usize(&args, &mut i, "--reps"),
            "--zero-fixture" => out.zero_fixture = true,
            "-h" | "--help" => usage(),
            flag => {
                eprintln!("unknown argument: {flag}");
                usage();
            }
        }
        i += 1;
    }
    if out.n == 0 || out.unique == 0 || out.reps == 0 {
        eprintln!("--n, --unique, and --reps must be non-zero");
        usage();
    }
    if out.slot >= (1usize << out.tau) {
        eprintln!("--slot must be smaller than 2^tau");
        usage();
    }
    if out.zero_fixture {
        if unique_set {
            eprintln!("warning: --unique is ignored with --zero-fixture");
        }
        if slot_set {
            eprintln!("warning: --slot is ignored with --zero-fixture");
        }
    }
    out
}

fn concat_pk_bytes(pks: &[LemurPk]) -> Vec<u8> {
    pks.iter()
        .flat_map(|pk| pk.0 .0.iter().flat_map(|&x| x.to_le_bytes()))
        .collect()
}

fn scale_mat_profile(w: &[i64], z: &KotsSig, profile: &Profile) -> Vec<i64> {
    let m = profile.m;
    if let Some(cfg) = profile.kots_crt() {
        scale_vec_crt(w, &z.0, m, &cfg.backend())
    } else if let Some(rp64) = profile.kots_ring64.as_ref() {
        lemur_rs::poly::scale_mat_u64(w, &z.0, 1, m, rp64)
    } else {
        lemur_rs::poly::scale_mat(w, &z.0, 1, m, profile.kots_ring_u32())
    }
}

fn zero_fixture(pp: &LemurPp, n: usize) -> (Vec<LemurPk>, LemurAggSig) {
    let profile = pp.profile;
    let pk_len = profile.omega * profile.d;
    let label_len = profile.omega * profile.kappa * profile.d;
    let u_len = profile.k * profile.n * profile.kappa_prime * profile.d;
    let z_len = profile.m * profile.d;

    let pks = (0..n)
        .map(|_| LemurPk(HvcCom(vec![0i64; pk_len])))
        .collect();
    let d_agg = HvcOpening {
        path_labels: vec![vec![0i64; label_len]; pp.hvc_pp.tau],
        sibling_labels: vec![vec![0i64; label_len]; pp.hvc_pp.tau],
        u: vec![0i64; u_len],
    };
    let sigma_agg = LemurAggSig {
        z_agg: KotsSig(vec![0i64; z_len]),
        d_agg,
        attempt: 1,
    };
    (pks, sigma_agg)
}

/// Build a verifying aggregate over `pks` by reusing `unique_sigs` modulo
/// `unique_sigs.len()`.  Skips per-signer `lemur_ivrfy` (the canonical
/// `lemur_aggregate` path runs it on every signature, which dominates at
/// large N) and feeds randomizers in chunks so the working set stays
/// bounded as N grows.  Within each chunk, scaling and accumulation run
/// in parallel via rayon.
fn aggregate_repeated(
    pp: &LemurPp,
    pks: &[LemurPk],
    unique_sigs: &[LemurSig],
    t: usize,
    msg: &[u8],
) -> LemurAggSig {
    let profile = pp.profile;
    let pks_bytes = concat_pk_bytes(pks);
    let z_len = profile.m * profile.d;
    let n = pks.len();
    let n_unique = unique_sigs.len();

    for attempt in 1..=profile.gamma {
        let mut xof = agg_randomizer_xof(t, msg, &pks_bytes, attempt);
        let mut z_agg = vec![0i64; z_len];
        let mut d_agg: Option<HvcOpening> = None;

        let mut chunk_start = 0;
        while chunk_start < n {
            let chunk_end = (chunk_start + AGG_CHUNK).min(n);
            let chunk_n = chunk_end - chunk_start;

            // Sequential XOF derivation for this chunk (single stream).
            let ws: Vec<Vec<i64>> = (0..chunk_n)
                .map(|_| xof_ternary_poly(&mut xof, profile.alpha_w, profile.d))
                .collect();

            // Parallel scale + reduce for Z.
            let chunk_z = (0..chunk_n)
                .into_par_iter()
                .map(|i| {
                    let sig = &unique_sigs[(chunk_start + i) % n_unique];
                    scale_mat_profile(&ws[i], &sig.z, profile)
                })
                .reduce(
                    || vec![0i64; z_len],
                    |mut acc, scaled| {
                        for (a, b) in acc.iter_mut().zip(scaled.iter()) {
                            *a += *b;
                        }
                        acc
                    },
                );

            // Parallel scale + reduce for the HVC opening.
            let chunk_d = (0..chunk_n)
                .into_par_iter()
                .map(|i| {
                    let sig = &unique_sigs[(chunk_start + i) % n_unique];
                    scale_opening_any(&ws[i], &sig.opening, profile)
                })
                .reduce_with(|a, b| add_openings(&a, &b))
                .expect("non-empty chunk");

            // Fold this chunk into the running totals.
            for (a, b) in z_agg.iter_mut().zip(chunk_z.iter()) {
                *a += *b;
            }
            d_agg = Some(match d_agg {
                Some(ref acc) => add_openings(acc, &chunk_d),
                None => chunk_d,
            });

            chunk_start = chunk_end;
        }

        let sigma_agg = LemurAggSig {
            z_agg: KotsSig(z_agg),
            d_agg: d_agg.expect("non-empty public-key list"),
            attempt,
        };
        if lemur_avrfy(pp, pks, t, msg, &sigma_agg).is_ok() {
            return sigma_agg;
        }
    }

    panic!(
        "failed to construct a verifying aggregate in {} attempts",
        profile.gamma
    );
}

fn main() {
    let args = parse_args();
    let profile = DEFAULT;
    let msg: &[u8] = b"batch verification benchmark message";
    let kots_seed = [1u8; 32];
    let hvc_seed = [2u8; 32];

    println!("--- Batch Verification Benchmark ---");
    println!(
        "profile={}, tau={}, n={}, unique={}, slot={}, reps={}, zero_fixture={}",
        profile.name, args.tau, args.n, args.unique, args.slot, args.reps, args.zero_fixture
    );

    let setup_t0 = Instant::now();
    let pp = if args.tau == profile.tau {
        lemur_setup_with_profile(&kots_seed, &hvc_seed, profile)
    } else {
        lemur_rs::lemur::lemur_setup_with_profile_and_tau(&kots_seed, &hvc_seed, profile, args.tau)
    };
    println!("Setup: {}", fmt_duration(setup_t0.elapsed()));

    let pk_bytes = args.n * profile.omega * profile.d * std::mem::size_of::<i64>();
    println!(
        "Public-key vector raw coefficient storage: {}",
        fmt_bytes(pk_bytes)
    );

    let (pks, sigma_agg) = if args.zero_fixture {
        let fixture_t0 = Instant::now();
        let fixture = zero_fixture(&pp, args.n);
        println!(
            "Built zero verification fixture: {}",
            fmt_duration(fixture_t0.elapsed())
        );
        fixture
    } else {
        let basegen_t0 = Instant::now();
        let mut base_pks = Vec::with_capacity(args.unique);
        let mut base_sigs = Vec::with_capacity(args.unique);
        for i in 0..args.unique {
            let mut seed = [3u8; 32];
            seed[0] = (i & 0xff) as u8;
            seed[1] = ((i >> 8) & 0xff) as u8;
            let (sk, _sk_state, pk) = lemur_keygen(&pp, &seed);
            let sig = lemur_sign_seed(&pp, &sk, args.slot, msg);
            base_pks.push(pk);
            base_sigs.push(sig);
            eprintln!("  generated unique signer {}/{}", i + 1, args.unique);
        }
        println!(
            "Base keygen + sign ({} signers): {}",
            args.unique,
            fmt_duration(basegen_t0.elapsed())
        );

        let replicate_t0 = Instant::now();
        let mut pks = Vec::with_capacity(args.n);
        for i in 0..args.n {
            pks.push(base_pks[i % base_pks.len()].clone());
        }
        println!(
            "Replicated public keys: {}",
            fmt_duration(replicate_t0.elapsed())
        );

        let agg_t0 = Instant::now();
        let sigma_agg = aggregate_repeated(&pp, &pks, &base_sigs, args.slot, msg);
        println!(
            "Constructed aggregate for verification in {} (attempt={})",
            fmt_duration(agg_t0.elapsed()),
            sigma_agg.attempt
        );
        (pks, sigma_agg)
    };

    let mut samples = Vec::with_capacity(args.reps);
    for rep in 0..args.reps {
        let wall = Instant::now();
        let result = lemur_avrfy(&pp, &pks, args.slot, msg, &sigma_agg);
        let elapsed = wall.elapsed();
        result.expect("batch verification failed");
        samples.push(elapsed);
        println!("Batch Verify rep {}: {}", rep + 1, fmt_duration(elapsed));
    }

    let total: Duration = samples.iter().copied().sum();
    let mean = total / (samples.len() as u32);
    println!("Batch Verify mean: {}", fmt_duration(mean));
}
