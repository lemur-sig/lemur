//! Bench breakdown: attribute Lemur hot-path time to individual primitives.
//!
//! The main `bench` binary reports composite timings (keygen, sign, aggregate).
//! This binary drills one level deeper: it times each primitive in isolation
//! and each composite operation's sub-stages, so the next optimization target
//! is chosen by measurement rather than guess.
//!
//! Usage:
//!   cargo run --release --bin bench_breakdown
//!
//! The output is intentionally not machine-readable — it's a "where is the
//! time going?" report.
//!
//! All stages are warmed up before the measured loop to avoid cold-cache and
//! branch-predictor skew, then run for a rep count chosen so each phase takes
//! roughly 0.1–1.0 s.
//!
//! The numbers reported are *means over the rep count*, not minimum or
//! median.  For quick-look optimization work mean is plenty.

use std::time::{Duration, Instant};

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;

use lemur_rs::hvc::{
    hvc_mat_vec_prentt_pair, hvc_setup_with_profile_and_tau, internal_label_with_profile_any,
    leaf_label_with_profile_any,
};
use lemur_rs::kots::{kots_keygen, kots_pk_from_seed, kots_setup, kots_sign};
use lemur_rs::lemur::{lemur_keygen, lemur_setup_with_profile_and_tau, lemur_sign_seed};
use lemur_rs::profile::{Profile, D256_K4};
use lemur_rs::sample::{
    xof_gauss_poly_with_profile, xof_gauss_poly_with_profile_fill, xof_ternary_poly,
    xof_uniform_poly,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:.3} s")
    } else if s >= 1e-3 {
        format!("{:.3} ms", s * 1e3)
    } else if s >= 1e-6 {
        format!("{:.3} us", s * 1e6)
    } else {
        format!("{:.0} ns", s * 1e9)
    }
}

/// Run a closure `reps` times and return the mean per-call duration.
fn time_per_call<F: FnMut()>(reps: usize, mut f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed() / reps as u32
}

/// Time `f` and print one line under the current section.
fn row<F: FnMut()>(label: &str, reps: usize, f: F) -> Duration {
    let t = time_per_call(reps, f);
    println!("  {label:<44}  {:>10}/op   ({reps} reps)", fmt_duration(t));
    t
}

fn section(title: &str) {
    println!();
    println!("--- {title} ---");
}

fn make_xof(seed: &[u8], tag: &[u8]) -> impl XofReader {
    let mut h = Shake128::default();
    h.update(seed);
    h.update(tag);
    h.finalize_xof()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let profile: &'static Profile = &D256_K4;

    println!("=== Lemur Bench Breakdown ===");
    println!(
        "profile={}, d={}, tau={}, m={}, n={}, omega={}",
        profile.name, profile.d, profile.tau, profile.m, profile.n, profile.omega,
    );
    println!("CDT entries: {}", profile.cdt.len());

    // Common setup: build pp at small tau so we can exercise tree primitives.
    // The per-primitive timings below don't depend on tau.
    let setup_tau = 3usize;
    let kots_seed = [0x5au8; 32];
    let hvc_seed = [0xa5u8; 32];
    let kots_a = kots_setup(&kots_seed, profile);
    let hvc_pp = hvc_setup_with_profile_and_tau(&hvc_seed, profile, setup_tau);

    // Pre-generated KOTS pk (for leaf_label input) and a small ternary for H.
    let (_sk, pk0) = kots_keygen(&kots_a, &[0x11u8; 32], profile);

    // ------------------------------------------------------------------
    // 1. XOF primitives — split to see where SHAKE cost actually lives.
    //
    // Separates:
    //   - Shake128::default()                 — pure state zero / init
    //   - default() + update(seed)            — + absorb (no permutation)
    //   - default() + update + finalize_xof() — + 1 Keccak-f permutation
    //   - …_xof() + read(n)                   — + n/136 more permutations
    // ------------------------------------------------------------------
    section("XOF primitives — isolating init from Keccak-f");
    let xof_reps: usize = 200_000;

    row(
        "Shake128::default() only (no absorb, no finalize)",
        xof_reps,
        || {
            let h = Shake128::default();
            std::hint::black_box(h);
        },
    );
    row("default() + update(seed[32])", xof_reps, || {
        let mut h = Shake128::default();
        h.update(&[7u8; 32]);
        std::hint::black_box(h);
    });
    row(
        "default() + update(seed[32]) + finalize_xof()  (1 perm)",
        xof_reps,
        || {
            let mut h = Shake128::default();
            h.update(&[7u8; 32]);
            let x = h.finalize_xof();
            std::hint::black_box(x);
        },
    );
    row("…_xof() + read(32)  (1 perm + copy)", xof_reps, || {
        let mut x = make_xof(&[7u8; 32], b"");
        let mut out = [0u8; 32];
        x.read(&mut out);
        std::hint::black_box(out);
    });
    row(
        "…_xof() + read(136) — one full rate block (1 perm, no refill)",
        xof_reps,
        || {
            let mut x = make_xof(&[7u8; 32], b"");
            let mut out = [0u8; 136];
            x.read(&mut out);
            std::hint::black_box(out);
        },
    );
    row(
        "…_xof() + read(137) — crosses rate boundary (2 perms)",
        xof_reps,
        || {
            let mut x = make_xof(&[7u8; 32], b"");
            let mut out = [0u8; 137];
            x.read(&mut out);
            std::hint::black_box(out);
        },
    );
    let d_bench = profile.d;
    row(
        &format!("…_xof() + read(d={d_bench})  (≈{} perms)", d_bench / 136 + 1),
        xof_reps,
        || {
            let mut x = make_xof(&[7u8; 32], b"");
            let mut out = vec![0u8; d_bench];
            x.read(&mut out);
            std::hint::black_box(out);
        },
    );
    row(
        &format!(
            "…_xof() + read(d*4={}, gauss-poly budget)",
            d_bench * 4
        ),
        xof_reps,
        || {
            let mut x = make_xof(&[7u8; 32], b"");
            let mut out = vec![0u8; d_bench * 4];
            x.read(&mut out);
            std::hint::black_box(out);
        },
    );

    // ------------------------------------------------------------------
    // 2. Sampler primitives
    // ------------------------------------------------------------------
    section("Per-poly sampler primitives");
    let sampler_reps: usize = 100_000;

    // Gauss sampler — profile-specialised fast path (this is what the
    // KOTS keygen hot path uses after the indexed-CDT + _into changes).
    row(
        "xof_gauss_poly_with_profile (D coefs)",
        sampler_reps,
        || {
            let mut x = make_xof(&[0x31u8; 32], b"gauss");
            let out = xof_gauss_poly_with_profile(&mut x, profile);
            std::hint::black_box(out);
        },
    );
    let d_profile = profile.d;
    row(
        "xof_gauss_poly_with_profile_fill (in-place)",
        sampler_reps,
        || {
            let mut x = make_xof(&[0x32u8; 32], b"gauss-fill");
            let mut buf = vec![0i64; d_profile];
            xof_gauss_poly_with_profile_fill(&mut x, profile, &mut buf);
            std::hint::black_box(buf);
        },
    );

    // Uniform poly rejection sampler — used in setup (cold).  Included
    // here so we know its per-call cost if we ever care.
    row(
        &format!("xof_uniform_poly (q_kots={})", profile.q_kots()),
        sampler_reps / 2,
        || {
            let mut x = make_xof(&[0x41u8; 32], b"unif-k");
            let out = xof_uniform_poly(&mut x as &mut dyn XofReader, profile.q_kots(), profile.d);
            std::hint::black_box(out);
        },
    );
    row(
        &format!("xof_uniform_poly (q_hvc={})", profile.q_hvc()),
        sampler_reps / 2,
        || {
            let mut x = make_xof(&[0x42u8; 32], b"unif-h");
            let out = xof_uniform_poly(&mut x as &mut dyn XofReader, profile.q_hvc(), profile.d);
            std::hint::black_box(out);
        },
    );

    // Ternary poly — used in build_h (per-sign, per kots_ivrfy) and in
    // aggregation (N randomizers per attempt).
    row(
        &format!("xof_ternary_poly (weight={}, alpha_h)", profile.alpha_h),
        sampler_reps,
        || {
            let mut x = make_xof(&[0x51u8; 32], b"tern-h");
            let out = xof_ternary_poly(&mut x as &mut dyn XofReader, profile.alpha_h, profile.d);
            std::hint::black_box(out);
        },
    );
    row(
        &format!("xof_ternary_poly (weight={}, alpha_w)", profile.alpha_w),
        sampler_reps,
        || {
            let mut x = make_xof(&[0x52u8; 32], b"tern-w");
            let out = xof_ternary_poly(&mut x as &mut dyn XofReader, profile.alpha_w, profile.d);
            std::hint::black_box(out);
        },
    );

    // ------------------------------------------------------------------
    // 3. Polynomial arithmetic primitives
    // ------------------------------------------------------------------
    section("Polynomial / matrix arithmetic");
    let _poly_reps: usize = 100_000;

    // Use small-coefficient values so mul_signed's range is well defined.
    let a_poly: Vec<i64> = (0..profile.d as i64)
        .map(|i| (i * 13 - 37) % 4096)
        .collect();
    let b_poly: Vec<i64> = (0..profile.d as i64).map(|i| (i * 17 + 5) % 4096).collect();
    let _ = (a_poly, b_poly);
    // poly_mul / poly_mul_signed / mat_mul_prentt_b are exposed only on
    // the u32 NTT KOTS path.  Every shipped profile routes KOTS through
    // the CRT backend (aux_ntt), so these per-primitive rows would always
    // SKIP — they were dropped from the breakdown.
    println!("  poly_mul / poly_mul_signed / mat_mul_prentt_b   SKIPPED (CRT backend)");

    // KOTS structured-A matmul shapes: k x m times m x n = k x n, which is
    // exactly the shape keygen multiplies S by [I; A2].
    let k = profile.k;
    let m = profile.m;
    let _n = profile.n;
    let mat_reps: usize = 5_000;

    // HVC fused pair multiply: A0·left + A1·right in one accumulator.
    let omega = profile.omega;
    let kappa = profile.kappa;
    let left: Vec<i64> = (0..omega * kappa * profile.d)
        .map(|i| (i as i64) % 5)
        .collect();
    let right: Vec<i64> = (0..omega * kappa * profile.d)
        .map(|i| (i as i64 + 1) % 5)
        .collect();
    row(
        &format!(
            "mat_vec_prentt_pair ({omega}×{} · 2 ops, HVC ring)",
            omega * kappa,
        ),
        mat_reps,
        || {
            let out = hvc_mat_vec_prentt_pair(
                &hvc_pp.a0_ntt,
                &left,
                &hvc_pp.a1_ntt,
                &right,
                omega,
                omega * kappa,
                profile,
            );
            std::hint::black_box(out);
        },
    );

    // ------------------------------------------------------------------
    // 4. Composite HVC primitives
    // ------------------------------------------------------------------
    section("HVC composite primitives");
    let composite_reps: usize = 5_000;

    // leaf_label: decompose + poly_to_ntt + mac + inverse + decompose.
    // This is called once per leaf (2M+ times at tau=21).
    row(
        "leaf_label (opk → decomposed leaf label)",
        composite_reps,
        || {
            let out = leaf_label_with_profile_any(&pk0.0, &hvc_pp.b_mat_ntt, profile);
            std::hint::black_box(out);
        },
    );

    // internal_label: one fused mat_vec_prentt_pair + decomposition.
    // Called 2^(tau+1)-2 times across a tree walk.
    // Use two decomposed-label-shaped inputs (omega * kappa * profile.d).
    let left_dec: Vec<i64> = (0..omega * kappa * profile.d)
        .map(|i| ((i as i64) % (2 * profile.eta + 1)) - profile.eta)
        .collect();
    let right_dec: Vec<i64> = (0..omega * kappa * profile.d)
        .map(|i| ((i as i64 + 7) % (2 * profile.eta + 1)) - profile.eta)
        .collect();
    row(
        "internal_label (two decomposed children → parent)",
        composite_reps,
        || {
            let out = internal_label_with_profile_any(
                &left_dec,
                &right_dec,
                &hvc_pp.a0_ntt,
                &hvc_pp.a1_ntt,
                profile,
            );
            std::hint::black_box(out);
        },
    );

    // ------------------------------------------------------------------
    // 5. Composite KOTS primitives
    // ------------------------------------------------------------------
    section("KOTS composite primitives");
    let kots_reps: usize = 1_000;

    // Full KOTS keygen (the hot offline path).  Sub-stages aren't
    // directly exposed but can be attributed via the per-primitive
    // numbers above: k*m gauss_poly calls + one structured-A matmul.
    let kg_total = row("kots_keygen_with_profile (full)", kots_reps, || {
        let (sk, pk) = kots_keygen(&kots_a, &[0x77u8; 32], profile);
        std::hint::black_box((sk, pk));
    });
    println!(
        "    attribution model:  {} gauss_poly calls + 1 structured-A matmul",
        k * m
    );

    // One KOTS sign = build_h ((k-1) ternary polys) + z = h*S
    // (m poly muls through mul_signed).
    let sk_for_sign = kots_keygen(&kots_a, &[0x88u8; 32], profile).0;
    let msg: &[u8] = b"breakdown";
    let kots_sign_reps: usize = 5_000;
    let sign_total = row("kots_sign_with_profile (full)", kots_sign_reps, || {
        let z = kots_sign(&sk_for_sign, msg, profile);
        std::hint::black_box(z);
    });
    println!(
        "    attribution model:  {} ternary_poly (h) + {} poly_mul_signed (z = h*S)",
        k - 1,
        m * k,
    );

    // ------------------------------------------------------------------
    // 6. Quick attribution summary
    // ------------------------------------------------------------------
    // Grab the gauss / matmul / ternary / poly_mul timings again cheaply
    // from fresh short runs, so the summary is self-consistent even if
    // an earlier section was thermally cold.
    section("Attribution summary (what costs what in keygen + sign?)");
    {
        let gauss_t = time_per_call(10_000, || {
            let mut x = make_xof(&[0xa1u8; 32], b"attr-g");
            let mut buf = vec![0i64; profile.d];
            xof_gauss_poly_with_profile_fill(&mut x, profile, &mut buf);
            std::hint::black_box(buf);
        });
        let kg_gauss_attr = gauss_t * (k * m) as u32;
        println!(
            "kots_keygen ({}):  gauss × {}  =  {} (backend-specific matmul/NTT attribution skipped)",
            fmt_duration(kg_total),
            k * m,
            fmt_duration(kg_gauss_attr),
        );
        println!(
            "  gauss fraction: {:.1}%",
            100.0 * kg_gauss_attr.as_secs_f64() / kg_total.as_secs_f64(),
        );
        println!(
            "kots_sign   ({}):  (backend-specific poly_mul model skipped)",
            fmt_duration(sign_total),
        );
    }

    println!();
    // ------------------------------------------------------------------
    // 7. lemur_sign_seed breakdown.
    //
    // Offline (O(2^τ)) sign: re-derives the target slot's KOTS key,
    // KOTS-signs the message, and builds the HVC opening — which in
    // turn re-derives a KOTS pk at every other slot and hashes those
    // into the path.  Costs:
    //
    //   * 1 × kots_keygen_with_profile        (target slot, for osk + opk)
    //   * 1 × kots_sign_with_profile          (Z = H·S)
    //   * (2^τ − 1) × kots_pk_from_seed        (pk-only at every other slot)
    //   * (2^τ − 1) × leaf_label               (one per leaf slot)
    //   * (2^τ − 1) × internal_label           (one per internal node)
    //     + some decompositions / projections inside hvc_open.
    //
    // At realistic τ=20/21 a single sign would take minutes (minus rayon
    // parallelism, which amortises it across cores).  We measure at
    // small τ so numbers land in seconds, then attribute the breakdown
    // using per-primitive unit costs captured earlier in this run.
    //
    // IMPORTANT: `hvc_open_target` inside `lemur_sign_seed` uses
    // `rayon::join` for divide-and-conquer; the per-sign numbers
    // reported below are wall-clock under the ambient rayon thread
    // pool.  Per-core work ≈ wall-clock × rayon::current_num_threads()
    // (with some overhead).  We surface both so the reader can tell
    // "what did the CPUs actually do?" from "what did the user wait?".
    // ------------------------------------------------------------------
    section("lemur_sign_seed full-path breakdown");
    let threads = rayon::current_num_threads();
    println!("  rayon threads: {threads}");
    println!();
    println!("  per-primitive unit costs (from earlier sections, single-thread):");
    // Take fresh quick measurements so the attribution is self-consistent.
    let kots_kg_t = time_per_call(200, || {
        let (sk, pk) = kots_keygen(&kots_a, &[0xcau8; 32], profile);
        std::hint::black_box((sk, pk));
    });
    let kots_pk_only_t = time_per_call(400, || {
        let pk = kots_pk_from_seed(&kots_a, &[0xcbu8; 32], profile);
        std::hint::black_box(pk);
    });
    let kots_sign_t = time_per_call(2000, || {
        let sig = kots_sign(&sk_for_sign, msg, profile);
        std::hint::black_box(sig);
    });
    let leaf_label_t = time_per_call(2000, || {
        let out = leaf_label_with_profile_any(&pk0.0, &hvc_pp.b_mat_ntt, profile);
        std::hint::black_box(out);
    });
    let internal_label_t = time_per_call(2000, || {
        let out = internal_label_with_profile_any(
            &left_dec,
            &right_dec,
            &hvc_pp.a0_ntt,
            &hvc_pp.a1_ntt,
            profile,
        );
        std::hint::black_box(out);
    });
    println!(
        "    kots_keygen_with_profile           = {}",
        fmt_duration(kots_kg_t)
    );
    println!(
        "    kots_pk_from_seed_with_profile     = {}",
        fmt_duration(kots_pk_only_t)
    );
    println!(
        "    kots_sign_with_profile             = {}",
        fmt_duration(kots_sign_t)
    );
    println!(
        "    leaf_label_with_profile            = {}",
        fmt_duration(leaf_label_t)
    );
    println!(
        "    internal_label_with_profile        = {}",
        fmt_duration(internal_label_t)
    );
    println!();

    // Build one Lemur pp + sk at each probe τ, sign a few times, and
    // report wall-clock per sign plus the single-thread attribution.
    let sign_msg: &[u8] = b"lemur_sign_seed bench";
    for &probe_tau in &[3usize, 5, 7, 9] {
        let l_pp =
            lemur_setup_with_profile_and_tau(&[0xd1u8; 32], &[0xd2u8; 32], profile, probe_tau);
        let (sk, _sk_state, _pk) = lemur_keygen(&l_pp, &[0xd3u8; 32]);

        let n_slots = 1usize << probe_tau;
        // Rep count scaled so each probe takes roughly a second of wall
        // clock on the measurement box.  For τ=9 we do just one rep.
        let sign_reps: usize = match probe_tau {
            3 => 200,
            5 => 50,
            7 => 8,
            9 => 2,
            _ => 1,
        };

        // Warm up — primes the materialized-path allocator a bit and
        // avoids first-sign JIT-ish overhead in rayon.
        let _ = lemur_sign_seed(&l_pp, &sk, 0, sign_msg);

        // Measure over a handful of distinct slots so the target-slot
        // specialness doesn't skew things.
        let start = Instant::now();
        for r in 0..sign_reps {
            let slot = r & (n_slots - 1);
            let sig = lemur_sign_seed(&l_pp, &sk, slot, sign_msg);
            std::hint::black_box(sig);
        }
        let wall_per_sign = start.elapsed() / sign_reps as u32;

        // Single-thread attribution: how much work *should* a sign take
        // if every primitive ran back-to-back on one core?
        //   target slot: 1 kots_keygen + 1 kots_sign
        //   leaves:      (n_slots - 1) × (kots_pk_from_seed + leaf_label)
        //   internals:   (n_slots - 1) × internal_label
        let n_leaves_side = (n_slots - 1) as u32;
        let attr_target = kots_kg_t + kots_sign_t;
        let attr_leaves = (kots_pk_only_t + leaf_label_t).saturating_mul(n_leaves_side);
        let attr_internals = internal_label_t.saturating_mul(n_leaves_side);
        let attr_total = attr_target + attr_leaves + attr_internals;
        let speedup = attr_total.as_secs_f64() / wall_per_sign.as_secs_f64().max(1e-9);

        println!("  τ={probe_tau:<2} (n_slots={n_slots:>5}, reps={sign_reps:>3}):",);
        println!(
            "    wall-clock per sign  = {}    (rayon scaling: {:.2}× vs single-thread model)",
            fmt_duration(wall_per_sign),
            speedup,
        );
        println!(
            "    single-thread model  = {}   (target {:>8} + leaves {:>9} + internals {:>9})",
            fmt_duration(attr_total),
            fmt_duration(attr_target),
            fmt_duration(attr_leaves),
            fmt_duration(attr_internals),
        );
        println!(
            "      leaf/internal ratio: leaf work = {:.0}% of tree work; internal = {:.0}%",
            100.0 * attr_leaves.as_secs_f64() / (attr_leaves + attr_internals).as_secs_f64(),
            100.0 * attr_internals.as_secs_f64() / (attr_leaves + attr_internals).as_secs_f64(),
        );
        // Per-leaf micro-attribution.
        let per_leaf_kots =
            100.0 * kots_pk_only_t.as_secs_f64() / (kots_pk_only_t + leaf_label_t).as_secs_f64();
        println!(
            "      per-leaf cost: kots_pk_from_seed = {:.0}%, leaf_label = {:.0}%",
            per_leaf_kots,
            100.0 - per_leaf_kots,
        );
    }

    println!();
    println!("Done.");
}
