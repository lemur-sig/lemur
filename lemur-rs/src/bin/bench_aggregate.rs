//! Sub-step timing breakdown for `lemur_aggregate` and `lemur_avrfy`.
//!
//! Times each sub-step in isolation at `N ∈ {1024, 8192}` so the paper
//! and README can attribute aggregation / verification wall-clock to the
//! individual operations.  The internal helpers are pulled in via the
//! `bench_internals` module of the lemur-rs crate.
//!
//! **Scope.**  This binary only measures the shipped implementation profile
//! `D256_K4`, which is pinned to the paper's `τ=20, N=2^10` cell
//! (`profile.n_signers = 1024`, `beta_agg = 175655`, `eta = 169`, `omega = 2`,
//! `kappa = 5`).  N=8192 is included as an in-profile scaling reference
//! (`bench`-compatible).  KOTS aggregation stays in the CRT-NTT path at both
//! N values when the auxiliary CRT headroom permits exact signed
//! reconstruction.  The paper's `N ∈ {2^15, 2^17, 2^20}` rows use different
//! parameter cells (larger `k, m, omega, eta, beta_agg`) and cannot be
//! measured by running this binary at N>8192 — that would only stress-test
//! the τ=20 N=1024 profile out of spec.
//!
//! Aggregation breakdown sub-steps (1-5) plus end-to-end `lemur_aggregate`:
//!     1. Individual pre-verify (rogue-key check, ×N, rayon-parallel)
//!     2. Randomizer derivation (SHAKE128 over `t || msg || concat(pks) || attempt`)
//!     3. KOTS aggregate `sum_i w_i * z_i`
//!     4. HVC opening aggregate `sum_i w_i * d_i`
//!     5. One avrfy probe (the close-loop check inside the retry loop)
//!
//! Batch-verify breakdown sub-steps (a-d) plus end-to-end `lemur_avrfy`:
//!     a. Randomizer derivation
//!     b. HVC commitment aggregate `sum_i w_i * T_i`
//!     c. HVC sVrfy (Babai decode + opening verify)
//!     d. KOTS sVrfy

use std::time::{Duration, Instant};

use lemur_rs::codec::{sig_decode, sig_encode};
use lemur_rs::hvc::{aggregate_openings_any, hvc_svrfy, HvcOpening};
use lemur_rs::kots::{kots_svrfy, KotsSig};
use lemur_rs::lemur::{
    bench_internals::{
        aggregate_kots_z, concat_pk_bytes_pub, hash_to_randomizers_pub,
        weighted_sum_commitments_pub,
    },
    lemur_aggregate, lemur_avrfy, lemur_ivrfy, lemur_keygen, lemur_setup_with_profile,
    lemur_sign_seed, LemurAggSig, LemurPk, LemurSig,
};
use lemur_rs::profile::{Profile, D256_K4};
use rayon::prelude::*;

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

fn time<F: FnMut()>(reps: u32, mut f: F) -> Duration {
    // Warm-up
    f();
    let wall = Instant::now();
    for _ in 0..reps {
        f();
    }
    wall.elapsed() / reps
}

fn main() {
    let profile: &'static Profile = &D256_K4;
    let kots_seed = [0u8; 32];
    let hvc_seed = [1u8; 32];
    let slot: usize = 0;
    let msg: &[u8] = b"benchmark message";

    let tau = profile.tau;
    let threads = rayon::current_num_threads();

    println!();
    println!(
        "=== Lemur Aggregate Breakdown \
         (profile={}, d={}, tau={tau}, lambda=128) ===",
        profile.name, profile.d,
    );
    println!("  Threads: {threads}");

    let pp = lemur_setup_with_profile(&kots_seed, &hvc_seed, profile);

    // ---------------------------------------------------------------
    // Pre-generate `unique_signers` real keys & sigs, then replicate
    // them to `max_n` via codec round-trip (LemurSig is not Clone).
    // Mirrors the prep strategy of `bench --fast`.
    // ---------------------------------------------------------------
    // N choice rationale:
    //   * 1024  -- the shipped D256_K4 profile cell (matches `profile.n_signers`).
    //   * 8192  -- in-profile scaling reference; KOTS Z aggregation still
    //              uses the CRT-NTT path when auxiliary-prime headroom allows.
    //
    // We deliberately do not include N ∈ {2^15, 2^17, 2^20} here: the paper's
    // larger-N rows use *different* parameter cells (larger `k, m, omega, eta,
    // beta_agg` — see `parameter/summary.txt`), so a D256_K4 run at those N
    // would not measure the paper's parameters, only stress-test the τ=20,
    // N=1024 profile out of spec.  For large-N batch verification, use
    // `bench_verify --zero-fixture`; for large-N aggregation timings, plug the
    // appropriate cell into `parameter/summary.txt` and instantiate a new
    // profile.
    let ns: &[usize] = &[1024, 8192];
    let max_n = *ns.iter().max().unwrap_or(&1024);

    // Unique signer count: the fixture replicates `unique_signers` real
    // (pk, sig) pairs to fill the N-slot test.  With low unique counts the
    // bunched per-pk randomizer sum amplifies the aggregated norms past
    // `beta_agg` and `lemur_aggregate` exhausts its `γ=10` retry budget.
    // Empirically at D256_K4:
    //   * replication ≤ 4096 (e.g. N=8192 with 2 signers) -- always succeeds
    //   * replication ≥ 8192 (e.g. N=32768 with 4 signers) -- can fail
    // 2 signers is sufficient for N ≤ 8192, which is the default sweep.
    let unique_signers: usize = 2;

    println!();
    println!("--- Pre-generating {unique_signers} unique signers (keygen+sign each) ---");
    let prep_start = Instant::now();
    let mut base_pks = Vec::with_capacity(unique_signers);
    let mut base_sig_bytes = Vec::with_capacity(unique_signers);
    for i in 0..unique_signers {
        let mut seed = [3u8; 32];
        seed[0] = (i & 0xFF) as u8;
        seed[1] = ((i >> 8) & 0xFF) as u8;
        let (sk_i, _state_i, pk_i) = lemur_keygen(&pp, &seed);
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

    let mut all_pks: Vec<LemurPk> = Vec::with_capacity(max_n);
    let mut all_sigs: Vec<LemurSig> = Vec::with_capacity(max_n);
    for i in 0..max_n {
        let idx = i % unique_signers;
        all_pks.push(base_pks[idx].clone());
        all_sigs.push(sig_decode(&base_sig_bytes[idx], &pp, slot).expect("sig round-trip failed"));
    }
    println!("  Replicated {unique_signers} -> {max_n} signers");

    // ---------------------------------------------------------------
    // Per-N breakdown
    // ---------------------------------------------------------------
    for &n in ns {
        let pks: &[LemurPk] = &all_pks[..n];
        let sigs: &[LemurSig] = &all_sigs[..n];

        println!();
        println!("--- Aggregation breakdown at N={n} ---");

        // Rep counts: per-step cost scales with N, so larger N gets fewer
        // reps.  Tuned so each N-row takes a few minutes wall-clock at
        // most.
        let reps_preverify: u32 = match n {
            n if n >= 32768 => 1,
            n if n >= 8192 => 2,
            _ => 3,
        };
        let reps_xof_pk_z_open: u32 = match n {
            n if n >= 32768 => 3,
            n if n >= 8192 => 5,
            _ => 10,
        };
        let reps_avrfy: u32 = match n {
            n if n >= 32768 => 2,
            n if n >= 8192 => 3,
            _ => 5,
        };
        let reps_full_agg: u32 = match n {
            n if n >= 32768 => 1,
            n if n >= 8192 => 2,
            _ => 3,
        };
        let reps_concat: u32 = match n {
            n if n >= 32768 => 5,
            n if n >= 8192 => 10,
            _ => 20,
        };

        // 1. Pre-verify (rayon-parallel, same as inside lemur_aggregate).
        let t_preverify = time(reps_preverify, || {
            pks.par_iter()
                .zip(sigs.par_iter())
                .try_for_each(|(pk, sig)| lemur_ivrfy(&pp, pk, slot, msg, sig))
                .expect("pre-verify failed");
        });

        // 2a. PK serialization (concat_pk_bytes).  Production code runs
        // this once inside `lemur_aggregate` and once inside `lemur_avrfy`;
        // we time it here as its own row so the breakdown is self-attributable.
        let t_concat = time(reps_concat, || {
            let b = concat_pk_bytes_pub(pks);
            std::hint::black_box(b);
        });

        // Build the randomizer-XOF inputs once for the per-step timings.
        let pks_bytes = concat_pk_bytes_pub(pks);

        // Run `lemur_aggregate` once up-front to discover the successful
        // attempt number.  Attempt 1 can fail the avrfy norm bound and the
        // retry loop advances.  Using the actual successful attempt guarantees
        // the synthetic aggregate built below mirrors a real, accepting
        // aggregate.
        let probe_agg = lemur_aggregate(&pp, pks, slot, msg, sigs).expect("aggregate probe failed");
        let success_attempt = probe_agg.attempt;
        drop(probe_agg);

        // 2b. Randomizer derivation (at the successful attempt).
        let t_randomizers = time(reps_xof_pk_z_open, || {
            let ws = hash_to_randomizers_pub(slot, msg, &pks_bytes, success_attempt, n, profile);
            std::hint::black_box(ws);
        });

        // Materialise ws once for the remaining steps.
        let ws = hash_to_randomizers_pub(slot, msg, &pks_bytes, success_attempt, n, profile);

        // 3. KOTS Z aggregation.
        let t_kots_z = time(reps_xof_pk_z_open, || {
            let z = aggregate_kots_z(&ws, sigs, profile);
            std::hint::black_box(z);
        });

        // 4. HVC opening aggregation.
        let opening_refs: Vec<&HvcOpening> = sigs.iter().map(|s| &s.opening).collect();
        let t_hvc_open = time(reps_xof_pk_z_open, || {
            let d_agg = aggregate_openings_any(&ws, &opening_refs, profile);
            std::hint::black_box(d_agg);
        });

        // Build a real aggregate so we can probe avrfy on it.  The
        // attempt number matches the one that `lemur_aggregate` succeeded
        // with above, so this aggregate is byte-identical to a real
        // accepting aggregate.
        let z_agg = aggregate_kots_z(&ws, sigs, profile);
        let d_agg = aggregate_openings_any(&ws, &opening_refs, profile);
        let sigma_agg = LemurAggSig {
            z_agg: KotsSig(z_agg),
            d_agg,
            attempt: success_attempt,
        };

        // Confirm avrfy actually accepts this aggregate (sanity).
        lemur_avrfy(&pp, pks, slot, msg, &sigma_agg)
            .expect("avrfy accept (synthetic aggregate from successful attempt)");

        // 5. One avrfy probe inside the retry loop.
        let t_avrfy_probe = time(reps_avrfy, || {
            let ok = lemur_avrfy(&pp, pks, slot, msg, &sigma_agg).is_ok();
            std::hint::black_box(ok);
        });

        // End-to-end secure aggregation.  Successful attempt was probed
        // up-front to populate `success_attempt`; we report that.
        let t_full_agg = time(reps_full_agg, || {
            let agg = lemur_aggregate(&pp, pks, slot, msg, sigs).expect("aggregate");
            debug_assert_eq!(agg.attempt, success_attempt);
            std::hint::black_box(agg);
        });

        // Print aggregation breakdown.
        let pct =
            |x: Duration| -> f64 { 100.0 * x.as_secs_f64() / t_full_agg.as_secs_f64().max(1e-12) };
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "1. Individual pre-verify (×N, rayon)",
            fmt_duration(t_preverify),
            pct(t_preverify)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "2a. PK serialization (concat_pk_bytes)",
            fmt_duration(t_concat),
            pct(t_concat)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "2b. Randomizer derivation (SHAKE128)",
            fmt_duration(t_randomizers),
            pct(t_randomizers)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "3. KOTS aggregate Σ wᵢ·zᵢ",
            fmt_duration(t_kots_z),
            pct(t_kots_z)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "4. HVC opening aggregate Σ wᵢ·dᵢ",
            fmt_duration(t_hvc_open),
            pct(t_hvc_open)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "5. avrfy probe (close-loop check)",
            fmt_duration(t_avrfy_probe),
            pct(t_avrfy_probe)
        );
        let sum_isolated =
            t_preverify + t_concat + t_randomizers + t_kots_z + t_hvc_open + t_avrfy_probe;
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "── Sum of isolated steps",
            fmt_duration(sum_isolated),
            pct(sum_isolated)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%   (attempts: {success_attempt})",
            "── End-to-end lemur_aggregate",
            fmt_duration(t_full_agg),
            100.0
        );

        // ---------------------------------------------------------
        // Batch verify (= lemur_avrfy) breakdown
        // ---------------------------------------------------------
        println!();
        println!("--- Batch Verify breakdown at N={n} ---");

        // a-prelude. PK serialization — same `concat_pk_bytes` call that
        // production avrfy makes once before the randomizer derivation.
        // Reuses the t_concat measurement from the aggregation block.
        let t_a_prelude = t_concat;

        // a. Randomizer derivation — same call, retimed in isolation.
        let t_a = time(reps_xof_pk_z_open, || {
            let ws = hash_to_randomizers_pub(slot, msg, &pks_bytes, 1, n, profile);
            std::hint::black_box(ws);
        });

        // b. HVC commitment aggregation Σ wᵢ·Tᵢ.
        let t_b = time(reps_xof_pk_z_open, || {
            let c = weighted_sum_commitments_pub(&ws, pks, profile);
            std::hint::black_box(c);
        });

        // c. HVC sVrfy.
        let c_agg = weighted_sum_commitments_pub(&ws, pks, profile);
        let t_c = time(reps_avrfy, || {
            let opk = hvc_svrfy(&pp.hvc_pp, &c_agg, slot, &sigma_agg.d_agg).expect("hvc_svrfy");
            std::hint::black_box(opk);
        });

        // d. KOTS sVrfy.
        let opk_agg = hvc_svrfy(&pp.hvc_pp, &c_agg, slot, &sigma_agg.d_agg).expect("hvc_svrfy");
        let t_d = time(reps_avrfy * 4, || {
            let ok = kots_svrfy(&pp.kots_a, &opk_agg, msg, &sigma_agg.z_agg, profile).is_ok();
            std::hint::black_box(ok);
        });

        // End-to-end lemur_avrfy.
        let t_avrfy_full = time(reps_avrfy, || {
            let ok = lemur_avrfy(&pp, pks, slot, msg, &sigma_agg).is_ok();
            std::hint::black_box(ok);
        });

        let pct_v = |x: Duration| -> f64 {
            100.0 * x.as_secs_f64() / t_avrfy_full.as_secs_f64().max(1e-12)
        };
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "a-prelude. PK serialization (concat_pk_bytes)",
            fmt_duration(t_a_prelude),
            pct_v(t_a_prelude)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "a. Randomizer derivation",
            fmt_duration(t_a),
            pct_v(t_a)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "b. HVC commitment aggregate Σ wᵢ·Tᵢ",
            fmt_duration(t_b),
            pct_v(t_b)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "c. HVC sVrfy (Babai decode + opening verify)",
            fmt_duration(t_c),
            pct_v(t_c)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "d. KOTS sVrfy",
            fmt_duration(t_d),
            pct_v(t_d)
        );
        let sum_v = t_a_prelude + t_a + t_b + t_c + t_d;
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "── Sum of isolated steps",
            fmt_duration(sum_v),
            pct_v(sum_v)
        );
        println!(
            "  {:<50}  {:>10}    {:>6.1}%",
            "── End-to-end lemur_avrfy",
            fmt_duration(t_avrfy_full),
            100.0
        );
    }

    println!();
    println!("Notes:");
    println!("  - 'avrfy probe' inside aggregation is the same as 'End-to-end lemur_avrfy';");
    println!("    listed separately so the aggregation column is self-attributable.");
    println!("  - Sub-step totals can exceed the end-to-end measurement when sub-steps");
    println!("    share rayon scheduling overhead; under-attribution by a small percent");
    println!("    indicates per-step setup amortises in the end-to-end path.");
}
