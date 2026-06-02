# Lemur: Scalable Post-Quantum Synchronized Multi-Signatures

This repository accompanies the paper:

> Yini Lin, Muhammed F. Esgin, Amin Sakzad, Ron Steinfeld, and
> Markku-Juhani O. Saarinen.
> **"Lemur: Scalable Post-Quantum Synchronized Multi-Signatures"**.
> IACR Cryptology ePrint Archive, Report 2026/xxxx, 2026.
> <https://eprint.iacr.org/2026/xxxx>

Citation:

```bibtex
@misc{cryptoeprint:2026/xxxx,
  author       = {Yini Lin and Muhammed F. Esgin and Amin Sakzad and Ron Steinfeld and Markku-Juhani O. Saarinen},
  title        = {Lemur: Scalable Post-Quantum Synchronized Multi-Signatures},
  howpublished = {IACR Cryptology ePrint Archive, Report 2026/xxxx},
  year         = {2026},
  url          = {https://eprint.iacr.org/2026/xxxx}
}
```

The artifact contains three main deliverables:

| Directory | Purpose |
| --- | --- |
| `lemur-py/` | Python reference implementation of Lemur, including CLI tools, serialization, and cross-checkable test-vector generation. |
| `lemur-rs/` | Performance-oriented Rust implementation, including the CLI, tests, benchmarks, materialized-tree option, and byte-compatible serialization. |
| `parameter/` | Sage-based parameter estimator and notes explaining how the concrete implementation parameters are generated. |

The artifact is intended to let reviewers inspect the scheme implementation,
reproduce byte-level Python/Rust interoperability, run the Rust test suite and
benchmarks, and inspect the parameter-search methodology.

## Quick Start

### Python Reference

Run from `lemur-py/`:

```sh
cd lemur-py
python3 -m pip install -r requirements.txt

python3 ring.py
python3 kots.py
python3 hvc.py
python3 lemur.py

python3 cli.py vectors --tau 3 --signers 2 --slot 0 --msg "artifact check" --out /tmp/lemur-py-vectors.json
python3 cli.py sizes
```

The Python CLI supports `setup`, `keygen`, `sign`, `verify`, `aggregate`,
`batch-verify`, `sizes`, and `vectors`.  See `lemur-py/usage.md` and
`lemur-py/api.md` for details.

### Rust Implementation

Run from `lemur-rs/`:

```sh
cd lemur-rs
cargo test

cargo run --release --bin lemur -- vectors --tau 3 --signers 2 --slot 0 --msg "artifact check" --out /tmp/lemur-rs-vectors.json
cargo run --release --bin lemur -- sizes

cargo run --release --bin bench -- --fast
cargo run --release --bin bench_aggregate
cargo run --release --bin bench_breakdown
cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1
```

The Rust CLI mirrors the Python CLI.  The Rust crate also includes integration
tests for serialization, stateful signing, robustness against malformed inputs,
and Python/Rust-compatible test-vector behavior.

## Benchmarking And Sizes

The Python and Rust implementations ship the fixed
`d=256, k=4, tau=20, N=1024` parameter cell.  The paper also reports
estimator-derived rows for larger signer counts; those larger rows are not
separate implementation profiles in this artifact.

Run:

```sh
cd lemur-rs
cargo run --release --bin lemur -- sizes
cargo run --release --bin bench -- --fast
cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1
```

### Reproducing The Paper Implementation Table

The paper's implementation-performance table reports the shipped `tau=20,
N=2^10` implementation cell plus extrapolated or estimator-derived entries for
`N in {2^15, 2^20}`.

Run the deterministic size calculations:

```sh
cd lemur-rs
cargo run --release --bin lemur -- sizes --n 1024
```

This reproduces the implemented aggregated-signature-size cell (`N=2^10`,
185.5 KB).  The larger signature-size cells are predicted Rice-encoded sizes
for the corresponding `tau=20` rows in `parameter/summary.txt`, computed via:

```sh
cd parameter
python3 rice_sizes.py                       # totals only (paper Table 6 "Real" column)
python3 rice_sizes.py --breakdown           # tau=20 cells with KOTS vs HVC split
python3 rice_sizes.py --all --breakdown     # every (tau, N) cell, split
```

Aggregate-size breakdown for the `tau=20` cells (KB).  KOTS and HVC totals
sum (with a 1-byte attempt prefix) to the Rice-encoded total in the last
column; the worst-case column is the Table 5 fixed-width upper bound from
`parameter/summary.txt`.  HVC dominates at every cell (~97% of the bytes),
which is why the encoder only searches Rice for the HVC components — KOTS
`Z_agg` is always fixed-width with an `N`-dependent bound.

| N | Worst-case (Table 5) | KOTS `Z_agg` | HVC opening | Rice-encoded total |
| ---: | ---: | ---: | ---: | ---: |
| `2^10` | 232 KB | 5.6 KB | 179.9 KB | 185.5 KB |
| `2^15` | 315 KB | 6.9 KB | 257.6 KB | 264.5 KB |
| `2^20` | 444 KB | 7.8 KB | 371.7 KB | 379.6 KB |

Rice compresses **HVC only** (the KOTS savings vs Table 5 — 7→5.6, 8→6.9,
9→7.8 KB — come from the `N`-dependent statistical bound on `Z_agg`, not
Rice).  At `N=2^10` the HVC opening drops from 226 KB (Table 5) to 179.9 KB,
a ~20% reduction; because HVC dominates the bytes (~97%) this carries through
to the *total*, which drops from 232 KB to 185.5 KB.

Run the main timing benchmark:

```sh
cargo run --release --bin bench -- --fast
```

This produces the measured `N=2^10` entries used in the paper table:

| Row | Source line |
| --- | --- |
| Signing, BDS08 | `Stateful Signing (BDS08, mean ...)` |
| Aggregation, `N=2^10` | `Secure Aggregation` under `--- Aggregation (N=1024) ---` |
| Batch verification, `N=2^10` | `Batch Verify` under `--- Aggregation (N=1024) ---` |

The tree-in-memory signing row is optional because it materializes the HVC tree
and needs substantial RAM:

```sh
cargo run --release --bin bench -- --fast --with-tree
```

Use the `Tree Sign (mean ...)` line for the table.  At `tau=20`, the tree
allocation is about 8 GiB ((2^21 − 1) · ω · d · 8 B with ω=2, d=256).

Run the large-`N` verification-only benchmark:

```sh
cargo run --release --bin bench_verify -- --zero-fixture --n 32768 --reps 3
cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1
```

Use the `Batch Verify mean` lines for the `N=2^15` and `N=2^20` batch
verification cells.  The benchmark uses an accepting all-zero public-key and
aggregate-signature fixture so it measures `lemur_avrfy` without first running
large aggregation.

The large-`N` aggregation cells in the paper table are marked as extrapolated.  To
recompute them, take the measured `N=8192` `Secure Aggregation` time from
`bench --fast` and scale linearly.  This `N=8192` benchmark is an extrapolation
anchor and is not itself reported as a paper-table column:

```text
Aggregation(2^15) = Aggregation(8192) * 32768 / 8192
Aggregation(2^20) = Aggregation(8192) * 1048576 / 8192
```

For the 24-thread run on AMD Ryzen AI 9 HX 370 used in the paper, `Secure
Aggregation` at `N=8192` measures `2.79 s` (1 attempt on this fixture; the
2-unique-signer fixture replicated 4096-fold can occasionally force a retry,
but a deployment with N distinct signers would not).  Linear scaling gives
approximately `11.2 s` at `N=2^15` and `6.0 min` at `N=2^20`.

Representative serialized sizes for `tau=20, N=1024`:

| Object | Size |
| --- | ---: |
| Public parameters | 65 B |
| Secret seed | 32 B |
| Stateful signer cache | 110.0 KB |
| Public key | 2.7 KB |
| Individual signature | 78.2 KB |
| Aggregated signature, `N=1024` | 185.5 KB |

Representative `bench --fast` timings from a 24-thread run on AMD Ryzen AI 9
HX 370 (single-core boost up to 5.16 GHz; all-core sustained ≈ 3.5–4 GHz under
the default `powersave` governor):

| Operation | Time |
| --- | ---: |
| Key generation | 1.3 min |
| Online signing, KOTS only | 398 us |
| Full signing, including HVC open | 1.3 min |
| Stateful signing, BDS08 | 4.17 ms |
| Tree-backed signing (`--with-tree`) | 1.62 ms |
| Individual pre-verify, `N=1024` | 1.60 s |
| Aggregate after verified inputs, `N=1024` | 983 ms |
| Secure aggregation, `N=1024` | 1.16 s |
| Batch verification, `N=1024` | 15.0 ms |
| Individual pre-verify, `N=8192` | 12.89 s |
| Aggregate after verified inputs, `N=8192` | 1.66 s |
| Secure aggregation, `N=8192` | 2.79 s (1 attempt on this fixture) |
| Batch verification, `N=8192` | 111.2 ms |

Timings are machine-dependent and laptop-thermal-state sensitive; with
`powersave` governor and frequency-boost enabled, per-run noise of a few
percent is normal. To minimise jitter, use a stable thermal envelope (e.g.
desktop / server) or pin the CPU frequency with
`cpupower frequency-set`.

### Aggregation sub-step breakdown

`cargo run --release --bin bench_aggregate` produces an end-to-end-to-sub-step
attribution of `lemur_aggregate` and `lemur_avrfy`. Representative results at
$\tau=20$ on the same 24-thread machine:

**Aggregation (`lemur_aggregate`) sub-steps:**

| Step | `N=1024` | `N=8192` |
| --- | ---: | ---: |
| Individual pre-verify (×N, rayon) | 177.6 ms (15.4%) | 1.31 s (46.7%) |
| PK serialization (`concat_pk_bytes`) | 6.2 ms (0.5%) | 59.9 ms (2.1%) |
| Randomizer derivation (SHAKE128) | 5.4 ms (0.5%) | 43.6 ms (1.6%) |
| KOTS aggregate `Σ wᵢ·zᵢ` | 9.8 ms (0.8%) | 64.1 ms (2.3%) |
| HVC opening aggregate `Σ wᵢ·dᵢ` | 937.7 ms (81.2%) | 1.32 s (47.2%) |
| `avrfy` probe (close-loop check) | 15.5 ms (1.3%) | 113.7 ms (4.1%) |
| **End-to-end `lemur_aggregate`** | **1.16 s** | **2.80 s** |

**Batch verification (`lemur_avrfy`) sub-steps:**

| Step | `N=1024` | `N=8192` |
| --- | ---: | ---: |
| PK serialization | 6.2 ms (40.0%) | 59.9 ms (52.6%) |
| Randomizer derivation | 5.3 ms (34.1%) | 46.8 ms (41.1%) |
| HVC commitment aggregate `Σ wᵢ·Tᵢ` | 0.89 ms (5.8%) | 5.63 ms (4.9%) |
| HVC sVrfy (Babai decode + verify) | 2.72 ms (17.5%) | 1.30 ms (1.1%) |
| KOTS sVrfy | 0.56 ms (3.6%) | 0.27 ms (0.2%) |
| **End-to-end `lemur_avrfy`** | **15.5 ms** | **114.0 ms** |

`bench_aggregate` is also the source of the new NTT-domain aggregation
microbenchmark used in the paper's implementation-notes section.

#### Scope: this profile only measures the `N=2^10` parameter cell

The Rust artifact ships a single parameter profile, `D256_K4`, with
`profile.n_signers = 1024`, `beta_agg = 175 655`, `eta = 169`, `omega = 2`,
`kappa = 5`. The paper's `N ∈ {2^15, 2^17, 2^20}` rows are **different
parameter cells** (larger `k, m, omega, eta, beta_agg`; see
`parameter/summary.txt`), not just the same scheme at larger N. Running
`bench_aggregate` at N > 8192 against the shipped profile would only
stress-test it out of spec, not measure the paper's larger-N cells.

For the paper's `N=2^10` row, use the `bench_aggregate` numbers above. For
`N ∈ {2^15, 2^17, 2^20}`:

- **Sizes**: reproduce via `python3 parameter/rice_sizes.py`, which applies
  the Rice-encoding cost model to the appropriate `summary.txt` row (no
  separate implementation profile needed).
- **Batch-verification timings**: use `bench_verify --zero-fixture`, which
  times only `lemur_avrfy` on an accepting all-zero PK/aggregate fixture
  (no per-N aggregation prep needed, no profile switch needed for the
  asymptotic behaviour).
- **Aggregation timings**: keep as extrapolated from the `bench --fast`
  `N=8192` row, noting that the linear scaling is approximate because the
  larger paper rows use different parameter cells. The D256_K4 KOTS
  aggregate remains on the CRT-NTT path at `N=8192` when auxiliary-prime
  headroom permits exact signed reconstruction. A more faithful measurement
  would require instantiating a fresh profile for each `summary.txt` row.

Note for future maintainers: `bench_aggregate`'s fixture replicates 2 unique
signer keypairs to fill the N-slot test. At replication ≥ 4096, the
bunched-randomizers per-pk sum can amplify the aggregated norms past
`beta_agg`, causing `lemur_aggregate` to exhaust its `γ=10` retry budget.
This is a fixture artifact (real deployments have N distinct signers, not
replicated copies), so 2 unique signers is fine for the supported N ≤ 8192;
bumping past N=8192 would also require a higher `unique_signers` setting.

A note on aggregated-signature sizes:  the `lemur sizes` numbers in the
serialized-size table above (e.g. 185.5 KB at `N=1024`) are the **predicted**
encoding lengths.  For Rice-coded components — Babai path, sibling labels,
and `u` — the per-coefficient cost is the analytic mean of the Rice
codeword length under a continuous-Gaussian model `X ~ N(0, σ²)` on the
coefficient:

```
E[L] = k + 2 − P(X = 0) + Σ_{j ≥ 1} erfc(j·2^k / (σ·√2))
```

(`k+1` if `X = 0`, else `k + 2 + ⌊|X|/2^k⌋`).  This is computed in
`lemur-py/codec.py:_rice_bits_per_coef`, `lemur-rs/src/codec.rs:rice_bits_per_coef`
(via `libm::erfc`), and `parameter/rice_sizes.py:rice_bits_per_coef`.
Per-polynomial byte counts are byte-aligned with `ceil`.  `lemur sizes`
marks these totals with a leading `~` to flag that they are estimates.
A `bench --fast` run also prints an `Agg Sig Size` line, but that is the
**realised** length of one specific encoded aggregate, which fluctuates
by a few percent around the formula because Rice output length is
data-dependent.  Treat the formula number as the headline figure; the
per-run measurement is informational only.

For large batch-verification timings, use `bench_verify --zero-fixture`.  It
times only `lemur_avrfy` on an accepting all-zero public-key/aggregate fixture,
avoiding the main benchmark's individual preverification and aggregation
measurements.  The `N=2^20` run still materializes the public-key list and may
need several GiB of memory.

## Python/Rust Interoperability Check

Both implementations are designed to produce identical byte-level artifacts for
the same public parameters, seeds, messages, slots, and signer counts.  A quick
cross-check is:

```sh
cd lemur-py
python3 cli.py vectors --tau 3 --signers 2 --slot 0 --msg "artifact check" --out /tmp/lemur-py-vectors.json

cd ../lemur-rs
cargo run --release --bin lemur -- vectors --tau 3 --signers 2 --slot 0 --msg "artifact check" --out /tmp/lemur-rs-vectors.json

python3 - <<'PY'
import json

py = json.load(open("/tmp/lemur-py-vectors.json"))
rs = json.load(open("/tmp/lemur-rs-vectors.json"))

for key in ["pp", "signatures", "ivrfy", "avrfy", "agg_attempt", "aggregate"]:
    print(f"{key}: {'MATCH' if py[key] == rs[key] else 'MISMATCH'}")

for i, (p, r) in enumerate(zip(py["signers"], rs["signers"])):
    print(f"pk {i}: {'MATCH' if p['pk'] == r['pk'] else 'MISMATCH'}")
PY
```

## Parameter Generation

The concrete parameter cells used by the implementations are generated and
checked with the Sage estimator in `parameter/`.

Run from `parameter/`:

```sh
cd parameter
sage lemur_param.sage
```

With `verbosity = 1`, the script writes `summary.txt`.  The estimator README
explains the search space, prime-selection conventions, `beta_encode`, the
Chipmunk comparison scripts, and how to map an estimator cell into the Python
and Rust profile definitions:

```text
parameter/README.md
```

## Artifact Scope

This artifact is scoped to the implementation and parameter-generation
deliverables listed above.  Paper drafts, review notes, local development
metadata, and assistant/tooling notes are not part of the artifact.
