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

This reproduces the implemented aggregated-signature-size cell:

| N | Table entry |
| ---: | ---: |
| `2^10` | 201.2 KB |

The larger signature-size cells are predicted Rice-encoded sizes for the
corresponding `tau=20` rows in `parameter/summary.txt`.  Reproduce them with:

```sh
cd parameter
python3 rice_sizes.py
```

| N | Worst-case row in `parameter/summary.txt` | Table entry |
| ---: | ---: | ---: |
| `2^15` | 331 KB | 283.5 KB |
| `2^20` | 457 KB | 394.4 KB |

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

For the 24-thread run used in the paper, `Aggregation(8192) = 5.73 s`, giving
approximately `23 s` and `12 min`.

Representative serialized sizes for `tau=20, N=1024`:

| Object | Size |
| --- | ---: |
| Public parameters | 65 B |
| Secret seed | 32 B |
| Stateful signer cache | 134.4 KB |
| Public key | 3.4 KB |
| Individual signature | 89.5 KB |
| Aggregated signature, `N=1024` | 201.2 KB |

Representative `bench --fast` timings from a 24-thread run:

| Operation | Time |
| --- | ---: |
| Key generation | 1.3 min |
| Online signing, KOTS only | 347 us |
| Full signing, including HVC open | 1.3 min |
| Stateful signing, BDS08 | 4.13 ms |
| Individual pre-verify, `N=1024` | 1.67 s |
| Aggregate after verified inputs, `N=1024` | 2.40 s |
| Secure aggregation, `N=1024` | 567 ms |
| Batch verification, `N=1024` | 30.1 ms |
| Individual pre-verify, `N=8192` | 13.6 s |
| Aggregate after verified inputs, `N=8192` | 37.6 s |
| Secure aggregation, `N=8192` | 5.73 s |
| Batch verification, `N=8192` | 223 ms |

Timings are machine-dependent.

A note on aggregated-signature sizes:  the `lemur sizes` numbers in the
serialized-size table above (e.g. 201.2 KB at `N=1024`) are the **predicted**
encoding lengths.  For Rice-coded components — Babai path, sibling labels,
and `u` — the per-coefficient cost is estimated from the folded-Gaussian
mean unary tail (`0.7979·σ / 2^k`) plus a small conservative pad for the
sign bit and unary terminator.  `lemur sizes` marks these totals with a
leading `~` to flag that they are estimates.  A `bench --fast` run also
prints an `Agg Sig Size` line, but that is the **realised** length of one
specific encoded aggregate, which fluctuates by a few percent around the
formula because Rice output length is data-dependent.  Treat the formula
number as the headline figure; the per-run measurement is informational
only.

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
