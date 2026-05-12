# Lemur Rust Implementation

This directory contains the performance-oriented Rust implementation of
Lemur.  It provides the same scheme pipeline and serialized formats as the
Python reference, plus integration tests and benchmarking binaries.

The default profile is the representative artifact cell
`d=256, k=4, tau=20, N=1024`.  Larger signer counts reported in the paper are
derived from the parameter estimator and benchmark extrapolations described in
the repository-level `README.md`.

## Prerequisites

Install a stable Rust toolchain with Cargo.  The checked-in `Cargo.lock` pins
the dependency resolution used for the artifact.

## Tests

Run the full Rust test suite from this directory:

```sh
cargo test
```

The tests cover profile invariants, KOTS/HVC consistency, stateful BDS08
signing, serialization, robustness against malformed encodings, materialized
tree signing, and the default end-to-end pipeline.

For a fast profile-only check:

```sh
cargo test profile::tests --lib
```

## CLI

Build and run the Lemur CLI with Cargo:

```sh
cargo run --release --bin lemur -- sizes
cargo run --release --bin lemur -- vectors --tau 3 --signers 2 --slot 0 \
    --msg "artifact check" --out /tmp/lemur-rs-vectors.json
```

The CLI commands mirror the Python implementation:

```text
setup, keygen, sign, verify, aggregate, batch-verify, sizes, vectors
```

## Benchmarks

The main artifact benchmark is:

```sh
cargo run --release --bin bench -- --fast
```

The optional tree-in-memory signing benchmark materializes the HVC tree and
requires about 8 GiB of RAM at `tau=20`:

```sh
cargo run --release --bin bench -- --fast --with-tree
```

Large-`N` batch verification can be measured without first running large
aggregation by using the zero-fixture benchmark:

```sh
cargo run --release --bin bench_verify -- --zero-fixture --n 32768 --reps 3
cargo run --release --bin bench_verify -- --zero-fixture --n 1048576 --reps 1
```

The repository-level `README.md` explains how these outputs map to the paper's
implementation-performance table.
