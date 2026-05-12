# Lemur Python Reference

This directory contains the readable Python reference implementation of Lemur.
It covers the full scheme pipeline: KOTS, HVC, key generation, stateful
signing, individual verification, aggregation, batch verification, and
byte-level serialization.

The Python implementation is intended as the golden reference for checking
functional behavior and generating deterministic test vectors.  It is
byte-compatible with the Rust implementation for the same public parameters,
seeds, slots, messages, and signer counts.

## Prerequisites

Use Python 3 with the packages listed in `requirements.txt`:

```sh
python3 -m pip install -r requirements.txt
```

## Quick Checks

Run the module self-checks from this directory:

```sh
python3 ring.py
python3 kots.py
python3 hvc.py
python3 lemur.py
```

Generate a small deterministic vector set:

```sh
python3 cli.py vectors --tau 3 --signers 2 --slot 0 \
    --msg "artifact check" --out /tmp/lemur-py-vectors.json
```

Print representative serialized sizes for the shipped parameter profile:

```sh
python3 cli.py sizes
```

## Command-Line Interface

The CLI supports:

```text
setup, keygen, sign, verify, aggregate, batch-verify, sizes, vectors
```

See `usage.md` for command examples and file formats.  See `api.md` for a
more detailed implementation/API reference.

## Parameter Profile

The shipped implementation uses the representative parameter cell
`d=256, k=4, tau=20, N=1024`.  Alternate tree depths can be used for local
testing and vector generation with `--tau`, for example `--tau 3`.
