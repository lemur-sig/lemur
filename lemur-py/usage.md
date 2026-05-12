# Lemur CLI Usage

All commands read and write binary files. Run from the `lemur-py/` directory.

The implementation uses the fixed parameter set `d=256, k=4` with default
tree depth `tau=20`.  Public parameters encode only the two setup seeds and
the tree depth.

## 1. Generate Public Parameters

```bash
python cli.py setup --out pp.bin
```

Outputs `pp.bin` (**65 bytes**): `kots_seed(32) || hvc_seed(32) || tau_u8`.
Distribute this file to all signers.

To use deterministic seeds:

```bash
python cli.py setup --out pp.bin \
    --kots-seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f \
    --hvc-seed  202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f
```

To use a different tree depth:

```bash
python cli.py setup --out pp.bin --tau 24
```

For fast local checks, use `--tau 3`.

## 2. Key Generation

Each signer runs key generation independently:

```bash
python cli.py keygen --pp pp.bin --sk alice.sk --stateful-sk alice.state --pk alice.pk
python cli.py keygen --pp pp.bin --sk bob.sk   --stateful-sk bob.state   --pk bob.pk
```

- `*.sk` is the raw 32-byte seed secret key.
- `*.state` is the mutable BDS08 traversal state for stateful signing.
- `*.pk` is the public key shared with the aggregator.

At the default `tau=20`, key generation computes `2^20` KOTS keypairs and can
take several minutes.

## 3. Sign

Raw seed workflow:

```bash
python cli.py sign --pp pp.bin --sk alice.sk --slot 2 --msg "approve" --out alice.sig
```

Stateful workflow:

```bash
python cli.py sign --pp pp.bin --stateful-sk alice.state --msg "approve" --out alice.sig
```

When `--stateful-sk` is used, the state file is advanced and rewritten in
place.  An explicit `--slot` is optional; if supplied, it must match the
state's current slot.

## 4. Verify

```bash
python cli.py verify --pp pp.bin --pk alice.pk --slot 2 --msg "approve" --sig alice.sig
```

The command prints `OK` and exits 0 on success, otherwise `FAIL` and exits 1.

## 5. Aggregate

```bash
python cli.py aggregate \
    --pp   pp.bin \
    --slot 2 \
    --msg  "approve" \
    --pks  alice.pk bob.pk \
    --sigs alice.sig bob.sig \
    --out  agg.sig
```

The aggregator verifies every individual signature before combining.  It
retries up to `gamma=10` times with different randomizers.

## 6. Batch Verify

```bash
python cli.py batch-verify \
    --pp   pp.bin \
    --slot 2 \
    --msg  "approve" \
    --pks  alice.pk bob.pk \
    --sig  agg.sig
```

The public keys must be in the same order used during aggregation.

## 7. Sizes

```bash
python cli.py sizes
```

Shows byte counts for public parameters, keys, individual signatures, and
aggregated signatures for several signer counts.  The output uses the fixed
parameter set at default `tau=20`.

## 8. Deterministic Vectors

```bash
python cli.py vectors --out vectors.json
```

Options:

```bash
python cli.py vectors --signers 3 --slot 1 --msg "committee vote" --out vectors.json
python cli.py vectors --tau 3 --out vectors.json
python cli.py vectors --tau 24 --out vectors.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--signers N` | 2 | Number of signers |
| `--slot T` | 0 | Time slot index |
| `--msg TEXT` | `"test vector"` | Message string |
| `--out FILE` | stdout | Output file path |
| `--tau TAU` | 20 | Tree depth |

When `--tau` is specified, `beta_agg` and `beta_encode` are recomputed for
that tree depth.

## Representative Sizes

| File | Size at `tau=20, N=1024` |
|------|--------------------------:|
| `pp.bin` | 65 B |
| `*.pk` | ~3.4 KB |
| `*.sk` | 32 B |
| `*.state` | ~134 KB |
| `*.sig` | ~89.5 KB |
| `agg.sig` | ~201 KB |

`sig_decode` requires `pp` and the slot index so path labels can be
reconstructed.  `agg_sig_decode` also requires the signer count, because the
aggregated-signature encoding is N-dependent.
