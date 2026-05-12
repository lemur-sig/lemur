# Lemur Parameter Estimator

This directory contains the Sage driver used to reproduce the Lemur
implementation parameters.  The artifact uses one fixed parameter family:

```text
d = 256
k = 4
ell = 1
secpar = 128
epsilon = 2^-15
RHF = 1.0045
```

The estimator evaluates that family for:

```text
tau in {12, 16, 20, 24}
N   in {1024, 32768, 131072, 1048576}
```

The Python and Rust implementations currently instantiate the representative
cell `tau = 20, N = 1024`, while the code paths also support alternate tree
depths such as `tau = 24`.

## Inputs

The active Lemur driver is `lemur_param.sage`.  It combines:

- the local lattice-estimator fork in `parameter/estimator/`
- the MSIS estimator in `parameter/msis_estimator/`
- the fixed implementation family `(d, k) = (256, 4)`

The generated constants are copied into:

- `lemur-py/profiles.py`
- `lemur-rs/src/profile.rs`
- `lemur-rs/gen_tables.py`

## Environment

Install SageMath and make sure `sage` is on your `PATH`:

```sh
sage -c 'print("sage ok")'
```

## Generate the Parameter Table

Run the estimator from this directory:

```sh
cd parameter
sage lemur_param.sage
```

With `verbosity = 1`, the script writes `summary.txt` in the current working
directory.  The rows correspond to the fixed `(d,k)=(256,4)` family across the
supported `(tau,N)` cells.

## Chipmunk Comparison Scripts

Two companion scripts reproduce the Chipmunk comparison data used by the paper:

- `chipmunk_param.sage` starts from the open-source Chipmunk parameter script,
  fixes the aggregated-opening bound by including the missing signer-count
  factor, uses the MSIS estimator, records opening sizes, and evaluates
  dimension `d=1024`.
- `chipmunk_original.sage` keeps the original Chipmunk parameter generation path
  and estimates the concrete Ring-SIS security of those parameters.

Run them from this directory:

```sh
sage chipmunk_param.sage
sage chipmunk_original.sage
```

`chipmunk_param.sage` writes `chipmunk_summary.txt`.  It uses the paper's
decoupled Chipmunk RHF settings:

| `rho` | `RHF_KOTS` | `RHF_HVC` |
| ---: | ---: | ---: |
| 1024 | 1.0045 | 1.0045 |
| 32768 | 1.00464 | 1.0045 |
| 131072 | 1.0048 | 1.0045 |
| 1048576 | 1.00501 | 1.0045 |

`chipmunk_original.sage` writes `chipmunk_original_security_summary.txt`; it is
only for assessing the original Chipmunk parameter security and is not used to
generate Lemur implementation constants.

## Rice-Encoded Size Entries

The paper reports worst-case sizes from `summary.txt` and also reports
Rice-encoded aggregate-size estimates for selected `tau=20` rows.  Reproduce
those encoded-size entries with:

```sh
python3 rice_sizes.py
```

The script reads `summary.txt` and prints the shipped `N=1024` cell plus the
larger `N=32768` and `N=1048576` rows used in the paper's implementation
performance table.

## Prime Conventions

KOTS uses CRT multiplication.  The estimator chooses `q'` with:

```text
q' == 17 mod 32
```

HVC only needs `q == 1 mod 2d` for the negacyclic NTT, but the spreadsheet uses
the stronger search convention:

```text
q == 1 mod 4d
```

`Profile::validate()` enforces the minimum runtime invariant `q == 1 mod 2d`.
The stronger `q == 1 mod 4d` convention is a search-time choice used to match
the spreadsheet cells.

## Representative Implementation Cell

The shipped Python and Rust constants use:

```text
tau = 20
N = 1024
```

For that cell, the estimator produces:

| field | value |
| --- | ---: |
| `d` | 256 |
| `k` | 4 |
| `ell` | 1 |
| `n` | 4 |
| `m` | 9 |
| `omega` | 2 |
| `alpha` | 87 |
| `alpha_mlwe` | 1.6 |
| `alpha_w` | 23 |
| `alpha_H` | 60 |
| `eta` | 776 |
| `kappa` | 5 |
| `kappaprime` | 3 |
| `qprime` | 3469416721 |
| `q` | 9007199254746113 |
| `beta_z` | 14046 |
| `beta_sigma` | 13229351 |
| `beta_agg` | 919945 |
| `beta_encode` | 593 |

`beta_encode` is:

```text
ceil(beta_agg / (2 * eta))
```

Do not use the older `ceil(beta_agg / (2*eta) + 1/2)` expression.

## Updating Python

Copy the representative cell into `lemur-py/profiles.py`.

Required fields:

- `name`
- `d`, `tau`, `n_signers`
- `k`, `ell`, `m`, `n`
- `q_prime`
- `alpha`, `alpha_h`, `beta_z`, `beta_sigma`
- `q`, `omega`, `eta`, `beta_agg`
- `alpha_w`, `gamma`

The Python helper computes:

```python
sigma = alpha / sqrt(2*pi)
```

Import-time assertions check:

- `q_prime % 32 == 17`
- `(q - 1) % (2*d) == 0`

## Updating Rust

Copy the same cell into `lemur-rs/src/profile.rs`.

Additional Rust-only fields:

- `kappa`
- `kappa_prime`
- `beta_encode`
- `HvcRing::U64(...)`
- `kots_crt: Some(KotsCrtCfg { q: q_prime, d })`

`Profile::validate()` checks:

- `beta_encode == ceil(beta_agg / (2*eta))`
- HVC NTT compatibility via `q == 1 mod 2d`
- KOTS `q' == 17 mod 32`
- CRT backend capacity for the KOTS accumulation shapes

The estimator and both implementations call the aggregation-attempt parameter
`gamma`, matching the paper.

## Regenerating Rust Tables

After editing `lemur-rs/gen_tables.py`, regenerate the Rust table module:

```sh
cd lemur-rs
python3 gen_tables.py > src/tables_d256_k4.rs
```

Then ensure `lemur-rs/src/profile.rs` points `D256_K4` at the generated table
constants and `lemur-rs/src/lib.rs` declares `tables_d256_k4`.

## Verification

Run the focused checks after updating constants:

```sh
python3 -m py_compile lemur-py/profiles.py lemur-py/codec.py lemur-py/kots.py lemur-py/cli.py
cd lemur-rs
cargo check
cargo test profile::tests --lib
```

For a full Rust verification pass:

```sh
cd lemur-rs
cargo test
```
