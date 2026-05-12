# api.md — Internal API Reference

Python reference implementation of the Lemur lattice-based multi-signature scheme.

Module dependency order:

```
profiles.py
    ↓
sample.py  →  ring.py  →  kots.py  →  hvc.py  →  lemur.py  →  codec.py  →  cli.py
```

The **`profiles.py`** module at the top of the graph supplies the fixed parameter set (`D256_K4`); every lower module either accepts a `profile=` kwarg with a `DEFAULT = D256_K4` fallback or reads the profile from its context (`HVC` inherits from its `KOTS`, `LEMUR` from its `KOTS`, `LemurCodec` from its `LEMUR`).

---

## Shared conventions

### Array shapes

| Object | Shape | dtype | Notes |
|--------|-------|-------|-------|
| polynomial | `(d,)` | int64 | coefficients in Z (or Z_q, Z_{q'}) |
| polynomial vector | `(n, d)` | int64 | n polynomials |
| polynomial matrix | `(r, c, d)` | int64 | r×c polynomial matrix |
| KOTS secret key S | `(k, m, d)` | int64 | Gaussian, small coefficients |
| KOTS public key T | `(k, n, d)` | int64 | in R_{q'} |
| KOTS signature Z | `(ell, m, d)` | int64 | over Z (not reduced mod q') |
| HVC label | `(omega*kappa, d)` | int64 | balanced digit representation |
| HVC commitment c | `(omega, d)` | int64 | in R_q, values in [0, q-1] |
| HVC opening d | `tuple` | — | `(path_labels, sibling_labels, u)` |

### Ring arithmetic

All polynomial operations are negacyclic: the ring is `R = Z[X]/(X^d + 1)`.
Multiplication by a constant polynomial `X^d` wraps with sign flip (`X^d = -1`).

### HVC opening type

```python
d: tuple[
    list[np.ndarray],   # path_labels:    tau arrays, each (omega*kappa, d)
    list[np.ndarray],   # sibling_labels: tau arrays, each (omega*kappa, d)
    np.ndarray,         # u:              (rho*nu*kappa_prime, d)
]
```

`path_labels[j-1]` is the label of the path node at depth `j`; `sibling_labels[j-1]` is its
sibling. Both are stored as raw digit arrays. `u = dec_{q'}(vec(M_t))` is the decomposed
leaf public key.

---

## `profiles.py` — Parameter set

### `LemurProfile` dataclass

```python
@dataclass(frozen=True)
class LemurProfile:
    name: str
    # Common
    d: int; tau: int; n_signers: int
    # KOTS
    k: int; ell: int; m: int; n: int
    q_prime: int
    alpha: float           # paper's Gaussian width parameter
    sigma: float           # actual sampling stddev (= α/√(2π))
    alpha_h: int
    beta_z: int
    beta_sigma: int
    # HVC
    q: int; omega: int; eta: int
    beta_agg: int
    # Aggregation
    alpha_w: int
    gamma: int = 10
    # Sampler
    cdt_bits: int = 32
    tailcut: int = 5
```

The profile maps to the representative `(d, k, τ, N) = (256, 4, 20, 1024)` cell of `doc/Lemur Parameter Setting.ods`.

### Named parameter set

| Constant | name | `(d, k, τ, N)` | Notes |
|---|---|---|---|
| `D256_K4` | `"d256_k4"` | (256, 4, 20, 1024) | **Default** |

The profile is built via the `_profile(...)` helper, which computes `sigma = alpha / sqrt(2 * pi)` and asserts both `q_prime % 32 == 17` and `(q - 1) % (2 * d) == 0`.

The KOTS modulus `q'` is deliberately not native length-`d` NTT-friendly. `ring.Ring` uses direct schoolbook multiplication for those KOTS products while HVC is required to remain on the native NTT path.

### Helpers

```python
profiles.DEFAULT: LemurProfile                   # = D256_K4
profiles.PROFILES: dict[str, LemurProfile]       # name → profile
profiles.get(name: str) -> LemurProfile          # case-insensitive lookup
```

---

## `sample.py` — XOF-based polynomial samplers

Standalone deterministic samplers.  Each takes an explicit SHAKE XOF handle so
the caller controls domain separation and seed derivation.  All three use
**batched XOF reads**: a single `xof.read(N)` per polynomial rather than per
coefficient.  SHAKE buffers one rate block internally (168 B for SHAKE128,
136 B for SHAKE256), so batching does not change the Keccak-f permutation
count — it eliminates per-call dispatch / allocation overhead.  Because each
caller hands in an *ephemeral* XOF seeded per matrix element (counter-mode
sub-keys; see the setup/keygen XOF construction in `kots.py` and `hvc.py`),
any byte read past what
the sampler needs is discarded with the XOF and byte-for-byte output is
identical to an unbatched read sequence.

### `GaussianSampler` dataclass

```python
@dataclass(frozen=True)
class GaussianSampler:
    sigma: float
    cdt_bits: int = 32
    tailcut: int = 5

    @property
    def cdt_bytes(self) -> int: ...       # cdt_bits // 8; raises if not byte-aligned
```

Profile for CDT-based discrete-Gaussian sampling.  `sigma` is the *sampling
standard deviation* of `D_σ` over `ℤ` and is **deliberately independent of
the paper symbol α**: the Lemur spec's formal Gaussian definition uses α as
a width parameter (σ = α/√(2π)).  Profiles resolve the convention
before passing σ here; the sampler itself is agnostic.

`cdt_bits = 32` is the byte-aligned implementation of the **31-bit CDF
precision bound** from the discrete-Gaussian Rényi-divergence analysis.
The 32-bit read yields 31 bits of CDF comparison plus one sign bit (LSB),
matching `prec_re = 31` at λ=128 for the shipped parameter set.
`tailcut = 5·σ` by the same analysis (`tc_re = 5`).

### Functions

```python
xof_uniform_poly(xof, q: int, d: int) -> np.ndarray   # (d,) in [0, q)
```
Rejection-samples `d` uniform coefficients in `[0, q)` from a SHAKE XOF
stream using `ceil(log2(q))`-bit chunks with rejection (ML-KEM/ML-DSA
matrix-expansion convention).  Reads `2·d·byte_q` bytes up front — enough
for ~2× the expected number of draws at the 23 % Q_KOTS / 3 % Q_HVC
rejection rate — and performs a small `d·byte_q`-byte fallback read if the
initial batch is exhausted.

```python
build_cdt(sigma: float, cdt_bits: int, tailcut: int = 14) -> list[int]
xof_gauss_poly(xof, cdt: list[int], cdt_bits: int, d: int) -> np.ndarray
```

`build_cdt(sigma, cdt_bits, tailcut)` precomputes a CDT table for `|X|`
where `X ~ D_σ`: `cdt[k] = floor(Pr[|X| ≤ k] · 2^cdt_bits)` for `k =
0 .. floor(tailcut·σ) + 1`.  Every KOTS caller passes `tailcut = 5`
via `GaussianSampler`.  Computed in `mpmath` at
`max(2·cdt_bits + 64, 256)` bits so the accumulated CDF is exact at the
table's precision; the last entry is clamped to `2^cdt_bits` to guarantee
binary search termination.  Falls back to `decimal` at proportional
precision when `mpmath` is unavailable.

`xof_gauss_poly` reads `d · (cdt_bits // 8)` bytes in a single batched
`xof.read()`, then for each coefficient extracts the sign from the LSB of
the word (so XOF consumption is exactly `cdt_bits // 8` bytes per
coefficient) and binary-searches `cdt` for the smallest `k` with
`cdt[k] > u`.  Returns `±k` based on the sign bit.

```python
xof_ternary_poly(xof, weight: int, d: int) -> np.ndarray   # (d,) in {-1,0,1}
```
Samples a fixed-weight ternary polynomial with exactly `weight` nonzero
coefficients from `{-1, +1}`.  Reads 9 bytes per batch: 8 position bytes +
1 sign-bit byte (packed as 1 bit per candidate).  Retries on collision or
out-of-range position.  Valid for `d ≤ 256`.

---

## `ring.py` — Negacyclic NTT and Ring class

### Module-level helpers

```python
find_primitive_root(q: int, d: int) -> int
```
Returns the smallest primitive `2d`-th root of unity in `Z_q`.
Requires `(q-1) % (2*d) == 0`.

```python
make_zeta_table(q: int, d: int, zeta: int) -> list[int]
```
Builds the length-`d` NTT twiddle-factor table in bit-reversed order.
`table[k] = zeta^bitrev(k, log2(d)) mod q`.  Matches the ML-DSA twiddle convention.

```python
ntt(f: list[int], zetas: list[int], q: int) -> list[int]
```
Forward NTT: Cooley-Tukey butterfly.  Input in normal order, output in
bit-reversed order, all values in `[0, q)`.

```python
intt(f: list[int], zetas: list[int], q: int) -> list[int]
```
Inverse NTT: Gentleman-Sande butterfly.  Input in bit-reversed order, output
in normal order, all values in `[0, q)`.

```python
poly_mul_schoolbook(a, b, q) -> list[int]
```
Schoolbook negacyclic multiply in `Z_q[X]/(X^d+1)`. Used as the fallback
for non-NTT-friendly moduli.

### `Ring(q, d)` class

NTT-backed arithmetic for `Z_q[X]/(X^d + 1)`. Native NTT is used when
`q ≡ 1 (mod 2d)`. Otherwise, for supported dimensions, multiplication
uses direct schoolbook multiplication.

**Attributes:** `q`, `d`, `zeta` (primitive `2d`-th root on native NTT moduli, otherwise `None`), `zetas` (twiddle table on native NTT moduli, otherwise `None`).

**NTT-domain helpers (internal):**

| Method | Description |
|--------|-------------|
| `_ntt(a)` | Forward NTT; signed inputs accepted (reduced mod q via Python `%`) |
| `_intt(a)` | Inverse NTT; output in `[0, q)` |
| `_lift(a)` | Lift `[0, q)` to canonical signed `[-q//2, q//2]` |

**Arithmetic methods:**

```python
ring.mul(a, b) -> np.ndarray   # (d,) in [0, q)
```
Negacyclic polynomial multiplication mod `q`.

```python
ring.mul_signed(a, b) -> np.ndarray   # (d,) in [-q//2, q//2]
```
Same as `mul`, then lifted to signed range.  Recovers the exact integer
product when the true result fits in `(-q/2, q/2)`.

```python
ring.mat_vec(A, v) -> np.ndarray   # (r, d) in [0, q)
```
Polynomial matrix–vector product mod `q`.  `A: (r, c, d)`, `v: (c, d)`.
Native NTT pre-transforms `v` columns once and accumulates each row in the NTT domain; non-NTT-friendly rings use direct polynomial products.

```python
ring.mat_mul(A, B) -> np.ndarray   # (r, t, d) in [0, q)
```
Polynomial matrix–matrix product mod `q`.  `A: (r, s, d)`, `B: (s, t, d)`.

```python
ring.scale_vec(w, v) -> np.ndarray   # (n, d) signed exact
```
Multiply scalar poly `w` by each row of `v: (n, d)`.  Native NTT pre-transforms `w` once.
Result lifted to signed range.

```python
ring.scale_mat(w, M) -> np.ndarray   # (r, c, d) signed exact
```
Multiply scalar poly `w` by each entry of `M: (r, c, d)`.  Native NTT pre-transforms `w` once.
Result lifted to signed range.

Signed exact operations are valid only when the mathematical coefficient result fits in `(-q/2, q/2)`.

---

## `kots.py` — Key-Homomorphic One-Time Signature

Implements the Lemur KOTS construction.

### `KOTS(profile=DEFAULT, **overrides)` class

`KOTS(profile=DEFAULT)` builds the default-case instance; any field from the
profile can be overridden via the optional keyword arguments listed below.

**Default profile (D256_K4):** `d=256`, `q=3_469_416_721` (`q_prime`), `k=4`,
`ell=1`, `m=9`, `n=4`, `alpha=87.0`, `sigma=34.71...` (= α/√(2π)),
`alpha_h=60`, `beta_z=14_046`, `beta_sigma=13_229_351`.

**Derived attributes:**
- `profile: LemurProfile` — the active profile (for downstream modules to inherit)
- `logq`, `bits_z`, `bits_zagg` — bit widths for codec
- `ring: Ring` — shared ring instance for all polynomial arithmetic
- `sampler: GaussianSampler` — the Gaussian sampler profile actually used for keygen (σ, cdt_bits, tailcut all from the profile unless overridden)
- `cdt: list[int]` — CDT table built from `sampler.sigma` / `sampler.cdt_bits`

**Constructor parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `profile` | `DEFAULT` (= `D256_K4`) | Active `LemurProfile`; supplies defaults for every other kwarg |
| `d` | `profile.d` | Ring dimension |
| `q` | `profile.q_prime` | KOTS modulus q' |
| `k` | `profile.k` | Secret key rows; also HVC message rows ρ |
| `ell` | `profile.ell` | Signature rows; `H = [I_ell | H']` |
| `m` | `profile.m` | Columns of S and rows of A |
| `n` | `profile.n` | Columns of A; also HVC message cols ν |
| `alpha` | `profile.alpha` | Paper symbol α (Gaussian width parameter) |
| `alpha_h` | `profile.alpha_h` | Weight of ternary message-hash polynomial H(μ) |
| `beta_z` | `profile.beta_z` | ‖Z‖_∞ bound for iVrfy |
| `beta_sigma` | `profile.beta_sigma` | ‖Z‖_∞ bound for sVrfy (2× for wVrfy) |
| `sigma` | `profile.sigma` | Sampling σ for secret key S |
| `sampler` | `None` | Optional explicit `GaussianSampler` instance; takes precedence over `sigma` when supplied |
| `lam` | `128` | Security-parameter tag carried on the instance (`self.lam`); not read by the algorithms and kept only so callers can tell which target level a `KOTS` object was built for |

**Static method:**

```python
KOTS._inf_norm(Z) -> int
```
Maximum absolute coefficient across all elements of `Z` (any shape ending in `d`).
Module-level alias: `from kots import inf_norm`.

### KOTS algorithms

```python
kots.setup(seed: bytes) -> np.ndarray   # A2: (m-n, n, d)
```
Expands and stores only the lower block `A2` of the structured public matrix
`A = [I_n; A2] ∈ R_{q'}^{m×n}` from a 32-byte seed.
`A2[i][j] = xof_uniform(SHAKE128(seed || [i,j] || b'A'))`.

```python
kots.keygen(A2, seed: bytes) -> tuple[np.ndarray, np.ndarray]
```
Returns `(S, T)`:
- `S ∈ R^{k×m}`, shape `(k, m, d)`, discrete Gaussian with standard
  deviation `sampler.sigma` (defaults to `profile.sigma`).
  `S[i][j] = xof_gauss(SHAKE256(seed || [i,j] || b'S'))` — one
  counter-mode sub-seed per matrix entry, so the `K·M` samplers are
  independent and trivially parallelisable.
- `T = S·A mod q'`, shape `(k, n, d)`, using the implicit structured matrix
  `A = [I_n; A2]`.

```python
kots.sign(A2, S, mu: bytes) -> np.ndarray   # Z: (ell, m, d)
```
Returns `Z = H·S` computed over Z (signed exact arithmetic via `ring.mul_signed`).
`H = [I_ell | H']` where `H'[i,j] = xof_ternary(SHAKE256(mu || j || b'H'))`.

```python
kots.vrfy(A2, T, mu: bytes, Z, beta: int) -> bool
```
Returns `True` iff `‖Z‖_∞ ≤ beta` and `Z·A ≡ H·T (mod q')`.

```python
kots.ivrfy(A2, T, mu, Z) -> bool   # beta = beta_z
kots.svrfy(A2, T, mu, Z) -> bool   # beta = beta_sigma
kots.wvrfy(A2, T, mu, Z) -> bool   # beta = 2*beta_sigma
```

---

## `hvc.py` — Homomorphic Vector Commitment

Implements the Lemur HVC construction. Commits to a vector of KOTS public keys
via an Ajtai-hash binary Merkle tree with `(2η+1)`-ary digit decomposition.

Uses a streaming O(τ) space algorithm: the tree is never materialized in memory.
Instead, `_subtree_root` processes leaves left-to-right with a stack-based merge.

### `HVC(kots, profile=None, **overrides)` class

Requires a `KOTS` instance to read `d`, `q'` (as `q_prime`), and key shape
(`rho = kots.k`, `nu = kots.n`). If `profile=None` the profile is inherited
from `kots.profile`.

**Defaults from D256_K4 profile (default):** `q=9_007_199_254_746_113`,
`omega=2`, `eta=776`, `tau=20`, `alpha_w=23`, `n_signers=1024`.

**Derived attributes:**
- `profile: LemurProfile`
- `b_val = 2*eta + 1`, `n_slots = 1 << tau`
- `kappa = ceil(log_{b_val}(q))`, `kappa_prime = ceil(log_{b_val}(q'))`
- `beta_agg` — aggregated opening norm bound; taken verbatim from the profile when `tau` and `n_signers` match, otherwise recomputed
- `beta_encode = ceil(beta_agg / (2*eta))` — Babai encoding bound
- `logq`, `bits_dig`, `bits_diag`, `bits_babai` — bit widths for codec
- `ring: Ring` — ring instance at the HVC modulus

**Constructor parameters:**

| Name | Default | Description |
|------|---------|-------------|
| `profile` | `kots.profile` | Active `LemurProfile` |
| `q` | `profile.q` | HVC modulus |
| `omega` | `profile.omega` | Commitment output dimension |
| `eta` | `profile.eta` | Half-base; `b_val = 2η+1` |
| `tau` | `profile.tau` | Tree depth |
| `alpha_w` | `profile.alpha_w` | Randomizer ternary weight |
| `n_signers` | `profile.n_signers` | Number of signers for β_agg derivation |
| `beta_agg` | derived | Override if the caller has already computed β_agg for a non-profile (τ, N) cell |

### Decomposition helpers

```python
hvc._decompose_coeff(c, modulus, kappa) -> list[int]
```
Balanced base-`b_val` decomposition of one integer into `kappa` digits in `[-eta, eta]`.

```python
hvc._dec_poly(a, modulus, kappa) -> np.ndarray   # (kappa, d)
hvc._proj_poly(digits) -> np.ndarray             # (d,)
hvc._dec_vec(v, modulus, kappa) -> np.ndarray    # (n*kappa, d)
hvc._proj_vec(digits, kappa) -> np.ndarray       # (n, d)
```
Polynomial and polynomial-vector decomposition/reconstruction.
`proj(dec(x)) ≡ x (mod modulus)` (over Z, via `sum_k digits[k] * b_val^k`).

### Tree internals

```python
hvc._leaf_label(M_t, B_mat) -> np.ndarray   # (omega*kappa, d)
```
`dec_q(B · dec_{q'}(vec(M_t)))`.

```python
hvc._internal_label(left, right, A0, A1) -> np.ndarray   # (omega*kappa, d)
```
`dec_q((A0 · left + A1 · right) mod q)`.

```python
hvc._subtree_root(pp, leaf_fn, leaf_start, leaf_count) -> np.ndarray
```
Stack-based streaming subtree root computation. Processes leaves
`[leaf_start, leaf_start + leaf_count)` left-to-right with O(log(leaf_count))
labels on the stack. `leaf_count` must be a power of 2.

```python
hvc._proj_label(label) -> np.ndarray   # (omega, d) in [0, q-1]
```
Project a label `(omega*kappa, d)` to `R_q^omega` by reconstructing each
`kappa`-block and reducing mod `q`.

### HVC algorithms

```python
hvc.setup(seed: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```
Returns `(B, A0, A1)`, all uniform in R_q, expanded from a 32-byte seed via SHAKE128.
- `B: (omega, rho*nu*kappa_prime, d)`
- `A0, A1: (omega, omega*kappa, d)`

```python
hvc.com(pp, leaf_fn) -> np.ndarray   # (omega, d)
```
Commits via streaming Merkle tree. `leaf_fn(t)` returns the KOTS public key
for slot `t` as `(rho, nu, d)`. Returns `c = proj_label(root)` in R_q^omega.
Memory: O(τ) intermediate labels.

```python
hvc.open(pp, t: int, leaf_fn) -> tuple
```
Returns opening `d = (path_labels, sibling_labels, u)` for slot `t`.
Computes τ sibling subtree roots via `_subtree_root`, then builds path labels
bottom-up from the leaf. Total work: O(2^τ) leaf evaluations.

```python
hvc.vrfy(pp, c, t: int, d_open, beta: int) -> np.ndarray | None
```
Verifies the opening and returns the recovered KOTS public key `T_t` on success,
`None` on any failure.  Verification steps:
1. Check `‖u‖_∞ ≤ beta`.
2. Compute `hint = B · u ∈ R_q^omega`.
3. For `j = tau, …, 1`: decode check, norm checks, update `hint` up the tree.
4. Check `hint == c`.
5. Return `inv_poly_vec(proj_vec(u, kappa_prime) % q', rho, nu)`.

```python
hvc.ivrfy(pp, c, t, d) -> np.ndarray | None   # beta = eta
hvc.svrfy(pp, c, t, d) -> np.ndarray | None   # beta = beta_agg
hvc.wvrfy(pp, c, t, d) -> np.ndarray | None   # beta = 2*beta_agg
```

### Babai encoding

Babai encoding compresses HVC openings by exploiting the decomposition constraint.
Path labels can be encoded with fewer bits (11 vs 16 per coefficient for aggregated),
or omitted entirely for individual signatures.

```python
hvc.babai_encode_block(digits, hint_block) -> (np.ndarray, np.ndarray)
```
Encode one omega-block of `(kappa, d)` digit polynomials given the known hint
`(d,)` in `[0, q)`. Returns `(a_star, alphas)` where `a_star` is `(d,)` and
`alphas` is `(kappa-1, d)`, both bounded by `beta_encode`.

```python
hvc.babai_decode_block(a_star, alphas, hint_block) -> np.ndarray
```
Decode `(kappa, d)` digit polynomials from `a_star`, `alphas`, and hint.
Derives the last alpha from the ZZ-decomposition carry.

```python
hvc.babai_encode_label(label, hint) -> list[tuple]
hvc.babai_decode_label(encoded, hint) -> np.ndarray
```
Label-level wrappers over omega blocks.

```python
hvc.reconstruct_path_labels_ind(pp, t, sibling_labels, u) -> list
```
Reconstruct individual-signature path labels from sibling labels and `u`.
Path labels are exact decompositions of the hash-chain hints (no Babai data needed).

```python
hvc.reconstruct_path_labels_agg(pp, t, path_encoded, sibling_labels, u) -> list
```
Reconstruct aggregated-signature path labels from Babai-encoded data.
`path_encoded` is a list of tau entries, each a list of omega `(a_star, alphas)` tuples.

### BDS08 traversal (stateful signing)

Stateful signers carry a Buchmann–Dahmen–Schneider (PQCrypto 2008) auth-path
cache so each sign pays amortised `(τ - K) / 2` leaf evaluations instead of
the full `2^τ`.  The cache lives in `sk_state["bds"]` and is advanced in
place (after a deep copy, so the caller's snapshot stays valid) by the
top-level `LEMUR.sign_stateful`.

`K = bds_choose_k(τ)` sets the treehash budget.  For even τ the choice is
`K = 2`; for odd τ it is `K = 3`.  The cache size is dominated by the
dense `auth[τ]` vector plus the `2^K − K − 1` pre-computed right-sibling
labels in the retain FIFOs, giving ~134 KB at τ=20.

#### `TreehashInst` class

```python
class TreehashInst:
    h: int                     # target node height
    stack: list                # tail-node stack of (height, label) tuples
    node: np.ndarray | None    # completed height-h output, or None
    leaf_index: int
    leaves_remaining: int
    finished: bool

    def initialize(self, leaf_index: int) -> None
    def set_ready(self, node: np.ndarray) -> None    # load a pre-computed output
    def active(self) -> bool
    def height_metric(self) -> float                  # BDS scheduler key
    def update(self, hvc, pp, leaf_fn) -> None        # consume one leaf, hash up
    def clone(self) -> TreehashInst                   # independent deep copy
```

One treehash instance per level in `[0, τ-k)`.  Each owns its own tail-node
stack; the paper's shared-stack micro-optimisation is not used because the
total state is still `O(τ^2)` labels.

#### Module-level helpers

```python
bds_copy_state(state: dict) -> dict
```
Deep-copy a BDS state dict (master seed, counters, `auth`, `keep`, `retain`,
`treehash` instances).  Used by `sign_stateful` to avoid mutating the
caller's input.

```python
bds_choose_k(tau: int) -> int
```
Return the BDS `K` parameter for a given τ (2 for even τ, 3 for odd τ;
clamped so `K ≤ τ` at very small τ).

#### `HVC` methods

```python
hvc.bds_init(pp, leaf_fn, K=None) -> tuple[np.ndarray, dict]
```
Fused tree walk that produces both the HVC commitment `c` (root projection)
and the initial BDS state at `phi=0` in a single pass over the leaves.
`leaf_fn(t)` returns the KOTS public key for slot `t`.  The state dict
carries keys `H, K, phi, auth, keep, retain, treehash` — the exact
structure `codec.sk_state_encode` expects.

```python
hvc.bds_advance(state, pp, leaf_fn) -> None
```
Advance the traversal state from slot `phi` to slot `phi+1` in place.  Does
nothing when `phi + 1 == 2^τ`.

```python
hvc.bds_opening(state, pp, t, leaf_fn) -> tuple
```
Assemble an opening for slot `t` from the current BDS cache.  Raises
`ValueError` (not a Python assert) if `t != state["phi"]` or if the
internal cache and `phi` disagree, so the error survives `python -O`.

---

## `lemur.py` — Full Multi-Signature Scheme

Implements the Lemur multi-signature construction. Composes `KOTS` and `HVC`.

### `LEMUR(kots, hvc, profile=None, alpha_w=None, gamma=None)` class

If `profile=None`, it is inherited from `kots.profile`. `alpha_w` and `gamma`
default to the profile's values (`23, 10`).

**Attributes:** `profile`, `kots`, `hvc`, `alpha_w`, `gamma`, `n_slots = hvc.n_slots`.

### Internal helpers

```python
LEMUR._slot_seed(master_seed, t) -> bytes   # @staticmethod
```
Per-slot seed: `SHAKE256(master_seed || b'slot' || t.to_bytes(4, 'little')).read(32)`.

```python
LEMUR.make_stateful_sk(master_seed, bds) -> dict   # @staticmethod
```
Build a stateful signer key from a 32-byte master seed and a populated
BDS08 state.  Validates input shapes and returns `{"master_seed":
bytes(master_seed), "bds": bds}`.  The "current slot" is read from
`bds["phi"]` — there is no separate `next_slot` field.

```python
lemur._hash_to_randomizers(t, m, P, attempt) -> list[np.ndarray]
```
Streams N ternary randomizers `w^i ∈ T_{alpha_w}` from a single
`SHAKE256(t || len(m) || m || pk_bytes || attempt)` XOF.

```python
LEMUR._add_openings(d1, d2) -> tuple   # @staticmethod
```
Componentwise integer addition of two HVC openings over Z.

```python
lemur._scale_opening(w, opening) -> tuple
```
Scale an HVC opening `(path_labels, sibling_labels, u)` by ternary poly `w`
using `hvc.ring.scale_vec`.

```python
lemur._weighted_sum_commitments(ws, pks) -> np.ndarray   # (omega, d)
```
`c_agg = sum_i w^i * c_i  mod q`.

### Key and signature types

```python
sk_seed: bytes     # 32-byte master seed (immutable, never written back)

sk_state: dict     # {
                   #   "master_seed": bytes,  # 32 bytes
                   #   "bds":         dict,   # BDS08 traversal state (always populated)
                   # }

pk: np.ndarray     # HVC commitment c, shape (omega, d)

sigma: tuple[np.ndarray, tuple]          # (Z, d_open)  — individual
sigma_agg: tuple[np.ndarray, tuple, int] # (Z_agg, d_agg, attempt)  — aggregated
```

Per-slot KOTS keys are re-derived on demand from the master seed via `_slot_seed`.
Neither secret-key form stores OPK or OSK arrays.  The "current slot" of a
stateful sk is the single counter `sk_state["bds"]["phi"]`; there is no
separate `next_slot` field.

The stateful sk always carries an in-memory BDS08 traversal cache that makes
`sign_stateful` amortised O(τ) per call instead of O(2^τ).  The cache is
populated eagerly by `keygen` and carried between calls both in-process and
on disk.

On disk, `codec.sk_state_encode` writes the cache with no magic bytes:

```
master_seed(32) || phi_u32 || tau_u32 || k_u32 ||
auth[tau] || keep[tau] || retain[tau] || treehash[tau-k]
```

`auth` is a dense sequence of `tau` labels; `keep` is per-level
`u8 present [+ label]`; `retain` is per-level `u16 count, count × label` in
pop-front order; `treehash` is one record per level in `[0, tau-k)` carrying
`finished` flag, `leaf_index`, `leaves_remaining`, optional completed `node`,
and the tail-node stack.  Labels are bit-packed at
`dx_dig = ceil(log2(2*eta+1))` bits per coefficient with offset `+eta`,
reusing the sibling-label encoding from individual signatures
(`dx_dig = 11` under D256_K4 where η = 776).  A fresh keygen state at
τ=3 is about 16 KB; at τ=20 it is ~134 KB.

`sign_stateful` does not mutate its input `sk_state`: it deep-copies the BDS
state before advancing, so callers holding a pre-sign snapshot can re-use
it for rollback or reproduction.  Use the returned new state for the next
call in the normal flow.

### LEMUR algorithms

```python
lemur.setup(kots_seed: bytes, hvc_seed: bytes) -> tuple
```
Returns `pp = (kots_pp, hvc_pp)`.

```python
lemur.keygen(pp, seed: bytes) -> tuple[bytes, dict, np.ndarray]
```
Returns `(sk_seed, sk_state, pk)` where `sk_seed = seed` (32 bytes),
`sk_state = {"master_seed": seed, "bds": <init state at phi=0>}`, and
`pk = c` (HVC commitment). A single fused tree walk via `hvc.bds_init`
produces both the commitment and the initial BDS08 traversal state.
O(2^τ) leaf computations (no extra cost beyond the plain commitment).

```python
lemur.sign_seed(pp, sk_seed: bytes, t: int, m: bytes) -> tuple
```
Returns `(Z, d_open)`. Re-derives the per-slot KOTS key from `sk_seed` for
online signing, then calls `hvc.open` with a `leaf_fn` closure for the
offline HVC opening. O(2^τ) leaf computations for the opening.  The caller
passes `t` explicitly; `sign_seed` never consults or updates a persistent
counter.

```python
lemur.sign_stateful(pp, sk_state: dict, m: bytes, t: int | None = None) -> tuple
```
Returns `(sigma, next_sk_state, slot_used)`. If `t` is omitted, uses
`sk_state["bds"]["phi"]`.  The BDS08 traversal state is deep-copied
before advancing, so `sk_state` is left untouched — use `next_sk_state`
for the next call. Amortised cost per call: `(τ - K) / 2` leaf
evaluations plus one KOTS sign (K=2 for even τ, K=3 for odd τ).  Raises
`ValueError` if the slot is out of range, mismatched against the
cache's `phi`, or if the BDS state's internal position disagrees with
the requested slot.

```python
lemur.ivrfy(pp, pk, t: int, m: bytes, sigma) -> bool
```
Individual verification: HVC `ivrfy` (β=η) to recover `opk`, then KOTS `ivrfy`
(β=β_z) — tight individual bounds from the active profile.  Catches all exceptions
internally and returns `False` on any error (malformed signature, decode failure,
truncated input).

```python
lemur.aggregate(pp, pks, t, m, sigs) -> tuple | None
```
Rejects if any individual signature is invalid.  Retries up to `gamma` times
with different randomizer draws until `avrfy` passes.  Returns `None` on failure.

```python
lemur.avrfy(pp, pks, t: int, m: bytes, sigma_agg) -> bool
```
Aggregated verification:
1. Recompute `ws` from `(t, m, pks, attempt)`.
2. `c_agg = sum_i w^i * c_i`.
3. `opk_agg = hvc.svrfy(hvc_pp, c_agg, t, d_agg)`.
4. `kots.svrfy(kots_pp, opk_agg, m, Z_agg)`.

Catches all exceptions internally and returns `False` on any error.

---

## `codec.py` — Compact serialization

### Module-level bit-packing primitives

```python
poly_serial(poly, dx: int, offset: int = 0) -> bytes
poly_deserial(data: bytes, dx: int, d: int, offset: int = 0) -> tuple[list, int]
vec_serial(v, dx: int, offset: int = 0) -> bytes
vec_deserial(data: bytes, dx: int, n: int, d: int, offset: int = 0) -> tuple[list, int]
```
mmCipher-style bit packing.  Signed values use offset encoding:
store `val + offset` as unsigned `dx`-bit integer.
`poly_deserial` / `vec_deserial` return `(coefficients, bytes_consumed)`.

### Rice coding primitives

```python
poly_serial_rice(poly, rice_k: int, bound: int) -> bytes
poly_deserial_rice(data: bytes, d: int, rice_k: int, bound: int) -> tuple[list, int]
vec_serial_rice(v, rice_k: int, bound: int) -> bytes
vec_deserial_rice(data: bytes, n: int, d: int, rice_k: int, bound: int) -> tuple[list, int]
```
Golomb-Rice encoding for polynomials with coefficients in `[-bound, bound]`.
Low bits (rice_k) stored verbatim, high bits as unary + stop bit, followed by
sign bit.  Each polynomial is byte-aligned.  The `vec_*` wrappers iterate the
poly primitives.  Deserialization rejects out-of-range coefficients and caps
unary run length; `poly_deserial_rice` also rejects nonzero padding after the
final byte.

### pp header inspection helpers

```python
PP_BYTES                                                       # 65
pp_peek_tau(data: bytes) -> int                                # extract τ
make_codec(tau: int, profile: LemurProfile | None = None)      # fresh codec
```

`pp_peek_tau` reads the trailing byte of a pp blob and returns τ
without instantiating any scheme objects.  `make_codec(tau, profile=DEFAULT)`
builds a fresh `LemurCodec` whose underlying `KOTS`/`HVC` match the
requested τ and profile, so the CLI can rebuild a correctly configured
codec before calling `codec.pp_decode`.

### `compute_agg_encoding(kots, hvc, alpha_w, n_signers)` function

Determines encoding parameters for aggregated signatures from public info.
Returns a dict with bit widths, Rice parameters, and whether Rice coding is
beneficial for each component (Z_agg, Babai path labels, sibling labels, u).
Parameters depend on N (number of signers) and the scheme constants.

For each Rice-eligible component, the helper picks between fixed-width
`dx_fixed = bit_length(2*max_bound)` and Rice coding by directly minimising
expected bits per coefficient: search `k ∈ [0, dx_fixed]` for the minimum
of `k + μ/2^k + 2` (folded-Gaussian approximation, `μ = 0.7979·σ`) and
compare against `dx_fixed`.  There is no hysteresis margin and no low-σ
cutoff — the formula decides.  `Z_agg` always uses N-dependent fixed-width
because its bound shrinks with N.

The CLT estimate for the Z-component σ uses `kots.sampler.sigma` (the actual
sampling stddev) rather than the paper symbol `kots.alpha` — the two differ
by √(2π) and conflating them would mis-size the Rice vs fixed-width choice
by ~2.5×.

### `LemurCodec(scheme)` class

All layout constants derived from the injected `LEMUR` instance.

**Bit width attributes:**

| Attribute | Description |
|-----------|-------------|
| `dx_pk` | `hvc.logq` — pk unsigned |
| `dx_z` | `kots.bits_z` — Z signed with offset `beta_z` |
| `dx_dig` | `hvc.bits_dig` — individual labels/u with offset `eta` |
| `dx_babai` | `hvc.bits_babai` — Babai-encoded coefficients with offset `beta_encode` |

**Key/parameter management:**

```python
codec.setup(kots_seed=None, hvc_seed=None) -> tuple[pp, bytes, bytes]
codec.keygen(pp, master_seed=None) -> tuple[bytes, dict, np.ndarray]
codec.sign_seed(pp, sk_seed: bytes, t, m) -> tuple
codec.sign_stateful(pp, sk_state: dict, m, t=None) -> tuple
```
`keygen` returns a raw seed key and a stateful signer key.  `sign_seed`
re-derives the KOTS key and runs a fresh `hvc.open` on each call (O(2^τ)
leaves); it takes the slot as an explicit argument and never writes back
to the raw seed.  `sign_stateful` re-derives the KOTS key but pulls the
HVC auth path from the in-memory BDS08 traversal cache carried inside
`sk_state["bds"]`, advancing it on each call (amortised O(τ-K) leaves
per sign).  The on-disk form written by `codec.sk_state_encode` persists
the full cache, so reloading it in a fresh process keeps the amortised
cost across restarts.

**Encode/decode:**

```python
codec.pp_encode(kots_seed, hvc_seed) -> bytes        # 65 B (seeds + tau)
codec.pp_decode(data) -> tuple[pp, bytes, bytes]     # rejects tau mismatch
codec.sk_encode(sk: bytes) -> bytes                  # 32 B
codec.sk_decode(data: bytes) -> bytes                # 32-byte seed
codec.sk_state_encode(sk_state: dict) -> bytes       # BDS state (~16 KB at τ=3, ~134 KB at τ=20 under D256_K4)
codec.sk_state_decode(data: bytes) -> dict
codec.pk_encode(pk) -> bytes
codec.pk_decode(data) -> np.ndarray
codec.sig_encode(sigma) -> bytes
codec.sig_decode(data, pp, t) -> tuple
codec.sig_bytes() -> int                             # individual-sig byte count
codec.agg_sig_encode(sigma_agg, n_signers) -> bytes
codec.agg_sig_decode(data, pp, t, n_signers) -> tuple
codec.sizes(n_signers: int = 1024) -> dict[str, int]
```

pp does not carry the profile — both sides must agree out-of-band on which profile to construct the codec under.  `pp_decode` validates that the τ byte matches the codec's scheme and rejects the file otherwise.

`sig_decode` requires `pp` (public parameters) and `t` (slot index) to
reconstruct path labels from sibling labels and `u`.

`agg_sig_encode(sigma_agg, n_signers)` and
`agg_sig_decode(data, pp, t, n_signers)` both require `n_signers` to
determine N-dependent encoding parameters via `compute_agg_encoding()`;
`agg_sig_decode` additionally takes `pp` and the slot `t` so it can
reconstruct path labels just like `sig_decode`.

**Canonical deserialization checks:** out-of-range coefficients rejected,
nonzero padding bits rejected, trailing bytes rejected, truncated input
detected, Rice unary run length capped.

**File format:**

No magic bytes — the format is unambiguous.

| File | Format |
|------|--------|
| pp | `kots_seed (32B) \|\| hvc_seed (32B) \|\| tau_u8 (1B)` — **65 B total** |
| sk | `master_seed (32B)` |
| sk.state | `master_seed(32) \|\| phi_u32 \|\| tau_u32 \|\| k_u32 \|\| auth \|\| keep \|\| retain \|\| treehash` |
| pk | `omega polys bit-packed at logq bits (unsigned)` |
| *.sig | `Z \|\| sibling_labels \|\| u` (path labels reconstructed by verifier) |
| agg.sig | `attempt (1B) \|\| Z_agg \|\| path_babai \|\| sibling \|\| u` (variable size, N-dependent encoding) |

`pp_decode` rejects any file whose τ does not match the codec's scheme; module-level `pp_peek_tau(data)` and `make_codec(tau, profile)` in `codec.py` let callers (including the CLI) rebuild a correctly configured scheme from the file before decoding.

---

## Dependency graph

```
profiles.py         (LemurProfile, D256_K4 [default], DEFAULT)
  │
  ├── kots.py       (KOTS, inf_norm)
  │     ├── profiles.py
  │     ├── ring.py    (Ring)
  │     └── sample.py  (xof_uniform_poly, xof_gauss_poly, xof_ternary_poly)
  ├── hvc.py        (HVC)
  │     ├── profiles.py
  │     ├── kots.py    (KOTS, inf_norm)
  │     ├── ring.py    (Ring)
  │     └── sample.py  (xof_uniform_poly)
  └── lemur.py      (LEMUR)
        ├── profiles.py
        ├── kots.py
        └── hvc.py
codec.py
  └── profiles.py, lemur.py, kots.py, hvc.py
cli.py
  └── profiles.py, codec.py, lemur.py, kots.py, hvc.py
```
