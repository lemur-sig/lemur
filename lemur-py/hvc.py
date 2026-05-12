"""
HVC: Homomorphic Vector Commitment.

Commits to a vector of KOTS public keys M = (M_0,...,M_{2^tau - 1}),
each M_t in R_{q'}^{rho x nu}.  Node labels are short integer vectors
in R_q^{omega*kappa} verified via a (2eta+1)-ary lattice hash chain.

Stateful signers maintain an auth path via the BDS08 tree-traversal
algorithm (Buchmann-Dahmen-Schneider, PQCrypto 2008); see the helpers
`bds_init`, `bds_advance`, and `bds_opening` at the bottom of this file.
"""

import math
import numpy as np
from Crypto.Hash import SHAKE128

from kots import KOTS, inf_norm
from profiles import LemurProfile
from ring import Ring
from sample import xof_uniform_poly


class TreehashInst:
    """One BDS08 treehash instance for computing a single height-h node.

    Each instance owns its own tail-node stack.  Sharing a single stack
    across instances is a memory optimisation we do not use here; total
    state is still O(tau^2).
    """

    __slots__ = ("h", "stack", "node", "leaf_index", "leaves_remaining",
                 "finished")

    def __init__(self, h: int):
        self.h                = h
        self.stack: list      = []      # list of (height, label) tuples
        self.node             = None    # finished height-h output (or None)
        self.leaf_index       = 0       # only read when leaves_remaining > 0
        self.leaves_remaining = 0
        self.finished         = False

    def initialize(self, leaf_index: int) -> None:
        self.stack            = []
        self.node             = None
        self.leaf_index       = leaf_index
        self.leaves_remaining = 1 << self.h
        self.finished         = False

    def set_ready(self, node: np.ndarray) -> None:
        """Mark this instance as already holding a completed node."""
        self.stack            = []
        self.node             = node
        self.leaf_index       = 0
        self.leaves_remaining = 0
        self.finished         = True

    def active(self) -> bool:
        return (not self.finished) and self.leaves_remaining > 0

    def height_metric(self) -> float:
        """Lowest stored tail-node height, or `h` if stack empty, or inf if done."""
        if self.finished or self.leaves_remaining == 0:
            return math.inf
        if not self.stack:
            return float(self.h)
        return float(min(e[0] for e in self.stack))

    def update(self, hvc: "HVC", pp: tuple, leaf_fn) -> None:
        """Execute one treehash step: consume one leaf, hash up while possible."""
        if self.finished or self.leaves_remaining == 0:
            return
        B_mat, A0, A1 = pp
        leaf = hvc._leaf_label(leaf_fn(self.leaf_index), B_mat)
        cur_h = 0
        cur_lab = leaf
        while self.stack and self.stack[-1][0] == cur_h:
            top_h, top_lab = self.stack.pop()
            cur_lab = hvc._internal_label(top_lab, cur_lab, A0, A1)
            cur_h += 1
        self.stack.append((cur_h, cur_lab))
        self.leaf_index       += 1
        self.leaves_remaining -= 1
        if self.leaves_remaining == 0:
            assert len(self.stack) == 1 and self.stack[0][0] == self.h
            self.node     = self.stack[0][1]
            self.stack    = []
            self.finished = True

    def clone(self) -> "TreehashInst":
        """Return an independent deep copy of this instance."""
        new = TreehashInst(self.h)
        new.stack = [(h, lab.copy()) for (h, lab) in self.stack]
        new.node = None if self.node is None else self.node.copy()
        new.leaf_index       = self.leaf_index
        new.leaves_remaining = self.leaves_remaining
        new.finished         = self.finished
        return new


def bds_copy_state(state: dict) -> dict:
    """Return an independent deep copy of a BDS08 traversal state.

    Label arrays, stacks, and treehash instances are all duplicated so that
    mutating the returned state leaves the input untouched.  Used by
    `LEMUR.sign_stateful` to preserve the caller's pre-sign snapshot.
    """
    return {
        "H":        state["H"],
        "K":        state["K"],
        "phi":      state["phi"],
        "auth":     [None if a is None else a.copy() for a in state["auth"]],
        "keep":     {h: lab.copy() for h, lab in state["keep"].items()},
        "retain":   {h: [lab.copy() for lab in labs]
                     for h, labs in state["retain"].items()},
        "treehash": {h: th.clone() for h, th in state["treehash"].items()},
    }


def bds_choose_k(tau: int) -> int:
    """Pick the BDS08 general-version parameter K.

    The BDS08 general traversal requires K >= 2 with (tau - K) even.  We take the smallest
    valid K: 2 for even tau, 3 for odd tau.
    """
    if tau < 2:
        return tau  # degenerate, nothing for BDS to do
    return 2 if tau % 2 == 0 else 3


class HVC:
    """Homomorphic Vector Commitment scheme.

    Requires a KOTS instance to read the shared ring dimension d,
    the KOTS modulus q' (used for leaf messages), and the KOTS key
    shape (rho = k, nu = n).
    """

    def __init__(self, kots: KOTS, *, profile: LemurProfile | None = None,
                 q=None, omega=None, eta=None, tau=None,
                 alpha_w=None, n_signers=None, beta_agg=None):
        if profile is None:
            profile = kots.profile
        self.profile   = profile
        self.d         = kots.d
        self.q         = q         if q         is not None else profile.q
        self.q_prime   = kots.q
        self.omega     = omega     if omega     is not None else profile.omega
        self.eta       = eta       if eta       is not None else profile.eta
        self.tau       = tau       if tau       is not None else profile.tau
        self.rho       = kots.k
        self.nu        = kots.n
        # Derived
        self.b_val     = 2 * self.eta + 1
        self.n_slots   = 1 << self.tau
        self.kappa     = math.ceil(math.log(self.q)       / math.log(self.b_val))
        self.kappa_prime = math.ceil(math.log(self.q_prime) / math.log(self.b_val))
        # beta_agg: aggregation-opening norm bound.  Profiles carry the authoritative
        # value from the ODS; if the caller overrides `tau` or `n_signers`
        # to a cell the profile doesn't cover, we recompute from the formula.
        aw       = alpha_w   if alpha_w   is not None else profile.alpha_w
        ns       = n_signers if n_signers is not None else profile.n_signers
        recompute = (beta_agg is None
                     and (self.tau != profile.tau or ns != profile.n_signers))
        if beta_agg is not None:
            self.beta_agg = beta_agg
        elif not recompute:
            self.beta_agg = profile.beta_agg
        else:
            eps = 1 / 32768
            n_coeffs = ns * (2 * self.tau * self.omega * self.kappa
                             + kots.k * kots.n * self.kappa_prime
                             + 2 * self.tau * self.omega)
            self.beta_agg = math.ceil(
                self.eta * math.sqrt(2 * aw * ns
                                     * (math.log(2 * kots.d / eps)
                                        + math.log(n_coeffs)))
            )
        self.beta_encode = math.ceil(self.beta_agg / (2 * self.eta))
        self.logq      = (self.q - 1).bit_length()
        self.bits_dig  = (2 * self.eta).bit_length()
        self.bits_diag = (2 * self.beta_agg).bit_length()
        self.bits_babai = (2 * self.beta_encode).bit_length()
        # Ring with NTT (HVC modulus)
        self.ring      = Ring(self.q, kots.d)

    # -----------------------------------------------------------------------
    # (2eta+1)-ary decomposition and projection
    # -----------------------------------------------------------------------

    def _decompose_coeff(self, c, modulus, kappa):
        """Balanced base-B decomposition of one integer into kappa digits."""
        c = int(c) % modulus
        if c > modulus // 2:
            c -= modulus
        digits = []
        for _ in range(kappa):
            r = c % self.b_val
            if r > self.eta:
                r -= self.b_val
            digits.append(r)
            c = (c - r) // self.b_val
        return digits

    def _dec_poly(self, a, modulus, kappa):
        """Decompose poly a into kappa digit polys.  Returns (kappa, d)."""
        result = np.zeros((kappa, self.d), dtype=np.int64)
        for i in range(self.d):
            for ki, dv in enumerate(self._decompose_coeff(int(a[i]), modulus, kappa)):
                result[ki, i] = dv
        return result

    def _proj_poly(self, digits):
        """Reconstruct poly from kappa digit polys (kappa, d) -> (d,)."""
        result = np.zeros(self.d, dtype=np.int64)
        base = np.int64(1)
        for k in range(digits.shape[0]):
            result += digits[k] * base
            base *= self.b_val
        return result

    def _dec_vec(self, v, modulus, kappa):
        """Decompose poly vector (n, d) into (n*kappa, d)."""
        n = v.shape[0]
        result = np.zeros((n * kappa, self.d), dtype=np.int64)
        for i in range(n):
            result[i * kappa:(i + 1) * kappa] = self._dec_poly(v[i], modulus, kappa)
        return result

    def _proj_vec(self, digits, kappa):
        """Reconstruct poly vector from (n*kappa, d) -> (n, d)."""
        n = digits.shape[0] // kappa
        result = np.zeros((n, self.d), dtype=np.int64)
        for i in range(n):
            result[i] = self._proj_poly(digits[i * kappa:(i + 1) * kappa])
        return result

    # -----------------------------------------------------------------------
    # Babai encoding for compressed HVC openings
    # -----------------------------------------------------------------------

    def _decompose_coeff_zz(self, c, kappa):
        """Balanced base-B decomposition over ZZ (no mod reduction)."""
        c = int(c)
        digits = []
        for _ in range(kappa):
            r = c % self.b_val
            if r > self.eta:
                r -= self.b_val
            digits.append(r)
            c = (c - r) // self.b_val
        return digits

    def _dec_poly_zz(self, a, kappa):
        """Decompose polynomial over ZZ into kappa digit polys. (kappa, d)."""
        result = np.zeros((kappa, self.d), dtype=np.int64)
        for i in range(self.d):
            for ki, dv in enumerate(self._decompose_coeff_zz(int(a[i]), kappa)):
                result[ki, i] = dv
        return result

    def babai_encode_block(self, digits, hint_block):
        """Babai-encode one omega-block of kappa digit polys.

        digits:     (kappa, d) — digit polynomials
        hint_block: (d,) — proj_label output for this block, in [0, q)

        Returns (a_star, alphas):
            a_star: (d,) — integer, bounded by ~beta_encode
            alphas: (kappa-1, d) — carry corrections, bounded by ~beta_encode
        """
        kappa = self.kappa
        q = self.q
        proj_zz = self._proj_poly(digits)
        hint_signed = hint_block.copy().astype(np.int64)
        hint_signed[hint_signed > q // 2] -= q
        diff = proj_zz - hint_signed
        assert np.all(diff % q == 0), "proj_zz - hint must be divisible by q"
        a_star = diff // q
        w = self._dec_poly_zz(proj_zz, kappa)
        alphas = np.zeros((kappa - 1, self.d), dtype=np.int64)
        residual = digits[0] - w[0]
        assert np.all(residual % self.b_val == 0)
        alphas[0] = -(residual // self.b_val)
        for k in range(1, kappa - 1):
            residual = digits[k] - w[k] - alphas[k - 1]
            assert np.all(residual % self.b_val == 0)
            alphas[k] = -(residual // self.b_val)
        return a_star, alphas

    def babai_decode_block(self, a_star, alphas, hint_block):
        """Babai-decode one omega-block back to kappa digit polys.

        a_star:     (d,)
        alphas:     (kappa-1, d)
        hint_block: (d,) in [0, q)

        Returns: (kappa, d) digit polynomials.
        """
        kappa = self.kappa
        q = self.q
        hint_signed = hint_block.copy().astype(np.int64)
        hint_signed[hint_signed > q // 2] -= q
        h_zz = hint_signed + q * a_star
        w = self._dec_poly_zz(h_zz, kappa)
        # Derive the last alpha from the carry
        w_sum = self._proj_poly(w)
        carry = (h_zz - w_sum) // (self.b_val ** kappa)
        alpha_last = -carry
        # Reconstruct delta_v from all kappa alphas
        digits = np.zeros((kappa, self.d), dtype=np.int64)
        digits[0] = w[0] - self.b_val * alphas[0]
        for k in range(1, kappa - 1):
            digits[k] = w[k] + alphas[k - 1] - self.b_val * alphas[k]
        digits[kappa - 1] = w[kappa - 1] + alphas[kappa - 2] - self.b_val * alpha_last
        return digits

    def babai_encode_label(self, label, hint):
        """Babai-encode a full label (omega*kappa, d) given hint (omega, d).

        Returns list of omega (a_star, alphas) tuples.
        """
        encoded = []
        for r in range(self.omega):
            block = label[r * self.kappa:(r + 1) * self.kappa]
            encoded.append(self.babai_encode_block(block, hint[r]))
        return encoded

    def babai_decode_label(self, encoded, hint):
        """Decode a full label from Babai data and hint (omega, d).

        encoded: list of omega (a_star, alphas) tuples.
        Returns: (omega*kappa, d) label.
        """
        label = np.zeros((self.omega * self.kappa, self.d), dtype=np.int64)
        for r in range(self.omega):
            a_star, alphas = encoded[r]
            block = self.babai_decode_block(a_star, alphas, hint[r])
            label[r * self.kappa:(r + 1) * self.kappa] = block
        return label

    def reconstruct_path_labels_ind(self, pp, t, sibling_labels, u):
        """Reconstruct individual-sig path labels from sibling labels and u.

        The path labels are exact decompositions of the hash-chain hints,
        so no Babai data is needed — they are fully deterministic.
        """
        B_mat, A0, A1 = pp
        hint = self.ring.mat_vec(B_mat, u)  # (omega, d) in [0, q)
        path_labels = [None] * self.tau
        for j in range(self.tau, 0, -1):
            path_labels[j - 1] = self._dec_vec(hint, self.q, self.kappa)
            s_j = sibling_labels[j - 1]
            bit = (t >> (self.tau - j)) & 1
            A_path = A0 if bit == 0 else A1
            A_sib = A1 if bit == 0 else A0
            hint = (self.ring.mat_vec(A_path, path_labels[j - 1])
                    + self.ring.mat_vec(A_sib, s_j)) % self.q
        return path_labels

    def reconstruct_path_labels_agg(self, pp, t, path_encoded, sibling_labels, u):
        """Reconstruct aggregated-sig path labels from Babai data.

        path_encoded: list of tau entries, each a list of omega (a_star, alphas).
        """
        B_mat, A0, A1 = pp
        hint = self.ring.mat_vec(B_mat, u)
        path_labels = [None] * self.tau
        for j in range(self.tau, 0, -1):
            path_labels[j - 1] = self.babai_decode_label(
                path_encoded[j - 1], hint
            )
            s_j = sibling_labels[j - 1]
            bit = (t >> (self.tau - j)) & 1
            A_path = A0 if bit == 0 else A1
            A_sib = A1 if bit == 0 else A0
            hint = (self.ring.mat_vec(A_path, path_labels[j - 1])
                    + self.ring.mat_vec(A_sib, s_j)) % self.q
        return path_labels

    # -----------------------------------------------------------------------
    # Message vectorisation: R^{rho x nu} <-> R^{rho*nu}
    # -----------------------------------------------------------------------

    def _poly_vec(self, M):
        return M.reshape(self.rho * self.nu, self.d)

    def _inv_poly_vec(self, v):
        return v.reshape(self.rho, self.nu, self.d)

    # -----------------------------------------------------------------------
    # Label functions for the HVC binary tree
    # -----------------------------------------------------------------------

    def _proj_label(self, label):
        """Project (omega*kappa, d) label to R_q^omega.  Returns (omega, d)."""
        result = np.zeros((self.omega, self.d), dtype=np.int64)
        for r in range(self.omega):
            result[r] = self._proj_poly(
                label[r * self.kappa:(r + 1) * self.kappa]
            ) % self.q
        return result

    def _proj_poly_zz(self, digits):
        """Unreduced ZZ-valued projection.  Returns (d,) of Python ints.

        Implements proj_{eta,kappa} from Sec 6.2 (codomain R, not R_q).
        Uses Python integers (dtype=object) because at wVrfy bounds the
        per-coefficient sum can exceed int64 range.
        """
        out = np.zeros(self.d, dtype=object)
        base = 1
        for k in range(digits.shape[0]):
            # element-wise: out[i] += int(digits[k, i]) * base
            for i in range(self.d):
                out[i] = out[i] + int(digits[k, i]) * base
            base *= self.b_val
        return out

    def _proj_label_zz(self, label):
        """Unreduced ZZ-valued projection of an HVC label.  Returns (omega, d)
        of Python ints.  Used for the proj-norm bound check in vrfy
        (paper Fig. HVC step 4c uses proj_{eta,kappa}, not proj_q)."""
        result = np.zeros((self.omega, self.d), dtype=object)
        for r in range(self.omega):
            result[r] = self._proj_poly_zz(
                label[r * self.kappa:(r + 1) * self.kappa]
            )
        return result

    def _leaf_label(self, M_t, B_mat):
        u   = self._dec_vec(self._poly_vec(M_t), self.q_prime, self.kappa_prime)
        raw = self.ring.mat_vec(B_mat, u)
        return self._dec_vec(raw, self.q, self.kappa)

    def _internal_label(self, left, right, A0, A1):
        contrib = (
            self.ring.mat_vec(A0, left) + self.ring.mat_vec(A1, right)
        ) % self.q
        return self._dec_vec(contrib, self.q, self.kappa)

    def _subtree_root(self, pp, leaf_fn, leaf_start, leaf_count):
        """Stack-based streaming subtree root computation.

        Processes leaves [leaf_start, leaf_start + leaf_count) left-to-right.
        Memory: O(log(leaf_count)) labels on the stack.
        leaf_count must be a power of 2.
        """
        B_mat, A0, A1 = pp
        stack = []  # list of (height, label)
        for i in range(leaf_count):
            leaf = self._leaf_label(leaf_fn(leaf_start + i), B_mat)
            node = (0, leaf)
            while stack and stack[-1][0] == node[0]:
                left_h, left = stack.pop()
                merged = self._internal_label(left, node[1], A0, A1)
                node = (left_h + 1, merged)
            stack.append(node)
        assert len(stack) == 1
        return stack[0][1]

    # -----------------------------------------------------------------------
    # HVC algorithms
    # -----------------------------------------------------------------------

    def setup(self, seed):
        """Setup(seed) -> (B, A0, A1)."""
        def _expand(tag, rows, cols):
            mat = np.zeros((rows, cols, self.d), dtype=np.int64)
            for i in range(rows):
                for j in range(cols):
                    xof = SHAKE128.new(seed + bytes([i, j]) + tag)
                    mat[i, j] = xof_uniform_poly(xof, self.q, self.d)
            return mat

        return (
            _expand(b'B',  self.omega, self.rho * self.nu * self.kappa_prime),
            _expand(b'A0', self.omega, self.omega * self.kappa),
            _expand(b'A1', self.omega, self.omega * self.kappa),
        )

    def com(self, pp, leaf_fn):
        """Com(pp, leaf_fn) -> c in R_q^omega.

        leaf_fn(t) returns the KOTS public key for slot t as (rho, nu, d).
        Streams over all n_slots leaves with O(tau) working memory.
        """
        root = self._subtree_root(pp, leaf_fn, 0, self.n_slots)
        return self._proj_label(root)

    # -----------------------------------------------------------------------
    # BDS08 stateful tree traversal
    # -----------------------------------------------------------------------

    def bds_init(self, pp, leaf_fn, K=None):
        """Build initial BDS08 state and root commitment in a single pass.

        The walk produces the HVC root (so keygen can reuse it) while saving
        the specific nodes needed by the algorithm:
          - auth[h]         = y_h[1]             for h in 0..tau-1
          - treehash[h].node = y_h[3]             for h in 0..tau-K-1
          - retain[h]       = [y_h[3], y_h[5], ...]
                                                  for h in tau-K..tau-2

        Returns (c, state) where c is the root label projected to R_q^omega.
        """
        H = self.tau
        if K is None:
            K = bds_choose_k(H)
        B_mat, A0, A1 = pp

        treehash = {h: TreehashInst(h) for h in range(max(H - K, 0))}
        retain   = {h: [] for h in range(max(H - K, 0), max(H - 1, 0))}
        state = {
            "H":        H,
            "K":        K,
            "phi":      0,
            "auth":     [None] * H,
            "keep":     {},
            "retain":   retain,
            "treehash": treehash,
        }

        def maybe_save(h, j, lab):
            if h < H and j == 1:
                state["auth"][h] = lab.copy()
            if h < H - K and j == 3:
                state["treehash"][h].set_ready(lab.copy())
            if H - K <= h < H - 1 and j >= 3 and (j & 1) == 1:
                state["retain"][h].append(lab.copy())

        stack: list = []  # list of (height, index, label)
        n_leaves = 1 << H
        for i in range(n_leaves):
            leaf_lab = self._leaf_label(leaf_fn(i), B_mat)
            cur = (0, i, leaf_lab)
            maybe_save(*cur)
            while stack and stack[-1][0] == cur[0]:
                lh, lj, ll = stack.pop()
                merged = self._internal_label(ll, cur[2], A0, A1)
                cur = (lh + 1, lj // 2, merged)
                maybe_save(*cur)
            stack.append(cur)

        assert len(stack) == 1 and stack[0][0] == H
        root_label = stack[0][2]
        return self._proj_label(root_label), state

    def bds_advance(self, state, pp, leaf_fn):
        """Advance state from 'auth path for leaf phi' to 'auth path for phi+1'.

        Implements the general BDS08 traversal step.
        State must be at phi in [0, 2^tau - 1); callers must not advance
        past the last leaf.
        """
        H = state["H"]
        K = state["K"]
        phi = state["phi"]
        assert 0 <= phi < (1 << H) - 1, \
            f"bds_advance: phi={phi} out of range for H={H}"
        B_mat, A0, A1 = pp

        # 1. tau_local: height of the first parent of leaf phi that is a left node
        if (phi & 1) == 0:
            tau_l = 0
        else:
            tau_l = 0
            m = phi + 1
            while (m & 1) == 0:
                tau_l += 1
                m >>= 1

        # 2. Save current Auth_tau into Keep_tau if it will be needed later.
        #    Condition: parent of phi on height tau+1 is a left node (so the
        #    current Auth_tau is a right node), and tau < H-1 (we never keep
        #    the sibling at the root level).
        if tau_l < H - 1 and ((phi >> (tau_l + 1)) & 1) == 0:
            state["keep"][tau_l] = state["auth"][tau_l].copy()

        # 3. If leaf phi is a left node, the new leaf-level auth entry is
        #    simply y_0[phi] (the left sibling of leaf phi+1).
        if tau_l == 0:
            state["auth"][0] = self._leaf_label(leaf_fn(phi), B_mat)
        else:
            # 4(a) new right auth node on height tau
            state["auth"][tau_l] = self._internal_label(
                state["auth"][tau_l - 1],
                state["keep"][tau_l - 1],
                A0, A1,
            )
            del state["keep"][tau_l - 1]

            # 4(b) fetch new right auth nodes on heights 0..tau-1
            for h in range(tau_l):
                if h < H - K:
                    th = state["treehash"][h]
                    assert th.finished and th.node is not None, \
                        f"treehash_{h} not ready at phi={phi}"
                    state["auth"][h] = th.node
                    th.set_ready(None)  # clear for reuse
                    th.finished = False
                else:
                    # retain queue; consume in FIFO order (oldest = smallest j)
                    state["auth"][h] = state["retain"][h].pop(0)

            # 4(c) reinitialize treehash instances for heights 0..min(tau-1, H-K-1)
            for h in range(min(tau_l, H - K)):
                new_start = phi + 1 + 3 * (1 << h)
                if new_start < (1 << H):
                    state["treehash"][h].initialize(new_start)

        # 5. Spend the budget of (H-K)/2 treehash updates.
        budget = max((H - K) // 2, 0)
        for _ in range(budget):
            best_h = None
            best_metric = math.inf
            for h in range(H - K):
                th = state["treehash"][h]
                metric = th.height_metric()
                if metric < best_metric:
                    best_metric = metric
                    best_h = h
            if best_h is None or math.isinf(best_metric):
                break
            state["treehash"][best_h].update(self, pp, leaf_fn)

        state["phi"] = phi + 1
        return state

    def bds_opening(self, state, pp, t, leaf_fn):
        """Assemble an HVC opening for slot `t` from a BDS state.

        The state must hold the auth path for leaf `t`.  Returns
        (path_labels, sibling_labels, u) in the same format that
        `HVC.open` produces.

        Raises:
            ValueError: if the state's current slot does not match `t`.
                This guards against user-side state-machine errors and
                is a real exception (not an `assert`) so it survives
                `python -O`.
        """
        if state["phi"] != t:
            raise ValueError(
                f"BDS state at phi={state['phi']} cannot open slot {t}"
            )
        B_mat, _, _ = pp
        leaf_opk = leaf_fn(t)
        u = self._dec_vec(
            self._poly_vec(leaf_opk), self.q_prime, self.kappa_prime
        )
        # sibling_labels[j-1] = sibling at depth j = auth[tau-j]
        sibling_labels = [state["auth"][self.tau - 1 - i].copy()
                          for i in range(self.tau)]
        path_labels = self.reconstruct_path_labels_ind(
            pp, t, sibling_labels, u
        )
        return path_labels, sibling_labels, u

    # -----------------------------------------------------------------------
    # Streaming open (used by sign_seed and by tests)
    # -----------------------------------------------------------------------

    def open(self, pp, t, leaf_fn):
        """Open(pp, t, leaf_fn) -> (path_labels, sibling_labels, u).

        leaf_fn(t) returns the KOTS public key for slot t as (rho, nu, d).
        Computes tau sibling subtree roots then builds path labels bottom-up.
        """
        B_mat, A0, A1 = pp

        # Sibling at depth j (1-indexed) covers 2^(tau-j) leaves
        sibling_labels = []
        for j in range(1, self.tau + 1):
            h = self.tau - j
            sib_start = ((t >> h) ^ 1) << h
            sib_count = 1 << h
            sib_label = self._subtree_root(
                pp, leaf_fn, sib_start, sib_count
            )
            sibling_labels.append(sib_label)

        # Compute leaf for slot t
        leaf_opk = leaf_fn(t)
        u = self._dec_vec(self._poly_vec(leaf_opk), self.q_prime, self.kappa_prime)
        leaf = self._leaf_label(leaf_opk, B_mat)

        # Build path labels bottom-up
        path_labels = [None] * self.tau
        path_labels[self.tau - 1] = leaf.copy()
        current = leaf
        for j in range(self.tau - 1, 0, -1):
            bit = (t >> (self.tau - j - 1)) & 1
            sib = sibling_labels[j]
            if bit == 0:
                left, right = current, sib
            else:
                left, right = sib, current
            parent = self._internal_label(left, right, A0, A1)
            path_labels[j - 1] = parent.copy()
            current = parent

        return path_labels, sibling_labels, u

    def vrfy(self, pp, c, t, d_open, beta):
        """Vrfy(pp, c, t, d_open, beta) -> T (recovered KOTS pk) or None."""
        B_mat, A0, A1 = pp
        path_labels, sibling_labels, u = d_open
        t_bar = tuple(int(b) for b in f"{t:0{self.tau}b}")

        if inf_norm(u) > beta:
            return None

        hint = self.ring.mat_vec(B_mat, u)

        for j in range(self.tau, 0, -1):
            p_bar_j = path_labels[j - 1]
            s_j     = sibling_labels[j - 1]

            if not np.all(self._proj_label(p_bar_j) == hint):
                return None
            p_j = p_bar_j

            if inf_norm(p_j) > beta or inf_norm(s_j) > beta:
                return None

            # proj_{eta,kappa} (Sec 6.2 over ZZ, not mod q) — paper Fig. step 4c.
            # (2*eta+1)^kappa > q, so reducing mod q before bounding would
            # discard the carry that the threshold is meant to constrain.
            threshold  = self.q * beta // (2 * self.eta)
            p_j_proj   = self._proj_label_zz(p_j)
            s_j_proj   = self._proj_label_zz(s_j)
            if (max(abs(int(x)) for x in p_j_proj.ravel()) > threshold or
                max(abs(int(x)) for x in s_j_proj.ravel()) > threshold):
                return None

            bit = t_bar[j - 1]
            A_path    = A0 if bit == 0 else A1
            A_sibling = A1 if bit == 0 else A0
            hint = (
                self.ring.mat_vec(A_path, p_j) +
                self.ring.mat_vec(A_sibling, s_j)
            ) % self.q

        if not np.all(c == hint):
            return None

        opk_flat = self._proj_vec(u, self.kappa_prime) % self.q_prime
        return self._inv_poly_vec(opk_flat)

    def ivrfy(self, pp, c, t, d_open):
        return self.vrfy(pp, c, t, d_open, self.eta)

    def svrfy(self, pp, c, t, d_open):
        return self.vrfy(pp, c, t, d_open, self.beta_agg)

    def wvrfy(self, pp, c, t, d_open):
        return self.vrfy(pp, c, t, d_open, 2 * self.beta_agg)


# ---------------------------------------------------------------------------
# Quick correctness test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kots = KOTS()
    hvc  = HVC(kots, tau=3)  # small tau for quick self-test
    print(f"HVC: d={hvc.d}, q={hvc.q}, omega={hvc.omega}, eta={hvc.eta}, "
          f"tau={hvc.tau} ({hvc.n_slots} slots)")
    print(f"  B={hvc.b_val}, kappa={hvc.kappa}, kappa'={hvc.kappa_prime}")
    print(f"  rho={hvc.rho}, nu={hvc.nu}, beta_agg={hvc.beta_agg}")
    print(f"  ring zeta={hvc.ring.zeta}")
    print()

    A      = kots.setup(bytes(32))
    hvc_pp = hvc.setup(bytes(range(32)))

    # Pre-generate all KOTS keys and build leaf_fn
    M    = np.zeros((hvc.n_slots, hvc.rho, hvc.nu, hvc.d), dtype=np.int64)
    for i in range(hvc.n_slots):
        _, opk = kots.keygen(A, i.to_bytes(32, 'little'))
        M[i] = opk
    leaf_fn = lambda t: M[t]

    c = hvc.com(hvc_pp, leaf_fn)
    for t in range(hvc.n_slots):
        d_open = hvc.open(hvc_pp, t, leaf_fn)
        ok = hvc.ivrfy(hvc_pp, c, t, d_open) is not None
        print(f"  slot {t}: {'PASS' if ok else 'FAIL'}")

    d0 = hvc.open(hvc_pp, 0, leaf_fn)
    wrong_1 = hvc.ivrfy(hvc_pp, c, 1, d0)
    wrong_4 = hvc.ivrfy(hvc_pp, c, 4, d0)
    print(f"\n  slot 0 opening at slot 1: "
          f"{'FAIL (expected)' if wrong_1 is None else 'PASS (unexpected!)'}")
    print(f"  slot 0 opening at slot 4: "
          f"{'FAIL (expected)' if wrong_4 is None else 'PASS (unexpected!)'}")
