from math import *
from model_BKZ import *
from proba_util import gaussian_center_weight

log_infinity = 9999

STEPS_b = 5
STEPS_m = 5


########################################
# CACHES (major speedup)
########################################

BKZ_SHAPE_CACHE = {}
BKZ_LENGTH_CACHE = {}
SVP_COST_CACHE = {}
NVEC_CACHE = {}


def cached_BKZ_shape(q, h, w, b):
    key = (q, h, w, b)
    if key not in BKZ_SHAPE_CACHE:
        BKZ_SHAPE_CACHE[key] = construct_BKZ_shape_randomized(q, h, w, b)
    return BKZ_SHAPE_CACHE[key]


def cached_BKZ_first_length(q, h, w, b):
    key = (q, h, w, b)
    if key not in BKZ_LENGTH_CACHE:
        BKZ_LENGTH_CACHE[key] = BKZ_first_length(q, h, w, b)
    return BKZ_LENGTH_CACHE[key]


def get_svp_cost(b, model):
    key = (b, model)
    if key not in SVP_COST_CACHE:
        SVP_COST_CACHE[key] = model(b)
    return SVP_COST_CACHE[key]


def get_nvec(b):
    if b not in NVEC_CACHE:
        NVEC_CACHE[b] = nvec_sieve(b)
    return NVEC_CACHE[b]


########################################
# PARAMETER CLASS
########################################

class MSISParameterSet:
    def __init__(self, n, w, h, B, q, norm=""):
        self.n = n
        self.w = w
        self.h = h
        self.B = B
        self.q = q
        self.norm = norm


########################################
# L2 ATTACK
########################################

def SIS_l2_cost(q, w, h, B, b, cost_svp=svp_classical, verbose=False):

    if B >= q:
        if verbose:
            print("Cannot handle B >= q in l2 norm")
        return 0

    l = cached_BKZ_first_length(q, h, w - h, b)

    if l > B:
        return log_infinity

    if verbose:
        print("Attack uses block-size %d and %d equations" % (b, h))
        print("shortest vector length %.2f (q=%d)" % (l, q))

    return get_svp_cost(b, cost_svp)


########################################
# LINF ATTACK
########################################

def SIS_linf_cost(q, w, h, B, b, cost_svp=svp_classical, verbose=False):

    (i, j, L) = cached_BKZ_shape(q, h, w - h, b)

    l = exp(L[i])
    d = j - i + 1

    sigma = l / sqrt(d)

    p_middle = gaussian_center_weight(sigma, B)
    p_head = 2.0 * B / q

    log2_eps = d * log(p_middle, 2) + i * log(p_head, 2)

    log2_R = max(0, -log2_eps - get_nvec(b))

    if verbose:
        print("Attack uses block-size %d in dimension %d" % (b, w))
        print("log2(epsilon) = %.2f" % log2_eps)
        print("shortest vector length %.2f" % l)

    return get_svp_cost(b, cost_svp) + log2_R


########################################
# ATTACK OPTIMIZATION
########################################

def SIS_optimize_attack(q, max_w, h, B,
                        cost_attack=SIS_linf_cost,
                        cost_svp=svp_classical,
                        verbose=False):

    best_cost = log_infinity
    best_w = None
    best_b = None

    w = max_w

    for b in range(50, max_w, STEPS_b):

        svp_cost = get_svp_cost(b, cost_svp)

        if svp_cost > best_cost:
            break

        cost = cost_attack(q, w, h, B, b, cost_svp)

        if cost <= best_cost:
            best_cost = cost
            best_w = w
            best_b = b

    if verbose and best_b is not None:
        cost_attack(q, best_w, h, B, best_b,
                    cost_svp=cost_svp,
                    verbose=True)

    return (best_w, best_b, best_cost)


########################################
# CONSISTENCY CHECK
########################################

def check_eq(a, b, c):
    if a != b or b != c:
        print("Warning: parameters differ between models")


########################################
# MAIN SUMMARY FUNCTION
########################################

def MSIS_summarize_attacks(ps):

    q = ps.q
    h = ps.n * ps.h
    max_w = ps.n * ps.w
    B = ps.B

    if ps.norm == "linf":
        attack = SIS_linf_cost
    elif ps.norm == "l2":
        attack = SIS_l2_cost
    else:
        raise ValueError("Unknown norm: " + ps.norm)

    (m_pc, b_pc, c_pc) = SIS_optimize_attack(
        q, max_w, h, B,
        cost_attack=attack,
        cost_svp=svp_classical,
        verbose=False
    )

    (m_pq, b_pq, c_pq) = SIS_optimize_attack(
        q, max_w, h, B,
        cost_attack=attack,
        cost_svp=svp_quantum
    )

    (m_pp, b_pp, c_pp) = SIS_optimize_attack(
        q, max_w, h, B,
        cost_attack=attack,
        cost_svp=svp_plausible
    )

    check_eq(m_pc, m_pq, m_pp)
    check_eq(b_pc, b_pq, b_pp)

   
    return (
        b_pq,
        int(floor(c_pc)),
        int(floor(c_pq)),
        int(floor(c_pp))
    )