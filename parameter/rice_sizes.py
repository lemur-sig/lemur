#!/usr/bin/env python3
"""Compute Lemur Rice-encoded aggregate-size estimates from summary.txt.

The Rust/Python implementations ship the tau=20, N=1024 parameter cell.  The
paper's larger-N aggregate-size entries apply the same Rice-encoding model to
the corresponding estimator rows in this directory.  This helper reproduces
those paper entries directly from summary.txt.
"""

from __future__ import annotations

import math
from pathlib import Path


SUMMARY = Path(__file__).with_name("summary.txt")
TARGETS = {(20, 1024), (20, 32768), (20, 1048576)}


def bit_length(v: int) -> int:
    return int(v).bit_length()


def poly_bytes(d: int, dx: int) -> int:
    return (d * dx + 7) // 8


def rice_params(sigma: float, max_bound: int) -> tuple[str, int]:
    dx_fixed = bit_length(2 * max_bound)
    mu = 0.7979 * max(sigma, 0.0)

    best_k = 0
    best_bits = mu + 2.0
    for k in range(1, dx_fixed + 1):
        bits = k + 2.0 + mu / (1 << k)
        if bits < best_bits:
            best_bits = bits
            best_k = k

    if best_bits < dx_fixed:
        return ("rice", best_k)
    return ("fixed", dx_fixed)


def rice_bits_per_coef(sigma: float, rice_k: int) -> float:
    """Expected Rice codeword bits per coefficient under X ~ N(0, sigma^2).

    Codeword length:
      L = k+1                       if  X = 0
      L = k+2+j  (j >= 0)           if  j*2^k <= |X| < (j+1)*2^k
    E[L] = k + 2 - P(X=0) + 2 * sum_{j>=1} (1 - Phi(j*2^k/sigma)).

    Tight; the earlier `k + 0.7979*sigma/2^k + 2` substituted
    E[|X|]/2^k for E[floor(|X|/2^k)] and biased the estimate ~3% high.
    """
    sigma = max(float(sigma), 0.0)
    if sigma == 0.0:
        return float(rice_k + 1)
    p_zero = math.erf(0.5 / (sigma * math.sqrt(2)))
    step = float(1 << rice_k)
    inv_sqrt2 = 1.0 / math.sqrt(2)
    e_floor = 0.0
    j = 1
    while True:
        p = math.erfc(j * step / sigma * inv_sqrt2)
        e_floor += p
        if p < 1e-15 and j > 5:
            break
        j += 1
        if j > 10_000:
            break
    return rice_k + 2.0 - p_zero + e_floor


def rice_poly_bytes_est(d: int, sigma: float, rice_k: int) -> int:
    return math.ceil(d * rice_bits_per_coef(sigma, rice_k) / 8.0)


def encoded_size(row: dict[str, int | float]) -> int:
    n_signers = int(row["N"])
    d = int(row["d"])
    tau = int(row["tau"])
    ell = int(row["ell"])
    k = int(row["k"])
    n = int(row["n"])
    m = int(row["m"])
    omega = int(row["omega"])
    eta = int(row["eta"])
    kappa = int(row["kappa"])
    kappa_prime = int(row["kappaprime"])
    alpha = float(row["alpha"])
    alpha_h = float(row["alpha_H"])
    alpha_w = float(row["alpha_w"])
    beta_sigma = int(row["beta_sigma"])
    beta_agg = int(row["beta_agg"])
    beta_encode = int(row["beta_encode"])

    var_digit = eta * (eta + 1.0) / 3.0
    sigma_label = math.sqrt(n_signers * alpha_w * var_digit)
    sigma = alpha / math.sqrt(2.0 * math.pi)
    sigma_z_ind = sigma * math.sqrt(1.0 + (k - ell) * alpha_h)
    sigma_zagg = sigma_z_ind * math.sqrt(n_signers * alpha_w)
    sigma_babai = sigma_label / (2.0 * eta)

    n_zagg_coeffs = ell * m * d
    c_zagg = math.sqrt(2.0 * math.log(2.0 * n_zagg_coeffs * 256.0))
    zagg_bound = min(math.ceil(c_zagg * sigma_zagg), beta_sigma)
    zagg_dx = bit_length(2 * zagg_bound)
    pb_zagg = poly_bytes(d, zagg_dx)

    babai_mode, babai_param = rice_params(sigma_babai, beta_encode)
    agg_mode, agg_param = rice_params(sigma_label, beta_agg)
    pb_babai = (
        rice_poly_bytes_est(d, sigma_babai, babai_param)
        if babai_mode == "rice"
        else poly_bytes(d, babai_param)
    )
    pb_agg = (
        rice_poly_bytes_est(d, sigma_label, agg_param)
        if agg_mode == "rice"
        else poly_bytes(d, agg_param)
    )

    n_label = omega * kappa
    n_u = k * n * kappa_prime
    babai_total = tau * omega * kappa * pb_babai
    sib_total = tau * n_label * pb_agg
    u_total = n_u * pb_agg
    return 1 + ell * m * pb_zagg + babai_total + sib_total + u_total


def parse_summary() -> list[dict[str, int | float]]:
    rows = []
    keys = [
        "secpar",
        "tau",
        "N",
        "d",
        "epsilon",
        "alpha_w",
        "gamma",
        "ell",
        "k",
        "n",
        "m",
        "omega",
        "RHF_LWE_KOTS",
        "RHF_SIS_KOTS",
        "RHF_SIS_HVC",
        "alpha",
        "alpha_mlwe",
        "alpha_H",
        "beta_z",
        "beta_sigma",
        "beta_agg",
        "beta_encode",
        "eta",
        "q",
        "q_bit",
        "kappa",
        "qprime",
        "qprime_bit",
        "kappaprime",
    ]
    for line in SUMMARY.read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        values = parts[: len(keys)]
        row: dict[str, int | float] = {}
        for key, value in zip(keys, values):
            if "/" in value:
                row[key] = value
            elif "." in value:
                row[key] = float(value)
            else:
                row[key] = int(value)
        rows.append(row)
    return rows


def main() -> None:
    print("tau,N,worst_case_KB,rice_encoded_KB")
    for row in parse_summary():
        tau = int(row["tau"])
        n_signers = int(row["N"])
        if (tau, n_signers) not in TARGETS:
            continue
        # summary.txt stores the worst-case total as the final numeric KB value.
        parts = next(
            line.split()
            for line in SUMMARY.read_text().splitlines()
            if line.split()[:3] == [str(row["secpar"]), str(tau), str(n_signers)]
        )
        worst_case_kb = int(parts[-2])
        rice_kb = encoded_size(row) / 1024.0
        print(f"{tau},{n_signers},{worst_case_kb},{rice_kb:.1f}")


if __name__ == "__main__":
    main()
