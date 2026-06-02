"""
lemur_param.sage
This code is built upon the Chipmunk parameter scripts:
https://github.com/GottfriedHerold/Chipmunk/tree/open-source/scripts

This file searches for Lemur parameters using the MSIS estimator and the
lattice estimator for MLWE estimation.
"""


try:
  from tabulate import tabulate
except ModuleNotFoundError:
  def tabulate(rows, headers, tablefmt=None):
    rows = [[str(x) for x in row] for row in rows]
    headers = [str(x) for x in headers]
    widths = [
      max([len(headers[i])] + [len(row[i]) for row in rows])
      for i in range(len(headers))
    ]
    def fmt(row):
      return "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(widths)))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows])

import sys
from pathlib import Path
current_script_dir = Path(sys.argv[0]).resolve().parent
sys.path.append(str(current_script_dir.resolve()))
from estimator import *

# Optimized variant of the file "MSIS_security.py".
msis_estimator_path = current_script_dir / "msis_estimator"
sys.path.append(str(msis_estimator_path.resolve()))
file_to_load = msis_estimator_path / "MSIS_security_optimized.py"
load(str(file_to_load.resolve()))



# uncomment the following to use the unoptimized MSIS estimator
# load("MSIS_security.py")


def round_half_up(x):
  return floor(x + 0.5) if x >= 0 else ceil(x - 0.5)

def cardinality_of_set_of_ternary_poly(d,alpha):
  """ Determines the size of the set of ternary polynomials of degree d and Hamming weight alpha.
  """

  return binomial(d,alpha)*2^alpha


def find_hamming_weight(d,l):
  """ Finds the minimal value alpha, such that the set of ternary polynomials of degree d and Hamming weight alpha has size at least l.
  """

  for alpha in range(d+1):
    if cardinality_of_set_of_ternary_poly(d,alpha) >= l:
      return alpha
  raise ValueError("There does not exist a Hamming weight satisfying the specified conditions.")


def get_alpha_w(secpar,d):
  """ Determines the minimal value alpha_w, such that |T_{alpha_w}| >= 2^secpar is satisfied.
  """
  return find_hamming_weight(d,2^secpar)


def find_hvc_prime(d,beta):
  """ Finds the smallest prime q > beta, such that Z[x]/(x^d + 1) is NTT friendly.
  """

  q = next_prime(beta)
  # Negacyclic NTT friendliness only requires q ≡ 1 (mod 2*d), but the
  # April 2026 spreadsheet uses the stronger q ≡ 1 (mod 4*d) convention.
  # Keep that convention here so the estimator reproduces the shipped cells.
  while q % (4*d) != 1:
    q = next_prime(q)
  return q


def find_kots_prime(d,beta):
  """ Finds the smallest CRT-friendly KOTS prime q' > beta. We set t2 = 8 satisfying Constraint 11.
  """
  q = next_prime(beta)
  while q % 32 != 17:
    q = next_prime(q)
  return q


def get_alpha_and_alpha_mlwe(d, k):
  # These values are obtained from the numerical experiments from alpha.sage
  if d == 256 and k == 4:
    return (83, 1.6)

  raise ValueError("Unsupported (d, k) pair for current alpha values")

 


def msis_estimation(q, d, nrows, ncols, beta_infty, RHF):
    try:
      params = MSISParameterSet(d, ncols, nrows, beta_infty, q, norm="linf")
      result = MSIS_summarize_attacks(params)
      if result is None:
        return False, None

      bkz_block = result[0]
      rhf = round(delta_BKZ(bkz_block), 5)

      return (rhf <= RHF), rhf
    except Exception:
      return False, None


def get_alpha_H(secpar,d,k):
  for alpha_H in range(d+1):
    num = cardinality_of_set_of_ternary_poly(d, alpha_H)
    if (k - 3) * ln(num) >= 2 * secpar * ln(2):
      return alpha_H
  raise ValueError("Can't find alpha_H.")





def get_beta_z(alpha,alpha_H,k):
  """ Determines the norm bound of individual signature 
  """
  return ZZ(ceil(6*alpha*sqrt(1+(k-1)*alpha_H)))

def get_beta_sigma(beta_z,alpha_w,N,m,d,epsilon_hom):
  """ Determines the norm bound of aggregated signature beta_sigma
  """
  return ZZ(ceil(beta_z*sqrt(2*alpha_w*N*ln(2*m*d/epsilon_hom))))


def get_beta_kots(alpha_w,beta_sigma,beta_z):
  """ Calculates the norm bound of the MSIS instance corresponding to the given KOTS parameters.
  """
  return ZZ(2*beta_sigma+2*alpha_w*beta_z)


def get_beta_agg(d,tau,k,n,omega,eta,kappa,kappaprime,alpha_w,N,epsilon_hom):
  """ Determines the minimal value beta_agg, such that Constraint 1 is satisfied.
  """
  return ZZ(ceil(eta * sqrt(2 * alpha_w * N * (ln(2 * d / epsilon_hom) + ln(2 * tau * omega * kappa + k * n * kappaprime + 2 * tau * omega)))))



def encoded_elements_size(d, eta, beta_agg, number_of_ring_element):
  """ compute the encoded size of a dim-d ring element whose norm is bounded by beta_agg
  """

  # size of alpha_star, alpha_1, alpha_2, ..., alpha_{number_of_ring_element-1}
  beta_encoded = ceil(beta_agg/(2*eta))
  alpha_sizes = number_of_ring_element * ceil(log(2*beta_encoded+1, 2)) * d

  return alpha_sizes


def compute_total_size_bits(alpha_w, gamma, kots_param, hvc_param):
  return (
    kots_param["size"]
    + hvc_param["size"]
    + ceil(log(gamma, 2))
    )



def find_kots_params(d, secpar, N, alpha_w, epsilon_hom, k, RHF):
  """ Finds parameters for the key homomorphic one-time signature scheme compatible with the inputs.

  Specifically, the parameters should result in a scheme with secpar bits security that supports aggregation of up to N signatures.
  When aggregating using uniformly random ternary polynomials with Hamming weight alpha_w, the aggregated signature will verify with probability at least 1 - epsilon_hom.
  """

  # Constraint 15 in Lemur Parameter Constraints Table
  alpha_H = get_alpha_H(secpar, d, k)
  # Constraints 5, 13, 16 in Lemur Parameter Constraints Table
  (alpha, alpha_mlwe) = get_alpha_and_alpha_mlwe(d, k)
  # Constraint 6 in Lemur Parameter Constraints Table
  beta_z = get_beta_z(alpha, alpha_H, k)


  params = {}
  min_size = oo 
  # Range start/end dynamic based on d
  n_start = ceil(512/d)
  n_end = ceil(2048/d) + 1
  offset_start = ceil(512/d)
  offset_end = ceil(2048/d) + 1
  for n in range(n_start, n_end): 
    for m in range(n+offset_start, n+offset_end):
      # Constraint 8 in Lemur Parameter Constraints Table
      beta_sigma = get_beta_sigma(beta_z,alpha_w,N,m,d,epsilon_hom)

      size = m * d * ceil(log(2 * beta_sigma + 1,2)) 
      if size >= min_size:
        continue
      
      beta_kots = get_beta_kots(alpha_w,beta_sigma,beta_z)
      base_q = 2 * beta_kots
      for q_multiplier in [1, 2, 4, 8, 16, 32, 64, 128, 256]: 
        # Constraint 11 in Lemur Parameter Constraints Table
        q = find_kots_prime(d, base_q * q_multiplier)


        # Constraint 17 in Lemur Parameter Constraints Table
        msis_ok, RHF_SIS_KOTS = msis_estimation(q, d, n, m, beta_kots, RHF)
        if not msis_ok:
          continue

        # Constraint 16 in Lemur Parameter Constraints Table
        lwe_param = LWE.Parameters(n=(m-n)*d, q=q, Xs=ND.DiscreteGaussian(alpha_mlwe), Xe=ND.DiscreteGaussian(alpha_mlwe))
        RHF_LWE_KOTS = LWE.estimate.rough(lwe_param)['usvp']['delta']
        if RHF_LWE_KOTS > RHF:
          continue
        min_size = size
        params = {"alpha": alpha, "alpha_mlwe": alpha_mlwe, "alpha_H" : alpha_H, "n": n, "m": m, "beta_z": beta_z, "beta_sigma" : beta_sigma, "q'" : q, "size" : size, "RHF_LWE_KOTS": RHF_LWE_KOTS, "RHF_SIS_KOTS": RHF_SIS_KOTS, "qprime_bit": ceil(log(q, 2))}
        break

  if not params:
    return None
   
  return params


def find_hvc_params(d, secpar, N, tau, alpha_w, qprime, epsilon_hom, k, n, RHF):
  """ Finds parameters for the homomorphic vector commitment compatible with the inputs.

  Specifically, the parameters should result in a vector commitment with secpar bits security that supports vectors of length 2^tau of payloads consisting of k * n R_{qprime} elements and aggregation of up to N openings.
  When aggregating using uniformly random ternary polynomials with Hamming weight alpha_w, the aggregated opening will verify correctly with probability at least 1-epsilon_hom.
  """


  hvc_params = {}
  hvc_min_size = oo       

  for qbit in range(16,65):
    for omega in range(512/d, 2048/d+1):
      for guess_kappa in range(2, 9):
        # this q is a guessed "q"
        q = pow(2, qbit)

        # the following eta, kappa, kappaprime are all "guessed", we will correct them after determining q
        eta = ceil((pow(q, 1.0/guess_kappa) - 1 )/2.0)
        kappa = ceil(log(q,2*eta+1))
        kappaprime = ceil(log(qprime,2*eta+1))
        # Constraint 1 in Lemur Parameter Constraints Table
        beta_agg = get_beta_agg(d, tau, k, n, omega, eta, kappa, kappaprime, alpha_w, N, epsilon_hom)
        beta_hvc = 4 * beta_agg
        if q < 2 * beta_hvc:
          continue

        q = find_hvc_prime(d, q)
        if kappa != ceil(log(q,2*eta+1)):
          continue

        size = encoded_elements_size(d, eta, beta_agg, tau * omega * kappa)
        size += d * ceil(log(2*beta_agg+1, 2)) * tau * omega * kappa
        size += d * ceil(log(2*beta_agg+1, 2)) * k * n * kappaprime
        if size >= hvc_min_size:  
          continue
        else:
          # Constraint 2 in Lemur Parameter Constraints Table
          check1 = msis_estimation(q, d, omega, 2 * omega * kappa, beta_hvc, RHF)[0]
          # Constraint 3 in Lemur Parameter Constraints Table
          check2 = msis_estimation(q, d, omega, k * n * kappaprime, beta_hvc, RHF)[0]
          msis_is_hard = (check1 and check2)
          if msis_is_hard: 
            rhf1 = msis_estimation(q, d, omega, 2 * omega * kappa, beta_hvc, RHF)[1]
            rhf2 = msis_estimation(q, d, omega, k * n * kappaprime, beta_hvc, RHF)[1]
            RHF_SIS_HVC = max(rhf1, rhf2)
            hvc_min_size = size
            # Constraint 4 in Lemur Parameter Constraints Table
            beta_encode = ZZ(ceil(beta_agg/(2*eta)))
            hvc_params = {"eta" : eta, "kappa" : kappa, "kappa'" : kappaprime ,"beta_agg" : beta_agg, "beta_encode" : beta_encode, "q" : q, "SIS beta" : beta_hvc, "size" : size, "epsilon_hom" : epsilon_hom, "omega": omega, "RHF_SIS_HVC": RHF_SIS_HVC, "q_bit": ceil(log(q, 2))}

  if not hvc_params:
    return None
  return hvc_params

def find_param(d, secpar, N, tau, epsilon_hom, k, RHF):
  """
    Finds parameters for the Lemur multi-signature scheme compatible with the inputs.

    Specifically, the parameters should result in a synchronized multi-signature scheme with secpar bits security that supports 2^tau time periods and aggregation of up to N signatures, where any individual aggregation attempt will fail with probability at most epsilon_hom.
  """
  print("Finding params for secpar = " + str(secpar) + " tau = " + str(tau) + " N = " + str(N) + ", and epsilon_hom=" + str(epsilon_hom))

  
  # Constraint 19 in Lemur Parameter Constraints Table
  alpha_w = get_alpha_w(secpar,d)
  # Constraint 18 in Lemur Parameter Constraints Table
  gamma = ZZ(ceil(secpar/log(1/(2*epsilon_hom), 2)))

  kots_param = find_kots_params(d, secpar, N, alpha_w, epsilon_hom, k, RHF)
  if kots_param is None:
    raise ValueError("KOTS parameters not found")

  hvc_param = find_hvc_params(d, secpar, N, tau, alpha_w, kots_param["q'"],
                        epsilon_hom, k, kots_param["n"], RHF)
  if hvc_param is None:
    raise ValueError("HVC parameters not found")

  return (alpha_w, gamma, kots_param, hvc_param)

def find_params(dk_pairs, secpars, taus, N_list, epsilon_hom_list, RHF):
  params = {}
  best_results = {}   # (secpar, tau, N) -> dict
  to_tabulate = []

  for secpar in secpars:
    for tau in taus:
      for N in N_list:

        best_key = (secpar, tau, N)

        if best_key not in best_results:
          best_results[best_key] = {
            "total_size": float('inf'),
            "params": None
          }

        for d, k in dk_pairs:
          for epsilon_hom in epsilon_hom_list:
              try:
                alpha_w, gamma, kots_param, hvc_param = find_param(
                  d, secpar, N, tau, epsilon_hom, k, RHF
                )
              except (ValueError, KeyError, AssertionError):
                continue

              total_size = compute_total_size_bits(
                alpha_w, gamma, kots_param, hvc_param
              )

              if total_size < best_results[best_key]["total_size"]:
                best_results[best_key]["total_size"] = total_size
                best_results[best_key]["params"] = {
                  "secpar": secpar,
                  "tau": tau,
                  "N": N,
                  "d": d,
                  "epsilon_hom": epsilon_hom,
                  "k": k,
                  "alpha_w": alpha_w,
                  "gamma": gamma,
                  "kots": kots_param,
                  "hvc": hvc_param,
                  "total_size": total_size
                }

  # Build table only from best results
  for key, entry in best_results.items():
    if entry["params"] is None:
      continue

    p = entry["params"]
    kots = p["kots"]
    hvc = p["hvc"]

    to_tabulate.append([
      str(p["secpar"]),
      str(p["tau"]),
      str(p["N"]),
      str(p["d"]),
      str(p["epsilon_hom"]),
      str(p["alpha_w"]),
      str(p["gamma"]),
      str(p["k"]),
      str(kots["n"]),
      str(kots["m"]),
      str(hvc["omega"]),
      str(kots["RHF_LWE_KOTS"]),
      str(kots["RHF_SIS_KOTS"]),
      str(hvc["RHF_SIS_HVC"]),
      str(kots["alpha"]),
      str(kots["alpha_mlwe"]),
      str(kots["alpha_H"]),
      str(kots["beta_z"]),
      str(kots["beta_sigma"]),
      str(hvc["beta_agg"]),
      str(hvc["beta_encode"]),
      str(hvc["eta"]),
      str(hvc["q"]),
      str(hvc["q_bit"]),
      str(hvc["kappa"]),
      str(kots["q'"]),
      str(kots["qprime_bit"]),
      str(hvc["kappa'"]),
      f'{round_half_up(kots["size"] / 8 / 1024)} KB',
      f'{round_half_up(hvc["size"] / 8 / 1024)} KB',
      f'{round_half_up(p["total_size"] / 8 / 1024)} KB'
    ])


  with open("summary.txt", "w") as f:
    f.write(tabulate(
      to_tabulate,
      headers=[
        "secpar", "tau", "N", "d", "epsilon_hom", "alpha_w", "gamma", "k",
        "n", "m", "omega", "RHF_LWE_KOTS", "RHF_SIS_KOTS", "RHF_SIS_HVC",
        "alpha", "alpha_mlwe", "alpha_H", "beta_z", "beta_sigma", "beta_agg", "beta_encode",
        "eta", "q", "q_bit", "kappa", "qprime", "qprime_bit", "kappaprime",
        "sig size", "open size", "total size"
      ],
      tablefmt="grid"
    ))

  return best_results



# security parameter
secpars = [128]
# Constraint 9 and 12 in Lemur Parameter Constraints Table
dk_pairs = [(256, 4)]
# number of users: [1024, 32768, 131072, 1048576]
N_list = [1024, 32768, 131072, 1048576]
# height of the tree: [12, 16, 20, 24]
taus = [12, 16, 20, 24]
# targeted failure probability: [2^(-10),2^(-15),2^(-16)]
epsilon_hom_list = [2^(-15)]  
RHF = 1.0045




params = find_params(
  dk_pairs,
  secpars,
  taus,
  N_list,
  epsilon_hom_list,
  RHF
)


print("Complete")
print()
