# All constraints referenced within this script refer to the numbered constraints found in Table 3 of the Chipmunk paper.

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
from sage.all import oo, ceil, log, ZZ, binomial, sqrt, next_prime, pi, e, false, true

import sys
from pathlib import Path
current_script_dir = Path(sys.argv[0]).resolve().parent
sys.path.append(str(current_script_dir.resolve()))
from estimator import SIS




def estimate_rsis_rop(n, q, m_ring, inf_bound):
    """
    Helper function to run the lattice estimator on a Ring-SIS instance.
    Returns the base-2 logarithm of the minimum required operations (rop).
    """
    m_sis = n * m_ring
    sis_param = SIS.Parameters(n=n, q=q, length_bound=inf_bound, norm=oo, m=m_sis)
    
    try:
        res = SIS.estimate.rough(sis_param)
        # Find the minimum rop from all available attack estimates
        min_rop = min([float(log(algo['rop'], 2)) for algo in res.values() if 'rop' in algo])
        return round(min_rop, 1)
    except Exception as err:
        return f"Err: {err}"

def encoded_elements_size(n, q, eta, beta_agg, number_of_ring_element):
  """ compute the encoded size of a dim-n ring element whose norm is bounded by beta_agg
  """
  # size of the hint
  # hint_size = ceil(log(q, 2)) * n

  # size of alpha_star, alpha_1, alpha_2, ..., alpha_{number_of_ring_element-1}
  beta_encoded = ceil(beta_agg/2/eta + 1/2)
  alpha_sizes = number_of_ring_element * ceil(log(beta_encoded, 2) + 1) * n

  return alpha_sizes

def cardinality_of_set_of_ternary_poly(n,alpha):
  """ Determines the size of the set of ternary polynomials of degree n and Hamming weight alpha.
  """
  return binomial(n,alpha)*2^alpha

def find_hamming_weight(n,l):
  """ Finds the minimal value alpha, such that the set of ternary polynomials of degree n and Hamming weight alpha has size at least l.
  """
  for alpha in range(n+1):
    if cardinality_of_set_of_ternary_poly(n,alpha) >= l:
      return alpha
  raise ValueError("There does not exist a Hamming weight satisfying the specified conditions.")

def get_gamma(secpar,delta,n,q,phi):
  """ Determines the minimal value gamma, such that constraint 6 is satisfied.
  """
  return ZZ(ceil((((3*secpar+delta)/n)+log(q,2))/log(phi+.5,2)))

def get_alpha_w(secpar,n):
  """ Determines the minimal value alpha_w, such that constraint 11 is satisfied.
  """
  return find_hamming_weight(n,2^secpar)

def get_alpha_H_and_delta(secpar,n):
  """ Determines the minimal values alpha_H and delta, such that constraints 5 and 7 are satisfied.
  """
  alpha_H = find_hamming_weight(n, 2^(2*secpar))
  delta = ZZ(ceil(log(cardinality_of_set_of_ternary_poly(n,alpha_H),2))-2*secpar)
  return (alpha_H,delta)

def get_beta_sigma(n,alpha_H,alpha_w,rho,phi,gamma,epsilon):
  """ Determines the minimal value beta_sigma, such that constraint 4 is satisfied.
  """
  return ZZ(ceil(4*phi*alpha_H*sqrt(.5*alpha_w*rho*log(2*n*gamma/epsilon))))

def get_beta_kots(alpha_w,alpha_H,phi,beta_sigma):
  """ Calculates the norm bound of the SIS instance corresponding to the given KOTS parameters.
  """
  return ZZ(2*beta_sigma+4*alpha_w*alpha_H*phi)

def get_beta_agg(n,tau,xi,eta,kappa,kappaprime,alpha_w,rho,epsilon):
  """ Determines the minimal value beta_agg, such that constraint 1 is satisfied.
  """
  return ZZ(ceil(eta*sqrt(2*alpha_w*rho*(log(2*n/epsilon)+log(2*tau*kappa+xi*kappaprime)))))

def find_ntt_friendly_prime(n,beta):
  """ Finds the smallest prime q > beta, such that Z[x]/(x^n + 1) is NTT friendly.
  """
  q = next_prime(beta)
  # NTT friendliness requires that q ≡ 1 (mod 4*n)
  while q % (4*n) != 1:
    q = next_prime(q)
  return q

def rsisIsHard(beta, q, n, m, c):
  """ Determines whether the Ring-SIS instance described by the inputs is hard.
  """
  # check that SIS problem is not trivial
  if beta >= q/2:
    return (false, "SIS is trivial")

  # find the best m for the attacker
  # if there are more m than required, the attacker can ignore some of them
  m = min(m, ceil(sqrt(n * log(q, 2) * log(c, 2))))

  # check that it is not possible to find short solution, using infinity norm
  if 2 * beta >= c^(n * m) * q^(1/m) - 1:
    return (false, "infinity norm fails")

  # check that it is not possible to find short solution, using l2 norm
  if sqrt(n * m) * beta >= c^(n * m) * sqrt( n * m / 2 / pi / e) * q^(1/m):
    return (false, "l2 norm fails")

  return (true, "pass")

def get_root_hermite_factor(secpar):
  """ Uses handwaving to translates the security parameter into a root Hermite factor.
  """
  if secpar == 128:
    return 1.004
  else:
    if secpar == 112:
      return 1.005
    else:
      raise ValueError("Input security parameter should be either 112 or 128")

def find_kots_params(n, secpar, rho, alpha_w, epsilon, verbose):
  """ Finds parameters for the key homomorphic one-time signature scheme compatible with the inputs.
  """
  c = get_root_hermite_factor(secpar)
  (alpha_H,delta) = get_alpha_H_and_delta(secpar,n)

  params = {}
  min_params = 0
  min_size = oo

  sis_is_hard = True
  phi = 1
  while sis_is_hard:
    guessed_gamma = 1
    gamma_too_big = True
    while gamma_too_big:
      guessed_gamma +=1
      beta_sigma = get_beta_sigma(n,alpha_H,alpha_w,rho,phi,guessed_gamma,epsilon)
      beta_kots = get_beta_kots(alpha_w,alpha_H,phi,beta_sigma)
      q = find_ntt_friendly_prime(n,max(2*beta_kots,16*alpha_w*alpha_H*phi))
      gamma = get_gamma(secpar,delta,n,q,phi)
      gamma_too_big = bool(guessed_gamma < gamma)

    (sis_is_hard, sis_check_msg) = rsisIsHard(beta_kots, q, n, gamma, c)
    if sis_is_hard:
      size = (gamma * n) * ceil(log(2 * beta_sigma + 1,2))
      if size < min_size:
        min_params = phi
        min_size = size
        params[phi] = {"alpha_H" : alpha_H, "delta" : delta, "phi" : phi, "gamma" : gamma, "beta_sigma" : beta_sigma, "q'" : q, "size" : size}
    phi += 1

  to_tabulate = []
  if verbose:
    for p in params.values():
      to_tabulate.append([str(p["alpha_H"]),str(p["delta"]), str(p["phi"]),str(p["gamma"]),str(p["beta_sigma"]),str(p["q'"]),str(ZZ(ceil(p["size"]/8/1024))) + " KB"])
    print(tabulate(to_tabulate,headers=["alpha_H", "delta", "phi","gamma","beta_sigma","q'","signature size"],tablefmt="simple_outline"),"\n")
  return params[min_params]

def find_hvc_params(n, secpar, rho, tau, alpha_w, xi, qprime, epsilon, verbose):
  """ Finds parameters for the homomorphic vector commitment compatible with the inputs.
  """
  c = get_root_hermite_factor(secpar)

  hvc_params = {}
  hvc_min_params = 0
  hvc_min_size = oo

  eta = 2
  sis_is_hard = True
  while sis_is_hard:
    kappaprime = ceil(log(qprime,2*eta+1))
    guessed_kappa = 0
    kappa_too_big = True
    while kappa_too_big:
      guessed_kappa+=1
      beta_agg = get_beta_agg(n,tau,xi,eta,guessed_kappa,kappaprime,alpha_w,rho,epsilon)
      beta_hvc = 4 * beta_agg
      q = find_ntt_friendly_prime(n,2*beta_hvc)
      kappa = ceil(log(q,2*eta+1))
      kappa_too_big = bool(guessed_kappa < kappa)

    (path_sis_is_hard, path_error_msg) = rsisIsHard(beta_hvc, q, n, 2 * kappa, c)
    (payload_sis_is_hard, payload_error_msg) = rsisIsHard(beta_hvc, q, n, xi*kappaprime, c)
    sis_is_hard = (path_sis_is_hard and payload_sis_is_hard)
    if sis_is_hard:
      size = encoded_elements_size(n, q, eta, beta_agg, tau*kappa)
      size += n * (ceil(log(beta_agg, 2)) + 1) * tau * kappa
      size += n * (ceil(log(beta_agg, 2)) + 1) * xi * kappaprime
      if size < hvc_min_size:
        hvc_min_params = eta
        hvc_min_size = size
        hvc_params[eta] = {"eta" : eta, "kappa" : kappa, "kappa'" : kappaprime ,"beta_agg" : beta_agg, "q" : q, "SIS beta" : beta_hvc, "SIS width" : 2*kappa, "size" : size, "epsilon" : epsilon}
    eta+=1

  if verbose:
    to_tabulate = []
    for p in hvc_params.values():
      to_tabulate.append([str(p["eta"]),str(p["kappa"]),str(p["kappa'"]),str(p["beta_agg"]),str(p["q"]),str(p["SIS beta"]),str(p["SIS width"]),str(ZZ(ceil(p["size"]/8/1024))) + " KB",p["epsilon"]])
    print(tabulate(to_tabulate,headers=["eta","kappa","kappa'","beta_agg","q","beta_hvc","SIS width","opening size","epsilon"],tablefmt="simple_outline"),"\n")
  return hvc_params[hvc_min_params]

def find_param(n, secpar, rho, tau, epsilon, verbose):
  """ Finds parameters for the Chipmunk multi-signature scheme compatible with the inputs.
  """
  print("Finding params for secpar = " + str(secpar) + " tau = " + str(tau) + " rho = " + str(rho) + ", and epsilon=" + str(epsilon))

  alpha_w = get_alpha_w(secpar,n)
  chi = ZZ(ceil(secpar/log(1/(2*epsilon))))

  kots_param = find_kots_params(n, secpar, rho, alpha_w, epsilon, verbose)
  hvc_param = find_hvc_params(n, secpar, rho, tau, alpha_w, 2, kots_param["q'"], epsilon, verbose)
  return (alpha_w,chi,kots_param,hvc_param)

def find_params(n,secpars,taus,rhos,epsilons,verbosity):
  """ Returns a dictionary containing valid parameter sets for all combinations of input constraints.
  """
  params = {}
  to_tabulate = []
  for secpar in secpars:
    params[secpar] = {}
    for tau in taus:
      params[secpar][tau] = {}
      for rho in rhos:
        params[secpar][tau][rho] = {}
        for epsilon in epsilons:
          params[secpar][tau][rho][epsilon] = find_param(n, secpar, rho, tau, epsilon, bool(verbosity>1))
          
          # Extract variables for RSIS estimator
          alpha_w = params[secpar][tau][rho][epsilon][0]
          chi = params[secpar][tau][rho][epsilon][1]
          kots_p = params[secpar][tau][rho][epsilon][2]
          hvc_p = params[secpar][tau][rho][epsilon][3]
          
          q_prime = kots_p["q'"]
          gamma = kots_p["gamma"]
          alpha_H = kots_p["alpha_H"]
          phi = kots_p["phi"]
          beta_sigma = kots_p["beta_sigma"]
          
          q = hvc_p["q"]
          eta = hvc_p["eta"]
          beta_agg = hvc_p["beta_agg"]
          
          # (1) RSIS_{q, 2\lceil log_{2eta+1} q \rceil, 4beta_agg}
          m_ring_1 = 2 * ceil(log(q, 2*eta + 1))
          bound_1 = 4 * beta_agg
          rop_1 = estimate_rsis_rop(n, q, m_ring_1, bound_1)
          
          # (2) RSIS_{q, 2\lceil log_{2eta+1} q' \rceil, 4beta_agg}
          m_ring_2 = 2 * ceil(log(q_prime, 2*eta + 1))
          bound_2 = 4 * beta_agg
          rop_2 = estimate_rsis_rop(n, q, m_ring_2, bound_2)
          
          # (3) RSIS_{q', gamma, 2beta_sigma + 4alpha_w * alpha_H * phi}
          m_ring_3 = gamma
          bound_3 = 2 * beta_sigma + 4 * alpha_w * alpha_H * phi
          rop_3 = estimate_rsis_rop(n, q_prime, m_ring_3, bound_3)
          
          to_tabulate.append([
            str(secpar),
            str(tau),
            str(rho),
            str(epsilon),
            str(alpha_w),
            str(chi),
            str(alpha_H),
            str(phi),
            str(gamma),
            str(beta_sigma),
            str(q_prime),
            str(eta),
            str(beta_agg),
            str(q),
            str(round((kots_p["size"]+hvc_p["size"]+ceil(log(chi,2)))/8/1024))+" KB",
            str(rop_1),  # Added RSIS 1
            str(rop_2),  # Added RSIS 2
            str(rop_3)   # Added RSIS 3
          ])
  if verbosity > 0:
    with open("chipmunk_original_security_summary.txt", "w") as f:
      table_string = tabulate(to_tabulate,headers=[
        "secpar",
        "tau",
        "rho",
        "epsilon",
        "alpha_w",
        "chi",
        "alpha_H",
        "phi",
        "gamma",
        "beta_sigma",
        "q'",
        "eta",
        "beta_agg",
        "q",
        "size",
        "RSIS1 rop",
        "RSIS2 rop",
        "RSIS3 rop"
      ],tablefmt="simple_outline")
      f.write(table_string)
  return params

# security parameter
secpars = [112, 128]
# polynomial degree
n = 512
# number of users
rhos = [1024, 8192, 131072]
# height of the tree
taus = [21, 23, 24, 26]
# targeted failure probability
#epsilons = [2^(-10),2^(-15),2^(-16)]
epsilons = [2^(-15)]

verbosity = 1

params = find_params(n,secpars,taus,rhos,epsilons,verbosity)
