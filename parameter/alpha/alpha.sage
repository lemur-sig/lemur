from sage.all import *
from sage.misc.prandom import choice, sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_ring(d, p):
    ZZPR.<x> = PolynomialRing(ZZ)
    RZ = ZZPR.quotient(x^d + 1, 'Xz')

    Zp = Integers(p)
    ZpPR.<y> = PolynomialRing(Zp)
    Rp = ZpPR.quotient(y^d + 1, 'Xp')

    return Zp, RZ, Rp


# ---------------------------------------------------------------------------------------
# Utility Functions for Statistics and Plotting
# ---------------------------------------------------------------------------------------
def stats_analysis(data_list, ddof=1):
    """ Returns (maxv, minv, mean, stdev, mean_plus_12stdev). """
    arr = np.array(data_list, dtype=float)
    maxv = float(np.max(arr))
    minv = float(np.min(arr))
    mean = float(np.mean(arr))
    stdev = float(np.std(arr, ddof=ddof))
    bound = mean + 12.0 * stdev
    return maxv, minv, mean, stdev, bound



def plot_distribution(data_list, title, filename, xlabel):
    """ Helper to plot and save a histogram and density curve with dual y-axes. """
    fig, ax1 = plt.subplots()
    
    # --- LEFT Y-AXIS: Density (Red) ---
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Density", color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    # Calculate and plot the smooth distribution curve (KDE)
    kde = gaussian_kde(data_list)
    x_eval = np.linspace(min(data_list), max(data_list), 200)
    line1 = ax1.plot(x_eval, kde(x_eval), color='red', linewidth=2, label='Density Curve')
    
    # --- RIGHT Y-AXIS: Frequency (Blue) ---
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    ax2.set_ylabel("Frequency", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot the histogram (counts/frequency, so density=False)
    # Using 'skyblue' for bars to stay consistent with the blue right-axis theme
    patches = ax2.hist(data_list, bins=20, density=False, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')
    
    # --- DISPLAY ORDER & LEGENDS ---
    # The second axis (ax2) is drawn over the first (ax1) by default. 
    # We bring ax1 to the front so the red curve isn't hidden behind the bars.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False) # Make the background of ax1 transparent
    
    # Combine the legends from both axes into a single box
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------------------------------------------------------------------------------------
# Core Cryptographic Helper Functions
# ---------------------------------------------------------------------------------------
def sample_poly_ternary_bounded(RZ, d, alpha_H):
    coeffs = [0] * d
    positions = sample(range(d), alpha_H)
    for i in positions:
        coeffs[i] = choice([-1, 1])
    return RZ(coeffs)

def rand_mat_RZ(RZ, rows, cols, d, alpha_H):
    return Matrix(RZ, rows, cols,
                  [sample_poly_ternary_bounded(RZ, d, alpha_H) for _ in range(rows*cols)])

def reduce_RZ_to_Rp(a, d, Zp, Rp):
    return Rp(a.lift())

def reduce_matrix_mod_p(MZ, d, Zp, Rp):
    return Matrix(Rp, MZ.nrows(), MZ.ncols(),
                  [reduce_RZ_to_Rp(MZ[i,j], d, Zp, Rp)
                   for i in range(MZ.nrows())
                   for j in range(MZ.ncols())])

def centered_lift_Rp_to_RZ(aRp, d, p, RZ):
    coeffs = aRp.lift().list()  
    out = [0]*d
    for i in range(d):
        c = int(coeffs[i]) if i < len(coeffs) else 0
        if c > p//2:
            c -= p
        out[i] = c
    return RZ(out)

def centered_lift_mat_Rp_to_RZ(MRp, d, p, RZ):
    return Matrix(RZ, MRp.nrows(), MRp.ncols(),
                  [centered_lift_Rp_to_RZ(MRp[i,j], d, p, RZ)
                   for i in range(MRp.nrows())
                   for j in range(MRp.ncols())])

def coeff_vector_RZ_elem(a, d):
    lst = a.lift().list()
    out = [0]*d
    for i in range(min(d, len(lst))):
        out[i] = ZZ(lst[i])
    return vector(ZZ, out)

def embed_R_vector_to_Z_vector(vR, d):
    flat = []
    for i in range(len(vR)):
        flat.extend(list(coeff_vector_RZ_elem(vR[i], d)))
    return vector(ZZ, flat)

def expand_R_basis_to_Z_basis(RBasis, *, d, RZ):
    rows = RBasis.nrows()
    cols = RBasis.ncols()

    xbar = RZ.gen()
    Xpow = [RZ(1)]
    for t in range(1, d):
        Xpow.append(Xpow[-1] * xbar)

    cols_Z = []
    for j in range(cols):
        b = RBasis.column(j)
        for t in range(d):
            bt = vector(RZ, [b[i] * Xpow[t] for i in range(rows)])
            cols_Z.append(embed_R_vector_to_Z_vector(bt, d))

    return Matrix(ZZ, cols_Z).transpose()

def col_norms_exact(BZ):
    norms_sq = []
    norms = []
    for j in range(BZ.ncols()):
        col = BZ.column(j)
        s = col * col 
        norms_sq.append(s)
        norms.append(col.norm())
    return norms_sq, norms

def spectral_norm_matrix(B):
    return matrix(RDF, B).norm(2)

def eta_vareps(n, vareps, max_basis_norm):
    return RR(sqrt(ln(2.0 * n * (1.0 + 1.0/vareps)) / RR(pi)) * max_basis_norm)


# ---------------------------------------------------------------------------------------
# LAMBDA 12 BASIS AND ALPHA CALCULATIONS
# ---------------------------------------------------------------------------------------
def Lambda12_RBasis(params):
    k, d, alpha_H, p = params["k"], params["d"], params["alpha_H"], params["p"]

    Zp, RZ, Rp = get_ring(d, p)

    H1_1 = rand_mat_RZ(RZ,1,1,d,alpha_H)
    H2_1 = rand_mat_RZ(RZ,1,1,d,alpha_H)
    H1_2 = rand_mat_RZ(RZ,1,k-3,d,alpha_H)
    H2_2 = rand_mat_RZ(RZ,1,k-3,d,alpha_H)

    XZ = H2_1-H1_1
    Xp = reduce_matrix_mod_p(XZ, d, Zp, Rp)

    if not Xp.is_invertible():
        raise ValueError("(H2_1-H1_1) mod p not invertible.")

    Yp = Xp.inverse()
    YZ = centered_lift_mat_Rp_to_RZ(Yp, d, p, RZ)

    M = -YZ*(H2_2-H1_2)

    A = block_matrix([[M],[identity_matrix(RZ,k-3)]])
    B = block_matrix([[identity_matrix(RZ,1)],
                      [zero_matrix(RZ,k-3,1)]])

    return A.augment(p*B)



def L12_max_basis_norm_exp(params, L12_Rbasis):
    """ Returns ONLY the maximum L2 norm of the basis vectors """
    d, p = params["d"], params["p"]
    Zp, RZ, Rp = get_ring(d, p)

    L12_Zbasis = expand_R_basis_to_Z_basis(L12_Rbasis, d=d, RZ=RZ)
    L12_Zbasis_lll = L12_Zbasis.transpose().LLL().transpose()
    _, norms_after = col_norms_exact(L12_Zbasis_lll)

    maxnorm = RR(max(norms_after))
    return float(maxnorm)


# ---------------------------------------------------------------------------------------
# ALPHA 2 & 3 FUNCTIONS
# ---------------------------------------------------------------------------------------
def get_alpha_2(params, trials, raw_filename):
    k, d, alpha_H, p = params["k"], params["d"], params["alpha_H"], params["p"]
    vareps = params["vareps_2"]

    Zp, RZ, Rp = get_ring(d, p)

    alpha0_init_sq = 2 * pi / 3
    constant_term = RR((1 + alpha_H) * ln(2 * (k - 1) * d * (1 + 1 / vareps)) / pi)

    # Trackers
    spec_norm_list = []
    spec_norm_sq_list = []

    for i in range(trials):
        print(f"\r  [Alpha 2] Trial {i+1}/{trials}...", end="")
        Hprime = rand_mat_RZ(RZ, 1, k-1, d, alpha_H)
        R_Basis = block_matrix([[-Hprime],
                                [identity_matrix(RZ, k-1)]])

        Z_Basis = expand_R_basis_to_Z_basis(R_Basis, d=d, RZ=RZ)

        spectral_norm = float(spectral_norm_matrix(Z_Basis))
        spectral_norm_sq = spectral_norm**2

        spec_norm_list.append(spectral_norm)
        spec_norm_sq_list.append(spectral_norm_sq)

    print() # Newline

    # Write Raw Tabular Data (Only Spectral Norms)
    with open(raw_filename, 'w') as f:
        f.write(f"{'Run':<10} {'Spectral_Norm':<20} {'Spectral_Norm_Sq':<20}\n")
        for i in range(trials):
            f.write(f"{i+1:<10} {spec_norm_list[i]:<20.8f} {spec_norm_sq_list[i]:<20.8f}\n")

    # Plots (Only Spectral Norms)
    plot_distribution(spec_norm_list, f"Spectral Norm Z_Basis (d={d}, num_trials={trials})", f"spectral_norm_dist_d{d}.png", "||B||_2")
    plot_distribution(spec_norm_sq_list, f"Spectral Norm Squared (d={d})", f"spectral_norm_sq_dist_d{d}.png", "||B||_2^2")

    # Calculate final bounding value for spectral norm
    _, _, mean, stdev, bound = stats_analysis(spec_norm_list)
    final_spec_norm = bound
    final_spec_norm_sq = final_spec_norm**2

    # Calculate single final Alpha 2
    final_alpha2_sq = float(alpha0_init_sq * final_spec_norm_sq + constant_term)
    final_alpha2 = float(sqrt(final_alpha2_sq))

    return final_alpha2, constant_term, final_spec_norm_sq, spec_norm_list, spec_norm_sq_list


def get_alpha_3(params, trials, raw_filename):
    d = params["d"]
    n = (params["k"] - 3) * d
    vareps = params["vareps_3"]
    
    # Trackers
    max_l2_norm_L12_basis_vector_list = []

    for i in range(trials):
        print(f"\r  [Alpha 3] Trial {i+1}/{trials}...", end="")
        L12_Rbasis = Lambda12_RBasis(params)
        max_l2_norm_L12_basis_vector = L12_max_basis_norm_exp(params, L12_Rbasis)
        
        max_l2_norm_L12_basis_vector_list.append(max_l2_norm_L12_basis_vector)
        
    print() # Newline

    # Write Raw Tabular Data (Only Max L2 Norms)
    with open(raw_filename, 'w') as f:
        f.write(f"{'Run':<10} {'Max_L2_Norm_L12_Basis':<25}\n")
        for i in range(trials):
            f.write(f"{i+1:<10} {max_l2_norm_L12_basis_vector_list[i]:<25.8f}\n")

    # Plots
    plot_distribution(max_l2_norm_L12_basis_vector_list, f"Max L2 Norm of Lambda_12 Basis Vector (d={d})", f"max_l2_norm_L12_dist_d{d}.png", "Max L2 Norm of Lambda_12 Basis Vector")

    # Calculate final bounding value for max L2 norm
    _, _, mean, stdev, bound = stats_analysis(max_l2_norm_L12_basis_vector_list)
    final_max_l2_norm = bound

    # Calculate single final Alpha 3
    final_alpha3 = float(sqrt(log(2*n*(1+1/vareps))/pi) * final_max_l2_norm)
    
    return final_alpha3, max_l2_norm_L12_basis_vector_list


def alpha_3_theory(params):
    k, d, alpha_H, p, vareps = params["k"], params["d"], params["alpha_H"], params["p"], params["vareps_3"]
    n = (k - 3) * d
    max_basis_norm = sqrt(2 * alpha_H)
    alpha_3 = RR(p * eta_vareps(n, vareps, max_basis_norm))
    return alpha_3

# ---------------------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------------------
def get_alpha_and_alpha0(params, num_trials):
    d = params["d"]
    
    # Filenames definitions
    file1_raw_alpha2 = f"raw_alpha2_d={d}.txt"
    file2_raw_alpha3 = f"raw_alpha3_d={d}.txt"
    file3_summary    = f"summary_statistics_d={d}.txt"

    # 1. Lemma 29
    print(f"--- Starting Alpha 1 evaluations ---")
    alpha_1 = float(eta_vareps(1, params["vareps_1"], 1))
    print(f"alpha_1: {alpha_1}")

    # 2. Theorem 1 
    print(f"--- Starting Alpha 2 evaluations ---")
    alpha_2, const_term, final_spec_sq, spec_list, spec_sq_list = get_alpha_2(params, num_trials, file1_raw_alpha2)

    # 3. Lemma 33
    print(f"--- Starting Alpha 3 evaluations ---")

    alpha_3_theoretical = float(alpha_3_theory(params))

    # Final Calculation for Alpha0 (using the empirical alpha_3 bound)
    alpha = max(alpha_1, alpha_2, alpha_3_theoretical)
    alpha0_sq = float((alpha**2 - const_term) / final_spec_sq)
    alpha0 = float(sqrt(alpha0_sq))

    # File 3: Write Summary Statistics
    data_dict = {
        "Spectral Norm": spec_list,
        "Spectral Norm Squared": spec_sq_list,
        #"Max L2 Norm L12 Basis Vector": max_l2_list
    }

    with open(file3_summary, "w") as f:
        f.write(f"SUMMARY STATISTICS (d={d})\n")
        f.write("="*60 + "\n\n")

        # --- NEW BLOCK FOR FINAL VALUES ---
        f.write("--- FINAL DERIVED VALUES ---\n")
        f.write(f"alpha_1 : {alpha_1:.8f}\n")
        f.write(f"alpha_2 : {alpha_2:.8f}\n")
        # f.write(f"alpha_3 (empirical bound)   : {alpha_3:.8f}\n")
        f.write(f"alpha_3 (theoretical bound) : {alpha_3_theoretical:.8f}\n")
        f.write(f"final alpha = max(alpha_1, alpha_2, alpha_3) = max({alpha_1:.8f}, {alpha_2:.8f}, {alpha_3_theoretical:.8f})\n")
        f.write(f"The final alpha = {alpha:.8f}, alpha0 = {alpha0:.8f}\n\n")
        f.write("="*60 + "\n\n")
        # ----------------------------------

        for name, data_list in data_dict.items():
            maxv, minv, mean, stdev, bound = stats_analysis(data_list)
            f.write(f"--- {name} ---\n")
            f.write(f"Max             : {maxv:.8f}\n")
            f.write(f"Min             : {minv:.8f}\n")
            f.write(f"Mean            : {mean:.8f}\n")
            f.write(f"Stdev           : {stdev:.8f}\n")
            f.write(f"Mean + 12*Stdev : {bound:.8f}\n\n")

    print(f"\n{'='*40}")
    print(f"Execution Complete! Files Generated:")
    print(f"1. {file1_raw_alpha2}")
    print(f"2. {file2_raw_alpha3}")
    print(f"3. {file3_summary}")
    print(f"The final alpha = {alpha:.8f}, alpha0 = {alpha0:.8f}")
    print(f"{'='*40}")

    return alpha, alpha0

def epsilon_cor(params):
    # maximum value of m = 20, according to our parameter search
    k, d, alpha_H, p, vareps_1 = params["k"], params["d"], params["alpha_H"], params["p"], params["vareps_1"]
    return 20 * d * (2**(-162) + 2 * (k - 1) * alpha_H * vareps_1) <= 2**(-128)
    

def get_vareps_3(params):
    k, d, alpha_H, p, vareps_3 = params["k"], params["d"], params["alpha_H"], params["p"], params["vareps_3"]
    m = 20
    t1 = 2
    return ((1+vareps_3)/(1-vareps_3))**(2*m) * p**(-d * m / t1) <= 2**(-128)



    


if __name__=="__main__":

    params = dict(k=4, d=256, alpha_H=60, p=5, vareps_1=2**(-150), vareps_2=2**(-136), vareps_3=2**(-0.5))
    num_trials = 1000

    alpha, alpha0 = get_alpha_and_alpha0(params, num_trials)
    
    print(epsilon_cor(params))
    print(get_vareps_3(params))
