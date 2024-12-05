import math
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import multinomial, binom, norm
import seaborn as sns
sns.set(style="whitegrid", palette="muted")


def lower_edge(n, k, alph):
    res = 10**-5
    m = np.arange(res, k-1-res, res)  
    t = m / (m + n - k)
    
    binom_pdf = binom.pmf(k, n, t) 
    obj = (alph / (binom_pdf * t**(m - k)))**(1 / (k - 1 - m))
    
    a = np.max(obj)
    ind = np.argmax(obj)
    m_opt = m[ind]
    
    return a, m_opt


def upper_edge(n, k, alph):
    res = 10**-3
    r = np.arange(1, 20 + res, res)  
    t = (r + k - 1) / (r + n - 1)
    
    obj = (binom.pmf(k, n, t) * t**(r - 1) / alph)**(1 / r)
    
    b = np.min(obj)
    ind = np.argmin(obj)
    r_opt = r[ind]
    
    return b, r_opt


def CI_for_non_frequent_symbols(n, k_max, alph):
    k_min = 0
    k_vec = np.arange(k_min, k_max + 1) 
    CI_marginals = np.zeros((len(k_vec), 2))
    r_vec = np.zeros(len(k_vec))
    m_vec = np.zeros(len(k_vec))

    for k_ind, k in enumerate(k_vec):
        if k >= 2:
            b, r_opt = upper_edge(n, k, alph / 2)
            a, m_opt = lower_edge(n, k, alph / 2)
        else:
            b, r_opt = upper_edge(n, k, alph)
            a = 0
            m_opt = -3

        CI_marginals[k_ind, 0] = a
        CI_marginals[k_ind, 1] = b
        r_vec[k_ind] = r_opt
        m_vec[k_ind] = m_opt

    # Resolution for c and t
    c_res = 10**-3
    c_vec = np.arange(1, 10 + c_res, c_res)
    t_res = 10**-6
    t = np.arange(t_res, 1 - t_res, t_res)

    ind = 0
    done = False

    while not done:
        c = c_vec[ind]
        current_CI_marginals = np.zeros_like(CI_marginals)
        mean_CI = CI_marginals.mean(axis=1)
        diff_CI = 0.5 * (CI_marginals[:, 1] - CI_marginals[:, 0])

        current_CI_marginals[:, 0] = np.maximum(mean_CI - c * diff_CI, 0)
        current_CI_marginals[:, 1] = np.maximum(mean_CI + c * diff_CI, 0)

        obj = np.zeros_like(t)
        for k_ind, k in enumerate(k_vec):
            term1 = (current_CI_marginals[k_ind, 1]**-r_vec[k_ind]) * t**(r_vec[k_ind] - 1)
            term2 = (current_CI_marginals[k_ind, 0]**(-m_vec[k_ind] + k - 1)) * t**(m_vec[k_ind] - k)
            obj += binom.pmf(k, n, t) * (term1 + term2)

        max_obj = np.max(obj)
        if max_obj < alph:
            done = True
        else:
            ind += 1
            if ind >= len(c_vec):
                raise RuntimeError("Convergence not achieved within range of c_vec.")

        print(f"k_max: {k_max}, ind: {ind}, max_obj: {max_obj}")

    c_opt = c_vec[ind]
    CI = current_CI_marginals

    return CI


def expected_max_E_ri(n, i):
    def bound_function(t):
        if t <= 0 or t >= 1:
            return -float('inf')
        return comb(n, i) * (t**(i - 1)) * ((1 - t)**(n - i))

    result = minimize_scalar(lambda t: -bound_function(t), bounds=(0, 1), method='bounded')
    return -result.fun

def expected_log_S_with_max_E_ri(n, alpha, k_max, differences, m):
    expectations = [expected_max_E_ri(n, i) for i in range(k_max + 1)]
    
    Z_value = norm.ppf(1 - alpha / m)
    log_Z_term = math.log(Z_value * math.sqrt(1 / n))
    
    summation_term = sum(expectations[i] * math.log(differences[i]) for i in range(k_max + 1))
    remaining_term = log_Z_term * (n - sum(expectations))
    
    return summation_term + remaining_term
