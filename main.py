import math
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize_scalar
from scipy.stats import multinomial, binom, norm, beta


def lower_edge(n, k, alpha):

    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
        
    res = 1e-5
    m = np.arange(res, k-1-res, res)  
    t = m / (m + n - k)
    
    binom_pdf = binom.pmf(k, n, t) 
    obj = (alpha / (binom_pdf * t**(m - k)))**(1 / (k - 1 - m))
    
    a = np.max(obj)
    m_opt = m[np.argmax(obj)]
    
    return a, m_opt


def upper_edge(n, k, alpha):
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
    
    res = 1e-3
    r = np.arange(1, 20 + res, res)  
    t = (r + k - 1) / (r + n - 1)
    
    obj = (binom.pmf(k, n, t) * t**(r - 1) / alpha)**(1 / r)
    
    b = np.min(obj)
    r_opt = r[np.argmin(obj)]
    
    return b, r_opt


def CI_for_non_frequent_symbols(n, k_max, alpha):

    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in (0, 1), bounds excluded')
        
    k_vec = np.arange(0, k_max + 1) 
    CI_marginals = np.zeros((len(k_vec), 2))
    r_vec = np.zeros(len(k_vec))
    m_vec = np.zeros(len(k_vec))

    for k_ind, k in enumerate(k_vec):
        if k >= 2:
            b, r_opt = upper_edge(n, k, alpha / 2)
            a, m_opt = lower_edge(n, k, alpha / 2)
        else:
            b, r_opt = upper_edge(n, k, alpha)
            a = 0
            m_opt = -3

        CI_marginals[k_ind, 0] = a
        CI_marginals[k_ind, 1] = b
        r_vec[k_ind] = r_opt
        m_vec[k_ind] = m_opt

    # Resolution for c and t
    c_res = 1e-3
    c_vec = np.arange(1, 10 + c_res, c_res)
    t_res = 1e-6
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
        if max_obj < alpha:
            done = True
        else:
            ind += 1
            if ind >= len(c_vec):
                raise RuntimeError("Convergence not achieved within range of c_vec.")

        print(f"k_max: {k_max}, ind: {ind}, max_obj: {max_obj}")

    c_opt = c_vec[ind]
    CI = current_CI_marginals

    return CI

def binofit(count, n, alpha):
    """
    Calculate binomial confidence intervals using the Clopper-Pearson method.

    Parameters
    ----------
    count : Number of observations in category.
    alpha : float in (0, 1), optional
        Significance level, defaults to 0.05.
    n : total observations
    """
    lower_bound = beta.ppf(alpha / 2, count, n - count + 1)
    upper_bound = beta.ppf(1 - alpha / 2, count + 1, n - count)
    return np.clip(lower_bound, 0, 1), np.clip(upper_bound, 0, 1)

def compute_pci_mine(counts, alpha, k_max):
    """
    Confidence intervals for multinomial proportions.

    Parameters
    ----------
    counts : array_like of int, 1-D
        Number of observations in each category.
    alpha : float in (0, 1), optional
        Significance level, defaults to 0.05.

    Returns
    -------
    confint : ndarray, 2-D
        Array of [lower, upper] confidence levels for each category, such that
        overall coverage is (approximately) `1-alpha`.
    """
    n = counts.sum()
    ab_size = len(counts)

    CI = CI_for_non_frequent_symbols(n, k_max, alpha * (1 - n / (ab_size * (k_max + 1)))) 

    # Compute naive binomial confidence intervals
    pci_naive = np.array([binofit(x, n, alpha / ab_size) for x in counts])
    pci_naive = np.stack(pci_naive)
    
    # Modify the naive confidence intervals based on the counts
    pci_mine = pci_naive.copy()
    for ind in range(k_max + 1):
        mask = counts == ind  
        pci_mine[mask, :] = np.tile(CI[ind, :], (np.sum(mask), 1)) 
    
    return pci_mine

def expected_max_E_ri(n, i):
    def bound_function(t):
        if t <= 0 or t >= 1:
            return -float('inf')
        return comb(n, i) * (t**(i - 1)) * ((1 - t)**(n - i))

    result = minimize_scalar(lambda t: -bound_function(t), bounds=(0, 1), method='bounded')
    return -result.fun

def expected_log_S_with_max_E_ri(counts, alpha, k_max):
    """
    Compute the expected log term with max expectations of E_ri.
    """
    n = counts.sum()
    ab_size = len(counts)

    CI = CI_for_non_frequent_symbols(n, k_max, alpha * (1 - n / (ab_size * (k_max + 1))))
    differences = CI[:, 1] - CI[:, 0]

    expectations = [expected_max_E_ri(n, i) for i in range(k_max + 1)]

    Z_value = norm.ppf(1 - alpha / ab_size)
    log_Z_term = math.log(Z_value * math.sqrt(1 / n))

    summation_term = np.sum(expectations * np.log(differences))
    remaining_term = log_Z_term * (n - np.sum(expectations))

    return summation_term + remaining_term
