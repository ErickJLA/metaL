import numpy as np

def calculate_hedges_g(xe, sde, ne, xc, sdc, nc):
    """
    Calculate Hedges' g and variance from summary statistics.
    Uses exact Gamma correction.
    """
    from scipy.special import gamma
    
    # Pooled SD
    df = ne + nc - 2
    sd_pooled = np.sqrt(((ne - 1)*sde**2 + (nc - 1)*sdc**2) / df)
    
    # Cohen's d
    d = (xe - xc) / sd_pooled
    
    # Correction Factor (J)
    J = gamma(df / 2) / (np.sqrt(df / 2) * gamma((df - 1) / 2))
    
    g = d * J
    
    # Variance
    var_g = ((ne + nc) / (ne * nc) + (g**2 / (2 * (ne + nc)))) * J**2
    
    return g, var_g

def calculate_lnrr(xe, sde, ne, xc, sdc, nc):
    """Calculate Log Response Ratio and variance."""
    lnrr = np.log(xe / xc)
    var = (sde**2 / (ne * xe**2)) + (sdc**2 / (nc * xc**2))
    return lnrr, var

def estimate_tau_dl(y, v):
    """DerSimonian-Laird Estimator for Tau-Squared."""
    k = len(y)
    w = 1 / v
    w_sum = np.sum(w)
    mu_hat = np.sum(w * y) / w_sum
    Q = np.sum(w * (y - mu_hat)**2)
    C = w_sum - np.sum(w**2) / w_sum
    
    tau2 = max(0, (Q - (k - 1)) / C)
    return tau2