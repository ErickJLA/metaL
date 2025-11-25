"""
Effect Size and Variance Estimators for Meta-Analysis.

This module provides pure functions for calculating effect sizes and estimating
between-study heterogeneity (tau-squared) using various methods.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.special import gamma
from scipy.optimize import minimize_scalar


# =============================================================================
# EFFECT SIZE CALCULATORS
# =============================================================================

def calculate_hedges_g(xe, sde, ne, xc, sdc, nc):
    """
    Calculate Hedges' g and variance from summary statistics.

    Uses exact Gamma correction factor to match R's metafor package.

    Parameters
    ----------
    xe : float or array-like
        Mean of experimental group
    sde : float or array-like
        Standard deviation of experimental group
    ne : int or array-like
        Sample size of experimental group
    xc : float or array-like
        Mean of control group
    sdc : float or array-like
        Standard deviation of control group
    nc : int or array-like
        Sample size of control group

    Returns
    -------
    g : float or ndarray
        Hedges' g (corrected standardized mean difference)
    var_g : float or ndarray
        Variance of Hedges' g

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size
    and related estimators. Journal of Educational Statistics, 6(2), 107-128.

    Examples
    --------
    >>> g, var_g = calculate_hedges_g(10.5, 2.3, 50, 8.2, 2.1, 48)
    >>> print(f"Hedges' g = {g:.3f}, SE = {np.sqrt(var_g):.3f}")
    """
    # Convert to numpy arrays for vectorized operations
    xe, sde, ne = np.asarray(xe), np.asarray(sde), np.asarray(ne)
    xc, sdc, nc = np.asarray(xc), np.asarray(sdc), np.asarray(nc)

    # Degrees of freedom
    df = ne + nc - 2

    # Pooled standard deviation
    sd_pooled = np.sqrt(((ne - 1) * sde**2 + (nc - 1) * sdc**2) / df)

    # Cohen's d (uncorrected)
    d = (xe - xc) / sd_pooled

    # Hedges' correction factor J (EXACT formula using Gamma function)
    # J = Γ(df/2) / (√(df/2) × Γ((df-1)/2))
    J = gamma(df / 2) / (np.sqrt(df / 2) * gamma((df - 1) / 2))

    # Apply correction
    g = d * J

    # Variance of Hedges' g (exact formula)
    var_g = ((ne + nc) / (ne * nc) + (g**2 / (2 * (ne + nc)))) * J**2

    return g, var_g


def calculate_lnrr(xe, sde, ne, xc, sdc, nc):
    """
    Calculate log response ratio (lnRR) and variance.

    Commonly used in ecology and biology for ratio/fold-change data.

    Parameters
    ----------
    xe : float or array-like
        Mean of experimental group
    sde : float or array-like
        Standard deviation of experimental group
    ne : int or array-like
        Sample size of experimental group
    xc : float or array-like
        Mean of control group
    sdc : float or array-like
        Standard deviation of control group
    nc : int or array-like
        Sample size of control group

    Returns
    -------
    lnrr : float or ndarray
        Log response ratio
    var_lnrr : float or ndarray
        Variance of log response ratio

    Notes
    -----
    Requires positive means (xe > 0, xc > 0). Returns NaN for non-positive values.

    References
    ----------
    Hedges, L. V., Gurevitch, J., & Curtis, P. S. (1999). The meta-analysis of
    response ratios in experimental ecology. Ecology, 80(4), 1150-1156.

    Examples
    --------
    >>> lnrr, var = calculate_lnrr(12.5, 3.2, 40, 10.0, 2.8, 38)
    >>> print(f"lnRR = {lnrr:.3f}, SE = {np.sqrt(var):.3f}")
    """
    xe, sde, ne = np.asarray(xe), np.asarray(sde), np.asarray(ne)
    xc, sdc, nc = np.asarray(xc), np.asarray(sdc), np.asarray(nc)

    # Check for positive means
    if np.any(xe <= 0) or np.any(xc <= 0):
        warnings.warn("lnRR requires positive means. Non-positive values will result in NaN.")

    # Log response ratio
    lnrr = np.log(xe / xc)

    # Variance (using delta method approximation)
    var_lnrr = (sde**2 / (ne * xe**2)) + (sdc**2 / (nc * xc**2))

    return lnrr, var_lnrr


# =============================================================================
# VARIANCE ESTIMATORS (TAU-SQUARED)
# =============================================================================

def calculate_tau_squared_DL(df, effect_col, var_col):
    """
    DerSimonian-Laird estimator for tau-squared.

    Advantages:
    - Simple, fast
    - Non-iterative
    - Always converges

    Disadvantages:
    - Can underestimate tau² in small samples
    - Negative values truncated to 0
    - Less efficient than ML methods

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column

    Returns
    -------
    tau_squared : float
        DerSimonian-Laird estimate of between-study variance

    References
    ----------
    DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.
    """
    k = len(df)
    if k < 2:
        return 0.0

    try:
        # Fixed-effects weights
        w = 1 / df[var_col]
        sum_w = w.sum()

        if sum_w <= 0:
            return 0.0

        # Fixed-effects pooled estimate
        pooled_effect = (w * df[effect_col]).sum() / sum_w

        # Q statistic
        Q = (w * (df[effect_col] - pooled_effect)**2).sum()
        df_Q = k - 1

        # C constant
        sum_w_sq = (w**2).sum()
        C = sum_w - (sum_w_sq / sum_w)

        # Tau-squared
        if C > 0 and Q > df_Q:
            tau_sq = (Q - df_Q) / C
        else:
            tau_sq = 0.0

        return max(0.0, tau_sq)

    except Exception as e:
        warnings.warn(f"Error in DL estimator: {e}")
        return 0.0


def calculate_tau_squared_REML(df, effect_col, var_col, max_iter=100, tol=1e-8):
    """
    REML estimator for tau-squared (RECOMMENDED - Gold Standard).

    Advantages:
    - Unbiased for tau²
    - Accounts for uncertainty in estimating mu
    - Better performance in small samples
    - Generally preferred in literature

    Disadvantages:
    - Iterative (slightly slower)
    - Can fail to converge in extreme cases

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column
    max_iter : int, default=100
        Maximum iterations for optimization
    tol : float, default=1e-8
        Convergence tolerance

    Returns
    -------
    tau_squared : float
        REML estimate of between-study variance

    References
    ----------
    Viechtbauer, W. (2005). Bias and efficiency of meta-analytic variance
    estimators in the random-effects model. Journal of Educational and
    Behavioral Statistics, 30(3), 261-293.
    """
    k = len(df)
    if k < 2:
        return 0.0

    try:
        # Extract data
        yi = df[effect_col].values
        vi = df[var_col].values

        # Remove any infinite or negative variances
        valid_mask = np.isfinite(vi) & (vi > 0)
        if not valid_mask.all():
            warnings.warn(f"Removed {(~valid_mask).sum()} observations with invalid variances")
            yi = yi[valid_mask]
            vi = vi[valid_mask]
            k = len(yi)

        if k < 2:
            return 0.0

        # REML objective function (negative log-likelihood)
        def reml_objective(tau2):
            tau2 = max(0, tau2)
            wi = 1 / (vi + tau2)
            sum_wi = wi.sum()

            if sum_wi <= 0:
                return 1e10

            mu = (wi * yi).sum() / sum_wi
            Q = (wi * (yi - mu)**2).sum()

            # REML log-likelihood (negative for minimization)
            # L = -0.5 * [sum(log(vi + tau2)) + log(sum(wi)) + Q]
            log_lik = -0.5 * (
                np.sum(np.log(vi + tau2)) +
                np.log(sum_wi) +
                Q
            )

            return -log_lik

        # Get reasonable bounds for tau2
        var_yi = np.var(yi, ddof=1) if k > 2 else 1.0
        upper_bound = max(10 * var_yi, 100)

        # Optimize
        result = minimize_scalar(
            reml_objective,
            bounds=(0, upper_bound),
            method='bounded',
            options={'maxiter': max_iter, 'xatol': tol}
        )

        if result.success:
            tau_sq = result.x
        else:
            warnings.warn("REML optimization did not converge, using DL fallback")
            tau_sq = calculate_tau_squared_DL(df, effect_col, var_col)

        return max(0.0, tau_sq)

    except Exception as e:
        warnings.warn(f"Error in REML estimator: {e}, using DL fallback")
        return calculate_tau_squared_DL(df, effect_col, var_col)


def calculate_tau_squared_ML(df, effect_col, var_col, max_iter=100, tol=1e-8):
    """
    Maximum Likelihood estimator for tau-squared.

    Advantages:
    - Efficient asymptotically
    - Produces valid estimates

    Disadvantages:
    - Biased downward (underestimates tau²)
    - Less preferred than REML

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-8
        Convergence tolerance

    Returns
    -------
    tau_squared : float
        ML estimate of between-study variance
    """
    k = len(df)
    if k < 2:
        return 0.0

    try:
        yi = df[effect_col].values
        vi = df[var_col].values

        valid_mask = np.isfinite(vi) & (vi > 0)
        if not valid_mask.all():
            yi = yi[valid_mask]
            vi = vi[valid_mask]
            k = len(yi)

        if k < 2:
            return 0.0

        # ML objective function
        def ml_objective(tau2):
            tau2 = max(0, tau2)
            wi = 1 / (vi + tau2)
            sum_wi = wi.sum()

            if sum_wi <= 0:
                return 1e10

            mu = (wi * yi).sum() / sum_wi
            Q = (wi * (yi - mu)**2).sum()

            # ML log-likelihood (without the constant term)
            log_lik = -0.5 * (np.sum(np.log(vi + tau2)) + Q)

            return -log_lik

        var_yi = np.var(yi, ddof=1) if k > 2 else 1.0
        upper_bound = max(10 * var_yi, 100)

        result = minimize_scalar(
            ml_objective,
            bounds=(0, upper_bound),
            method='bounded',
            options={'maxiter': max_iter, 'xatol': tol}
        )

        if result.success:
            tau_sq = result.x
        else:
            warnings.warn("ML optimization did not converge, using DL fallback")
            tau_sq = calculate_tau_squared_DL(df, effect_col, var_col)

        return max(0.0, tau_sq)

    except Exception as e:
        warnings.warn(f"Error in ML estimator: {e}, using DL fallback")
        return calculate_tau_squared_DL(df, effect_col, var_col)


def calculate_tau_squared_PM(df, effect_col, var_col, max_iter=100, tol=1e-8):
    """
    Paule-Mandel estimator for tau-squared.

    Advantages:
    - Exact solution to Q = k-1 equation
    - Non-iterative in principle
    - Good performance

    Disadvantages:
    - Can be unstable with few studies
    - Requires iterative solution in practice

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-8
        Convergence tolerance

    Returns
    -------
    tau_squared : float
        Paule-Mandel estimate of between-study variance

    References
    ----------
    Paule, R. C., & Mandel, J. (1982). Consensus values and weighting factors.
    Journal of Research of the National Bureau of Standards, 87(5), 377-385.
    """
    k = len(df)
    if k < 2:
        return 0.0

    try:
        yi = df[effect_col].values
        vi = df[var_col].values

        valid_mask = np.isfinite(vi) & (vi > 0)
        if not valid_mask.all():
            yi = yi[valid_mask]
            vi = vi[valid_mask]
            k = len(yi)

        if k < 2:
            return 0.0

        df_Q = k - 1

        # PM objective: Find tau2 such that Q(tau2) = k - 1
        def pm_objective(tau2):
            tau2 = max(0, tau2)
            wi = 1 / (vi + tau2)
            sum_wi = wi.sum()

            if sum_wi <= 0:
                return 1e10

            mu = (wi * yi).sum() / sum_wi
            Q = (wi * (yi - mu)**2).sum()

            # We want Q = k - 1
            return (Q - df_Q)**2

        var_yi = np.var(yi, ddof=1) if k > 2 else 1.0
        upper_bound = max(10 * var_yi, 100)

        result = minimize_scalar(
            pm_objective,
            bounds=(0, upper_bound),
            method='bounded',
            options={'maxiter': max_iter, 'xatol': tol}
        )

        if result.success and result.fun < 1:  # Good convergence
            tau_sq = result.x
        else:
            tau_sq = calculate_tau_squared_DL(df, effect_col, var_col)

        return max(0.0, tau_sq)

    except Exception as e:
        warnings.warn(f"Error in PM estimator: {e}, using DL fallback")
        return calculate_tau_squared_DL(df, effect_col, var_col)


def calculate_tau_squared_SJ(df, effect_col, var_col):
    """
    Sidik-Jonkman estimator for tau-squared.

    Advantages:
    - Simple, non-iterative
    - Good performance with few studies
    - Conservative (tends to produce larger estimates)

    Disadvantages:
    - Can be overly conservative
    - Less commonly used

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column

    Returns
    -------
    tau_squared : float
        Sidik-Jonkman estimate of between-study variance

    References
    ----------
    Sidik, K., & Jonkman, J. N. (2005). Simple heterogeneity variance
    estimation for meta-analysis. Journal of the Royal Statistical Society,
    Series C, 54(2), 367-384.
    """
    k = len(df)
    if k < 3:  # Need at least 3 studies for SJ
        return calculate_tau_squared_DL(df, effect_col, var_col)

    try:
        yi = df[effect_col].values
        vi = df[var_col].values

        valid_mask = np.isfinite(vi) & (vi > 0)
        if not valid_mask.all():
            yi = yi[valid_mask]
            vi = vi[valid_mask]
            k = len(yi)

        if k < 3:
            return calculate_tau_squared_DL(df, effect_col, var_col)

        # Weights for typical average
        wi = 1 / vi
        sum_wi = wi.sum()

        # Typical average (weighted mean)
        y_bar = (wi * yi).sum() / sum_wi

        # SJ estimator
        numerator = ((yi - y_bar)**2 / vi).sum()
        denominator = k - 1

        tau_sq = (numerator / denominator) - (k / sum_wi)

        return max(0.0, tau_sq)

    except Exception as e:
        warnings.warn(f"Error in SJ estimator: {e}, using DL fallback")
        return calculate_tau_squared_DL(df, effect_col, var_col)


def calculate_tau_squared(df, effect_col, var_col, method='REML', **kwargs):
    """
    Unified function to calculate tau-squared using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Data with effect sizes and variances
    effect_col : str
        Name of effect size column
    var_col : str
        Name of variance column
    method : str, default='REML'
        Estimation method: 'DL', 'REML', 'ML', 'PM', 'SJ'
        REML is recommended for most applications
    **kwargs : dict
        Additional arguments passed to estimator (e.g., max_iter, tol)

    Returns
    -------
    tau_squared : float
        Estimate of between-study variance
    info : dict
        Dictionary containing:
        - method: Method used
        - tau_squared: The estimate
        - tau: Square root of tau_squared
        - success: Whether estimation succeeded
        - fallback: Whether fallback to DL was used (if applicable)
        - error: Error message (if applicable)

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'effect': [0.5, 0.8, 0.3], 'variance': [0.1, 0.12, 0.09]})
    >>> tau_sq, info = calculate_tau_squared(data, 'effect', 'variance', method='REML')
    >>> print(f"Tau-squared = {tau_sq:.4f}, Tau = {info['tau']:.4f}")
    """
    method = method.upper()

    estimators = {
        'DL': calculate_tau_squared_DL,
        'REML': calculate_tau_squared_REML,
        'ML': calculate_tau_squared_ML,
        'PM': calculate_tau_squared_PM,
        'SJ': calculate_tau_squared_SJ
    }

    if method not in estimators:
        warnings.warn(f"Unknown method '{method}', using REML")
        method = 'REML'

    try:
        tau_sq = estimators[method](df, effect_col, var_col, **kwargs)

        info = {
            'method': method,
            'tau_squared': tau_sq,
            'tau': np.sqrt(tau_sq),
            'success': True
        }

        return tau_sq, info

    except Exception as e:
        warnings.warn(f"Error with {method}, falling back to DL: {e}")
        tau_sq = calculate_tau_squared_DL(df, effect_col, var_col)

        info = {
            'method': 'DL',
            'tau_squared': tau_sq,
            'tau': np.sqrt(tau_sq),
            'success': False,
            'fallback': True,
            'error': str(e)
        }

        return tau_sq, info
