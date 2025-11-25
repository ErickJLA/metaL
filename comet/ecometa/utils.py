"""
Utility Functions for Meta-Analysis.

This module provides helper functions for heterogeneity assessment,
model diagnostics, and influence analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import t, norm, chi2
import warnings


def calculate_i2(q_stat, df, i2_type='total'):
    """
    Calculate I-squared heterogeneity statistic.

    I² represents the percentage of total variability due to
    heterogeneity rather than sampling error.

    Parameters
    ----------
    q_stat : float
        Cochran's Q statistic
    df : int
        Degrees of freedom (k - 1 for k studies)
    i2_type : str, default='total'
        Type of I²: 'total', 'between', or 'within'
        (for three-level models)

    Returns
    -------
    i2 : float
        I² statistic as a percentage (0-100)

    References
    ----------
    Higgins, J. P. T., & Thompson, S. G. (2002). Quantifying heterogeneity in
    a meta-analysis. Statistics in Medicine, 21(11), 1539-1558.

    Examples
    --------
    >>> calculate_i2(q_stat=25.5, df=10)
    60.78
    """
    if df <= 0:
        return 0.0

    if q_stat <= df:
        return 0.0

    i2 = ((q_stat - df) / q_stat) * 100
    return max(0.0, min(100.0, i2))


def calculate_i2_3level(tau2, sigma2, typical_variance):
    """
    Calculate I² statistics for three-level meta-analysis.

    Partitions heterogeneity into between-study (Level 3),
    within-study (Level 2), and sampling error (Level 1).

    Parameters
    ----------
    tau2 : float
        Between-study variance (Level 3)
    sigma2 : float
        Within-study variance (Level 2)
    typical_variance : float
        Typical sampling variance (Level 1)

    Returns
    -------
    i2_dict : dict
        Dictionary containing:
        - i2_total: Total heterogeneity (%)
        - i2_between: Between-study heterogeneity (%)
        - i2_within: Within-study heterogeneity (%)

    Examples
    --------
    >>> i2 = calculate_i2_3level(tau2=0.05, sigma2=0.03, typical_variance=0.02)
    >>> print(f"Total I²: {i2['i2_total']:.1f}%")
    Total I²: 80.0%
    """
    total_var = tau2 + sigma2 + typical_variance

    if total_var == 0:
        return {
            'i2_total': 0.0,
            'i2_between': 0.0,
            'i2_within': 0.0
        }

    i2_total = ((tau2 + sigma2) / total_var) * 100
    i2_between = (tau2 / total_var) * 100
    i2_within = (sigma2 / total_var) * 100

    return {
        'i2_total': max(0.0, min(100.0, i2_total)),
        'i2_between': max(0.0, min(100.0, i2_between)),
        'i2_within': max(0.0, min(100.0, i2_within))
    }


def calculate_q_statistic(effect_sizes, variances, pooled_effect=None):
    """
    Calculate Cochran's Q statistic for heterogeneity.

    Q measures the weighted sum of squared deviations from
    the pooled effect.

    Parameters
    ----------
    effect_sizes : array-like
        Effect size estimates
    variances : array-like
        Sampling variances
    pooled_effect : float, optional
        Pooled effect estimate. If None, computed from data

    Returns
    -------
    q_stat : float
        Cochran's Q statistic
    p_value : float
        P-value from chi-square test
    df : int
        Degrees of freedom (k - 1)

    Examples
    --------
    >>> effects = np.array([0.5, 0.8, 0.3, 0.6])
    >>> variances = np.array([0.1, 0.12, 0.09, 0.11])
    >>> q, p, df = calculate_q_statistic(effects, variances)
    >>> print(f"Q = {q:.2f}, p = {p:.3f}")
    """
    y = np.asarray(effect_sizes)
    v = np.asarray(variances)

    # Calculate weights
    w = 1 / v
    sum_w = np.sum(w)

    # Calculate pooled effect if not provided
    if pooled_effect is None:
        pooled_effect = np.sum(w * y) / sum_w

    # Calculate Q statistic
    q_stat = np.sum(w * (y - pooled_effect)**2)

    # Degrees of freedom
    df = len(y) - 1

    # P-value from chi-square distribution
    p_value = 1 - chi2.cdf(q_stat, df) if df > 0 else 1.0

    return q_stat, p_value, df


def calculate_aic(log_likelihood, n_params):
    """
    Calculate Akaike Information Criterion (AIC).

    AIC = 2k - 2ln(L)

    Lower values indicate better model fit, penalized for complexity.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters

    Returns
    -------
    aic : float
        AIC value

    Examples
    --------
    >>> aic = calculate_aic(log_likelihood=-12.5, n_params=3)
    >>> print(f"AIC = {aic:.2f}")
    AIC = 31.00
    """
    return 2 * n_params - 2 * log_likelihood


def calculate_bic(log_likelihood, n_params, n_obs):
    """
    Calculate Bayesian Information Criterion (BIC).

    BIC = k×ln(n) - 2ln(L)

    Lower values indicate better model fit, with stronger
    penalty for model complexity than AIC.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters
    n_obs : int
        Number of observations

    Returns
    -------
    bic : float
        BIC value

    Examples
    --------
    >>> bic = calculate_bic(log_likelihood=-12.5, n_params=3, n_obs=50)
    >>> print(f"BIC = {bic:.2f}")
    BIC = 36.71
    """
    return n_params * np.log(n_obs) - 2 * log_likelihood


def calculate_aicc(log_likelihood, n_params, n_obs):
    """
    Calculate corrected AIC (AICc) for small sample sizes.

    AICc = AIC + (2k²+ 2k) / (n - k - 1)

    Recommended when n/k < 40.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of estimated parameters
    n_obs : int
        Number of observations

    Returns
    -------
    aicc : float
        Corrected AIC value

    Examples
    --------
    >>> aicc = calculate_aicc(log_likelihood=-12.5, n_params=3, n_obs=20)
    >>> print(f"AICc = {aicc:.2f}")
    """
    aic = calculate_aic(log_likelihood, n_params)

    # Correction term
    if n_obs - n_params - 1 > 0:
        correction = (2 * n_params**2 + 2 * n_params) / (n_obs - n_params - 1)
        aicc = aic + correction
    else:
        warnings.warn("Sample size too small for AICc correction. Returning AIC.")
        aicc = aic

    return aicc


def compare_models(models_dict):
    """
    Compare multiple models using information criteria.

    Parameters
    ----------
    models_dict : dict
        Dictionary with model names as keys and dicts as values.
        Each value dict must contain:
        - 'log_likelihood': float
        - 'n_params': int
        - 'n_obs': int

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame with AIC, BIC, AICc, and delta values

    Examples
    --------
    >>> models = {
    ...     'Fixed Effects': {'log_likelihood': -15.2, 'n_params': 1, 'n_obs': 30},
    ...     'Random Effects': {'log_likelihood': -12.8, 'n_params': 2, 'n_obs': 30}
    ... }
    >>> df = compare_models(models)
    >>> print(df)
    """
    results = []

    for name, info in models_dict.items():
        ll = info['log_likelihood']
        k = info['n_params']
        n = info['n_obs']

        results.append({
            'Model': name,
            'LogLik': ll,
            'Params': k,
            'AIC': calculate_aic(ll, k),
            'BIC': calculate_bic(ll, k, n),
            'AICc': calculate_aicc(ll, k, n)
        })

    df = pd.DataFrame(results)

    # Calculate delta values (difference from best model)
    df['Delta_AIC'] = df['AIC'] - df['AIC'].min()
    df['Delta_BIC'] = df['BIC'] - df['BIC'].min()

    # Sort by AIC
    df = df.sort_values('AIC').reset_index(drop=True)

    return df


def leave_one_out_analysis(model_class, data, effect_col, var_col, study_id_col,
                           fit_kwargs=None):
    """
    Perform leave-one-out influence analysis.

    Refits the model k times, each time excluding one study,
    to assess the influence of individual studies on the
    pooled effect.

    Parameters
    ----------
    model_class : class
        Model class to use (e.g., ThreeLevelMeta)
    data : pd.DataFrame
        Full dataset
    effect_col : str
        Column name for effect sizes
    var_col : str
        Column name for variances
    study_id_col : str
        Column name for study IDs
    fit_kwargs : dict, optional
        Additional arguments for model.fit()

    Returns
    -------
    loo_results : pd.DataFrame
        DataFrame with one row per study containing:
        - study_id: Study identifier
        - pooled_effect_loo: Pooled effect without this study
        - se_loo: Standard error without this study
        - ci_lower_loo: Lower CI without this study
        - ci_upper_loo: Upper CI without this study
        - influence: Absolute change in pooled effect
        - percent_change: Percentage change in pooled effect

    Examples
    --------
    >>> from ecometa import ThreeLevelMeta
    >>> loo_df = leave_one_out_analysis(
    ...     ThreeLevelMeta, data, 'effect', 'variance', 'study_id'
    ... )
    >>> # Find most influential studies
    >>> print(loo_df.nlargest(3, 'influence'))
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    # Get unique studies
    studies = data[study_id_col].unique()

    # Fit full model for reference
    full_model = model_class(data, effect_col, var_col, study_id_col)
    full_model.fit(**fit_kwargs)
    full_effect = full_model.results['pooled_effect']

    # Leave-one-out iterations
    loo_results = []

    for study in studies:
        # Create dataset without this study
        loo_data = data[data[study_id_col] != study].copy()

        # Skip if insufficient data
        if len(loo_data) < 2:
            continue

        try:
            # Fit model
            loo_model = model_class(loo_data, effect_col, var_col, study_id_col)
            loo_model.fit(**fit_kwargs)

            # Extract results
            loo_effect = loo_model.results['pooled_effect']
            loo_se = loo_model.results['se']
            loo_ci_lower = loo_model.results['ci_lower']
            loo_ci_upper = loo_model.results['ci_upper']

            # Calculate influence
            influence = abs(full_effect - loo_effect)
            percent_change = (influence / abs(full_effect)) * 100 if full_effect != 0 else 0

            loo_results.append({
                'study_id': study,
                'pooled_effect_loo': loo_effect,
                'se_loo': loo_se,
                'ci_lower_loo': loo_ci_lower,
                'ci_upper_loo': loo_ci_upper,
                'influence': influence,
                'percent_change': percent_change
            })

        except Exception as e:
            warnings.warn(f"Failed to fit model without study {study}: {e}")
            continue

    loo_df = pd.DataFrame(loo_results)

    # Sort by influence (descending)
    if not loo_df.empty:
        loo_df = loo_df.sort_values('influence', ascending=False).reset_index(drop=True)

    return loo_df


def format_p_value(p, threshold=0.001, decimals=3):
    """
    Format p-value for publication.

    Parameters
    ----------
    p : float
        P-value to format
    threshold : float, default=0.001
        Threshold below which to use "< threshold" format
    decimals : int, default=3
        Number of decimal places

    Returns
    -------
    formatted : str
        Formatted p-value string

    Examples
    --------
    >>> format_p_value(0.0001)
    '< 0.001'
    >>> format_p_value(0.042)
    '= 0.042'
    >>> format_p_value(0.5)
    '= 0.500'
    """
    if p < threshold:
        return f"< {threshold}"
    else:
        return f"= {p:.{decimals}f}"


def interpret_heterogeneity(i2):
    """
    Provide qualitative interpretation of I² statistic.

    Based on Higgins et al. (2003) guidelines.

    Parameters
    ----------
    i2 : float
        I² statistic (percentage)

    Returns
    -------
    interpretation : str
        Qualitative interpretation

    References
    ----------
    Higgins, J. P. T., Thompson, S. G., Deeks, J. J., & Altman, D. G. (2003).
    Measuring inconsistency in meta-analyses. BMJ, 327(7414), 557-560.

    Examples
    --------
    >>> interpret_heterogeneity(25)
    'Low heterogeneity'
    >>> interpret_heterogeneity(60)
    'Moderate heterogeneity'
    >>> interpret_heterogeneity(85)
    'High heterogeneity'
    """
    if i2 < 25:
        return "Low heterogeneity"
    elif i2 < 50:
        return "Moderate heterogeneity"
    elif i2 < 75:
        return "Substantial heterogeneity"
    else:
        return "High heterogeneity"


def interpret_tau2(tau2, scale='log'):
    """
    Provide qualitative interpretation of tau² (between-study variance).

    Parameters
    ----------
    tau2 : float
        Tau-squared value
    scale : str, default='log'
        Scale of effect sizes: 'log' (lnRR, log OR) or 'standardized' (SMD, Hedges' g)

    Returns
    -------
    interpretation : str
        Qualitative interpretation

    Examples
    --------
    >>> interpret_tau2(0.01, scale='log')
    'Low heterogeneity'
    >>> interpret_tau2(0.15, scale='standardized')
    'Substantial heterogeneity'
    """
    if scale == 'log':
        # Guidelines for log-scale effect sizes (lnRR, log OR)
        if tau2 < 0.04:
            return "Low heterogeneity"
        elif tau2 < 0.16:
            return "Moderate heterogeneity"
        else:
            return "High heterogeneity"
    else:  # standardized
        # Guidelines for standardized effect sizes (SMD, Hedges' g)
        if tau2 < 0.04:
            return "Low heterogeneity"
        elif tau2 < 0.09:
            return "Moderate heterogeneity"
        else:
            return "High heterogeneity"
