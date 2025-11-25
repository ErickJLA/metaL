import numpy as np
from scipy.stats import t, norm

def calculate_i2(q_stat, df):
    """Calculate I-squared statistic."""
    if q_stat <= df:
        return 0.0
    return 100 * (q_stat - df) / q_stat

def knapp_hartung_adjustment(model_results):
    """
    Apply Knapp-Hartung correction to standard errors.
    """
    # Logic for KH adjustment
    pass

def format_p_value(p):
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"