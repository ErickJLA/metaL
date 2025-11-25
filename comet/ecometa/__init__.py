"""
ecometa: A Professional Python Library for Three-Level Meta-Analysis

This library provides tools for conducting meta-analyses with multiple effect
sizes per study (three-level or multilevel meta-analysis), including:

- Effect size calculation (Hedges' g, log response ratio)
- Variance estimation (DL, REML, ML, PM, SJ methods)
- Three-level random-effects models
- Meta-regression (linear and non-linear/spline)
- Publication bias assessment
- Visualization tools (forest, funnel, orchard, spline plots)
- Sensitivity analysis (leave-one-out)
- Model diagnostics (AIC, BIC, I²)

Example Usage
-------------
>>> from ecometa import ThreeLevelMeta, MetaRegression
>>> import pandas as pd
>>>
>>> # Basic three-level meta-analysis
>>> model = ThreeLevelMeta(data, 'effect', 'variance', 'study_id')
>>> model.fit()
>>> print(model.summary())
>>>
>>> # Meta-regression with moderators
>>> reg_model = MetaRegression(data, 'effect', 'variance', 'study_id', 'year')
>>> reg_model.fit()
>>> print(reg_model.summary())
>>>
>>> # Create publication-ready plots
>>> from ecometa.plots import forest_plot, funnel_plot
>>> fig, ax = forest_plot(data, 'effect', 'variance', overall_effect=0.5, overall_se=0.1)
>>> plt.show()
"""

__version__ = "0.1.0"

# Core models
from .models import ThreeLevelMeta

# Regression models
from .regression import MetaRegression, SplineMetaRegression

# Effect size and variance estimators
from .estimators import (
    calculate_hedges_g,
    calculate_lnrr,
    calculate_tau_squared,
    calculate_tau_squared_DL,
    calculate_tau_squared_REML,
    calculate_tau_squared_ML,
    calculate_tau_squared_PM,
    calculate_tau_squared_SJ
)

# Plotting functions
from .plots import (
    orchard_plot,
    forest_plot,
    funnel_plot,
    trim_and_fill_plot,
    spline_plot
)

# Interactive widgets
from .interactive.plot_guis import (
    orchard_plot_interactive,
    forest_plot_interactive,
    funnel_plot_interactive,
    trim_and_fill_plot_interactive,
    spline_plot_interactive
)

# Utility functions
from .utils import (
    calculate_i2,
    calculate_i2_3level,
    calculate_q_statistic,
    calculate_aic,
    calculate_bic,
    calculate_aicc,
    compare_models,
    leave_one_out_analysis,
    format_p_value,
    interpret_heterogeneity,
    interpret_tau2
)

__all__ = [
    # Core Models
    "ThreeLevelMeta",
    "MetaRegression",
    "SplineMetaRegression",

    # Effect Size Calculators
    "calculate_hedges_g",
    "calculate_lnrr",

    # Variance Estimators
    "calculate_tau_squared",
    "calculate_tau_squared_DL",
    "calculate_tau_squared_REML",
    "calculate_tau_squared_ML",
    "calculate_tau_squared_PM",
    "calculate_tau_squared_SJ",

    # Visualization (Static)
    "orchard_plot",
    "forest_plot",
    "funnel_plot",
    "trim_and_fill_plot",
    "spline_plot",

    # Visualization (Interactive)
    "orchard_plot_interactive",
    "forest_plot_interactive",
    "funnel_plot_interactive",
    "trim_and_fill_plot_interactive",
    "spline_plot_interactive",

    # Utility Functions
    "calculate_i2",
    "calculate_i2_3level",
    "calculate_q_statistic",
    "calculate_aic",
    "calculate_bic",
    "calculate_aicc",
    "compare_models",
    "leave_one_out_analysis",
    "format_p_value",
    "interpret_heterogeneity",
    "interpret_tau2",
]
