"""
Visualization Functions for Meta-Analysis.

This module provides publication-ready plotting functions for meta-analysis results.
All functions are stateless and accept explicit arguments instead of global configuration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings


def forest_plot(data,
                effect_col,
                var_col,
                study_col=None,
                group_col=None,
                overall_effect=None,
                overall_se=None,
                overall_ci=None,
                figsize=None,
                title="Forest Plot",
                xlabel="Effect Size",
                marker_size=100,
                alpha=0.05,
                color='steelblue',
                overall_color='darkred',
                show_weights=True,
                show_ci_text=False):
    """
    Create a classic forest plot for meta-analysis results.

    Displays individual study effect sizes with confidence intervals,
    optional subgroups, and overall pooled effect.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing effect sizes and variances
    effect_col : str
        Column name for effect sizes
    var_col : str
        Column name for sampling variances
    study_col : str, optional
        Column name for study labels (if None, uses index)
    group_col : str, optional
        Column name for grouping (subgroup analysis)
    overall_effect : float, optional
        Overall pooled effect size
    overall_se : float, optional
        Standard error of overall effect
    overall_ci : tuple, optional
        (lower, upper) confidence interval for overall effect
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None
    title : str, default="Forest Plot"
        Plot title
    xlabel : str, default="Effect Size"
        X-axis label
    marker_size : float, default=100
        Size of effect size markers (scaled by precision)
    alpha : float, default=0.05
        Significance level for confidence intervals
    color : str, default='steelblue'
        Color for study markers
    overall_color : str, default='darkred'
        Color for overall effect diamond
    show_weights : bool, default=True
        Show study weights as marker sizes
    show_ci_text : bool, default=False
        Show CI values as text annotations

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects

    Examples
    --------
    >>> import pandas as pd
    >>> from ecometa.plots import forest_plot
    >>> data = pd.DataFrame({
    ...     'study': ['Study A', 'Study B', 'Study C'],
    ...     'effect': [0.5, 0.8, 0.3],
    ...     'variance': [0.1, 0.12, 0.09]
    ... })
    >>> fig, ax = forest_plot(data, 'effect', 'variance', 'study',
    ...                        overall_effect=0.53, overall_se=0.08)
    >>> plt.show()
    """
    # Prepare data
    plot_data = data.copy()

    # Calculate standard errors and CIs
    plot_data['se'] = np.sqrt(plot_data[var_col])
    z_crit = norm.ppf(1 - alpha / 2)
    plot_data['ci_lower'] = plot_data[effect_col] - z_crit * plot_data['se']
    plot_data['ci_upper'] = plot_data[effect_col] + z_crit * plot_data['se']

    # Calculate weights for marker sizing
    if show_weights:
        plot_data['weight'] = 1 / plot_data[var_col]
        plot_data['weight_norm'] = (plot_data['weight'] / plot_data['weight'].sum()) * 1000
    else:
        plot_data['weight_norm'] = marker_size

    # Study labels
    if study_col:
        plot_data['label'] = plot_data[study_col].astype(str)
    else:
        plot_data['label'] = plot_data.index.astype(str)

    # Grouping
    if group_col:
        plot_data['group'] = plot_data[group_col].astype(str)
        groups = plot_data['group'].unique()
    else:
        plot_data['group'] = 'All'
        groups = ['All']

    # Calculate figure height
    n_studies = len(plot_data)
    n_groups = len(groups)
    add_overall = 1 if overall_effect is not None else 0
    total_rows = n_studies + n_groups + add_overall

    if figsize is None:
        fig_height = max(6, total_rows * 0.4 + 2)
        figsize = (10, fig_height)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Y-position counter
    y_pos = 0
    y_positions = []
    y_labels = []

    # Plot groups
    for group in groups:
        group_data = plot_data[plot_data['group'] == group]

        # Group header (if multiple groups)
        if len(groups) > 1:
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else plot_data[effect_col].min(),
                    y_pos, f"  {group}", fontweight='bold', fontsize=11,
                    va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
            y_pos += 1

        # Plot studies in group
        for idx, row in group_data.iterrows():
            # Confidence interval line
            ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos],
                    color=color, linewidth=2, alpha=0.6, zorder=1)

            # Effect size marker
            ax.scatter(row[effect_col], y_pos,
                       s=row['weight_norm'], color=color,
                       edgecolors='white', linewidth=1.5,
                       alpha=0.8, zorder=3)

            # Label
            y_positions.append(y_pos)
            y_labels.append(row['label'])

            # Optional CI text
            if show_ci_text:
                ci_text = f"{row[effect_col]:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
                ax.text(ax.get_xlim()[1] * 0.95, y_pos, ci_text,
                        ha='right', va='center', fontsize=8, color='gray')

            y_pos += 1

        # Add spacing between groups
        if len(groups) > 1:
            y_pos += 0.5

    # Add overall effect (diamond)
    if overall_effect is not None:
        y_pos += 0.5  # Add spacing

        if overall_ci is not None:
            ci_lower, ci_upper = overall_ci
        elif overall_se is not None:
            ci_lower = overall_effect - z_crit * overall_se
            ci_upper = overall_effect + z_crit * overall_se
        else:
            ci_lower = ci_upper = overall_effect

        # Draw diamond
        diamond_height = 0.3
        diamond_x = [ci_lower, overall_effect, ci_upper, overall_effect, ci_lower]
        diamond_y = [y_pos, y_pos + diamond_height, y_pos, y_pos - diamond_height, y_pos]

        ax.fill(diamond_x, diamond_y, color=overall_color, alpha=0.7,
                edgecolor='white', linewidth=2, zorder=5)

        y_positions.append(y_pos)
        y_labels.append("Overall Effect")

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add null line
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Invert y-axis (top to bottom)
    ax.invert_yaxis()

    # Remove top and right spines
    sns.despine(ax=ax)

    # Set x-axis limits with padding
    x_min = min(plot_data['ci_lower'].min(), ci_lower if overall_effect else 0) * 1.1
    x_max = max(plot_data['ci_upper'].max(), ci_upper if overall_effect else 0) * 1.1
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    return fig, ax


def funnel_plot(effect_sizes,
                variances,
                center=None,
                figsize=(8, 8),
                title="Funnel Plot",
                xlabel="Effect Size",
                ylabel="Standard Error",
                color='steelblue',
                alpha=0.6,
                marker_size=60,
                show_contours=True,
                contour_levels=[0.90, 0.95, 0.99],
                invert_y=True):
    """
    Create a funnel plot for assessing publication bias.

    Plots effect sizes against their standard errors, with optional
    pseudo-confidence interval contours.

    Parameters
    ----------
    effect_sizes : array-like
        Effect size estimates
    variances : array-like
        Sampling variances
    center : float, optional
        Center of the funnel (pooled effect). If None, uses mean
    figsize : tuple, default=(8, 8)
        Figure size (width, height)
    title : str, default="Funnel Plot"
        Plot title
    xlabel : str, default="Effect Size"
        X-axis label
    ylabel : str, default="Standard Error"
        Y-axis label
    color : str, default='steelblue'
        Color for data points
    alpha : float, default=0.6
        Transparency of markers
    marker_size : float, default=60
        Size of markers
    show_contours : bool, default=True
        Whether to show pseudo-confidence interval contours
    contour_levels : list, default=[0.90, 0.95, 0.99]
        Confidence levels for contours (as proportions)
    invert_y : bool, default=True
        Invert y-axis (larger studies at top)

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects

    Examples
    --------
    >>> from ecometa.plots import funnel_plot
    >>> import numpy as np
    >>> effects = np.array([0.5, 0.8, 0.3, 0.6, 0.4])
    >>> variances = np.array([0.1, 0.12, 0.09, 0.11, 0.10])
    >>> fig, ax = funnel_plot(effects, variances, center=0.52)
    >>> plt.show()
    """
    # Convert to numpy arrays
    y = np.asarray(effect_sizes)
    v = np.asarray(variances)
    se = np.sqrt(v)

    # Determine center
    if center is None:
        # Use inverse-variance weighted mean
        weights = 1 / v
        center = np.sum(weights * y) / np.sum(weights)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    ax.scatter(y, se, s=marker_size, color=color, alpha=alpha,
               edgecolors='white', linewidth=1, zorder=3)

    # Draw contours
    if show_contours:
        se_range = np.linspace(0, se.max() * 1.1, 100)

        for level in contour_levels:
            z_crit = norm.ppf(1 - (1 - level) / 2)
            ci_width = z_crit * se_range

            # Draw symmetric contours
            ax.plot(center + ci_width, se_range, color='gray',
                    linestyle='--', linewidth=1, alpha=0.5, zorder=1)
            ax.plot(center - ci_width, se_range, color='gray',
                    linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # Add center line
    ax.axvline(center, color='darkred', linestyle='-', linewidth=2,
               alpha=0.7, zorder=2, label=f'Pooled Effect = {center:.3f}')

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    if invert_y:
        ax.invert_yaxis()

    ax.legend(loc='best', frameon=True, framealpha=0.9)
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig, ax


def trim_and_fill_plot(effect_sizes,
                        variances,
                        filled_effects=None,
                        filled_variances=None,
                        center_original=None,
                        center_adjusted=None,
                        figsize=(9, 8),
                        title="Trim-and-Fill Funnel Plot",
                        xlabel="Effect Size",
                        ylabel="Standard Error",
                        original_color='steelblue',
                        filled_color='coral',
                        alpha=0.7,
                        marker_size=60,
                        show_contours=True):
    """
    Create a trim-and-fill funnel plot showing imputed studies.

    Visualizes the original studies and imputed "missing" studies
    identified by the trim-and-fill method for publication bias.

    Parameters
    ----------
    effect_sizes : array-like
        Original effect size estimates
    variances : array-like
        Original sampling variances
    filled_effects : array-like, optional
        Imputed (filled) effect sizes
    filled_variances : array-like, optional
        Imputed (filled) variances
    center_original : float, optional
        Original pooled effect
    center_adjusted : float, optional
        Adjusted pooled effect after trim-and-fill
    figsize : tuple, default=(9, 8)
        Figure size (width, height)
    title : str, default="Trim-and-Fill Funnel Plot"
        Plot title
    xlabel : str, default="Effect Size"
        X-axis label
    ylabel : str, default="Standard Error"
        Y-axis label
    original_color : str, default='steelblue'
        Color for original studies
    filled_color : str, default='coral'
        Color for imputed studies
    alpha : float, default=0.7
        Transparency of markers
    marker_size : float, default=60
        Size of markers
    show_contours : bool, default=True
        Whether to show pseudo-confidence interval contours

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects

    Examples
    --------
    >>> from ecometa.plots import trim_and_fill_plot
    >>> import numpy as np
    >>> effects = np.array([0.5, 0.8, 0.3, 0.6])
    >>> variances = np.array([0.1, 0.12, 0.09, 0.11])
    >>> filled_effects = np.array([0.2, 0.1])  # Imputed studies
    >>> filled_variances = np.array([0.10, 0.11])
    >>> fig, ax = trim_and_fill_plot(effects, variances, filled_effects,
    ...                                filled_variances, 0.55, 0.42)
    >>> plt.show()
    """
    # Convert to numpy arrays
    y_orig = np.asarray(effect_sizes)
    v_orig = np.asarray(variances)
    se_orig = np.sqrt(v_orig)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot original studies
    ax.scatter(y_orig, se_orig, s=marker_size, color=original_color,
               alpha=alpha, edgecolors='white', linewidth=1.5,
               label='Observed Studies', zorder=3)

    # Plot filled studies
    if filled_effects is not None and filled_variances is not None:
        y_fill = np.asarray(filled_effects)
        v_fill = np.asarray(filled_variances)
        se_fill = np.sqrt(v_fill)

        ax.scatter(y_fill, se_fill, s=marker_size, color=filled_color,
                   alpha=alpha, edgecolors='darkred', linewidth=2,
                   marker='o', facecolors='none',
                   label=f'Imputed Studies (k={len(y_fill)})', zorder=3)

        all_se = np.concatenate([se_orig, se_fill])
        all_y = np.concatenate([y_orig, y_fill])
    else:
        all_se = se_orig
        all_y = y_orig

    # Draw contours
    if show_contours:
        se_range = np.linspace(0, all_se.max() * 1.1, 100)
        center_for_contour = center_adjusted if center_adjusted is not None else center_original

        if center_for_contour is not None:
            for level in [0.95, 0.99]:
                z_crit = norm.ppf(1 - (1 - level) / 2)
                ci_width = z_crit * se_range

                ax.plot(center_for_contour + ci_width, se_range,
                        color='gray', linestyle='--', linewidth=1,
                        alpha=0.4, zorder=1)
                ax.plot(center_for_contour - ci_width, se_range,
                        color='gray', linestyle='--', linewidth=1,
                        alpha=0.4, zorder=1)

    # Add center lines
    if center_original is not None:
        ax.axvline(center_original, color=original_color, linestyle='--',
                   linewidth=2, alpha=0.8,
                   label=f'Original: {center_original:.3f}', zorder=2)

    if center_adjusted is not None:
        ax.axvline(center_adjusted, color=filled_color, linestyle='-',
                   linewidth=2, alpha=0.8,
                   label=f'Adjusted: {center_adjusted:.3f}', zorder=2)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()

    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='gray')
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig, ax


def spline_plot(moderator_values,
                predictions,
                ci_lower=None,
                ci_upper=None,
                observed_effects=None,
                observed_moderators=None,
                observed_sizes=None,
                figsize=(10, 6),
                title="Spline Meta-Regression",
                xlabel="Moderator",
                ylabel="Predicted Effect Size",
                line_color='darkred',
                ci_color='lightcoral',
                point_color='steelblue',
                alpha=0.7,
                show_ci=True,
                show_points=True):
    """
    Create a spline meta-regression plot with smooth fitted curve.

    Visualizes the non-linear relationship between a continuous moderator
    and effect sizes using natural cubic splines.

    Parameters
    ----------
    moderator_values : array-like
        Values of the moderator variable for prediction
    predictions : array-like
        Predicted effect sizes from spline model
    ci_lower : array-like, optional
        Lower confidence interval bounds
    ci_upper : array-like, optional
        Upper confidence interval bounds
    observed_effects : array-like, optional
        Observed effect sizes (raw data)
    observed_moderators : array-like, optional
        Observed moderator values (raw data)
    observed_sizes : array-like, optional
        Sizes for observed data points (e.g., by precision)
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    title : str, default="Spline Meta-Regression"
        Plot title
    xlabel : str, default="Moderator"
        X-axis label
    ylabel : str, default="Predicted Effect Size"
        Y-axis label
    line_color : str, default='darkred'
        Color for fitted line
    ci_color : str, default='lightcoral'
        Color for confidence interval band
    point_color : str, default='steelblue'
        Color for observed data points
    alpha : float, default=0.7
        Transparency for confidence band
    show_ci : bool, default=True
        Whether to show confidence interval band
    show_points : bool, default=True
        Whether to show observed data points

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects

    Examples
    --------
    >>> from ecometa.plots import spline_plot
    >>> import numpy as np
    >>> x_pred = np.linspace(10, 30, 100)
    >>> y_pred = 0.5 + 0.02 * x_pred - 0.0005 * x_pred**2  # Quadratic
    >>> ci_low = y_pred - 0.1
    >>> ci_high = y_pred + 0.1
    >>> fig, ax = spline_plot(x_pred, y_pred, ci_low, ci_high)
    >>> plt.show()
    """
    # Convert to numpy arrays
    x_pred = np.asarray(moderator_values)
    y_pred = np.asarray(predictions)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confidence interval band
    if show_ci and ci_lower is not None and ci_upper is not None:
        ci_low = np.asarray(ci_lower)
        ci_high = np.asarray(ci_upper)

        ax.fill_between(x_pred, ci_low, ci_high,
                        color=ci_color, alpha=alpha,
                        label='95% CI', zorder=1)

    # Plot fitted spline curve
    ax.plot(x_pred, y_pred, color=line_color, linewidth=3,
            label='Fitted Spline', zorder=3)

    # Plot observed data points
    if show_points and observed_effects is not None and observed_moderators is not None:
        obs_y = np.asarray(observed_effects)
        obs_x = np.asarray(observed_moderators)

        if observed_sizes is not None:
            sizes = np.asarray(observed_sizes)
        else:
            sizes = 100

        ax.scatter(obs_x, obs_y, s=sizes, color=point_color,
                   alpha=0.6, edgecolors='white', linewidth=1.5,
                   label='Observed Data', zorder=2)

    # Add null line
    ax.axhline(0, color='black', linestyle='--', linewidth=1,
               alpha=0.5, zorder=0)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='gray')
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig, ax


def orchard_plot(model,
                 group_by=None,
                 figsize=(10, 6),
                 alpha=0.05,
                 trunk_alpha=0.5,
                 leaf_alpha=0.6,
                 leaf_size=50,
                 color_palette="viridis",
                 show_pi=True):
    """
    Generates an Orchard Plot for 3-Level Meta-Analysis.

    Visualizes the "Trunk" (Prediction Interval), "Fruit" (Confidence Interval),
    and "Leaves" (Raw Effect Sizes).

    Parameters
    ----------
    model : ThreeLevelMeta
        A fitted instance of ThreeLevelMeta.
    group_by : str, optional
        Column name to calculate subgroups. If provided, the function will
        automatically fit separate models for each subgroup and plot them
        alongside the overall model.
    figsize : tuple
        Dimensions of the plot (width, height).
    alpha : float
        Significance level (default 0.05 for 95% CIs).
    trunk_alpha : float
        Transparency of the prediction interval bar.
    leaf_alpha : float
        Transparency of the raw data points.
    leaf_size : int
        Size of the raw data points.
    color_palette : str
        Seaborn/Matplotlib color palette name.
    show_pi : bool
        Whether to show the Prediction Interval (Trunk).

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    from .models import ThreeLevelMeta

    # 1. Prepare Results Table
    # ------------------------
    results_list = []

    # A. Overall Result
    if model.results is None:
        raise ValueError("Model not fitted. Call .fit() first.")

    overall_res = model.results
    overall_res['group'] = 'Overall'
    overall_res['k'] = model.n_obs
    results_list.append(overall_res)

    # B. Subgroup Results (if requested)
    if group_by:
        if group_by not in model.data.columns:
            raise ValueError(f"Group column '{group_by}' not found in data.")

        groups = model.data[group_by].unique()
        for grp in groups:
            # Subset data
            sub_data = model.data[model.data[group_by] == grp].copy()

            # Skip tiny groups
            if len(sub_data) < 2:
                continue

            # Fit separate model for subgroup
            # We use the same column settings as the parent model
            sub_model = ThreeLevelMeta(
                sub_data,
                model.effect_col,
                model.var_col,
                model.study_id_col
            )
            try:
                sub_model.fit()
                res = sub_model.results
                res['group'] = str(grp)
                res['k'] = len(sub_data)
                results_list.append(res)
            except Exception:
                continue  # Skip groups that fail to converge

    # Create plotting dataframe
    plot_df = pd.DataFrame(results_list)

    # Calculate Prediction Intervals (Trunks)
    # PI_SD = sqrt(SE^2 + tau^2 + sigma^2)
    z_score = 1.96  # Approximation for 95%

    plot_df['pi_sd'] = np.sqrt(
        plot_df['se']**2 +
        plot_df['tau2'] +
        plot_df['sigma2']
    )
    plot_df['pi_lower'] = plot_df['pooled_effect'] - z_score * plot_df['pi_sd']
    plot_df['pi_upper'] = plot_df['pooled_effect'] + z_score * plot_df['pi_sd']

    # 2. Setup Plot
    # -------------
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    groups = plot_df['group'].tolist()

    # Map colors
    palette = sns.color_palette(color_palette, n_colors=len(groups))
    color_map = {g: c for g, c in zip(groups, palette)}
    # Force Overall to be black/grey
    color_map['Overall'] = 'black'

    # Y-positions (spaced out)
    y_positions = np.arange(len(groups))

    # 3. Draw Elements
    # ----------------
    for i, grp in enumerate(groups):
        row = plot_df[plot_df['group'] == grp].iloc[0]
        y = i
        col = color_map[grp]

        # A. Draw Trunk (Prediction Interval)
        if show_pi:
            ax.plot([row['pi_lower'], row['pi_upper']], [y, y],
                    color=col, alpha=trunk_alpha, linewidth=4,
                    solid_capstyle='round', zorder=1)

        # B. Draw Leaves (Raw Data)
        # We need the raw data for this group
        if grp == 'Overall':
            raw_y = model.data[model.effect_col].values
        else:
            raw_y = model.data[model.data[group_by] == str(grp)][model.effect_col].values

        # Add jitter to y-axis
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.15, 0.15, size=len(raw_y))

        ax.scatter(raw_y, y + jitter,
                   s=leaf_size, color=col, alpha=leaf_alpha,
                   edgecolors='none', zorder=2)

        # C. Draw Fruit (Confidence Interval + Mean)
        # CI Line
        ax.plot([row['ci_lower'], row['ci_upper']], [y, y],
                color='white',
                linewidth=2, zorder=3)  # Inner white line for contrast

        ax.plot([row['ci_lower'], row['ci_upper']], [y, y],
                color=col, linewidth=1.5, linestyle='-', zorder=4)

        # Mean Point
        ax.plot(row['pooled_effect'], y, marker='D', markersize=10,
                color=col, markeredgecolor='white', zorder=5)

        # Add Annotation (k)
        ax.text(row['pi_upper'] if show_pi else row['ci_upper'],
                y + 0.25,
                f" k={row['k']}",
                va='center', fontsize=9, color='#555')

    # 4. Final Formatting
    # -------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Effect Size", fontsize=12, fontweight='bold')

    # Add zero line
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=0)

    # Remove top/right spines
    sns.despine(ax=ax, left=True)

    # Invert Y axis so first group is at top
    # Standard forest plots go Top->Bottom.
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax
