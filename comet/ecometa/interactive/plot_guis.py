"""
Interactive Plot Widgets for Meta-Analysis.

This module provides ipywidgets-based interactive interfaces for all
static plot functions, allowing users to customize visualizations
in Jupyter notebooks.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from ..plots import orchard_plot, forest_plot, funnel_plot, trim_and_fill_plot, spline_plot


def orchard_plot_interactive(model):
    """
    Launches an interactive GUI for the Orchard Plot.

    Features:
    - Dropdown to select subgrouping variable (Group By).
    - Sliders for plot dimensions.
    - Toggles for visual elements (Trunk, Fruit, Leaves).
    - Generates reproducible Python code for the current view.

    Parameters
    ----------
    model : ThreeLevelMeta
        A fitted instance of the ThreeLevelMeta model.
    """

    # 1. Setup Widgets
    # ----------------

    # Identify potential grouping columns (categorical/object types with <20 unique values)
    df = model.data
    potential_groups = ['None']
    for col in df.columns:
        if df[col].dtype == 'object' or hasattr(df[col], 'cat'):
            if df[col].nunique() < 20 and col != model.study_id_col:
                potential_groups.append(col)

    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='95%')

    # Data Widgets
    w_group = widgets.Dropdown(
        options=potential_groups,
        value='None',
        description='Group By:',
        style=style, layout=layout
    )

    # Appearance Widgets
    w_width = widgets.FloatSlider(value=10, min=5, max=15, step=0.5, description='Width (in):', style=style, layout=layout)
    w_height = widgets.FloatSlider(value=6, min=4, max=12, step=0.5, description='Height (in):', style=style, layout=layout)
    w_palette = widgets.Dropdown(
        options=['viridis', 'plasma', 'coolwarm', 'tab10', 'Set2'],
        value='viridis',
        description='Color Palette:',
        style=style, layout=layout
    )

    # Toggle Widgets
    w_show_pi = widgets.Checkbox(value=True, description='Show Prediction Interval (Trunk)')
    w_leaf_size = widgets.IntSlider(value=50, min=10, max=150, description='Leaf Size:', style=style, layout=layout)

    # 2. Layout
    # ---------
    ui = widgets.VBox([
        widgets.HTML("<b>DATA & GROUPING</b>"),
        w_group,
        widgets.HTML("<hr><b>APPEARANCE</b>"),
        widgets.HBox([w_width, w_height]),
        w_palette,
        w_leaf_size,
        w_show_pi
    ])

    # 3. Output Areas
    # ---------------
    out_plot = widgets.Output()
    out_code = widgets.Output()

    # 4. Update Logic
    # ---------------
    def update_view(change=None):
        # Get values
        group_col = None if w_group.value == 'None' else w_group.value
        width = w_width.value
        height = w_height.value
        palette = w_palette.value
        leaf_size = w_leaf_size.value
        show_pi = w_show_pi.value

        # A. Generate Plot
        with out_plot:
            clear_output(wait=True)
            try:
                fig, ax = orchard_plot(
                    model=model,
                    group_by=group_col,
                    figsize=(width, height),
                    color_palette=palette,
                    leaf_size=leaf_size,
                    show_pi=show_pi
                )
                plt.show(fig)
            except Exception as e:
                print(f"Error generating plot: {e}")

        # B. Generate Reproducible Code
        with out_code:
            clear_output()
            group_arg = f"'{group_col}'" if group_col else "None"

            code = (
                f"# --- Reproducible Code ---\n"
                f"from ecometa.plots import orchard_plot\n\n"
                f"fig, ax = orchard_plot(\n"
                f"    model=my_model,\n"
                f"    group_by={group_arg},\n"
                f"    figsize=({width}, {height}),\n"
                f"    color_palette='{palette}',\n"
                f"    leaf_size={leaf_size},\n"
                f"    show_pi={show_pi}\n"
                f")"
            )
            print(code)

    # 5. Bind Events
    # --------------
    w_group.observe(update_view, names='value')
    w_width.observe(update_view, names='value')
    w_height.observe(update_view, names='value')
    w_palette.observe(update_view, names='value')
    w_leaf_size.observe(update_view, names='value')
    w_show_pi.observe(update_view, names='value')

    # Initial call
    update_view()

    # Display
    display(widgets.HBox([ui, widgets.VBox([out_plot, out_code])]))


def forest_plot_interactive(data, effect_col, var_col, study_col=None, group_col=None,
                            overall_effect=None, overall_se=None):
    """
    Launches an interactive GUI for the Forest Plot.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing effect sizes and variances
    effect_col : str
        Column name for effect sizes
    var_col : str
        Column name for sampling variances
    study_col : str, optional
        Column name for study labels
    group_col : str, optional
        Column name for grouping
    overall_effect : float, optional
        Overall pooled effect
    overall_se : float, optional
        Standard error of overall effect
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='95%')

    # Widgets
    w_width = widgets.FloatSlider(value=10, min=6, max=14, step=0.5, description='Width (in):', style=style, layout=layout)
    w_title = widgets.Text(value='Forest Plot', description='Title:', style=style, layout=layout)
    w_xlabel = widgets.Text(value='Effect Size', description='X Label:', style=style, layout=layout)
    w_color = widgets.Dropdown(options=['steelblue', 'darkgreen', 'coral', 'purple'], value='steelblue', description='Color:', style=style, layout=layout)
    w_show_weights = widgets.Checkbox(value=True, description='Show Weights (marker size)')
    w_show_ci_text = widgets.Checkbox(value=False, description='Show CI Text')

    # Layout
    ui = widgets.VBox([
        widgets.HTML("<b>PLOT OPTIONS</b>"),
        w_title,
        w_width,
        w_xlabel,
        w_color,
        w_show_weights,
        w_show_ci_text
    ])

    out_plot = widgets.Output()
    out_code = widgets.Output()

    def update_view(change=None):
        with out_plot:
            clear_output(wait=True)
            try:
                fig, ax = forest_plot(
                    data=data,
                    effect_col=effect_col,
                    var_col=var_col,
                    study_col=study_col,
                    group_col=group_col,
                    overall_effect=overall_effect,
                    overall_se=overall_se,
                    figsize=(w_width.value, None),
                    title=w_title.value,
                    xlabel=w_xlabel.value,
                    color=w_color.value,
                    show_weights=w_show_weights.value,
                    show_ci_text=w_show_ci_text.value
                )
                plt.show(fig)
            except Exception as e:
                print(f"Error: {e}")

        with out_code:
            clear_output()
            code = (
                f"from ecometa.plots import forest_plot\n\n"
                f"fig, ax = forest_plot(\n"
                f"    data=data,\n"
                f"    effect_col='{effect_col}',\n"
                f"    var_col='{var_col}',\n"
                f"    study_col={repr(study_col)},\n"
                f"    group_col={repr(group_col)},\n"
                f"    overall_effect={overall_effect},\n"
                f"    overall_se={overall_se},\n"
                f"    title='{w_title.value}',\n"
                f"    xlabel='{w_xlabel.value}',\n"
                f"    color='{w_color.value}',\n"
                f"    show_weights={w_show_weights.value}\n"
                f")"
            )
            print(code)

    # Bind events
    for widget in [w_width, w_title, w_xlabel, w_color, w_show_weights, w_show_ci_text]:
        widget.observe(update_view, names='value')

    update_view()
    display(widgets.HBox([ui, widgets.VBox([out_plot, out_code])]))


def funnel_plot_interactive(effect_sizes, variances, center=None):
    """
    Launches an interactive GUI for the Funnel Plot.

    Parameters
    ----------
    effect_sizes : array-like
        Effect size estimates
    variances : array-like
        Sampling variances
    center : float, optional
        Center of the funnel (pooled effect)
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='95%')

    # Widgets
    w_width = widgets.FloatSlider(value=8, min=6, max=12, step=0.5, description='Width (in):', style=style, layout=layout)
    w_height = widgets.FloatSlider(value=8, min=6, max=12, step=0.5, description='Height (in):', style=style, layout=layout)
    w_title = widgets.Text(value='Funnel Plot', description='Title:', style=style, layout=layout)
    w_color = widgets.Dropdown(options=['steelblue', 'darkgreen', 'coral'], value='steelblue', description='Color:', style=style, layout=layout)
    w_show_contours = widgets.Checkbox(value=True, description='Show Contours')
    w_invert_y = widgets.Checkbox(value=True, description='Invert Y-axis')

    ui = widgets.VBox([
        widgets.HTML("<b>FUNNEL PLOT OPTIONS</b>"),
        w_title,
        widgets.HBox([w_width, w_height]),
        w_color,
        w_show_contours,
        w_invert_y
    ])

    out_plot = widgets.Output()
    out_code = widgets.Output()

    def update_view(change=None):
        with out_plot:
            clear_output(wait=True)
            try:
                fig, ax = funnel_plot(
                    effect_sizes=effect_sizes,
                    variances=variances,
                    center=center,
                    figsize=(w_width.value, w_height.value),
                    title=w_title.value,
                    color=w_color.value,
                    show_contours=w_show_contours.value,
                    invert_y=w_invert_y.value
                )
                plt.show(fig)
            except Exception as e:
                print(f"Error: {e}")

        with out_code:
            clear_output()
            code = (
                f"from ecometa.plots import funnel_plot\n\n"
                f"fig, ax = funnel_plot(\n"
                f"    effect_sizes=effect_sizes,\n"
                f"    variances=variances,\n"
                f"    center={center},\n"
                f"    figsize=({w_width.value}, {w_height.value}),\n"
                f"    title='{w_title.value}',\n"
                f"    color='{w_color.value}',\n"
                f"    show_contours={w_show_contours.value}\n"
                f")"
            )
            print(code)

    for widget in [w_width, w_height, w_title, w_color, w_show_contours, w_invert_y]:
        widget.observe(update_view, names='value')

    update_view()
    display(widgets.HBox([ui, widgets.VBox([out_plot, out_code])]))


def trim_and_fill_plot_interactive(effect_sizes, variances, filled_effects=None,
                                   filled_variances=None, center_original=None,
                                   center_adjusted=None):
    """
    Launches an interactive GUI for the Trim-and-Fill Plot.

    Parameters
    ----------
    effect_sizes : array-like
        Original effect sizes
    variances : array-like
        Original variances
    filled_effects : array-like, optional
        Imputed effect sizes
    filled_variances : array-like, optional
        Imputed variances
    center_original : float, optional
        Original pooled effect
    center_adjusted : float, optional
        Adjusted pooled effect
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='95%')

    w_width = widgets.FloatSlider(value=9, min=6, max=12, step=0.5, description='Width (in):', style=style, layout=layout)
    w_height = widgets.FloatSlider(value=8, min=6, max=12, step=0.5, description='Height (in):', style=style, layout=layout)
    w_title = widgets.Text(value='Trim-and-Fill Funnel Plot', description='Title:', style=style, layout=layout)
    w_show_contours = widgets.Checkbox(value=True, description='Show Contours')

    ui = widgets.VBox([
        widgets.HTML("<b>TRIM-AND-FILL OPTIONS</b>"),
        w_title,
        widgets.HBox([w_width, w_height]),
        w_show_contours
    ])

    out_plot = widgets.Output()
    out_code = widgets.Output()

    def update_view(change=None):
        with out_plot:
            clear_output(wait=True)
            try:
                fig, ax = trim_and_fill_plot(
                    effect_sizes=effect_sizes,
                    variances=variances,
                    filled_effects=filled_effects,
                    filled_variances=filled_variances,
                    center_original=center_original,
                    center_adjusted=center_adjusted,
                    figsize=(w_width.value, w_height.value),
                    title=w_title.value,
                    show_contours=w_show_contours.value
                )
                plt.show(fig)
            except Exception as e:
                print(f"Error: {e}")

        with out_code:
            clear_output()
            code = (
                f"from ecometa.plots import trim_and_fill_plot\n\n"
                f"fig, ax = trim_and_fill_plot(\n"
                f"    effect_sizes=effect_sizes,\n"
                f"    variances=variances,\n"
                f"    filled_effects=filled_effects,\n"
                f"    filled_variances=filled_variances,\n"
                f"    center_original={center_original},\n"
                f"    center_adjusted={center_adjusted},\n"
                f"    figsize=({w_width.value}, {w_height.value}),\n"
                f"    title='{w_title.value}'\n"
                f")"
            )
            print(code)

    for widget in [w_width, w_height, w_title, w_show_contours]:
        widget.observe(update_view, names='value')

    update_view()
    display(widgets.HBox([ui, widgets.VBox([out_plot, out_code])]))


def spline_plot_interactive(moderator_values, predictions, ci_lower=None, ci_upper=None,
                           observed_effects=None, observed_moderators=None):
    """
    Launches an interactive GUI for the Spline Plot.

    Parameters
    ----------
    moderator_values : array-like
        Moderator values for prediction
    predictions : array-like
        Predicted effect sizes
    ci_lower : array-like, optional
        Lower CI bounds
    ci_upper : array-like, optional
        Upper CI bounds
    observed_effects : array-like, optional
        Observed effect sizes
    observed_moderators : array-like, optional
        Observed moderator values
    """
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='95%')

    w_width = widgets.FloatSlider(value=10, min=6, max=14, step=0.5, description='Width (in):', style=style, layout=layout)
    w_height = widgets.FloatSlider(value=6, min=4, max=10, step=0.5, description='Height (in):', style=style, layout=layout)
    w_title = widgets.Text(value='Spline Meta-Regression', description='Title:', style=style, layout=layout)
    w_xlabel = widgets.Text(value='Moderator', description='X Label:', style=style, layout=layout)
    w_ylabel = widgets.Text(value='Predicted Effect Size', description='Y Label:', style=style, layout=layout)
    w_show_ci = widgets.Checkbox(value=True, description='Show Confidence Interval')
    w_show_points = widgets.Checkbox(value=True, description='Show Observed Data')

    ui = widgets.VBox([
        widgets.HTML("<b>SPLINE PLOT OPTIONS</b>"),
        w_title,
        widgets.HBox([w_width, w_height]),
        w_xlabel,
        w_ylabel,
        w_show_ci,
        w_show_points
    ])

    out_plot = widgets.Output()
    out_code = widgets.Output()

    def update_view(change=None):
        with out_plot:
            clear_output(wait=True)
            try:
                fig, ax = spline_plot(
                    moderator_values=moderator_values,
                    predictions=predictions,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    observed_effects=observed_effects,
                    observed_moderators=observed_moderators,
                    figsize=(w_width.value, w_height.value),
                    title=w_title.value,
                    xlabel=w_xlabel.value,
                    ylabel=w_ylabel.value,
                    show_ci=w_show_ci.value,
                    show_points=w_show_points.value
                )
                plt.show(fig)
            except Exception as e:
                print(f"Error: {e}")

        with out_code:
            clear_output()
            code = (
                f"from ecometa.plots import spline_plot\n\n"
                f"fig, ax = spline_plot(\n"
                f"    moderator_values=moderator_values,\n"
                f"    predictions=predictions,\n"
                f"    ci_lower=ci_lower,\n"
                f"    ci_upper=ci_upper,\n"
                f"    observed_effects=observed_effects,\n"
                f"    observed_moderators=observed_moderators,\n"
                f"    figsize=({w_width.value}, {w_height.value}),\n"
                f"    title='{w_title.value}',\n"
                f"    xlabel='{w_xlabel.value}',\n"
                f"    ylabel='{w_ylabel.value}',\n"
                f"    show_ci={w_show_ci.value},\n"
                f"    show_points={w_show_points.value}\n"
                f")"
            )
            print(code)

    for widget in [w_width, w_height, w_title, w_xlabel, w_ylabel, w_show_ci, w_show_points]:
        widget.observe(update_view, names='value')

    update_view()
    display(widgets.HBox([ui, widgets.VBox([out_plot, out_code])]))
