import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .models import ThreeLevelMeta

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
                continue # Skip groups that fail to converge

    # Create plotting dataframe
    plot_df = pd.DataFrame(results_list)
    
    # Calculate Prediction Intervals (Trunks)
    # PI_SD = sqrt(SE^2 + tau^2 + sigma^2)
    z_score = 1.96 # Approximation for 95%
    
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
    # Ensure "Overall" is last or first? Usually Overall is separate.
    # Let's put Overall at the top (index 0)
    
    # Map colors
    palette = sns.color_palette(color_palette, n_colors=len(groups))
    color_map = {g: c for g, c in zip(groups, palette)}
    # Force Overall to be black/grey if desired, or keep palette
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
                color='white' if grp != 'Overall' else 'white', 
                linewidth=2, zorder=3) # Inner white line for contrast
        
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
    
    # Invert Y axis so first group is at top? 
    # Standard forest plots go Top->Bottom. 
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig, ax