import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from ..plots import orchard_plot

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
            # Construct the code string
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