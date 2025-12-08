import matplotlib.pyplot as plt
from collections import defaultdict
import scienceplots
import seaborn as sns
import numpy as np
plt.style.use(['science','ieee'])
from tqdm import tqdm
from scipy.stats import truncnorm, norm
from itertools import cycle

from .plant import *

# HELPER FUNCTIONS
def try_clear_output(*args, **kwargs):
    try:
        from IPython.display import clear_output
        clear_output(*args, **kwargs)
    except ImportError:
        pass

def get_original_value(plant, full_key):
    """
    Retrieve the value of a parameter from the plant object.
    Supports both dot-notation attributes and nested dictionaries
    such as variable_opex_inputs.<key>.price
    """
    keys = full_key.split('.')
    ref = plant
    for k in keys:
        if isinstance(ref, dict):
            ref = ref[k]["price"]
        else:
            ref = getattr(ref, k)
    return ref

def update_and_evaluate(plant, factor, value, nested_price_keys):
    """
    Update the plant parameter (top-level or nested price) and 
    recalculate LCOP. Returns the resulting LCOP.
    
    Parameters
    ----------
    plant : object
        The plant object to copy and evaluate.
    factor : str
        Parameter name (e.g., 'fixed_capital', 'variable_opex_inputs.electricity')
    value : float
        New value to plug into the plant model.
    nested_price_keys : list of str
        Keys belonging to variable_opex_inputs.<key>
    """
    plant_copy = deepcopy(plant)

    if factor == 'fixed_capital':
        plant_copy.calculate_fixed_capital(fc=value)

    elif factor == 'fixed_opex':
        plant_copy.calculate_fixed_opex(fp=value)

    elif factor in nested_price_keys:
        parts = factor.split('.')   # ['variable_opex_inputs', '<name>']
        config = {
            'variable_opex_inputs': {
                parts[1]: {
                    'price': value,
                }
            }
        }
        plant_copy.update_configuration(config)

    else:
        # Generic top-level parameter update
        config = {factor: value}
        plant_copy.update_configuration(config)

    plant_copy.calculate_levelized_cost()
    return plant_copy.levelized_cost

# type_costs = defaultdict(float)
# for eq in mp_equipment:
#     type_costs[eq.type] += eq.purchased_cost

# # Convert to sorted list (largest at bottom)
# equip_sorted = sorted(type_costs.items(), key=lambda x: x[1], reverse=True)

# labels = [e[0].replace("&", r"\&") for e in equip_sorted]
# costs  = [e[1] for e in equip_sorted]

# # Generate colors from the Plasma colormap
# colors = sns.color_palette("plasma", n_colors=len(costs))

# fig, ax = plt.subplots(figsize=(2.5, 1.8))

# bottom = 0
# for (label, cost, color) in zip(labels, costs, colors):
#     ax.bar("MP Plant", cost, bottom=bottom, label=label, color=color)
#     bottom += cost

# ax.set_ylabel(r"Purchased cost / [\$]")
# # ax.set_title("Stacked Equipment Cost (largest → smallest)")
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
# plt.tight_layout()
# plt.show()

def sensitivity_plot(plants, parameter, plus_minus_value, n_points=21, figsize=(3.2, 2.2), label=r'Levelized cost of product'):
    """
    Compare sensitivity of multiple plants to the same parameter.

    Parameters
    ----------
    plants : list
        List of plant objects [plant1, plant2, ...].
    parameter : str
        Parameter to vary (same syntax as before, e.g. 'interest_rate' or
        'variable_opex_inputs.electricity', or shorthand like 'co2_tax').
    plus_minus_value : float
        Fractional range for variation (e.g. 0.2 for ±20%).
    n_points : int
        Number of points in the sweep.
    label : str
        Y-axis label.
    """

    # Allow single plant for convenience
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    # Color cycle for multiple plants
    line_colors = cycle(plt.cm.Set2.colors)

    # Use the first plant to define allowed parameters & label map
    first_plant = plants[0]

    top_level_keys = [
        'fixed_capital',
        'fixed_opex',
        'project_lifetime',
        'interest_rate',
        'operator_hourly_rate' 
    ]

    # Get the nested price keys from the first plant
    nested_price_keys_first = [
        f"variable_opex_inputs.{k}" for k in first_plant.variable_opex_inputs.keys()
    ]

    # Collect all the nested price keys from all plants
    nested_price_keys_all = set(nested_price_keys_first)
    for plant in plants[1:]:
        nested_price_keys_all.update([f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs.keys()])

    # Combine the top level keys and all nested price keys
    valid_parameters = set(top_level_keys + list(nested_price_keys_all))

    # --- Allow shorthand input like "co2_tax" instead of full path ---
    short_to_full = {}

    # Collect all shorthand to full mappings across all plants
    for plant in plants:
        short_to_full.update({
            k: f"variable_opex_inputs.{k}"
            for k in plant.variable_opex_inputs.keys()
        })

    # If parameter is shorthand, convert to full path
    if parameter in short_to_full:
        parameter = short_to_full[parameter]

    # Ensure the parameter is valid
    if parameter not in valid_parameters:
        raise ValueError(f"Unrecognized parameter: {parameter}")

    # Generate percent deviations (e.g., -20% to +20%) – same for all plants
    pct_changes = np.linspace(-plus_minus_value, plus_minus_value, n_points)
    pct_axis = pct_changes * 100  # for plotting in %

    # Build cleaner x-label (based on all plants)
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }

    # Collect all unique variable_opex_inputs from all plants
    for plant in plants:
        for var in plant.variable_opex_inputs:
            label_map[f"variable_opex_inputs.{var}"] = f"{var.capitalize()} price"

    label_raw = label_map.get(
        parameter,
        parameter.replace("variable_opex_inputs.", "")
                .replace(".price", "")
    )

    # Apply underscore replacement + capitalization to BOTH cases
    label_clean = label_raw.replace("_", " ").capitalize()
    x_label = label_clean + r" / [$\pm$ \%]"

    plt.figure(figsize=figsize)

    # Loop over plants and plot each sensitivity curve
    for i, plant in enumerate(plants):
        # Rebuild nested_price_keys for this specific plant
        nested_price_keys = [
            f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs.keys()
        ]

        # Valid parameters for THIS plant (includes top-level and nested ones)
        plant_valid_params = set(top_level_keys + nested_price_keys)

        # Get baseline for this plant (LCOP at nominal parameter)
        lcop_base = plant.levelized_cost

        color = next(line_colors)
        plant_label = getattr(plant, "name", f"Plant {i+1}")

        # If this plant does NOT have the parameter ➜ flat horizontal line
        if parameter not in plant_valid_params:
            lcop_values = np.full_like(pct_axis, fill_value=lcop_base, dtype=float)
        else:
            # Get original (baseline) value for this plant
            if parameter in ['fixed_capital', 'fixed_opex']:
                original_value = 1.0
            else:
                original_value = get_original_value(plant, parameter)

            # Convert pct changes into actual parameter values (per plant)
            param_values = original_value * (1 + pct_changes)

            # Compute LCOP for each variation for this plant
            lcop_values = [
                update_and_evaluate(plant, parameter, v, nested_price_keys)
                for v in param_values
            ]

        plt.plot(
            pct_axis,
            lcop_values,
            linewidth=1,
            color=color,
            label=plant_label,
            linestyle='-'
        )

    plt.xlabel(x_label)
    plt.ylabel(label)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()



def tornado_plot(plant, plus_minus_value, figsize=(3.4, 2.4), label=r'Levelized cost of product'):
    """
    Generate a tornado plot to visualize the sensitivity of the levelized cost of product to key input parameters.
    This function performs a one-at-a-time sensitivity analysis on selected parameters in the `config` dictionary,
    varying each parameter by ±`plus_minus_value` (as a fraction, e.g., 0.2 for ±20%) and recalculating the
    levelized cost of product (lcop). The results are displayed as a horizontal bar chart (tornado plot), showing
    the impact of each parameter on the lcop.
    Parameters
    ----------
    config : dict
        Configuration dictionary containing all techno-economic parameters, including top-level keys and nested
        variable OPEX input prices.
    plus_minus_value : float
        Fractional change to apply to each parameter for sensitivity analysis (e.g., 0.2 for ±20%).
    Returns
    -------
    None
        Displays a matplotlib tornado plot. Does not return any value.
    Notes
    -----
    - The function assumes the existence of several calculation functions:
      `calculate_levelized_cost`, `calculate_variable_opex`, `calculate_fixed_opex`,
      `calculate_fixed_capital`, and `calculate_cash_flow`.
    - The tornado plot highlights which parameters have the largest effect on the lcop.
    - The plot uses blue bars for -X% changes and red bars for +X% changes.
    """

    # Top-level keys (assuming these are variable names already defined)
    top_level_keys = [
        'fixed_capital',
        'fixed_opex',
        'project_lifetime',
        'interest_rate',
        'operator_hourly_rate'
    ]

    # Get all nested 'price' keys from variable_opex_inputs
    nested_price_keys = [
        f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs.keys()
    ]

    all_keys = top_level_keys + nested_price_keys

    lcop_base = plant.levelized_cost

    # Perform sensitivity analysis
    sensitivity_results = {}
    for key in all_keys:
        if key in ['fixed_capital', 'fixed_opex']:
            low = (1 - plus_minus_value)
            high = (1 + plus_minus_value)
        else:
            original = get_original_value(plant, key)
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)

        lcop_low = update_and_evaluate(plant, key, low, nested_price_keys)
        lcop_high = update_and_evaluate(plant, key, high, nested_price_keys)
        
        sensitivity_results[key] = [lcop_low, lcop_high]

    # Extract lcop values
    factors = list(sensitivity_results.keys())
    lcop_lows = np.array([sensitivity_results[f][0] for f in factors])
    lcop_highs = np.array([sensitivity_results[f][1] for f in factors])
    total_effects = np.abs(lcop_highs - lcop_lows)

    # Sort by total effect
    sorted_indices = np.argsort(total_effects)
    factors_sorted = [factors[i] for i in sorted_indices]
    lcop_lows_sorted = lcop_lows[sorted_indices]
    lcop_highs_sorted = lcop_highs[sorted_indices]

    # # Prepare bar components
    # bar_left = np.minimum(lcop_lows_sorted, lcop_highs_sorted)
    # bar_width = np.abs(lcop_highs_sorted - lcop_lows_sorted)

    # Assign colors: blue for -X%, red for +X%
    colors_low = ['#87CEEB'] * len(factors_sorted)   # blue for -X%
    colors_high = ['#FF9999'] * len(factors_sorted)  # red for +X%

    # Label mapping
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }
    for var in plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}.price"] = f"{var.capitalize()} price"

    labels_sorted = [label_map.get(f, f.replace("variable_opex_inputs.", "").replace(".price", "").replace("_", " ").capitalize() + " price" if "variable_opex_inputs" in f else f) for f in factors_sorted]
    y_pos = np.arange(len(labels_sorted))

    # Plot
    plt.figure(figsize=figsize)
    for i in range(len(y_pos)):
        # Bar for -X% (blue)
        plt.barh(y_pos[i], abs(lcop_lows_sorted[i] - lcop_base), left=min(lcop_base,lcop_lows_sorted[i]),
                 color=colors_low[i], edgecolor='black', label=r'-{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")
        # Bar for +X% (red)
        plt.barh(y_pos[i], abs(lcop_highs_sorted[i] - lcop_base), left=min(lcop_base,lcop_highs_sorted[i]),
                 color=colors_high[i], edgecolor='black', label=r'+{}\%'.format(int(plus_minus_value * 100)) if i == 0 else "")

    plt.axvline(x=lcop_base, color='black', linestyle='--', linewidth=0.75)
    plt.yticks(y_pos, labels_sorted)
    plt.xlim(min(lcop_lows_sorted) * 0.95, max(lcop_highs_sorted) * 1.05)
    plt.xlabel(label)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def monte_carlo(plant, num_samples: int = 1_000_000, batch_size: int = 1000, 
                show_input_distributions: bool = False, 
                show_plot_updates: bool = False,
                show_final_plot: bool = True,
                figsize=None,
                label=r'Levelized cost of product'):
    
    plant.calculate_fixed_capital()
    plant.calculate_variable_opex()
    plant.calculate_fixed_opex()
    plant.calculate_cash_flow()
    plant.calculate_levelized_cost()

    plant_copy = deepcopy(plant)
    num_samples = int(num_samples)
    num_batches = num_samples // batch_size

    def truncated_normal_samples(mean, std, low, high, size):
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

    # Preallocate arrays
    fixed_capitals = np.zeros(num_samples)
    fixed_opexs = np.zeros(num_samples)
    operator_hourlys = np.zeros(num_samples)
    project_lifetimes = np.zeros(num_samples)
    interests = np.zeros(num_samples)
    lcops = np.zeros(num_samples)

    # Preallocate variable opex price samples
    variable_opex_price_samples = {
        item: np.zeros(num_samples)
        for item in plant_copy.variable_opex_inputs.keys()
    }

    def plot_monte_carlo(lcops, label=label):
        mu_lcop, std_lcop = norm.fit(lcops)
        plt.figure()
        plt.hist(lcops, bins=30, density=True, 
                color='skyblue', edgecolor='black', zorder=2)
        # Plot fitted normal curve
        x = np.linspace(min(lcops), max(lcops), 1000)
        p = norm.pdf(x, mu_lcop, std_lcop)
        plt.plot(x, p, '-', color='indianred', 
                label=fr'$\mu$={mu_lcop:.3g}, $\sigma$={std_lcop:.2e}', zorder=2)
        plt.xlabel(label)
        plt.ylabel("Probability density")
        plt.legend(loc='best', fontsize='x-small')
        plt.show()

    update_interval = num_batches // 10  # Every 1/10 of simulation

    for i in tqdm(range(num_batches), desc="Running Monte Carlo"):
        start = i * batch_size
        end = start + batch_size

        fixed_capitals[start:end] = truncated_normal_samples(
            1, 0.6/2, 0.25, 2, batch_size
        )
        fixed_opexs[start:end] = truncated_normal_samples(
            1, 0.6/2, 0.25, 2, batch_size
        )
        operator_hourlys[start:end] = truncated_normal_samples(
            plant.operator_hourly_rate, 20/2, 10, 100, batch_size
        )
        project_lifetimes[start:end] = truncated_normal_samples(
            plant.project_lifetime, 10/2, 5, 40, batch_size
        )
        interests[start:end] = truncated_normal_samples(
            plant.interest_rate, 0.03, 0.01, 2, batch_size
        )

        # Sample variable opex prices
        for item, props in plant.variable_opex_inputs.items():
            price_mean = props['price']
            price_std = props['price_std']
            price_min = props['price_min']
            price_max = props['price_max']
            variable_opex_price_samples[item][start:end] = truncated_normal_samples(
                price_mean, price_std, price_min, price_max, batch_size
            )

        # Update batch config
        plant_copy.update_configuration({
            'operator_hourly_rate': operator_hourlys[start:end],
            'project_lifetime': project_lifetimes[start:end],
            'interest_rate': interests[start:end],
        })

        # Update variable opex prices in config for this batch
        for item in plant_copy.variable_opex_inputs.keys():
            plant_copy.variable_opex_inputs[item]['price'] = variable_opex_price_samples[item][start:end]

        # Run calculations
        plant_copy.calculate_fixed_capital(fc=fixed_capitals[start:end])
        plant_copy.calculate_variable_opex()
        plant_copy.calculate_fixed_opex(fp=fixed_opexs[start:end])
        plant_copy.calculate_cash_flow()
        plant_copy.calculate_levelized_cost()
        lcops[start:end] = plant_copy.levelized_cost

        # Show live plot every 1/10 of the simulation
        if show_plot_updates:
            if (i + 1) % update_interval == 0 or (i + 1) == num_batches:
                try_clear_output(wait=True)  # Clear previous output
                plot_monte_carlo(lcops[:end])

    # Final plot after all batches
    if show_final_plot:
        try_clear_output(wait=True)
        plot_monte_carlo(lcops)
    plant.monte_carlo_lcops = lcops

    # Plotting
    if show_input_distributions:
        # Collect all input parameter arrays for plotting
        input_distributions = {
            'Fixed capital investment': fixed_capitals,
            'Fixed production costs': fixed_opexs,
            'Operator hourly rate': operator_hourlys,
            'Project lifetime': project_lifetimes,
            'Interest rate': interests,
        }

        # Include variable opex price inputs
        for item, samples in variable_opex_price_samples.items():
            input_distributions[f'{item.capitalize()} price'] = samples

        # Set up subplots: adjust layout depending on number of variables
        n_params = len(input_distributions)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()  # Flatten for easy indexing

        # Plot each distribution
        for idx, (label, data) in enumerate(input_distributions.items()):
            ax = axes[idx]
            # mu, std = norm.fit(data)
            ax.hist(data, bins=50, density=True, color='#FFFFE0', edgecolor='black', zorder=2)
            x = np.linspace(min(data), max(data), 1000)
            # ax.plot(x, norm.pdf(x, a, b, loc=mu, scale=std), 'r-', label=fr'$\mu$={mu:.2e}, $\sigma$={std:.2e}')
            ax.set_xlabel(label)
            # ax.legend(loc='best', fontsize='medium')

        # Turn off any unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        fig.tight_layout()
        plt.show()


def plot_multiple_monte_carlo(plants, bins=30, figsize=None, label=r'Levelized cost of product'):
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    
    # Separate color cycles for histograms and lines
    hist_colors = cycle(plt.cm.tab10.colors)   # e.g., from tab10 colormap
    line_colors = cycle(plt.cm.tab10.colors)    # different palette for lines
    
    for plant in plants:
        if plant.monte_carlo_lcops is not None:
            hist_color = next(hist_colors)
            line_color = next(line_colors)
            
            # Fit normal distribution
            mu_lcop, std_lcop = norm.fit(plant.monte_carlo_lcops)
            
            # Histogram
            plt.hist(
                plant.monte_carlo_lcops,
                bins=bins,
                alpha=0.5,
                density=True,
                edgecolor='black',
                color=hist_color,
                zorder=1,
                label=plant.name
            )
            
            # Normal curve
            x = np.linspace(min(plant.monte_carlo_lcops), max(plant.monte_carlo_lcops), 1000)
            p = norm.pdf(x, mu_lcop, std_lcop)
            plt.plot(
                x, p, '-',
                color=line_color,
                linewidth=1.2,
                label=fr'$\mu$={mu_lcop:.3g}, $\sigma$={std_lcop:.2e}',
                zorder=2
            )
    
    plt.xlabel(label)
    plt.ylabel("Probability density")
    
      # --- Adaptive legend settings ---
    handles, labels = plt.gca().get_legend_handles_labels()
    n_items = len(labels)
    
    # Dynamic number of columns
    if n_items <= 4:
        ncol = 1
    elif n_items <= 6:
        ncol = 3
    else:
        ncol = 4
    
    # For many items, force legend outside plot
    if n_items > 4:
        loc = 'upper center'
        bbox_to_anchor = (0.5, 1.20)  # place below plot
    else:
        loc = 'best'
        bbox_to_anchor = None
    
    plt.legend(
        loc=loc, ncol=ncol, fontsize=4,
        frameon=True, facecolor='white', framealpha=0.6,
        fancybox=True, bbox_to_anchor=bbox_to_anchor
    )

     # Adjust layout if legend outside
    if bbox_to_anchor:
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space on top
    else:
        plt.tight_layout()
    plt.show()