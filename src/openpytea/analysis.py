import matplotlib.pyplot as plt
import scienceplots
import numpy as np
plt.style.use(['science','ieee'])
from tqdm import tqdm
from scipy.stats import truncnorm, norm
from itertools import cycle
from collections.abc import Mapping, Sequence

from .plant import *

# HELPER FUNCTIONS
def make_label(s: str) -> str:
    s = s.replace("_", " ")
    return s[:1].upper() + s[1:]

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

def update_and_evaluate(plant, factor, value, nested_price_keys, metric="LCOP", additional_capex: bool = False):
    """
    Update the plant parameter (top-level or nested price) and 
    recalculate the requested economic metric. Returns that metric.

    Parameters
    ----------
    plant : object
        The plant object to copy and evaluate.
    factor : str
        Parameter name (e.g., 'fixed_capital',
        'operator_hourly_rate',
        'variable_opex_inputs.electricity',
        'plant_products.methanol').
    value : float
        New value to plug into the plant model.
    nested_price_keys : list of str
        Keys belonging to variable_opex_inputs.<key> and/or plant_products.<key>.
    metric : {"LCOP", "ROI", "NPV", "PBT", "IRR"}
        Economic metric to evaluate.
    """
    plant_copy = deepcopy(plant)
    metric = metric.upper()

    # --- 1. Apply parameter change ---

    if factor == 'fixed_capital':
        plant_copy.calculate_fixed_capital(fc=value)

    elif factor == 'fixed_opex':
        plant_copy.calculate_fixed_opex(fp=value)

    elif factor in nested_price_keys:
        # factor can be:
        #   "variable_opex_inputs.<name>"  or
        #   "plant_products.<name>"
        parts = factor.split('.')   # ['variable_opex_inputs' | 'plant_products', '<name>']
        root, name = parts[0], parts[1]

        if root == "variable_opex_inputs":
            config = {
                "variable_opex_inputs": {
                    name: {
                        "price": value,
                    }
                }
            }
        elif root == "plant_products":
            config = {
                "plant_products": {
                    name: {
                        "price": value,
                    }
                }
            }
        else:
            raise ValueError(f"Unsupported nested price root '{root}' in factor '{factor}'.")

        plant_copy.update_configuration(config)

    elif factor == "operator_hourly_rate":
        # Support both dict-style {"rate": ...} and scalar-style operator_hourly_rate
        current = getattr(plant_copy, "operator_hourly_rate", None)
        if isinstance(current, dict):
            config = {
                "operator_hourly_rate": {
                    "rate": value
                }
            }
        else:
            config = {"operator_hourly_rate": value}
        plant_copy.update_configuration(config)

    else:
        # Generic top-level parameter update, e.g. 'interest_rate', 'project_lifetime'
        config = {factor: value}
        plant_copy.update_configuration(config)

    # --- 2. Recompute economics ---

    # This builds fixed_capital, opex, revenue, cash_flow, etc.
    plant_copy.calculate_levelized_cost()

    # --- 3. Return requested metric ---

    if metric == "LCOP":
        return plant_copy.levelized_cost

    elif metric == "ROI":
        plant_copy.calculate_roi(additional_capex=additional_capex)
        return plant_copy.roi

    elif metric == "NPV":
        # With MC-aware calculate_npv this can be scalar or array.
        # In sensitivity/tornado we are effectively in a single-scenario context.
        return plant_copy.calculate_npv()

    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        return plant_copy.calculate_payback_time(additional_capex=additional_capex)
    elif metric == "IRR":
        plant_copy.calculate_irr()
        return plant_copy.irr

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Use 'LCOP', 'ROI', 'NPV', 'PBT', or 'IRR'."
        )

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Mapping, Sequence


def _plot_stacked_bar_from_components(
    components,
    xlabel,
    ylabel: str,
    figsize=(1.2, 1.8),
    pct: bool = False,
):
    """
    Plot one or multiple stacked vertical bars.

    Parameters
    ----------
    components : dict[str, float] or list[dict[str, float]]
        - Single dict  -> one stacked bar.
        - List of dict -> one stacked bar per dict (side-by-side).
    xlabel : str or list[str]
        Label(s) for the x-axis (one per bar if list).
    ylabel : str
        Base label for the y-axis (units are appended automatically).
    pct : bool, default False
        If True, each bar is normalized to 100% (per bar), and the y-axis
        unit becomes [%]. Otherwise, unit is [$].
    """

    # --- Normalize to list of dicts ---
    if isinstance(components, Mapping):
        components_list = [dict(components)]
    else:
        components_list = [dict(c) for c in components]

    if not components_list:
        raise ValueError("No components to plot (empty list).")

    n_bars = len(components_list)

    # --- Normalize xlabels ---
    if isinstance(xlabel, str):
        if n_bars == 1:
            xlabels = [xlabel]
        else:
            # Simple default if a single string is given for many bars
            xlabels = [f"{xlabel} {i+1}" for i in range(n_bars)]
    else:
        xlabels = list(xlabel)
        if len(xlabels) != n_bars:
            raise ValueError("Number of xlabels must match number of bars.")

    # --- Convert values to percentages per bar if requested ---
    if pct:
        converted = []
        for comp in components_list:
            vals = np.array(list(comp.values()), dtype=float)
            total = vals.sum()
            if total == 0:
                raise ValueError("Cannot compute percentages: total value is zero in one bar.")
            factor = 100.0 / total
            converted.append({k: float(v) * factor for k, v in comp.items()})
        components_list = converted
        ylabel = ylabel + r" / [\%]"
    else:
        ylabel = ylabel + r" / [\$]"

    # --- Collect all component names across all bars ---
    all_names = set()
    for comp in components_list:
        all_names.update(comp.keys())

    if not all_names:
        raise ValueError("All component dictionaries are empty.")

    # Sort components by their total contribution across all bars (largest → smallest)
    totals = {}
    for name in all_names:
        total = 0.0
        for comp in components_list:
            total += float(comp.get(name, 0.0))
        totals[name] = total

    names_sorted = sorted(all_names, key=lambda n: totals[n], reverse=True)

    # --- Colormap: consistent color per component across all bars ---
    cmap = plt.cm.plasma
    colors = [cmap(i) for i in np.linspace(0.15, 0.95, len(names_sorted))]
    color_map = dict(zip(names_sorted, colors))

    # --- X positions for bars ---
    x = np.arange(n_bars)  # THIS is the "iterate through x"
    bottoms = np.zeros(n_bars, dtype=float)

    # --- Automatic width adjustment for multiple bars ---
    if isinstance(figsize, (tuple, list)) and len(figsize) == 2:
        base_w, base_h = figsize
    else:
        base_w, base_h = 1.2, 1.8

    auto_width = max(base_w * n_bars, base_w)
    plt.figure(figsize=(auto_width, base_h))

    # --- Draw stacked bars ---
    for name in names_sorted:
        vals = np.array([comp.get(name, 0.0) for comp in components_list], dtype=float)
        if np.allclose(vals, 0.0):
            # This component is zero everywhere; skip
            continue

        # Legend label:
        # Single bar + pct=True → include percentage in legend
        if n_bars == 1 and pct:
            label = rf"{name} ({vals[0]:.1f}\%)"
        else:
            label = name

        plt.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.6,
            color=color_map[name],
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
        bottoms += vals

    plt.ylabel(ylabel)
    plt.xticks(x, xlabels)
    plt.xlim(-0.5, n_bars - 0.5)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
        frameon=False,
    )

    plt.show()


def plot_direct_costs_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
):
    """
    Stacked vertical bar(s) of fixed capital investment per equipment item,
    using eq.direct_cost as the contribution.

    Parameters
    ----------
    plants : Plant or sequence of Plant
        Plant object(s) with `equipment_list`.
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        components = {}
        for eq in plant.equipment_list:
            components[eq.name] = float(eq.direct_cost)

        components_list.append(components)
        xlabels.append(plant.name)

    _plot_stacked_bar_from_components(
        components=components_list,
        xlabel=xlabels,
        ylabel=r"Direct costs",
        figsize=figsize,
        pct=pct,
    )

def plot_fixed_capital_bar(
    plants,
    figsize=(1.2, 1.8),
    additional_capex: bool = False,
    pct: bool = False,
):
    """
    Stacked vertical bar(s) of fixed capital components for one or more plants.

    Uses the component attributes computed in Plant.calculate_fixed_capital().
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        plant.calculate_fixed_capital(fc=None)

        components = {
            "ISBL":                 plant.isbl,
            "OSBL":                 plant.osbl,
            r"Design \& engineering": plant.dne,
            "Contingency":          plant.contigency,
        }

        if additional_capex:

            extra = getattr(plant, "additional_capex_cost", None)

            if extra is None:
                total_extra = 0.0

            elif isinstance(extra, (list, tuple)):
                total_extra = float(sum(extra)) if len(extra) > 0 else 0.0

            else:
                # numeric single value (int, float, numpy scalar, etc.)
                try:
                    total_extra = float(extra)
                except Exception:
                    total_extra = 0.0  # fallback if something unexpected

            # add only if nonzero
            if total_extra != 0:
                components["Additional CAPEX"] = total_extra

        components_list.append(components)
        xlabels.append(plant.name)

    _plot_stacked_bar_from_components(
        components=components_list,
        xlabel=xlabels,
        ylabel=r"Fixed capital investment",
        figsize=figsize,
        pct=pct,
    )

def plot_variable_opex_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
):
    """
    Stacked vertical bar(s) of variable OPEX by input stream.

    Uses plant.variable_opex_inputs, trying 'annual_cost', then 'cost',
    then falls back to consumption * price if available.
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        components = {}

        for name, props in plant.variable_opex_inputs.items():
            if "annual_cost" in props:
                val = props["annual_cost"]
            elif "cost" in props:
                val = props["cost"]
            elif "consumption" in props and "price" in props:
                val = props["consumption"] * props["price"]
            else:
                continue

            # Format label: remove underscores and capitalize first letter
            label = make_label(name)
            components[label] = float(val)

        components_list.append(components)
        xlabels.append(plant.name)

    _plot_stacked_bar_from_components(
        components=components_list,
        xlabel=xlabels,
        ylabel=r"Annual variable OPEX",
        figsize=figsize,
        pct=pct,
    )

def plot_fixed_opex_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
):
    """
    Stacked vertical bar(s) of fixed OPEX components for one or more plants.

    Uses the component attributes computed in Plant.calculate_fixed_opex().
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        plant.calculate_fixed_opex(fp=None)

        components = {
            "Operating labor":                 plant.operating_labor_costs,
            "Supervision":                     plant.supervision_costs,
            "Direct salary overhead":          plant.direct_salary_overhead,
            "Laboratory charges":              plant.laboratory_charges,
            "Maintenance":                     plant.maintenance_costs,
            r"Taxes \& insurance":             plant.taxes_insurance_costs,
            "Rent of land":                    plant.rent_of_land_costs,
            "Environmental charges":           plant.environmental_charges,
            "Operating supplies":              plant.operating_supplies,
            "General plant overhead":          plant.general_plant_overhead,
            "Interest on working capital":     plant.interest_working_capital,
            r"Patents \& royalties":           plant.patents_royalties,
            r"Distribution \& selling":        plant.distribution_selling_costs,
            r"R\&D":                           plant.RnD_costs,
        }

        components_list.append(components)
        xlabels.append(plant.name)

    _plot_stacked_bar_from_components(
        components=components_list,
        xlabel=xlabels,
        ylabel=r"Annual fixed OPEX",
        figsize=figsize,
        pct=pct,
    )


def sensitivity_plot(
    plants,
    parameter,
    plus_minus_value,
    n_points=21,
    figsize=(3.2, 2.2),
    metric="LCOP",
    label=None,
    additional_capex: bool = False,
):
    """
    Compare sensitivity of multiple plants to the same parameter.
    """

    # Normalize plants input
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    metric = metric.upper()

    # Default y-axis labels if not provided
    if label is None:
        if metric == "LCOP":
            label = r"Levelized cost of product"
        elif metric == "ROI":
            label = r"Return on investment [\%]"
        elif metric == "NPV":
            label = r"Net present value [\$]"
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            label = "Payback time [years]"
        elif metric == "IRR":
            label = "Internal rate of return [-]"
        else:
            label = metric

    # Color cycle for multiple plants
    line_colors = cycle(plt.cm.Set2.colors)

    # True top-level scalar/factor keys
    top_level_keys = [
        "fixed_capital",
        "fixed_opex",
        "project_lifetime",
        "interest_rate",
        "operator_hourly_rate",
    ]

    # --- Nested price keys across ALL plants ---

    # variable_opex_inputs.<name> across plants
    var_opex_keys_all = set(
        f"variable_opex_inputs.{k}"
        for plant in plants
        for k in plant.variable_opex_inputs
    )

    # plant_products.<name> across plants
    product_keys_all = set(
        f"plant_products.{k}"
        for plant in plants
        for k in plant.plant_products
    )

    # Byproducts across plants = plant_products excluding the first key per plant
    byproduct_keys_all = set()
    for plant in plants:
        prod_keys = list(plant.plant_products.keys())  # preserves order
        for k in prod_keys[1:]:  # everything except the first = byproducts
            byproduct_keys_all.add(f"plant_products.{k}")

    # Choose nested keys depending on metric
    if metric == "LCOP":
        # LCOP: cost-side prices + byproducts
        nested_price_keys_all = var_opex_keys_all.union(byproduct_keys_all)
    else:
        # Other metrics: cost-side + all products
        nested_price_keys_all = var_opex_keys_all.union(product_keys_all)

    valid_parameters = set(top_level_keys).union(nested_price_keys_all)

    # --- Allow shorthand input like "co2_tax" instead of full path ---
    short_to_full = {}

    for plant in plants:
        # cost-side
        for k in plant.variable_opex_inputs:
            full = f"variable_opex_inputs.{k}"
            if k in short_to_full and short_to_full[k] != full:
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'. Please use full path."
                )
            short_to_full[k] = full

        # revenue-side (includes byproducts too)
        for k in plant.plant_products:
            full = f"plant_products.{k}"
            if k in short_to_full and short_to_full[k] != full:
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'. Please use full path."
                )
            short_to_full[k] = full

    # If parameter is shorthand, convert to full path
    parameter = short_to_full.get(parameter, parameter)

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

    for plant in plants:
        for var in plant.variable_opex_inputs:
            label_map[f"variable_opex_inputs.{var}"] = f"{make_label(var)} price"
        for prod in plant.plant_products:
            label_map[f"plant_products.{prod}"] = f"{make_label(prod)} price"

    label_raw = label_map.get(
        parameter,
        parameter.replace("variable_opex_inputs.", "")
        .replace("plant_products.", "")
        .replace(".price", ""),
    )

    label_clean = make_label(label_raw)
    x_label = label_clean + r" / [$\pm$ \%]"

    plt.figure(figsize=figsize)

    # Loop over plants and plot each sensitivity curve
    for i, plant in enumerate(plants):
        # Nested price keys for THIS plant
        var_opex_keys = set(f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs)

        prod_key_list = list(plant.plant_products.keys())  # preserves order
        all_prod_keys = set(f"plant_products.{k}" for k in prod_key_list)
        byprod_keys = set(f"plant_products.{k}" for k in prod_key_list[1:])  # exclude first

        if metric == "LCOP":
            nested_price_keys = var_opex_keys.union(byprod_keys)
        else:
            nested_price_keys = var_opex_keys.union(all_prod_keys)

        plant_valid_params = set(top_level_keys).union(nested_price_keys)

        # Baseline value for this metric
        if metric == "LCOP":
            if not hasattr(plant, "levelized_cost"):
                plant.calculate_levelized_cost()
            base_value = plant.levelized_cost
        elif metric == "ROI":
            plant.calculate_levelized_cost()
            plant.calculate_roi(additional_capex=additional_capex)
            base_value = plant.roi
        elif metric == "NPV":
            plant.calculate_levelized_cost()
            base_value = plant.calculate_npv()
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            plant.calculate_levelized_cost()
            base_value = plant.calculate_payback_time(additional_capex=additional_capex)
        elif metric == "IRR":
            plant.calculate_levelized_cost()
            plant.calculate_irr()
            base_value = plant.irr
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")

        color = next(line_colors)
        plant_label = getattr(plant, "name", f"Plant {i+1}")

        if parameter not in plant_valid_params:
            metric_values = np.full_like(pct_axis, fill_value=base_value, dtype=float)
        else:
            if parameter in ["fixed_capital", "fixed_opex"]:
                original_value = 1.0
            else:
                original_value = get_original_value(plant, parameter)

            param_values = original_value * (1 + pct_changes)

            metric_values = [
                update_and_evaluate(
                    plant,
                    parameter,
                    v,
                    list(nested_price_keys),
                    metric=metric,
                    additional_capex=additional_capex,
                )
                for v in param_values
            ]

        plt.plot(
            pct_axis,
            metric_values,
            linewidth=1,
            color=color,
            label=plant_label,
            linestyle="-",
        )

    plt.xlabel(x_label)
    plt.ylabel(label)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def tornado_plot(
    plant,
    plus_minus_value,
    metric="LCOP",
    figsize=(3.4, 2.4),
    label=None,
):
    """
    Generate a tornado plot to visualize the sensitivity of a chosen
    economic metric to key input parameters.

    Parameters
    ----------
    plant : Plant
        A single Plant object.
    plus_minus_value : float
        Fractional change to apply to each parameter for sensitivity
        analysis (e.g., 0.2 for ±20%).
    metric : {"LCOP", "ROI", "NPV", "PBT", "IRR"}
        Economic metric to evaluate and display on the x-axis.
    figsize : tuple
        Matplotlib figure size.
    label : str or None
        X-axis label; if None, a default based on `metric` is used.
    """

    metric = metric.upper()

    # Default x-axis labels if not provided
    if label is None:
        if metric == "LCOP":
            label = r"Levelized cost of product"
        elif metric == "ROI":
            label = r"Return on investment [\%]"
        elif metric == "NPV":
            label = r"Net present value [\$]"
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            label = "Payback time [years]"
        elif metric == "IRR":
            label = "Internal rate of return [-]"
        else:
            label = metric

    # --- Keys to vary (true top-level scalars/factors) ---
    top_level_keys = [
        'fixed_capital',
        'fixed_opex',
        'project_lifetime',
        'interest_rate',
        'operator_hourly_rate',
    ]

    # Nested price keys: variable OPEX + product prices
    var_opex_price_keys = [
        f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs.keys()
    ]
    product_price_keys = [
        f"plant_products.{k}" for k in plant.plant_products.keys()
    ]

    # For LCOP we typically don't include product prices (revenue-only)
    if metric == "LCOP":
        nested_price_keys = var_opex_price_keys
    else:
        nested_price_keys = var_opex_price_keys + product_price_keys

    all_keys = top_level_keys + nested_price_keys

    # --- Baseline value for the selected metric ---
    if metric == "LCOP":
        plant.calculate_levelized_cost()
        base_value = plant.levelized_cost
    elif metric == "ROI":
        plant.calculate_levelized_cost()
        plant.calculate_roi()
        base_value = plant.roi
    elif metric == "NPV":
        plant.calculate_levelized_cost()
        base_value = plant.calculate_npv()
    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        plant.calculate_levelized_cost()
        base_value = plant.calculate_payback_time()
    elif metric == "IRR":
        plant.calculate_levelized_cost()
        plant.calculate_irr()
        base_value = plant.irr
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")

    # --- Sensitivity analysis: low / high for each parameter ---
    sensitivity_results = {}
    for key in all_keys:
        if key in ['fixed_capital', 'fixed_opex']:
            # These are treated as multiplicative factors
            low = (1 - plus_minus_value)
            high = (1 + plus_minus_value)

        elif key == "operator_hourly_rate":
            # Support dict-style {"rate": ...} and scalar-style
            current = getattr(plant, "operator_hourly_rate", None)
            if isinstance(current, dict):
                original = current.get("rate", 0.0)
            else:
                original = 0.0 if current is None else float(current)
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)

        else:
            # For:
            #   - interest_rate
            #   - project_lifetime
            #   - variable_opex_inputs.<name>
            #   - plant_products.<name>
            # get_original_value will return the current scalar (price, etc.)
            original = get_original_value(plant, key)
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)

        metric_low = update_and_evaluate(
            plant, key, low, nested_price_keys, metric=metric
        )
        metric_high = update_and_evaluate(
            plant, key, high, nested_price_keys, metric=metric
        )

        sensitivity_results[key] = [metric_low, metric_high]

    # Extract values
    factors = list(sensitivity_results.keys())
    lows = np.array([sensitivity_results[f][0] for f in factors])
    highs = np.array([sensitivity_results[f][1] for f in factors])
    total_effects = np.abs(highs - lows)

    # Sort by total effect (small → large, so most sensitive at top)
    sorted_indices = np.argsort(total_effects)
    factors_sorted = [factors[i] for i in sorted_indices]
    lows_sorted = lows[sorted_indices]
    highs_sorted = highs[sorted_indices]

    # Colors: blue for -X%, red for +X%
    colors_low = ['#87CEEB'] * len(factors_sorted)   # blue
    colors_high = ['#FF9999'] * len(factors_sorted)  # red

    # --- Label mapping for pretty y-axis names ---
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }
    for var in plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}"] = f"{make_label(var)} price"

    for prod in plant.plant_products:
        label_map[f"plant_products.{prod}"] = f"{make_label(prod)} price"
    labels_sorted = [
        label_map.get(f, make_label(f))
        for f in factors_sorted
    ]

    y_pos = np.arange(len(labels_sorted))

    # --- Plot ---
    plt.figure(figsize=figsize)
    for i in range(len(y_pos)):
        low_val = lows_sorted[i]
        high_val = highs_sorted[i]

        # Bar for -X% (blue)
        plt.barh(
            y_pos[i],
            abs(low_val - base_value),
            left=min(base_value, low_val),
            color=colors_low[i],
            edgecolor='black',
            label=r'-{}\%'.format(int(plus_minus_value * 100)) if i == 0 else ""
        )
        # Bar for +X% (red)
        plt.barh(
            y_pos[i],
            abs(high_val - base_value),
            left=min(base_value, high_val),
            color=colors_high[i],
            edgecolor='black',
            label=r'+{}\%'.format(int(plus_minus_value * 100)) if i == 0 else ""
        )

    plt.axvline(x=base_value, color='black', linestyle='--', linewidth=0.75)
    plt.yticks(y_pos, labels_sorted)

    # Combine all relevant x-values
    x_all = np.concatenate([
        lows_sorted,
        highs_sorted,
        np.atleast_1d(base_value)
    ])
    xmin, xmax = x_all.min(), x_all.max()

    # --- Symmetric padding based on the total span ---
    if xmin == xmax:
        # Degenerate case: all values the same
        pad = 0.05 * (1.0 if xmax == 0 else abs(xmax))
        left, right = xmin - pad, xmax + pad
    else:
        span = xmax - xmin
        pad_frac = 0.05   # 5% of the span on each side
        pad = pad_frac * span
        left = xmin - pad
        right = xmax + pad

    # (Optional) ensure zero is included if you care about it:
    # left = min(left, 0)
    # right = max(right, 0)

    plt.xlim(left, right)

    plt.xlabel(label)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def truncated_normal_samples(mean, std, low, high, size):
    a, b = (low - mean) / std, (high - mean) / std

    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def monte_carlo(
    plant,
    num_samples: int = 1_000_000,
    batch_size: int = 1000,
    additional_capex: bool = False,
):
    """
    Run Monte Carlo on a Plant object and compute ALL metrics (LCOP, ROI, NPV, PBT).
    Returns results on plant.monte_carlo_metrics.
    """

    # Ensure plant is baseline-initialized
    plant.calculate_fixed_capital()
    plant.calculate_variable_opex()
    plant.calculate_fixed_opex()
    plant.calculate_cash_flow()
    plant.calculate_levelized_cost()

    plant_copy = deepcopy(plant)
    num_batches = num_samples // batch_size

    # ---- Allocate arrays for ALL metrics ----
    mc_metrics = {
        "LCOP": np.zeros(num_samples),
        "ROI":  np.zeros(num_samples),
        "NPV":  np.zeros(num_samples),
        "PBT":  np.zeros(num_samples),
    }

    # ---- Allocate all input distributions (same as before) ----
    op_cfg = plant.operator_hourly_rate
    op_mean = op_cfg.get("rate", 38.11)
    op_std  = op_cfg.get("std", 20/2)
    op_min  = op_cfg.get("min", 10)
    op_max  = op_cfg.get("max", 100)

    fixed_capitals = np.zeros(num_samples)
    fixed_opexs = np.zeros(num_samples)
    operator_hourlys = np.zeros(num_samples)
    project_lifetimes = np.zeros(num_samples)
    interests = np.zeros(num_samples)

    variable_opex_price_samples = {
        item: np.zeros(num_samples)
        for item in plant.variable_opex_inputs
    }

    # Product revenues only needed for ROI, NPV, PBT
    have_product_prices = all(
        "price" in props for props in plant.plant_products.values()
    )

    product_price_samples = {
        prod: np.zeros(num_samples)
        for prod in plant.plant_products
    } if have_product_prices else {}

    # ---- Sampling loop ----
    for b in tqdm(range(num_batches), desc="Monte Carlo"):
        start = b * batch_size
        end = start + batch_size

        # ---- Sample inputs ----
        fixed_capitals[start:end] = truncated_normal_samples(1, 0.3, 0.25, 1.75, batch_size)
        fixed_opexs[start:end]    = truncated_normal_samples(1, 0.3, 0.25, 1.75, batch_size)
        operator_hourlys[start:end] = truncated_normal_samples(op_mean, op_std, op_min, op_max, batch_size)
        project_lifetimes[start:end] = truncated_normal_samples(plant.project_lifetime, 5, max(5,plant.project_lifetime-2*5), plant.project_lifetime+2*5, batch_size)
        interests[start:end] = truncated_normal_samples(plant.interest_rate, 0.03, max(0.02,plant.interest_rate-2*0.03), plant.interest_rate+2*0.03, batch_size)

        for item, props in plant.variable_opex_inputs.items():
            variable_opex_price_samples[item][start:end] = truncated_normal_samples(
                props["price"], props["std"], props["min"], props["max"], batch_size
            )

        if have_product_prices:
            for prod, props in plant.plant_products.items():
                product_price_samples[prod][start:end] = truncated_normal_samples(
                    props["price"], props["std"], props["min"], props["max"], batch_size
                )

        # ---- Apply sampled inputs ----
        plant_copy.operator_hourly_rate["rate"] = operator_hourlys[start:end]
        plant_copy.update_configuration({
            "project_lifetime": project_lifetimes[start:end],
            "interest_rate": interests[start:end],
        })

        for item in plant.variable_opex_inputs:
            plant_copy.variable_opex_inputs[item]["price"] = variable_opex_price_samples[item][start:end]

        if have_product_prices:
            for prod in plant.plant_products:
                plant_copy.plant_products[prod]["price"] = product_price_samples[prod][start:end]

        # ---- Economic calculations ----
        plant_copy.calculate_fixed_capital(fc=fixed_capitals[start:end])
        plant_copy.calculate_variable_opex()
        plant_copy.calculate_fixed_opex(fp=fixed_opexs[start:end])
        plant_copy.calculate_cash_flow()
        plant_copy.calculate_levelized_cost()

        # ---- Store LCOP always ----
        mc_metrics["LCOP"][start:end] = plant_copy.levelized_cost

        # ---- If revenue available, compute all other metrics ----
        if have_product_prices:
            mc_metrics["ROI"][start:end] = plant_copy.calculate_roi(additional_capex=additional_capex)
            mc_metrics["NPV"][start:end] = plant_copy.calculate_npv()
            mc_metrics["PBT"][start:end] = plant_copy.calculate_payback_time(additional_capex=additional_capex)

    # ---- Store all results on plant ----
    plant.monte_carlo_metrics = mc_metrics
    plant.monte_carlo_inputs = {
        "Fixed capital factor": fixed_capitals,
        "Fixed opex factor": fixed_opexs,
        "Operator hourly rate": operator_hourlys,
        "Project lifetime": project_lifetimes,
        "Interest rate": interests,
        **{f"{k} price": v for k, v in variable_opex_price_samples.items()},
        **{f"{k} product price": v for k, v in product_price_samples.items()},
    }

    return mc_metrics, plant.monte_carlo_inputs

def default_metric_label(metric: str) -> str:
    metric = metric.upper()
    if metric == "LCOP":
        return r"Levelized cost of product"
    elif metric == "ROI":
        return r"Return on investment [\%]"
    elif metric == "NPV":
        return r"Net present value [\$]"
    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        return "Payback time [years]"
    elif metric == "IRR":
        return "Internal rate of return [-]"
    return metric

def plot_monte_carlo(
    plant,
    metric: str = None,
    bins: int = 30,
    label: str | None = None,
    show: bool = True,
):
    """
    Plot MC distribution from:
    - a Plant object with .monte_carlo_metrics, or
    - a NumPy array of samples (then use `metric` just for labeling)
    """

    # --- Accept both Plant and array ---
    if hasattr(plant, "monte_carlo_metrics"):
        # Using a Plant: pick metric from the dict
        if metric is None:
            metric = "LCOP"  # default if not specified
        metric = metric.upper()

        if metric not in plant.monte_carlo_metrics:
            available = ", ".join(plant.monte_carlo_metrics.keys())
            raise ValueError(
                f"Metric '{metric}' not found in plant.monte_carlo_metrics. "
                f"Available metrics: {available}"
            )

        values = plant.monte_carlo_metrics[metric]

    else:
        # Using raw array-like input
        values = np.asarray(plant)
        if metric is None:
            metric = "LCOP"  # fallback for label purposes

    if label is None:
        label = default_metric_label(metric)

    mu_val, std_val = norm.fit(values)

    hist_colors = cycle(plt.cm.tab10.colors)
    line_colors = cycle(plt.cm.tab10.colors)
    hist_color = next(hist_colors)
    line_color = next(line_colors)

    plt.figure()
    plt.hist(
        values,
        bins=bins,
        density=True,
        color=hist_color,
        edgecolor='black',
        alpha=0.6,
        zorder=1,
        label="Samples",
    )

    x = np.linspace(values.min(), values.max(), 1000)
    p = norm.pdf(x, mu_val, std_val)
    plt.plot(
        x, p, '-',
        label=fr'$\mu$={mu_val:.3g}, $\sigma$={std_val:.2e}',
        color=line_color,
        zorder=2
    )

    plt.xlabel(label)
    plt.ylabel("Probability density")
    plt.legend(loc='best', fontsize='x-small')

    if show:
        plt.show()

def plot_monte_carlo_inputs(
    plant,
    figsize=None,
    bins: int = 50,
):
    """
    Plot input distributions from:
    - a Plant object holding .monte_carlo_inputs, or
    - a raw {label: array} dictionary
    """

    hist_colors = cycle(plt.cm.tab10.colors)
    hist_color = next(hist_colors)
    hist_color = next(hist_colors)

    if hasattr(plant, "monte_carlo_inputs"):
        inputs = plant.monte_carlo_inputs
    else:
        inputs = plant  # assume dict

    n_params = len(inputs)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (label, arr) in enumerate(inputs.items()):
        ax = axes[idx]
        ax.hist(
            arr,
            bins=bins,
            density=True,
            color=hist_color,
            edgecolor='black',
            alpha=0.7
        )
        ax.set_title(label, fontsize=9)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    plt.show()

def plot_multiple_monte_carlo(
    plants,
    metric="LCOP",
    bins=30,
    figsize=None,
    label=None
):
    """
    Compare multiple plants on a chosen Monte Carlo metric.
    """

    metric = metric.upper()

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    # Colors
    hist_colors = cycle(plt.cm.tab10.colors)
    line_colors = cycle(plt.cm.tab10.colors)

    for plant in plants:
        if hasattr(plant, "monte_carlo_metrics") and metric in plant.monte_carlo_metrics:

            values = plant.monte_carlo_metrics[metric]
            hist_color = next(hist_colors)
            line_color = next(line_colors)

            # Fit distribution
            mu, std = norm.fit(values)

            # Histogram
            plt.hist(
                values,
                bins=bins,
                alpha=0.5,
                density=True,
                edgecolor='black',
                color=hist_color,
                zorder=1,
                label=plant.name
            )

            # Normal curve
            x = np.linspace(values.min(), values.max(), 1000)
            p = norm.pdf(x, mu, std)

            plt.plot(
                x, p, '-',
                color=line_color,
                linewidth=1.2,
                label=fr'$\mu$={mu:.3g}, $\sigma$={std:.2e}',
                zorder=2
            )

    # Axis labels
    if label is None:
        label = default_metric_label(metric)

    plt.xlabel(label)
    plt.ylabel("Probability density")

    # Legend formatting
    handles, labels_list = plt.gca().get_legend_handles_labels()
    n_items = len(labels_list)

    if n_items <= 4:
        ncol = 1
        loc = 'best'
        bbox_to_anchor = None
    elif n_items <= 6:
        ncol = 3
        loc = 'upper center'
        bbox_to_anchor = (0.5, 1.15)
    else:
        ncol = 4
        loc = 'upper center'
        bbox_to_anchor = (0.5, 1.20)

    plt.legend(
        loc=loc, ncol=ncol, fontsize=4,
        frameon=True, facecolor='white', framealpha=0.6,
        fancybox=True, bbox_to_anchor=bbox_to_anchor
    )

    if bbox_to_anchor:
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()

    plt.show()

# def monte_carlo(
#     plant,
#     num_samples: int = 1_000_000,
#     batch_size: int = 1000, 
#     show_input_distributions: bool = False, 
#     show_plot_updates: bool = False,
#     show_final_plot: bool = True,
#     bins: int = 30,
#     figsize=None,
#     metric: str = "LCOP",
#     label: str | None = None,
# ):
#     """
#     Monte Carlo simulation on a Plant object.

#     Parameters
#     ----------
#     plant : Plant
#         Plant instance to evaluate.
#     num_samples : int
#         Total number of Monte Carlo samples.
#     batch_size : int
#         Number of samples per batch.
#     show_input_distributions : bool
#         If True, also plot histograms of the sampled input parameters.
#     show_plot_updates : bool
#         If True, update the output distribution plot as batches complete.
#     show_final_plot : bool
#         If True, show the final output distribution plot at the end.
#     figsize : tuple or None
#         Figure size for input-distribution plots.
#     metric : {"LCOP", "ROI", "NPV", "PBT", "IRR"}
#         Economic metric to sample.
#     label : str or None
#         Y-axis label for the metric histogram. If None, a default based on
#         `metric` is used.
#     """

#     # Ensure baseline so the plant is fully initialized
#     plant.calculate_fixed_capital()
#     plant.calculate_variable_opex()
#     plant.calculate_fixed_opex()
#     plant.calculate_cash_flow()
#     plant.calculate_levelized_cost()

#     plant_copy = deepcopy(plant)
#     num_samples = int(num_samples)
#     num_batches = num_samples // batch_size

#     metric = metric.upper()

#     # Default label based on metric if not provided
#     if label is None:
#         if metric == "LCOP":
#             label = r"Levelized cost of product"
#         elif metric == "ROI":
#             label = r"Return on investment [\%]"
#         elif metric == "NPV":
#             label = r"Net present value [\$]"
#         elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
#             label = "Payback time [years]"
#         elif metric == "IRR":
#             label = "Internal rate of return [-]"
#         else:
#             label = metric

#     def truncated_normal_samples(mean, std, low, high, size):
#         a, b = (low - mean) / std, (high - mean) / std
#         return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

#     # --- Operator hourly rate distribution (always dict) ---
#     op_cfg = plant.operator_hourly_rate
#     op_mean = op_cfg.get("rate", 38.11)
#     op_std  = op_cfg.get("std", 20 / 2)
#     op_min  = op_cfg.get("min", 10)
#     op_max  = op_cfg.get("max", 100)

#     # Preallocate arrays for inputs
#     fixed_capitals = np.zeros(num_samples)
#     fixed_opexs = np.zeros(num_samples)
#     operator_hourlys = np.zeros(num_samples)
#     project_lifetimes = np.zeros(num_samples)
#     interests = np.zeros(num_samples)

#     # Preallocate array for the chosen metric
#     metric_values = np.zeros(num_samples)

#     # Preallocate variable opex price samples
#     variable_opex_price_samples = {
#         item: np.zeros(num_samples)
#         for item in plant_copy.variable_opex_inputs.keys()
#     }

#     # Preallocate product price samples (from plant_products)
#     product_price_samples = {
#         prod: np.zeros(num_samples)
#         for prod in plant_copy.plant_products.keys()
#     }

#     def plot_monte_carlo(values, label=label):
#         mu_val, std_val = norm.fit(values)
#         plt.figure()
#         plt.hist(
#             values,
#             bins=bins,
#             density=True,
#             color='skyblue',
#             edgecolor='black',
#             zorder=2,
#         )
#         # Plot fitted normal curve
#         x = np.linspace(min(values), max(values), 1000)
#         p = norm.pdf(x, mu_val, std_val)
#         plt.plot(
#             x,
#             p,
#             '-',
#             color='indianred',
#             label=fr'$\mu$={mu_val:.3g}, $\sigma$={std_val:.2e}',
#             zorder=2,
#         )
#         plt.xlabel(label)
#         plt.ylabel("Probability density")
#         plt.legend(loc='best', fontsize='x-small')
#         plt.show()

#     update_interval = max(num_batches // 10, 1)  # Every ~1/10 of simulation

#     for i in tqdm(range(num_batches), desc="Running Monte Carlo"):
#         start = i * batch_size
#         end = start + batch_size

#         # --- Sample uncertain inputs ---
#         fixed_capitals[start:end] = truncated_normal_samples(
#             1, 0.6 / 2, 0.25, 2, batch_size
#         )
#         fixed_opexs[start:end] = truncated_normal_samples(
#             1, 0.6 / 2, 0.25, 2, batch_size
#         )
#         operator_hourlys[start:end] = truncated_normal_samples(
#             op_mean, op_std, op_min, op_max, batch_size
#         )
#         project_lifetimes[start:end] = truncated_normal_samples(
#             plant.project_lifetime, 10 / 2, 5, 40, batch_size
#         )
#         interests[start:end] = truncated_normal_samples(
#             plant.interest_rate, 0.03, 0.01, 2, batch_size
#         )

#         # Sample variable opex prices
#         for item, props in plant.variable_opex_inputs.items():
#             price_mean = props['price']
#             price_std = props['std']
#             price_min = props['min']
#             price_max = props['max']
#             variable_opex_price_samples[item][start:end] = truncated_normal_samples(
#                 price_mean, price_std, price_min, price_max, batch_size
#             )

#         # Sample product prices (from plant_products) only if metric uses revenue
#         if metric != "LCOP":
#             for prod, props in plant.plant_products.items():
#                 price_mean = props['price']
#                 price_std = props['std']
#                 price_min = props['min']
#                 price_max = props['max']
#                 product_price_samples[prod][start:end] = truncated_normal_samples(
#                     price_mean, price_std, price_min, price_max, batch_size
#                 )

#         # --- Update configuration for this batch ---

#         # Operator hourly rate: dict with 'rate' field
#         plant_copy.operator_hourly_rate["rate"] = operator_hourlys[start:end]

#         plant_copy.update_configuration({
#             'project_lifetime': project_lifetimes[start:end],
#             'interest_rate': interests[start:end],
#         })

#         # Update variable opex prices in config for this batch
#         for item in plant_copy.variable_opex_inputs.keys():
#             plant_copy.variable_opex_inputs[item]['price'] = (
#                 variable_opex_price_samples[item][start:end]
#             )

#         # Update product prices in config for this batch
#         for prod in plant_copy.plant_products.keys():
#             plant_copy.plant_products[prod]['price'] = (
#                 product_price_samples[prod][start:end]
#             )

#         # --- Run economic calculations for this batch ---
#         plant_copy.calculate_fixed_capital(fc=fixed_capitals[start:end])
#         plant_copy.calculate_variable_opex()
#         plant_copy.calculate_fixed_opex(fp=fixed_opexs[start:end])
#         plant_copy.calculate_cash_flow()
#         plant_copy.calculate_levelized_cost()

#         # --- Evaluate the chosen metric (scalar → broadcast over batch) ---
#         if metric == "LCOP":
#             metric_batch = plant_copy.levelized_cost
#         elif metric == "ROI":
#             plant_copy.calculate_roi()
#             metric_batch = plant_copy.roi
#         elif metric == "NPV":
#             metric_batch = plant_copy.calculate_npv()
#         elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
#             metric_batch = plant_copy.calculate_payback_time()
#         elif metric == "IRR":
#             plant_copy.calculate_irr()
#             metric_batch = plant_copy.irr
#         else:
#             raise ValueError(
#                 "Unsupported metric '{metric}'. Use 'LCOP', 'ROI', 'NPV', 'PBT', or 'IRR'."
#             )

#         # Broadcast metric value over this batch slice
#         metric_values[start:end] = metric_batch

#         # --- Optional live plotting ---
#         if show_plot_updates:
#             if (i + 1) % update_interval == 0 or (i + 1) == num_batches:
#                 try_clear_output(wait=True)
#                 plot_monte_carlo(metric_values[:end])

#     # --- Final plot after all batches ---
#     if show_final_plot:
#         try_clear_output(wait=True)
#         plot_monte_carlo(metric_values)

#     # Store results on the plant object
#     plant.monte_carlo_results = metric_values
#     plant.monte_carlo_metric = metric

#     # --- Optional: input distributions ---
#     if show_input_distributions:
#         input_distributions = {
#             'Fixed capital investment factor': fixed_capitals,
#             'Fixed production cost factor': fixed_opexs,
#             'Operator hourly rate': operator_hourlys,
#             'Project lifetime': project_lifetimes,
#             'Interest rate': interests,
#         }

#         for item, samples in variable_opex_price_samples.items():
#             input_distributions[f'{item.capitalize()} price'] = samples

#         if metric != "LCOP":
#             for prod, samples in product_price_samples.items():
#                 input_distributions[f'{prod.capitalize()} price (product)'] = samples

#         n_params = len(input_distributions)
#         n_cols = 3
#         n_rows = (n_params + n_cols - 1) // n_cols
#         if figsize is None:
#             figsize = (n_cols * 5, n_rows * 3)
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
#         axes = axes.flatten()

#         for idx, (lab, data) in enumerate(input_distributions.items()):
#             ax = axes[idx]
#             ax.hist(
#                 data,
#                 bins=50,
#                 density=True,
#                 color='#FFFFE0',
#                 edgecolor='black',
#                 zorder=2,
#             )
#             x = np.linspace(min(data), max(data), 1000)
#             ax.set_xlabel(lab)

#         for i in range(n_params, len(axes)):
#             axes[i].axis('off')

#         fig.tight_layout()
#         plt.show()

# def monte_carlo(
#     plant,
#     num_samples: int = 1_000_000,
#     batch_size: int = 1000, 
#     show_input_distributions: bool = False, 
#     show_final_plot: bool = True,
#     bins: int = 30,
#     figsize=None,
#     metric: str = "LCOP",
#     label: str | None = None,
# ):
#     """
#     Backwards-compatible convenience wrapper:
#     - runs Monte Carlo
#     - optionally plots output + input distributions
#     """

#     metric_values, input_distributions = monte_carlo_sampling(
#         plant=plant,
#         num_samples=num_samples,
#         batch_size=batch_size,
#         metric=metric,
#     )
#     # NOTE: monte_carlo_sampling should do:
#     #   plant.monte_carlo_results = metric_values
#     #   plant.monte_carlo_metric  = metric
#     #   plant.monte_carlo_inputs  = input_distributions

#     if show_final_plot:
#         # Use the unified signature: pass the plant, like in plot_multiple_monte_carlo
#         plot_monte_carlo_distribution(
#             plant,          # <--- same style of input as plot_multiple_monte_carlo
#             metric=metric,
#             bins=bins,
#             label=label,
#             show=True,
#         )

#     if show_input_distributions:
#         # Also pass the plant here
#         plot_input_distributions(
#             plant,          # <--- plant instead of input_distributions dict
#             figsize=figsize,
#             bins=50,
#         )

#     return metric_values, input_distributions