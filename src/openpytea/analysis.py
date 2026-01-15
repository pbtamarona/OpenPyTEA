from tqdm import tqdm
from copy import deepcopy
from itertools import cycle
from collections.abc import Mapping
from scipy.stats import truncnorm, norm
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import re


plt.style.use(["science", "ieee"])
_ = scienceplots  # mark as used for Flake8


# HELPER FUNCTIONS
def make_label(s: str) -> str:
    """
    Convert a string to a label format by replacing underscores with spaces
    and capitalizing the first character.

    Preserves LaTeX math segments (text enclosed in $...$) without
    modification, while replacing underscores with spaces in non-math segments.

    Args:
        s: Input string that may contain underscores and
        LaTeX math expressions.

    Returns:
        A formatted label string with underscores replaced by spaces
        (outside math segments)
        and the first character capitalized.

    Example:
        >>> make_label("my_variable_$x^2$")
        "My variable $x^2$"
    """
    parts = re.split(r"(\$.*?\$)", s)  # keep math segments
    parts = [
        p.replace("_", " ") if not p.startswith("$") else p
        for p in parts
    ]
    s = "".join(parts)
    return s[:1].upper() + s[1:]


def try_clear_output(*args, **kwargs):
    """
    Attempt to clear the output of the current IPython cell.
    This function tries to import and call `clear_output` from IPython.display.
    If IPython is not available, the function silently passes without
    raising an error.
    Args:
        *args:
        Variable length args list passed to IPython's clear_output function.
        **kwargs:
        Arbitrary keyword args passed to IPython's clear_output function.
            Common kwargs include:
            - wait (bool): If True, wait for next cell output before clearing.
    Returns:
        None
    Note:
        This function is useful in environments where
        IPython may not be installed,
        allowing code to run without raising ImportError exceptions.
    """
    try:
        from IPython.display import clear_output

        clear_output(*args, **kwargs)
    except ImportError:
        pass


def get_original_value(plant, full_key):
    """
    Retrieve the original value from a nested structure
    using a dot-separated key path.

    This function navigates through a potentially nested
    combination of dictionaries and objects to extract a value
    at the location specified by the full_key parameter.

    Args:
        plant: The root object or dictionary to traverse.
        Can be either a dictionary or an object with attributes.
        full_key (str):
        A dot-separated string representing the path to the value
        (e.g., "level1.level2.level3").

    Returns:
        The value found at the specified key path. For dictionary entries,
        returns the "price" field of the value.

    Raises:
        KeyError: If a key is not found in a dictionary.
        AttributeError: If an attribute is not found in an object.
        TypeError:
        If attempting to access a key/attribute on an unsupported type.

    Examples:
        >>> plant = {"item": {"price": 100}}
        >>> get_original_value(plant, "item")
        100
    """
    keys = full_key.split(".")
    ref = plant
    for k in keys:
        if isinstance(ref, dict):
            ref = ref[k]["price"]
        else:
            ref = getattr(ref, k)
    return ref


def update_and_evaluate(
    plant,
    factor,
    value,
    nested_price_keys,
    metric="LCOP",
    additional_capex: bool = False,
):
    """
    Update a plant parameter and recalculate the specified economic metric.
    This function creates a deep copy of the plant object,
    applies a parameter change, recomputes the economic calculations,
    and returns the requested metric.
    Parameters
    ----------
    plant : object
        The plant object to be evaluated. Must have methods for updating
        configuration and calculating economic metrics.
    factor : str
        The parameter to update. Can be one of:
        - "fixed_capital": updates fixed capital cost
        - "fixed_opex": updates fixed operating expenditure
        - "variable_opex_inputs.<name>": updates price of a variable input
        - "plant_products.<name>": updates price of a plant product
        - "operator_hourly_rate": updates operator hourly rate
        - Any other top-level parameter
        (e.g., "interest_rate", "project_lifetime")
    value : float or dict
        The new value for the parameter specified by factor.
    nested_price_keys : list or set
        Collection of valid nested price keys in the format "category.<name>"
        (e.g., ["variable_opex_inputs.steam", "plant_products.electricity"]).
    metric : str, optional
        The economic metric to return. Default is "LCOP".
        Supported values: "LCOP", "ROI", "NPV", "PBT", "IRR".
    additional_capex : bool, optional
        Whether to include additional capital expenditure in ROI and PBT
        calculations. Default is False.
    Returns
    -------
    float or array-like
        The requested metric value after applying the parameter update.
        - "LCOP": Levelized cost of product
        - "ROI": Return on investment (%)
        - "NPV": Net present value
        - "PBT"/"PAYBACK"/"PAYBACK_TIME": Payback time (years)
        - "IRR": Internal rate of return (%)
    Raises
    ------
    ValueError
        If an unsupported nested price root is provided or if an unsupported
        metric is requested.
    """
    plant_copy = deepcopy(plant)
    metric = metric.upper()

    # --- 1. Apply parameter change ---

    if factor == "fixed_capital":
        plant_copy.calculate_fixed_capital(fc=value)

    elif factor == "fixed_opex":
        plant_copy.calculate_fixed_opex(fp=value)

    elif factor in nested_price_keys:
        # factor can be:
        #   "variable_opex_inputs.<name>"  or
        #   "plant_products.<name>"
        parts = factor.split(
            "."
        )  # ['variable_opex_inputs' | 'plant_products', '<name>']
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
            raise ValueError(
                f"Unsupported nested price root '{root}' in factor '{factor}'."
            )

        plant_copy.update_configuration(config)

    elif factor == "operator_hourly_rate":
        # Support both dict-style {"rate": ...} and
        # scalar-style operator_hourly_rate
        current = getattr(
            plant_copy, "operator_hourly_rate", None
        )
        if isinstance(current, dict):
            config = {
                "operator_hourly_rate": {"rate": value}
            }
        else:
            config = {"operator_hourly_rate": value}
        plant_copy.update_configuration(config)

    else:
        # Generic top-level parameter update,
        # e.g. 'interest_rate', 'project_lifetime'
        config = {factor: value}
        plant_copy.update_configuration(config)

    # --- 2. Recompute economics ---

    # This builds fixed_capital, opex, revenue, cash_flow, etc.
    plant_copy.calculate_levelized_cost()

    # --- 3. Return requested metric ---

    if metric == "LCOP":
        return plant_copy.levelized_cost

    elif metric == "ROI":
        plant_copy.calculate_roi(
            additional_capex=additional_capex
        )
        return plant_copy.roi

    elif metric == "NPV":
        # With MC-aware calculate_npv this can be scalar or array.
        # In sensitivity/tornado we are effectively in a single-scenario.
        return plant_copy.calculate_npv()

    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        return plant_copy.calculate_payback_time(
            additional_capex=additional_capex
        )
    elif metric == "IRR":
        plant_copy.calculate_irr()
        return plant_copy.irr

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. \n"
            f"Use 'LCOP', 'ROI', 'NPV', 'PBT', or 'IRR'."
        )


def _plot_stacked_bar_from_components(
    components,
    xlabel,
    ylabel: str,
    figsize=(1.2, 1.8),
    pct: bool = False,
    ax=None,
    show: bool = True,
):
    """
    Plot a stacked bar chart from component data.
    Creates a stacked bar chart visualization from component dictionaries,
    with automatic color mapping and legend generation. Supports single or
    multiple bars with optional percentage normalization.
    Parameters
    ----------
    components : dict or list of dict
        Component data where keys are component names
        and values are numeric values.
        If a single dict is provided,
        it is converted to a list with one element.
    xlabel : str or list of str
        Label(s) for the x-axis.
        If a string is provided and multiple bars exist,
        labels are auto-generated as "{xlabel} 1", "{xlabel} 2", etc.
    ylabel : str
        Base label for the y-axis. Units are automatically appended ("[%]" for
        percentages or "[$]" for absolute values).
    figsize : tuple of float, optional
        Figure size as (width, height). Default is (1.2, 1.8).
        Width is automatically adjusted based on the number of bars.
    pct : bool, optional
        If True, normalize values to percentages per bar. Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axes object to plot on.
        If None, a new figure and axes are created.
        Default is None.
    show : bool, optional
        If True and a new figure was created, display the plot.
        Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Raises
    ------
    ValueError
        If components list is empty, all component dictionaries are empty,
        the number of xlabels does not match the number of bars, or if
        percentage computation is requested with zero total values.
    Notes
    -----
    - Components are sorted by total value across all bars (descending).
    - Colors are assigned from the plasma colormap.
    - Legend is positioned outside the plot area on the right side.
    - For single bars with percentages, the value is displayed in the label.
    - Scientific notation is used for y-labels when displaying absolute values.
    """
    # --- Normalize to list of dicts ---
    if isinstance(components, Mapping):
        components_list = [dict(components)]
    else:
        components_list = [dict(c) for c in components]

    if not components_list:
        raise ValueError(
            "No components to plot (empty list)."
        )

    n_bars = len(components_list)

    # --- Normalize xlabels ---
    if isinstance(xlabel, str):
        if n_bars == 1:
            xlabels = [xlabel]
        else:
            xlabels = [
                f"{xlabel} {i+1}" for i in range(n_bars)
            ]
    else:
        xlabels = list(xlabel)
        if len(xlabels) != n_bars:
            raise ValueError(
                "Number of xlabels must match number of bars."
            )

    # --- Convert values to percentages per bar if requested ---
    if pct:
        converted = []
        for comp in components_list:
            vals = np.array(
                list(comp.values()), dtype=float
            )
            total = vals.sum()
            if total == 0:
                raise ValueError(
                    "Cannot compute percentages: total value "
                    "is zero in one bar."
                )
            factor = 100.0 / total
            converted.append(
                {
                    k: float(v) * factor
                    for k, v in comp.items()
                }
            )
        components_list = converted
        ylabel = ylabel + r" / [\%]"
    else:
        ylabel = ylabel + r" / [\$]"

    # --- Collect all component names across all bars ---
    all_names = set()
    for comp in components_list:
        all_names.update(comp.keys())
    if not all_names:
        raise ValueError(
            "All component dictionaries are empty."
        )

    totals = {
        name: sum(
            float(comp.get(name, 0.0))
            for comp in components_list
        )
        for name in all_names
    }
    names_sorted = sorted(
        all_names, key=lambda n: totals[n], reverse=True
    )

    cmap = plt.cm.plasma
    colors = [
        cmap(i)
        for i in np.linspace(0.15, 0.95, len(names_sorted))
    ]
    color_map = dict(zip(names_sorted, colors))

    spacing = 0.75  # < 1.0 pulls bars together
    bar_width = 0.45
    x = np.arange(n_bars) * spacing
    bottoms = np.zeros(n_bars, dtype=float)

    # --- Ax/fig handling ---
    created_fig = None
    if ax is None:
        if (
            isinstance(figsize, (tuple, list))
            and len(figsize) == 2
        ):
            base_w, base_h = figsize
        else:
            base_w, base_h = 1.2, 1.8
        auto_width = max(base_w * n_bars, base_w)
        created_fig, ax = plt.subplots(
            figsize=(auto_width, base_h)
        )

    # --- Draw stacked bars ---
    for name in names_sorted:
        vals = np.array(
            [
                comp.get(name, 0.0)
                for comp in components_list
            ],
            dtype=float,
        )
        if np.allclose(vals, 0.0):
            continue

        if n_bars == 1 and pct:
            label = rf"{name} ({vals[0]:.1f}\%)"
        else:
            label = name

        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=bar_width,
            color=color_map[name],
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )
        bottoms += vals

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    left = x[0] - bar_width / 2
    right = x[-1] + bar_width / 2
    ax.set_xlim(left - 0.2, right + 0.2)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
        frameon=False,
    )

    if not pct:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)

    if show and created_fig is not None:
        plt.show()

    return ax


def plot_direct_costs_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
    ax=None,
    show=True,
):
    """
    Plot a stacked bar chart of direct costs for given plant(s).
    Parameters
    ----------
    plants : Plant or list of Plant or tuple of Plant
        One or more Plant objects to plot direct costs for.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches. Default is (1.2, 1.8).
    pct : bool, optional
        If True, display costs as percentages.
        If False, display absolute values. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure is created.
        Default is None.
    show : bool, optional
        If True, display the plot. If False, only create the plot object.
        Default is True.
    Returns
    -------
    None
        Displays or creates a stacked bar chart of direct costs by equipment.
    Notes
    -----
    Direct costs are aggregated by equipment name across all plants.
    Each plant is represented as a separate bar in the chart.
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
        ax=ax,
        show=show,
    )


def plot_fixed_capital_bar(
    plants,
    figsize=(1.2, 1.8),
    additional_capex: bool = False,
    pct: bool = False,
    ax=None,
    show=True,
):
    """
    Plot a stacked bar chart of fixed capital costs for one or more plants.
    Parameters
    ----------
    plants : Plant or list of Plant or tuple of Plant
        One or more plant objects to plot fixed capital costs for.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (1.2, 1.8).
    additional_capex : bool, optional
        If True, include additional CAPEX costs in the plot. Default is False.
    pct : bool, optional
        If True, display values as percentages. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure is created.
        Default is None.
    show : bool, optional
        If True, display the plot. Default is True.
    Returns
    -------
    None
    Notes
    -----
    The function calculates fixed capital for each plant and displays a stacked
    bar chart with the following components:
        - ISBL (Inside Battery Limits)
        - OSBL (Outside Battery Limits)
        - Design & engineering
        - Contingency
        - Additional CAPEX (if additional_capex=True and cost is non-zero)
    The function calls calculate_fixed_capital() internally for each plant.
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        plant.calculate_fixed_capital(fc=None)

        components = {
            "ISBL": plant.isbl,
            "OSBL": plant.osbl,
            r"Design \& engineering": plant.dne,
            "Contingency": plant.contigency,
        }

        if additional_capex:

            extra = getattr(
                plant, "additional_capex_cost", None
            )

            if extra is None:
                total_extra = 0.0

            elif isinstance(extra, (list, tuple)):
                total_extra = (
                    float(sum(extra))
                    if len(extra) > 0
                    else 0.0
                )

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
        ylabel=r"Fixed CAPEX",
        figsize=figsize,
        pct=pct,
        ax=ax,
        show=show,
    )


def plot_variable_opex_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
    ax=None,
    show=True,
):
    """
    Plot a stacked bar chart of variable OPEX for one or more plants.
    This function visualizes the annual variable oOPEX broken down by component
    for each plant. It extracts cost information from plant variable OPEX
    inputs and displays them as a stacked bar chart.
    Parameters
    ----------
    plants : Plant or list of Plant
        A single plant object or list of plant objects to plot.
        If a single plant is provided, it will be converted to a list.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (1.2, 1.8).
    pct : bool, optional
        If True, display values as percentages of the total.
        If False, display absolute values. Default is False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on.
        If None, a new figure and axes will be created.
        Default is None.
    show : bool, optional
        If True, display the plot.
        If False, only create the plot object without displaying.
        Default is True.
    Notes
    -----
    - Cost values are extracted from plant variable OPEX inputs
    in the following priority:
      1. 'annual_cost' key if present
      2. 'cost' key if present
      3. Product of 'consumption' and 'price' keys if both present
    - Names are formatted by removing underscores and capitalizing 1st letter.
    - Components with no cost data are skipped.
    Returns
    -------
    None
        Displays the plot according to the `show` parameter.
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        components = {}

        for (
            name,
            props,
        ) in plant.variable_opex_inputs.items():
            if "annual_cost" in props:
                val = props["annual_cost"]
            elif "cost" in props:
                val = props["cost"]
            elif (
                "consumption" in props and "price" in props
            ):
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
        ax=ax,
        show=show,
    )


def plot_fixed_opex_bar(
    plants,
    figsize=(1.2, 1.8),
    pct: bool = False,
    ax=None,
    show=True,
):
    """
    Plot a stacked bar chart of fixed OPEX for one or more plants.
    This function calculates and visualizes the breakdown of
    fixed OPEX components including operating labor,
    supervision, maintenance, taxes & insurance, and
    other overhead costs for the specified plant(s).
    Parameters
    ----------
    plants : Plant or list or tuple
        A single plant object or a list/tuple of plant objects to plot.
        If a single plant is provided, it will be converted to a list.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (1.2, 1.8).
    pct : bool, optional
        If True, display values as percentages.
        If False (default), display absolute values.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on.
        If None, a new figure will be created.
    show : bool, optional
        If True (default), display the plot.
        If False, the plot is not displayed.
    Returns
    -------
    None
        The function displays or stores the plot
        based on the ax and show parameters.
    Notes
    -----
    The fixed OPEX components included in the plot are:
    - Operating labor
    - Supervision
    - Direct salary overhead
    - Laboratory charges
    - Maintenance
    - Taxes & insurance
    - Rent of land
    - Environmental charges
    - Operating supplies
    - General plant overhead
    - Interest on working capital
    - Patents & royalties
    - Distribution & selling
    - R&D
    The function calls calculate_fixed_opex() on each plant before plotting.
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    components_list = []
    xlabels = []

    for plant in plants:
        plant.calculate_fixed_opex(fp=None)

        components = {
            "Operating labor": plant.operating_labor_costs,
            "Supervision": plant.supervision_costs,
            "Direct salary overhead": plant.direct_salary_overhead,
            "Laboratory charges": plant.laboratory_charges,
            "Maintenance": plant.maintenance_costs,
            r"Taxes \& insurance": plant.taxes_insurance_costs,
            "Rent of land": plant.rent_of_land_costs,
            "Environmental charges": plant.environmental_charges,
            "Operating supplies": plant.operating_supplies,
            "General plant overhead": plant.general_plant_overhead,
            "Interest on working capital": plant.interest_working_capital,
            r"Patents \& royalties": plant.patents_royalties,
            r"Distribution \& selling": plant.distribution_selling_costs,
            r"R\&D": plant.RnD_costs,
        }

        components_list.append(components)
        xlabels.append(plant.name)

    _plot_stacked_bar_from_components(
        components=components_list,
        xlabel=xlabels,
        ylabel=r"Annual fixed OPEX",
        figsize=figsize,
        pct=pct,
        ax=ax,
        show=show,
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
    ax=None,
    show: bool = True,
):
    """
    Generate a sensitivity analysis plot for one or more plants.
    This function creates a line plot showing how a specified metric changes
    when a given parameter varies by a certain percentage. It supports multiple
    plants and various financial/technical metrics.
    Parameters
    ----------
    plants : Plant or list of Plant
        One or more Plant objects to analyze.
    parameter : str
        The parameter to vary. Can be a full path (e.g.,
        "variable_opex_inputs.co2_tax") or a shorthand key name.
        Valid top-level parameters: "fixed_capital", "fixed_opex",
        "project_lifetime", "interest_rate", "operator_hourly_rate".
    plus_minus_value : float
        The range of variation as a fraction (e.g., 0.2 for ±20%).
    n_points : int, optional
        Number of points to evaluate along the parameter range.
        Default is 21.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches. Default is (3.2, 2.2).
    metric : str, optional
        The metric to plot. Options: "LCOP", "ROI", "NPV", "PBT"/"PAYBACK"/
        "PAYBACK_TIME", "IRR". Default is "LCOP".
    label : str, optional
        Custom y-axis label. If None, a default label is generated based
        on the metric.
    additional_capex : bool, optional
        Whether to include additional CAPEX in ROI/payback calculations.
        Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    show : bool, optional
        Whether to display the plot. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the sensitivity plot.
    Raises
    ------
    ValueError
        If the parameter is unrecognized, ambiguous across plants, or
        if the metric is unsupported.
    """
    # Normalize plants input
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    metric = metric.upper()

    # Default y-axis labels if not provided
    if label is None:
        if metric == "LCOP":
            label = (
                r"LCOH / [\$$\cdot$kg$^{-1}_\mathrm{H_2}$]"
            )
        elif metric == "ROI":
            label = r"Return on investment / [\%]"
        elif metric == "NPV":
            label = r"Net present value / [\$]"
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            label = "Payback time / [years]"
        elif metric == "IRR":
            label = "Internal rate of return / [-]"
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
    var_opex_keys_all = set(
        f"variable_opex_inputs.{k}"
        for plant in plants
        for k in plant.variable_opex_inputs
    )

    product_keys_all = set(
        f"plant_products.{k}"
        for plant in plants
        for k in plant.plant_products
    )

    byproduct_keys_all = set()
    for plant in plants:
        prod_keys = list(plant.plant_products.keys())
        for k in prod_keys[1:]:
            byproduct_keys_all.add(f"plant_products.{k}")

    if metric == "LCOP":
        nested_price_keys_all = var_opex_keys_all.union(
            byproduct_keys_all
        )
    else:
        nested_price_keys_all = var_opex_keys_all.union(
            product_keys_all
        )

    valid_parameters = set(top_level_keys).union(
        nested_price_keys_all
    )

    # --- Allow shorthand input like "co2_tax" instead of full path ---
    short_to_full = {}
    for plant in plants:
        for k in plant.variable_opex_inputs:
            full = f"variable_opex_inputs.{k}"
            if (
                k in short_to_full
                and short_to_full[k] != full
            ):
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'. \n"
                    f"Please use full path."
                )
            short_to_full[k] = full

        for k in plant.plant_products:
            full = f"plant_products.{k}"
            if (
                k in short_to_full
                and short_to_full[k] != full
            ):
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'. \n"
                    f"Please use full path."
                )
            short_to_full[k] = full

    parameter = short_to_full.get(parameter, parameter)

    if parameter not in valid_parameters:
        raise ValueError(
            f"Unrecognized parameter: {parameter}"
        )

    pct_changes = np.linspace(
        -plus_minus_value, plus_minus_value, n_points
    )
    pct_axis = pct_changes * 100

    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }

    for plant in plants:
        for var in plant.variable_opex_inputs:
            key = f"variable_opex_inputs.{var}"
            label_map[key] = f"{make_label(var)} price"
        for prod in plant.plant_products:
            key = f"plant_products.{prod}"
            label_map[key] = f"{make_label(prod)} price"

    label_raw = label_map.get(
        parameter,
        parameter.replace("variable_opex_inputs.", "")
        .replace("plant_products.", "")
        .replace(".price", ""),
    )
    label_clean = make_label(label_raw)
    x_label = label_clean + r" / [$\pm$ \%]"

    # --- Ax/fig handling ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    # Loop over plants and plot each sensitivity curve
    for i, plant in enumerate(plants):
        var_opex_keys = set(
            f"variable_opex_inputs.{k}"
            for k in plant.variable_opex_inputs
        )

        prod_key_list = list(plant.plant_products.keys())
        all_prod_keys = set(
            f"plant_products.{k}" for k in prod_key_list
        )
        byprod_keys = set(
            f"plant_products.{k}" for k in prod_key_list[1:]
        )

        if metric == "LCOP":
            nested_price_keys = var_opex_keys.union(
                byprod_keys
            )
        else:
            nested_price_keys = var_opex_keys.union(
                all_prod_keys
            )

        plant_valid_params = set(top_level_keys).union(
            nested_price_keys
        )

        # Baseline metric
        if metric == "LCOP":
            if not hasattr(plant, "levelized_cost"):
                plant.calculate_levelized_cost()
            base_value = plant.levelized_cost
        elif metric == "ROI":
            plant.calculate_levelized_cost()
            plant.calculate_roi(
                additional_capex=additional_capex
            )
            base_value = plant.roi
        elif metric == "NPV":
            plant.calculate_levelized_cost()
            base_value = plant.calculate_npv()
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            plant.calculate_levelized_cost()
            base_value = plant.calculate_payback_time(
                additional_capex=additional_capex
            )
        elif metric == "IRR":
            plant.calculate_levelized_cost()
            plant.calculate_irr()
            base_value = plant.irr
        else:
            raise ValueError(
                f"Unsupported metric '{metric}'."
            )

        color = next(line_colors)
        plant_label = getattr(plant, "name", f"Plant {i+1}")

        if parameter not in plant_valid_params:
            metric_values = np.full_like(
                pct_axis, fill_value=base_value, dtype=float
            )
        else:
            if parameter in ["fixed_capital", "fixed_opex"]:
                original_value = 1.0
            else:
                original_value = get_original_value(
                    plant, parameter
                )

            param_values = original_value * (
                1 + pct_changes
            )

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

        ax.plot(
            pct_axis,
            metric_values,
            linewidth=1,
            color=color,
            label=plant_label,
            linestyle="-",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(label)
    ax.legend(loc="best")

    # Only tighten/show if we created the figure
    # (prevents messing up your dashboard layout)
    if created_fig is not None:
        created_fig.tight_layout()
        if show:
            plt.show()
    else:
        if show:
            ax.figure.canvas.draw_idle()

    return ax


def tornado_plot(
    plant,
    plus_minus_value,
    metric="LCOP",
    figsize=(3.4, 2.4),
    label=None,
    ax=None,
    show: bool = True,
):
    """
    Generate a tornado plot for sensitivity analysis of a plant economic model.
    A tornado plot visualizes the impact of variations in input parameters on a
    selected economic metric. Each horizontal bar shows the range of the metric
    value when a parameter is varied by a specified percentage above and below
    its baseline value.
    Parameters
    ----------
    plant : object
        A plant object with economic parameters
        and methods to calculate metrics.
        Expected attributes include:
        fixed_capital, fixed_opex, project_lifetime, interest_rate,
        operator_hourly_rate, variable_opex_inputs, and plant_products.
    plus_minus_value : float
        The fractional variation applied to parameters (e.g., 0.1 for ±10%).
    metric : str, optional
        The economic metric to analyze. Options are:
        - "LCOP": Levelized cost of product (default)
        - "ROI": Return on investment
        - "NPV": Net present value
        - "PBT"/"PAYBACK"/"PAYBACK_TIME": Payback time in years
        - "IRR": Internal rate of return
        Default is "LCOP".
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (3.4, 2.4).
    label : str, optional
        Custom x-axis label.
        If None, a default label is generated based on the metric.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axes object to plot on.
        If None, a new figure is created.
    show : bool, optional
        If True and a new figure was created, displays the plot.
        Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the tornado plot.
    Notes
    -----
    - Parameters are sorted by their total effect (sensitivity) on the metric,
      with the largest effects at the top.
    - Blue bars represent the impact of decreasing a parameter;
    red bars represent increasing it.
    - A vertical dashed line indicates the baseline metric value.
    - The plot uses a label mapping for cleaner parameter names on the y-axis.
    Raises
    ------
    ValueError
        If an unsupported metric is specified.
    """
    metric = metric.upper()

    # Default x-axis labels if not provided
    if label is None:
        if metric == "LCOP":
            label = (
                r"Levelized cost / [\$$\cdot$unit$^{-1}$]"
            )
        elif metric == "ROI":
            label = r"Return on investment / [\%]"
        elif metric == "NPV":
            label = r"Net present value / [\$]"
        elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
            label = "Payback time / [years]"
        elif metric == "IRR":
            label = "Internal rate of return / [-]"
        else:
            label = metric

    top_level_keys = [
        "fixed_capital",
        "fixed_opex",
        "project_lifetime",
        "interest_rate",
        "operator_hourly_rate",
    ]

    var_opex_price_keys = [
        f"variable_opex_inputs.{k}"
        for k in plant.variable_opex_inputs.keys()
    ]
    product_price_keys = [
        f"plant_products.{k}"
        for k in plant.plant_products.keys()
    ]

    nested_price_keys = (
        var_opex_price_keys
        if metric == "LCOP"
        else (var_opex_price_keys + product_price_keys)
    )
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
        if key in ["fixed_capital", "fixed_opex"]:
            low = 1 - plus_minus_value
            high = 1 + plus_minus_value

        elif key == "operator_hourly_rate":
            current = getattr(
                plant, "operator_hourly_rate", None
            )
            if isinstance(current, dict):
                original = current.get("rate", 0.0)
            else:
                original = (
                    0.0
                    if current is None
                    else float(current)
                )
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)

        else:
            original = get_original_value(plant, key)
            low = original * (1 - plus_minus_value)
            high = original * (1 + plus_minus_value)

        metric_low = update_and_evaluate(
            plant,
            key,
            low,
            nested_price_keys,
            metric=metric,
        )
        metric_high = update_and_evaluate(
            plant,
            key,
            high,
            nested_price_keys,
            metric=metric,
        )

        sensitivity_results[key] = [metric_low, metric_high]

    factors = list(sensitivity_results.keys())
    lows = np.array(
        [sensitivity_results[f][0] for f in factors],
        dtype=float,
    )
    highs = np.array(
        [sensitivity_results[f][1] for f in factors],
        dtype=float,
    )
    total_effects = np.abs(highs - lows)

    sorted_indices = np.argsort(
        total_effects
    )  # small -> large (largest appears at top in barh)
    factors_sorted = [factors[i] for i in sorted_indices]
    lows_sorted = lows[sorted_indices]
    highs_sorted = highs[sorted_indices]

    # --- Label mapping for pretty y-axis names ---
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }
    for var in plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}"] = (
            f"{make_label(var)} price"
        )
    for prod in plant.plant_products:
        label_map[f"plant_products.{prod}"] = (
            f"{make_label(prod)} price"
        )

    labels_sorted = [
        label_map.get(f, make_label(f))
        for f in factors_sorted
    ]
    y_pos = np.arange(len(labels_sorted))

    # --- Ax/fig handling ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    # Colors (keep your original)
    colors_low = ["#87CEEB"] * len(labels_sorted)  # blue
    colors_high = ["#FF9999"] * len(labels_sorted)  # red

    # --- Plot ---
    for i in range(len(y_pos)):
        low_val = lows_sorted[i]
        high_val = highs_sorted[i]

        ax.barh(
            y_pos[i],
            abs(low_val - base_value),
            left=min(base_value, low_val),
            color=colors_low[i],
            edgecolor="black",
            label=(
                rf"-{int(plus_minus_value * 100)}\%"
                if i == 0
                else ""
            ),
        )

        ax.barh(
            y_pos[i],
            abs(high_val - base_value),
            left=min(base_value, high_val),
            color=colors_high[i],
            edgecolor="black",
            label=(
                rf"+{int(plus_minus_value * 100)}\%"
                if i == 0
                else ""
            ),
        )

    ax.axvline(
        x=base_value,
        color="black",
        linestyle="--",
        linewidth=0.75,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted)

    # x-limits with padding
    x_all = np.concatenate(
        [
            lows_sorted,
            highs_sorted,
            np.atleast_1d(base_value),
        ]
    )
    xmin, xmax = float(x_all.min()), float(x_all.max())

    if xmin == xmax:
        pad = 0.05 * (1.0 if xmax == 0 else abs(xmax))
        left_lim, right_lim = xmin - pad, xmax + pad
    else:
        span = xmax - xmin
        pad = 0.05 * span
        left_lim, right_lim = xmin - pad, xmax + pad

    ax.set_xlim(left_lim, right_lim)
    ax.set_xlabel(label)
    ax.legend(loc="best")

    # Only manage layout/show if we created the figure
    if created_fig is not None:
        created_fig.tight_layout()
        if show:
            plt.show()
    else:
        if show:
            ax.figure.canvas.draw_idle()

    return ax


def truncated_normal_samples(mean, std, low, high, size):
    """
    Generate random samples from a truncated normal distribution.
    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
    low : float
        Lower bound of the truncation interval.
    high : float
        Upper bound of the truncation interval.
    size : int
        Number of random samples to generate.
    Returns
    -------
    ndarray
        Array of random samples from the truncated normal distribution.
        If std is zero or close to zero, returns an array filled with the
        clipped mean value.
    Notes
    -----
    When std is 0 or very close to 0, the function returns a deterministic
    result (the mean clipped to the [low, high] interval) instead of sampling.
    """
    if std == 0 or np.isclose(std, 0):
        return np.full(size, np.clip(mean, low, high))

    a, b = (low - mean) / std, (high - mean) / std

    return truncnorm.rvs(
        a, b, loc=mean, scale=std, size=size
    )


def get_sampling_params(
    props, default_min=-1, default_max=1
):
    """
    Extract sampling parameters from a properties dictionary.

    This function retrieves statistical parameters for sampling from a given
    properties dictionary. If certain keys are not present, default values
    are used for the min and max bounds.

    Args:
        props (dict): A dictionary containing sampling parameter definitions.
            Expected keys are:
            - "price" (optional): The mean value for sampling.
            Defaults to 0.
            - "std" (optional): The standard deviation for sampling.
            Defaults to 0.
            - "min" (optional): The minimum bound for sampling.
            Defaults to default_min.
            - "max" (optional): The maximum bound for sampling.
            Defaults to default_max.
        default_min (float, optional): The default minimum bound value.
        Defaults to -1.
        default_max (float, optional): The default maximum bound value.
        Defaults to 1.

    Returns:
        tuple: A tuple containing four values in order:
            - mean (float): The mean value for sampling.
            - std (float): The standard deviation for sampling.
            - min_ (float): The minimum bound for sampling.
            - max_ (float): The maximum bound for sampling.
    """
    mean = props.get("price", 0)
    std = props.get("std", 0)
    min_ = props.get("min", default_min)
    max_ = props.get("max", default_max)
    return mean, std, min_, max_


def monte_carlo(
    plant,
    num_samples: int = 1_000_000,
    batch_size: int = 1000,
    additional_capex: bool = False,
):
    """
    Perform Monte Carlo simulation on a plant's economic metrics.
    This function conducts a probabilistic analysis of a plant's
    economic performance by sampling from distributions of various
    input parameters and computing multiple
    economic metrics across the samples.
    Parameters
    ----------
    plant : Plant
        The plant object to analyze. Must have economic calculation methods and
        configuration attributes initialized.
    num_samples : int, optional
        Total number of Monte Carlo samples to generate. Default is 1,000,000.
    batch_size : int, optional
        Number of samples to process per batch. Default is 1,000.
        Reduces memory usage by processing samples in chunks.
    additional_capex : bool, optional
        Whether to include additional capital expenditure in ROI and PBT
        calculations. Default is False.
    Returns
    -------
    tuple[dict, dict]
        - mc_metrics : dict
            Dictionary containing arrays of computed metrics for all samples:
            - "LCOP" : np.ndarray
                Levelized cost of product for each sample.
            - "ROI" : np.ndarray
                Return on investment for each sample
                (only if product prices available).
            - "NPV" : np.ndarray
                Net present value for each sample
                (only if product prices available).
            - "PBT" : np.ndarray
                Payback time for each sample
                (only if product prices available).
        - mc_inputs : dict
            Dictionary containing the sampled input parameters for each sample:
            - "Fixed capital factor" : np.ndarray
            - "Fixed opex factor" : np.ndarray
            - "Operator hourly rate" : np.ndarray
            - "Project lifetime" : np.ndarray
            - "Interest rate" : np.ndarray
            - Variable opex price samples for each item
            - Product price samples for each product
    Notes
    -----
    - The plant object is deep copied to avoid modifying the original
    during sampling.
    - All input parameters are sampled from truncated normal distributions.
    - Economic metrics (ROI, NPV, PBT) are only computed if product prices
    are defined.
    - Results are stored in the original plant object as
    `monte_carlo_metrics` and `monte_carlo_inputs` attributes.
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
        "ROI": np.zeros(num_samples),
        "NPV": np.zeros(num_samples),
        "PBT": np.zeros(num_samples),
    }

    # ---- Allocate all input distributions (same as before) ----
    op_cfg = plant.operator_hourly_rate
    op_mean = op_cfg.get("rate", 38.11)
    op_std = op_cfg.get("std", 20 / 2)
    op_min = op_cfg.get("min", 10)
    op_max = op_cfg.get("max", 100)

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
        "price" in props
        for props in plant.plant_products.values()
    )

    product_price_samples = (
        {
            prod: np.zeros(num_samples)
            for prod in plant.plant_products
        }
        if have_product_prices
        else {}
    )

    # ---- Sampling loop ----
    for b in tqdm(range(num_batches), desc="Monte Carlo"):
        start = b * batch_size
        end = start + batch_size

        # ---- Sample inputs ----
        fixed_capitals[start:end] = (
            truncated_normal_samples(
                1, 0.3, 0.25, 1.75, batch_size
            )
        )
        fixed_opexs[start:end] = truncated_normal_samples(
            1, 0.3, 0.25, 1.75, batch_size
        )
        operator_hourlys[start:end] = (
            truncated_normal_samples(
                op_mean, op_std, op_min, op_max, batch_size
            )
        )
        project_lifetimes[start:end] = (
            truncated_normal_samples(
                plant.project_lifetime,
                5,
                max(5, plant.project_lifetime - 2 * 5),
                plant.project_lifetime + 2 * 5,
                batch_size,
            )
        )
        interests[start:end] = truncated_normal_samples(
            plant.interest_rate,
            0.03,
            max(0.02, plant.interest_rate - 2 * 0.03),
            plant.interest_rate + 2 * 0.03,
            batch_size,
        )

        for (
            item,
            props,
        ) in plant.variable_opex_inputs.items():
            mean, std, min_, max_ = get_sampling_params(
                props
            )
            samples = truncated_normal_samples(
                mean, std, min_, max_, batch_size
            )
            variable_opex_price_samples[item][
                start:end
            ] = samples

        if have_product_prices:
            for prod, props in plant.plant_products.items():
                mean, std, min_, max_ = get_sampling_params(
                    props
                )

                samples = truncated_normal_samples(
                    mean, std, min_, max_, batch_size
                )
                product_price_samples[prod][
                    start:end
                ] = samples

        # ---- Apply sampled inputs ----
        plant_copy.operator_hourly_rate["rate"] = (
            operator_hourlys[start:end]
        )
        plant_copy.update_configuration(
            {
                "project_lifetime": project_lifetimes[
                    start:end
                ],
                "interest_rate": interests[start:end],
            }
        )

        for item in plant.variable_opex_inputs:
            plant_copy.variable_opex_inputs[item][
                "price"
            ] = variable_opex_price_samples[item][start:end]

        if have_product_prices:
            for prod in plant.plant_products:
                plant_copy.plant_products[prod]["price"] = (
                    product_price_samples[prod][start:end]
                )

        # ---- Economic calculations ----
        plant_copy.calculate_fixed_capital(
            fc=fixed_capitals[start:end]
        )
        plant_copy.calculate_variable_opex()
        plant_copy.calculate_fixed_opex(
            fp=fixed_opexs[start:end]
        )
        plant_copy.calculate_cash_flow()
        plant_copy.calculate_levelized_cost()

        # ---- Store LCOP always ----
        mc_metrics["LCOP"][
            start:end
        ] = plant_copy.levelized_cost

        # ---- If revenue available, compute all other metrics ----
        if have_product_prices:
            mc_metrics["ROI"][start:end] = (
                plant_copy.calculate_roi(
                    additional_capex=additional_capex
                )
            )
            mc_metrics["NPV"][
                start:end
            ] = plant_copy.calculate_npv()
            mc_metrics["PBT"][start:end] = (
                plant_copy.calculate_payback_time(
                    additional_capex=additional_capex
                )
            )

    # ---- Store all results on plant ----
    plant.monte_carlo_metrics = mc_metrics
    plant.monte_carlo_inputs = {
        "Fixed capital factor": fixed_capitals,
        "Fixed opex factor": fixed_opexs,
        "Operator hourly rate": operator_hourlys,
        "Project lifetime": project_lifetimes,
        "Interest rate": interests,
        **{
            f"{k} price": v
            for k, v in variable_opex_price_samples.items()
        },
        **{
            f"{k} product price": v
            for k, v in product_price_samples.items()
        },
    }

    return mc_metrics, plant.monte_carlo_inputs


def default_metric_label(metric: str) -> str:
    """
    Generate a default metric label for a given metric name.

    Parameters
    ----------
    metric : str
        The name of the metric to generate a label for. Case-insensitive.
        Supported metrics: 'LCOP', 'ROI', 'NPV', 'PBT', 'IRR'.

    Returns
    -------
    str
        A formatted label string for the metric,
        including units where applicable.
        - 'LCOP': Levelized cost with units [$/unit]
        - 'ROI': Return on investment with units [%]
        - 'NPV': Net present value with units [$]
        - 'PBT', 'PAYBACK', 'PAYBACK_TIME': Payback time with units [years]
        - 'IRR': Internal rate of return with units [-]
        - Any other metric: Returns the uppercase version of the input metric

    Examples
    --------
    >>> default_metric_label('lcop')
    'Levelized cost / [\\$$\\cdot$unit$^{-1}$]'
    >>> default_metric_label('roi')
    'Return on investment / [\\%]'
    >>> default_metric_label('payback_time')
    'Payback time / [years]'
    """
    metric = metric.upper()
    if metric == "LCOP":
        return r"Levelized cost / [\$$\cdot$unit$^{-1}$]"
    elif metric == "ROI":
        return r"Return on investment / [\%]"
    elif metric == "NPV":
        return r"Net present value / [\$]"
    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        return "Payback time / [years]"
    elif metric == "IRR":
        return "Internal rate of return / [-]"
    return metric


def plot_monte_carlo(
    plant,
    metric: str = None,
    bins: int = 30,
    label: str | None = None,
    ax=None,
    show: bool = True,
):
    """
    Plot a histogram of Monte Carlo simulation results
    with a fitted normal distribution.
    Parameters
    ----------
    plant : Plant or array-like
        Either a Plant object with monte_carlo_metrics attribute,
        or a numpy array of metric values to plot.
    metric : str, optional
        The metric to plot. Must be a key in plant.monte_carlo_metrics if plant
        is a Plant object. Default is "LCOP". Case-insensitive.
    bins : int, optional
        Number of histogram bins. Default is 30.
    label : str or None, optional
        Label for the x-axis. If None, a default label is generated based on
        the metric name. Default is None.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure and axes are
        created. Default is None.
    show : bool, optional
        Whether to display the plot using plt.show(). Only applies if a new
        figure is created. Default is True.
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Raises
    ------
    ValueError
        If the specified metric is not found in plant.monte_carlo_metrics.
    Notes
    -----
    The normal distribution is fitted to the data using scipy.stats.norm.fit().
    The standard deviation is displayed in scientific notation with mantissa
    and exponent separated for readability.
    """
    # --- Accept both Plant and array ---
    if hasattr(plant, "monte_carlo_metrics"):
        if metric is None:
            metric = "LCOP"
        metric = metric.upper()

        if metric not in plant.monte_carlo_metrics:
            available = ", ".join(
                plant.monte_carlo_metrics.keys()
            )
            raise ValueError(
                f"Metric '{metric}' not found."
                f" Available: {available}"
            )

        values = plant.monte_carlo_metrics[metric]
    else:
        values = np.asarray(plant)
        if metric is None:
            metric = "LCOP"

    if label is None:
        label = default_metric_label(metric)

    mu, std = norm.fit(values)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()

    hist_color = next(cycle(plt.cm.tab10.colors))
    line_color = next(cycle(plt.cm.tab10.colors))

    ax.hist(
        values,
        bins=bins,
        density=True,
        color=hist_color,
        edgecolor="black",
        alpha=0.6,
        zorder=1,
        label="Samples",
    )

    x = np.linspace(values.min(), values.max(), 1000)
    p = norm.pdf(x, mu, std)

    std_exp = int(np.floor(np.log10(std)))
    std_mant = std / 10**std_exp
    ax.plot(
        x,
        p,
        color=line_color,
        zorder=2,
        label=(
            rf"$\mu$={mu:.3g}, "
            rf"$\sigma$={std_mant:.2f}$\times 10^{{{std_exp}}}$"
        ),
    )

    ax.set_xlabel(label)
    ax.set_ylabel("Probability density")
    ax.legend(loc="best", fontsize="x-small")

    if created_fig is not None and show:
        created_fig.tight_layout()
        plt.show()

    return ax


def plot_monte_carlo_inputs(
    plant,
    figsize=None,
    bins: int = 50,
    show: bool = True,
):
    """
    Plot histograms of Monte Carlo input parameters.
    Creates a grid of histograms visualizing the distribution of Monte Carlo
    input parameters. If the plant object has a monte_carlo_inputs attribute,
    it uses that; otherwise, treats the plant argument as a dictionary of
    inputs.
    Parameters
    ----------
    plant : object or dict
        A plant object with a monte_carlo_inputs attribute, or a dictionary
        where keys are parameter names and values are arrays of sampled values.
    figsize : tuple, optional
        Figure size as (width, height) in inches. If None, defaults to
        (3*5, n_rows*3) where n_rows is calculated based on the number of
        parameters.
    bins : int, default=50
        Number of bins to use for each histogram.
    show : bool, default=True
        If True, calls plt.tight_layout() and plt.show() to display the figure.
    Returns
    -------
    numpy.ndarray
        Flattened array of matplotlib Axes objects from the subplot grid.
    Notes
    -----
    - Histograms are plotted with density=True for normalized distributions.
    - Unused subplot axes are turned off.
    - Parameters are arranged in a grid with 3 columns.
    """
    if hasattr(plant, "monte_carlo_inputs"):
        inputs = plant.monte_carlo_inputs
    else:
        inputs = plant

    n_params = len(inputs)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 3)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize
    )
    axes = axes.flatten()

    hist_color = next(cycle(plt.cm.tab10.colors))

    for idx, (label, arr) in enumerate(inputs.items()):
        ax = axes[idx]
        ax.hist(
            arr,
            bins=bins,
            density=True,
            color=hist_color,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(label, fontsize=9)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    if show:
        fig.tight_layout()
        plt.show()

    return axes


def plot_multiple_monte_carlo(
    plants,
    metric="LCOP",
    bins=30,
    figsize=None,
    label=None,
    ax=None,
    show: bool = True,
):
    """
    Plot multiple Monte Carlo simulation results as overlaid histograms with
    fitted normal distributions.
    Parameters
    ----------
    plants : list
        List of plant objects containing monte_carlo_metrics data.
    metric : str, optional
        The metric to plot from monte_carlo_metrics (default: "LCOP").
        The string is converted to uppercase.
    bins : int, optional
        Number of histogram bins (default: 30).
    figsize : tuple, optional
        Figure size as (width, height).
        If None, uses matplotlib default (default: None).
    label : str, optional
        Label for the x-axis.
        If None, uses default_metric_label(metric) (default: None).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
        If None, creates a new figure and axes (default: None).
    show : bool, optional
        If True, displays the plot.
        If False, closes the figure (default: True).
        Only applies when ax is None (i.e., when a new figure is created).
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    Notes
    -----
    - Each plant's data is plotted as a semi-transparent
    histogram with black edges.
    - A fitted normal distribution curve is overlaid for each plant.
    - The legend displays plant names and
    distribution parameters (μ and σ).
    - Legend position and number of columns are
    automatically adjusted based on the number of items.
    - Only plants with the specified metric in their
    monte_carlo_metrics are plotted.
    Raises
    ------
    AttributeError
        If a plant object lacks the monte_carlo_metrics attribute or the
        specified metric is not present.
    """
    metric = metric.upper()

    created_fig = None
    if ax is None:
        if figsize is None:
            created_fig, ax = plt.subplots()
        else:
            created_fig, ax = plt.subplots(figsize=figsize)

    hist_colors = cycle(plt.cm.tab10.colors)
    line_colors = cycle(plt.cm.tab10.colors)

    for plant in plants:
        if (
            hasattr(plant, "monte_carlo_metrics")
            and metric in plant.monte_carlo_metrics
        ):
            values = plant.monte_carlo_metrics[metric]
            hist_color = next(hist_colors)
            line_color = next(line_colors)

            mu, std = norm.fit(values)

            ax.hist(
                values,
                bins=bins,
                alpha=0.5,
                density=True,
                edgecolor="black",
                color=hist_color,
                zorder=1,
                label=plant.name,
            )

            x = np.linspace(
                values.min(), values.max(), 1000
            )
            p = norm.pdf(x, mu, std)

            std_exp = int(np.floor(np.log10(std)))
            std_mant = std / 10**std_exp

            ax.plot(
                x,
                p,
                color=line_color,
                linewidth=1.2,
                zorder=2,
                linestyle="-",
                label=(
                    rf"$\mu$={mu:.3g}, "
                    rf"$\sigma$={std_mant:.2f}$\times 10^{{{std_exp}}}$"
                ),
            )

    if label is None:
        label = default_metric_label(metric)

    ax.set_xlabel(label)
    ax.set_ylabel("Probability density")

    handles, labels_list = ax.get_legend_handles_labels()
    n_items = len(labels_list)

    if n_items <= 4:
        ncol, loc, bbox = 1, "best", None
    elif n_items <= 6:
        ncol, loc, bbox = 3, "upper center", (0.5, 1.15)
    else:
        ncol, loc, bbox = 4, "upper center", (0.5, 1.20)

    ax.legend(
        loc=loc,
        ncol=ncol,
        fontsize=4,
        frameon=True,
        facecolor="white",
        framealpha=0.6,
        fancybox=True,
        bbox_to_anchor=bbox,
    )

    if created_fig is not None:
        if bbox:
            created_fig.tight_layout(rect=[0, 0, 1, 0.92])
        else:
            created_fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.close(created_fig)

    return ax
