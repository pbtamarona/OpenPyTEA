from copy import deepcopy
from tqdm import tqdm
import numpy as np

from openpytea.helpers import (_make_label,
                               _get_original_value,
                               _update_and_evaluate,
                               _get_sampling_params,
                               _truncated_normal_samples,
                               _default_metric_label,
                               _ensure_list,
                               _build_bar_data,
                               _evaluate_metric,
                               _collect_sensitivity_keys,
                               _run_tornado_sensitivity,
                               _build_tornado_labels)


# ======================================================
# DATA PREPARATION (MAIN API)
# ======================================================

def direct_costs_data(plants, pct=False):
    """
    Extract and organize direct cost data from one or more plants.
    This function aggregates direct cost information from equipment lists
    across one or more plants and prepares the data for visualization as
    a bar chart.
    Parameters
    ----------
    plants : Plant or list of Plant
        A single plant object or a list of plant objects from which to extract
        direct cost data.
    pct : bool, optional
        If True, return direct costs as percentages of the total. If False
        (default), return absolute cost values.
    Returns
    -------
    dict
        A dictionary containing structured data for bar chart visualization,
        including:
        - Component costs keyed by equipment name
        - Plant names as x-axis labels
        - Currency symbol
        - Chart title and formatting information
    Notes
    -----
    - If plants list is empty, USD currency symbol is used as default
    - Currency is automatically extracted from the first plant in the list
    - Each equipment's direct cost is converted to float for numerical
    operations
    Examples
    --------
    >>> plant1 = Plant(name="Plant A", currency="$")
    >>> data = direct_costs_data(plant1)
    >>> data = direct_costs_data([plant1, plant2], pct=True)
    """
    plants = _ensure_list(plants)
    currency = plants[0].currency if plants else r"\$"

    components_list = []
    xlabels = []

    for plant in plants:
        components = {
            eq.name: float(eq.direct_cost)
            for eq in plant.equipment_list
        }
        components_list.append(components)
        xlabels.append(plant.name)

    return _build_bar_data(components_list, xlabels,
                           "Direct costs", currency, pct)


def fixed_capital_data(plants, additional_capex=False, pct=False):
    """
    Generate fixed capital expenditure data for one or more plants.
    This function calculates and aggregates the fixed capital costs for given
    plants, breaking down costs into components (ISBL, OSBL,
    Design & Engineering, and Contingency). Optionally includes additional
    CAPEX costs if available.
    Args:
        plants (Plant or list[Plant]): A single plant object or list of plant
        objects to generate fixed capital data for.
        additional_capex (bool, optional):
        If True, includes additional CAPEX costs
            from the plant's `additional_capex_cost` attribute.
            Defaults to False.
        pct (bool, optional): If True, returns data as percentages
        of total CAPEX.
            If False, returns absolute values. Defaults to False.
    Returns:
        dict: A dictionary containing structured bar chart data with keys:
            - "components": List of dictionaries with CAPEX component
            breakdowns
            - "labels": List of plant names (x-axis labels)
            - "title": Chart title ("Fixed CAPEX")
            - "currency": Currency symbol or code
            - "percentage": Boolean indicating if values are percentages
    Raises:
        AttributeError: If plant objects lack required attributes (isbl, osbl,
        dne, etc.).
    Example:
        >>> plants = [plant1, plant2]
        >>> data = fixed_capital_data(plants, additional_capex=True, pct=False)
        >>> # Returns fixed CAPEX breakdown for both plants with additional
        >>> # costs in absolute values
    """
    plants = _ensure_list(plants)
    currency = plants[0].currency if plants else r"\$"

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
            extra = getattr(plant, "additional_capex_cost", None)

            if isinstance(extra, (list, tuple, np.ndarray)):
                total_extra = float(
                    sum(x for x in extra if isinstance(x, (int, float)))
                )
            else:
                try:
                    total_extra = float(extra) if extra is not None else 0.0
                except (TypeError, ValueError):
                    total_extra = 0.0

            if total_extra != 0:
                components["Additional CAPEX"] = total_extra

        components_list.append(components)
        xlabels.append(plant.name)

    return _build_bar_data(components_list, xlabels,
                           "Fixed CAPEX", currency, pct)


def variable_opex_data(plants, pct=False):
    """
    Extract variable operational expenditure (OPEX) data from '
    one or more plants. This function processes plant objects to compile their
    variable OPEX components and returns formatted data suitable for
    visualization. It handles multiple cost definition formats and supports
    currency representation.
    Args:
        plants (Plant or list[Plant]): A single plant object
        or list of plant objects from which to extract variable OPEX data.
        pct (bool, optional): If True, display values as percentages.
        Default is False.
    Returns:
        dict: A dictionary containing structured data for visualization,
        including:
            - Components breakdown for each plant
            - X-axis labels (plant names)
            - Title: "Annual variable OPEX"
            - Currency symbol or format
            - Data formatted as percentages if pct=True
    Notes:
        - Cost values are determined from (in priority order):
            1. "annual_cost" field
            2. "cost" field
            3. "consumption" * "price" calculation
        - If none of these fields exist, the component is skipped.
        - Component names are formatted via _make_label() function.
        - Currency is extracted from the first plant,
        defaulting to "$" if no plants provided.
    """
    plants = _ensure_list(plants)
    currency = plants[0].currency if plants else r"\$"

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

            label = _make_label(name)
            components[label] = float(val)

        components_list.append(components)
        xlabels.append(plant.name)

    return _build_bar_data(components_list, xlabels,
                           "Annual variable OPEX", currency, pct)


def fixed_opex_data(plants, pct=False):
    """
    Generate fixed operating expenditure (OPEX) data for one or more plants.
    This function calculates and aggregates the fixed OPEX components for the
    given plants, including operating labor, supervision, maintenance, taxes,
    insurance, and other operational costs.
    Parameters
    ----------
    plants : Plant or list of Plant
        A single Plant object or a list of Plant objects for which to
        calculate fixed OPEX data.
    pct : bool, optional
        If True, return OPEX data as percentages. If False (default),
        return absolute values.
    Returns
    -------
    dict
        A dictionary containing structured bar chart data with OPEX components
        and plant names.
        The structure includes:
        - Component costs (Operating labor, Supervision, Maintenance, etc.)
        - Plant names as x-axis labels
        - Currency information
        - Annual fixed OPEX totals
    Notes
    -----
    The function calculates the following fixed OPEX components:
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
    - Research & Development (R&D)
    Examples
    --------
    >>> result = fixed_opex_data(plant1)
    >>> result = fixed_opex_data([plant1, plant2], pct=True)
    """
    plants = _ensure_list(plants)
    currency = plants[0].currency if plants else r"\$"

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

    return _build_bar_data(components_list, xlabels,
                           "Annual fixed OPEX", currency, pct)


def sensitivity_data(plants,
                     parameter,
                     plus_minus_value,
                     n_points=21,
                     metric="LCOP",
                     label=None,
                     additional_capex: bool = False):
    """
    Perform sensitivity analysis on one or more plants by varying a parameter.
    This function computes how a specified metric (e.g., LCOP) changes as a
    parameter is varied by a given percentage range. It supports both top-level
    parameters (capital, opex, etc.) and nested parameters (variable costs,
    product prices, etc.).
    Parameters
    ----------
    plants : Plant or list of Plant
        One or more Plant objects to analyze. If a single plant is provided,
        it is converted to a list.
    parameter : str
        The parameter to vary. Can be specified as:
        - A top-level key: "fixed_capital", "fixed_opex", "project_lifetime",
          "interest_rate", or "operator_hourly_rate"
        - A nested key: "variable_opex_inputs.{key}" or "plant_products.{key}"
        - A shorthand: "{key}" (resolved to full path if unambiguous)
    plus_minus_value : float
        The fraction (0-1) to vary the parameter by in both directions.
        For example, 0.2 varies from -20% to +20%.
    n_points : int, optional
        Number of points along the variation range. Default is 21.
    metric : str, optional
        The metric to compute. Default is "LCOP".
        Will be converted to uppercase.
    label : str, optional
        Custom label for the y-axis. If None, a default label is generated
        based on the metric and plant currency.
    additional_capex : bool, optional
        Whether to include additional capital expenditure in calculations.
        Default is False.
    Returns
    -------
    dict
        A dictionary containing:
        - "curves" : list of dict
            List of results for each plant, each containing:
            - "plant" : str
                Plant name or identifier
            - "x" : ndarray
                Percentage changes along the variation range
            - "y" : ndarray or list
                Metric values corresponding to each point
            - "baseline" : float
                Metric value at the baseline (0% variation)
        - "xlabel" : str
            Label for the x-axis (parameter name with % unit)
        - "ylabel" : str
            Label for the y-axis (metric name and unit)
        - "parameter" : str
            Full parameter name that was varied
        - "metric" : str
            Metric that was computed (uppercase)
    Raises
    ------
    ValueError
        If parameter is ambiguous across plants or unrecognized.
    Notes
    -----
    - For "fixed_capital" and "fixed_opex",
    the original value is assumed to be 1.0
    - If a parameter does not exist for a particular plant,
    a flat baseline curve is returned
    - Shorthand parameters are resolved from full nested keys
    (e.g., "CO2" -> "variable_opex_inputs.CO2")
    """
    if not isinstance(plants, (list, tuple)):
        plants = [plants]

    metric = metric.upper()

    # --- Label ---
    if label is None:
        label = _default_metric_label(
            plants[0].currency if plants else r"\$", metric
        )

    # --- Top-level parameters ---
    top_level_keys = [
        "fixed_capital",
        "fixed_opex",
        "project_lifetime",
        "interest_rate",
        "operator_hourly_rate",
    ]

    # --- Nested price keys across all plants ---
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

    # --- Shorthand resolution with ambiguity check ---
    short_to_full = {}
    for plant in plants:
        for k in plant.variable_opex_inputs:
            full = f"variable_opex_inputs.{k}"
            if k in short_to_full and short_to_full[k] != full:
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'.\n"
                    f"Please use full path."
                )
            short_to_full[k] = full

        for k in plant.plant_products:
            full = f"plant_products.{k}"
            if k in short_to_full and short_to_full[k] != full:
                raise ValueError(
                    f"Ambiguous shorthand '{k}' across plants.\n"
                    f"Seen both '{short_to_full[k]}' and '{full}'.\n"
                    f"Please use full path."
                )
            short_to_full[k] = full

    parameter = short_to_full.get(parameter, parameter)

    if parameter not in valid_parameters:
        raise ValueError(f"Unrecognized parameter: {parameter}")

    # --- X axis ---
    pct_changes = np.linspace(
        -plus_minus_value, plus_minus_value, n_points
    )
    pct_axis = pct_changes * 100

    # --- X label ---
    label_clean = _make_label(parameter.split(".")[-1])
    if parameter in top_level_keys:
        x_label = label_clean + r" / [$\pm$ \%]"
    else:
        x_label = label_clean + r" price / [$\pm$ \%]"

    # --- Core computation ---
    results = []

    for i, plant in enumerate(plants):
        # Plant-specific valid parameters
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
            nested_price_keys = var_opex_keys.union(byprod_keys)
        else:
            nested_price_keys = var_opex_keys.union(all_prod_keys)

        plant_valid_params = set(top_level_keys).union(
            nested_price_keys
        )

        # Baseline
        base_value = _evaluate_metric(
            plant, metric, additional_capex
        )

        # If parameter does not exist for this plant,
        # return a flat baseline curve
        if parameter not in plant_valid_params:
            metric_values = np.full_like(
                pct_axis, fill_value=base_value, dtype=float
            )
        else:
            if parameter in ["fixed_capital", "fixed_opex"]:
                original_value = 1.0
            else:
                original_value = _get_original_value(
                    plant, parameter
                )

            param_values = original_value * (1 + pct_changes)

            metric_values = [
                _update_and_evaluate(
                    plant,
                    parameter,
                    v,
                    list(nested_price_keys),
                    metric=metric,
                    additional_capex=additional_capex,
                )
                for v in param_values
            ]

        results.append(
            {
                "plant": getattr(plant, "name", f"Plant {i+1}"),
                "x": pct_axis,
                "y": metric_values,
                "baseline": base_value,
            }
        )

    return {
        "curves": results,
        "xlabel": x_label,
        "ylabel": label,
        "parameter": parameter,
        "metric": metric,
    }


def tornado_data(plant,
                 plus_minus_value,
                 metric="LCOP",
                 label=None,
                 additional_capex: bool = False):
    """
    Generate tornado plot data for sensitivity analysis (no plotting).
    This function performs a sensitivity analysis on a plant model by varying
    key parameters and calculating their impact on a specified metric.
    The results are sorted by total effect magnitude to facilitate tornado
    plot visualization.
    Parameters
    ----------
    plant : Plant
        The plant object containing model parameters and configuration.
    plus_minus_value : float
        The percentage or absolute value to vary each parameter by
        (e.g., 0.1 for ±10%).
    metric : str, optional
        The metric to analyze. Default is "LCOP" (Levelized Cost of Power).
        Common metrics: "LCOP", "LCOH", "IRR", "NPV".
    label : str, optional
        Custom label for the metric on the x-axis. If None, uses default label
        based on currency and metric type.
    additional_capex : bool, optional
        Whether to include additional capital expenditure in calculations.
        Default is False.
    dict
        Dictionary containing tornado plot data with keys:
        - factors : list[str]
            Sorted list of parameter names by
            sensitivity magnitude (ascending).
        - lows : np.ndarray
            Metric values when each factor is reduced
            (sorted by effect size).
        - highs : np.ndarray
            Metric values when each factor is increased
            (sorted by effect size).
        - base_value : float
            Metric value with baseline parameters.
        - labels : list[str]
            Display labels for each factor (sorted by effect size).
        - plus_minus_value : float
            The sensitivity variation used.
        - metric : str
            The analyzed metric in uppercase.
        - xlabel : str
            Label for the x-axis.
    Examples
    --------
    >>> tornado_data = tornado_data(plant, plus_minus_value=0.1, metric="LCOP")
    >>> factors = tornado_data["factors"]
    >>> lows = tornado_data["lows"]
    >>> highs = tornado_data["highs"]
    """
    metric = metric.upper()
    if label is None:
        label = _default_metric_label(plant.currency, metric)

    keys, nested_price_keys = _collect_sensitivity_keys(plant, metric)

    base_value = _evaluate_metric(plant, metric, additional_capex)

    sensitivity_results = _run_tornado_sensitivity(
        plant,
        keys,
        nested_price_keys,
        plus_minus_value,
        metric,
        additional_capex=additional_capex,
    )

    factors = list(sensitivity_results.keys())
    lows = np.array([sensitivity_results[f][0] for f in factors], dtype=float)
    highs = np.array([sensitivity_results[f][1] for f in factors], dtype=float)

    total_effects = np.abs(highs - lows)
    sorted_indices = np.argsort(total_effects)

    factors_sorted = [factors[i] for i in sorted_indices]
    lows_sorted = lows[sorted_indices]
    highs_sorted = highs[sorted_indices]

    labels_sorted = _build_tornado_labels(plant, factors_sorted)

    return {
        "factors": factors_sorted,
        "lows": lows_sorted,
        "highs": highs_sorted,
        "base_value": base_value,
        "labels": labels_sorted,
        "plus_minus_value": plus_minus_value,   # ✅ add this
        "metric": metric,                       # optional
        "xlabel": label,
    }


def monte_carlo(plant,
                num_samples: int = 1_000_000,
                batch_size: int = 1000,
                additional_capex: bool = False):
    """
    This function conducts a probabilistic analysis of a plant's economic
    performance by sampling from distributions of various input parameters
    and computing multiple economic metrics across the samples. The simulation
    processes samples in batches to optimize memory usage while maintaining
    computational efficiency.
        configuration attributes initialized. The plant will be deep copied
        internally to preserve the original state.
        Must be divisible by batch_size for optimal performance.
        Reduces memory usage by processing samples in chunks rather than
        all at once. Smaller batches use less memory but may be slower.
        calculations. Default is False. Only applies when product prices
        are available.
    dict
        A dictionary containing the Monte Carlo simulation results with keys:
        - "name" : str
            Name of the analyzed plant.
        - "metrics" : dict
                (only populated if product prices available).
                (only populated if product prices available).
                (only populated if product prices available).
        - "inputs" : dict
            - Variable opex price samples for each item formatted as
              "{Item} price"
            - Product price samples for each product formatted as
              "{Product} product price"
        - "num_samples" : int
            Number of samples generated in the simulation.
        - "additional_capex" : bool
            Whether additional CAPEX was included in calculations.
        - "currency" : str
            Currency symbol used in the plant's economic calculations.
    - The plant object is deep copied to avoid modifying the original during
      sampling. Results are stored back to the original plant as
    - All input parameters are sampled from truncated normal distributions
      with parameter-specific mean, standard deviation, and bounds.
      are defined in the plant configuration.
    - Progress is displayed via a progress bar showing batch processing status.
    - Fixed capital and fixed opex factors use a standard distribution with
      mean=1, std=0.3, bounds=[0.25, 1.75].
    - Operator hourly rate is sampled from configurable
    distribution parameters.
    - Project lifetime and interest rate use 2-sigma bounds around baseline
      values.
    Raises
    ------
    AttributeError
        If the plant object lacks required economic calculation methods or
        configuration attributes.
    Examples
    --------
    >>> results = monte_carlo(plant, num_samples=10000, batch_size=500)
    >>> lcop_values = results['metrics']['LCOP']
    >>> roi_values = results['metrics']['ROI']
    """
    currency = plant.currency if hasattr(plant, "currency") else r"\$"
    # Ensure plant is baseline-initialized
    plant.calculate_fixed_capital()
    plant.calculate_variable_opex()
    plant.calculate_fixed_opex()
    plant.calculate_cash_flow()
    plant.calculate_levelized_cost()

    num_batches = (num_samples + batch_size - 1) // batch_size

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

    # ---- Sample ALL inputs once ----
    fixed_capitals = _truncated_normal_samples(
        1, 0.3, 0.25, 1.75, num_samples
    )

    fixed_opexs = _truncated_normal_samples(
        1, 0.3, 0.25, 1.75, num_samples
    )

    operator_hourlys = _truncated_normal_samples(
        op_mean, op_std, op_min, op_max, num_samples
    )

    project_lifetimes = _truncated_normal_samples(
        plant.project_lifetime,
        5,
        max(5, plant.project_lifetime - 2 * 5),
        plant.project_lifetime + 2 * 5,
        num_samples,
    )

    interests = _truncated_normal_samples(
        plant.interest_rate,
        0.03,
        max(0.02, plant.interest_rate - 2 * 0.03),
        plant.interest_rate + 2 * 0.03,
        num_samples,
    )

    variable_opex_price_samples = {}
    for item, props in plant.variable_opex_inputs.items():
        mean, std, min_, max_ = _get_sampling_params(props)
        variable_opex_price_samples[item] = _truncated_normal_samples(
            mean, std, min_, max_, num_samples
        )

    have_product_prices = all(
        "price" in props for props in plant.plant_products.values()
    )

    product_price_samples = {}
    if have_product_prices:
        for prod, props in plant.plant_products.items():
            mean, std, min_, max_ = _get_sampling_params(props)
            product_price_samples[prod] = _truncated_normal_samples(
                mean, std, min_, max_, num_samples
            )

    # ---- Batch calculation loop ----
    for b in tqdm(range(num_batches), desc="Monte Carlo"):
        start = b * batch_size
        end = min(start + batch_size, num_samples)

        # Fresh copy for each batch
        plant_copy = deepcopy(plant)

        # ---- Apply sampled inputs ----
        plant_copy.operator_hourly_rate["rate"] = operator_hourlys[start:end]

        plant_copy.update_configuration(
            {
                "project_lifetime": project_lifetimes[start:end],
                "interest_rate": interests[start:end],
            }
        )

        for item in plant.variable_opex_inputs:
            plant_copy.variable_opex_inputs[item]["price"] = (
                variable_opex_price_samples[item][start:end]
            )

        if have_product_prices:
            for prod in plant.plant_products:
                plant_copy.plant_products[prod]["price"] = (
                    product_price_samples[prod][start:end]
                )

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
            mc_metrics["NPV"][start:end] = plant_copy.calculate_npv()
            mc_metrics["ROI"][start:end] = plant_copy.calculate_roi(
                additional_capex=additional_capex
            )
            mc_metrics["PBT"][start:end] = (
                plant_copy.calculate_payback_time(
                    additional_capex=additional_capex
                )
            )

    mc_inputs = {
        "Fixed capital factor": fixed_capitals,
        "Fixed opex factor": fixed_opexs,
        "Operator hourly rate": operator_hourlys,
        "Project lifetime": project_lifetimes,
        "Interest rate": interests,
        **{
            f"{k.replace('_', ' ').title()} price": v
            for k, v in variable_opex_price_samples.items()
        },
        **{
            f"{k.replace('_', ' ').title()} product price": v
            for k, v in product_price_samples.items()
        },
    }

    # ---- Store on plant ----
    plant.monte_carlo_metrics = mc_metrics
    plant.monte_carlo_inputs = mc_inputs

    return {
        "name": plant.name,
        "metrics": mc_metrics,
        "inputs": mc_inputs,
        "num_samples": num_samples,
        "additional_capex": additional_capex,
        "currency": currency,
    }
