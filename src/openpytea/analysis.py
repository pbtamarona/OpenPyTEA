from copy import deepcopy
from tqdm.auto import tqdm
from scipy import stats
import numpy as np

from openpytea.helpers import (_make_label,
                               _get_original_value,
                               _update_and_evaluate,
                               _default_metric_label,
                               _ensure_list,
                               _build_bar_data,
                               _evaluate_metric,
                               _evaluate_baseline_metric,
                               _collect_sensitivity_keys,
                               _run_tornado_sensitivity,
                               _build_tornado_labels,
                               _PROJECT_SCALAR_PARAMS,
                               _parse_dependency_driver,
                               _describe_dependency_node,
                               _collect_dependency_specs,
                               _dependency_parents,
                               _resolve_dependency_dag,
                               _sensitivity_key_node,
                               _node_sensitivity_key)


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
        loc = plant._resolve_loc_factor()
        components = {
            eq.name: float(eq.direct_cost * loc * plant.exchange_rate)
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
        - Each item's annual cost is computed as
            consumption * price * 365 * plant_utilization.
        - Items missing either "consumption" or "price" are skipped.
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
            if "consumption" in props and "price" in props:
                val = (
                    props["consumption"] * props["price"]
                    * 365 * plant.plant_utilization
                )
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


def levelized_cost_data(plants, pct=False):
    """
    Generate levelized cost of production (LCOP) breakdown data for one or
    more plants.
    This function discounts capital costs, cash costs, side-product revenue,
    and production over each plant's project lifetime at its interest rate
    (mirroring ``Plant.calculate_levelized_cost``), then divides the
    discounted CAPEX, OPEX, and side revenue by the discounted production so
    each component is expressed per unit of main product. Side revenue is
    negated (since it is subtracted from the LCOP numerator), so the
    components sum directly to the plant's LCOP: CAPEX + OPEX +
    Side revenue = LCOP.
    Parameters
    ----------
    plants : Plant or list of Plant
        A single plant object or a list of plant objects for which to build
        the levelized cost breakdown.
    pct : bool, optional
        If True, return the breakdown as percentages of the total. If False
        (default), return absolute values.
    Returns
    -------
    dict
        A dictionary containing structured bar chart data with keys:
        - CAPEX
        - OPEX
        - Side revenue
        (each expressed per unit of main product), along with plant names,
        currency, and formatting information.
    Notes
    -----
    - Only the scalar (non-Monte Carlo) case is supported; each plant's
      ``project_lifetime`` and ``interest_rate`` must be scalar values.
    Examples
    --------
    >>> data = levelized_cost_data(plant1)
    >>> data = levelized_cost_data([plant1, plant2], pct=True)
    """
    plants = _ensure_list(plants)
    base_currency = plants[0].currency if plants else r"\$"
    currency = rf"{base_currency}$\cdot$unit$^{{-1}}$"

    components_list = []
    xlabels = []

    for plant in plants:
        plant.calculate_levelized_cost()

        n_years = int(plant.project_lifetime)
        years = np.arange(1, n_years + 1, dtype=float)
        discount_factors = (1 + plant.interest_rate) ** years

        capital_cost = np.asarray(plant.capital_cost_array, dtype=float)[0, :n_years]
        cash_cost = np.asarray(plant.cash_cost_array, dtype=float)[0, :n_years]
        side_rev = np.asarray(plant.side_revenue_array, dtype=float)[0, :n_years]
        prod = np.asarray(plant.prod_array, dtype=float)[0, :n_years]

        disc_capex = float(np.sum(capital_cost / discount_factors))
        disc_opex = float(np.sum(cash_cost / discount_factors))
        disc_side_rev = float(np.sum(side_rev / discount_factors))
        disc_prod = float(np.sum(prod / discount_factors))

        components = {
            "CAPEX": disc_capex / disc_prod,
            "OPEX": disc_opex / disc_prod,
            "Side revenue": -(disc_side_rev / disc_prod),
        }

        components_list.append(components)
        xlabels.append(plant.name)

    return _build_bar_data(components_list, xlabels,
                           "Levelized cost", currency, pct)


def cash_flow_data(plants):
    """
    Prepare cumulative cash flow data for one or more plants, for plotting
    the classic project cash flow diagram (cumulative cash position vs.
    time): a dip into debt during construction/start-up, a minimum
    ("maximum investment"), a break-even point where the curve crosses
    back above zero, and a rise into profit for the remainder of the
    project life.

    Parameters
    ----------
    plants : Plant or list of Plant
        A single plant object or a list of plant objects to build the
        cash flow diagram data for. Each plant's ``calculate_cash_flow``
        is (re)run to ensure the underlying annual cash flow array is
        up to date.

    Returns
    -------
    dict
        A dictionary containing:
        - "curves" : list of dict
            One entry per plant, each containing:
            - "plant" : str
                Plant name.
            - "years" : ndarray
                Time axis from 0 (project start) to the project
                lifetime, one point per year.
            - "cumulative" : ndarray
                Cumulative cash position at each year in ``years``.
            - "max_investment" : float
                Depth of the deepest point of the cumulative cash flow
                curve (0 if the curve never goes negative).
            - "max_investment_year" : float
                Year at which ``max_investment`` occurs.
            - "breakeven_year" : float or None
                Year at which the cumulative cash flow first crosses
                back above zero after having been negative (linearly
                interpolated between the two surrounding years). None
                if the project never goes into debt or never recovers.
            - "payback_time" : float or None
                Alias of ``breakeven_year``.
            - "project_life" : float
                Final year in ``years`` (the plant's project lifetime).
        - "xlabel" : str
            Label for the x-axis.
        - "ylabel" : str
            Label for the y-axis (excluding currency units).
        - "currency" : str
            Currency symbol, taken from the first plant.

    Notes
    -----
    - Only the scalar (non-Monte Carlo) case is supported; if a plant's
      ``cash_flow`` has multiple rows (vectorised inputs), the first row
      is used.
    - The cumulative cash flow already reflects the plant's CAPEX ramp,
      working capital draw/release, production ramp, depreciation, and
      tax lag, as computed by ``Plant.calculate_cash_flow``.

    Examples
    --------
    >>> data = cash_flow_data(plant)
    >>> data = cash_flow_data([plant_a, plant_b])
    """
    plants = _ensure_list(plants)
    currency = plants[0].currency if plants else r"\$"

    curves = []
    for plant in plants:
        plant.calculate_cash_flow()

        cash_flow = np.asarray(plant.cash_flow, dtype=float)[0]
        n_years = cash_flow.shape[0]

        years = np.arange(0, n_years + 1, dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(cash_flow)))

        min_idx = int(np.argmin(cumulative))
        max_investment = max(0.0, -float(cumulative[min_idx]))
        max_investment_year = float(years[min_idx])

        breakeven_year = None
        for i in range(1, len(cumulative)):
            if cumulative[i - 1] < 0 <= cumulative[i]:
                span = cumulative[i] - cumulative[i - 1]
                frac = (-cumulative[i - 1] / span) if span != 0 else 0.0
                breakeven_year = float(years[i - 1] + frac)
                break

        curves.append({
            "plant": plant.name,
            "years": years,
            "cumulative": cumulative,
            "max_investment": max_investment,
            "max_investment_year": max_investment_year,
            "breakeven_year": breakeven_year,
            "payback_time": breakeven_year,
            "project_life": float(years[-1]),
        })

    return {
        "curves": curves,
        "xlabel": "Time / [years]",
        "ylabel": "Cumulative cash flow",
        "currency": currency,
    }


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
          "interest_rate", "operator_hourly_rate", "plant_utilization", or
          "tax_rate"
        - A nested price key: "variable_opex_inputs.{key}" or
          "plant_products.{key}"
        - A process quantity key:
          "variable_opex_inputs.{key}.consumption" or
          "plant_products.{key}.production"
        - A shorthand: "{key}" for a price, or "{key}.consumption" /
          "{key}.production" for a quantity (resolved to the full path if
          unambiguous)

        A parameter whose value is set by a dependency cannot be varied --
        it follows from its parents rather than moving on its own -- and
        raises ``ValueError``. Vary one of its parents instead; the change
        propagates down to it (see Notes).
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
        If parameter is ambiguous across plants, unrecognized, or is set by
        a dependency on one of the plants.
    Notes
    -----
    - For "fixed_capital" and "fixed_opex", the plant's configured
    fc/fp multiplier (1.0 when unset) is perturbed
    - If a parameter does not exist for a particular plant,
    a flat baseline curve is returned
    - Shorthand parameters are resolved from full nested keys
    (e.g., "CO2" -> "variable_opex_inputs.CO2")
    - Parameter dependencies configured on a plant are honoured at every
    point along the curve, exactly as they are in
    :func:`monte_carlo`: varying a parameter that drives others moves them
    with it (and anything downstream of *those*), so the curve shows the
    combined effect rather than the parameter in isolation. The baseline is
    resolved the same way. See
    :func:`~openpytea.helpers._apply_dependencies`.
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
        "plant_utilization",
        "tax_rate",
    ]

    # --- Process quantity keys across all plants ---
    # Consumption/production rates. These are dependency-graph nodes, so
    # varying one here also moves whatever depends on it.
    quantity_keys_all = set(
        f"variable_opex_inputs.{k}.consumption"
        for plant in plants
        for k in plant.variable_opex_inputs
    ).union(
        f"plant_products.{k}.production"
        for plant in plants
        for k in plant.plant_products
    )

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
    ).union(quantity_keys_all)

    # --- Shorthand resolution with ambiguity check ---
    short_to_full = {}
    ambiguous_keys = set()
    for plant in plants:
        for k in plant.variable_opex_inputs:
            full = f"variable_opex_inputs.{k}"
            if k in short_to_full and short_to_full[k] != full:
                ambiguous_keys.add(k)
            else:
                short_to_full[k] = full
            # Quantity shorthand: "<item>.consumption". Unambiguous by
            # construction -- only variable_opex_inputs items are consumed.
            short_to_full[f"{k}.consumption"] = f"{full}.consumption"

        for k in plant.plant_products:
            full = f"plant_products.{k}"
            if k in short_to_full and short_to_full[k] != full:
                ambiguous_keys.add(k)
            else:
                short_to_full[k] = full
            # Likewise "<product>.production" -- only products are produced.
            short_to_full[f"{k}.production"] = f"{full}.production"

    if parameter in ambiguous_keys:
        full_options = set()
        for plant in plants:
            if parameter in plant.variable_opex_inputs:
                full_options.add(f"variable_opex_inputs.{parameter}")
            if parameter in plant.plant_products:
                full_options.add(f"plant_products.{parameter}")
        raise ValueError(
            f"Ambiguous shorthand '{parameter}'.\n"
            f"Seen both {' and '.join(sorted(full_options))}.\n"
            f"Please use full path."
        )

    parameter = short_to_full.get(parameter, parameter)

    if parameter not in valid_parameters:
        raise ValueError(f"Unrecognized parameter: {parameter}")

    # --- Reject a parameter the dependency graph already determines ---
    # Its value is a function of its parents, so "holding everything else
    # constant" while moving it is not a scenario the plant can be in.
    parameter_node = _sensitivity_key_node(parameter)
    if parameter_node is not None:
        for plant in plants:
            dependents = _collect_dependency_specs(plant)
            if parameter_node not in dependents:
                continue
            parents = sorted(
                _node_sensitivity_key(pk)
                for pk in _dependency_parents(
                    dependents[parameter_node], parameter_node
                )
            )
            raise ValueError(
                f"Cannot vary '{parameter}' on plant "
                f"'{getattr(plant, 'name', 'unnamed')}': it is set by a "
                f"dependency on {', '.join(parents)}, so it has no value "
                "of its own to vary. Vary one of those instead -- the "
                f"change propagates through to '{parameter}'."
            )

    # --- X axis ---
    pct_changes = np.linspace(
        -plus_minus_value, plus_minus_value, n_points
    )
    pct_axis = pct_changes * 100

    # --- X label ---
    label_clean = _make_label(parameter.split(".")[-1])
    if parameter in top_level_keys:
        x_label = label_clean + r" / [$\pm$ \%]"
    elif parameter in quantity_keys_all:
        # "variable_opex_inputs.steam.consumption" ->
        # "Steam consumption / [+- %]"
        item, field = parameter.split(".")[1:]
        x_label = (
            f"{_make_label(item)} {field}" + r" / [$\pm$ \%]"
        )
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

        plant_quantity_keys = set(
            f"variable_opex_inputs.{k}.consumption"
            for k in plant.variable_opex_inputs
        ).union(
            f"plant_products.{k}.production" for k in prod_key_list
        )

        plant_valid_params = set(top_level_keys).union(
            nested_price_keys
        ).union(plant_quantity_keys)

        # Baseline, with this plant's dependencies resolved so it lines up
        # with the perturbed points either side of it
        base_value = _evaluate_baseline_metric(
            plant, metric, additional_capex
        )

        # If parameter does not exist for this plant,
        # return a flat baseline curve
        if parameter not in plant_valid_params:
            metric_values = np.full_like(
                pct_axis, fill_value=base_value, dtype=float
            )
        else:
            if parameter == "fixed_capital":
                # Perturb the plant's actual configured multiplier, not
                # an assumed 1.0 (see the tornado twin in helpers)
                original_value = (
                    1.0 if plant.fc is None else plant.fc
                )
            elif parameter == "fixed_opex":
                original_value = (
                    1.0 if plant.fp is None else plant.fp
                )
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
                 additional_capex: bool = False,
                 include_process_params: bool = False):
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
    include_process_params : bool, optional
        Whether to also rank the plant's *process* parameters -- every
        ``variable_opex_inputs`` item's consumption and every
        ``plant_products`` entry's production -- alongside the prices and
        economic scalars. Default is False, which keeps the factor list to
        prices and economic scalars.

        This is independent of the dependency graph: process quantities are
        ordinary economic drivers (a plant's production rate moves LCOP
        whether or not anything is tied to it), so configuring a dependency
        does not switch them on, and switching them on does not require
        one. A quantity that is *set by* a dependency is excluded even when
        this is True -- see Notes.
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
    Notes
    -----
    Parameter dependencies configured on the plant are honoured, exactly as
    they are in :func:`monte_carlo`:

    - Varying a factor that drives others moves them with it (and anything
      downstream of *those*), so its bar shows the combined effect rather
      than the parameter in isolation. The baseline is resolved the same
      way. See :func:`~openpytea.helpers._apply_dependencies`.
    - A parameter that is itself set by a dependency is never a factor --
      it has no value of its own to vary. This drops it from the top-level
      keys (e.g. ``"fixed_capital"`` when ``fixed_capital_factor`` is a
      dependent) and from whatever ``include_process_params`` would
      otherwise add.

    Dependencies do not, however, change *which kinds* of parameter are
    ranked: that is ``include_process_params`` alone. See
    :func:`~openpytea.helpers._collect_sensitivity_keys`.
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

    keys, nested_price_keys = _collect_sensitivity_keys(
        plant, metric, include_process_params=include_process_params
    )

    base_value = _evaluate_baseline_metric(plant, metric, additional_capex)

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


def make_distribution(dist_id, loc=None, scale=None, shape=None,
                       minimum=None, maximum=None):
    """
    Build a frozen ``scipy.stats`` distribution from an OpenPyTEA dist_id.

    Translates the compact ``(dist_id, loc, scale, shape, minimum, maximum)``
    parameterization used throughout the Monte Carlo module into the
    corresponding frozen SciPy distribution object.

    Parameters
    ----------
    dist_id : int
        Distribution family identifier:

        - 2 : Lognormal (``loc``=mu, ``scale``=sigma)
        - 3 : Normal (``loc``=mean, ``scale``=std)
        - 4 : Uniform (``minimum``, ``maximum``)
        - 5 : Triangular (``loc``=mode, ``minimum``, ``maximum``)
        - 6 : Bernoulli (``loc``=p, ``scale``=success value, default 1)
        - 7 : Discrete uniform (``minimum``, ``maximum``, inclusive)
        - 8 : Weibull (``loc``=offset, ``scale``=lambda, ``shape``=k)
        - 9 : Gamma (``loc``=offset, ``scale``=theta, ``shape``=k)
        - 10 : Beta (``loc``=alpha, ``shape``=beta, ``maximum``=upper bound)
        - 11 : GEV (``loc``=mu, ``scale``=sigma, ``shape``=xi)
        - 12 : Student's t (``loc``=median, ``scale``=scale, ``shape``=nu)
    loc : float, optional
        Location parameter; meaning depends on ``dist_id`` (see above).
    scale : float, optional
        Scale parameter; meaning depends on ``dist_id`` (see above).
    shape : float, optional
        Shape parameter, required for Weibull, Gamma, Beta, GEV, and
        Student's t.
    minimum : float, optional
        Lower bound, required for Uniform, Triangular, and Discrete uniform.
    maximum : float, optional
        Upper bound, required for Uniform, Triangular, Discrete uniform,
        and (optionally) Beta.

    Returns
    -------
    scipy.stats distribution
        A frozen distribution instance (continuous ``rv_continuous`` /
        ``rv_discrete``) exposing the usual ``rvs``, ``pdf``/``pmf``, etc.

    Raises
    ------
    ValueError
        If ``dist_id`` is not one of the supported values above (0/1 are
        handled separately by :func:`sample_distribution` as constants).

    See Also
    --------
    sample_distribution : Draws random samples, with optional truncation.
    """
    if dist_id == 2:  # Lognormal: loc=mu, scale=sigma
        return stats.lognorm(s=scale, scale=np.exp(loc))

    elif dist_id == 3:  # Normal: loc=mu, scale=sigma
        return stats.norm(loc=loc, scale=scale)

    elif dist_id == 4:  # Uniform: minimum, maximum
        return stats.uniform(loc=minimum, scale=maximum - minimum)

    elif dist_id == 5:  # Triangular: loc=mode, minimum, maximum
        c = (loc - minimum) / (maximum - minimum)
        return stats.triang(c, loc=minimum, scale=maximum - minimum)

    elif dist_id == 6:  # Bernoulli: loc=p, scale=success value (default 1)
        # Outcomes are 0 (failure) or scale (success), with prob 1-p / p
        p = loc
        success_value = scale if scale is not None else 1.0
        return stats.rv_discrete(name='bernoulli_scaled',
                                values=([0, success_value], [1 - p, p]))

    elif dist_id == 7:  # Discrete uniform: minimum, maximum
        return stats.randint(low=minimum, high=maximum + 1)

    elif dist_id == 8:  # Weibull: loc=offset, scale=lambda, shape=k
        return stats.weibull_min(c=shape, scale=scale, loc=loc)

    elif dist_id == 9:  # Gamma: loc=offset, scale=theta, shape=k
        return stats.gamma(a=shape, scale=scale, loc=loc)

    elif dist_id == 10:  # Beta: loc=alpha, shape=beta, maximum=upper bound
        upper = maximum if maximum is not None else 1.0
        return stats.beta(a=loc, b=shape, scale=upper)

    elif dist_id == 11:  # GEV: loc=mu, scale=sigma, shape=xi (scipy negates xi)
        return stats.genextreme(c=-shape, loc=loc, scale=scale)

    elif dist_id == 12:  # Student's t: loc=median, scale=scale, shape=nu
        return stats.t(df=shape, loc=loc, scale=scale)

    else:
        raise ValueError(f"Unsupported dist_id for make_distribution: {dist_id}")


def sample_distribution(dist_id, size, loc=None, scale=None, shape=None,
                         minimum=None, maximum=None, random_state=None):
    """
    Draw random samples for a Monte Carlo input, with optional truncation.

    Wraps :func:`make_distribution` to generate an array of samples. For
    ``dist_id`` 0 or 1 (fixed/constant values) it returns a constant array
    without touching ``random_state``. When ``minimum``/``maximum`` bounds
    are given for Lognormal, Normal, or Bernoulli (``dist_id`` 2, 3, 6),
    samples are drawn and re-drawn (rejection sampling) until ``size``
    values fall within ``[minimum, maximum]``.

    Parameters
    ----------
    dist_id : int
        Distribution family identifier, see :func:`make_distribution`.
        0 or 1 means "constant value equal to ``loc``".
    size : int
        Number of samples to draw.
    loc : float, optional
        Location parameter, forwarded to :func:`make_distribution`.
    scale : float, optional
        Scale parameter, forwarded to :func:`make_distribution`.
    shape : float, optional
        Shape parameter, forwarded to :func:`make_distribution`.
    minimum : float, optional
        Lower truncation bound (also used as a distribution parameter for
        some families, see :func:`make_distribution`).
    maximum : float, optional
        Upper truncation bound (also used as a distribution parameter for
        some families, see :func:`make_distribution`).
    random_state : numpy.random.Generator or int, optional
        Random state passed to ``scipy.stats``' ``rvs``. Pass a single
        shared ``Generator`` across calls to keep an entire Monte Carlo run
        reproducible from one seed.

    Returns
    -------
    numpy.ndarray
        Array of ``size`` samples.

    Notes
    -----
    Rejection sampling redraws in batches of ``2 * remaining`` until enough
    in-bounds values are collected, so very narrow ``[minimum, maximum]``
    windows relative to the distribution's spread can be slow.

    See Also
    --------
    make_distribution : Builds the underlying frozen SciPy distribution.
    """
    if dist_id in (0, 1):
        return np.full(size, loc if loc is not None else 0.0)

    dist = make_distribution(dist_id, loc=loc, scale=scale, shape=shape,
                              minimum=minimum, maximum=maximum)

    needs_truncation = dist_id in (2, 3, 6) and (
        minimum is not None or maximum is not None
    )
    if not needs_truncation:
        return dist.rvs(size=size, random_state=random_state)

    out = np.empty(size)
    filled = 0
    max_rounds = 200
    for _ in range(max_rounds):
        remaining = size - filled
        draw = dist.rvs(size=remaining * 2, random_state=random_state)
        if minimum is not None:
            draw = draw[draw >= minimum]
        if maximum is not None:
            draw = draw[draw <= maximum]
        n = min(len(draw), remaining)
        out[filled:filled + n] = draw[:n]
        filled += n
        if filled >= size:
            return out
    raise ValueError(
        f"Truncated sampling accepted only {filled}/{size} draws after "
        f"{max_rounds} rounds (dist_id={dist_id}, loc={loc}, "
        f"scale={scale}, minimum={minimum}, maximum={maximum}): the "
        "[minimum, maximum] window excludes nearly all of the "
        "distribution's probability mass. Check the input's baseline "
        "value and its min/max bounds."
    )


def _resolve_scale(cfg, default_scale=0.0):
    """
    Extract the scale/std/noise parameter from an uncertainty config dict.

    ``"noise"``, ``"std"``, and ``"scale"`` are all read here as spellings
    of the same underlying parameter -- this function alone doesn't know
    whether ``cfg`` belongs to a dependent or not, so it accepts all three.
    For a *dependent's* own uncertainty block specifically, the caller
    (:func:`_collect_dependency_nodes`, via
    :func:`_reject_std_scale_for_dependent`) requires ``"noise"`` and
    rejects ``"std"``/``"scale"`` before this function is ever reached for
    that block: there, this value is the standard deviation of the *noise*
    added on top of the item's DAG-implied mean, not the standard deviation
    of the item's own absolute value (which the dependency determines), so
    ``"std"``/``"scale"`` would be actively misleading rather than just an
    alternate spelling. Independent items are unaffected and keep accepting
    all three interchangeably.
    """
    return cfg.get("scale", cfg.get("std", cfg.get("noise", default_scale)))


def _has_uncertainty(cfg):
    """
    True if an uncertainty config dict actually specifies variability: a
    nonzero scale/std/noise value, or an explicit ``dist_id`` (which can
    describe a distribution, like Uniform, that has no separate scale
    parameter of its own at all).
    """
    return _resolve_scale(cfg) > 0 or "dist_id" in cfg


def _resolve_dist_params(cfg, default_loc=0.0, default_scale=0.0,
                          default_min=0, default_max=99999, default_id=3):
    """
    Extract ``(dist_id, loc, scale, shape, minimum, maximum)`` from a config dict.

    Lets Monte Carlo input blocks use whichever field name reads naturally
    for that input (e.g. ``"price"`` or ``"rate"`` instead of ``"loc"``,
    ``"std"``/``"noise"`` instead of ``"scale"``, ``"min"``/``"max"``
    instead of ``"minimum"``/``"maximum"``) while normalizing them to the
    positional arguments expected by :func:`make_distribution` /
    :func:`sample_distribution`.

    Parameters
    ----------
    cfg : dict
        Uncertainty configuration for one input, e.g. an entry from
        ``plant.project_uncertainties``, ``plant.variable_opex_inputs``, or
        ``plant.plant_products``. Recognized keys: ``dist_id``, ``loc``,
        ``mean``, ``price``, ``rate``, ``scale``, ``std``, ``noise``,
        ``shape``, ``minimum``, ``min``, ``maximum``, ``max``.
    default_loc : float, optional
        Fallback for ``loc`` when none of ``loc``/``mean``/``price``/``rate``
        is present in ``cfg``. Default is 0.0.
    default_scale : float, optional
        Fallback for ``scale`` when none of ``scale``/``std``/``noise`` is
        present in ``cfg``. Default is 0.0.
    default_min : float, optional
        Fallback for ``minimum`` when neither ``minimum`` nor ``min`` is
        present in ``cfg``. Default is 0.
    default_max : float, optional
        Fallback for ``maximum`` when neither ``maximum`` nor ``max`` is
        present in ``cfg``. Default is 99999.
    default_id : int, optional
        Fallback distribution id when ``cfg`` has no ``dist_id``. Default
        is 3 (Normal).

    Returns
    -------
    tuple
        ``(dist_id, loc, scale, shape, minimum, maximum)`` ready to unpack
        as arguments to :func:`make_distribution` or
        :func:`sample_distribution`. ``shape`` is ``None`` unless ``cfg``
        sets it explicitly.

    See Also
    --------
    _resolve_scale : Just the scale/std/noise extraction, used on its own
        where the rest of the distribution config isn't needed yet.
    """
    dist_id = cfg.get("dist_id", default_id)
    loc = cfg.get(
        "loc", cfg.get("mean", cfg.get("price", cfg.get("rate", default_loc)))
    )
    scale = _resolve_scale(cfg, default_scale)
    shape = cfg.get("shape")
    minimum = cfg.get("minimum", cfg.get("min", default_min))
    maximum = cfg.get("maximum", cfg.get("max", default_max))
    return dist_id, loc, scale, shape, minimum, maximum


def _resolve_price_dist_params(props):
    """
    Extract price-uncertainty ``(dist_id, loc, scale, shape, minimum,
    maximum)`` for one ``variable_opex_inputs``/``plant_products`` item.

    Preferred form: a nested ``"price_uncertainty"`` sub-dict (same
    ``std``/``min``/``max``/``dist_id`` fields as ``consumption_uncertainty``/
    ``production_uncertainty``), mirroring how every other per-item
    uncertainty block is namespaced. ``"loc"``/``"mean"`` inside it default
    to the item's own ``"price"``, and the default truncation bounds are
    ``baseline ± 2*std`` (floored at 0 for non-negative baselines), the
    same convention consumption/production use -- so negative baselines
    (e.g. disposal credits) and arbitrarily large ones sample correctly.

    Backward-compatible fallback: if ``"price_uncertainty"`` is absent, the
    distribution fields are read directly off ``props`` instead (the
    pre-``price_uncertainty`` layout, where ``dist_id``/``std``/``min``/
    ``max`` sat alongside ``"price"`` at the top level of the item). This
    keeps configs written before ``price_uncertainty`` existed working
    unchanged; it is not the recommended layout for new configs.

    Parameters
    ----------
    props : dict
        One item's full config dict (e.g. ``plant.variable_opex_inputs["electricity"]``
        or ``plant.plant_products["methanol"]``).

    Returns
    -------
    tuple
        ``(dist_id, loc, scale, shape, minimum, maximum)``, as returned by
        :func:`_resolve_dist_params`.
    """
    baseline = props.get("price", 0.0)
    price_cfg = props.get("price_uncertainty")
    cfg = price_cfg if price_cfg is not None else props
    std = _resolve_scale(cfg)
    # Default truncation window is centered on the baseline, the same
    # way consumption/production bounds are -- a hard-coded window would
    # exclude (and hang the rejection sampling on) out-of-range
    # baselines such as negative prices (disposal credits) or large
    # per-unit costs in JPY/IDR-scale currencies. Only a non-negative
    # baseline gets the zero floor; explicit min/max always win.
    default_min = baseline - 2 * std
    if baseline >= 0:
        default_min = max(0.0, default_min)
    return _resolve_dist_params(
        cfg, default_loc=baseline,
        default_min=default_min, default_max=baseline + 2 * std,
    )


def _resolve_rate_dist_params(props):
    """
    Extract operator-hourly-rate uncertainty ``(dist_id, loc, scale,
    shape, minimum, maximum)`` from ``plant.operator_hourly_rate``.

    Preferred form: a nested ``"rate_uncertainty"`` sub-dict (same
    ``std``/``min``/``max``/``dist_id`` fields as
    ``consumption_uncertainty``/``production_uncertainty``/
    ``price_uncertainty``), mirroring how every other per-item
    uncertainty block is namespaced. ``"loc"``/``"mean"`` inside it
    default to the item's own ``"rate"``, and the default truncation
    bounds are ``baseline ± 2*std`` (floored at 0), the same convention
    every other input uses.

    Backward-compatible fallback: if ``"rate_uncertainty"`` is absent,
    the distribution fields are read directly off ``props`` instead (the
    original flat layout, where ``dist_id``/``std``/``min``/``max`` sat
    alongside ``"rate"``). This keeps configs written before
    ``rate_uncertainty`` existed working unchanged; it is not the
    recommended layout for new configs.

    Parameters
    ----------
    props : dict
        The full ``plant.operator_hourly_rate`` dict.

    Returns
    -------
    tuple
        ``(dist_id, loc, scale, shape, minimum, maximum)``, as returned
        by :func:`_resolve_dist_params`.
    """
    baseline = props.get("rate", 38.11)
    rate_cfg = props.get("rate_uncertainty")
    cfg = rate_cfg if rate_cfg is not None else props
    std = _resolve_scale(cfg, 10)
    # Baseline-centered default window, like every other input's -- the
    # old hard-coded [10, 100] hung the rejection sampling for rates
    # above ~130 and silently piled samples under 100 for rates just
    # over it. Explicit min/max always win.
    default_min = baseline - 2 * std
    if baseline >= 0:
        default_min = max(0.0, default_min)
    return _resolve_dist_params(
        cfg, default_loc=baseline, default_scale=10,
        default_min=default_min, default_max=baseline + 2 * std,
    )


def _reject_std_scale_for_dependent(cfg, kind, name):
    """
    Raise ``ValueError`` if a dependent's own uncertainty block sets
    ``"std"``/``"scale"`` -- only ``"noise"`` is accepted there.

    For a dependent, this value is the standard deviation of the noise
    added on top of its DAG-implied mean, not the item's own standard
    deviation (the dependency, not this field, determines that), so
    ``"std"``/``"scale"`` would be actively misleading rather than just an
    alternate spelling of the same thing. Independent items are unaffected:
    their uncertainty still accepts ``"std"``/``"scale"``/``"noise"``
    interchangeably via :func:`_resolve_scale`.
    """
    bad_keys = sorted(set(cfg) & {"std", "scale"})
    if bad_keys:
        raise ValueError(
            f"{_describe_dependency_node(kind, name)} sets {bad_keys}, but "
            "a dependent's own noise must be specified via \"noise\" "
            "instead of \"std\"/\"scale\" -- those describe an item's own "
            "absolute value, which no longer applies once a dependency is "
            "defined."
        )


def _collect_dependency_nodes(plant):
    """
    Scan the plant for every dependency-capable node, returning
    ``(dependents, noise_cfg)`` dicts keyed by ``(kind, name)``.

    The ``dependents`` half is the plant's dependency graph itself, read by
    :func:`~openpytea.helpers._collect_dependency_specs` and shared with the
    deterministic sensitivity/tornado analyses. This function adds the
    Monte-Carlo-only half: each dependent's own *noise* configuration.

    A dependent's noise reuses whichever uncertainty fields that kind
    already has: the matching ``"consumption_uncertainty"``/
    ``"production_uncertainty"`` sub-dict for process nodes, or the
    ``noise``/``dist_id``/etc. fields already sitting alongside
    ``"dependency"`` for project nodes (there is no separate
    ``"project_uncertainty"`` sub-block). Only ``"noise"`` is accepted for
    the scale parameter there -- ``"std"``/``"scale"`` raise ``ValueError``
    (see :func:`_reject_std_scale_for_dependent`).
    """
    dependents = _collect_dependency_specs(plant)
    noise_cfg = {}

    for key in dependents:
        kind, name = key
        if kind == "consumption":
            cfg = plant.variable_opex_inputs[name].get(
                "consumption_uncertainty", {}
            )
        elif kind == "production":
            cfg = plant.plant_products[name].get(
                "production_uncertainty", {}
            )
        elif name == "operator_hourly_rate":
            # Preferred nested "rate_uncertainty" block; fall back to
            # the original flat layout (noise/dist fields alongside
            # "rate") for backward compatibility
            cfg = plant.operator_hourly_rate.get(
                "rate_uncertainty", plant.operator_hourly_rate
            )
        else:
            cfg = plant.project_uncertainties.get(name, {})

        _reject_std_scale_for_dependent(cfg, kind, name)
        if _has_uncertainty(cfg):
            noise_cfg[key] = cfg

    return dependents, noise_cfg


def _ensure_driver_available(plant, key, num_samples, driver_pool):
    """
    True if ``key`` is (or was just lazily seeded as) present in
    ``driver_pool``.

    ``plant_utilization`` and ``tax_rate`` are opt-in: unless configured
    with their own uncertainty (or a dependency), they're never
    independently sampled, so a *reference* to one from another node's
    ``depends_on`` falls back here to a constant at the plant's baseline
    value. Every other node kind is always independently sampled by the
    caller before dependency resolution runs, so it's always already in
    ``driver_pool`` by the time this is reached.
    """
    if key in driver_pool:
        return True
    kind, name = key
    if kind == "project" and name == "plant_utilization":
        driver_pool[key] = np.full(num_samples, plant.plant_utilization)
        return True
    if kind == "project" and name == "tax_rate":
        driver_pool[key] = np.full(num_samples, plant.tax_rate)
        return True
    return False


def _resolve_quantity_dependencies(plant, num_samples, consumption_samples,
                                    production_samples, project_samples, rng):
    """
    Resolve quantities defined as a function of one or more other
    quantities, adding them to ``consumption_samples``/
    ``production_samples``/``project_samples`` in place.

    This is the Monte Carlo face of the plant's dependency graph -- a small
    structural causal model over its process *and* economic parameters.
    Nodes are ``variable_opex_inputs`` consumption, ``plant_products``
    production, and the seven economic scalars in
    :data:`~openpytea.helpers._PROJECT_SCALAR_PARAMS`
    (``plant.project_uncertainties`` entries plus ``operator_hourly_rate``)
    — any of them can be a parent, a dependent, or both, e.g. a higher
    production capacity driving up ``fixed_capital_factor``. A node becomes
    a dependent ("child") by setting ``"consumption_dependency"``/
    ``"production_dependency"`` (process nodes) or ``"dependency"``
    (economic scalars) to a dict with:

    - ``"depends_on"``: a non-empty dict mapping one or more parent
      references (``"production:<product>"``, ``"consumption:<item>"``, or
      ``"project:<param>"``) to their linear weight, e.g.
      ``{"production:methanol": 9.3}``.
    - ``"offset"`` (default 0.0).

    giving ``dependent = sum(weight_i * parent_i) + offset``. The graph
    walk itself lives in
    :func:`~openpytea.helpers._resolve_dependency_dag`, shared with the
    deterministic sensitivity and tornado analyses, so the same
    configuration means the same thing there; this function supplies the
    sampled arrays it works over and the per-dependent noise.
    ``plant_utilization``/``tax_rate`` are the only nodes not always
    already sampled going in (they're opt-in); a reference to one that
    isn't independently varying falls back to a constant at its baseline
    (see :func:`_ensure_driver_available`).

    A dependent may *also* define its matching uncertainty fields: unlike a
    non-dependent (whose uncertainty parameterizes its absolute value,
    ``loc`` defaulting to its baseline), for a dependent this is additive
    noise on top of the DAG mean, with ``loc`` defaulting to 0 and the
    scale parameter required as ``"noise"`` rather than ``"std"``/``"scale"``
    (which raise ``ValueError`` there -- see
    :func:`_reject_std_scale_for_dependent`). Process nodes read this from
    their separate ``"consumption_uncertainty"``/``"production_uncertainty"``
    sub-dict; ``operator_hourly_rate`` from its nested
    ``"rate_uncertainty"`` sub-dict when present (falling back to the
    legacy flat fields alongside ``"rate"``); other economic scalars
    reuse whichever ``noise``/``dist_id``/etc. fields already sit
    alongside ``"dependency"`` (see :func:`_collect_dependency_nodes`).
    The noise is always centered at 0 unless the block sets an explicit
    ``"loc"`` -- the ``"mean"``/``"price"``/``"rate"`` loc aliases are
    ignored here, since they spell the item's own baseline, which the
    dependency determines. Each dependent's noise is drawn
    independently, and because the walk feeds each parent's *final* value
    to its children, noise propagates downstream through the graph.

    Parameters
    ----------
    plant : Plant
        The plant being simulated; read-only here.
    num_samples : int
        Number of Monte Carlo draws.
    consumption_samples : dict
        ``variable_opex_inputs`` samples for every non-dependent item
        (sampled if ``"consumption_uncertainty"`` was set, otherwise a
        constant at baseline); mutated in place with resolved dependents.
    production_samples : dict
        ``plant_products`` samples for every non-dependent item (sampled if
        ``"production_uncertainty"`` was set, otherwise a constant at
        baseline); mutated in place with resolved dependents.
    project_samples : dict
        Samples for every independently-varying economic scalar (see
        :data:`~openpytea.helpers._PROJECT_SCALAR_PARAMS`) already sampled
        by the caller, keyed by param name
        (``plant_utilization``/``tax_rate`` may be absent if not
        independently configured); mutated in place with resolved
        dependents.
    rng : numpy.random.Generator
        Shared RNG, advanced by this call for any dependents' own noise.

    Raises
    ------
    ValueError
        If a ``"depends_on"`` entry is malformed or points at an unknown
        item, or the dependency graph has a cycle.
    """
    dependents, noise_cfg = _collect_dependency_nodes(plant)

    if not dependents:
        return

    # ---- Seed the driver pool from the already-sampled non-dependent
    # nodes (the caller guarantees every one of those has an entry, except
    # possibly plant_utilization/tax_rate -- see _ensure_driver_available) ----
    driver_pool = {}
    for name, arr in consumption_samples.items():
        driver_pool[("consumption", name)] = arr
    for name, arr in production_samples.items():
        driver_pool[("production", name)] = arr
    for name, arr in project_samples.items():
        driver_pool[("project", name)] = arr

    def seed_missing(key):
        return _ensure_driver_available(plant, key, num_samples, driver_pool)

    def add_noise(key, value):
        if key not in noise_cfg:
            return value
        unc = noise_cfg[key]
        noise_std = _resolve_scale(unc)
        dist_id, _, scale, shape, minimum, maximum = _resolve_dist_params(
            unc,
            default_loc=0.0,
            default_scale=noise_std,
            default_min=-2 * noise_std,
            default_max=2 * noise_std,
        )
        # Noise is centered on the DAG-implied mean, so only an explicit
        # "loc" may shift it. The "mean"/"price"/"rate" loc aliases spell
        # an item's own baseline (which the dependency determines here)
        # and may sit in the same dict -- e.g. operator_hourly_rate's
        # flat layout -- so reading them as the noise center would bias
        # the noise by the baseline, or hang the truncated sampling when
        # [-2*noise_std, 2*noise_std] lies far from it.
        loc = unc.get("loc", 0.0)
        return value + sample_distribution(
            dist_id, num_samples, loc=loc, scale=scale, shape=shape,
            minimum=minimum, maximum=maximum, random_state=rng,
        )

    resolved = _resolve_dependency_dag(
        dependents, driver_pool, seed_missing=seed_missing, noise=add_noise,
    )

    for (kind, name), value in resolved.items():
        if kind == "consumption":
            consumption_samples[name] = value
        elif kind == "production":
            production_samples[name] = value
        else:
            project_samples[name] = value


def monte_carlo(plant,
                 num_samples: int = 1_000_000,
                 batch_size: int = 1000,
                 additional_capex: bool = False,
                 random_seed: int = None):
    """
    Run a Monte Carlo uncertainty simulation over a plant's financial metrics.

    Samples every configured uncertain input (project-level factors such as
    fixed capital/OPEX, project lifetime, and interest rate; optionally
    plant utilization and tax rate; variable OPEX item prices and,
    optionally, consumption rates; and product prices and, optionally,
    production rates) and re-evaluates the plant's economics ``num_samples``
    times, producing a distribution of outcomes for LCOP and, when product
    prices are configured, NPV, ROI, and payback time.

    Parameters
    ----------
    plant : Plant
        A configured :class:`~openpytea.plant.Plant`. Uncertainty ranges are
        read from ``plant.project_uncertainties``, ``plant.operator_hourly_rate``,
        ``plant.variable_opex_inputs``, and ``plant.plant_products`` — see
        the Monte Carlo section of the user guide for the configuration
        format. The plant is first baseline-initialized (fixed capital,
        variable/fixed OPEX, cash flow, levelized cost) but is not mutated
        by the simulation itself; each batch operates on a deep copy.

        Each entry in ``variable_opex_inputs`` samples its ``"price"`` by
        default (as before); adding a ``"consumption_uncertainty"`` dict
        (with the same ``std``/``min``/``max``/``dist_id`` fields used
        elsewhere) additionally samples that item's ``"consumption"`` around
        its configured baseline value. Likewise, each entry in
        ``plant_products`` samples ``"price"`` when all products have one
        configured, and a ``"production_uncertainty"`` dict additionally
        samples its ``"production"``. Both uncertainty sub-blocks are
        opt-in for actual *variability* — omitted, consumption/production
        stay fixed at their baseline value — but every item's consumption
        and every product's production is always included in
        ``"inputs"`` (as a constant when no uncertainty is configured), so
        the full set of process parameters is always visible, e.g. via
        :func:`~openpytea.plotting.plot_monte_carlo_inputs`.

        A ``"consumption"``/``"production"`` value can also be made a
        function of one or more *other* quantities via
        ``"consumption_dependency"``/``"production_dependency"``: a dict
        with ``"depends_on"`` — a non-empty dict mapping one or more parent
        references to their linear weight — plus an optional ``"offset"``
        (default 0.0), giving ``dependent = sum(weight_i * parent_i) +
        offset``. A parent reference is ``"production:<product>"``,
        ``"consumption:<item>"``, or ``"project:<param>"`` — the last
        naming one of the seven economic scalars in
        :data:`_PROJECT_SCALAR_PARAMS` (the six ``project_uncertainties``
        entries plus ``operator_hourly_rate``), which are dependency-capable
        the same way via a ``"dependency"`` key of their own — so a process
        parameter and an economic one can drive each other in either
        direction, e.g. ``fixed_capital_factor`` scaling up with production
        capacity. Together, every dependent (process or economic) forms a
        small DAG (nodes are these parameters, edges are the ``depends_on``
        references) that's resolved in topological order — chains and
        multi-parent nodes work automatically, using each parent's own
        *final* value (its mean plus any noise of its own), so noise
        propagates downstream through the graph; a cycle raises
        ``ValueError``. A dependent may *also* define its matching
        uncertainty fields (the ``*_uncertainty`` sub-dict for process
        nodes; the same ``noise``/``dist_id``/etc. fields already sitting
        alongside ``"dependency"`` for economic scalars): unlike a
        non-dependent, this becomes additive noise on top of the DAG-implied
        mean (``loc`` defaults to 0, not the baseline, and the scale
        parameter must be ``"noise"`` — ``"std"``/``"scale"`` raise
        ``ValueError`` for a dependent). Each dependent's noise is drawn
        independently. See :func:`_resolve_quantity_dependencies` and
        :func:`_collect_dependency_nodes`.
    num_samples : int, optional
        Total number of Monte Carlo draws. Default is 1,000,000.
    batch_size : int, optional
        Number of samples evaluated per batch (each batch deep-copies the
        plant and vectorizes the economic calculations over the batch).
        Larger values are faster but use more memory. Default is 1000.
    additional_capex : bool, optional
        Whether to include additional CAPEX events when computing ROI and
        payback time. Default is False.
    random_seed : int, optional
        Seed for the single ``numpy.random.Generator`` shared across all
        parameter draws, for reproducible runs. Default is None
        (nondeterministic).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"name"`` : the plant's name.
        - ``"metrics"`` : dict mapping ``"LCOP"``, ``"ROI"``, ``"NPV"``,
          ``"PBT"`` to ``numpy.ndarray`` of length ``num_samples`` (ROI,
          NPV, PBT stay zero-filled if product prices aren't configured).
        - ``"inputs"`` : dict mapping each sampled input's display name to
          its ``numpy.ndarray`` of drawn values.
        - ``"num_samples"`` : the requested sample count.
        - ``"additional_capex"`` : the flag used for ROI/PBT.
        - ``"currency"`` : the plant's currency symbol.

    Notes
    -----
    - The same results are also stored on the plant as
      ``plant.monte_carlo_metrics`` and ``plant.monte_carlo_inputs`` for use
      by :func:`~openpytea.plotting.plot_monte_carlo` and
      :func:`~openpytea.plotting.plot_monte_carlo_inputs`.
    - All inputs are sampled once up front (in a fixed order, from one
      shared RNG) and then consumed batch-by-batch, so results are
      reproducible for a given ``random_seed`` regardless of ``batch_size``.

    See Also
    --------
    sample_distribution : Underlying per-input sampling routine.
    """
    currency = plant.currency if hasattr(plant, "currency") else r"\$"

    # Ensure plant is baseline-initialized
    plant.calculate_fixed_capital()
    plant.calculate_variable_opex()
    plant.calculate_fixed_opex()
    plant.calculate_cash_flow()
    plant.calculate_levelized_cost()

    num_batches = (num_samples + batch_size - 1) // batch_size

    # ---- Single shared RNG for full reproducibility ----
    # One Generator is created here and passed to every sample_distribution()
    # call below, in a fixed order, so it advances its internal state once
    # per draw rather than being reseeded each time (reseeding each call
    # would make every parameter draw the same underlying sequence).
    rng = np.random.default_rng(random_seed)

    # ---- Allocate arrays for ALL metrics ----
    mc_metrics = {
        "LCOP": np.zeros(num_samples),
        "ROI": np.zeros(num_samples),
        "NPV": np.zeros(num_samples),
        "PBT": np.zeros(num_samples),
    }

    # ---- Resolve project uncertainty parameters ----
    pu = plant.project_uncertainties

    fc_id, fc_loc, fc_scale, fc_shape, fc_min, fc_max = _resolve_dist_params(
        pu.get("fixed_capital_factor", {}),
        default_loc=1, default_scale=0.3, default_min=0.25, default_max=1.75,
    )

    fo_id, fo_loc, fo_scale, fo_shape, fo_min, fo_max = _resolve_dist_params(
        pu.get("fixed_opex_factor", {}),
        default_loc=1, default_scale=0.3, default_min=0.25, default_max=1.75,
    )

    lt_cfg = pu.get("project_lifetime", {})
    lt_std_default = _resolve_scale(lt_cfg, 5)
    lt_id, lt_loc, lt_scale, lt_shape, lt_min, lt_max = _resolve_dist_params(
        lt_cfg,
        default_loc=plant.project_lifetime,
        default_scale=lt_std_default,
        default_min=max(5, plant.project_lifetime - 2 * lt_std_default),
        default_max=plant.project_lifetime + 2 * lt_std_default,
    )

    ir_cfg = pu.get("interest_rate", {})
    ir_std_default = _resolve_scale(ir_cfg, 0.03)
    ir_id, ir_loc, ir_scale, ir_shape, ir_min, ir_max = _resolve_dist_params(
        ir_cfg,
        default_loc=plant.interest_rate,
        default_scale=ir_std_default,
        default_min=max(0.02, plant.interest_rate - 2 * ir_std_default),
        default_max=plant.interest_rate + 2 * ir_std_default,
    )

    # ---- project_samples collects every independently-sampled economic
    # scalar (see _PROJECT_SCALAR_PARAMS), feeding the dependency DAG below
    # both as potential drivers and to receive any of these seven that are
    # themselves dependents. A param with its own "dependency" key skips
    # sampling here entirely -- _resolve_quantity_dependencies fills it in.
    project_samples = {}

    pu_util_cfg = pu.get("plant_utilization", {})
    pu_util_std = _resolve_scale(pu_util_cfg, 0)
    pu_util_is_dependent = pu_util_cfg.get("dependency") is not None
    if not pu_util_is_dependent and _has_uncertainty(pu_util_cfg):
        pu_util_mean = plant.plant_utilization
        (util_id, util_loc, util_scale, util_shape,
         util_min, util_max) = _resolve_dist_params(
            pu_util_cfg,
            default_loc=pu_util_mean,
            default_scale=pu_util_std,
            default_min=max(0.0, pu_util_mean - 2 * pu_util_std),
            default_max=min(1.0, pu_util_mean + 2 * pu_util_std),
        )
        project_samples["plant_utilization"] = sample_distribution(
            util_id, num_samples, loc=util_loc, scale=util_scale,
            shape=util_shape, minimum=util_min, maximum=util_max,
            random_state=rng,
        )

    tr_cfg = pu.get("tax_rate", {})
    tr_std = _resolve_scale(tr_cfg, 0)
    tr_is_dependent = tr_cfg.get("dependency") is not None
    if not tr_is_dependent and _has_uncertainty(tr_cfg):
        tr_mean = plant.tax_rate
        (tr_id, tr_loc, tr_scale, tr_shape,
         tr_min, tr_max) = _resolve_dist_params(
            tr_cfg,
            default_loc=tr_mean,
            default_scale=tr_std,
            default_min=max(0.0, tr_mean - 2 * tr_std),
            default_max=min(1.0, tr_mean + 2 * tr_std),
        )
        project_samples["tax_rate"] = sample_distribution(
            tr_id, num_samples, loc=tr_loc, scale=tr_scale,
            shape=tr_shape, minimum=tr_min, maximum=tr_max,
            random_state=rng,
        )

    # ---- Operator hourly rate ----
    op_cfg = plant.operator_hourly_rate
    op_id, op_loc, op_scale, op_shape, op_min, op_max = (
        _resolve_rate_dist_params(op_cfg)
    )

    # ---- Sample ALL inputs once (skipping any that are themselves a
    # dependent, deferred to _resolve_quantity_dependencies below) ----
    if pu.get("fixed_capital_factor", {}).get("dependency") is None:
        project_samples["fixed_capital_factor"] = sample_distribution(
            fc_id, num_samples, loc=fc_loc, scale=fc_scale, shape=fc_shape,
            minimum=fc_min, maximum=fc_max, random_state=rng,
        )

    if pu.get("fixed_opex_factor", {}).get("dependency") is None:
        project_samples["fixed_opex_factor"] = sample_distribution(
            fo_id, num_samples, loc=fo_loc, scale=fo_scale, shape=fo_shape,
            minimum=fo_min, maximum=fo_max, random_state=rng,
        )

    if op_cfg.get("dependency") is None:
        project_samples["operator_hourly_rate"] = sample_distribution(
            op_id, num_samples, loc=op_loc, scale=op_scale, shape=op_shape,
            minimum=op_min, maximum=op_max, random_state=rng,
        )

    if pu.get("project_lifetime", {}).get("dependency") is None:
        project_samples["project_lifetime"] = sample_distribution(
            lt_id, num_samples, loc=lt_loc, scale=lt_scale, shape=lt_shape,
            minimum=lt_min, maximum=lt_max, random_state=rng,
        )

    if pu.get("interest_rate", {}).get("dependency") is None:
        project_samples["interest_rate"] = sample_distribution(
            ir_id, num_samples, loc=ir_loc, scale=ir_scale, shape=ir_shape,
            minimum=ir_min, maximum=ir_max, random_state=rng,
        )

    variable_opex_price_samples = {}
    variable_opex_consumption_samples = {}
    for item, props in plant.variable_opex_inputs.items():
        (v_id, v_loc, v_scale, v_shape,
         v_min, v_max) = _resolve_price_dist_params(props)
        variable_opex_price_samples[item] = sample_distribution(
            v_id, num_samples, loc=v_loc, scale=v_scale, shape=v_shape,
            minimum=v_min, maximum=v_max, random_state=rng,
        )

        cons_cfg = props.get("consumption_uncertainty", {})
        cons_std = _resolve_scale(cons_cfg, 0)
        has_cons_uncertainty = _has_uncertainty(cons_cfg)

        # .get(...) is not None, matching _collect_dependency_specs: an
        # explicit "consumption_dependency": None means no dependency,
        # so the item must still be sampled here
        if props.get("consumption_dependency") is not None:
            # Deterministic (DAG) mean, plus any noise on top of it, is
            # resolved later by _resolve_quantity_dependencies.
            continue

        cons_baseline = props.get("consumption", 0)
        if has_cons_uncertainty:
            (c_id, c_loc, c_scale, c_shape,
             c_min, c_max) = _resolve_dist_params(
                cons_cfg,
                default_loc=cons_baseline,
                default_scale=cons_std,
                default_min=max(0.0, cons_baseline - 2 * cons_std),
                default_max=cons_baseline + 2 * cons_std,
            )
            variable_opex_consumption_samples[item] = sample_distribution(
                c_id, num_samples, loc=c_loc, scale=c_scale, shape=c_shape,
                minimum=c_min, maximum=c_max, random_state=rng,
            )
        else:
            # No uncertainty configured: still report a constant so this
            # process parameter always shows up alongside its price in the
            # Monte Carlo inputs/plots (dist_id 1 is a no-op for `rng`).
            variable_opex_consumption_samples[item] = sample_distribution(
                1, num_samples, loc=cons_baseline, random_state=rng,
            )

    have_product_prices = all(
        "price" in props for props in plant.plant_products.values()
    )

    product_price_samples = {}
    if have_product_prices:
        for prod, props in plant.plant_products.items():
            (p_id, p_loc, p_scale, p_shape,
             p_min, p_max) = _resolve_price_dist_params(props)
            product_price_samples[prod] = sample_distribution(
                p_id, num_samples, loc=p_loc, scale=p_scale, shape=p_shape,
                minimum=p_min, maximum=p_max, random_state=rng,
            )

    product_production_samples = {}
    for prod, props in plant.plant_products.items():
        prod_cfg = props.get("production_uncertainty", {})
        prod_std = _resolve_scale(prod_cfg, 0)
        has_prod_uncertainty = _has_uncertainty(prod_cfg)

        # see the consumption gate above: None means no dependency
        if props.get("production_dependency") is not None:
            # Deterministic (DAG) mean, plus any noise on top of it, is
            # resolved later by _resolve_quantity_dependencies.
            continue

        prod_baseline = props.get("production", 0)
        if has_prod_uncertainty:
            (pp_id, pp_loc, pp_scale, pp_shape,
             pp_min, pp_max) = _resolve_dist_params(
                prod_cfg,
                default_loc=prod_baseline,
                default_scale=prod_std,
                default_min=max(0.0, prod_baseline - 2 * prod_std),
                default_max=prod_baseline + 2 * prod_std,
            )
            product_production_samples[prod] = sample_distribution(
                pp_id, num_samples, loc=pp_loc, scale=pp_scale, shape=pp_shape,
                minimum=pp_min, maximum=pp_max, random_state=rng,
            )
        else:
            # No uncertainty configured: still report a constant so this
            # process parameter always shows up alongside its price in the
            # Monte Carlo inputs/plots (dist_id 1 is a no-op for `rng`).
            product_production_samples[prod] = sample_distribution(
                1, num_samples, loc=prod_baseline, random_state=rng,
            )

    _resolve_quantity_dependencies(
        plant,
        num_samples,
        variable_opex_consumption_samples,
        product_production_samples,
        project_samples,
        rng,
    )

    # ---- Every param in _PROJECT_SCALAR_PARAMS is now resolved in
    # project_samples, either sampled independently above or as a DAG
    # dependent; plant_utilization/tax_rate stay absent (None) if neither ----
    fixed_capitals = project_samples["fixed_capital_factor"]
    fixed_opexs = project_samples["fixed_opex_factor"]
    operator_hourlys = project_samples["operator_hourly_rate"]
    project_lifetimes = project_samples["project_lifetime"]
    interests = project_samples["interest_rate"]
    plant_utilizations = project_samples.get("plant_utilization")
    tax_rates = project_samples.get("tax_rate")

    # ---- Batch calculation loop ----
    for b in tqdm(range(num_batches), desc="Monte Carlo"):
        start = b * batch_size
        end = min(start + batch_size, num_samples)

        # Fresh copy for each batch
        plant_copy = deepcopy(plant)

        # ---- Apply sampled inputs ----
        plant_copy.operator_hourly_rate["rate"] = operator_hourlys[start:end]

        scalar_updates = {
            "project_lifetime": project_lifetimes[start:end],
            "interest_rate": interests[start:end],
        }
        if plant_utilizations is not None:
            scalar_updates["plant_utilization"] = plant_utilizations[start:end]
        if tax_rates is not None:
            scalar_updates["tax_rate"] = tax_rates[start:end]
        plant_copy.update_configuration(scalar_updates)

        for item in plant.variable_opex_inputs:
            plant_copy.variable_opex_inputs[item]["price"] = (
                variable_opex_price_samples[item][start:end]
            )
            plant_copy.variable_opex_inputs[item]["consumption"] = (
                variable_opex_consumption_samples[item][start:end]
            )

        if have_product_prices:
            for prod in plant.plant_products:
                plant_copy.plant_products[prod]["price"] = (
                    product_price_samples[prod][start:end]
                )

        for prod in product_production_samples:
            plant_copy.plant_products[prod]["production"] = (
                product_production_samples[prod][start:end]
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
        **({} if plant_utilizations is None
           else {"Plant utilization": plant_utilizations}),
        **({} if tax_rates is None
           else {"Tax rate": tax_rates}),
        **{
            f"{k.replace('_', ' ').title()} price": v
            for k, v in variable_opex_price_samples.items()
        },
        **{
            f"{k.replace('_', ' ').title()} consumption": v
            for k, v in variable_opex_consumption_samples.items()
        },
        **{
            f"{k.replace('_', ' ').title()} product price": v
            for k, v in product_price_samples.items()
        },
        **{
            f"{k.replace('_', ' ').title()} production": v
            for k, v in product_production_samples.items()
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
