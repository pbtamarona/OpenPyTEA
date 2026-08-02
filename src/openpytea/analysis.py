from copy import deepcopy
from tqdm import tqdm
from scipy import stats
import numpy as np

from openpytea.helpers import (_make_label,
                               _get_original_value,
                               _update_and_evaluate,
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
    ambiguous_keys = set()
    for plant in plants:
        for k in plant.variable_opex_inputs:
            full = f"variable_opex_inputs.{k}"
            if k in short_to_full and short_to_full[k] != full:
                ambiguous_keys.add(k)
            else:
                short_to_full[k] = full

        for k in plant.plant_products:
            full = f"plant_products.{k}"
            if k in short_to_full and short_to_full[k] != full:
                ambiguous_keys.add(k)
            else:
                short_to_full[k] = full

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
    while filled < size:
        remaining = size - filled
        draw = dist.rvs(size=remaining * 2, random_state=random_state)
        if minimum is not None:
            draw = draw[draw >= minimum]
        if maximum is not None:
            draw = draw[draw <= maximum]
        n = min(len(draw), remaining)
        out[filled:filled + n] = draw[:n]
        filled += n
    return out


def _resolve_dist_params(cfg, default_loc=0.0, default_scale=0.0,
                          default_min=0, default_max=99999, default_id=3):
    """
    Extract ``(dist_id, loc, scale, shape, minimum, maximum)`` from a config dict.

    Lets Monte Carlo input blocks use whichever field name reads naturally
    for that input (e.g. ``"price"`` or ``"rate"`` instead of ``"loc"``,
    ``"std"`` instead of ``"scale"``, ``"min"``/``"max"`` instead of
    ``"minimum"``/``"maximum"``) while normalizing them to the positional
    arguments expected by :func:`make_distribution` /
    :func:`sample_distribution`.

    Parameters
    ----------
    cfg : dict
        Uncertainty configuration for one input, e.g. an entry from
        ``plant.project_uncertainties``, ``plant.variable_opex_inputs``, or
        ``plant.plant_products``. Recognized keys: ``dist_id``, ``loc``,
        ``mean``, ``price``, ``rate``, ``scale``, ``std``, ``shape``,
        ``minimum``, ``min``, ``maximum``, ``max``.
    default_loc : float, optional
        Fallback for ``loc`` when none of ``loc``/``mean``/``price``/``rate``
        is present in ``cfg``. Default is 0.0.
    default_scale : float, optional
        Fallback for ``scale`` when neither ``scale`` nor ``std`` is present
        in ``cfg``. Default is 0.0.
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
    """
    dist_id = cfg.get("dist_id", default_id)
    loc = cfg.get(
        "loc", cfg.get("mean", cfg.get("price", cfg.get("rate", default_loc)))
    )
    scale = cfg.get("scale", cfg.get("std", default_scale))
    shape = cfg.get("shape")
    minimum = cfg.get("minimum", cfg.get("min", default_min))
    maximum = cfg.get("maximum", cfg.get("max", default_max))
    return dist_id, loc, scale, shape, minimum, maximum


def monte_carlo(plant,
                 num_samples: int = 1_000_000,
                 batch_size: int = 1000,
                 additional_capex: bool = False,
                 random_seed: int = None):
    """
    Run a Monte Carlo uncertainty simulation over a plant's financial metrics.

    Samples every configured uncertain input (project-level factors such as
    fixed capital/OPEX, project lifetime, and interest rate; optionally
    plant utilization and tax rate; variable OPEX item prices; and product
    prices) and re-evaluates the plant's economics ``num_samples`` times,
    producing a distribution of outcomes for LCOP and, when product prices
    are configured, NPV, ROI, and payback time.

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
    lt_std_default = lt_cfg.get("std", 5)
    lt_id, lt_loc, lt_scale, lt_shape, lt_min, lt_max = _resolve_dist_params(
        lt_cfg,
        default_loc=plant.project_lifetime,
        default_scale=lt_std_default,
        default_min=max(5, plant.project_lifetime - 2 * lt_std_default),
        default_max=plant.project_lifetime + 2 * lt_std_default,
    )

    ir_cfg = pu.get("interest_rate", {})
    ir_std_default = ir_cfg.get("std", 0.03)
    ir_id, ir_loc, ir_scale, ir_shape, ir_min, ir_max = _resolve_dist_params(
        ir_cfg,
        default_loc=plant.interest_rate,
        default_scale=ir_std_default,
        default_min=max(0.02, plant.interest_rate - 2 * ir_std_default),
        default_max=plant.interest_rate + 2 * ir_std_default,
    )

    pu_util_cfg = pu.get("plant_utilization", {})
    pu_util_std = pu_util_cfg.get("std", 0)
    if pu_util_std > 0 or "dist_id" in pu_util_cfg:
        pu_util_mean = plant.plant_utilization
        (util_id, util_loc, util_scale, util_shape,
         util_min, util_max) = _resolve_dist_params(
            pu_util_cfg,
            default_loc=pu_util_mean,
            default_scale=pu_util_std,
            default_min=max(0.0, pu_util_mean - 2 * pu_util_std),
            default_max=min(1.0, pu_util_mean + 2 * pu_util_std),
        )
        plant_utilizations = sample_distribution(
            util_id, num_samples, loc=util_loc, scale=util_scale,
            shape=util_shape, minimum=util_min, maximum=util_max,
            random_state=rng,
        )
    else:
        plant_utilizations = None

    tr_cfg = pu.get("tax_rate", {})
    tr_std = tr_cfg.get("std", 0)
    if tr_std > 0 or "dist_id" in tr_cfg:
        tr_mean = plant.tax_rate
        (tr_id, tr_loc, tr_scale, tr_shape,
         tr_min, tr_max) = _resolve_dist_params(
            tr_cfg,
            default_loc=tr_mean,
            default_scale=tr_std,
            default_min=max(0.0, tr_mean - 2 * tr_std),
            default_max=min(1.0, tr_mean + 2 * tr_std),
        )
        tax_rates = sample_distribution(
            tr_id, num_samples, loc=tr_loc, scale=tr_scale,
            shape=tr_shape, minimum=tr_min, maximum=tr_max,
            random_state=rng,
        )
    else:
        tax_rates = None

    # ---- Operator hourly rate ----
    op_cfg = plant.operator_hourly_rate
    op_id, op_loc, op_scale, op_shape, op_min, op_max = _resolve_dist_params(
        op_cfg, default_loc=38.11, default_scale=10, default_min=10, default_max=100,
    )

    # ---- Sample ALL inputs once ----
    fixed_capitals = sample_distribution(
        fc_id, num_samples, loc=fc_loc, scale=fc_scale, shape=fc_shape,
        minimum=fc_min, maximum=fc_max, random_state=rng,
    )

    fixed_opexs = sample_distribution(
        fo_id, num_samples, loc=fo_loc, scale=fo_scale, shape=fo_shape,
        minimum=fo_min, maximum=fo_max, random_state=rng,
    )

    operator_hourlys = sample_distribution(
        op_id, num_samples, loc=op_loc, scale=op_scale, shape=op_shape,
        minimum=op_min, maximum=op_max, random_state=rng,
    )

    project_lifetimes = sample_distribution(
        lt_id, num_samples, loc=lt_loc, scale=lt_scale, shape=lt_shape,
        minimum=lt_min, maximum=lt_max, random_state=rng,
    )

    interests = sample_distribution(
        ir_id, num_samples, loc=ir_loc, scale=ir_scale, shape=ir_shape,
        minimum=ir_min, maximum=ir_max, random_state=rng,
    )

    variable_opex_price_samples = {}
    for item, props in plant.variable_opex_inputs.items():
        (v_id, v_loc, v_scale, v_shape,
         v_min, v_max) = _resolve_dist_params(props)
        variable_opex_price_samples[item] = sample_distribution(
            v_id, num_samples, loc=v_loc, scale=v_scale, shape=v_shape,
            minimum=v_min, maximum=v_max, random_state=rng,
        )

    have_product_prices = all(
        "price" in props for props in plant.plant_products.values()
    )

    product_price_samples = {}
    if have_product_prices:
        for prod, props in plant.plant_products.items():
            (p_id, p_loc, p_scale, p_shape,
             p_min, p_max) = _resolve_dist_params(props)
            product_price_samples[prod] = sample_distribution(
                p_id, num_samples, loc=p_loc, scale=p_scale, shape=p_shape,
                minimum=p_min, maximum=p_max, random_state=rng,
            )

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
        **({} if plant_utilizations is None
           else {"Plant utilization": plant_utilizations}),
        **({} if tax_rates is None
           else {"Tax rate": tax_rates}),
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
