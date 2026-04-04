from scipy.stats import truncnorm
from copy import deepcopy
from pathlib import Path
import numpy as np
import json
import re


# HELPER FUNCTIONS
# For plottings
def _make_label(s: str) -> str:
    """
    Convert a string to a label format by replacing underscores with spaces
    and capitalizing the first character.

    Preserves LaTeX math segments (text enclosed in $...$) without
    modification, while replacing underscores with spaces in non-math segments.

    Parameters
    ----------
    s : str
        Input string that may contain underscores and LaTeX math expressions.

    Returns
    -------
    str
        A formatted label string with underscores replaced by spaces
        (outside math segments) and the first character capitalized.

    Examples
    --------
    >>> _make_label("my_variable_$x^2$")
    'My variable $x^2$'
    >>> _make_label("cost_per_unit")
    'Cost per unit'
    """
    parts = re.split(r"(\$.*?\$)", s)  # keep math segments
    parts = [
        p.replace("_", " ") if not p.startswith("$") else p
        for p in parts
    ]
    s = "".join(parts)
    return s[:1].upper() + s[1:]


def _default_metric_label(currency: str, metric: str) -> str:
    """
    Generate a default metric label for a given metric name.

    Parameters
    ----------
    currency : str
        The currency code to include in the label (e.g., 'USD', 'EUR').
    metric : str
        The name of the metric to generate a label for. Case-insensitive.
        Supported metrics: 'LCOP', 'ROI', 'NPV',
        'PBT'/'PAYBACK'/'PAYBACK_TIME', 'IRR'.

    Returns
    -------
    str
        A formatted label string for the metric,
        including units where applicable.
        - 'LCOP': Levelized cost with units [currency·unit⁻¹]
        - 'ROI': Return on investment with units [%]
        - 'NPV': Net present value with units [currency]
        - 'PBT', 'PAYBACK', 'PAYBACK_TIME': Payback time with units [years]
        - 'IRR': Internal rate of return with units [-]
        - Any other metric: Returns the uppercase version of the input metric

    Examples
    --------
    >>> _default_metric_label('USD', 'lcop')
    'Levelized cost / [USD$\\cdot$unit$^{-1}$]'
    >>> _default_metric_label('USD', 'roi')
    'Return on investment / [\\%]'
    >>> _default_metric_label('USD', 'payback_time')
    'Payback time / [years]'
    """
    metric = metric.upper()
    if metric == "LCOP":
        return rf"Levelized cost / [{currency}$\cdot$unit$^{-1}$]"
    elif metric == "ROI":
        return r"Return on investment / [\%]"
    elif metric == "NPV":
        return rf"Net present value / [{currency}]"
    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        return "Payback time / [years]"
    elif metric == "IRR":
        return "Internal rate of return / [-]"
    return metric


def _build_tornado_labels(plant, factors):
    """
    Build a mapping of factor names to display labels for tornado diagrams.
    This function creates human-readable labels for sensitivity analysis
    factors by mapping technical parameter names to descriptive labels.
    It handles predefined factors (fixed costs, rates, etc.) and dynamically
    generates labels for variable operating expense inputs and plant product
    prices.
    Args:
        plant: A plant object containing variable_opex_inputs
        and plant_products attributes. factors (list): A list of factor names
        (strings) to generate labels for.
    Returns:
        list: A list of display labels corresponding to the input factors,
        in the same order.
        Each label is either a predefined label, a dynamically generated label
        based on plant attributes, or a formatted version of the factor name.
    Example:
        >>> factors = ["fixed_capital",
        "variable_opex_inputs.electricity",
        "plant_products.power"]
        >>> labels = _build_tornado_labels(plant, factors)
        >>> labels
        ["Fixed CAPEX", "Electricity price", "Power price"]
    """
    label_map = {
        "fixed_capital": "Fixed CAPEX",
        "fixed_opex": "Fixed OPEX",
        "project_lifetime": "Project lifetime",
        "interest_rate": "Interest rate",
        "operator_hourly_rate": "Operator hourly rate",
    }

    for var in plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}"] = f"{_make_label(var)} price"

    for prod in plant.plant_products:
        label_map[f"plant_products.{prod}"] = f"{_make_label(prod)} price"

    return [label_map.get(f, _make_label(f)) for f in factors]


# For analysis
def _get_original_value(plant, full_key):
    """
    Retrieve the original value from a nested structure using a dot-separated
    key path. This function navigates through a potentially nested combination
    of dictionaries and objects to extract a value at the location specified
    by the full_key parameter.
    When traversing dictionaries, it automatically extracts the "price" field
    from the accessed value. For objects, it retrieves attributes directly by
    name.

    Args:
        plant: The root object or dictionary to traverse.
            Can be either a dictionary with nested structure or
            an object with attributes.
        full_key (str): A dot-separated string representing
        the path to the value

    Returns:
        The value found at the specified key path. For dictionary entries,
        returns the "price" field of the value. For object attributes,
        returns the attribute value directly.

    Raises:
        KeyError: If a key is not found in a dictionary or if the "price" field
            does not exist in a dictionary value.
        TypeError: If attempting to access a key/attribute on an unsupported
        type.

    Examples:
        >>> plant = {"level1": {"level2": {"price": 250}}}
        >>> _get_original_value(plant, "level1.level2")
        250
    """
    keys = full_key.split(".")
    ref = plant
    for k in keys:
        if isinstance(ref, dict):
            ref = ref[k]["price"]
        else:
            ref = getattr(ref, k)
    return ref


def _update_and_evaluate(
        plant,
        factor,
        value,
        nested_price_keys,
        metric="LCOP",
        additional_capex: bool = False,
        ):
    """
    Update a plant configuration parameter and evaluate the resulting economic
    metric. This function creates a deep copy of the plant object, applies
    a specified parameter change, recalculates economics, and returns the
    requested metric value. It is typically used for sensitivity analysis,
    tornado diagrams, or scenario evaluation.
    Parameters
    ----------
    plant : Plant
        The plant object to be evaluated. The original object is not modified.
    factor : str
        The parameter to update. Can be one of:
        - "fixed_capital": Updates fixed capital cost
        - "fixed_opex": Updates fixed operating expenses
        - "variable_opex_inputs.<name>": Updates price of a variable input
        - "plant_products.<name>": Updates price of a plant product
        - "operator_hourly_rate": Updates operator hourly rate
        - Any other top-level plant attribute (e.g., "interest_rate",
        "project_lifetime")
    value : float
        The new value for the parameter being updated.
    nested_price_keys : list or set
        Collection of valid nested price keys
        (e.g., ["variable_opex_inputs.item1", "plant_products.product1"])
        used to identify which factors are nested.
    metric : str, optional
        The economic metric to calculate and return, by default "LCOP".
        Supported metrics:
        - "LCOP": Levelized cost of product
        - "ROI": Return on investment
        - "NPV": Net present value
        - "PBT", "PAYBACK", "PAYBACK_TIME": Payback time
        - "IRR": Internal rate of return
    additional_capex : bool, optional
        Whether to include additional capital expenditure in
        ROI and payback time calculations, by default False.
    Returns
    -------
    float or array-like
        The calculated metric value. For most metrics returns a scalar;
        NPV may return an array if Monte Carlo analysis is enabled.
    Raises
    ------
    ValueError
        If the specified factor contains an unsupported nested root, or if the
        requested metric is not supported.
    Notes
    -----
    - The original plant object is not modified; a deep copy is created
    internally.
    - All metric calculations trigger a recalculation of plant economics via
        calculate_levelized_cost().
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


def _ensure_list(plants):
    """
    Ensure that the input is converted to a list if it isn't already.

    Converts a single plant object or other iterable into a list format.
    If the input is already a list or tuple, it is returned as-is.

    Args:
        plants: A plant object, list of plants, or tuple of plants.

    Returns:
        list or tuple: The input wrapped in a list if it was not already a
        list or tuple, otherwise the input unchanged.

    Examples:
        >>> _ensure_list("plant1")
        ["plant1"]
        >>> _ensure_list(["plant1", "plant2"])
        ["plant1", "plant2"]
        >>> _ensure_list(("plant1", "plant2"))
        ("plant1", "plant2")
    """
    return plants if isinstance(plants, (list, tuple)) else [plants]


def _build_bar_data(components_list, xlabels, ylabel, currency, pct):
    """
    Build structured data for bar chart visualization from component
    dictionaries. This function processes a list of component dictionaries
    and formats them into a standardized structure suitable for bar chart
    rendering. It aligns all components to common labels and optionally
    normalizes values to percentages.
    Args:
        components_list (list): List of dictionaries where each dictionary maps
            component names (str) to numeric values (float).
        xlabels (list): Labels for the x-axis of the bar chart.
        ylabel (str): Label for the y-axis of the bar chart.
        currency (str): Currency symbol or code to be used in chart display.
        pct (bool): If True, normalize all values to percentages (0-100).
            If False, keep original values.
    Returns:
        dict: A dictionary containing the following keys:
            - "components" (list): Original components_list.
            - "labels" (list): List of label sets, one per component.
            - "values" (list): List of aligned value rows, one per component.
                If pct=True, values are normalized to percentages.
            - "xlabels" (list): The provided x-axis labels.
            - "ylabel" (str): The provided y-axis label.
            - "currency" (str): The provided currency.
            - "pct" (bool): The percentage flag.
    Example:
        >>> components = [{"A": 100, "B": 50}, {"A": 200, "C": 75}]
        >>> result = _build_bar_data(components, ["Q1", "Q2"],
                                                "Revenue", "USD", False)
    """
    # collect all unique component names
    all_labels = sorted(set().union(*(c.keys() for c in components_list)))

    # build aligned rows
    values = []
    for c in components_list:
        row = [c.get(label, 0.0) for label in all_labels]
        if pct:
            total = sum(row)
            row = [v / total * 100 if total != 0 else 0.0 for v in row]
        values.append(row)

    labels = [all_labels for _ in components_list]

    return {
        "components": components_list,
        "labels": labels,
        "values": values,
        "xlabels": xlabels,
        "ylabel": ylabel,
        "currency": currency,
        "pct": pct,
    }


def _evaluate_metric(plant, metric, additional_capex=False):
    """
    Evaluate a specified metric for a plant object.
    This function calculates and returns various financial and performance
    metrics for a plant by calling the appropriate calculation methods
    on the plant object.
    Args:
        plant: A plant object with methods to calculate financial metrics.
        metric (str): The metric to evaluate. Supported values are:
            - "LCOP": Levelized Cost of Power
            - "ROI": Return on Investment
            - "NPV": Net Present Value
            - "PBT", "PAYBACK", "PAYBACK_TIME": Payback Time
            - "IRR": Internal Rate of Return
        additional_capex (bool, optional): Whether to include additional
            capital expenditure in calculations for ROI and payback time
            metrics. Defaults to False.
    Returns:
        float: The calculated value of the requested metric.
    Raises:
        ValueError: If the specified metric is not supported.
    Raises:
        AttributeError: If the plant object lacks required attributes or
            methods to calculate the requested metric.
    """
    if metric == "LCOP":
        if not hasattr(plant, "levelized_cost"):
            plant.calculate_levelized_cost()
        return plant.levelized_cost

    elif metric == "ROI":
        plant.calculate_levelized_cost()
        plant.calculate_roi(additional_capex=additional_capex)
        return plant.roi

    elif metric == "NPV":
        plant.calculate_levelized_cost()
        return plant.calculate_npv()

    elif metric in ("PBT", "PAYBACK", "PAYBACK_TIME"):
        plant.calculate_levelized_cost()
        return plant.calculate_payback_time(
            additional_capex=additional_capex
        )

    elif metric == "IRR":
        plant.calculate_levelized_cost()
        plant.calculate_irr()
        return plant.irr

    else:
        raise ValueError(f"Unsupported metric '{metric}'")


def _collect_sensitivity_keys(plant, metric):
    """
    Collect sensitivity analysis keys for a given plant and metric.
    This function identifies which parameters should be included in sensitivity
    analysis based on the specified metric. It returns both all relevant keys
    and the nested keys separately.
    Args:
        plant: A plant object containing variable_opex_inputs and
                plant_products attributes with their respective keys.
        metric (str): The metric type for sensitivity analysis.
                    Either "LCOP" or another metric type.
    Returns:
        tuple: A tuple containing:
            - all_keys (list): Complete list of all sensitivity keys including
                              top-level keys and nested keys based on metric
                              type.
            - nested_keys (list): List of nested keys (variable_opex_inputs and
                                 optionally plant_products keys).
                                 For "LCOP" metric: only variable_opex_inputs
                                 keys. For other metrics: both
                                 variable_opex_inputs and plant_products keys.
    Notes:
        Top-level keys always included: fixed_capital, fixed_opex,
        project_lifetime, interest_rate, operator_hourly_rate.
    """
    top_level_keys = [
        "fixed_capital",
        "fixed_opex",
        "project_lifetime",
        "interest_rate",
        "operator_hourly_rate",
    ]

    var_keys = [f"variable_opex_inputs.{k}"
                for k in plant.variable_opex_inputs]
    prod_keys = [f"plant_products.{k}" for k in plant.plant_products]

    nested = var_keys if metric == "LCOP" else (var_keys + prod_keys)

    return top_level_keys + nested, nested


def _run_tornado_sensitivity(plant, keys, nested_keys,
                             pm, metric, additional_capex=False):
    """
    Perform tornado sensitivity analysis on plant parameters.
    This function evaluates how changes in specified plant parameters affect
    a given metric. For each parameter, it calculates the metric value at both
    low and high perturbation levels (typically ±pm from the original value).
    Args:
        plant: Plant object containing parameters to be analyzed.
        keys (list): List of parameter names to perform sensitivity analysis
            on. nested_keys (list or dict): Nested key structure for accessing
            parameters in hierarchical plant configurations.
        pm (float): Perturbation multiplier as a fraction (e.g., 0.1 for ±10%).
            Used to calculate low and high parameter values as (1 - pm) and
            (1 + pm) of the original value.
        metric (str or callable): The metric to evaluate. Used to assess the
            impact of parameter changes on plant performance.
        additional_capex (bool, optional): If True, includes additional capital
            expenditure in the evaluation. Defaults to False.
    Returns:
        dict: A dictionary where keys are parameter names from the input `keys`
            list, and values are lists containing [metric_low, metric_high],
            representing the metric values at low and high perturbation levels
            respectively.
    Notes:
        - Special handling is applied to "fixed_capital" and "fixed_opex"
        parameters, which use direct multiplication by (1 ± pm).
        - "operator_hourly_rate" is handled specially to extract rate from
        dict format
            or convert scalar values to float.
        - All other parameters use _get_original_value() to retrieve their
        current value.
    """
    results = {}

    for key in keys:
        if key in ["fixed_capital", "fixed_opex"]:
            low = 1 - pm
            high = 1 + pm

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
            low = original * (1 - pm)
            high = original * (1 + pm)

        else:
            original = _get_original_value(plant, key)
            low = original * (1 - pm)
            high = original * (1 + pm)

        metric_low = _update_and_evaluate(plant, key, low,
                                          nested_keys, metric,
                                          additional_capex=additional_capex)
        metric_high = _update_and_evaluate(plant, key, high,
                                           nested_keys, metric,
                                           additional_capex=additional_capex)

        results[key] = [metric_low, metric_high]

    return results


# For Monte Carlo analysis
def _truncated_normal_samples(mean, std, low, high, size):
    """
    Generate samples from a truncated normal distribution.
    This function generates random samples from a normal distribution that is
    truncated to fall within a specified range [low, high].
    Parameters
    ----------
    mean : float
        The mean of the normal distribution.
    std : float
        The standard deviation of the normal distribution.
    low : float
        The lower bound of the truncation range.
    high : float
        The upper bound of the truncation range.
    size : int or tuple of ints
        The shape of the output. If an integer, the output is 1-D.
        If a tuple of integers, the output is N-D with shape size.
    Returns
    -------
    ndarray
        Random samples from the truncated normal distribution with the
        specified parameters. If std is 0 or very close to 0,
        returns an array filled with the clipped mean value.
    Notes
    -----
    - If std is 0 or numerically close to 0, the function returns the mean
    value clipped to the [low, high] range, rather than sampling.
    - Uses scipy.stats.truncnorm for sampling when std > 0.
    Examples
    --------
    >>> samples = _truncated_normal_samples(mean=0, std=1,
    low=-2, high=2, size=100)
    >>> len(samples)
    100
    """
    if std == 0 or np.isclose(std, 0):
        return np.full(size, np.clip(mean, low, high))

    a, b = (low - mean) / std, (high - mean) / std

    return truncnorm.rvs(
        a, b, loc=mean, scale=std, size=size
    )


def _get_sampling_params(props, default_min=0, default_max=99999):
    """
    Extract sampling parameters from a properties dictionary.

    Parameters
    ----------
    props : dict
        Dictionary containing sampling parameter properties with optional keys:
        - "price": float, the mean value for sampling (default: 0)
        - "std": float, the standard deviation (default: 0)
        - "min": float, the minimum bound (default: default_min)
        - "max": float, the maximum bound (default: default_max)
    default_min : float, optional
        Default minimum value if not specified in props (default: 0)
    default_max : float, optional
        Default maximum value if not specified in props (default: 99999)

    Returns
    -------
    tuple of (float, float, float, float)
        A tuple containing (mean, std, min_, max_) extracted from props or
        defaults.
        - mean: the price/mean value
        - std: the standard deviation
        - min_: the minimum bound
        - max_: the maximum bound

    Examples
    --------
    >>> props = {"price": 50, "std": 10, "min": 10, "max": 100}
    >>> _get_sampling_params(props)
    (50, 10, 10, 100)

    >>> props = {"price": 25}
    >>> _get_sampling_params(props)
    (25, 0, 0, 99999)
    """
    mean = props.get("price", 0)
    std = props.get("std", 0)
    min_ = props.get("min", default_min)
    max_ = props.get("max", default_max)
    return mean, std, min_, max_


# For reading and writing JSON files
def _read_json(filepath):
    """
    Read and parse a JSON file.

    Parameters
    ----------
    filepath : str or Path
        The path to the JSON file to read.

    Returns
    -------
    dict or list
        The parsed JSON content from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    JSONDecodeError
        If the file content is not valid JSON.
    IOError
        If there is an error reading the file.

    Examples
    --------
    >>> data = _read_json('config.json')
    >>> print(data)
    {'key': 'value'}
    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(obj):
    """
    Convert a Python object to a JSON-serializable format.

    This function recursively traverses through nested data structures
    and converts non-JSON-serializable objects (such as NumPy arrays
    and scalar types) into their JSON-compatible equivalents.

    Args:
        obj: The object to convert. Can be a dict, list, tuple, NumPy array,
             NumPy scalar, or any JSON-serializable type.

    Returns:
        A JSON-serializable representation of the input object, where:
        - dicts are recursively processed with all values converted
        - lists and tuples are recursively processed (returned as lists)
        - NumPy arrays are converted to lists via tolist()
        - NumPy scalars are converted to native Python types via item()
        - other objects are returned unchanged

    Examples:
        >>> import numpy as np
        >>> _to_jsonable({'array': np.array([1, 2, 3])})
        {'array': [1, 2, 3]}

        >>> _to_jsonable([np.float64(1.5), np.int32(42)])
        [1.5, 42]

        >>> _to_jsonable((np.array([1, 2]), [3, 4]))
        [[1, 2], [3, 4]]
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
