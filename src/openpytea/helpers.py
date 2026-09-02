from copy import deepcopy
from pathlib import Path
import numpy as np
import json
import re


# HELPER FUNCTIONS
# For plottings
def _tex_escape(symbol: str) -> str:
    """
    Escape ``symbol`` (e.g. ``"%"`` or ``"$"``) for LaTeX only when
    matplotlib's ``text.usetex`` is active.

    Plot labels must emit ``\\%``/``\\$`` when a real LaTeX run consumes
    them, but the bare symbol otherwise -- a hardcoded escape renders as
    a literal backslash on machines without LaTeX.
    """
    import matplotlib.pyplot as plt

    if plt.rcParams.get("text.usetex", False):
        return "\\" + symbol
    return symbol


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
    >>> _default_metric_label('USD', 'payback_time')
    'Payback time / [years]'

    Notes
    -----
    The ``%`` in the ROI label is LaTeX-escaped only when matplotlib's
    ``text.usetex`` is active (see :func:`_tex_escape`).
    """
    metric = metric.upper()
    if metric == "LCOP" or metric == "levelized_cost":
        return rf"Levelized cost / [{currency}$\cdot$unit$^{-1}$]"
    elif metric == "ROI":
        return f"Return on investment / [{_tex_escape('%')}]"
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

    # "consumption"/"production" are abbreviated: these sit in the y-tick
    # labels of a tornado plot, where the full words crowd out the axes.
    for var in plant.variable_opex_inputs:
        label_map[f"variable_opex_inputs.{var}"] = f"{_make_label(var)} price"
        label_map[f"variable_opex_inputs.{var}.consumption"] = (
            f"{_make_label(var)} cons."
        )

    for prod in plant.plant_products:
        label_map[f"plant_products.{prod}"] = f"{_make_label(prod)} price"
        label_map[f"plant_products.{prod}.production"] = (
            f"{_make_label(prod)} prod."
        )

    return [label_map.get(f, _make_label(f)) for f in factors]


# ======================================================
# PARAMETER DEPENDENCY GRAPH
# ======================================================
# Shared by every analysis that re-evaluates the plant under changed
# inputs: Monte Carlo (stochastic, arrays + noise) and the deterministic
# sensitivity/tornado analyses (scalars, no noise). The graph itself --
# which parameters exist as nodes, how a "depends_on" spec is parsed, and
# the topological resolution order -- is defined once here so that a
# dependency configured on a plant means the same thing in all of them.

# The seven economic "scalar" parameters that can participate in the
# dependency DAG as ("project", <name>) nodes: the six
# plant.project_uncertainties entries plus operator_hourly_rate (which
# lives in its own config dict but is otherwise treated identically).
# Unlike process parameters, these don't come from a per-item collection,
# so there is exactly one node per name.
_PROJECT_SCALAR_PARAMS = (
    "fixed_capital_factor", "fixed_opex_factor", "project_lifetime",
    "interest_rate", "plant_utilization", "tax_rate", "operator_hourly_rate",
)

# Sensitivity/tornado factor key -> DAG node, for the factors that name an
# economic scalar. Prices ("variable_opex_inputs.<item>" /
# "plant_products.<product>") are deliberately absent: they are not DAG
# nodes, so they neither drive nor are driven by a dependency.
_TOP_LEVEL_DEPENDENCY_NODES = {
    "fixed_capital": ("project", "fixed_capital_factor"),
    "fixed_opex": ("project", "fixed_opex_factor"),
    "project_lifetime": ("project", "project_lifetime"),
    "interest_rate": ("project", "interest_rate"),
    "operator_hourly_rate": ("project", "operator_hourly_rate"),
    "plant_utilization": ("project", "plant_utilization"),
    "tax_rate": ("project", "tax_rate"),
}


def _parse_dependency_driver(spec, context):
    """
    Parse a ``"depends_on"`` string into a ``(kind, name)`` pair.

    Parameters
    ----------
    spec : str
        Expected form ``"production:<product>"``, ``"consumption:<item>"``,
        or ``"project:<param>"``, referencing a ``plant_products`` entry, a
        ``variable_opex_inputs`` entry, or one of the economic scalars in
        :data:`_PROJECT_SCALAR_PARAMS` respectively.
    context : str
        Human-readable description of where ``spec`` came from, used in the
        error message.

    Returns
    -------
    tuple
        ``(kind, name)`` where ``kind`` is ``"production"``,
        ``"consumption"``, or ``"project"``.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Invalid 'depends_on' value {spec!r} for {context}; expected "
            "'production:<product>', 'consumption:<item>', or "
            "'project:<param>'."
        )
    kind, name = spec.split(":", 1)
    if kind not in ("production", "consumption", "project"):
        raise ValueError(
            f"Invalid 'depends_on' kind {kind!r} for {context}; must be "
            "'production', 'consumption', or 'project'."
        )
    if kind == "project" and name not in _PROJECT_SCALAR_PARAMS:
        raise ValueError(
            f"Invalid 'depends_on' reference 'project:{name}' for "
            f"{context}; 'project:' must name one of "
            f"{sorted(_PROJECT_SCALAR_PARAMS)}."
        )
    return kind, name


def _describe_dependency_node(kind, name):
    """Human-readable description of a ``(kind, name)`` node, for errors."""
    if kind == "consumption":
        return f"variable_opex_inputs['{name}']'s consumption_uncertainty"
    if kind == "production":
        return f"plant_products['{name}']'s production_uncertainty"
    if name == "operator_hourly_rate":
        return "operator_hourly_rate's uncertainty fields"
    return f"project_uncertainties['{name}']'s uncertainty fields"


def _dependency_context(kind, name):
    """Description of the block a dependency was declared in, for errors."""
    if kind == "project":
        return f"the 'dependency' block for project parameter '{name}'"
    return f"{kind}_dependency on '{name}'"


def _collect_dependency_specs(plant):
    """
    Map every dependency-driven node to its dependency block.

    A node opts in as a dependent by setting ``"consumption_dependency"``/
    ``"production_dependency"`` (on a ``variable_opex_inputs``/
    ``plant_products`` item) or ``"dependency"`` (on a
    ``plant.project_uncertainties`` entry or ``plant.operator_hourly_rate``,
    for the seven economic scalars in :data:`_PROJECT_SCALAR_PARAMS`).

    Only the graph *structure* is read here -- the ``"depends_on"``/
    ``"offset"`` block itself -- not the node's uncertainty configuration,
    which is Monte-Carlo-specific (see
    :func:`~openpytea.analysis._collect_dependency_nodes`).

    Returns
    -------
    dict
        ``{(kind, name): dependency_block}``, in plant configuration order.
    """
    specs = {}

    for item, props in plant.variable_opex_inputs.items():
        dep = props.get("consumption_dependency")
        if dep is not None:
            specs[("consumption", item)] = dep

    for prod, props in plant.plant_products.items():
        dep = props.get("production_dependency")
        if dep is not None:
            specs[("production", prod)] = dep

    pu = getattr(plant, "project_uncertainties", None) or {}
    for name in _PROJECT_SCALAR_PARAMS:
        props = (plant.operator_hourly_rate if name == "operator_hourly_rate"
                 else pu.get(name, {}))
        if isinstance(props, dict) and props.get("dependency") is not None:
            specs[("project", name)] = props["dependency"]

    return specs


def _dependency_parents(dep, key):
    """
    Parse and validate one dependency block's ``"depends_on"`` mapping.

    Parameters
    ----------
    dep : dict
        The dependency block, with a non-empty ``"depends_on"`` dict mapping
        parent references to their linear weights.
    key : tuple
        The ``(kind, name)`` node this block belongs to, for error messages.

    Returns
    -------
    dict
        ``{(kind, name): weight}`` for every parent of this node.
    """
    context = _dependency_context(*key)
    weights = dep.get("depends_on") if isinstance(dep, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            f"'depends_on' for {context} must be a non-empty "
            "dict mapping driver references to factors, e.g. "
            "{'production:methanol': 9.3}."
        )

    parents = {}
    for spec, factor in weights.items():
        if not isinstance(factor, (int, float)):
            raise ValueError(
                f"'depends_on' factor for {spec!r} in "
                f"{context} must be a number, got {factor!r}."
            )
        parents[_parse_dependency_driver(spec, context)] = factor
    return parents


def _resolve_dependency_dag(dependents, driver_pool,
                            seed_missing=None, noise=None):
    """
    Resolve every dependent in topological order.

    Each dependent's value is ``sum(weight_i * parent_i) + offset``, where
    each ``parent_i`` is that parent's own *final* value -- so a parent's
    noise (Monte Carlo) or perturbation (sensitivity/tornado) propagates
    downstream through the graph. Parents are resolved before their
    children via repeated passes over the pending set: a node becomes
    resolvable once every one of its parents already has a final value, and
    a pass that makes no progress means the remainder is unresolvable, due
    to an unknown reference or a cycle.

    The arithmetic is plain linear algebra over whatever ``driver_pool``
    holds, so the same walk serves scalar (deterministic) and array
    (Monte Carlo) values alike.

    Parameters
    ----------
    dependents : dict
        ``{(kind, name): dependency_block}``, from
        :func:`_collect_dependency_specs`.
    driver_pool : dict
        ``{(kind, name): value}`` for the nodes that already have a final
        value; mutated in place as dependents are resolved.
    seed_missing : callable, optional
        ``seed_missing(key) -> bool``, called for a parent absent from
        ``driver_pool`` to lazily add it; returning False marks it
        unresolvable.
    noise : callable, optional
        ``noise(key, value) -> value``, applied to each dependent's
        resolved value (Monte Carlo adds the node's own noise here).

    Returns
    -------
    dict
        ``{(kind, name): value}`` for the resolved dependents only.

    Raises
    ------
    ValueError
        If a ``"depends_on"`` entry is malformed or points at an unknown
        item, or the dependency graph has a cycle.
    """
    resolved = {}
    pending_keys = list(dependents)

    while pending_keys:
        progressed = False
        still_pending = []
        for key in pending_keys:
            dep = dependents[key]
            parents = _dependency_parents(dep, key)

            # A parent that is itself a pending dependent must never be
            # lazily seeded (that would hand its children a baseline
            # constant instead of its resolved value) -- wait for a
            # later pass to resolve it instead
            if not all(
                pk in driver_pool
                or (pk not in dependents
                    and seed_missing is not None
                    and seed_missing(pk))
                for pk in parents
            ):
                still_pending.append(key)
                continue

            value = None
            for parent_key, factor in parents.items():
                term = factor * driver_pool[parent_key]
                value = term if value is None else value + term
            value = value + dep.get("offset", 0.0)

            if noise is not None:
                value = noise(key, value)

            driver_pool[key] = value
            resolved[key] = value
            progressed = True

        pending_keys = still_pending
        if not progressed and pending_keys:
            unresolved = ", ".join(f"{k}:{n}" for k, n in pending_keys)
            raise ValueError(
                "Could not resolve consumption/production/project "
                f"dependency(ies) for: {unresolved}. Each 'depends_on' "
                "entry must reference a known variable_opex_inputs/"
                "plant_products item or one of "
                f"{sorted(_PROJECT_SCALAR_PARAMS)} (directly or "
                "transitively), and the dependency DAG must not contain a "
                "cycle."
            )

    return resolved


def _dependency_node_value(plant, key):
    """Current (baseline) value of a ``(kind, name)`` DAG node on ``plant``."""
    kind, name = key
    if kind == "consumption":
        return plant.variable_opex_inputs[name].get("consumption", 0.0)
    if kind == "production":
        return plant.plant_products[name].get("production", 0.0)

    # Economic scalars. fixed_capital_factor/fixed_opex_factor are the
    # multipliers `plant.fc`/`plant.fp`, which stand for 1.0 when unset.
    if name == "fixed_capital_factor":
        return 1.0 if plant.fc is None else plant.fc
    if name == "fixed_opex_factor":
        return 1.0 if plant.fp is None else plant.fp
    if name == "operator_hourly_rate":
        current = getattr(plant, "operator_hourly_rate", None)
        if isinstance(current, dict):
            return current.get("rate", 0.0)
        return 0.0 if current is None else float(current)
    return getattr(plant, name)


def _set_dependency_node_value(plant, key, value):
    """
    Write a resolved value back onto ``plant`` -- the inverse of
    :func:`_dependency_node_value`.
    """
    kind, name = key
    if kind == "consumption":
        plant.variable_opex_inputs[name]["consumption"] = value
    elif kind == "production":
        plant.plant_products[name]["production"] = value
    elif name == "fixed_capital_factor":
        plant.fc = value
    elif name == "fixed_opex_factor":
        plant.fp = value
    elif name == "operator_hourly_rate":
        if isinstance(getattr(plant, "operator_hourly_rate", None), dict):
            plant.operator_hourly_rate["rate"] = value
        else:
            plant.operator_hourly_rate = value
    else:
        setattr(plant, name, value)


def _apply_dependencies(plant):
    """
    Recompute every dependency-driven parameter on ``plant``, in place.

    This is the deterministic counterpart of
    :func:`~openpytea.analysis._resolve_quantity_dependencies`: the same
    DAG, resolved from the plant's *current* parameter values with no noise
    added. Calling it after a parameter has been perturbed makes that
    perturbation cascade to everything downstream of it, which is what lets
    sensitivity and tornado analyses honour dependencies the way Monte
    Carlo does.

    A no-op (and free) for a plant with no dependencies configured.

    Parameters
    ----------
    plant : Plant
        Mutated in place. Pass a copy unless you mean to change the plant.
    """
    dependents = _collect_dependency_specs(plant)
    if not dependents:
        return

    driver_pool = {}
    for item in plant.variable_opex_inputs:
        key = ("consumption", item)
        if key not in dependents:
            driver_pool[key] = _dependency_node_value(plant, key)
    for prod in plant.plant_products:
        key = ("production", prod)
        if key not in dependents:
            driver_pool[key] = _dependency_node_value(plant, key)
    for name in _PROJECT_SCALAR_PARAMS:
        key = ("project", name)
        if key not in dependents:
            driver_pool[key] = _dependency_node_value(plant, key)

    for key, value in _resolve_dependency_dag(dependents, driver_pool).items():
        _set_dependency_node_value(plant, key, value)


def _sensitivity_key_node(key):
    """
    DAG node named by a sensitivity/tornado factor key, or None.

    Maps ``"fixed_capital"`` -> ``("project", "fixed_capital_factor")``,
    ``"variable_opex_inputs.<item>.consumption"`` ->
    ``("consumption", "<item>")``, ``"plant_products.<product>.production"``
    -> ``("production", "<product>")``, and so on. Returns None for a key
    that isn't a DAG node at all -- notably the price keys
    ``"variable_opex_inputs.<item>"`` / ``"plant_products.<product>"``.
    """
    if key in _TOP_LEVEL_DEPENDENCY_NODES:
        return _TOP_LEVEL_DEPENDENCY_NODES[key]

    parts = key.split(".")
    if len(parts) == 3:
        root, name, field = parts
        if root == "variable_opex_inputs" and field == "consumption":
            return ("consumption", name)
        if root == "plant_products" and field == "production":
            return ("production", name)
    return None


def _node_sensitivity_key(node):
    """
    Sensitivity/tornado factor key for a DAG node -- the inverse of
    :func:`_sensitivity_key_node`.
    """
    kind, name = node
    if kind == "consumption":
        return f"variable_opex_inputs.{name}.consumption"
    if kind == "production":
        return f"plant_products.{name}.production"
    for factor_key, factor_node in _TOP_LEVEL_DEPENDENCY_NODES.items():
        if factor_node == node:
            return factor_key
    return name


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

    The two three-part quantity keys --
    "variable_opex_inputs.<item>.consumption" and
    "plant_products.<product>.production" -- name a dependency-graph node
    rather than a price, and are read through
    :func:`_dependency_node_value` instead.

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
    node = _sensitivity_key_node(full_key)
    if node is not None and node[0] in ("consumption", "production"):
        return _dependency_node_value(plant, node)

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
        - "variable_opex_inputs.<name>.consumption": Updates the consumption
        rate of a variable input
        - "plant_products.<name>.production": Updates the production rate of
        a plant product
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
    - Any parameter dependencies configured on the plant are re-resolved
    after the change and before the economics are recomputed, so a
    perturbation cascades to everything downstream of it in the dependency
    graph (see :func:`_apply_dependencies`).
    - All metric calculations trigger a recalculation of plant economics via
        calculate_levelized_cost().
    """
    plant_copy = deepcopy(plant)
    metric = metric.upper()

    # --- 1. Apply parameter change ---

    quantity_node = _sensitivity_key_node(factor)
    if quantity_node is not None and quantity_node[0] != "project":
        # "variable_opex_inputs.<name>.consumption" or
        # "plant_products.<name>.production" -- a process quantity rather
        # than a price, and a node of the dependency graph.
        kind, name = quantity_node
        root, field = (
            ("variable_opex_inputs", "consumption") if kind == "consumption"
            else ("plant_products", "production")
        )
        plant_copy.update_configuration({root: {name: {field: value}}})

    elif factor == "fixed_capital":
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

    # --- 2. Propagate the change through the dependency graph ---

    # Anything tied to the changed parameter (directly or transitively)
    # moves with it, exactly as it would in a Monte Carlo run. No-op when
    # the plant has no dependencies configured.
    _apply_dependencies(plant_copy)

    # --- 3. Recompute economics ---

    # This builds fixed_capital, opex, revenue, cash_flow, etc.
    plant_copy.calculate_levelized_cost()

    # --- 4. Return requested metric ---

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
        # Always recompute, like every other metric branch: a cached
        # levelized_cost is never invalidated, so trusting it here
        # de-centers sensitivity/tornado baselines after any direct
        # plant edit
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


def _evaluate_baseline_metric(plant, metric, additional_capex=False):
    """
    Evaluate a metric at the plant's unperturbed baseline, with any
    configured parameter dependencies resolved first.

    Sensitivity and tornado analyses compare perturbed cases -- which run
    through :func:`_update_and_evaluate` and therefore resolve the
    dependency graph -- against this baseline, so the baseline has to be
    resolved the same way or a plant whose dependencies don't exactly
    reproduce its configured values would show a spurious offset at 0 %.

    Unlike :func:`_evaluate_metric`, which recomputes on the plant it is
    given, this never mutates ``plant``: it works on a copy whenever there
    is a dependency to resolve, and otherwise falls straight through.
    """
    if not _collect_dependency_specs(plant):
        return _evaluate_metric(plant, metric, additional_capex)

    plant_copy = deepcopy(plant)
    _apply_dependencies(plant_copy)
    # Resolving the graph can move the plant's inputs off the values its
    # cached levelized_cost was computed from, and _evaluate_metric reuses
    # that cache for "LCOP" rather than recomputing.
    plant_copy.calculate_levelized_cost()
    return _evaluate_metric(plant_copy, metric, additional_capex)


def _collect_sensitivity_keys(plant, metric, include_process_params=False):
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
        include_process_params (bool): Whether to also return a factor for
                    each process quantity -- every item's consumption and
                    every product's production. Default False, which keeps
                    the factor set to prices and economic scalars.
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

        ``include_process_params=True`` additionally returns
        "variable_opex_inputs.<item>.consumption" and
        "plant_products.<product>.production" for every item and product.
        These are ordinary economic drivers -- a plant's production rate
        moves LCOP whether or not anything is tied to it -- so the flag is
        independent of the dependency graph: it neither requires
        dependencies nor is implied by them.

        Only the *dependency* structure narrows the result, and only for
        correctness: a parameter set by a dependency has no value of its
        own to vary, so it is never a factor. That drops dependent process
        quantities from what ``include_process_params`` would otherwise
        add, and drops an economic scalar (e.g. "fixed_capital" when
        ``fixed_capital_factor`` is a dependent) from the top-level keys.
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

    dependents = _collect_dependency_specs(plant)
    if not dependents and not include_process_params:
        return top_level_keys + nested, nested

    top_level_keys = [
        k for k in top_level_keys
        if _sensitivity_key_node(k) not in dependents
    ]

    quantity_keys = []
    if include_process_params:
        quantity_keys = [
            _node_sensitivity_key(("consumption", item))
            for item in plant.variable_opex_inputs
            if ("consumption", item) not in dependents
        ] + [
            _node_sensitivity_key(("production", prod))
            for prod in plant.plant_products
            if ("production", prod) not in dependents
        ]

    return top_level_keys + nested + quantity_keys, nested


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
