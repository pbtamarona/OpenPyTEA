"""MCP server for OpenPyTEA.

Exposes the library to AI assistants (Claude Desktop, Claude Code, or any
other Model Context Protocol client) as a set of tools: cost-correlation
lookup, equipment costing, plant evaluation, and sensitivity / tornado /
Monte Carlo analyses. Installed as the ``openpytea-mcp`` console command
(see ``[project.scripts]`` in ``pyproject.toml``); needs the optional
``mcp`` extra::

    pip install "openpytea[mcp]"
    claude mcp add openpytea -- openpytea-mcp

Inputs use the same JSON layout as the library's equipment and plant
files (see :func:`openpytea.io.load_equipment_config` and
:func:`openpytea.io.load_plant_config`), so anything built here can be
saved and run with ``run_tea`` or the ``openpytea`` CLI, and the other way
round.

The tool logic lives in plain functions (``cost_equipment``,
``evaluate_plant``, ...) that raise ``ValueError`` on bad input;
:func:`build_server` wraps them as MCP tools. Evaluated plants are kept in
memory for the life of the server process, keyed by plant name, so the
analysis tools can refer to a plant by name instead of resending its
configuration.
"""

import json
import math
import os
import re
from copy import deepcopy
from importlib import resources
from importlib.metadata import version

import numpy as np

from openpytea.analysis import monte_carlo, sensitivity_data, tornado_data
from openpytea.equipment import COST_DB_DF, Equipment
from openpytea.helpers import _collect_sensitivity_keys
from openpytea.io import _build_equipment_list
from openpytea.plant import Plant


__version__ = version("openpytea")

METRICS = ("LCOP", "NPV", "ROI", "IRR", "PBT")
MAX_MC_SAMPLES = 500_000

# Cost database columns worth showing for a correlation lookup
_CORRELATION_COLUMNS = {
    "key": "key",
    "category": "category",
    "type": "type",
    "units": "units",
    "s_lower": "size_lower",
    "s_upper": "size_upper",
    "s2_lower": "size2_lower",
    "s2_upper": "size2_upper",
    "form": "form",
    "cost_year": "cost_year",
    "default material": "default_material",
    "source": "source",
}

# Plants evaluated in this session: name -> {"equipment", "plant", "obj"}
_SESSION = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """numpy / NaN / inf -> plain JSON values (NaN and inf become None)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _plain(label):
    """
    A library display label (matplotlib mathtext, e.g.
    ``r"$\\mathrm{CO_{2}}$ price"``) as plain text (``"CO2 price"``).
    """
    if not isinstance(label, str):
        return label
    s = label.replace(r"\$", "\0")  # a literal dollar, not math mode
    s = s.replace("$", "")
    s = re.sub(r"\^\{?(-?\d+)\}?",
               lambda m: m.group(1).translate(_SUPERSCRIPTS), s)
    s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[_^](\w)", r"\1", s)
    s = re.sub(r"\\(?:mathrm|text|mathit|rm)\b", "", s)
    s = s.replace(r"\cdot", "·").replace(r"\,", " ")
    s = s.replace(r"\%", "%").replace(r"\&", "&")
    return s.replace("{", "").replace("}", "").replace("\0", "$")


def _unwrap(data, key):
    """Accept either ``{"<key>": {...}}`` (the file layout) or the bare block."""
    if isinstance(data, dict) and key in data and len(data) == 1:
        return data[key]
    return data


def _build_equipment(equipment):
    """Library equipment objects from a list of equipment entries."""
    equipment = _unwrap(equipment, "equipment")
    if not isinstance(equipment, list) or not equipment:
        raise ValueError(
            "equipment must be a non-empty list of equipment entries."
        )
    try:
        return _build_equipment_list({"equipment": deepcopy(equipment)})
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid equipment entry: {exc}") from exc


def _get_plant(name):
    if name not in _SESSION:
        known = ", ".join(sorted(_SESSION)) or "none yet"
        raise ValueError(
            f"No evaluated plant named {name!r} (evaluated plants: "
            f"{known}). Run evaluate_plant or load_example first."
        )
    return _SESSION[name]["obj"]


def _metric(metric):
    metric = str(metric).upper()
    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; use one of {', '.join(METRICS)}."
        )
    return metric


def _plant_summary(plant):
    """The headline results of a calculated plant."""
    out = plant.to_dict()
    cash_flow = out["cash_flow"]["cash_flow"]
    if cash_flow and isinstance(cash_flow[0], list):
        cash_flow = cash_flow[0]  # (1, N) -> N
    summary = {
        "plant_name": plant.name,
        "currency": plant.currency,
        "main_product": out["revenue"]["main_product"],
        "metrics": out["metrics"],
        "capital_costs": out["capital_costs"],
        "variable_opex": out["variable_opex"],
        "fixed_opex": out["fixed_opex"],
        "revenue": out["revenue"],
        "annual_cash_flow": cash_flow,
        "equipment": [eq.to_dict() for eq in plant.equipment_list],
        "notes": [
            "levelized_cost (LCOP) is the pre-tax cost per unit of the "
            "main product, in currency per the product's production unit.",
            "Costs and cash flows are annual values in the plant currency.",
        ],
    }
    return _jsonable(summary)


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p5": float(p5),
        "median": float(p50),
        "p95": float(p95),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Tool logic
# ---------------------------------------------------------------------------

def _example_files():
    folder = resources.files("openpytea") / "data" / "examples"
    return sorted(
        (f for f in folder.iterdir() if f.name.endswith(".json")),
        key=lambda f: f.name,
    )


def _read_example(example_id):
    for f in _example_files():
        if f.name[:-5] == example_id:
            return json.loads(f.read_text(encoding="utf-8"))
    ids = ", ".join(f.name[:-5] for f in _example_files())
    raise ValueError(f"Unknown example {example_id!r}; available: {ids}.")


def list_examples():
    """The bundled example plants (the GUI's examples)."""
    out = []
    for f in _example_files():
        data = json.loads(f.read_text(encoding="utf-8"))
        out.append({
            "id": f.name[:-5],
            "title": data.get("title", f.name[:-5]),
            "description": data.get("description", ""),
            "plant_name": data["plant"].get("plant_name"),
            "n_equipment": len(data["equipment"]),
        })
    return out


def get_example(example_id):
    """An example plant's equipment and plant configuration."""
    data = _read_example(example_id)
    return {
        "id": example_id,
        "title": data.get("title", example_id),
        "description": data.get("description", ""),
        "equipment": data["equipment"],
        "plant": data["plant"],
    }


def search_cost_correlations(query="", category=None, limit=25):
    """
    Search the bundled equipment cost correlations by key, category, type,
    units or source (case-insensitive; every word of ``query`` must match).
    """
    df = COST_DB_DF
    if category:
        df = df[df["category"].str.lower() == category.lower()]
        if df.empty:
            cats = sorted(COST_DB_DF["category"].dropna().unique())
            raise ValueError(
                f"Unknown category {category!r}; available: {', '.join(cats)}."
            )
    if query:
        text = (
            df[["key", "category", "type", "units", "source"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        mask = np.ones(len(df), dtype=bool)
        for word in query.lower().split():
            mask &= text.str.contains(word, regex=False).to_numpy()
        df = df[mask]

    limit = max(1, min(int(limit), 200))
    rows = df[list(_CORRELATION_COLUMNS)].head(limit)
    rows = rows.rename(columns=_CORRELATION_COLUMNS)
    rows = rows.astype(object).where(rows.notna(), None)
    return _jsonable({
        "total_matches": int(len(df)),
        "returned": int(len(rows)),
        "correlations": rows.to_dict(orient="records"),
        "materials": list(Equipment.material_factors),
        "usage": (
            "Use category and type (not key) in an equipment entry, with "
            "param = the size in the correlation's units (a [s1, s2] pair "
            "for 2-variable correlations). Sizes above size_upper are split "
            "into parallel units automatically."
        ),
    })


def cost_equipment(equipment):
    """Purchased and direct (installed) costs of a list of equipment."""
    items = [eq.to_dict() for eq in _build_equipment(equipment)]
    return _jsonable({
        "equipment": items,
        "totals": {
            "total_purchased_cost": sum(
                i.get("purchased_cost") or 0.0 for i in items
            ),
            "total_direct_cost": sum(
                i.get("direct_cost") or 0.0 for i in items
            ),
        },
        "notes": [
            "Costs are in USD of each item's target_year (default 2024); "
            "direct_cost includes the installation factors and material "
            "factor.",
        ],
    })


def evaluate_plant(equipment, plant, additional_capex=False):
    """
    Build and calculate a plant; keep it in the session under its name.
    """
    equipment_list = _build_equipment(equipment)
    config = deepcopy(_unwrap(plant, "plant"))
    if not isinstance(config, dict):
        raise ValueError("plant must be a plant configuration object.")
    config.pop("equipment", None)
    if not config.get("plant_name"):
        raise ValueError("plant needs a 'plant_name'.")
    if not config.get("plant_products"):
        raise ValueError(
            "plant needs 'plant_products' (the first one is the main "
            "product), e.g. {\"hydrogen\": {\"production\": 50000, "
            "\"price\": 5}}; production is per day."
        )

    built = Plant({**deepcopy(config), "equipment": equipment_list})
    built.calculate_all(additional_capex=additional_capex)

    replaced = built.name in _SESSION
    _SESSION[built.name] = {
        "equipment": deepcopy(_unwrap(equipment, "equipment")),
        "plant": config,
        "obj": built,
    }
    summary = _plant_summary(built)
    if replaced:
        summary["notes"].append(
            f"Replaced the earlier plant named {built.name!r} in this session."
        )
    return summary


def load_example(example_id):
    """Evaluate a bundled example plant and keep it in the session."""
    data = _read_example(example_id)
    return evaluate_plant(data["equipment"], data["plant"])


def list_plants():
    """Plants evaluated in this session, with their headline metrics."""
    out = []
    for name, entry in _SESSION.items():
        metrics = entry["obj"].to_dict()["metrics"]
        out.append({
            "plant_name": name,
            "currency": entry["obj"].currency,
            "levelized_cost": metrics.get("levelized_cost"),
            "npv": metrics.get("npv"),
            "irr": metrics.get("irr"),
        })
    return _jsonable(out)


def get_plant_config(plant_name):
    """
    A session plant's inputs, in the library's equipment / plant JSON file
    layout (ready to edit and pass back to evaluate_plant, or to save as
    equipment.json / plant.json for run_tea).
    """
    _get_plant(plant_name)
    entry = _SESSION[plant_name]
    return {
        "equipment": deepcopy(entry["equipment"]),
        "plant": deepcopy(entry["plant"]),
    }


def list_sensitivity_parameters(plant_name, metric="LCOP"):
    """The parameters sensitivity / tornado analyses can vary on a plant."""
    plant = _get_plant(plant_name)
    metric = _metric(metric)
    return {
        "economic": _collect_sensitivity_keys(plant, metric, "economic"),
        "process": _collect_sensitivity_keys(plant, metric, "process"),
        "notes": [
            "Prices can also be given by item name alone (e.g. "
            "'electricity'), quantities as '<item>.consumption' or "
            "'<product>.production'.",
            "For LCOP the main product's price is not a parameter: it is "
            "what LCOP solves for.",
        ],
    }


def run_sensitivity(plant_name, parameter, plus_minus_value=0.2,
                    metric="LCOP", n_points=11, additional_capex=False):
    """One-parameter sensitivity of a metric for a session plant."""
    plant = _get_plant(plant_name)
    if not 0 < plus_minus_value <= 10:
        raise ValueError("plus_minus_value must be a fraction in (0, 10].")
    n_points = max(3, min(int(n_points), 101))
    data = sensitivity_data(
        plant,
        parameter=parameter,
        plus_minus_value=plus_minus_value,
        n_points=n_points,
        metric=_metric(metric),
        additional_capex=additional_capex,
    )
    curve = data["curves"][0]
    return _jsonable({
        "plant_name": plant_name,
        "parameter": data["parameter"],
        "metric": data["metric"],
        "metric_label": _plain(data["ylabel"]),
        "parameter_label": _plain(data["xlabel_value"]),
        "baseline": curve["baseline"],
        "points": [
            {"change_pct": x, "parameter_value": xv, "metric_value": y}
            for x, xv, y in zip(
                curve["x"],
                curve["x_value"]
                if curve["x_value"] is not None
                else [None] * len(curve["x"]),
                curve["y"],
            )
        ],
    })


def run_tornado(plant_name, plus_minus_value=0.2, metric="LCOP",
                category="economic", additional_capex=False):
    """Rank the parameters by their effect on a metric (tornado data)."""
    plant = _get_plant(plant_name)
    if category not in ("economic", "process", "both"):
        raise ValueError("category must be 'economic', 'process' or 'both'.")
    if not 0 < plus_minus_value <= 10:
        raise ValueError("plus_minus_value must be a fraction in (0, 10].")
    data = tornado_data(
        plant,
        plus_minus_value=plus_minus_value,
        metric=_metric(metric),
        category=category,
        additional_capex=additional_capex,
    )
    base = float(data["base_value"])
    bars = [
        {
            "parameter": factor,
            "label": _plain(label),
            "metric_at_low": low,
            "metric_at_high": high,
            "swing": abs(high - low),
        }
        for factor, label, low, high in zip(
            data["factors"], data["labels"], data["lows"], data["highs"]
        )
    ]
    bars.sort(key=lambda b: b["swing"], reverse=True)  # largest first
    return _jsonable({
        "plant_name": plant_name,
        "metric": data["metric"],
        "metric_label": _plain(data["xlabel"]),
        "base_value": base,
        "plus_minus_value": plus_minus_value,
        "category": category,
        "bars": bars,
    })


def run_monte_carlo(plant_name, num_samples=20_000, random_seed=None,
                    additional_capex=False):
    """
    Monte Carlo uncertainty analysis for a session plant, using the
    uncertainty ranges in its configuration; returns summary statistics.
    """
    plant = _get_plant(plant_name)
    num_samples = int(num_samples)
    if not 100 <= num_samples <= MAX_MC_SAMPLES:
        raise ValueError(
            f"num_samples must be between 100 and {MAX_MC_SAMPLES:,}."
        )
    result = monte_carlo(
        plant,
        num_samples=num_samples,
        batch_size=min(10_000, num_samples),
        additional_capex=additional_capex,
        random_seed=random_seed,
    )
    prices_set = all(
        isinstance(p, dict) and p.get("price") is not None
        for p in plant.plant_products.values()
    )
    metrics = {}
    for name, values in result["metrics"].items():
        if name != "LCOP" and not prices_set:
            continue  # zero-filled without product prices
        metrics[name] = _stats(values)
    inputs = {}
    for name, values in result["inputs"].items():
        stats = _stats(values)
        if stats and stats["std"] > 0:  # skip the constants
            inputs[_plain(name)] = stats
    notes = [
        "Statistics over all samples; p5/p95 are the 5th/95th percentiles.",
        "Only inputs that were actually sampled are listed under inputs.",
    ]
    if not prices_set:
        notes.append(
            "NPV, ROI and PBT are left out: not every product has a price."
        )
    return _jsonable({
        "plant_name": plant_name,
        "currency": result["currency"],
        "num_samples": num_samples,
        "random_seed": random_seed,
        "metrics": metrics,
        "inputs": inputs,
        "notes": notes,
    })


INPUT_FORMAT_GUIDE = """\
OpenPyTEA input format (same as the library's JSON files)

EQUIPMENT: a list of entries.
  Costed from a correlation:
    {"name": "Feed pump", "process_type": "Fluids", "category": "Pumps",
     "type": "Single-stage centrifugal", "param": 15, "material": "Carbon steel"}
    (param 15 = flow in L/s, the units of that correlation)
    - category/type pick the cost correlation (search_cost_correlations);
      param is the size in that correlation's units ([s1, s2] for 2-variable
      correlations).
  Known purchased cost:
    {"name": "Stack", "process_type": "Fluids", "category": "Electrolysis",
     "type": "Stack", "purchased_cost": 17194000, "cost_year": 2020}
  Optional: num_units, target_year (default 2024), material, and
  installation-factor overrides erection_factor, piping_factor,
  instrumentation_factor, electrical_factor, civil_factor,
  structural_factor, lagging_factor, material_factor.
  A composite: {"name", "process_type", "components": [entries...]}.
  process_type: "Fluids", "Solids", "Mixed" or "Electrical".

PLANT: one object.
  Required: plant_name, process_type ("Fluids"/"Solids"/"Mixed"),
  plant_products (the first is the main product).
  Common: country, region, currency, exchange_rate, interest_rate (0.09),
  project_lifetime (20), plant_utilization (0-1), tax_rate,
  operator_hourly_rate ({"rate": 38.11}), working_capital (null = auto),
  capex_ramp ([0.3, 0.6, 0.1]), production_ramp ([0, 0, 0.4, 0.8]),
  fixed_capital_factors, fixed_opex_factors, fc, fp.
  variable_opex_inputs: {"electricity": {"consumption": 2352000,
    "price": 0.05}} -- consumption per day, price per unit.
  plant_products: {"hydrogen": {"production": 50000, "price": 5}}
    -- production per day; price optional for the main product (needed
    for NPV/IRR/ROI/payback).

UNCERTAINTY (Monte Carlo): on a price add "std", "min", "max" (and
  optionally "dist_id"); for quantities a "consumption_uncertainty" /
  "production_uncertainty" dict with the same fields; project-level ones
  under project_uncertainties (fixed_capital_factor, fixed_opex_factor,
  project_lifetime, interest_rate, plant_utilization, tax_rate).

The examples (list_examples / get_example) show complete, working inputs.
"""


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

INSTRUCTIONS = """\
OpenPyTEA: techno-economic assessment (TEA) of process plants and energy
systems. Typical flow: look at an example (list_examples / get_example) or
read the input format (get_input_format); cost equipment
(search_cost_correlations, cost_equipment); evaluate a plant
(evaluate_plant or load_example) to get CAPEX, OPEX, the levelized cost of
the main product (LCOP), NPV, IRR and payback; then run_sensitivity,
run_tornado or run_monte_carlo on it by plant name. To change a plant,
fetch its inputs with get_plant_config, edit them and call evaluate_plant
again. Fractions (rates, utilization, plus_minus_value) are 0-1, e.g. 0.2
for +/-20%. Report the currency and the unit basis with any number.
"""


def build_server():
    """The OpenPyTEA MCP server, with every tool registered."""
    import functools
    from typing import Annotated, Any, Literal

    from pydantic import Field

    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The OpenPyTEA MCP server needs the 'mcp' package (version 2 or "
            'later): pip install "openpytea[mcp]"'
        ) from exc

    server = MCPServer(
        name="openpytea",
        title="OpenPyTEA",
        instructions=INSTRUCTIONS,
        version=__version__,
        website_url="https://github.com/pbtamarona/OpenPyTEA",
    )

    read_only = ToolAnnotations(
        read_only_hint=True, idempotent_hint=True, open_world_hint=False
    )
    computes = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, idempotent_hint=True,
        open_world_hint=False,
    )

    def tool(annotations):
        """Register ``fn`` as a tool; library errors reach the model."""
        def register(fn):
            @functools.wraps(fn)
            def call(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except ToolError:
                    raise
                except (ValueError, KeyError, TypeError) as exc:
                    raise ToolError(str(exc)) from exc
            # the wrappers end in "_" so as not to shadow the module-level
            # functions they call
            server.tool(name=fn.__name__.rstrip("_"),
                        annotations=annotations)(call)
            return fn
        return register

    Metric = Literal["LCOP", "NPV", "ROI", "IRR", "PBT"]
    Equipments = Annotated[
        list[dict[str, Any]],
        Field(description="Equipment entries (see get_input_format)."),
    ]
    PlantName = Annotated[
        str, Field(description="Name of a plant evaluated in this session.")
    ]
    Fraction = Annotated[
        float,
        Field(gt=0, le=10, description="Variation as a fraction, 0.2 = ±20%."),
    ]
    AddCapex = Annotated[
        bool,
        Field(description="Include the plant's additional (later) CAPEX."),
    ]

    @tool(read_only)
    def get_input_format() -> str:
        """The equipment and plant input format, with examples."""
        return INPUT_FORMAT_GUIDE

    @tool(read_only)
    def list_examples_() -> list[dict[str, Any]]:
        """List the bundled example plants (hydrogen, geothermal, ...)."""
        return list_examples()

    @tool(read_only)
    def get_example_(
        example_id: Annotated[str, Field(description="Id from list_examples.")],
    ) -> dict[str, Any]:
        """An example's complete equipment list and plant configuration, a
        good starting point to adapt for a new plant."""
        return get_example(example_id)

    @tool(read_only)
    def search_cost_correlations_(
        query: Annotated[str, Field(
            description="Words to match, e.g. 'centrifugal pump' or "
                        "'shell tube'. Empty lists everything.")] = "",
        category: Annotated[str | None, Field(
            description="Exact category, e.g. 'Pumps', 'Heat exchangers'."
        )] = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 25,
    ) -> dict[str, Any]:
        """Search the equipment cost correlations (category, type, size
        units and valid size range, cost year, source) and list the
        construction materials."""
        return search_cost_correlations(query, category, limit)

    @tool(read_only)
    def cost_equipment_(equipment: Equipments) -> dict[str, Any]:
        """Purchased and installed (direct) cost of each equipment item,
        and the totals. Does not need a plant."""
        return cost_equipment(equipment)

    @tool(computes)
    def evaluate_plant_(
        equipment: Equipments,
        plant: Annotated[dict[str, Any], Field(
            description="Plant configuration (see get_input_format).")],
        additional_capex: AddCapex = False,
    ) -> dict[str, Any]:
        """Run the full TEA of a plant: capital costs, variable and fixed
        OPEX, revenue, cash flow, LCOP, NPV, IRR, ROI, payback. The plant is
        kept in the session under its plant_name (replacing one with the
        same name) for the analysis tools."""
        return evaluate_plant(equipment, plant, additional_capex)

    @tool(computes)
    def load_example_(
        example_id: Annotated[str, Field(description="Id from list_examples.")],
    ) -> dict[str, Any]:
        """Evaluate a bundled example plant and keep it in the session."""
        return load_example(example_id)

    @tool(read_only)
    def list_plants_() -> list[dict[str, Any]]:
        """Plants evaluated in this session, with headline metrics."""
        return list_plants()

    @tool(read_only)
    def get_plant_config_(plant_name: PlantName) -> dict[str, Any]:
        """A session plant's inputs (equipment + plant), in the library's
        JSON file layout, to edit and re-evaluate or save for run_tea."""
        return get_plant_config(plant_name)

    @tool(read_only)
    def list_sensitivity_parameters_(
        plant_name: PlantName, metric: Metric = "LCOP",
    ) -> dict[str, Any]:
        """The parameters a sensitivity or tornado analysis can vary on a
        plant, split into economic and process (quantity) parameters."""
        return list_sensitivity_parameters(plant_name, metric)

    @tool(read_only)
    def run_sensitivity_(
        plant_name: PlantName,
        parameter: Annotated[str, Field(
            description="From list_sensitivity_parameters, e.g. "
                        "'fixed_capital', 'electricity', "
                        "'hydrogen.production'.")],
        plus_minus_value: Fraction = 0.2,
        metric: Metric = "LCOP",
        n_points: Annotated[int, Field(ge=3, le=101)] = 11,
        additional_capex: AddCapex = False,
    ) -> dict[str, Any]:
        """How a metric changes as one parameter is varied by
        ±plus_minus_value: the metric at each % change and parameter
        value. Dependencies in the plant are followed."""
        return run_sensitivity(plant_name, parameter, plus_minus_value,
                               metric, n_points, additional_capex)

    @tool(read_only)
    def run_tornado_(
        plant_name: PlantName,
        plus_minus_value: Fraction = 0.2,
        metric: Metric = "LCOP",
        category: Literal["economic", "process", "both"] = "economic",
        additional_capex: AddCapex = False,
    ) -> dict[str, Any]:
        """Rank parameters by how much a ±plus_minus_value change moves a
        metric (tornado chart data, largest swing first)."""
        return run_tornado(plant_name, plus_minus_value, metric, category,
                           additional_capex)

    @tool(read_only)
    def run_monte_carlo_(
        plant_name: PlantName,
        num_samples: Annotated[int, Field(ge=100, le=MAX_MC_SAMPLES)] = 20_000,
        random_seed: Annotated[int | None, Field(
            description="Set for reproducible results.")] = None,
        additional_capex: AddCapex = False,
    ) -> dict[str, Any]:
        """Monte Carlo uncertainty analysis using the uncertainty ranges in
        the plant's configuration (std/min/max on prices etc.; defaults for
        the project-level factors). Returns mean, std and percentiles of
        LCOP (and NPV, ROI, PBT when products are priced) and of the
        sampled inputs."""
        return run_monte_carlo(plant_name, num_samples, random_seed,
                               additional_capex)

    return server


def main():
    """Run the OpenPyTEA MCP server over stdio."""
    # No Monte Carlo progress bars in the client's server log
    os.environ.setdefault("TQDM_DISABLE", "1")
    build_server().run()


if __name__ == "__main__":
    main()
