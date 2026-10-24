"""Sensitivity, tornado, and Monte Carlo analysis endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from openpytea.analysis import sensitivity_data, tornado_data, monte_carlo
from openpytea.helpers import _collect_dependency_specs, _sensitivity_key_node

from app import state
from app.plant_factory import build_plant, require_active_plant
from app.schemas import (
    SensitivityIn, TornadoIn, MonteCarloIn,
    SensitivityResult, TornadoResult,
    MonteCarloMultiResult,
    PlantInput,
    AnalysisImportIn, AnalysisImportOut,
)
from app.util import to_jsonable

router = APIRouter()


def _require_plant():
    return require_active_plant()


def _plants_for(data):
    """The plants a run covers: the open plant (unless left out) followed
    by the extra plants, in the order they were sent."""
    plants = [_require_plant()] if data.include_active else []
    plants += _rehydrate_extras(data.extra_plants)
    if not plants:
        raise HTTPException(status_code=400, detail="Select at least one plant")
    return plants


def _rehydrate_extras(extras: list[PlantInput]):
    """Build Plant objects from saved-JSON-shaped payloads.

    If the PlantInput carries a `name` (the label the user picked on the
    Compare tab), it overrides whatever `plant_name` was baked into the
    saved JSON — so analysis legends show the user's chosen name.
    """
    plants = []
    for extra in extras:
        p = build_plant(extra.equipment, extra.plant)
        if extra.name:
            p.name = extra.name
        plants.append(p)
    return plants


@router.get("/sensitivity/parameters", response_model=list[str])
def get_sensitivity_parameters():
    plant = _require_plant()
    top = ["fixed_capital", "fixed_opex", "project_lifetime", "interest_rate", "operator_hourly_rate"]
    # Bare item keys vary the price; explicit .consumption/.production keys
    # vary the process quantity (3.0).
    var_keys = [f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs]
    prod_keys = [f"plant_products.{k}" for k in plant.plant_products]
    qty_keys = [f"variable_opex_inputs.{k}.consumption" for k in plant.variable_opex_inputs]
    qty_keys += [f"plant_products.{k}.production" for k in plant.plant_products]
    # A parameter set by a dependency has no value of its own to vary —
    # sensitivity_data would reject it, so don't offer it.
    dependents = _collect_dependency_specs(plant)
    return [
        k for k in top + var_keys + prod_keys + qty_keys
        if _sensitivity_key_node(k) not in dependents
    ]


@router.post("/sensitivity", response_model=SensitivityResult)
def run_sensitivity(data: SensitivityIn):
    plants = _plants_for(data)
    try:
        result = sensitivity_data(
            plants,
            parameter=data.parameter,
            plus_minus_value=data.plus_minus_value,
            n_points=data.n_points,
            metric=data.metric,
            additional_capex=data.additional_capex,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Sensitivity analysis failed: {e}")
    if data.include_active:
        _record_sensitivity(data)
    return to_jsonable(result)


def _record_sensitivity(data: SensitivityIn):
    """Remember the run for the open plant's Save as JSON analysis file."""
    state.sensitivity_args[(data.parameter, data.metric.upper())] = {
        "parameter": data.parameter,
        "plus_minus_value": data.plus_minus_value,
        "n_points": data.n_points,
        "metric": data.metric,
        "additional_capex": data.additional_capex,
    }


@router.post("/tornado", response_model=TornadoResult)
def run_tornado(data: TornadoIn):
    plants = _plants_for(data)

    per_plant: list[dict] = []
    xlabel = ""
    for p in plants:
        try:
            r = tornado_data(
                p,
                plus_minus_value=data.plus_minus_value,
                metric=data.metric,
                additional_capex=data.additional_capex,
                category=data.category,
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Tornado analysis failed for plant '{getattr(p, 'plant_name', '?')}': {e}",
            )
        xlabel = r.get("xlabel", xlabel)
        per_plant.append({
            "name": getattr(p, "name", None) or "Plant",
            "factors": r["factors"],
            "labels": r["labels"],
            "lows": r["lows"],
            "highs": r["highs"],
            "base_value": r["base_value"],
        })

    if data.include_active:
        state.tornado_args = {
            "plus_minus_value": data.plus_minus_value,
            "metric": data.metric,
            "additional_capex": data.additional_capex,
            "category": data.category,
        }
    return to_jsonable({
        "plants": per_plant,
        "plus_minus_value": data.plus_minus_value,
        "metric": data.metric.upper(),
        "xlabel": xlabel,
    })


def _summarize_mc(result: dict) -> dict:
    """Histogram + percentile summary of one monte_carlo() result."""
    summary = {
        "name": result["name"],
        "num_samples": result["num_samples"],
        "currency": result["currency"],
        "metrics": {},
        "inputs": {},
    }

    for metric_name, values in result["metrics"].items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or np.all(arr == 0):
            continue
        counts, bin_edges = np.histogram(arr, bins=80)
        summary["metrics"][metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "histogram": {
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist(),
            },
        }

    for input_name, values in result["inputs"].items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        counts, bin_edges = np.histogram(arr, bins=50)
        summary["inputs"][input_name] = {
            "label": result.get("display_labels", {}).get(input_name),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "histogram": {
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist(),
            },
        }

    return summary


def _run_mc_for_plant(plant, num_samples: int, batch_size: int, additional_capex: bool) -> dict:
    try:
        return monte_carlo(
            plant,
            num_samples=num_samples,
            batch_size=batch_size,
            additional_capex=additional_capex,
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Monte Carlo analysis failed for plant '{getattr(plant, 'name', '?')}' — check configuration",
        )


@router.post("/monte-carlo", response_model=MonteCarloMultiResult)
def run_monte_carlo(data: MonteCarloIn):
    plants = _plants_for(data)

    raws: list[dict] = []
    summaries: list[dict] = []
    for p in plants:
        raw = _run_mc_for_plant(p, data.num_samples, data.batch_size, data.additional_capex)
        raws.append(raw)
        summaries.append(_summarize_mc(raw))

    # Cache the raw results (full sample arrays) so /api/plots/monte-carlo
    # can re-render this exact run with the library's matplotlib functions.
    state.mc_raw = raws
    # Only a run that includes the open plant (first) counts as its
    # Monte Carlo result for Save as JSON
    state.mc_results = summaries[0] if data.include_active else None
    state.mc_snapshot = state.calc_snapshot if data.include_active else None

    return {"plants": summaries}


# ── Analysis input files ───────────────────────────────────────────

_METRICS = ("LCOP", "NPV", "IRR", "ROI", "PBT")
_TOP_PARAMETERS = ("fixed_capital", "fixed_opex", "project_lifetime", "interest_rate", "operator_hourly_rate")
_DETERMINISTIC = ("direct_costs", "fixed_capital", "fixed_opex", "variable_opex", "levelized_cost", "cash_flow")


def _resolve_parameter(parameter: str, config: dict) -> str:
    """Full sensitivity key of a parameter as written in an analysis file,
    resolving shorthands ("electricity", "steam.consumption") against the
    open plant the way sensitivity_data does."""
    items = config.get("variable_opex_inputs") or {}
    products = config.get("plant_products") or {}
    if parameter in _TOP_PARAMETERS:
        return parameter
    head, _, field = parameter.partition(".")
    if head in ("variable_opex_inputs", "plant_products"):
        item, _, field = field.partition(".")
        pool = items if head == "variable_opex_inputs" else products
        if item in pool and field in ("", "consumption" if pool is items else "production"):
            return parameter
        raise ValueError(f"'{parameter}' is not in this plant")
    if field == "consumption" and head in items:
        return f"variable_opex_inputs.{parameter}"
    if field == "production" and head in products:
        return f"plant_products.{parameter}"
    if field:
        raise ValueError(f"'{parameter}' is not in this plant")
    if parameter in items and parameter in products:
        raise ValueError(f"'{parameter}' is both a variable OPEX item and a product; use the full path")
    if parameter in items:
        return f"variable_opex_inputs.{parameter}"
    if parameter in products:
        return f"plant_products.{parameter}"
    raise ValueError(f"'{parameter}' is not in this plant")


def _metric(value, where: str, notes: list[str]) -> str:
    metric = str(value or "LCOP").upper()
    if metric not in _METRICS:
        notes.append(f"{where}: metric {value!r} isn't available here; using LCOP.")
        return "LCOP"
    return metric


def _clamped(value, default, lo, hi, where: str, notes: list[str]):
    try:
        v = type(default)(value) if value is not None else default
    except (TypeError, ValueError):
        notes.append(f"{where}: {value!r} isn't a number; using {default}.")
        return default
    if not lo <= v <= hi:
        v = min(max(v, lo), hi)
        notes.append(f"{where}: {value} is outside {lo}–{hi}; using {v}.")
    return v


@router.post("/import-config", response_model=AnalysisImportOut)
def import_analysis_config(data: AnalysisImportIn):
    """Read an analysis input file into the settings of the Analysis and
    Monte Carlo tabs. Only blocks with "run": true are taken, as run_tea
    does; nothing is run."""
    block = data.data.get("analysis")
    if not isinstance(block, dict):
        raise HTTPException(status_code=400, detail="Not a valid analysis file: it must contain an 'analysis' key.")
    config = state.plant_config
    notes: list[str] = []
    out: dict = {}

    def runs(key):
        cfg = block.get(key)
        return isinstance(cfg, dict) and cfg.get("run")

    if runs("sensitivity"):
        cases = []
        for i, case in enumerate(block["sensitivity"].get("cases") or [], start=1):
            args = case.get("args") or {}
            name = case.get("name") or f"case_{i}"
            where = f"Sensitivity '{name}'"
            try:
                parameter = _resolve_parameter(str(args.get("parameter", "")), config)
            except ValueError as e:
                notes.append(f"{where} left out: {e}.")
                continue
            x_axis = (case.get("plot_args") or {}).get("x_axis", "percent")
            cases.append({
                "name": name,
                "parameter": parameter,
                "plus_minus_value": _clamped(args.get("plus_minus_value"), 0.2, 0.01, 10.0, where, notes),
                "n_points": _clamped(args.get("n_points"), 21, 3, 1000, where, notes),
                "metric": _metric(args.get("metric"), where, notes),
                "x_axis": x_axis if x_axis in ("percent", "value") else "percent",
            })
        out["sensitivity"] = cases

    if runs("tornado"):
        args = block["tornado"].get("args") or {}
        category = args.get("category")
        if category is None:
            category = "both" if args.get("include_process_params") else "economic"
        if category not in ("economic", "process", "both"):
            notes.append(f"Tornado: category {category!r} isn't known; using economic.")
            category = "economic"
        out["tornado"] = {
            "plus_minus_value": _clamped(args.get("plus_minus_value"), 0.2, 0.01, 10.0, "Tornado", notes),
            "metric": _metric(args.get("metric"), "Tornado", notes),
            "category": category,
        }

    if runs("monte_carlo"):
        args = block["monte_carlo"].get("args") or {}
        out["monte_carlo"] = {
            "num_samples": _clamped(args.get("num_samples"), 100_000, 100, 5_000_000, "Monte Carlo", notes),
            "batch_size": _clamped(args.get("batch_size"), 10_000, 10, 100_000, "Monte Carlo", notes),
        }
        if args.get("random_seed") is not None:
            notes.append("Monte Carlo: random_seed isn't used here; each run draws new samples.")

    if block.get("additional_capex") or any(
        isinstance(block.get(k), dict) and (block[k].get("args") or {}).get("additional_capex")
        for k in ("tornado", "monte_carlo", "fixed_capital")
    ):
        notes.append("additional_capex isn't supported here; the analyses run without it.")
    return {**out, "notes": notes}
