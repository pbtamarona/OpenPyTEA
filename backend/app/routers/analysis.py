"""Sensitivity, tornado, and Monte Carlo analysis endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from openpytea.analysis import sensitivity_data, tornado_data, monte_carlo

from app import state
from app.plant_factory import build_plant
from app.schemas import (
    SensitivityIn, TornadoIn, MonteCarloIn,
    SensitivityResult, TornadoResult,
    MonteCarloMultiResult,
    PlantInput,
)
from app.util import to_jsonable

router = APIRouter()


def _require_plant():
    if state.plant is None:
        raise HTTPException(status_code=400, detail="Run calculations first")
    return state.plant


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
    var_keys = [f"variable_opex_inputs.{k}" for k in plant.variable_opex_inputs]
    prod_keys = [f"plant_products.{k}" for k in plant.plant_products]
    return top + var_keys + prod_keys


@router.post("/sensitivity", response_model=SensitivityResult)
def run_sensitivity(data: SensitivityIn):
    plant = _require_plant()
    plants = [plant] + _rehydrate_extras(data.extra_plants)
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
    return to_jsonable(result)


@router.post("/tornado", response_model=TornadoResult)
def run_tornado(data: TornadoIn):
    plant = _require_plant()
    plants = [plant] + _rehydrate_extras(data.extra_plants)

    per_plant: list[dict] = []
    xlabel = ""
    for p in plants:
        try:
            r = tornado_data(
                p,
                plus_minus_value=data.plus_minus_value,
                metric=data.metric,
                additional_capex=data.additional_capex,
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
        result = monte_carlo(
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
    return _summarize_mc(result)


@router.post("/monte-carlo", response_model=MonteCarloMultiResult)
def run_monte_carlo(data: MonteCarloIn):
    plant = _require_plant()
    plants = [plant] + _rehydrate_extras(data.extra_plants)

    summaries: list[dict] = []
    for p in plants:
        summary = _run_mc_for_plant(p, data.num_samples, data.batch_size, data.additional_capex)
        summaries.append(summary)

    # Cache only the active-plant raw result for any future single-plant use.
    state.mc_results = summaries[0] if summaries else None

    return {"plants": summaries}
