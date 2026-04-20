"""Sensitivity, tornado, and Monte Carlo analysis endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from openpytea.analysis import sensitivity_data, tornado_data, monte_carlo

from app import state
from app.schemas import (
    SensitivityIn, TornadoIn, MonteCarloIn,
    SensitivityResult, TornadoResult, MonteCarloResult,
)
from app.util import to_jsonable

router = APIRouter()


def _require_plant():
    if state.plant is None:
        raise HTTPException(status_code=400, detail="Run calculations first")
    return state.plant


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
    try:
        result = sensitivity_data(
            plant,
            parameter=data.parameter,
            plus_minus_value=data.plus_minus_value,
            n_points=data.n_points,
            metric=data.metric,
            additional_capex=data.additional_capex,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return to_jsonable(result)


@router.post("/tornado", response_model=TornadoResult)
def run_tornado(data: TornadoIn):
    plant = _require_plant()
    try:
        result = tornado_data(
            plant,
            plus_minus_value=data.plus_minus_value,
            metric=data.metric,
            additional_capex=data.additional_capex,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return to_jsonable(result)


@router.post("/monte-carlo", response_model=MonteCarloResult)
def run_monte_carlo(data: MonteCarloIn):
    plant = _require_plant()
    try:
        result = monte_carlo(
            plant,
            num_samples=data.num_samples,
            batch_size=data.batch_size,
            additional_capex=data.additional_capex,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    state.mc_results = result

    # Summarize for JSON response (don't send million-element arrays)
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
