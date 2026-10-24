"""Plant configuration + calculation endpoints."""

import copy
import json

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from openpytea.io import _safe_filename
from openpytea.plant import DEFAULT_OPERATOR_HOURLY_RATE, Plant

from app import project_store, state
from app.plant_factory import mark_plant_fresh
from app.schemas import (
    PlantConfigIn, CalculationResults, OkResponse,
    ExportedFile, PlantImportIn, PlantImportOut,
)
from app.util import to_jsonable

router = APIRouter()


@router.get("/config")
def get_plant_config():
    return state.plant_config


@router.put("/config", response_model=OkResponse)
def set_plant_config(data: PlantConfigIn):
    config = data.model_dump()
    if config.get("project_uncertainties") is None:
        config.pop("project_uncertainties", None)
    state.plant_config = config
    return {"ok": True}


def _unused_name(name: str) -> str:
    """`name`, numbered if another plant of the project already has it."""
    others = {s["name"] for s in state.plants if s["id"] != state.active_plant_id}
    base, i = name, 2
    while name in others:
        name, i = f"{base} {i}", i + 1
    return name


@router.post("/import", response_model=PlantImportOut)
def import_plant(data: PlantImportIn):
    """Replace the open plant's configuration with a plant input file."""
    raw = data.data.get("plant")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Not a valid plant file: it must contain a top-level 'plant' key.")
    raw = copy.deepcopy(raw)
    raw.pop("equipment", None)  # equipment comes from the Equipment tab
    # The library takes a bare rate, and no rate means its default with no
    # uncertainty (the form's default adds some)
    rate = raw.get("operator_hourly_rate")
    if rate is None:
        raw["operator_hourly_rate"] = {"rate": DEFAULT_OPERATOR_HOURLY_RATE}
    elif not isinstance(rate, dict):
        raw["operator_hourly_rate"] = {"rate": rate}
    try:
        config = PlantConfigIn(**raw).model_dump()
    except ValidationError as e:
        err = e.errors()[0]
        where = ".".join(str(p) for p in err["loc"])
        raise HTTPException(status_code=400, detail=f"Not a valid plant file: {where}: {err['msg']}")
    if config.get("project_uncertainties") is None:
        config.pop("project_uncertainties", None)
    try:
        Plant({**copy.deepcopy(config), "equipment": state.equipment_list})
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Not a valid plant file: {e}")

    notes = []
    project_store.ensure_project()
    name = _unused_name(config.get("plant_name") or project_store.active_slot()["name"])
    if name != config.get("plant_name"):
        notes.append(f"Named '{name}': another plant of the project is called '{config.get('plant_name')}'.")
    config["plant_name"] = name
    project_store.active_slot()["name"] = name
    extra = sorted(set(config) - set(PlantConfigIn.model_fields))
    if extra:
        notes.append(f"Kept as in the file (no form for them here): {', '.join(extra)}.")
    state.plant_config = config
    return {"config": to_jsonable(config), "notes": notes}


@router.get("/export-input", response_model=ExportedFile)
def export_plant_input():
    """The open plant's configuration as a file load_plant_config reads."""
    config = {k: v for k, v in state.plant_config.items() if v is not None}
    name = _safe_filename(config.get("plant_name") or "plant")
    return {"name": f"{name}_plant_input.json", "content": json.dumps({"plant": to_jsonable(config)}, indent=2)}


@router.get("/locations")
def get_locations():
    return Plant.locFactors


@router.get("/process-types")
def get_process_types():
    return Plant.processTypes


@router.post("/calculate", response_model=CalculationResults)
def calculate():
    if not state.plant_config:
        raise HTTPException(status_code=400, detail="Plant not configured")
    if not state.equipment_list:
        raise HTTPException(status_code=400, detail="No equipment defined")

    config = dict(state.plant_config)
    config["equipment"] = state.equipment_list

    try:
        plant = Plant(config)
        plant.calculate_all()
        state.plant = plant
        mark_plant_fresh()
    except Exception:
        raise HTTPException(status_code=400, detail="Calculation failed — check equipment and plant configuration")

    results = _extract_results(plant)
    state.results = results
    return results


def _extract_results(plant: Plant) -> dict:
    """Extract all calculated results into a JSON-safe dict."""
    capital_costs = {
        "purchased_cost": float(getattr(plant, "purchased_cost", 0)),
        "isbl": float(getattr(plant, "isbl", 0)),
        "osbl": float(getattr(plant, "osbl", 0)),
        "design_and_engineering": float(getattr(plant, "dne", 0)),
        "contingency": float(getattr(plant, "contigency", 0)),
        "fixed_capital": float(getattr(plant, "fixed_capital", 0)),
        "working_capital": float(plant.working_capital) if plant.working_capital is not None else None,
    }

    variable_opex = {
        "breakdown": to_jsonable(getattr(plant, "variable_opex_breakdown", {})),
        "total": float(getattr(plant, "variable_production_costs", 0)),
    }

    fixed_opex = {
        "operating_labor": float(getattr(plant, "operating_labor_costs", 0)),
        "supervision": float(getattr(plant, "supervision_costs", 0)),
        "direct_salary_overhead": float(getattr(plant, "direct_salary_overhead", 0)),
        "laboratory_charges": float(getattr(plant, "laboratory_charges", 0)),
        "maintenance": float(getattr(plant, "maintenance_costs", 0)),
        "taxes_insurance": float(getattr(plant, "taxes_insurance_costs", 0)),
        "rent_of_land": float(getattr(plant, "rent_of_land_costs", 0)),
        "environmental_charges": float(getattr(plant, "environmental_charges", 0)),
        "operating_supplies": float(getattr(plant, "operating_supplies", 0)),
        "general_plant_overhead": float(getattr(plant, "general_plant_overhead", 0)),
        "interest_working_capital": float(getattr(plant, "interest_working_capital", 0)),
        "patents_royalties": float(getattr(plant, "patents_royalties", 0)),
        "distribution_selling": float(getattr(plant, "distribution_selling_costs", 0)),
        "rnd": float(getattr(plant, "RnD_costs", 0)),
        "total": float(getattr(plant, "fixed_production_costs", 0)),
    }

    revenue = {
        "breakdown": to_jsonable(getattr(plant, "revenue_breakdown", {})),
        "total": float(getattr(plant, "revenue", 0)),
    }

    def _flatten(arr):
        """Flatten 2D (1, N) arrays to 1D lists."""
        val = to_jsonable(arr)
        if val and isinstance(val[0], list):
            return val[0]
        return val

    cash_flow = {
        "capital_cost_array": _flatten(getattr(plant, "capital_cost_array", [])),
        "revenue_array": _flatten(getattr(plant, "revenue_array", [])),
        "cash_cost_array": _flatten(getattr(plant, "cash_cost_array", [])),
        "gross_profit_array": _flatten(getattr(plant, "gross_profit_array", [])),
        "depreciation_array": _flatten(getattr(plant, "depreciation_array", [])),
        "taxable_income_array": _flatten(getattr(plant, "taxable_income_array", [])),
        "tax_paid_array": _flatten(getattr(plant, "tax_paid_array", [])),
        "cash_flow": _flatten(getattr(plant, "cash_flow", [])),
        "production_array": _flatten(getattr(plant, "prod_array", [])),
    }

    metrics = {
        "levelized_cost": to_jsonable(getattr(plant, "levelized_cost", None)),
        "npv": to_jsonable(getattr(plant, "npv", None)),
        "irr": to_jsonable(getattr(plant, "irr", None)),
        "roi": to_jsonable(getattr(plant, "roi", None)),
        "payback_time": to_jsonable(getattr(plant, "payback_time", None)),
    }

    return to_jsonable({
        "capital_costs": capital_costs,
        "variable_opex": variable_opex,
        "fixed_opex": fixed_opex,
        "revenue": revenue,
        "cash_flow": cash_flow,
        "metrics": metrics,
    })
