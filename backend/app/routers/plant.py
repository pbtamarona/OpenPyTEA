"""Plant configuration + calculation endpoints."""

from fastapi import APIRouter, HTTPException
from openpytea.plant import Plant

from app import state
from app.schemas import PlantConfigIn
from app.util import to_jsonable

router = APIRouter()


@router.get("/config")
def get_plant_config():
    return state.plant_config


@router.put("/config")
def set_plant_config(data: PlantConfigIn):
    state.plant_config = data.model_dump()
    return {"ok": True}


@router.get("/locations")
def get_locations():
    return Plant.locFactors


@router.get("/process-types")
def get_process_types():
    return Plant.processTypes


@router.post("/calculate")
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

    cash_flow = {
        "capital_cost_array": to_jsonable(getattr(plant, "capital_cost_array", [])),
        "revenue_array": to_jsonable(getattr(plant, "revenue_array", [])),
        "cash_cost_array": to_jsonable(getattr(plant, "cash_cost_array", [])),
        "gross_profit_array": to_jsonable(getattr(plant, "gross_profit_array", [])),
        "depreciation_array": to_jsonable(getattr(plant, "depreciation_array", [])),
        "taxable_income_array": to_jsonable(getattr(plant, "taxable_income_array", [])),
        "tax_paid_array": to_jsonable(getattr(plant, "tax_paid_array", [])),
        "cash_flow": to_jsonable(getattr(plant, "cash_flow", [])),
        "production_array": to_jsonable(getattr(plant, "prod_array", [])),
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
