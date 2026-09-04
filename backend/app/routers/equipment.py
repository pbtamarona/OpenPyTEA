"""Equipment CRUD + cost database lookup endpoints."""

from fastapi import APIRouter, HTTPException
from openpytea.equipment import Equipment, CostCorrelationDB

from app import state
from app.plant_factory import equipment_from_entry
from app.schemas import EquipmentIn, EquipmentOut, OkResponse, CostDBEntry

router = APIRouter()

_db = CostCorrelationDB()


def _eq_to_out(i: int, eq: Equipment) -> dict:
    param = eq.param
    if isinstance(param, tuple):
        param = list(param)
    return EquipmentOut(
        index=i,
        name=eq.name,
        category=eq.category,
        type=eq.type,
        material=eq.material,
        process_type=eq.process_type,
        param=param,
        num_units=eq.num_units,
        num_units_input=getattr(eq, "_requested_num_units", None),
        cost_func=eq._cost_func,
        cost_year=eq.cost_year,
        target_year=eq.target_year,
        purchased_cost=float(eq.purchased_cost),
        direct_cost=float(eq.direct_cost),
    ).model_dump()


def _make_equipment(data: EquipmentIn) -> Equipment:
    return equipment_from_entry(data.model_dump())


@router.get("", response_model=list[EquipmentOut])
def list_equipment():
    return [_eq_to_out(i, eq) for i, eq in enumerate(state.equipment_list)]


@router.post("", response_model=EquipmentOut)
def add_equipment(data: EquipmentIn):
    try:
        eq = _make_equipment(data)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid equipment parameters: {e}")
    state.equipment_list.append(eq)
    return _eq_to_out(len(state.equipment_list) - 1, eq)


@router.put("/{index}", response_model=EquipmentOut)
def update_equipment(index: int, data: EquipmentIn):
    if index < 0 or index >= len(state.equipment_list):
        raise HTTPException(status_code=404, detail="Equipment not found")
    try:
        eq = _make_equipment(data)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid equipment parameters: {e}")
    state.equipment_list[index] = eq
    return _eq_to_out(index, eq)


@router.delete("/{index}", response_model=OkResponse)
def delete_equipment(index: int):
    if index < 0 or index >= len(state.equipment_list):
        raise HTTPException(status_code=404, detail="Equipment not found")
    state.equipment_list.pop(index)
    return {"ok": True}


@router.get("/cost-db/categories", response_model=dict[str, list[CostDBEntry]])
def get_cost_db_categories():
    """Return grouped categories with their types, units, and param ranges."""
    df = _db.df
    groups = {}
    for _, row in df.iterrows():
        cat = row.get("category", "")
        if cat not in groups:
            groups[cat] = []
        default_material = row.get("default material")
        if not isinstance(default_material, str) or default_material.strip().lower() in ("", "n.a.", "na", "none"):
            default_material = None
        groups[cat].append({
            "key": row.get("key", ""),
            "type": row.get("type", None),
            "units": row.get("units", ""),
            "s_lower": float(row["s_lower"]) if not _isnan(row.get("s_lower")) else None,
            "s_upper": float(row["s_upper"]) if not _isnan(row.get("s_upper")) else None,
            "s2_lower": float(row["s2_lower"]) if not _isnan(row.get("s2_lower")) else None,
            "s2_upper": float(row["s2_upper"]) if not _isnan(row.get("s2_upper")) else None,
            "default_material": default_material,
        })
    return groups


@router.get("/process-types", response_model=list[str])
def get_process_types():
    return list(Equipment.process_factors.keys())


@router.get("/materials", response_model=list[str])
def get_materials():
    return list(Equipment.material_factors.keys())


def _isnan(v):
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True
