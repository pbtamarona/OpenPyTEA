"""Equipment CRUD + cost database lookup endpoints."""

from fastapi import APIRouter, HTTPException
from openpytea.equipment import Equipment, CostCorrelationDB

from app import state
from app.schemas import EquipmentIn, EquipmentOut, OkResponse, CostDBEntry

router = APIRouter()

_db = CostCorrelationDB()


def _eq_to_out(i: int, eq: Equipment) -> dict:
    return EquipmentOut(
        index=i,
        name=eq.name,
        category=eq.category,
        type=eq.type,
        material=eq.material,
        process_type=eq.process_type,
        param=eq.param,
        num_units=eq.num_units,
        cost_year=eq.cost_year,
        target_year=eq.target_year,
        purchased_cost=float(eq.purchased_cost),
        direct_cost=float(eq.direct_cost),
    ).model_dump()


def _make_equipment(data: EquipmentIn) -> Equipment:
    return Equipment(
        name=data.name,
        param=data.param if data.param is not None else 0.0,
        process_type=data.process_type,
        category=data.category,
        type=data.type,
        material=data.material,
        num_units=data.num_units,
        purchased_cost=data.purchased_cost,
        cost_year=data.cost_year,
        cost_func=data.cost_func,
        target_year=data.target_year,
    )


@router.get("", response_model=list[EquipmentOut])
def list_equipment():
    return [_eq_to_out(i, eq) for i, eq in enumerate(state.equipment_list)]


@router.post("", response_model=EquipmentOut)
def add_equipment(data: EquipmentIn):
    try:
        eq = _make_equipment(data)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    state.equipment_list.append(eq)
    return _eq_to_out(len(state.equipment_list) - 1, eq)


@router.put("/{index}", response_model=EquipmentOut)
def update_equipment(index: int, data: EquipmentIn):
    if index < 0 or index >= len(state.equipment_list):
        raise HTTPException(status_code=404, detail="Equipment not found")
    try:
        eq = _make_equipment(data)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        groups[cat].append({
            "key": row.get("key", ""),
            "type": row.get("type", None),
            "units": row.get("units", ""),
            "s_lower": float(row["s_lower"]) if not _isnan(row.get("s_lower")) else None,
            "s_upper": float(row["s_upper"]) if not _isnan(row.get("s_upper")) else None,
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
