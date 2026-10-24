"""Equipment CRUD + cost database lookup endpoints."""

import copy
import json

from fastapi import APIRouter, HTTPException
from openpytea.equipment import Equipment, CompositeEquipment, CostCorrelationDB
from openpytea.io import _FACTOR_KEYS, _build_equipment_list, _safe_filename

from app import state
from app.plant_factory import MAX_EQUIPMENT, build_equipment_list, equipment_from_entry
from app.project_store import equipment_specs
from app.schemas import (
    EquipmentIn, EquipmentOut, OkResponse, CostDBEntry,
    EquipmentImportIn, ExportedFile, ImportResult,
)
from app.util import to_jsonable

router = APIRouter()

_db = CostCorrelationDB()


def _eq_to_out(i: int, eq) -> dict:
    param = eq.param
    if isinstance(param, tuple):
        param = list(param)
    is_composite = isinstance(eq, CompositeEquipment)
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
        cost_func=None if is_composite else eq._cost_func,
        cost_year=eq.cost_year,
        target_year=eq.target_year,
        purchased_cost=float(eq.purchased_cost),
        direct_cost=float(eq.direct_cost),
        is_composite=is_composite,
        installation=eq.installation if is_composite else None,
        components_purchased_cost=(
            float(eq.components_purchased_cost) if is_composite else None
        ),
        quoted_purchased_cost=(
            eq._input_spec.get("purchased_cost") if is_composite else None
        ),
        quoted_cost_year=(
            eq._input_spec.get("cost_year") if is_composite else None
        ),
        components=(
            [
                {
                    "spec": comp._input_spec,
                    "purchased_cost": float(comp.purchased_cost),
                    "direct_cost": float(comp.direct_cost),
                }
                for comp in eq.components
            ]
            if is_composite
            else None
        ),
        factors={k: v for k, v in getattr(eq, "_input_spec", {}).items() if k in _FACTOR_KEYS},
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


def _import_notes(entries: list[dict]) -> list[str]:
    """What the GUI builds differently from load_equipment_config."""
    notes = []
    for entry in entries:
        year = entry.get("target_year", 2024)
        for comp in entry.get("components") or []:
            if comp.get("target_year", year) != year:
                notes.append(
                    f"Components of '{entry['name']}' use its target year {year}, "
                    f"not their own ({comp.get('name', '?')}: {comp['target_year']})."
                )
                break
    return notes


@router.post("/import", response_model=ImportResult)
def import_equipment(data: EquipmentImportIn):
    """Read an equipment input file: replace the list or add to it."""
    # The library's loader decides what a valid file is (and gives its
    # error messages); the GUI then builds its own objects from the entries
    try:
        _build_equipment_list(copy.deepcopy(data.data))
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Not a valid equipment file: {e}")
    entries = data.data["equipment"]
    keep = state.equipment_list if data.mode == "add" else []
    if len(keep) + len(entries) > MAX_EQUIPMENT:
        raise HTTPException(status_code=400, detail=f"Too many equipment items (max {MAX_EQUIPMENT})")
    built = build_equipment_list(entries)
    state.equipment_list = [*keep, *built]
    return {"count": len(built), "notes": _import_notes(entries)}


@router.get("/export-input", response_model=ExportedFile)
def export_equipment_input():
    """The open plant's equipment as a file load_equipment_config reads."""
    name = _safe_filename(state.plant_config.get("plant_name") or "plant")
    payload = {"equipment": to_jsonable(equipment_specs(state.equipment_list))}
    return {"name": f"{name}_equipment_input.json", "content": json.dumps(payload, indent=2)}


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
