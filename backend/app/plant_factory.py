"""Helpers to build Plant objects from raw payloads (saved JSON or session state)."""

from fastapi import HTTPException

from openpytea.equipment import Equipment, CompositeEquipment
from openpytea.plant import Plant


MAX_EQUIPMENT = 500
MAX_COMPONENTS = 100

# Input keys worth preserving for a save/load round trip. Everything else
# on an entry (computed costs, UI leftovers) is dropped.
_EQUIPMENT_SPEC_KEYS = (
    "name", "param", "process_type", "category", "type", "material",
    "num_units", "purchased_cost", "cost_year", "cost_func", "target_year",
)
_COMPOSITE_SPEC_KEYS = (
    "name", "process_type", "category", "type", "installation",
    "num_units", "purchased_cost", "cost_year", "target_year",
)


def _clean_spec(entry: dict, keys: tuple) -> dict:
    return {k: entry[k] for k in keys if entry.get(k) is not None}


def equipment_from_entry(entry: dict, _depth: int = 0):
    """Build one Equipment or CompositeEquipment from a raw payload entry.

    An entry with a non-empty `components` list becomes a CompositeEquipment;
    components are built recursively (they may themselves be composites).

    `material` may be absent/None, in which case the library resolves the
    correlation's default material. `param` may be a 2-element list for
    2-variable correlations.

    The library overwrites `num_units` with the auto-parallelization result
    when the caller didn't set it, so the caller's original value is kept on
    `_requested_num_units` — save/serialize that, never the resolved value,
    or reloading would multiply an already-parallelized cost. For the same
    reason (and because a composite quote is inflation-adjusted in place),
    the sanitized original input is kept on `_input_spec`; saving writes
    that spec verbatim, so reloads always rebuild from user inputs.
    """
    if _depth > 3:
        raise ValueError("Composite equipment nested too deeply (max 3 levels)")

    components = entry.get("components")
    if components:
        if len(components) > MAX_COMPONENTS:
            raise ValueError(f"Too many components (max {MAX_COMPONENTS})")
        target_year = entry.get("target_year", 2024)
        built = [
            equipment_from_entry({**c, "target_year": target_year}, _depth + 1)
            for c in components
        ]
        eq = CompositeEquipment(
            name=entry["name"],
            process_type=entry["process_type"],
            components=built,
            category=entry.get("category") or "Composite",
            type=entry.get("type"),
            installation=entry.get("installation") or "component",
            purchased_cost=entry.get("purchased_cost"),
            cost_year=entry.get("cost_year"),
            target_year=target_year,
            num_units=entry.get("num_units") or 1,
        )
        eq._requested_num_units = entry.get("num_units")
        eq._input_spec = {
            **_clean_spec(entry, _COMPOSITE_SPEC_KEYS),
            "components": [c._input_spec for c in built],
        }
        return eq

    requested_num_units = entry.get("num_units")
    param = entry.get("param", 0.0)
    if isinstance(param, list):
        param = tuple(param)
    eq = Equipment(
        name=entry["name"],
        param=param if param is not None else 0.0,
        process_type=entry["process_type"],
        category=entry["category"],
        type=entry.get("type"),
        material=entry.get("material"),
        num_units=requested_num_units,
        purchased_cost=entry.get("purchased_cost"),
        cost_year=entry.get("cost_year"),
        cost_func=entry.get("cost_func"),
        target_year=entry.get("target_year", 2024),
    )
    eq._requested_num_units = requested_num_units
    eq._input_spec = _clean_spec(entry, _EQUIPMENT_SPEC_KEYS)
    return eq


def build_equipment_list(payload: list[dict]) -> list[Equipment]:
    if len(payload) > MAX_EQUIPMENT:
        raise HTTPException(status_code=400, detail=f"Too many equipment items (max {MAX_EQUIPMENT})")

    equipment_list: list[Equipment] = []
    for entry in payload:
        try:
            equipment_list.append(equipment_from_entry(entry))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid equipment entry '{entry.get('name', '?')}'",
            )
    return equipment_list


def build_plant(equipment_payload: list[dict], plant_config: dict) -> Plant:
    """Rehydrate a fully calculated Plant from raw equipment + plant config payloads."""
    if not plant_config:
        raise HTTPException(status_code=400, detail="Missing plant configuration")
    if not equipment_payload:
        raise HTTPException(status_code=400, detail="Missing equipment list")

    equipment_list = build_equipment_list(equipment_payload)
    config = dict(plant_config)
    config["equipment"] = equipment_list
    try:
        plant = Plant(config)
        plant.calculate_all()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to build plant '{plant_config.get('plant_name', '?')}' — check equipment and plant configuration",
        )
    return plant
