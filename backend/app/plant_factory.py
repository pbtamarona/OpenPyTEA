"""Helpers to build Plant objects from raw payloads (saved JSON or session state)."""

from fastapi import HTTPException

from openpytea.equipment import Equipment
from openpytea.plant import Plant


MAX_EQUIPMENT = 500


def build_equipment_list(payload: list[dict]) -> list[Equipment]:
    if len(payload) > MAX_EQUIPMENT:
        raise HTTPException(status_code=400, detail=f"Too many equipment items (max {MAX_EQUIPMENT})")

    equipment_list: list[Equipment] = []
    for entry in payload:
        try:
            eq = Equipment(
                name=entry["name"],
                param=entry.get("param", 0.0),
                process_type=entry["process_type"],
                category=entry["category"],
                type=entry.get("type"),
                material=entry.get("material", "Carbon steel"),
                num_units=entry.get("num_units"),
                purchased_cost=entry.get("purchased_cost"),
                cost_year=entry.get("cost_year"),
                target_year=entry.get("target_year", 2024),
            )
            equipment_list.append(eq)
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
