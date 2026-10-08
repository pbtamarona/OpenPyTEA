"""Multi-plant project bookkeeping on top of the single-plant session.

The *active* plant lives in the long-standing session fields
(state.equipment_list, state.plant_config, state.plant, state.results and
the recorded analysis runs), so every plant-level router keeps working
unchanged. Every plant of the project — the active one included — has a
slot in state.plants:

    {"id", "name", "equipment": [spec], "plant": config, "results": dict|None,
     "analysis": {"tornado_args", "sensitivity_args"}}

Only inactive slots are authoritative; the active slot is refreshed from
the live session by park_active() before anything reads it.
"""

import copy
import getpass
import uuid
from datetime import date

from openpytea.plant import Plant

from app import state
from app.plant_factory import build_equipment_list, mark_plant_fresh, _calc_snapshot
from app.schemas import PlantConfigIn


def default_plant_config(name: str) -> dict:
    """The Plant Config page's defaults, under the given plant name."""
    config = PlantConfigIn(plant_name=name).model_dump()
    config.pop("project_uncertainties", None)
    return config


def default_meta() -> dict:
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    return {"name": "", "description": "", "user": user,
            "created": date.today().isoformat()}


def equipment_specs(equipment_list) -> list[dict]:
    """Serialize equipment to the input specs it was built from."""
    # Every equipment object is built via plant_factory.equipment_from_entry,
    # which keeps the sanitized original input on _input_spec. Saving that
    # spec (never derived values like the resolved num_units or an
    # inflation-adjusted composite quote) makes save → load a clean rebuild
    # from user inputs.
    specs = []
    for eq in equipment_list:
        spec = getattr(eq, "_input_spec", None)
        if spec is None:
            spec = {
                "name": eq.name,
                "param": list(eq.param) if isinstance(eq.param, tuple) else eq.param,
                "process_type": eq.process_type,
                "category": eq.category,
                "type": eq.type,
                "material": eq.material,
                "num_units": getattr(eq, "_requested_num_units", None),
                "purchased_cost": float(eq.purchased_cost) if eq.param is None else None,
                "cost_year": eq.cost_year,
                "target_year": eq.target_year,
            }
        specs.append(spec)
    return specs


def _new_slot(name: str, equipment: list[dict], config: dict) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "equipment": equipment,
        "plant": config,
        "results": None,
        "analysis": {"tornado_args": None, "sensitivity_args": {}},
    }


def _calculate(equipment_list, config: dict):
    """Calculated Plant for a config + equipment list, or None if invalid."""
    if not config or not equipment_list:
        return None
    try:
        plant = Plant({**config, "equipment": equipment_list})
        plant.calculate_all()
        return plant
    except Exception:
        return None


def active_results() -> dict | None:
    """Results of the active plant, recalculated if its inputs changed."""
    from app.routers.plant import _extract_results
    if state.plant is None or state.calc_snapshot != _calc_snapshot():
        plant = _calculate(state.equipment_list, state.plant_config)
        state.plant = plant
        state.results = _extract_results(plant) if plant else {}
        if plant:
            mark_plant_fresh()
    return state.results or None


def ensure_project():
    """Create the project wrapper around the session on first use."""
    if state.plants:
        return
    state.project_meta = default_meta()
    name = state.plant_config.get("plant_name") or "Plant 1"
    if not state.plant_config and not state.equipment_list:
        state.plant_config = default_plant_config(name)
    slot = _new_slot(name, [], {})
    state.plants = [slot]
    state.active_plant_id = slot["id"]
    park_active()


def _slot(plant_id: str) -> dict | None:
    return next((s for s in state.plants if s["id"] == plant_id), None)


def active_slot() -> dict:
    return _slot(state.active_plant_id)


def park_active():
    """Copy the live session into the active plant's slot."""
    slot = active_slot()
    if slot is None:
        return
    slot["equipment"] = copy.deepcopy(equipment_specs(state.equipment_list))
    slot["plant"] = copy.deepcopy(state.plant_config)
    slot["name"] = state.plant_config.get("plant_name") or slot["name"]
    slot["results"] = active_results()
    slot["analysis"] = {
        "tornado_args": state.tornado_args,
        "sensitivity_args": dict(state.sensitivity_args),
    }


def _load_slot(slot: dict):
    """Make `slot` the live session (the caller has parked the old one)."""
    from app.routers.plant import _extract_results
    state.equipment_list = build_equipment_list(slot["equipment"])
    state.plant_config = copy.deepcopy(slot["plant"])
    state.active_plant_id = slot["id"]
    state.reset_analysis_runs()
    state.tornado_args = slot["analysis"]["tornado_args"]
    state.sensitivity_args = dict(slot["analysis"]["sensitivity_args"])
    plant = _calculate(state.equipment_list, state.plant_config)
    state.plant = plant
    state.results = _extract_results(plant) if plant else {}
    if plant:
        mark_plant_fresh()
    slot["results"] = state.results or None


def activate(plant_id: str) -> bool:
    slot = _slot(plant_id)
    if slot is None:
        return False
    if plant_id != state.active_plant_id:
        park_active()
        _load_slot(slot)
    return True


def _unique_name(base: str) -> str:
    taken = {s["name"] for s in state.plants}
    if base not in taken:
        return base
    i = 2
    while f"{base} {i}" in taken:
        i += 1
    return f"{base} {i}"


def add_plant(name: str | None = None, equipment: list[dict] | None = None,
              config: dict | None = None) -> dict:
    """Add a plant to the project and make it active."""
    park_active()
    if config:
        name = _unique_name(name or config.get("plant_name") or "Plant")
        config = {**copy.deepcopy(config), "plant_name": name}
    else:
        name = _unique_name(name or f"Plant {len(state.plants) + 1}")
        config = default_plant_config(name)
    slot = _new_slot(name, copy.deepcopy(equipment or []), config)
    state.plants.append(slot)
    _load_slot(slot)
    return slot


def active_is_blank() -> bool:
    """True for an untouched new plant (no equipment, default config)."""
    return (not state.equipment_list and
            state.plant_config in ({}, default_plant_config(active_slot()["name"])))


def replace_active(equipment: list[dict], config: dict):
    """Overwrite the active plant in place (used for a blank plant)."""
    slot = active_slot()
    slot["equipment"] = copy.deepcopy(equipment)
    slot["plant"] = copy.deepcopy(config)
    slot["name"] = config.get("plant_name") or slot["name"]
    slot["analysis"] = {"tornado_args": None, "sensitivity_args": {}}
    _load_slot(slot)


def duplicate(plant_id: str) -> dict | None:
    park_active()
    src = _slot(plant_id)
    if src is None:
        return None
    return add_plant(f"{src['name']} (copy)", src["equipment"], src["plant"])


def rename(plant_id: str, name: str) -> bool:
    slot = _slot(plant_id)
    if slot is None:
        return False
    slot["name"] = name
    if plant_id == state.active_plant_id:
        # replace (not mutate) the dict: the calc snapshot compares by value
        state.plant_config = {**state.plant_config, "plant_name": name}
    else:
        slot["plant"] = {**slot["plant"], "plant_name": name}
    return True


def delete(plant_id: str) -> bool:
    slot = _slot(plant_id)
    if slot is None or len(state.plants) <= 1:
        return False
    idx = state.plants.index(slot)
    if plant_id == state.active_plant_id:
        neighbour = state.plants[idx + 1] if idx + 1 < len(state.plants) else state.plants[idx - 1]
        _load_slot(neighbour)  # the deleted plant's live state is discarded
    state.plants.remove(slot)
    return True


def overview() -> dict:
    """Project metadata + one summary row per plant, for the Project tab."""
    ensure_project()
    park_active()
    rows = []
    for s in state.plants:
        metrics = (s["results"] or {}).get("metrics")
        rows.append({
            "id": s["id"],
            "name": s["name"],
            "currency": s["plant"].get("currency", ""),
            "equipment_count": len(s["equipment"]),
            "metrics": metrics,
        })
    return {
        "meta": state.project_meta,
        "active_plant_id": state.active_plant_id,
        "plants": rows,
    }


def snapshot(plant_id: str) -> dict | None:
    """Name, currency, results and source of one plant (for Compare)."""
    park_active()
    s = _slot(plant_id)
    if s is None or not s["results"]:
        return None
    return {
        "name": s["name"],
        "currency": s["plant"].get("currency", ""),
        "results": s["results"],
        "source": {"name": s["name"], "equipment": s["equipment"], "plant": s["plant"]},
    }


# ── Save / load ────────────────────────────────────────────────────


def to_file_payload() -> dict:
    """The project part of the .openpytea envelope (format version 2).

    The active plant is also mirrored at the top level ("equipment",
    "plant", "results") — the version-1 shape — so older app builds can
    still open the file (they see the active plant only).
    """
    ensure_project()
    park_active()
    plants = [
        {"name": s["name"], "equipment": s["equipment"], "plant": s["plant"],
         "results": s["results"]}
        for s in state.plants
    ]
    active = active_slot()
    return {
        "project": state.project_meta,
        "active_plant": state.plants.index(active),
        "plants": plants,
        "equipment": active["equipment"],
        "plant": active["plant"],
        "results": active["results"],
    }


def restore(data: dict) -> int:
    """Rebuild the project from a saved file (v2, or a v1 single plant).

    Returns the active plant's equipment count. Saved results are ignored;
    every plant is recalculated from its inputs.
    """
    if isinstance(data.get("plants"), list) and data["plants"]:
        entries = data["plants"]
        active_idx = data.get("active_plant", 0)
    else:
        entries = [{"equipment": data.get("equipment", []), "plant": data.get("plant", {})}]
        active_idx = 0
    if not isinstance(active_idx, int) or not 0 <= active_idx < len(entries):
        active_idx = 0

    from app.routers.plant import _extract_results
    slots = []
    for i, e in enumerate(entries, start=1):
        config = e.get("plant") or {}
        equipment = e.get("equipment") or []
        name = e.get("name") or config.get("plant_name") or f"Plant {i}"
        slot = _new_slot(name, equipment, config)
        plant = _calculate(build_equipment_list(equipment), config)
        slot["results"] = _extract_results(plant) if plant else None
        slots.append(slot)

    meta = {**default_meta(), "user": "", **(data.get("project") or {})}
    if not data.get("project") and data.get("saved_at"):
        meta["created"] = str(data["saved_at"])[:10]
    state.project_meta = meta
    state.plants = slots
    _load_slot(slots[active_idx])
    return len(state.equipment_list)


def reset():
    """A new, empty project with one default plant."""
    state.plants = []
    state.equipment_list = []
    state.plant_config = {}
    state.plant = None
    state.results = {}
    state.reset_analysis_runs()
    ensure_project()
