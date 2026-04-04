"""Save/load project + example presets endpoints."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from openpytea.equipment import Equipment

from app import state
from app.util import to_jsonable

router = APIRouter()

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


@router.post("/save")
def save_project():
    """Return the full project state as JSON."""
    equipment_data = []
    for eq in state.equipment_list:
        equipment_data.append({
            "name": eq.name,
            "param": eq.param,
            "process_type": eq.process_type,
            "category": eq.category,
            "type": eq.type,
            "material": eq.material,
            "num_units": eq.num_units,
            "purchased_cost": float(eq.purchased_cost) if eq.param is None else None,
            "cost_year": eq.cost_year,
            "target_year": eq.target_year,
        })

    project = {
        "equipment": equipment_data,
        "plant": state.plant_config,
        "results": to_jsonable(state.results) if state.results else None,
    }
    return project


@router.post("/load")
async def load_project(file: UploadFile = File(...)):
    """Load a project from an uploaded JSON file."""
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Restore equipment
    state.equipment_list = []
    for entry in data.get("equipment", []):
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
            state.equipment_list.append(eq)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error loading equipment '{entry.get('name', '?')}': {e}",
            )

    # Restore plant config
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}

    return {"ok": True, "equipment_count": len(state.equipment_list)}


@router.get("/examples")
def list_examples():
    """List available example presets."""
    examples = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            examples.append({
                "id": data.get("id", f.stem),
                "title": data.get("title", f.stem),
                "description": data.get("description", ""),
            })
        except Exception:
            continue
    return examples


@router.post("/examples/{example_id}")
def load_example(example_id: str):
    """Load an example preset into the session."""
    preset_file = PRESETS_DIR / f"{example_id}.json"
    if not preset_file.exists():
        raise HTTPException(status_code=404, detail=f"Example '{example_id}' not found")

    data = json.loads(preset_file.read_text())

    # Restore equipment
    state.equipment_list = []
    for entry in data.get("equipment", []):
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
            state.equipment_list.append(eq)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error loading equipment '{entry.get('name', '?')}': {e}",
            )

    # Restore plant config
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}

    return {"ok": True, "title": data.get("title"), "equipment_count": len(state.equipment_list)}
