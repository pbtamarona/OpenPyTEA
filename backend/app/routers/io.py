"""Save/load project + example presets endpoints."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from openpytea.equipment import Equipment

from app import state
from app.schemas import LoadResponse, LoadExampleResponse, ExamplePreset
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


MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_EQUIPMENT = 500


@router.post("/load", response_model=LoadResponse)
async def load_project(file: UploadFile = File(...)):
    """Load a project from an uploaded JSON file."""
    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 5 MB)")
        data = json.loads(content)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    # Restore equipment
    equipment_data = data.get("equipment", [])
    if len(equipment_data) > MAX_EQUIPMENT:
        raise HTTPException(status_code=400, detail=f"Too many equipment items (max {MAX_EQUIPMENT})")

    state.equipment_list = []
    for entry in equipment_data:
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
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid equipment entry '{entry.get('name', '?')}'",
            )

    # Restore plant config
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}

    return {"ok": True, "equipment_count": len(state.equipment_list)}


@router.get("/examples", response_model=list[ExamplePreset])
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


@router.post("/examples/{example_id}", response_model=LoadExampleResponse)
def load_example(example_id: str):
    """Load an example preset into the session."""
    preset_file = (PRESETS_DIR / f"{example_id}.json").resolve()
    if not str(preset_file).startswith(str(PRESETS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid example ID")
    if not preset_file.exists():
        raise HTTPException(status_code=404, detail="Example not found")

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
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid equipment entry '{entry.get('name', '?')}'",
            )

    # Restore plant config
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}

    return {"ok": True, "title": data.get("title"), "equipment_count": len(state.equipment_list)}
