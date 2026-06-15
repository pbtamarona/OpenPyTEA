"""Save/load project + example presets endpoints."""

import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from openpytea.equipment import Equipment

from app import state
from app.schemas import LoadResponse, LoadExampleResponse, ExamplePreset
from app.util import to_jsonable

router = APIRouter()

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

# Saved-project format identifier and current schema version. Bump `version`
# whenever the on-disk shape changes in a way that the loader needs to detect
# and migrate.
PROJECT_FORMAT = "openpytea-project"
PROJECT_VERSION = 1
APP_VERSION = "0.1.0"


@router.post("/new")
def new_project():
    """Clear the in-memory session — fresh start."""
    state.equipment_list = []
    state.plant_config = {}
    state.plant = None
    state.results = {}
    return {"ok": True}


@router.post("/save")
def save_project():
    """Return the full project state as a versioned JSON envelope."""
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
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "equipment": equipment_data,
        "plant": state.plant_config,
        "results": to_jsonable(state.results) if state.results else None,
    }
    return project


MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_EQUIPMENT = 500


def _restore_project_state(data: dict) -> int:
    """Apply a project payload to the in-memory state. Returns equipment count.

    Accepts both the current versioned envelope and the legacy flat shape —
    both have `equipment` and `plant` at the top level.
    """
    equipment_data = data.get("equipment", [])
    if len(equipment_data) > MAX_EQUIPMENT:
        raise HTTPException(
            status_code=400,
            detail=f"Too many equipment items (max {MAX_EQUIPMENT})",
        )

    rebuilt: list[Equipment] = []
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
            rebuilt.append(eq)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid equipment entry '{entry.get('name', '?')}'",
            )

    state.equipment_list = rebuilt
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}
    return len(rebuilt)


@router.post("/load", response_model=LoadResponse)
async def load_project(file: UploadFile = File(...)):
    """Load a project from a multipart-uploaded JSON file (browser path)."""
    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 5 MB)")
        data = json.loads(content)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    n = _restore_project_state(data)
    return {"ok": True, "equipment_count": n}


@router.post("/load_json", response_model=LoadResponse)
async def load_project_json(data: dict):
    """Load a project from a JSON body directly (Tauri path).

    Avoids the Blob → File → FormData → multipart round-trip that's
    fragile inside WebKit-based webviews.
    """
    n = _restore_project_state(data)
    return {"ok": True, "equipment_count": n}


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
