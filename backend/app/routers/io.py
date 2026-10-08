"""Save/load project + example presets endpoints."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File

from openpytea.analysis import (
    direct_costs_data, fixed_capital_data, fixed_opex_data,
    variable_opex_data, levelized_cost_data, cash_flow_data,
    sensitivity_data, tornado_data,
)
from openpytea.io import (
    export_equipment_results, export_plant_results,
    _write_json, _safe_filename,
)

from app import state
from app.plant_factory import build_equipment_list, require_active_plant
from app.schemas import (
    LoadResponse, LoadExampleResponse, ExamplePreset, ExportJsonResponse,
)
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
    state.reset_analysis_runs()
    return {"ok": True}


@router.post("/save")
def save_project():
    """Return the full project state as a versioned JSON envelope."""
    # Every equipment object is built via plant_factory.equipment_from_entry,
    # which keeps the sanitized original input on _input_spec. Saving that
    # spec (never derived values like the resolved num_units or an
    # inflation-adjusted composite quote) makes save → load a clean rebuild
    # from user inputs.
    equipment_data = []
    for eq in state.equipment_list:
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
        equipment_data.append(spec)

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


@router.post("/export-json", response_model=ExportJsonResponse)
def export_json():
    """Return the three result files run_tea writes for this plant.

    <plant>_equipment_results.json, <plant>_plant_results.json and
    <plant>_analysis_results.json are produced by the library's own
    exporters, so they match a Python run_tea() byte-for-byte in shape.
    The analysis file always holds the deterministic breakdowns; tornado,
    sensitivity and Monte Carlo are included when they were run in the GUI
    (MC only if the configuration hasn't changed since that run).
    """
    plant = require_active_plant()

    results = {
        "direct_costs": direct_costs_data(plant),
        "fixed_capital": fixed_capital_data(plant),
        "fixed_opex": fixed_opex_data(plant),
        "variable_opex": variable_opex_data(plant),
        "levelized_cost": levelized_cost_data(plant),
        "cash_flow": cash_flow_data(plant),
    }
    if state.tornado_args:
        try:
            results["tornado"] = tornado_data(plant, **state.tornado_args)
        except (ValueError, KeyError):
            pass  # e.g. a varied item was since removed from the config
    if state.mc_raw and state.mc_snapshot == state.calc_snapshot:
        results["monte_carlo"] = state.mc_raw[0]

    sensitivity = {}
    params = [p for p, _ in state.sensitivity_args]
    for (param, metric), args in state.sensitivity_args.items():
        # Case name is the parameter, plus the metric when the same
        # parameter was run against several metrics
        name = param if params.count(param) == 1 else f"{param} ({metric})"
        try:
            sensitivity[name] = sensitivity_data(plant, **args)
        except (ValueError, KeyError):
            pass
    if sensitivity:
        results["sensitivity"] = sensitivity

    fname = _safe_filename(plant.name)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        export_equipment_results(
            state.equipment_list, out / f"{fname}_equipment_results.json")
        export_plant_results(plant, out / f"{fname}_plant_results.json")
        _write_json(out / f"{fname}_analysis_results.json",
                    {"results": results})
        files = [
            {"name": f.name, "content": f.read_text(encoding="utf-8")}
            for f in sorted(out.glob("*.json"))
        ]
    return {"files": files}


MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB


def _restore_project_state(data: dict) -> int:
    """Apply a project payload to the in-memory state. Returns equipment count.

    Accepts both the current versioned envelope and the legacy flat shape —
    both have `equipment` and `plant` at the top level.
    """
    state.equipment_list = build_equipment_list(data.get("equipment", []))
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}
    state.reset_analysis_runs()
    return len(state.equipment_list)


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
    state.equipment_list = build_equipment_list(data.get("equipment", []))

    # Restore plant config
    state.plant_config = data.get("plant", {})
    state.plant = None
    state.results = {}
    state.reset_analysis_runs()

    return {"ok": True, "title": data.get("title"), "equipment_count": len(state.equipment_list)}
