"""Save/load project + example presets endpoints."""

import base64
import json
import mimetypes
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
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

from app import project_store, state
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
PROJECT_VERSION = 2  # 2: multi-plant projects (plants, project, active_plant)
APP_VERSION = "0.1.0"


@router.post("/new")
def new_project():
    """Clear the in-memory session — a new project with one default plant."""
    project_store.reset()
    return {"ok": True}


@router.post("/save")
def save_project():
    """Return the full project (all plants) as a versioned JSON envelope."""
    return to_jsonable({
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        **project_store.to_file_payload(),
    })


@router.post("/export-json", response_model=ExportJsonResponse)
def export_json(only: Literal["equipment", "plant", "analysis"] | None = None):
    """Return the three result files run_tea writes for this plant, or
    just the `only` one.

    <plant>_equipment_results.json, <plant>_plant_results.json and
    <plant>_analysis_results.json are produced by the library's own
    exporters, so they match a Python run_tea() byte-for-byte in shape.
    The analysis file always holds the deterministic breakdowns; tornado,
    sensitivity and Monte Carlo are included when they were run in the GUI
    (MC only if the configuration hasn't changed since that run).
    """
    if only == "equipment":
        # Equipment costs don't need a calculated plant
        if not state.equipment_list:
            raise HTTPException(status_code=400, detail="No equipment defined")
        fname = _safe_filename(state.plant_config.get("plant_name") or "plant")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{fname}_equipment_results.json"
            export_equipment_results(state.equipment_list, path)
            return {"files": [{"name": path.name, "content": path.read_text(encoding="utf-8")}]}

    plant = require_active_plant()
    if only == "plant":
        fname = _safe_filename(plant.name)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{fname}_plant_results.json"
            export_plant_results(plant, path)
            return {"files": [{"name": path.name, "content": path.read_text(encoding="utf-8")}]}

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
    if state.mc_active_raw and state.mc_snapshot is not None and state.mc_snapshot == state.calc_snapshot:
        results["monte_carlo"] = state.mc_active_raw

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
        if only is None:
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
MAX_PLANTS = 50


# Analysis sections of a project file that list their plants
ANALYSIS_PLANT_SECTIONS = ("sensitivity", "tornado", "monte_carlo")


def _adopt_compared_plants(data: dict) -> list[str]:
    """Turn an older file's `compared_plants` (plants imported on the
    Compare tab, which no longer exists) into plants of the project.

    Only compared plants saved with their inputs can be rebuilt; the
    project's own plants, which older files listed there too, are skipped.
    Returns notes for the user.
    """
    compared = data.get("compared_plants")
    if not isinstance(compared, list) or not compared:
        return []
    plants = data.get("plants")
    if not isinstance(plants, list) or not plants:
        plants = [{"equipment": data.get("equipment") or [], "plant": data.get("plant") or {}}]
    own = {p.get("name") or (p.get("plant") or {}).get("plant_name") for p in plants}
    adopted, skipped = [], []
    for c in compared:
        name = c.get("name") if isinstance(c, dict) else None
        if name in own:
            continue
        source = (c.get("source") or {}) if isinstance(c, dict) else {}
        if source.get("equipment") and source.get("plant"):
            plants.append({"name": name or source.get("name"),
                           "equipment": source["equipment"], "plant": source["plant"]})
            adopted.append(name or "a plant")
        else:
            skipped.append(name or "a plant")
    data["plants"] = plants
    notes = []
    if adopted:
        notes.append(f"Compared plants added to the project as plants: {', '.join(adopted)}.")
    if skipped:
        notes.append(f"Compared plants saved without their inputs were left out: {', '.join(skipped)}.")
    return notes


def _autorun(data: dict, notes: list[str]) -> dict | None:
    """The file's project-wide "analysis" block, with each section's
    "plants" (plant names; all plants when left out) as the loaded plants'
    ids, for the GUI to run right after loading."""
    block = data.get("analysis")
    if not isinstance(block, dict) or not block:
        return None
    ids_of: dict[str, str] = {}
    for s in state.plants:
        for name in (s["name"], s["plant"].get("plant_name")):
            if name:
                ids_of.setdefault(name, s["id"])
    analysis, run = {}, {}
    for key, cfg in block.items():
        if key in ANALYSIS_PLANT_SECTIONS and isinstance(cfg, dict):
            listed = cfg.get("plants")
            if not isinstance(listed, list):
                ids = [s["id"] for s in state.plants]
            else:
                ids = [ids_of[n] for n in listed if n in ids_of]
                missing = [str(n) for n in listed if n not in ids_of]
                if missing:
                    label = {"sensitivity": "Sensitivity", "tornado": "Tornado",
                             "monte_carlo": "Monte Carlo"}[key]
                    notes.append(f"The {label} analysis lists plants this project "
                                 f"doesn't have: {', '.join(missing)}.")
            run[f"{key}_plants"] = ids
            cfg = {k: v for k, v in cfg.items() if k != "plants"}
        analysis[key] = cfg
    return {"analysis": analysis, **run}


def _restore_project_state(data: dict) -> dict:
    """Apply a project payload to the in-memory state; returns the load
    response (equipment count, open plant, the file's analyses to re-run).

    Accepts the multi-plant envelope (v2) as well as v1 / legacy flat
    files, which have a single plant's `equipment` and `plant` at the top
    level and open as a one-plant project.
    """
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Not an OpenPyTEA project file")
    notes = _adopt_compared_plants(data)
    if len(data.get("plants") or []) > MAX_PLANTS:
        raise HTTPException(status_code=400, detail=f"Too many plants (max {MAX_PLANTS})")
    n = project_store.restore(data)
    return {"ok": True, "equipment_count": n, "active_plant_id": state.active_plant_id,
            "autorun": _autorun(data, notes), "notes": notes}


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

    return _restore_project_state(data)


@router.post("/load_json", response_model=LoadResponse)
async def load_project_json(data: dict):
    """Load a project from a JSON body directly (Tauri path).

    Avoids the Blob → File → FormData → multipart round-trip that's
    fragile inside WebKit-based webviews.
    """
    return _restore_project_state(data)


@router.get("/examples", response_model=list[ExamplePreset])
def list_examples():
    """List available example presets."""
    examples = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
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

    data = json.loads(preset_file.read_text(encoding="utf-8"))
    for entry in data.get("plants", []):
        build_equipment_list(entry.get("equipment", []))  # validate before touching the project
        # a preset's PFD is an image file next to it ("img/CS_1-Fig_1.png");
        # the project keeps images as data URLs
        pfd = entry.get("pfd")
        if isinstance(pfd, str) and not project_store.is_image_data_url(pfd):
            image = (PRESETS_DIR / pfd).resolve()
            mime = mimetypes.guess_type(image.name)[0] or ""
            entry["pfd"] = (
                f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}"
                if image.is_relative_to(PRESETS_DIR.resolve()) and image.is_file()
                and mime.startswith("image/") else None
            )

    # An example is a whole project (like File ▸ Open): it replaces the
    # current one, and its analyses are run right after loading
    return {**_restore_project_state(data), "title": data.get("title")}
