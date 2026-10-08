"""Project metadata + plant-list endpoints (the Project tab)."""

from fastapi import APIRouter, HTTPException

from app import project_store, state
from app.schemas import ProjectMeta, ProjectOverview, PlantNameIn, PlantSnapshot

router = APIRouter()

MAX_PLANTS = 50


def _overview() -> dict:
    return {**project_store.overview(), "results": state.results or None}


def _not_found():
    raise HTTPException(status_code=404, detail="Plant not found")


@router.get("", response_model=ProjectOverview)
def get_project():
    return _overview()


@router.put("/meta", response_model=ProjectOverview)
def set_meta(data: ProjectMeta):
    project_store.ensure_project()
    state.project_meta = data.model_dump()
    return _overview()


@router.post("/plants", response_model=ProjectOverview)
def add_plant(data: PlantNameIn):
    """Add a blank plant (default config, no equipment) and open it."""
    project_store.ensure_project()
    if len(state.plants) >= MAX_PLANTS:
        raise HTTPException(status_code=400, detail=f"Too many plants (max {MAX_PLANTS})")
    project_store.add_plant(data.name)
    return _overview()


@router.post("/plants/{plant_id}/activate", response_model=ProjectOverview)
def activate_plant(plant_id: str):
    project_store.ensure_project()
    if not project_store.activate(plant_id):
        _not_found()
    return _overview()


@router.post("/plants/{plant_id}/duplicate", response_model=ProjectOverview)
def duplicate_plant(plant_id: str):
    project_store.ensure_project()
    if len(state.plants) >= MAX_PLANTS:
        raise HTTPException(status_code=400, detail=f"Too many plants (max {MAX_PLANTS})")
    if project_store.duplicate(plant_id) is None:
        _not_found()
    return _overview()


@router.patch("/plants/{plant_id}", response_model=ProjectOverview)
def rename_plant(plant_id: str, data: PlantNameIn):
    project_store.ensure_project()
    if not data.name or not data.name.strip():
        raise HTTPException(status_code=400, detail="Plant name can't be empty")
    if not project_store.rename(plant_id, data.name.strip()):
        _not_found()
    return _overview()


@router.delete("/plants/{plant_id}", response_model=ProjectOverview)
def delete_plant(plant_id: str):
    project_store.ensure_project()
    if len(state.plants) <= 1:
        raise HTTPException(status_code=400, detail="A project needs at least one plant")
    if not project_store.delete(plant_id):
        _not_found()
    return _overview()


@router.get("/plants/{plant_id}/snapshot", response_model=PlantSnapshot)
def plant_snapshot(plant_id: str):
    """Results + inputs of one plant, for adding it to the Compare tab."""
    project_store.ensure_project()
    snap = project_store.snapshot(plant_id)
    if snap is None:
        raise HTTPException(
            status_code=400,
            detail="This plant can't be calculated yet — add equipment and check its configuration",
        )
    return snap
