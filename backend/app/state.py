"""In-memory session state for a single OpenPyTEA project."""

from openpytea.equipment import Equipment
from openpytea.plant import Plant

equipment_list: list[Equipment] = []
plant: Plant | None = None
results: dict = {}
mc_results: dict | None = None
# Raw monte_carlo() results (full sample arrays), one per plant of the
# Monte Carlo tab's last run (project-wide: kept when the open plant
# changes) — the /api/plots endpoints re-render them with the library's
# own matplotlib functions.
mc_raw: list[dict] | None = None
# The active plant's own raw run (from the last run that included it),
# replayed by /api/project/export-json; swapped with the plant's slot
mc_active_raw: dict | None = None
plant_config: dict = {}
# (config deep-copy, equipment ids) the active plant was built from —
# plant_factory.require_active_plant() rebuilds the plant when it drifts
calc_snapshot: tuple | None = None
# Multi-plant project wrapper (see project_store): metadata, one slot per
# plant, and which slot the fields above currently hold.
project_meta: dict = {}
plants: list[dict] = []
active_plant_id: str | None = None
# Analyses run on the active plant this session, replayed by
# /api/project/export-json: last tornado args, sensitivity args per
# (parameter, metric), and the calc_snapshot the cached MC run used.
tornado_args: dict | None = None
sensitivity_args: dict[tuple[str, str], dict] = {}
mc_snapshot: tuple | None = None


def reset_plant_runs():
    """Forget the active plant's recorded runs (another plant opens)."""
    global tornado_args, sensitivity_args, mc_active_raw, mc_results, mc_snapshot
    tornado_args = None
    sensitivity_args = {}
    mc_active_raw = None
    mc_results = None
    mc_snapshot = None


def reset_analysis_runs():
    """Forget all recorded analysis runs (new/loaded project)."""
    global mc_raw
    reset_plant_runs()
    mc_raw = None
