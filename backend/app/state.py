"""In-memory session state for a single OpenPyTEA project."""

from openpytea.equipment import Equipment
from openpytea.plant import Plant

equipment_list: list[Equipment] = []
plant: Plant | None = None
results: dict = {}
mc_results: dict | None = None
# Raw monte_carlo() results (full sample arrays), one per plant of the
# last run — the /api/plots endpoints re-render them with the library's
# own matplotlib functions.
mc_raw: list[dict] | None = None
plant_config: dict = {}
# (config deep-copy, equipment ids) the active plant was built from —
# plant_factory.require_active_plant() rebuilds the plant when it drifts
calc_snapshot: tuple | None = None
# Analyses run on the active plant this session, replayed by
# /api/project/export-json: last tornado args, sensitivity args per
# (parameter, metric), and the calc_snapshot the cached MC run used.
tornado_args: dict | None = None
sensitivity_args: dict[tuple[str, str], dict] = {}
mc_snapshot: tuple | None = None


def reset_analysis_runs():
    """Forget recorded analysis runs (new/loaded project)."""
    global tornado_args, sensitivity_args, mc_raw, mc_results, mc_snapshot
    tornado_args = None
    sensitivity_args = {}
    mc_raw = None
    mc_results = None
    mc_snapshot = None
