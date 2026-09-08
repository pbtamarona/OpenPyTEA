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
