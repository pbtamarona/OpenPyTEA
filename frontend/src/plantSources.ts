import { getPlantSnapshot } from "./api/client";
import type { PlantInput, ProjectOverview } from "./types";

/** Inputs of the ticked plants (other than the open one), in project order,
    for the extra_plants field of the analysis requests. */
export async function selectedSources(
  project: ProjectOverview | null,
  selected: Set<string>,
): Promise<PlantInput[]> {
  if (!project) return [];
  const ids = project.plants
    .filter((p) => p.id !== project.active_plant_id && p.metrics != null && selected.has(p.id))
    .map((p) => p.id);
  const snaps = await Promise.all(ids.map((id) => getPlantSnapshot(id)));
  return snaps.map((s) => s.source).filter((s): s is PlantInput => s != null);
}
