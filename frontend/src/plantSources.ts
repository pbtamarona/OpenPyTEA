import { useState } from "react";
import { getPlantSnapshot } from "./api/client";
import type { ComparedPlant, PlantInput, ProjectOverview } from "./types";

/** Which project plants an analysis run covers. The open plant starts
    ticked; every plant, the open one included, can be ticked or unticked. */
export function usePlantSelection(project: ProjectOverview | null) {
  const [includeOpen, setIncludeOpen] = useState(true);
  const [others, setOthers] = useState<Set<string>>(new Set());
  const activeId = project?.active_plant_id;
  const multi = !!project && project.plants.length > 1;

  const isTicked = (id: string) => (id === activeId ? includeOpen : others.has(id));

  const toggle = (id: string) => {
    if (id === activeId) {
      setIncludeOpen((v) => !v);
      return;
    }
    setOthers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  /** Ticked plants other than the open one that have results. */
  const otherIds = () =>
    multi
      ? project!.plants
          .filter((p) => p.id !== activeId && p.metrics != null && others.has(p.id))
          .map((p) => p.id)
      : [];

  /** Number of plants the run covers. */
  const count = () => (multi ? Number(includeOpen) + otherIds().length : 1);

  /** The include_active / extra_plants fields of an analysis request.
      The inputs of the other plants are fetched when the run starts, so
      they are current. */
  const request = async (): Promise<{ include_active: boolean; extra_plants: PlantInput[] }> => {
    const ids = otherIds();
    const include_active = !multi || includeOpen;
    if (!include_active && ids.length === 0) throw new Error("Tick at least one plant");
    const snaps = await Promise.all(ids.map((id) => getPlantSnapshot(id)));
    return { include_active, extra_plants: snaps.map((s) => s.source) };
  };

  return { isTicked, toggle, count, request };
}

/** Every calculated plant of the project, for the Compare tab. */
export async function projectComparison(project: ProjectOverview): Promise<ComparedPlant[]> {
  const rows = project.plants.filter((p) => p.metrics != null);
  const snaps = await Promise.all(rows.map((p) => getPlantSnapshot(p.id)));
  return snaps.map((s, i) => ({ id: `project:${rows[i].id}`, ...s }));
}
