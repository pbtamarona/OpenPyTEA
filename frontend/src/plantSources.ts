import { useState } from "react";
import { getPlantSnapshot } from "./api/client";
import type { ComparedPlant, PlantInput, ProjectOverview } from "./types";

/** Request fields that pick the plants of an analysis run. */
export interface PlantsBody {
  include_active: boolean;
  extra_plants: PlantInput[];
}

/** A run's plant fields plus the plant id of each result, in the order the
    backend returns them (the open plant first, then the extra plants). */
export interface PlantsRequest {
  body: PlantsBody;
  ids: string[];
}

/** Stand-in id for the open plant when there is no project overview yet. */
const OPEN = "__open__";

/** Which project plants an analysis run covers, and which results are
    shown. The open plant starts ticked; every plant, the open one
    included, can be ticked or unticked. With a single plant there are no
    checkboxes and everything is shown. */
export function usePlantSelection(project: ProjectOverview | null) {
  const [includeOpen, setIncludeOpen] = useState(true);
  const [others, setOthers] = useState<Set<string>>(new Set());
  const activeId = project?.active_plant_id ?? OPEN;
  const multi = !!project && project.plants.length > 1;

  const isTicked = (id: string) =>
    !multi || (id === activeId ? includeOpen : others.has(id));

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

  /** Ticked plants that can run (others need results), open plant first. */
  const tickedIds = (): string[] => {
    if (!multi) return [activeId];
    const rest = project!.plants
      .filter((p) => p.id !== activeId && p.metrics != null && others.has(p.id))
      .map((p) => p.id);
    return includeOpen ? [activeId, ...rest] : rest;
  };

  /** Request fields for the given plants. The inputs of plants other than
      the open one are fetched now, so they are current. */
  const requestFor = async (wanted: string[]): Promise<PlantsRequest> => {
    // Leave out plants deleted (or no longer calculated) since a run
    const ids = wanted.filter((id) =>
      id === activeId || !project || project.plants.some((p) => p.id === id && p.metrics != null));
    if (ids.length === 0) throw new Error("Tick at least one plant");
    const include_active = ids.includes(activeId);
    const extras = ids.filter((id) => id !== activeId);
    const snaps = await Promise.all(extras.map((id) => getPlantSnapshot(id)));
    return {
      body: { include_active, extra_plants: snaps.map((s) => s.source) },
      ids: include_active ? [activeId, ...extras] : extras,
    };
  };

  /** Request fields for the ticked plants. */
  const request = () => requestFor(tickedIds());

  /** Colour slot of a plant: its place in the project, so a plant keeps its
      colour whichever others are shown. */
  const colorIndex = (id: string, fallback: number) => {
    const i = project?.plants.findIndex((p) => p.id === id) ?? -1;
    return i >= 0 ? i : fallback;
  };

  const nameOf = (id: string) => project?.plants.find((p) => p.id === id)?.name ?? "";

  return { isTicked, toggle, tickedIds, requestFor, request, colorIndex, nameOf };
}

/** Every calculated plant of the project, for the Compare tab. */
export async function projectComparison(project: ProjectOverview): Promise<ComparedPlant[]> {
  const rows = project.plants.filter((p) => p.metrics != null);
  const snaps = await Promise.all(rows.map((p) => getPlantSnapshot(p.id)));
  return snaps.map((s, i) => ({ id: `project:${rows[i].id}`, ...s }));
}
