import type { ProjectOverview } from "../types";

interface Props {
  project: ProjectOverview | null;
  /** Called with the chosen plant's id (not called for the open plant). */
  onSwitch: (id: string) => void;
  /** Shows "+ New Plant" when given. */
  onAdd?: () => void;
  disabled?: boolean;
  /** Label for the open plant, e.g. its name as currently typed. */
  activeLabel?: string;
  hint?: string;
}

/** Dropdown of the project's plants, by Plant Name — the plant the page
    (Plant Config, Equipment, Results) works on. */
export default function PlantPicker({ project, onSwitch, onAdd, disabled, activeLabel, hint }: Props) {
  if (!project) return null;
  const n = project.plants.length;
  return (
    <div className="card plant-picker">
      <label htmlFor="plant-picker-select">Plant</label>
      <select
        id="plant-picker-select"
        value={project.active_plant_id}
        disabled={disabled}
        onChange={(e) => {
          // read now: a caller that awaits a save first would otherwise see
          // React reset the controlled select to the open plant
          const id = e.target.value;
          if (id !== project.active_plant_id) onSwitch(id);
        }}
      >
        {project.plants.map((p) => (
          <option key={p.id} value={p.id}>
            {p.id === project.active_plant_id ? activeLabel || p.name : p.name}
          </option>
        ))}
      </select>
      {onAdd && (
        <button className="btn-secondary" disabled={disabled} onClick={onAdd}>
          + New Plant
        </button>
      )}
      <span className="plant-picker-hint">
        {n} plant{n !== 1 ? "s" : ""} in this project{hint ? ` · ${hint}` : ""}
      </span>
    </div>
  );
}
