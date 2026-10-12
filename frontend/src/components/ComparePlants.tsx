import type { ProjectOverview } from "../types";

interface Props {
  project: ProjectOverview | null;
  isTicked: (id: string) => boolean;
  onToggle: (id: string) => void;
  note?: string;
}

/** Checkbox per project plant, choosing the plants a run covers (the open
    plant starts ticked). Plants that have never been calculated have no
    results yet and can't be ticked. Shown when the project has more than
    one plant, in place of the header plant switcher. */
export default function ComparePlants({ project, isTicked, onToggle, note }: Props) {
  if (!project || project.plants.length < 2) return null;
  return (
    <div className="compare-plants">
      <span className="compare-plants-label">Plants</span>
      <div className="compare-plants-list">
        {project.plants.map((p) => {
          const ready = p.id === project.active_plant_id || p.metrics != null;
          return (
            <label
              key={p.id}
              className={ready ? "" : "is-disabled"}
              title={ready ? undefined : "Calculate this plant on the Results tab first"}
            >
              <input
                type="checkbox"
                checked={ready && isTicked(p.id)}
                disabled={!ready}
                onChange={() => onToggle(p.id)}
              />
              {p.name}
              {!ready && <span className="compare-plants-tag">not calculated</span>}
            </label>
          );
        })}
      </div>
      {note && <p className="compare-plants-note">{note}</p>}
    </div>
  );
}
