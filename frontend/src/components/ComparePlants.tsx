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
        {[...project.plants]
          .sort((a, b) => Number(b.id === project.active_plant_id) - Number(a.id === project.active_plant_id))
          .map((p) => {
            const open = p.id === project.active_plant_id;
            const ready = open || p.metrics != null;
            return (
              <label
                key={p.id}
                className={ready ? "" : "is-disabled"}
                title={open ? "The open plant (switch it on the Results tab)" : ready ? undefined : "Calculate this plant on the Results tab first"}
              >
                <input
                  type="checkbox"
                  checked={ready && isTicked(p.id)}
                  disabled={!ready}
                  onChange={() => onToggle(p.id)}
                />
                {p.name}
                {open && <span className="compare-plants-tag">open</span>}
                {!ready && <span className="compare-plants-tag">not calculated</span>}
              </label>
            );
          })}
      </div>
      {note && <p className="compare-plants-note">{note}</p>}
    </div>
  );
}
