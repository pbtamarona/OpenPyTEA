import { useEffect, useState } from "react";
import {
  setProjectMeta, addPlant, activatePlant, duplicatePlant, renamePlant,
  deletePlant, getPlantSnapshot,
} from "../api/client";
import type { ProjectMeta, ProjectOverview, PlantSnapshot } from "../types";
import HelpTip from "../components/HelpTip";

interface Props {
  project: ProjectOverview | null;
  /** New overview after an action; `switched` = the open plant changed. */
  onProjectChange: (o: ProjectOverview, switched: boolean) => void;
  onOpenPlant: () => void;
  onAddToComparison: (snap: PlantSnapshot) => void;
  setError: (e: string | null) => void;
  markDirty: () => void;
}

const fmt = (n: number | null | undefined) =>
  n != null ? n.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "–";

const fmtCompact = (n: number | null | undefined) => {
  if (n == null) return "–";
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + "k";
  return n.toFixed(2);
};

const pct = (n: number | null | undefined) => (n != null ? (n * 100).toFixed(2) + "%" : "–");

/** Native confirm in Tauri (window.confirm is suppressed in its webview). */
async function askConfirm(message: string, title: string): Promise<boolean> {
  try {
    const { isTauri } = await import("@tauri-apps/api/core");
    if (isTauri()) {
      const { ask } = await import("@tauri-apps/plugin-dialog");
      return await ask(message, { title, okLabel: "Delete", cancelLabel: "Cancel", kind: "warning" });
    }
  } catch {
    // fall through to the browser dialog
  }
  return window.confirm(message);
}

export default function ProjectPage({
  project, onProjectChange, onOpenPlant, onAddToComparison, setError, markDirty,
}: Props) {
  const [meta, setMeta] = useState<ProjectMeta | null>(project?.meta ?? null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [busy, setBusy] = useState(false);

  // The form is local while editing — re-syncing on every overview would
  // clobber a field being typed in while another field's save returns. The
  // parent remounts this page (new key) when a different project loads.
  useEffect(() => {
    if (project && !meta) setMeta(project.meta);
  }, [project, meta]);

  if (!project || !meta) return <div className="card">Loading project…</div>;

  const run = async (action: () => Promise<ProjectOverview>, opts: { switched?: boolean; dirty?: boolean } = {}) => {
    setBusy(true);
    setError(null);
    try {
      const o = await action();
      onProjectChange(o, opts.switched ?? false);
      if (opts.dirty ?? true) markDirty();
      return o;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Project action failed");
      return null;
    } finally {
      setBusy(false);
    }
  };

  const saveMeta = () => {
    const p = project.meta;
    if (meta.name === p.name && meta.description === p.description && meta.user === p.user) return;
    run(() => setProjectMeta(meta));
  };

  const commitRename = async (id: string) => {
    if (renamingId !== id) return; // Enter already committed; this is the blur
    const name = renameValue.trim();
    setRenamingId(null);
    const current = project.plants.find((p) => p.id === id)?.name;
    if (!name || name === current) return;
    await run(() => renamePlant(id, name), { switched: id === project.active_plant_id });
  };

  const handleDelete = async (id: string, name: string) => {
    if (!(await askConfirm(`Delete plant "${name}" from the project? This can't be undone.`, "Delete Plant"))) return;
    await run(() => deletePlant(id), { switched: id === project.active_plant_id });
  };

  const handleOpen = async (id: string) => {
    const o = id === project.active_plant_id
      ? project
      : await run(() => activatePlant(id), { switched: true, dirty: false });
    if (o) onOpenPlant();
  };

  const addToCompare = async (ids: string[]) => {
    setError(null);
    const failed: string[] = [];
    for (const id of ids) {
      try {
        onAddToComparison(await getPlantSnapshot(id));
      } catch {
        failed.push(project.plants.find((p) => p.id === id)?.name ?? id);
      }
    }
    if (failed.length) setError(`Not added to Compare (can't be calculated yet): ${failed.join(", ")}`);
  };

  const calculable = project.plants.filter((p) => p.metrics);

  return (
    <div className="project-page">
      <div className="card">
        <h2>Project Details</h2>
        <div className="form-grid">
          <div className="form-group">
            <label>Project Name<HelpTip id="project.name" /></label>
            <input
              value={meta.name}
              placeholder="Untitled project"
              onChange={(e) => setMeta({ ...meta, name: e.target.value })}
              onBlur={saveMeta}
            />
          </div>
          <div className="form-group">
            <label>User Name<HelpTip id="project.user" /></label>
            <input
              value={meta.user}
              onChange={(e) => setMeta({ ...meta, user: e.target.value })}
              onBlur={saveMeta}
            />
          </div>
          <div className="form-group">
            <label>Date Created<HelpTip id="project.created" /></label>
            <input value={meta.created} readOnly disabled />
          </div>
        </div>
        <div className="form-group" style={{ marginTop: 12 }}>
          <label>Project Description<HelpTip id="project.description" /></label>
          <textarea
            rows={3}
            value={meta.description}
            placeholder="Scope, assumptions, data sources…"
            onChange={(e) => setMeta({ ...meta, description: e.target.value })}
            onBlur={saveMeta}
          />
        </div>
      </div>

      <div className="card">
        <div className="project-plants-header">
          <h2>Plants<HelpTip id="project.plants" /></h2>
          <div className="project-plants-actions">
            <button
              className="btn-secondary"
              disabled={busy || calculable.length === 0}
              onClick={() => addToCompare(calculable.map((p) => p.id))}
              title="Add every calculated plant to the Compare tab"
            >
              Add all to Compare
            </button>
            <button className="btn-primary" disabled={busy} onClick={() => run(() => addPlant(), { switched: true })}>
              + Add Plant
            </button>
          </div>
        </div>
        <p className="project-hint">
          Plant Config, Equipment, Results, Analysis and Monte Carlo work on the open plant (●).
          Examples are added to the project as new plants.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th style={{ width: 24 }}></th>
                <th>Plant</th><th>Currency</th><th>Equipment</th>
                <th>LCOP</th><th>NPV</th><th>IRR</th><th>Payback (yr)</th><th></th>
              </tr>
            </thead>
            <tbody>
              {project.plants.map((p) => {
                const active = p.id === project.active_plant_id;
                return (
                  <tr key={p.id} className={active ? "project-row-active" : undefined}>
                    <td className="project-dot" title={active ? "Open plant" : undefined}>{active ? "●" : "○"}</td>
                    <td>
                      {renamingId === p.id ? (
                        <input
                          className="project-rename"
                          autoFocus
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onBlur={() => commitRename(p.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitRename(p.id);
                            else if (e.key === "Escape") setRenamingId(null);
                          }}
                        />
                      ) : (
                        <span className="project-plant-name">{p.name}</span>
                      )}
                    </td>
                    <td>{p.currency}</td>
                    <td className="number">{p.equipment_count}</td>
                    <td className="number">{fmt(p.metrics?.levelized_cost)}</td>
                    <td className="number">{fmtCompact(p.metrics?.npv)}</td>
                    <td className="number">{pct(p.metrics?.irr)}</td>
                    <td className="number">{fmt(p.metrics?.payback_time)}</td>
                    <td className="project-row-actions">
                      <button className="btn-primary btn-small" disabled={busy} onClick={() => handleOpen(p.id)}>
                        Open
                      </button>
                      <button
                        className="btn-secondary btn-small"
                        disabled={busy}
                        onClick={() => { setRenamingId(p.id); setRenameValue(p.name); }}
                      >
                        Rename
                      </button>
                      <button
                        className="btn-secondary btn-small"
                        disabled={busy}
                        onClick={() => run(() => duplicatePlant(p.id), { switched: true })}
                      >
                        Duplicate
                      </button>
                      <button
                        className="btn-secondary btn-small"
                        disabled={busy || !p.metrics}
                        title={p.metrics ? "Add to the Compare tab" : "Calculate this plant first"}
                        onClick={() => addToCompare([p.id])}
                      >
                        Compare
                      </button>
                      <button
                        className="btn-danger"
                        disabled={busy || project.plants.length <= 1}
                        title={project.plants.length <= 1 ? "A project needs at least one plant" : undefined}
                        onClick={() => handleDelete(p.id, p.name)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
