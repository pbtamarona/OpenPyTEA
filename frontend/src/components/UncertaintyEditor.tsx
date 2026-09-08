import { useState } from "react";
import type { UncertaintyBlock } from "../types";

// The distribution families the library's make_distribution() supports,
// with the fields each one actually reads. `key` is the config key the
// value is written to; labels spell out what the parameter means for
// that family. Fields marked optional may be left blank.
interface FieldSpec {
  key: "loc" | "std" | "scale" | "shape" | "min" | "max";
  label: string;
  optional?: boolean;
  placeholder?: string;
}

interface DistSpec {
  id: number;
  name: string;
  fields: FieldSpec[];
  note?: string;
}

export const DISTRIBUTIONS: DistSpec[] = [
  {
    id: 3, name: "Normal",
    fields: [
      { key: "loc", label: "Mean", optional: true, placeholder: "baseline" },
      { key: "std", label: "Std dev" },
      { key: "min", label: "Min (truncation)", optional: true },
      { key: "max", label: "Max (truncation)", optional: true },
    ],
  },
  {
    id: 2, name: "Lognormal",
    // μ must be explicit: a blank one falls back to the parameter's
    // baseline value, which as a log-mean overflows immediately
    fields: [
      { key: "loc", label: "μ (log-mean)" },
      { key: "std", label: "σ (log-std)" },
      { key: "min", label: "Min (truncation)" },
      { key: "max", label: "Max (truncation)" },
    ],
  },
  {
    id: 4, name: "Uniform",
    fields: [
      { key: "min", label: "Minimum" },
      { key: "max", label: "Maximum" },
    ],
  },
  {
    id: 5, name: "Triangular",
    fields: [
      { key: "loc", label: "Mode", optional: true, placeholder: "baseline" },
      { key: "min", label: "Minimum" },
      { key: "max", label: "Maximum" },
    ],
  },
  {
    id: 8, name: "Weibull",
    // The library doesn't truncate this family — no min/max offered
    fields: [
      { key: "shape", label: "k (shape)" },
      { key: "scale", label: "λ (scale)" },
      { key: "loc", label: "Offset", optional: true, placeholder: "baseline" },
    ],
    note: "Offset defaults to the parameter's baseline value; samples are unbounded above.",
  },
  {
    id: 9, name: "Gamma",
    fields: [
      { key: "shape", label: "k (shape)" },
      { key: "scale", label: "θ (scale)" },
      { key: "loc", label: "Offset", optional: true, placeholder: "baseline" },
    ],
    note: "Offset defaults to the parameter's baseline value; samples are unbounded above.",
  },
  {
    id: 10, name: "Beta",
    fields: [
      { key: "loc", label: "α (alpha)" },
      { key: "shape", label: "β (beta)" },
      { key: "max", label: "Upper bound" },
    ],
  },
  {
    id: 11, name: "GEV",
    fields: [
      { key: "loc", label: "μ (location)" },
      { key: "scale", label: "σ (scale)" },
      { key: "shape", label: "ξ (shape)" },
    ],
    note: "Samples are not truncated for this family.",
  },
  {
    id: 12, name: "Student's t",
    fields: [
      { key: "loc", label: "Median", optional: true, placeholder: "baseline" },
      { key: "scale", label: "Scale" },
      { key: "shape", label: "ν (deg. of freedom)" },
    ],
    note: "Samples are not truncated for this family.",
  },
  {
    id: 7, name: "Discrete uniform",
    fields: [
      { key: "min", label: "Minimum (integer)" },
      { key: "max", label: "Maximum (integer, inclusive)" },
    ],
  },
  {
    id: 6, name: "Bernoulli",
    fields: [
      { key: "loc", label: "p (success probability)" },
      { key: "scale", label: "Success value", optional: true, placeholder: "1" },
    ],
    note: "Outcome is 0 with probability 1−p, the success value with probability p.",
  },
];

// Compact summary for table buttons: "Normal (σ=0.3)", "Uniform [0.5, 5]", "—".
export function uncertaintySummary(block: UncertaintyBlock | null | undefined): string {
  if (!block) return "—";
  const id = block.dist_id ?? 3;
  const dist = DISTRIBUTIONS.find((d) => d.id === id);
  if (!dist) return `dist ${id}`;
  const spread = block.std ?? block.scale;
  if (id === 4 || id === 7) return `${dist.name} [${block.min ?? "?"}, ${block.max ?? "?"}]`;
  if (id === 5) return `Triangular [${block.min ?? "?"}, ${block.max ?? "?"}]`;
  if (id === 3 && !((spread ?? 0) > 0)) return "—";
  return spread != null ? `${dist.name} (σ=${spread})` : dist.name;
}

interface Props {
  title: string;
  value: UncertaintyBlock | null;
  onSave: (block: UncertaintyBlock | null) => void;
  onClose: () => void;
}

export default function UncertaintyEditor({ title, value, onSave, onClose }: Props) {
  const [distId, setDistId] = useState<number>(value?.dist_id ?? 3);
  const [fields, setFields] = useState<Record<string, number | null>>({
    loc: value?.loc ?? null,
    std: value?.std ?? null,
    scale: value?.scale ?? null,
    shape: value?.shape ?? null,
    min: value?.min ?? null,
    max: value?.max ?? null,
  });
  const [err, setErr] = useState<string | null>(null);

  const dist = DISTRIBUTIONS.find((d) => d.id === distId)!;

  const save = () => {
    const block: UncertaintyBlock = { dist_id: distId };
    for (const f of dist.fields) {
      const v = fields[f.key];
      if (v == null) {
        if (!f.optional) {
          setErr(`"${f.label}" is required for a ${dist.name} distribution`);
          return;
        }
        continue;
      }
      block[f.key] = v;
    }
    // A Normal with no spread is no uncertainty at all
    if (distId === 3 && !((block.std ?? 0) > 0)) {
      onSave(null);
      return;
    }
    onSave(block);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 460 }} onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        {err && <div style={{ color: "#e63946", marginBottom: 12, fontSize: 13 }}>{err}</div>}
        <div className="form-grid">
          <div className="form-group">
            <label>Distribution</label>
            <select value={distId} onChange={(e) => { setDistId(+e.target.value); setErr(null); }}>
              {DISTRIBUTIONS.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          {dist.fields.map((f) => (
            <div className="form-group" key={f.key}>
              <label>{f.label}{f.optional ? "" : " *"}</label>
              <input
                type="number" step="any"
                placeholder={f.placeholder ?? (f.optional ? "(blank = default)" : "")}
                value={fields[f.key] ?? ""}
                onChange={(e) => setFields({ ...fields, [f.key]: e.target.value === "" ? null : +e.target.value })}
              />
            </div>
          ))}
        </div>
        {dist.note && <p style={{ fontSize: 12, color: "#868e96", marginTop: 8 }}>{dist.note}</p>}
        <div className="modal-actions">
          <button className="btn-danger" onClick={() => onSave(null)}>No uncertainty</button>
          <button className="btn-secondary" style={{ color: "#495057", borderColor: "#dee2e6" }} onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={save}>Apply</button>
        </div>
      </div>
    </div>
  );
}
