import { useEffect, useMemo, useState } from "react";
import {
  getSensitivityParameters, runSensitivity, runTornado,
} from "../api/client";
import type { SensitivityResult, TornadoResult, ComparedPlant } from "../types";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, ReferenceLine, Cell, Legend,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";

const METRICS = ["LCOP", "NPV", "IRR", "ROI", "PBT"];
const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

const cleanLatex = (s: string) =>
  s.replace(/\$\\cdot\$/g, "·").replace(/\$\^-1\$/g, "⁻¹").replace(/\\%/g, "%").replace(/\$.*?\$/g, "").replace(/\\/g, "");

const formatParam = (p: string) => {
  const last = p.split(".").pop() ?? p;
  const label = last.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
  return p.includes(".") ? `${label} price` : label;
};

interface Panel {
  id: string;
  parameter: string;
  metric: string;
  plus_minus_value: number;
  result: SensitivityResult | null;
  loading: boolean;
}

interface Props {
  setError: (e: string | null) => void;
  comparedPlants: ComparedPlant[];
}

const newPanel = (parameter = "", metric = "LCOP", plus_minus_value = 0.2): Panel => ({
  id: crypto.randomUUID(),
  parameter,
  metric,
  plus_minus_value,
  result: null,
  loading: false,
});

export default function AnalysisPage({ setError, comparedPlants }: Props) {
  const [parameters, setParameters] = useState<string[]>([]);
  const [sensPoints, setSensPoints] = useState(21);
  const [panels, setPanels] = useState<Panel[]>([newPanel()]);
  const [selectedExtras, setSelectedExtras] = useState<Set<string>>(new Set());

  const [tornPM, setTornPM] = useState(0.2);
  const [tornMetric, setTornMetric] = useState("LCOP");
  const [tornResult, setTornResult] = useState<TornadoResult | null>(null);
  const [tornLoading, setTornLoading] = useState(false);

  const overlayCandidates = useMemo(
    () => comparedPlants.filter((p) => p.source != null),
    [comparedPlants],
  );

  useEffect(() => {
    getSensitivityParameters().then((p) => {
      setParameters(p);
      setPanels((prev) =>
        prev.map((panel) => (panel.parameter === "" && p.length > 0 ? { ...panel, parameter: p[0] } : panel)),
      );
    }).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load parameters");
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updatePanel = (id: string, patch: Partial<Panel>) =>
    setPanels((prev) => prev.map((p) => (p.id === id ? { ...p, ...patch } : p)));

  const removePanel = (id: string) =>
    setPanels((prev) => (prev.length > 1 ? prev.filter((p) => p.id !== id) : prev));

  const addPanel = () => {
    const last = panels[panels.length - 1];
    setPanels((prev) => [
      ...prev,
      newPanel(last?.parameter || parameters[0] || "", last?.metric || "LCOP", last?.plus_minus_value ?? 0.2),
    ]);
  };

  const toggleExtra = (id: string) => {
    setSelectedExtras((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const collectExtras = () =>
    overlayCandidates
      .filter((p) => selectedExtras.has(p.id))
      .map((p) => p.source!)
      .filter(Boolean);

  const runAll = async () => {
    setError(null);
    const extra_plants = collectExtras();
    setPanels((prev) => prev.map((p) => ({ ...p, loading: true })));

    const tasks = panels.map(async (panel) => {
      try {
        const r = await runSensitivity({
          parameter: panel.parameter,
          plus_minus_value: panel.plus_minus_value,
          n_points: sensPoints,
          metric: panel.metric,
          additional_capex: false,
          extra_plants,
        });
        return { id: panel.id, result: r as SensitivityResult | null, error: null as string | null };
      } catch (e: unknown) {
        return { id: panel.id, result: null, error: e instanceof Error ? e.message : "Sensitivity failed" };
      }
    });

    const outcomes = await Promise.all(tasks);
    const errors = outcomes.filter((o) => o.error).map((o) => o.error!);
    if (errors.length) setError(errors.join(" • "));

    setPanels((prev) =>
      prev.map((p) => {
        const out = outcomes.find((o) => o.id === p.id);
        return out ? { ...p, loading: false, result: out.result ?? p.result } : { ...p, loading: false };
      }),
    );
  };

  const doTornado = async () => {
    setTornLoading(true);
    setError(null);
    try {
      const r = await runTornado({ plus_minus_value: tornPM, metric: tornMetric, additional_capex: false });
      setTornResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Tornado failed");
    } finally {
      setTornLoading(false);
    }
  };

  const anyLoading = panels.some((p) => p.loading);
  const isGrid = panels.length > 1;

  // Tornado chart data — bars are deltas from base_value; x-axis ticks are offset to show actual values
  const tornMaxDelta = tornResult
    ? Math.max(...tornResult.lows.concat(tornResult.highs).map((v) => Math.abs(v - tornResult.base_value)))
    : 0;
  const tornScale = (tornResult && Math.abs(tornResult.base_value) >= 1e6) || tornMaxDelta >= 1e6 ? 1e6 : 1;

  const tornChartData = tornResult
    ? tornResult.labels
        .map((label, i) => ({
          label,
          low: (tornResult.lows[i] - tornResult.base_value) / tornScale,
          high: (tornResult.highs[i] - tornResult.base_value) / tornScale,
          range: Math.abs(tornResult.highs[i] - tornResult.lows[i]),
        }))
        .sort((a, b) => b.range - a.range)
    : [];

  const tornBase = tornResult ? tornResult.base_value / tornScale : 0;

  const tornXLabel = tornResult
    ? (tornScale === 1e6 ? cleanLatex(tornResult.xlabel).replace("/ [", "/ [million ") : cleanLatex(tornResult.xlabel))
    : "";

  return (
    <div>
      {/* Sensitivity */}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Sensitivity Analysis</h2>

        {overlayCandidates.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "var(--text-secondary)" }}>
              Compare with (from Compare tab)
            </label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              {overlayCandidates.map((p) => (
                <label key={p.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
                  <input
                    type="checkbox"
                    checked={selectedExtras.has(p.id)}
                    onChange={() => toggleExtra(p.id)}
                  />
                  {p.name}
                </label>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <button className="btn-primary" onClick={runAll} disabled={anyLoading}>
            {anyLoading && <span className="spinner" />}Run Sensitivity{panels.length > 1 ? ` (${panels.length} panels)` : ""}
          </button>
          <button
            onClick={addPanel}
            title="Add another panel with its own parameter & metric"
            style={{
              padding: "8px 16px",
              border: "1px solid var(--accent)",
              background: "transparent",
              color: "var(--accent)",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500,
            }}
          >
            + Add panel
          </button>
          <label style={{ fontSize: 13, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
            Points
            <input
              type="number"
              value={sensPoints}
              onChange={(e) => setSensPoints(+e.target.value)}
              style={{ width: 70 }}
            />
          </label>
        </div>

        <div
          style={
            isGrid
              ? { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 16 }
              : undefined
          }
        >
          {panels.map((panel, idx) => (
            <SensitivityPanel
              key={panel.id}
              panel={panel}
              parameters={parameters}
              showRemove={panels.length > 1}
              compact={isGrid}
              label={isGrid ? `(${String.fromCharCode(97 + idx)})` : undefined}
              onChange={(patch) => updatePanel(panel.id, patch)}
              onRemove={() => removePanel(panel.id)}
            />
          ))}
        </div>
      </div>

      {/* Tornado */}
      <div className="card">
        <h2>Tornado Analysis</h2>
        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="form-group">
            <label>+/- Variation</label>
            <input type="number" step="0.05" value={tornPM} onChange={(e) => setTornPM(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Metric</label>
            <select value={tornMetric} onChange={(e) => setTornMetric(e.target.value)}>
              {METRICS.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
        </div>
        <button className="btn-primary" onClick={doTornado} disabled={tornLoading}>
          {tornLoading && <span className="spinner" />}Run Tornado
        </button>

        {tornResult && tornChartData.length > 0 && (
          <DownloadableChart filename="tornado" height={Math.max(300, tornChartData.length * 30 + 120)} style={{ marginTop: 20 }}>
            <ResponsiveContainer>
              <BarChart data={tornChartData} layout="vertical" margin={{ left: 160, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={(v) => (v + tornBase).toFixed(tornMetric === "IRR" ? 3 : 2)} label={{ value: tornXLabel, position: "insideBottom", offset: -20, style: { fontWeight: "bold" } }} />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12, fontWeight: "bold" }} />
                <Tooltip formatter={(v) => { const unit = tornXLabel.match(/\[(.+)\]/)?.[1] ?? ""; const actual = typeof v === "number" ? v + tornBase : v; const val = typeof actual === "number" ? actual.toFixed(tornMetric === "IRR" ? 3 : 2) : actual; return (unit && tornMetric !== "IRR") ? `${val} ${unit}` : val; }} labelStyle={{ fontWeight: "bold", color: "#000" }} />
                <Legend verticalAlign="bottom" align="right" content={() => (
                  <div style={{ display: "flex", gap: 16, justifyContent: "flex-end", fontSize: 12, color: "var(--text-secondary)" }}>
                    {[{ color: "#4361ee", label: `-${tornPM * 100}%` }, { color: "#e63946", label: `+${tornPM * 100}%` }].map(({ color, label }) => (
                      <span key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ width: 12, height: 12, background: color, display: "inline-block" }} />
                        {label}
                      </span>
                    ))}
                  </div>
                )} />
                <ReferenceLine x={0} stroke="#333" strokeDasharray="3 3" />
                <Bar dataKey="low" name="Low">
                  {tornChartData.map((_, i) => <Cell key={i} fill="#4361ee" />)}
                </Bar>
                <Bar dataKey="high" name="High">
                  {tornChartData.map((_, i) => <Cell key={i} fill="#e63946" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </DownloadableChart>
        )}
      </div>
    </div>
  );
}

interface PanelProps {
  panel: Panel;
  parameters: string[];
  showRemove: boolean;
  compact: boolean;
  label?: string;
  onChange: (patch: Partial<Panel>) => void;
  onRemove: () => void;
}

function SensitivityPanel({ panel, parameters, showRemove, compact, label, onChange, onRemove }: PanelProps) {
  const result = panel.result;

  const allY = result ? result.curves.flatMap((c) => c.y).filter((v): v is number => typeof v === "number") : [];
  const maxAbs = allY.length ? Math.max(...allY.map(Math.abs)) : 0;
  const scale = maxAbs >= 1e6 ? 1e6 : 1;

  const chartData = useMemo(() => {
    if (!result || result.curves.length === 0) return [];
    const xs = result.curves[0].x;
    return xs.map((x, i) => {
      const row: Record<string, number> = { x };
      result.curves.forEach((c) => {
        const y = c.y[i];
        if (typeof y === "number") row[c.plant] = y / scale;
      });
      return row;
    });
  }, [result, scale]);

  const yLabel = result
    ? (scale === 1e6 ? cleanLatex(result.ylabel).replace("/ [", "/ [million ") : cleanLatex(result.ylabel))
    : "";
  const xLabel = result ? cleanLatex(result.xlabel) : "";
  const chartHeight = compact ? 280 : 380;

  return (
    <div style={{ border: compact ? "1px solid var(--border)" : undefined, borderRadius: compact ? 6 : 0, padding: compact ? 12 : 0, marginBottom: compact ? 0 : 12 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "end", gap: 12, marginBottom: 12 }}>
        {label && <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>{label}</span>}
        <div className="form-group" style={{ flex: "1 1 200px", marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>Parameter</label>
          <select value={panel.parameter} onChange={(e) => onChange({ parameter: e.target.value })}>
            {parameters.map((p) => <option key={p} value={p}>{formatParam(p)}</option>)}
          </select>
        </div>
        <div className="form-group" style={{ width: 100, marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>Metric</label>
          <select value={panel.metric} onChange={(e) => onChange({ metric: e.target.value })}>
            {METRICS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div className="form-group" style={{ width: 90, marginBottom: 0 }}>
          <label style={{ fontSize: 12 }}>+/- Var</label>
          <input
            type="number"
            step="0.05"
            value={panel.plus_minus_value}
            onChange={(e) => onChange({ plus_minus_value: +e.target.value })}
          />
        </div>
        {showRemove && (
          <button className="btn-danger" onClick={onRemove} style={{ marginBottom: 0 }}>Remove</button>
        )}
      </div>

      {result && chartData.length > 0 && (
        <DownloadableChart filename={`sensitivity_${panel.parameter}_${panel.metric}`} height={chartHeight} style={{ marginTop: 4 }}>
          <div style={{ position: "absolute", left: 0, top: 0, bottom: 50, width: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ transform: "rotate(-90deg)", whiteSpace: "nowrap", fontWeight: "bold", fontSize: compact ? 12 : 14, color: "#666" }}>{yLabel}</span>
          </div>
          <div style={{ position: "absolute", left: 24, right: 0, top: 0, bottom: 0 }}>
            <ResponsiveContainer>
              <LineChart data={chartData} margin={{ left: 10, bottom: 40, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" tickFormatter={(v) => parseFloat(v.toFixed(4)).toString()} label={{ value: xLabel, position: "insideBottom", offset: -20, style: { fontWeight: "bold", fontSize: compact ? 12 : 14, fill: "#666" } }} />
                <YAxis width={60} />
                <Tooltip
                  labelFormatter={(v) => `x: ${typeof v === "number" ? v.toFixed(2) : parseFloat(String(v)).toFixed(2)} %`}
                  formatter={(v, name) => {
                    const unit = yLabel.match(/\[(.+)\]/)?.[1] ?? "";
                    const val = typeof v === "number" ? v.toFixed(panel.metric === "IRR" ? 3 : 2) : v;
                    return [`${val}${unit ? ` ${unit}` : ""}`, name];
                  }}
                  labelStyle={{ color: "#000" }}
                />
                {result.curves.length > 1 && <Legend verticalAlign="top" height={28} />}
                {result.curves.map((c, i) => (
                  <Line key={c.plant} type="monotone" dataKey={c.plant} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} connectNulls />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </DownloadableChart>
      )}
    </div>
  );
}
