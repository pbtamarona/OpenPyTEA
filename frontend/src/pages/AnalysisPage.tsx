import { useEffect, useState } from "react";
import {
  getSensitivityParameters, runSensitivity, runTornado,
} from "../api/client";
import type { SensitivityResult, TornadoResult } from "../types";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, ReferenceLine, Cell, Legend,
} from "recharts";

const METRICS = ["LCOP", "NPV", "IRR", "ROI", "PBT"];

interface Props {
  setError: (e: string | null) => void;
}

export default function AnalysisPage({ setError }: Props) {
  const [parameters, setParameters] = useState<string[]>([]);
  const [sensParam, setSensParam] = useState("");
  const [sensPM, setSensPM] = useState(0.2);
  const [sensPoints, setSensPoints] = useState(21);
  const [sensMetric, setSensMetric] = useState("LCOP");
  const [sensResult, setSensResult] = useState<SensitivityResult | null>(null);
  const [sensLoading, setSensLoading] = useState(false);

  const [tornPM, setTornPM] = useState(0.2);
  const [tornMetric, setTornMetric] = useState("LCOP");
  const [tornResult, setTornResult] = useState<TornadoResult | null>(null);
  const [tornLoading, setTornLoading] = useState(false);

  const fetchParams = () => {
    getSensitivityParameters().then((p) => {
      setParameters(p);
      if (p.length > 0 && !sensParam) setSensParam(p[0]);
    }).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : "Failed to load parameters");
    });
  };

  useEffect(fetchParams, []);

  const doSensitivity = async () => {
    setSensLoading(true);
    setError(null);
    try {
      const r = await runSensitivity({
        parameter: sensParam, plus_minus_value: sensPM,
        n_points: sensPoints, metric: sensMetric, additional_capex: false,
      });
      setSensResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Sensitivity failed");
    } finally {
      setSensLoading(false);
    }
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

  // Sensitivity chart data
  const sensChartData = sensResult
    ? sensResult.curves[0]?.x.map((x, i) => ({
        x,
        y: sensResult.curves[0]?.y[i] ?? 0,
      }))
    : [];

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

  const tornXLabel = (() => {
    if (!tornResult) return "";
    const raw = tornResult.xlabel
      .replace(/\$\\cdot\$/g, "·")
      .replace(/\$\^-1\$/g, "⁻¹")
      .replace(/\\%/g, "%")
      .replace(/\$.*?\$/g, "")
      .replace(/\\/g, "");
    return tornScale === 1e6 ? raw.replace("/ [", "/ [million ") : raw;
  })();

  return (
    <div>
      {/* Sensitivity */}
      <div className="card">
        <h2>Sensitivity Analysis</h2>
        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="form-group">
            <label>Parameter</label>
            <select value={sensParam} onChange={(e) => setSensParam(e.target.value)}>
              {parameters.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>+/- Variation</label>
            <input type="number" step="0.05" value={sensPM} onChange={(e) => setSensPM(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Points</label>
            <input type="number" value={sensPoints} onChange={(e) => setSensPoints(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Metric</label>
            <select value={sensMetric} onChange={(e) => setSensMetric(e.target.value)}>
              {METRICS.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
        </div>
        <button className="btn-primary" onClick={doSensitivity} disabled={sensLoading}>
          {sensLoading && <span className="spinner" />}Run Sensitivity
        </button>

        {sensResult && sensChartData.length > 0 && (
          <div className="chart-section" style={{ height: 350, marginTop: 20 }}>
            <ResponsiveContainer>
              <LineChart data={sensChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" label={{ value: sensResult.xlabel.replace(/\$.*?\$/g, "").replace(/\\/g, ""), position: "bottom" }} />
                <YAxis label={{ value: sensResult.metric, angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line type="monotone" dataKey="y" stroke="#4361ee" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
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
          <div className="chart-section" style={{ height: Math.max(300, tornChartData.length * 30 + 120), marginTop: 20 }}>
            <ResponsiveContainer>
              <BarChart data={tornChartData} layout="vertical" margin={{ left: 160, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={(v) => (v + tornBase).toFixed(tornMetric === "IRR" ? 3 : 2)} label={{ value: tornXLabel, position: "insideBottom", offset: -20, style: { fontWeight: "bold" } }} />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12, fontWeight: "bold" }} />
                <Tooltip formatter={(v) => { const unit = tornXLabel.match(/\[(.+)\]/)?.[1] ?? ""; const actual = typeof v === "number" ? v + tornBase : v; const val = typeof actual === "number" ? actual.toFixed(tornMetric === "IRR" ? 3 : 2) : actual; return (unit && tornMetric !== "IRR") ? `${val} ${unit}` : val; }} labelStyle={{ fontWeight: "bold", color: "#000" }} />
                <Legend verticalAlign="bottom" align="right" content={() => (
                  <div style={{ display: "flex", gap: 16, justifyContent: "flex-end", fontSize: 12, color: "#ccc" }}>
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
          </div>
        )}
      </div>
    </div>
  );
}
