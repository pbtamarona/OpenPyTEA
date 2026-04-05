import { useEffect, useState } from "react";
import {
  getSensitivityParameters, runSensitivity, runTornado,
} from "../api/client";
import type { SensitivityResult, TornadoResult } from "../types";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, ReferenceLine, Cell,
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
    }).catch(() => {});
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

  // Tornado chart data
  const tornChartData = tornResult
    ? tornResult.labels.map((label, i) => ({
        label,
        low: tornResult.lows[i] - tornResult.base_value,
        high: tornResult.highs[i] - tornResult.base_value,
      }))
    : [];

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
          <div className="chart-section" style={{ height: Math.max(300, tornChartData.length * 30 + 80), marginTop: 20 }}>
            <ResponsiveContainer>
              <BarChart data={tornChartData} layout="vertical" margin={{ left: 160 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12 }} />
                <Tooltip />
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
