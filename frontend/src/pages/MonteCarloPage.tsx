import { useMemo, useState } from "react";
import { runMonteCarlo } from "../api/client";
import type { MonteCarloMultiResult, MonteCarloResult, ComparedPlant } from "../types";
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";

const COLORS = ["#4361ee", "#e63946", "#06d6a0", "#f77f00", "#7209b7", "#4cc9f0", "#d62828", "#2a9d8f", "#e9c46a", "#264653"];

interface Props {
  setError: (e: string | null) => void;
  comparedPlants: ComparedPlant[];
}

const SQRT_2PI = Math.sqrt(2 * Math.PI);
const normalPdf = (x: number, mu: number, sigma: number) => {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-(z * z) / 2) / (sigma * SQRT_2PI);
};

interface OverlayRow {
  x: number;
  [series: string]: number;
}

function buildOverlayData(plants: MonteCarloResult[], metric: string, pdfPoints = 200): OverlayRow[] {
  const series = plants
    .map((p) => ({ name: p.name, num_samples: p.num_samples, stats: p.metrics[metric] }))
    .filter((s) => s.stats != null);
  if (series.length === 0) return [];

  let xMin = Infinity, xMax = -Infinity;
  series.forEach((s) => {
    const edges = s.stats.histogram.bin_edges;
    if (edges[0] < xMin) xMin = edges[0];
    if (edges[edges.length - 1] > xMax) xMax = edges[edges.length - 1];
    // Extend a bit beyond histogram so the fitted curve has tails.
    xMin = Math.min(xMin, s.stats.mean - 4 * s.stats.std);
    xMax = Math.max(xMax, s.stats.mean + 4 * s.stats.std);
  });

  // Collect x points: bin centers (per plant) + uniform PDF grid
  const xSet = new Set<number>();
  for (let i = 0; i <= pdfPoints; i++) xSet.add(xMin + ((xMax - xMin) * i) / pdfPoints);
  series.forEach((s) => {
    const edges = s.stats.histogram.bin_edges;
    for (let i = 0; i < s.stats.histogram.counts.length; i++) {
      xSet.add((edges[i] + edges[i + 1]) / 2);
    }
  });
  const xs = Array.from(xSet).sort((a, b) => a - b);

  return xs.map((x) => {
    const row: OverlayRow = { x };
    series.forEach((s) => {
      // Histogram density at x: lookup which bin the x falls inside
      const edges = s.stats.histogram.bin_edges;
      const counts = s.stats.histogram.counts;
      for (let i = 0; i < counts.length; i++) {
        const lo = edges[i];
        const hi = edges[i + 1];
        const center = (lo + hi) / 2;
        if (Math.abs(center - x) < 1e-9) {
          const w = hi - lo;
          if (w > 0) row[`${s.name}__hist`] = counts[i] / (s.num_samples * w);
          break;
        }
      }
      row[`${s.name}__pdf`] = normalPdf(x, s.stats.mean, s.stats.std);
    });
    return row;
  });
}

export default function MonteCarloPage({ setError, comparedPlants }: Props) {
  const [numSamples, setNumSamples] = useState(50000);
  const [batchSize, setBatchSize] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MonteCarloMultiResult | null>(null);
  const [selectedExtras, setSelectedExtras] = useState<Set<string>>(new Set());

  const overlayCandidates = useMemo(
    () => comparedPlants.filter((p) => p.source != null),
    [comparedPlants],
  );

  const toggleExtra = (id: string) => {
    setSelectedExtras((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const extra_plants = overlayCandidates
        .filter((p) => selectedExtras.has(p.id))
        .map((p) => p.source!)
        .filter(Boolean);
      const r = await runMonteCarlo({
        num_samples: numSamples, batch_size: batchSize, additional_capex: false,
        extra_plants,
      });
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Monte Carlo failed");
    } finally {
      setLoading(false);
    }
  };

  const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });
  const plants = result?.plants ?? [];

  // Collect all metric names that appeared on at least one plant
  const metricNames = useMemo(() => {
    const set = new Set<string>();
    plants.forEach((p) => Object.keys(p.metrics).forEach((m) => set.add(m)));
    return Array.from(set);
  }, [plants]);

  return (
    <div>
      <div className="card">
        <h2>Monte Carlo Uncertainty Analysis</h2>
        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="form-group">
            <label>Number of Samples</label>
            <input type="number" value={numSamples} onChange={(e) => setNumSamples(+e.target.value)} />
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
          </div>
        </div>

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
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
              Each selected plant runs an independent Monte Carlo simulation with the same sample count.
            </p>
          </div>
        )}

        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Running Monte Carlo..." : "Run Monte Carlo"}
        </button>
        {loading && (
          <p style={{ color: "#868e96", fontSize: 13, marginTop: 8 }}>
            This may take a while with {numSamples.toLocaleString()} samples
            {selectedExtras.size > 0 ? ` × ${1 + selectedExtras.size} plants` : ""}...
          </p>
        )}
      </div>

      {plants.length > 0 && (
        <>
          {/* Summary statistics — one row per (plant, metric) */}
          <div className="card">
            <h2>Results Summary ({plants[0].num_samples.toLocaleString()} samples)</h2>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Plant</th><th>Metric</th><th>Mean</th><th>Std</th>
                    <th>P5</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th>
                    <th>Min</th><th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {plants.flatMap((p, pi) =>
                    Object.entries(p.metrics).map(([name, stats]) => (
                      <tr key={`${p.name}-${name}`}>
                        <td>
                          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: COLORS[pi % COLORS.length], marginRight: 8, verticalAlign: "middle" }} />
                          <strong>{p.name}</strong>
                        </td>
                        <td>{name}</td>
                        <td className="number">{fmt(stats.mean)}</td>
                        <td className="number">{fmt(stats.std)}</td>
                        <td className="number">{fmt(stats.p5)}</td>
                        <td className="number">{fmt(stats.p25)}</td>
                        <td className="number">{fmt(stats.p50)}</td>
                        <td className="number">{fmt(stats.p75)}</td>
                        <td className="number">{fmt(stats.p95)}</td>
                        <td className="number">{fmt(stats.min)}</td>
                        <td className="number">{fmt(stats.max)}</td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Histograms for each metric — overlaid across plants */}
          {metricNames.map((metric) => {
            const data = buildOverlayData(plants, metric);
            if (data.length === 0) return null;
            const seriesPlants = plants.filter((p) => p.metrics[metric] != null);

            return (
              <div key={metric} className="card">
                <h2>{metric} Distribution</h2>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 18, fontSize: 13, color: "var(--text-secondary)", marginBottom: 8 }}>
                  {seriesPlants.map((p) => {
                    const s = p.metrics[metric]!;
                    return (
                      <span key={p.name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 12, height: 12, background: COLORS[plants.indexOf(p) % COLORS.length], display: "inline-block" }} />
                        <strong>{p.name}</strong> μ={fmt(s.mean)}, σ={fmt(s.std)}
                      </span>
                    );
                  })}
                </div>
                <DownloadableChart filename={`mc_${metric}`} height={340}>
                  <ResponsiveContainer>
                    <ComposedChart data={data} margin={{ bottom: 30, left: 10, top: 10, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="x"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={(v: number) => v.toFixed(2)}
                        label={{ value: metric, position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#666" } }}
                      />
                      <YAxis label={{ value: "Probability density", angle: -90, position: "insideLeft", offset: 4, style: { fontSize: 12, fill: "#666" } }} />
                      <Tooltip
                        labelFormatter={(v) => `${metric}: ${Number(v).toFixed(3)}`}
                        formatter={(v, name) => {
                          const label = String(name);
                          const isHist = label.endsWith("__hist");
                          const isPdf = label.endsWith("__pdf");
                          const plantName = label.replace(/__(hist|pdf)$/, "");
                          const tag = isHist ? "histogram" : isPdf ? "fitted normal" : "";
                          return [Number(v).toExponential(2), `${plantName}${tag ? ` (${tag})` : ""}`];
                        }}
                      />
                      {seriesPlants.map((p) => {
                        const color = COLORS[plants.indexOf(p) % COLORS.length];
                        return [
                          <Area
                            key={`${p.name}-hist`}
                            type="stepAfter"
                            dataKey={`${p.name}__hist`}
                            stroke={color}
                            strokeOpacity={0.5}
                            fill={color}
                            fillOpacity={0.25}
                            isAnimationActive={false}
                            connectNulls={false}
                          />,
                          <Line
                            key={`${p.name}-pdf`}
                            type="monotone"
                            dataKey={`${p.name}__pdf`}
                            stroke={color}
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />,
                        ];
                      })}
                    </ComposedChart>
                  </ResponsiveContainer>
                </DownloadableChart>
              </div>
            );
          })}

          {/* Input distributions — show active plant only (extras share the same priors) */}
          {plants[0] && Object.keys(plants[0].inputs).length > 0 && (
            <div className="card">
              <h2>Input Parameter Distributions ({plants[0].name})</h2>
              <table>
                <thead>
                  <tr><th>Input</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                </thead>
                <tbody>
                  {Object.entries(plants[0].inputs).map(([name, stats]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="number">{fmt(stats.mean)}</td>
                      <td className="number">{fmt(stats.std)}</td>
                      <td className="number">{fmt(stats.min)}</td>
                      <td className="number">{fmt(stats.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
