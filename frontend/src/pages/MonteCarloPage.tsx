import { useState } from "react";
import { runMonteCarlo } from "../api/client";
import type { MonteCarloResult } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

interface Props {
  setError: (e: string | null) => void;
}

export default function MonteCarloPage({ setError }: Props) {
  const [numSamples, setNumSamples] = useState(50000);
  const [batchSize, setBatchSize] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MonteCarloResult | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await runMonteCarlo({ num_samples: numSamples, batch_size: batchSize, additional_capex: false });
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Monte Carlo failed");
    } finally {
      setLoading(false);
    }
  };

  const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

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
        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Running Monte Carlo..." : "Run Monte Carlo"}
        </button>
        {loading && (
          <p style={{ color: "#868e96", fontSize: 13, marginTop: 8 }}>
            This may take a while with {numSamples.toLocaleString()} samples...
          </p>
        )}
      </div>

      {result && (
        <>
          {/* Summary statistics */}
          <div className="card">
            <h2>Results Summary ({result.num_samples.toLocaleString()} samples)</h2>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Metric</th><th>Mean</th><th>Std</th>
                    <th>P5</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th>
                    <th>Min</th><th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.metrics).map(([name, stats]) => (
                    <tr key={name}>
                      <td><strong>{name}</strong></td>
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
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Histograms for each metric */}
          {Object.entries(result.metrics).map(([name, stats]) => {
            const histData = stats.histogram.counts.map((count, i) => ({
              bin: ((stats.histogram.bin_edges[i] + stats.histogram.bin_edges[i + 1]) / 2),
              count,
            }));

            return (
              <div key={name} className="card">
                <h2>{name} Distribution</h2>
                <p style={{ fontSize: 13, color: "#868e96" }}>
                  Mean: {fmt(stats.mean)} | Std: {fmt(stats.std)} | 90% CI: [{fmt(stats.p5)}, {fmt(stats.p95)}]
                </p>
                <div style={{ height: 300, marginTop: 12 }}>
                  <ResponsiveContainer>
                    <BarChart data={histData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bin" tickFormatter={(v: number) => v.toFixed(1)} />
                      <YAxis />
                      <Tooltip
                        formatter={(v: number) => v.toLocaleString()}
                        labelFormatter={(v: number) => `Value: ${Number(v).toFixed(2)}`}
                      />
                      <Bar dataKey="count" fill="#4361ee" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}

          {/* Input distributions */}
          {Object.keys(result.inputs).length > 0 && (
            <div className="card">
              <h2>Input Parameter Distributions</h2>
              <table>
                <thead>
                  <tr><th>Input</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                </thead>
                <tbody>
                  {Object.entries(result.inputs).map(([name, stats]) => (
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
