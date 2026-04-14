import { useState } from "react";
import { runCalculations } from "../api/client";
import type { CalculationResults } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

interface Props {
  results: CalculationResults | null;
  setResults: (r: CalculationResults | null) => void;
  setError: (e: string | null) => void;
}

export default function ResultsPage({ results, setResults, setError }: Props) {
  const [loading, setLoading] = useState(false);

  const calculate = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await runCalculations();
      setResults(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Calculation failed");
    } finally {
      setLoading(false);
    }
  };

  const fmt = (n: number | null | undefined) =>
    n != null ? n.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "-";

  const pct = (n: number | null | undefined) =>
    n != null ? (n * 100).toFixed(2) + "%" : "-";

  if (!results) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 40 }}>
        <p style={{ color: "#868e96", marginBottom: 16 }}>Configure equipment and plant, then run calculations.</p>
        <button className="btn-primary" style={{ padding: "12px 32px", fontSize: 16 }} onClick={calculate} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Calculating..." : "Run Calculations"}
        </button>
      </div>
    );
  }

  const m = results.metrics;

  // CAPEX breakdown chart data
  const capex = results.capital_costs;
  const capexData = [
    { name: "ISBL", value: capex.isbl ?? 0 },
    { name: "OSBL", value: capex.osbl ?? 0 },
    { name: "D&E", value: capex.design_and_engineering ?? 0 },
    { name: "Contingency", value: capex.contingency ?? 0 },
  ].filter((d) => d.value > 0);

  // Fixed OPEX breakdown
  const fixedOpex = results.fixed_opex;
  const fixedOpexData = Object.entries(fixedOpex)
    .filter(([k, v]) => k !== "total" && typeof v === "number" && v > 0)
    .map(([k, v]) => ({ name: k.replace(/_/g, " "), value: v as number }));

  // Variable OPEX breakdown
  const varOpexData = Object.entries(results.variable_opex.breakdown)
    .filter(([, v]) => typeof v === "number" && (v as number) > 0)
    .map(([k, v]) => ({ name: k, value: v as number }));

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
        <div />
        <button className="btn-primary" onClick={calculate} disabled={loading}>
          {loading && <span className="spinner" />}Recalculate
        </button>
      </div>

      {/* Key metrics */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="label">Levelized Cost</div>
          <div className="value">{fmt(m.levelized_cost)}</div>
        </div>
        <div className="metric-card">
          <div className="label">NPV</div>
          <div className="value">{fmt(m.npv)}</div>
        </div>
        <div className="metric-card">
          <div className="label">IRR</div>
          <div className="value">{pct(m.irr)}</div>
        </div>
        <div className="metric-card">
          <div className="label">ROI</div>
          <div className="value">{pct(m.roi)}</div>
        </div>
        <div className="metric-card">
          <div className="label">Payback Time</div>
          <div className="value">{fmt(m.payback_time)} yr</div>
        </div>
      </div>

      {/* CAPEX */}
      <div className="card">
        <h2>Capital Costs</h2>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 300 }}>
            <table>
              <tbody>
                <tr><td>ISBL</td><td className="number">{fmt(capex.isbl)}</td></tr>
                <tr><td>OSBL</td><td className="number">{fmt(capex.osbl)}</td></tr>
                <tr><td>Design & Engineering</td><td className="number">{fmt(capex.design_and_engineering)}</td></tr>
                <tr><td>Contingency</td><td className="number">{fmt(capex.contingency)}</td></tr>
                <tr style={{ fontWeight: 700 }}><td>Fixed Capital</td><td className="number">{fmt(capex.fixed_capital)}</td></tr>
                <tr><td>Working Capital</td><td className="number">{fmt(capex.working_capital)}</td></tr>
              </tbody>
            </table>
          </div>
          {capexData.length > 0 && (
            <div style={{ flex: 1, minWidth: 300, height: 250 }}>
              <ResponsiveContainer>
                <BarChart data={capexData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#4361ee" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Fixed OPEX */}
      <div className="card">
        <h2>Fixed OPEX</h2>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 300 }}>
            <table>
              <tbody>
                {fixedOpexData.map((d) => (
                  <tr key={d.name}><td style={{ textTransform: "capitalize" }}>{d.name}</td><td className="number">{fmt(d.value)}</td></tr>
                ))}
                <tr style={{ fontWeight: 700 }}><td>Total</td><td className="number">{fmt(fixedOpex.total)}</td></tr>
              </tbody>
            </table>
          </div>
          {fixedOpexData.length > 1 && (
            <div style={{ flex: 1, minWidth: 300, height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={fixedOpexData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={(v: number) => (v / 1e3).toFixed(0) + "k"} />
                  <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#f72585" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Variable OPEX */}
      {varOpexData.length > 0 && (
        <div className="card">
          <h2>Variable OPEX (Annual: {fmt(results.variable_opex.total)})</h2>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 300 }}>
              <table>
                <tbody>
                  {varOpexData.map((d) => (
                    <tr key={d.name}><td>{d.name}</td><td className="number">{fmt(d.value)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ flex: 1, minWidth: 300, height: 250 }}>
              <ResponsiveContainer>
                <BarChart data={varOpexData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#7209b7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Revenue */}
      {results.revenue.total > 0 && (
        <div className="card">
          <h2>Revenue (Annual: {fmt(results.revenue.total)})</h2>
          <table>
            <tbody>
              {Object.entries(results.revenue.breakdown).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td className="number">{fmt(v as number)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Cash Flow Table */}
      {results.cash_flow.cash_flow && results.cash_flow.cash_flow.length > 0 && (
        <div className="card">
          <h2>Cash Flow</h2>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Year</th><th>CAPEX</th><th>Revenue</th><th>Costs</th>
                  <th>Gross Profit</th><th>Tax</th><th>Cash Flow</th>
                </tr>
              </thead>
              <tbody>
                {results.cash_flow.cash_flow.map((_val, yearIdx) => (
                  <tr key={yearIdx}>
                    <td>{yearIdx + 1}</td>
                    <td className="number">{fmt(results.cash_flow.capital_cost_array[yearIdx] ?? 0)}</td>
                    <td className="number">{fmt(results.cash_flow.revenue_array[yearIdx] ?? 0)}</td>
                    <td className="number">{fmt(results.cash_flow.cash_cost_array[yearIdx] ?? 0)}</td>
                    <td className="number">{fmt(results.cash_flow.gross_profit_array[yearIdx] ?? 0)}</td>
                    <td className="number">{fmt(results.cash_flow.tax_paid_array[yearIdx] ?? 0)}</td>
                    <td className="number">{fmt(results.cash_flow.cash_flow[yearIdx] ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
