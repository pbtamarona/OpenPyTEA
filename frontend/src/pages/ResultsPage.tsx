import { useState, useEffect } from "react";
import { runCalculations, getPlantConfig } from "../api/client";
import type { CalculationResults, PlantConfig } from "../types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import DownloadableChart from "../components/DownloadableChart";

interface Props {
  results: CalculationResults | null;
  setResults: (r: CalculationResults | null) => void;
  setError: (e: string | null) => void;
}

export default function ResultsPage({ results, setResults, setError }: Props) {
  const [loading, setLoading] = useState(false);
  const [plantConfig, setPlantConfig] = useState<PlantConfig | null>(null);

  useEffect(() => {
    getPlantConfig().then(setPlantConfig).catch(() => {});
  }, []);

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

  const byDesc = (a: { value: number }, b: { value: number }) => b.value - a.value;

  // CAPEX breakdown chart data
  const capex = results.capital_costs;
  const capexData = [
    { name: "ISBL", value: capex.isbl ?? 0 },
    { name: "OSBL", value: capex.osbl ?? 0 },
    { name: "D&E", value: capex.design_and_engineering ?? 0 },
    { name: "Contingency", value: capex.contingency ?? 0 },
  ].filter((d) => d.value > 0).sort(byDesc);

  // Fixed OPEX breakdown
  const fixedOpex = results.fixed_opex;
  const fixedOpexData = Object.entries(fixedOpex)
    .filter(([k, v]) => k !== "total" && typeof v === "number" && v > 0)
    .map(([k, v]) => ({ name: k.replace(/_/g, " "), value: v as number }))
    .sort(byDesc);

  // Variable OPEX breakdown
  const varOpexData = Object.entries(results.variable_opex.breakdown)
    .filter(([, v]) => typeof v === "number" && (v as number) > 0)
    .map(([k, v]) => ({ name: k, value: v as number }))
    .sort(byDesc);

  // Revenue breakdown
  const revenueData = Object.entries(results.revenue.breakdown)
    .map(([k, v]) => ({ name: k, value: v as number }))
    .sort(byDesc);

  const currency = plantConfig?.currency || "USD";
  const lcUnit = `${currency}/Unit`;

  const discountRate = plantConfig?.interest_rate ?? 0;
  const cumulativeNPV = results.cash_flow.cash_flow.reduce<number[]>((acc, cf, i) => {
    const discounted = cf / Math.pow(1 + discountRate, i + 1);
    acc.push((acc[i - 1] ?? 0) + discounted);
    return acc;
  }, []);

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
          <div className="label">Levelized Cost (in {lcUnit})</div>
          <div className="value">{fmt(m.levelized_cost)}</div>
        </div>
        <div className="metric-card">
          <div className="label">NPV (in {currency})</div>
          <div className="value">{fmt(m.npv)}</div>
        </div>
        <div className="metric-card">
          <div className="label">IRR</div>
          <div className="value">{pct(m.irr)}</div>
        </div>
        <div className="metric-card">
          <div className="label">ROI</div>
          <div className="value">{m.roi != null ? m.roi.toFixed(2) + "%" : "-"}</div>
        </div>
        <div className="metric-card">
          <div className="label">Payback Time</div>
          <div className="value">{fmt(m.payback_time)} yr</div>
        </div>
      </div>

      {/* CAPEX */}
      <div className="card">
        <h2>Capital Costs (in {currency})</h2>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
          <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
            <tbody>
              {capexData.map((d) => (
                <tr key={d.name}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
              ))}
              <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>Fixed Capital</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(capex.fixed_capital)}</td></tr>
              <tr><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>Working Capital</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(capex.working_capital)}</td></tr>
            </tbody>
          </table>
          {capexData.length > 0 && (
            <DownloadableChart filename="capital_costs" height={capexData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={capexData} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 30, left: 8 }}>
                  <XAxis type="number" tickFormatter={(v: number) => (v / 1e6).toFixed(0) + "M"} tick={{ fontSize: 11 }} domain={[0, Math.max(...capexData.map(d => d.value)) * 1.1]} label={{ value: `Cost (${currency})`, position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#4361ee" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          )}
        </div>
      </div>

      {/* Fixed OPEX */}
      <div className="card">
        <h2>Fixed OPEX (in {currency})</h2>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
          <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
            <tbody>
              {fixedOpexData.map((d) => (
                <tr key={d.name}><td style={{ textTransform: "capitalize", paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
              ))}
              <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12 }}>Total</td><td className="number">{fmt(fixedOpex.total)}</td></tr>
            </tbody>
          </table>
          {fixedOpexData.length > 0 && (
            <DownloadableChart filename="fixed_opex" height={fixedOpexData.length * 30 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fixedOpexData} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 30, left: 8 }}>
                  <XAxis type="number" tickFormatter={(v: number) => (v / 1e3).toFixed(0) + "k"} tick={{ fontSize: 11 }} domain={[0, Math.max(...fixedOpexData.map(d => d.value)) * 1.1]} label={{ value: `Cost (${currency})`, position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 10, textTransform: "capitalize" } as object} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#f72585" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          )}
        </div>
      </div>

      {/* Variable OPEX */}
      {varOpexData.length > 0 && (
        <div className="card">
          <h2>Annual variable OPEX (in {currency})</h2>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
            <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
              <tbody>
                {varOpexData.map((d) => (
                  <tr key={d.name}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
                ))}
                <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12 }}>Total</td><td className="number">{fmt(results.variable_opex.total)}</td></tr>
              </tbody>
            </table>
            <DownloadableChart filename="variable_opex" height={varOpexData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={varOpexData} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 30, left: 8 }}>
                  <XAxis type="number" tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"} tick={{ fontSize: 11 }} domain={[0, Math.max(...varOpexData.map(d => d.value)) * 1.1]} label={{ value: `Annual cost (${currency})`, position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#666" } }} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmt(Number(v))} />
                  <Bar dataKey="value" fill="#7209b7" radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </DownloadableChart>
          </div>
        </div>
      )}

      {/* Revenue */}
      {results.revenue.total > 0 && (
        <div className="card">
          <h2>Annual revenue (in {currency})</h2>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
            <table style={{ flexShrink: 0, width: "50%", fontSize: 13 }}>
              <tbody>
                {revenueData.map((d) => (
                  <tr key={d.name}><td style={{ paddingRight: 12, whiteSpace: "nowrap" }}>{d.name}</td><td className="number" style={{ whiteSpace: "nowrap" }}>{fmt(d.value)}</td></tr>
                ))}
                <tr style={{ fontWeight: 700 }}><td style={{ paddingRight: 12 }}>Total</td><td className="number">{fmt(results.revenue.total)}</td></tr>
              </tbody>
            </table>
            {revenueData.length > 0 && (
              <DownloadableChart filename="revenue" height={revenueData.length * 40 + 60} style={{ flex: 1, minWidth: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={revenueData} layout="vertical" barCategoryGap={0} margin={{ top: 4, right: 16, bottom: 30, left: 8 }}>
                    <XAxis type="number" tickFormatter={(v: number) => (v / 1e6).toFixed(1) + "M"} tick={{ fontSize: 11 }} domain={[0, Math.max(...revenueData.map(d => d.value)) * 1.1]} label={{ value: `Annual revenue (${currency})`, position: "insideBottom", offset: -16, style: { fontSize: 12, fill: "#666" } }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => fmt(Number(v))} />
                    <Bar dataKey="value" fill="#4cc9f0" radius={[0, 4, 4, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </DownloadableChart>
            )}
          </div>
        </div>
      )}

      {/* Cash Flow Table */}
      {results.cash_flow.cash_flow && results.cash_flow.cash_flow.length > 0 && (
        <div className="card">
          <h2>Cash Flow (in {currency})</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ textAlign: "center" }}>
              <thead>
                <tr>
                  <th>Year</th><th>CAPEX</th><th>Revenue</th><th>Operating Costs</th>
                  <th>Depreciation</th><th>Gross Profit</th><th>Tax</th><th>Cash Flow</th><th>NPV</th>
                </tr>
              </thead>
              <tbody>
                {results.cash_flow.cash_flow.map((_val, yearIdx) => (
                  <tr key={yearIdx}>
                    <td>{yearIdx + 1}</td>
                    <td>{fmt(results.cash_flow.capital_cost_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.revenue_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.cash_cost_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.depreciation_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.gross_profit_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.tax_paid_array[yearIdx] ?? 0)}</td>
                    <td>{fmt(results.cash_flow.cash_flow[yearIdx] ?? 0)}</td>
                    <td>{fmt(cumulativeNPV[yearIdx])}</td>
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
