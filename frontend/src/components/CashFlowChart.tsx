import { useEffect, useState } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Legend,
} from "recharts";
import DownloadableChart from "./DownloadableChart";
import { fetchPlotPng, runCashFlow, type CashFlowCurve } from "../api/client";
import type { PlantInput } from "../types";

interface Props {
  /** Plants to show: the open plant (include_active, default) and/or saved
      plant inputs, as for the other analysis requests */
  body: { extra_plants?: PlantInput[]; include_active?: boolean };
  /** Refetch when this changes (e.g. new results) */
  refreshKey?: unknown;
  /** Line colour of each curve, in the order of the request */
  colors: string[];
  currency: string;
  height?: number;
  style?: React.CSSProperties;
}

const fmt = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

/** Cumulative cash flow diagram (analysis.cash_flow_data): debt shaded,
    break-even marked; the download is the library's plot_cash_flow. */
export default function CashFlowChart({ body, refreshKey, colors, currency, height = 300, style }: Props) {
  const [curves, setCurves] = useState<CashFlowCurve[]>([]);
  const [error, setError] = useState<string | null>(null);
  const bodyKey = JSON.stringify(body);

  useEffect(() => {
    let stale = false;
    runCashFlow(body)
      .then((r) => { if (!stale) { setCurves(r.curves); setError(null); } })
      .catch((e: unknown) => { if (!stale) { setCurves([]); setError(e instanceof Error ? e.message : "Cash flow failed"); } });
    return () => { stale = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bodyKey, refreshKey]);

  if (curves.length === 0) {
    return <p style={{ ...style, color: "var(--text-muted)", fontSize: 13 }}>{error ?? "Loading the cash flow diagram…"}</p>;
  }

  // One row per year: curve i's cumulative position (c<i>) and its part
  // below zero (d<i>, shaded as debt)
  const nYears = Math.max(...curves.map((c) => c.years.length));
  const data = Array.from({ length: nYears }, (_, year) => {
    const row: Record<string, number> = { year };
    curves.forEach((c, i) => {
      if (year < c.cumulative.length) {
        row[`c${i}`] = c.cumulative[year];
        row[`d${i}`] = Math.min(c.cumulative[year], 0);
      }
    });
    return row;
  });
  const color = (i: number) => colors[i % colors.length];

  return (
    <DownloadableChart filename="cash_flow" serverPlot={(opts) => fetchPlotPng("/plots/cash-flow", body, opts)} height={height} style={style}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 42, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" type="number" domain={[0, nYears - 1]} tick={{ fontSize: 12 }} label={{ value: "Time (years)", position: "insideBottom", offset: -28, style: { fontSize: 14, fill: "#666" } }} />
          <YAxis tickFormatter={(v: number) => (v / 1e6).toFixed(0) + "M"} tick={{ fontSize: 12 }} width={56} label={{ value: `Cumulative cash flow (${currency})`, angle: -90, position: "insideLeft", offset: 0, style: { fontSize: 12, fill: "#666", textAnchor: "middle" } }} />
          <Tooltip formatter={(v) => fmt(Number(v))} labelFormatter={(y) => `Year ${y}`} />
          {curves.length > 1 && <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />}
          <ReferenceLine y={0} stroke="var(--text-muted)" />
          {curves.map((c, i) => c.breakeven_year !== null && (
            <ReferenceLine
              key={`be${i}`} x={c.breakeven_year} stroke={color(i)} strokeDasharray="4 3"
              label={curves.length === 1 ? { value: `Break-even ${c.breakeven_year.toFixed(1)} y`, position: "insideTopLeft", fontSize: 11, fill: "#888" } : undefined}
            />
          ))}
          {curves.map((_, i) => (
            <Area key={`d${i}`} dataKey={`d${i}`} type="linear" fill={color(i)} fillOpacity={0.2} stroke="none" isAnimationActive={false} tooltipType="none" legendType="none" />
          ))}
          {curves.map((c, i) => (
            <Line key={`c${i}`} dataKey={`c${i}`} name={c.plant} type="linear" stroke={color(i)} strokeWidth={2} dot={false} isAnimationActive={false} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </DownloadableChart>
  );
}
