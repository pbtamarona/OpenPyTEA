/** Number formatting shared by chart axes and tooltips (en-US separators). */

/** Axis tick with thousand separators: whole numbers from 1,000 up, up to
    2 decimals below that, 3 significant digits for small fractions. */
export function fmtTick(v: number): string {
  if (!Number.isFinite(v)) return "";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 1) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return v.toLocaleString("en-US", { maximumSignificantDigits: 3 });
}

/** Tooltip readout: separators, `digits` decimals below 1,000. */
export function fmtValue(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "";
  return v.toLocaleString("en-US", {
    minimumFractionDigits: Math.abs(v) >= 1000 ? 0 : digits,
    maximumFractionDigits: Math.abs(v) >= 1000 ? 0 : digits,
  });
}

/** Round tick positions (steps of 1, 2 or 5 × 10^n) inside [min, max], so
    an axis reads 0, 500,000,000, 1,000,000,000 rather than starting at an
    arbitrary data minimum like -1,157,636,956. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return [min];
  const raw = span / Math.max(1, count - 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(12))); // drop float noise (0.30000000000000004)
  }
  return ticks;
}

// Unicode sub/superscripts; a character without one is kept as it is
const SUB: Record<string, string> = Object.fromEntries(
  [..."0123456789+-=()aehijklmnoprstuvx"].map((c, i) => [c, [..."₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"][i]]),
);
const SUP: Record<string, string> = Object.fromEntries(
  [..."0123456789+-=()abcdefghijklmnoprstuvwxyz"].map((c, i) => [c, [..."⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"][i]]),
);
const sub = (s: string) => [...s].map((c) => SUB[c] ?? c).join("");
const sup = (s: string) => [...s].map((c) => SUP[c] ?? c).join("");

const ESCAPED_DOLLAR = "\u0000";

/** Plain text of one $...$ math segment (the library's mathtext). */
function mathToText(m: string): string {
  let s = m;
  for (let prev = ""; prev !== s;) {
    prev = s;
    s = s.replace(/\\mathrm\{([^{}]*)\}/g, "$1");
  }
  return s
    .replace(/\\cdot/g, "·").replace(/\\pm/g, "±").replace(/\\times/g, "×").replace(/\\,/g, " ")
    .replace(/_\{([^{}]*)\}|_([+-]?\d+|.)/g, (_, a, b) => sub(a ?? b))
    .replace(/\^\{([^{}]*)\}|\^([+-]?\d+|.)/g, (_, a, b) => sup(a ?? b))
    .replace(/\\%/g, "%").replace(/[\\{}]/g, "");
}

/** A library label (matplotlib mathtext: "H$_{\mathrm{2}}$O price / [USD$\cdot$unit$^{-1}$]")
    as plain text for the charts: "H₂O price / [USD·unit⁻¹]". */
export function texToText(s: string): string {
  return s
    .replace(/\\\$/g, ESCAPED_DOLLAR)
    .split(/(\$[^$]*\$)/)
    .map((p, i) => (i % 2 ? mathToText(p.slice(1, -1)) : p.replace(/\\%/g, "%").replace(/\\_/g, "_")))
    .join("")
    .replace(new RegExp(ESCAPED_DOLLAR, "g"), "$");
}

/** A name as typed ("CO_2_tax", "Fe^{3+}") shown the way the library labels
    it in plots: _2 / _{in} subscript, ^3 / ^{3+} superscript, any other
    underscore a space, first letter capitalised -> "CO₂ tax", "Fe³⁺". */
export function nameLabel(name: string): string {
  const s = name
    .split(/(\$[^$]*\$)/)
    .map((p, i) => (i % 2 ? p : p.replace(
      /_(\{[^{}]*\}|\d+)|\^(\{[^{}]*\}|[+-]?\d+|[+-])|_/g,
      (_, a?: string, b?: string) => (a ? sub(a.replace(/[{}]/g, "")) : b ? sup(b.replace(/[{}]/g, "")) : " "),
    )))
    .join("");
  const text = texToText(s);
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** A cost factor to up to 4 significant digits, without float noise
    (0.30000000000000004 -> 0.3) */
export const fmtFactor = (v: number) => String(Number(v.toPrecision(4)));

/** Display name of a fixed OPEX item key ("rent_of_land" -> "Rent of Land") */
export function fixedOpexLabel(k: string): string {
  return ({
    rent_of_land: "Rent of Land", rnd: "R&D", taxes_insurance: "Taxes & Insurance",
    patents_royalties: "Patents & Royalties", distribution_selling: "Distribution & Selling",
    interest_working_capital: "Interest on Working Capital",
  } as Record<string, string>)[k] ?? k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Chart rows as a percentage of their total (the library's pct=True) */
export function asShare<T extends { value: number }>(rows: T[]): T[] {
  const total = rows.reduce((s, r) => s + r.value, 0);
  return rows.map((r) => ({ ...r, value: total ? (r.value / total) * 100 : 0 }));
}

/** A percentage for an axis tick or tooltip: 12.5% */
export const fmtPct = (v: number) => `${Number(v.toFixed(1))}%`;
