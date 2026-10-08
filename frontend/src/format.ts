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
