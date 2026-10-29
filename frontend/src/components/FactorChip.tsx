import { useState } from "react";
import { fmtFactor } from "../format";

interface Props {
  value: number;
  /** Set by the user rather than the default */
  custom: boolean;
  /** Can't be edited (the component's value is set directly) */
  locked?: string;
  disabled?: boolean;
  /** "amount": a cost in the value column rather than a factor */
  variant?: "factor" | "amount";
  /** New value, or null to go back to the default */
  onCommit: (v: number | null) => void;
}

const fmtAmount = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 2 });

/** A cost factor (or amount) that edits in place: click to edit, Enter or
    click away to apply, Esc to cancel, empty to restore the default. */
export default function FactorChip({ value, custom, locked, disabled, variant = "factor", onCommit }: Props) {
  const [draft, setDraft] = useState<string | null>(null);
  const amount = variant === "amount";
  const show = amount ? fmtAmount : fmtFactor;

  if (locked) {
    return <span className="factor-chip locked" title={locked}>fixed</span>;
  }
  if (draft === null) {
    return (
      <button
        className={`${amount ? "value-edit" : "factor-chip"}${custom ? " custom" : ""}`}
        disabled={disabled}
        title={amount
          ? `Click to set this amount directly${custom ? " (set; clear it to compute it again)" : ""}`
          : `Click to edit${custom ? " (custom; clear it to use the default)" : " (default)"}`}
        onClick={() => setDraft(amount ? String(Math.round(value * 100) / 100) : fmtFactor(value))}
      >
        {show(value)}
      </button>
    );
  }

  const commit = () => {
    setDraft(null);
    const t = draft.trim();
    if (t === "") { if (custom) onCommit(null); return; }
    const v = Number(t);
    if (Number.isFinite(v) && v >= 0 && show(v) !== show(value)) onCommit(v);
  };
  return (
    <input
      className="factor-chip-input"
      style={amount ? { width: 130, textAlign: "right" } : undefined}
      type="number" min={0} step="any" autoFocus
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") setDraft(null);
      }}
    />
  );
}
