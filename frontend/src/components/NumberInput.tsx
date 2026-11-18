import { useState } from "react";

const show = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 10 });

/** A number box that reads 2,352,000 and shows the plain number while typing;
    commas typed or pasted are accepted. */
export default function NumberInput({ value, onChange, style, title }: {
  value: number;
  onChange: (v: number) => void;
  style?: React.CSSProperties;
  title?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="text" inputMode="decimal" title={title} style={style}
      value={draft ?? show(value)}
      onFocus={() => setDraft(String(value))}
      onChange={(e) => {
        setDraft(e.target.value);
        const v = Number(e.target.value.replace(/,/g, ""));
        if (e.target.value.trim() !== "" && Number.isFinite(v)) onChange(v);
      }}
      onBlur={() => setDraft(null)}
      onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
    />
  );
}
