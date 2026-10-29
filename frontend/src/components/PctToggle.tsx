interface Props {
  pct: boolean;
  onChange: (pct: boolean) => void;
}

/** Switch a breakdown chart between values and percentage of the total. */
export default function PctToggle({ pct, onChange }: Props) {
  return (
    <div className="seg-toggle" role="group" aria-label="Show values or percentages">
      <button className={pct ? "" : "active"} onClick={() => onChange(false)}>Value</button>
      <button className={pct ? "active" : ""} onClick={() => onChange(true)}>%</button>
    </div>
  );
}
