import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getHelp } from "../help/help";
import { openDocs } from "../docs";

interface Props {
  /** Key into the help map (help/help.ts), e.g. "plant.interest_rate". */
  id: string;
}

/** A small (?) that shows the input's documentation on hover/focus and
    opens the matching docs section on click. Renders nothing when the
    key has no help text. */
export default function HelpTip({ id }: Props) {
  const help = getHelp(id);
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean } | null>(null);
  if (!help) return null;

  const show = () => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    // Fixed-position bubble in a portal: never clipped by a scrolling table
    // or card. Above the icon unless it's near the top of the window.
    const above = r.top > 180;
    const left = Math.min(Math.max(r.left + r.width / 2, 170), window.innerWidth - 170);
    setPos({ left, top: above ? r.top - 8 : r.bottom + 8, above });
  };

  return (
    <>
      <span
        ref={ref}
        className="help-tip"
        role="button"
        tabIndex={0}
        aria-label={help.text}
        onMouseEnter={show}
        onMouseLeave={() => setPos(null)}
        onFocus={show}
        onBlur={() => setPos(null)}
        onClick={(e) => {
          // inside a <label>: don't focus/toggle the associated input
          e.preventDefault();
          e.stopPropagation();
          if (help.doc) openDocs(help.doc);
        }}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && help.doc) {
            e.preventDefault();
            openDocs(help.doc);
          }
        }}
      >
        ?
      </span>
      {pos &&
        createPortal(
          <div
            className={`help-bubble ${pos.above ? "above" : "below"}`}
            style={{ left: pos.left, top: pos.top }}
            role="tooltip"
          >
            {help.text}
            {help.doc && <div className="help-bubble-more">Click ? to open the documentation</div>}
          </div>,
          document.body,
        )}
    </>
  );
}
