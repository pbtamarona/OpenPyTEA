import { useEffect, useRef, useState } from "react";

/** One plant's process flow diagram: a panel that minimizes to its title bar
    and maximizes to fill the screen (Esc or the button closes it). */
export default function PfdWindow({ title, image, busy, onChange }: {
  title: string;
  image: string | null;
  busy: boolean;
  onChange: (image: string | null) => void;
}) {
  const [minimized, setMinimized] = useState(false);
  const [maximized, setMaximized] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!maximized) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMaximized(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [maximized]);

  const pick = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onChange(reader.result as string);
    reader.readAsDataURL(file);
  };

  const [dragging, setDragging] = useState(false);
  const isFileDrag = (e: React.DragEvent) => e.dataTransfer.types.includes("Files");
  const drop = {
    onDragOver: (e: React.DragEvent) => {
      if (busy || !isFileDrag(e)) return;
      e.preventDefault();
      setDragging(true);
    },
    onDragLeave: (e: React.DragEvent) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) pick(file);
    },
  };

  const bar = (
    <div className="pfd-bar">
      <strong>{title}</strong>
      <div className="pfd-actions">
        <button className="btn-secondary btn-small" disabled={busy} onClick={() => fileRef.current?.click()}>
          {image ? "Replace…" : "Upload…"}
        </button>
        {image && <button className="btn-secondary btn-small" disabled={busy} onClick={() => onChange(null)}>Remove</button>}
        {!maximized && (
          <button className="btn-secondary btn-small" title={minimized ? "Expand" : "Minimize"} aria-label={minimized ? "Expand" : "Minimize"} onClick={() => setMinimized(!minimized)}>
            {minimized ? "▢" : "–"}
          </button>
        )}
        {image && (
          <button className="btn-secondary btn-small" title={maximized ? "Restore (Esc)" : "Maximize"} aria-label={maximized ? "Restore" : "Maximize"} onClick={() => setMaximized(!maximized)}>
            {maximized ? "✕" : "⛶"}
          </button>
        )}
      </div>
      <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { pick(e.target.files?.[0]); e.target.value = ""; }} />
    </div>
  );

  const body = image
    ? <img src={image} alt={`${title} process flow diagram`} onDoubleClick={() => setMaximized(!maximized)} />
    : <p className="pfd-empty">No process flow diagram yet. Upload or drop an image here (PNG, JPG, SVG…).</p>;

  return (
    <div className={`pfd-window${dragging ? " pfd-drag" : ""}`} {...drop}>
      {bar}
      {!minimized && !maximized && <div className="pfd-body">{body}</div>}
      {maximized && (
        <div className="modal-overlay" onClick={() => setMaximized(false)}>
          <div className={`pfd-window pfd-max${dragging ? " pfd-drag" : ""}`} onClick={(e) => e.stopPropagation()}>
            {bar}
            <div className="pfd-body">{body}</div>
          </div>
        </div>
      )}
    </div>
  );
}
