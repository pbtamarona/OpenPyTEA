import { useRef, useCallback, useEffect, useState, type ReactNode } from "react";
import { saveFigure } from "../save";

interface Props {
  filename?: string;
  children: ReactNode;
  height: number | string;
  style?: React.CSSProperties;
  // Box constraint: charts render at most this wide, centered, instead of
  // stretching across the whole card. Pass null to fill the container.
  maxWidth?: number | null;
  // When given, the download button fetches the backend's matplotlib
  // rendering of this chart — the exact figure the library produces in
  // Jupyter — instead of rasterizing the on-screen SVG. Falls back to the
  // screen render if the request fails.
  serverPlot?: () => Promise<Blob>;
}

export default function DownloadableChart({
  filename = "chart", children, height, style, maxWidth = 620, serverPlot,
}: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  // Grid toggle, offered only when the chart draws a grid. Recharts adds
  // it after measuring its container, so watch for it rather than check once.
  const [hasGrid, setHasGrid] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  useEffect(() => {
    const el = chartRef.current;
    if (!el) return;
    const check = () => setHasGrid(!!el.querySelector(".recharts-cartesian-grid"));
    check();
    const obs = new MutationObserver(check);
    obs.observe(el, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, []);

  const downloadScreenRender = useCallback(() => {
    if (!chartRef.current) return;

    // Get all SVGs inside the chart area (button is outside this div)
    const svgs = chartRef.current.querySelectorAll("svg");
    let svg: SVGSVGElement | null = null;
    let maxArea = 0;
    svgs.forEach((s) => {
      const r = s.getBoundingClientRect();
      if (r.width * r.height > maxArea) {
        maxArea = r.width * r.height;
        svg = s as SVGSVGElement;
      }
    });
    if (!svg) return;

    const clone = (svg as SVGSVGElement).cloneNode(true) as SVGSVGElement;
    const w = (svg as SVGSVGElement).getBoundingClientRect().width;
    const h = (svg as SVGSVGElement).getBoundingClientRect().height;
    clone.setAttribute("width", String(w));
    clone.setAttribute("height", String(h));
    clone.setAttribute("viewBox", `0 0 ${w} ${h}`);

    // Deep-inline all computed styles so the PNG looks identical to the screen
    const allSource = (svg as SVGSVGElement).querySelectorAll("*");
    const allTarget = clone.querySelectorAll("*");
    allSource.forEach((srcEl, i) => {
      const tgtEl = allTarget[i] as HTMLElement | undefined;
      if (!tgtEl) return;
      const cs = window.getComputedStyle(srcEl);
      const props = [
        "fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-opacity",
        "fill-opacity", "opacity",
        "font-size", "font-family", "font-weight", "font-style",
        "text-anchor", "dominant-baseline", "alignment-baseline",
        "letter-spacing", "word-spacing", "text-decoration",
        "transform", "visibility", "display", "color",
      ];
      for (const p of props) {
        const v = cs.getPropertyValue(p);
        if (v && v !== "none" && v !== "normal" && v !== "visible" && v !== "0px"
          && v !== "inline" && v !== "auto") {
          tgtEl.style.setProperty(p, v);
        }
      }
      for (const p of ["fill", "stroke", "font-size", "font-family", "font-weight", "text-anchor", "dominant-baseline", "transform"]) {
        const v = cs.getPropertyValue(p);
        if (v) tgtEl.style.setProperty(p, v);
      }
    });

    // A hidden grid stays hidden in the PNG (its display:none isn't inlined)
    if (!showGrid) clone.querySelectorAll(".recharts-cartesian-grid").forEach((g) => g.remove());

    // White background
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", "100%");
    rect.setAttribute("height", "100%");
    rect.setAttribute("fill", "white");
    clone.insertBefore(rect, clone.firstChild);

    const svgData = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      const scale = 3;
      const canvas = document.createElement("canvas");
      canvas.width = w * scale;
      canvas.height = h * scale;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      canvas.toBlob((png) => { if (png) saveFigure(png, filename); }, "image/png");
    };
    img.src = url;
  }, [filename, showGrid]);

  const download = useCallback(async () => {
    if (serverPlot) {
      setBusy(true);
      try {
        const blob = await serverPlot();
        await saveFigure(blob, filename);
        return;
      } catch (e) {
        console.warn("Publication figure failed, falling back to screen render:", e);
      } finally {
        setBusy(false);
      }
    }
    downloadScreenRender();
  }, [serverPlot, filename, downloadScreenRender]);

  return (
    <div className="chart-panel" style={{
      position: "relative", height,
      ...(maxWidth != null ? { maxWidth, marginLeft: "auto", marginRight: "auto" } : {}),
      ...style,
    }}>
      {hasGrid && (
        <button
          className={`chart-tool${showGrid ? "" : " off"}`}
          style={{ right: 34 }}
          onClick={() => setShowGrid((v) => !v)}
          title={showGrid ? "Hide grid lines" : "Show grid lines"}
          aria-pressed={showGrid}
        >
          #
        </button>
      )}
      <button
        className="chart-tool"
        style={{ right: 6, fontSize: 17, cursor: busy ? "wait" : "pointer", opacity: busy ? 1 : undefined }}
        onClick={download}
        disabled={busy}
        title={serverPlot
          ? "Download publication figure (rendered by the OpenPyTEA library, as in Jupyter)"
          : "Download as PNG"}
      >
        {busy ? "⏳" : "⬇"}
      </button>
      {/* position: relative so absolutely placed overlays stay inside the padding */}
      <div ref={chartRef} className={showGrid ? undefined : "no-grid"} style={{ position: "relative", width: "100%", height: "100%" }}>
        {children}
      </div>
    </div>
  );
}
