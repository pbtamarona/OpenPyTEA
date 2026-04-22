import { useRef, useCallback, type ReactNode } from "react";

interface Props {
  filename?: string;
  children: ReactNode;
  height: number | string;
  style?: React.CSSProperties;
}

export default function DownloadableChart({ filename = "chart", children, height, style }: Props) {
  const chartRef = useRef<HTMLDivElement>(null);

  const download = useCallback(() => {
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

      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = `${filename}.png`;
      a.click();
    };
    img.src = url;
  }, [filename]);

  return (
    <div style={{ position: "relative", height, ...style }}>
      <button
        onClick={download}
        title="Download as PNG"
        style={{
          position: "absolute", top: 4, right: 4, zIndex: 10,
          background: "none", border: "none", cursor: "pointer",
          opacity: 0.4, padding: 4, lineHeight: 1, fontSize: 18,
        }}
        onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }}
        onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.4"; }}
      >
        &#11015;
      </button>
      <div ref={chartRef} style={{ width: "100%", height: "100%" }}>
        {children}
      </div>
    </div>
  );
}
