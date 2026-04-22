import { useRef, useCallback, type ReactNode } from "react";

interface Props {
  filename?: string;
  children: ReactNode;
  height: number | string;
  style?: React.CSSProperties;
}

export default function DownloadableChart({ filename = "chart", children, height, style }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  const download = useCallback(() => {
    const svg = containerRef.current?.querySelector(".recharts-wrapper svg") as SVGSVGElement | null;
    if (!svg) return;

    const clone = svg.cloneNode(true) as SVGSVGElement;
    const w = svg.getBoundingClientRect().width;
    const h = svg.getBoundingClientRect().height;
    clone.setAttribute("width", String(w));
    clone.setAttribute("height", String(h));
    clone.setAttribute("viewBox", `0 0 ${w} ${h}`);

    // Deep-inline all computed styles so the PNG looks identical to the screen
    const allSource = svg.querySelectorAll("*");
    const allTarget = clone.querySelectorAll("*");
    allSource.forEach((srcEl, i) => {
      const tgtEl = allTarget[i] as HTMLElement | undefined;
      if (!tgtEl) return;
      const cs = window.getComputedStyle(srcEl);
      // Copy every property that matters for SVG rendering
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
      // Always copy these even if "default-looking"
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
    <div ref={containerRef} style={{ position: "relative", height, ...style }}>
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
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </button>
      {children}
    </div>
  );
}
