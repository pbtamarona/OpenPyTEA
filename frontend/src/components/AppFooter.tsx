import { openExternal } from "../docs";

const PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S2352711026003080";

/** Bottom ribbon: the paper to cite, and who made the GUI. */
export default function AppFooter() {
  return (
    <footer className="app-footer">
      <span className="app-footer-cite">
        If you use OpenPyTEA, please cite:{" "}
        <a
          href={PAPER_URL}
          onClick={(e) => {
            e.preventDefault();
            openExternal(PAPER_URL);
          }}
          title="OpenPyTEA: An open-source Python toolkit for techno-economic assessment of chemical process plants and energy systems with economic sensitivity and uncertainty evaluation"
        >
          Tamarona, Vlugt &amp; Ramdin, <em>SoftwareX</em> 35, 102816 (2026) ↗
        </a>
      </span>
      <span className="app-footer-credit">GUI developed by Othonas A. Moultos</span>
    </footer>
  );
}
