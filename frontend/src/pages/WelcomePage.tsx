interface Props {
  onContinue: () => void;
}

export default function WelcomePage({ onContinue }: Props) {
  return (
    <div className="welcome">
      <div className="welcome-card">
        <img src="/logo.png" alt="OpenPyTEA" className="welcome-logo" />
        <h1 className="welcome-title">OpenPyTEA</h1>
        <p className="welcome-subtitle">
          Open-source toolkit for techno-economic assessment of chemical and energy systems
        </p>

        <ul className="welcome-features">
          <li>
            <strong>Equipment costing</strong> — estimate purchased and installed costs from
            built-in correlations and CEPCI escalation.
          </li>
          <li>
            <strong>Plant economics</strong> — assemble CAPEX/OPEX and run discounted cash flow
            with NPV, IRR, payback, and break-even price.
          </li>
          <li>
            <strong>Sensitivity & tornado</strong> — vary parameters one at a time to see which
            drivers matter most.
          </li>
          <li>
            <strong>Monte Carlo</strong> — quantify uncertainty across thousands of scenarios
            with histograms and summary statistics.
          </li>
          <li>
            <strong>Compare plants</strong> — load multiple configurations side-by-side to
            evaluate design trade-offs.
          </li>
        </ul>

        <p className="welcome-hint">
          Tip: try <em>Examples</em> in the top bar to load a pre-built case study.
        </p>

        <button className="btn-primary welcome-cta" onClick={onContinue}>
          Get Started →
        </button>
      </div>
    </div>
  );
}
