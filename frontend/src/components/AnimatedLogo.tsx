interface Props {
  animating: boolean;
  className?: string;
}

const LAYERS = "/logo-layers";

/**
 * The OpenPyTEA cup. While `animating`, the logo is drawn from separate
 * layers (public/logo-layers/) so the steam can waver and the icons drift;
 * otherwise it is the plain static logo.
 */
export default function AnimatedLogo({ animating, className = "" }: Props) {
  if (!animating) {
    return <img src="/logo.png" alt="OpenPyTEA" className={className} />;
  }

  return (
    <div className={`animated-logo ${className}`} role="img" aria-label="OpenPyTEA">
      <svg className="animated-logo-defs" aria-hidden="true" focusable="false">
        <filter id="steam-waver" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" baseFrequency="0.006 0.022" numOctaves="1" seed="7" result="noise">
            <animate
              attributeName="baseFrequency"
              values="0.006 0.022;0.008 0.03;0.006 0.022"
              dur="5s"
              repeatCount="indefinite"
            />
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="6" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
      <img src={`${LAYERS}/logo-cup.png`} alt="" />
      <img src={`${LAYERS}/logo-steam.png`} alt="" className="animated-logo-steam" />
      <img src={`${LAYERS}/logo-dollar.png`} alt="" className="animated-logo-icon animated-logo-dollar" />
      <img src={`${LAYERS}/logo-molecule.png`} alt="" className="animated-logo-icon animated-logo-molecule" />
      <img src={`${LAYERS}/logo-flask.png`} alt="" className="animated-logo-icon animated-logo-flask" />
    </div>
  );
}
