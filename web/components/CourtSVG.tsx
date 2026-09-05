import type { ReactNode } from "react";

/**
 * The ONE shared SVG badminton court (docs/plan.md: "one shared SVG court component")
 * reused by the match report, positioning heatmaps, and trajectory views. Do not fork
 * per-view court drawings — pass overlays as children instead.
 *
 * Geometry mirrors pipeline/src/synchro_pipeline/domain/court.py EXACTLY and must be
 * kept in sync with it (that module pins the coordinate contract):
 *   - origin at court centre, x across [-3.05, +3.05] m, y along [-6.70, +6.70] m
 *   - near side (broadcast camera side) is y < 0 and renders at the BOTTOM
 *   - "left" means x < 0 as seen from the broadcast camera behind the near court
 *
 * SVG user units are court metres x 10 (decimetres) for clean numbers: full court is
 * 61 x 134 units. Lines are 40 mm wide (0.4 units), stroked CENTRED on the canonical
 * line coordinates — the same convention court.py uses for its keypoint intersections,
 * so overlays projected through the pipeline's homographies land on the drawn lines.
 */

// keep in sync with domain/court.py (values are metres x SCALE)
const SCALE = 10;
const X_DOUBLES = 3.05 * SCALE; // 30.5
const X_SINGLES = 2.59 * SCALE; // 25.9
const Y_BASELINE = 6.7 * SCALE; // 67
const Y_SHORT_SERVICE = 1.98 * SCALE; // 19.8, from the net
const Y_DOUBLES_LONG_SERVICE = Y_BASELINE - 0.76 * SCALE; // 59.4, 0.76 m from the back
const LINE_W = 0.04 * SCALE; // 40 mm BWF line width, to scale

export const COURT_SVG_WIDTH = 2 * X_DOUBLES; // 61
export const COURT_SVG_HEIGHT = 2 * Y_BASELINE; // 134

/**
 * Court metres (domain/court.py frame) -> this SVG's user units.
 * Use for overlay children so pipeline outputs (player positions, shuttle landing
 * points, heatmap cells) can be plotted without re-deriving the transform.
 */
export function courtToSvg(xMetres: number, yMetres: number): { x: number; y: number } {
  return {
    x: (xMetres + X_DOUBLES / SCALE) * SCALE,
    y: (Y_BASELINE / SCALE - yMetres) * SCALE, // near side (y < 0) at the bottom
  };
}

export interface CourtSVGProps {
  /** Portrait only for now (broadcast-style, near court at the bottom). */
  orientation?: "portrait";
  /** "full" shows both halves; "half" shows the near half only (net at the top edge). */
  view?: "full" | "half";
  /** Draw the net as a dashed line on the full-court view. */
  showNet?: boolean;
  className?: string;
  /**
   * Overlay slot: children render in the SAME coordinate system (metres x 10, origin
   * top-left of the doubles court, near side at the bottom). Use courtToSvg().
   */
  children?: ReactNode;
}

/** Padding around the court so centred strokes and slight out-of-court overlays fit. */
const PAD = 1.5;

export default function CourtSVG({
  orientation = "portrait",
  view = "full",
  showNet = true,
  className,
  children,
}: CourtSVGProps) {
  void orientation; // single supported value; prop reserved for a future landscape mode
  const half = view === "half";
  const minY = half ? Y_BASELINE - PAD : -PAD;
  const viewH = half ? Y_BASELINE + 2 * PAD : COURT_SVG_HEIGHT + 2 * PAD;

  const line = {
    stroke: "var(--court-line, #f2f5f7)",
    strokeWidth: LINE_W,
  } as const;

  return (
    <svg
      viewBox={`${-PAD} ${minY} ${COURT_SVG_WIDTH + 2 * PAD} ${viewH}`}
      className={className}
      role="img"
      aria-label={half ? "Badminton court (near half)" : "Badminton court"}
    >
      {/* mat / surface */}
      <rect
        x={-PAD}
        y={minY}
        width={COURT_SVG_WIDTH + 2 * PAD}
        height={viewH}
        fill="var(--court-surface, #1e5f46)"
      />

      {/* doubles boundary: sidelines + baselines */}
      <rect x={0} y={0} width={COURT_SVG_WIDTH} height={COURT_SVG_HEIGHT} fill="none" {...line} />

      {/* singles sidelines, full length */}
      <line x1={X_DOUBLES - X_SINGLES} y1={0} x2={X_DOUBLES - X_SINGLES} y2={COURT_SVG_HEIGHT} {...line} />
      <line x1={X_DOUBLES + X_SINGLES} y1={0} x2={X_DOUBLES + X_SINGLES} y2={COURT_SVG_HEIGHT} {...line} />

      {/* short service lines, 1.98 m either side of the net */}
      <line
        x1={0}
        y1={Y_BASELINE - Y_SHORT_SERVICE}
        x2={COURT_SVG_WIDTH}
        y2={Y_BASELINE - Y_SHORT_SERVICE}
        {...line}
      />
      <line
        x1={0}
        y1={Y_BASELINE + Y_SHORT_SERVICE}
        x2={COURT_SVG_WIDTH}
        y2={Y_BASELINE + Y_SHORT_SERVICE}
        {...line}
      />

      {/* doubles long service lines, 0.76 m in from each baseline */}
      <line
        x1={0}
        y1={Y_BASELINE - Y_DOUBLES_LONG_SERVICE}
        x2={COURT_SVG_WIDTH}
        y2={Y_BASELINE - Y_DOUBLES_LONG_SERVICE}
        {...line}
      />
      <line
        x1={0}
        y1={Y_BASELINE + Y_DOUBLES_LONG_SERVICE}
        x2={COURT_SVG_WIDTH}
        y2={Y_BASELINE + Y_DOUBLES_LONG_SERVICE}
        {...line}
      />

      {/* centre lines: between each short service line and its baseline (court.py: T -> baseline centre) */}
      <line
        x1={X_DOUBLES}
        y1={0}
        x2={X_DOUBLES}
        y2={Y_BASELINE - Y_SHORT_SERVICE}
        {...line}
      />
      <line
        x1={X_DOUBLES}
        y1={Y_BASELINE + Y_SHORT_SERVICE}
        x2={X_DOUBLES}
        y2={COURT_SVG_HEIGHT}
        {...line}
      />

      {/* net (not a floor line — drawn dashed to distinguish); top edge in half view */}
      {showNet && (
        <line
          x1={-PAD}
          y1={Y_BASELINE}
          x2={COURT_SVG_WIDTH + PAD}
          y2={Y_BASELINE}
          stroke="var(--court-line, #f2f5f7)"
          strokeWidth={LINE_W / 2}
          strokeDasharray="1 1"
          opacity={0.8}
        />
      )}

      {/* overlay slot: heatmaps, trajectories, player markers */}
      <g data-slot="overlay">{children}</g>
    </svg>
  );
}
