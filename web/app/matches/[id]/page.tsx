import CourtSVG from "@/components/CourtSVG";
import {
  ApiError,
  ApiUnreachableError,
  getMatch,
  listRallies,
  type Match,
  type Rally,
} from "@/lib/api";

/**
 * Match report placeholder (plan build order step 1: score worm, headline tiles,
 * shot distribution, rally table — every stat click-through to clips). Phase 0
 * renders the skeleton only: header, confidence badge, rally table, and the shared
 * CourtSVG that heatmap/trajectory overlays will plug into later.
 */

function confidenceBadge(match: Match) {
  const pct = match.quality_report.pct_rallies_clean;
  if (pct == null) {
    return <span className="badge">confidence: pending</span>;
  }
  // Mirrors the plan's honest-abstention posture: low-confidence rallies are
  // excluded and flagged, and the share of clean rallies is surfaced up front.
  const cls = pct >= 90 ? "ok" : pct >= 70 ? "warn" : "bad";
  return (
    <span className={`badge ${cls}`}>{pct.toFixed(0)}% rallies clean</span>
  );
}

function scoreBefore(rally: Rally): string {
  if (!rally.score_before) return "—";
  return `${rally.score_before.a}-${rally.score_before.b}`;
}

function serveCell(rally: Rally): string {
  if (!rally.server) return "—";
  return rally.serve_type ? `${rally.server} (${rally.serve_type})` : rally.server;
}

function lengthCell(rally: Rally): string {
  const strokes = rally.n_strokes != null ? `${rally.n_strokes} shots` : null;
  const secs = `${rally.duration_s.toFixed(0)}s`;
  return strokes ? `${strokes} · ${secs}` : secs;
}

export default async function MatchPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let match: Match;
  let rallies: Rally[];
  try {
    [match, rallies] = await Promise.all([getMatch(id), listRallies(id)]);
  } catch (err) {
    const detail =
      err instanceof ApiUnreachableError
        ? "The API is not running. Start it with: uv run uvicorn synchro_api.main:app --reload"
        : err instanceof ApiError && err.status === 404
          ? `No match with id "${id}".`
          : err instanceof ApiError
            ? `The API answered ${err.status}.`
            : "Unexpected error talking to the API.";
    return (
      <>
        <h1>Match report</h1>
        <div className="card notice">
          <strong>Could not load match</strong>
          <p className="muted">{detail}</p>
        </div>
      </>
    );
  }

  const nameA = match.players.A?.name ?? "Player A";
  const nameB = match.players.B?.name ?? "Player B";

  return (
    <>
      <h1>
        {nameA} vs {nameB}
      </h1>
      <p>
        <span className="score-line">
          {match.sets.length > 0
            ? match.sets
                .map((g) => `${g.final_score.a}-${g.final_score.b}`)
                .join("  ")
            : "no games processed"}
        </span>{" "}
        {confidenceBadge(match)}
      </p>
      <p className="muted">
        {match.discipline} · pipeline {match.pipeline_version}
        {match.source_meta.broadcaster_guess
          ? ` · ${match.source_meta.broadcaster_guess}`
          : ""}
      </p>

      <h2>Rallies</h2>
      {rallies.length === 0 ? (
        <div className="card">
          <p className="muted">No rallies extracted for this match yet.</p>
        </div>
      ) : (
        <div className="card">
          <table className="rally-table">
            <thead>
              <tr>
                <th>Game</th>
                <th>Score before</th>
                <th>Serve</th>
                <th>Length</th>
                <th>Winner</th>
                <th>Clip</th>
              </tr>
            </thead>
            <tbody>
              {rallies.map((r) => (
                <tr key={r.rally_id}>
                  <td>{r.set_no}</td>
                  <td>{scoreBefore(r)}</td>
                  <td>{serveCell(r)}</td>
                  <td>{lengthCell(r)}</td>
                  <td>{r.winner ?? "—"}</td>
                  <td>
                    {/* stat -> clip loop is the product (plan); player lands Phase 1 */}
                    {r.clip_uri ? (
                      <a href={r.clip_uri}>watch</a>
                    ) : (
                      <span className="muted">clip pending</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h2>Court views</h2>
      <div className="card">
        <p className="muted">
          Positioning heatmaps and trajectories arrive in later phases — all of them
          overlay this one shared court component.
        </p>
        <div className="court-wrap">
          <CourtSVG view="full" />
        </div>
      </div>
    </>
  );
}
