import Link from "next/link";
import {
  ApiError,
  ApiUnreachableError,
  listMatches,
  type Match,
} from "@/lib/api";

/**
 * Match list — the dashboard entry point (plan build order step 1 leads to the
 * match report; this page is just the index into it). Server component: fetches
 * from the FastAPI service per-request, degrades gracefully when it is not running.
 */

function playerLine(match: Match): string {
  const a = match.players.A?.name ?? "Player A";
  const b = match.players.B?.name ?? "Player B";
  return `${a} vs ${b}`;
}

function gameScores(match: Match): string {
  if (match.sets.length === 0) return "no games processed";
  return match.sets.map((g) => `${g.final_score.a}-${g.final_score.b}`).join(", ");
}

export default async function HomePage() {
  let matches: Match[];
  try {
    matches = await listMatches();
  } catch (err) {
    const detail =
      err instanceof ApiUnreachableError
        ? "The API is not running. Start it with: uv run uvicorn synchro_api.main:app --reload"
        : err instanceof ApiError
          ? `The API answered ${err.status} for ${err.url}.`
          : "Unexpected error talking to the API.";
    return (
      <>
        <h1>Matches</h1>
        <div className="card notice">
          <strong>API not running</strong>
          <p className="muted">{detail}</p>
        </div>
      </>
    );
  }

  if (matches.length === 0) {
    return (
      <>
        <h1>Matches</h1>
        <div className="card">
          <p>No matches yet.</p>
          <p className="muted">
            Upload a match video through the pipeline (docs/plan.md, Phase 0 walking
            skeleton) and it will appear here once processed.
          </p>
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Matches</h1>
      <ul className="match-list">
        {matches.map((m) => (
          <li key={m.match_id} className="card">
            <Link href={`/matches/${encodeURIComponent(m.match_id)}`}>
              {playerLine(m)}
            </Link>
            <div className="muted">
              {m.discipline} · {gameScores(m)}
            </div>
          </li>
        ))}
      </ul>
    </>
  );
}
