/**
 * Typed fetch helpers against the Synchronicity FastAPI service (api/src/synchro_api).
 *
 * The interfaces below mirror the pipeline's Pydantic contracts in
 * pipeline/src/synchro_pipeline/schemas/records.py (MatchRecord / RallyRecord /
 * GameSummary / QualityReport and friends) — that file is the source of truth
 * per docs/plan.md ("FastAPI + Pydantic v2, shared schema models with the pipeline").
 *
 * COORDINATION NOTE: the API service is being scaffolded in parallel; exact route
 * paths and field names may drift until Phase 0 lands. When they do, update THIS
 * file only — pages consume these types, never raw JSON.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// --- shared shapes (records.py mirrors) -------------------------------------------

export type PlayerRef = "A" | "B";
export type Side = "near" | "far";
export type ServeType = "short" | "long";
export type Discipline = "MS" | "WS" | "MD" | "WD" | "XD";

export interface ScorePair {
  a: number;
  b: number;
}

export interface PlayerInfo {
  name: string | null;
  handedness: "left" | "right" | null;
}

export interface GameSummary {
  set_no: number; // game number; "set" follows ShuttleSet naming (records.py)
  final_score: ScorePair;
  rally_ids: string[];
}

export interface QualityReport {
  pct_rallies_clean: number | null;
  pct_strokes_classified: number | null;
  flags_summary: Record<string, number>;
}

export interface SourceMeta {
  width: number;
  height: number;
  fps_effective: number;
  duration_s: number;
  broadcaster_guess: string | null;
}

/** Mirrors records.MatchRecord (list endpoints may return a subset). */
export interface Match {
  match_id: string;
  source_meta: SourceMeta;
  discipline: Discipline;
  players: Partial<Record<PlayerRef, PlayerInfo>>;
  first_server: PlayerRef | null;
  sets: GameSummary[];
  pipeline_version: string;
  quality_report: QualityReport;
}

export interface RallyConfidence {
  segmentation: number | null;
  ocr: number | null;
  hits: number | null;
  tracking: number | null;
}

/** Mirrors records.RallyRecord. */
export interface Rally {
  rally_id: string;
  match_id: string;
  set_no: number;
  seq: number;
  start_frame: number;
  end_frame: number;
  duration_s: number;
  score_before: ScorePair | null;
  score_after: ScorePair | null;
  server: PlayerRef | null;
  serve_type: ServeType | null;
  winner: PlayerRef | null;
  lose_reason: string | null;
  n_strokes: number | null;
  score_verified: boolean;
  confidence: RallyConfidence;
  qa_flags: string[];
  clip_uri: string | null;
}

// --- errors -----------------------------------------------------------------------

/** The API answered with a non-2xx status. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly url: string,
    message?: string,
  ) {
    super(message ?? `API request failed: ${status} ${url}`);
    this.name = "ApiError";
  }
}

/** The API could not be reached at all (service down, wrong NEXT_PUBLIC_API_URL). */
export class ApiUnreachableError extends Error {
  constructor(public readonly url: string) {
    super(`API unreachable at ${url} — is the FastAPI service running?`);
    this.name = "ApiUnreachableError";
  }
}

// --- fetch core -------------------------------------------------------------------

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  let res: Response;
  try {
    // no-store: analytics data updates as the pipeline reprocesses matches;
    // stale caches would contradict the confidence-first product promise.
    res = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiUnreachableError(url);
  }
  if (!res.ok) {
    throw new ApiError(res.status, url);
  }
  return (await res.json()) as T;
}

// --- endpoints --------------------------------------------------------------------

export async function listMatches(): Promise<Match[]> {
  return fetchJson<Match[]>("/matches");
}

export async function getMatch(matchId: string): Promise<Match> {
  return fetchJson<Match>(`/matches/${encodeURIComponent(matchId)}`);
}

export async function listRallies(matchId: string): Promise<Rally[]> {
  return fetchJson<Rally[]>(`/matches/${encodeURIComponent(matchId)}/rallies`);
}
