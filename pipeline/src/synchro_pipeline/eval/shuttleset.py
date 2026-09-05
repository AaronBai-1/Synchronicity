"""ShuttleSet → golden-set alignment (docs/plan.md Phase 0: the golden set is CI, and
docs/golden-set.md budgets 2–3 person-days of hand labels per match).

For matches that already appear in ShuttleSet (wywyWang/CoachAI-Projects, 44 pro
singles matches with human per-stroke labels), most of that budget is re-deriving
facts ShuttleSet already recorded: a frame number, a stroke order and a score for
every hit. The only obstacle is that ShuttleSet's `frame_num` indexes THEIR video
encode while golden labels index OUR 720p30 mezzanine. This module closes that gap:

    1. a human identifies 2–4 ANCHOR correspondences (the same visually-identified
       contact in both timelines: ShuttleSet frame_num ↔ mezzanine frame),
    2. `Alignment.fit` least-squares a linear map mezz = scale·ss + offset (constant
       frame-rate on both sides ⇒ the true map IS linear) and REFUSES when any anchor
       residual exceeds tolerance — a bad anchor must fail loudly, not shift every hit,
    3. `strokes_to_golden` transforms every stroke into proposed GoldenRally spans,
       GoldenHit frames and an approximate score timeline for human review.

Days of scrubbing become minutes of anchoring plus a review pass in label_rallies.

Trust posture (plan: abstain over guessing; every automated output is human-reviewed
before it becomes ground truth): rally boundaries are PROPOSED (hit span ± pad_s —
the golden schema cannot mark provenance, so the ConversionReport and
docs/golden-set.md carry that caveat), score-timeline frames are approximations,
and any rally whose sides/frames cannot be derived honestly is skipped with a
named reason in the report rather than emitted with a guess.

Verified CSV schema — fetched 2026-09-04 from the CoachAI-Projects repo (main branch,
ShuttleSet/set/<match>/set{1,2,3}.csv; spot-checked across three matches, identical):

    rally, ball_round, time, frame_num, roundscore_A, roundscore_B, player, server,
    type, aroundhead, backhand, hit_height, hit_area, hit_x, hit_y, landing_height,
    landing_area, landing_x, landing_y, lose_reason, win_reason, getpoint_player,
    flaw, player_location_area, player_location_x, player_location_y,
    opponent_location_area, opponent_location_x, opponent_location_y, db

Facts about that data this module relies on (each checked against the real files):

    * `type` values are CHINESE strings (殺球, 長球, …); ShuttleSet/README.md carries
      the English translation table, and OUR domain.taxonomy.ShotType values are
      exactly those English strings — so translation is a fixed dict, with 未知球種
      ("unknown shot type", present in the data but absent from the table) mapping to
      None. Untranslatable values become type=None with the raw string preserved.
    * `player_location_x/y` are CAMERA-FRAME PIXELS, not court metres: their ranges
      match the per-match court corners in ShuttleSet/set/homography.csv (e.g.
      upleft_y≈300 far end, downleft_y≈670 near end for a 720p frame). Larger y =
      lower in frame = the end nearest the broadcast camera = our "near" side
      (domain/court.py convention), which is what makes honest per-rally side
      resolution from locations possible (`resolve_sides_by_location`).
    * `roundscore_A/B` is the set score INCLUDING the current rally's outcome
      (rally 1 already reads 1–0), constant across the rally's strokes; player A is
      the match winner in ShuttleSet's convention. Golden `a`/`b` therefore map
      A→a, B→b, and per-rally winners derive from roundscore deltas — cross-checked
      with domain.scoring.is_valid_transition.
    * `frame_num` is a float column (e.g. "10418.0"); rows can have it empty.
    * the stroke of `ball_round` 1 is the serve, so the rally's server is that
      stroke's `player` — the seed for `make_serve_side_resolver`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from synchro_pipeline.domain.scoring import (
    DECIDING_GAME_SWITCH_SCORE,
    GameScore,
    is_valid_transition,
)
from synchro_pipeline.domain.taxonomy import ShotType
from synchro_pipeline.eval.golden import GoldenHit, GoldenRally, GoldenScoreEntry
from synchro_pipeline.schemas.records import PlayerRef, Side

# English ShotType values double as their own keys so a CSV that already carries the
# translated strings (e.g. a hand-fixed export) loads identically.
SHOT_TYPE_TRANSLATIONS: dict[str, ShotType] = {
    # Chinese → English, verbatim from ShuttleSet/README.md's translation table.
    "放小球": ShotType.NET_SHOT,
    "擋小球": ShotType.RETURN_NET,
    "殺球": ShotType.SMASH,
    "點扣": ShotType.WRIST_SMASH,
    "挑球": ShotType.LOB,
    "防守回挑": ShotType.DEFENSIVE_RETURN_LOB,
    "長球": ShotType.CLEAR,
    "平球": ShotType.DRIVE,
    "小平球": ShotType.DRIVEN_FLIGHT,
    "後場抽平球": ShotType.BACK_COURT_DRIVE,
    "切球": ShotType.DROP,
    "過渡切球": ShotType.PASSIVE_DROP,
    # The actual set CSVs spell passive drop 過度切球 (度, not the README table's 渡) —
    # verified in two real matches (41 occurrences, zero of the README spelling).
    "過度切球": ShotType.PASSIVE_DROP,
    "推球": ShotType.PUSH,
    "撲球": ShotType.RUSH,
    "防守回抽": ShotType.DEFENSIVE_RETURN_DRIVE,
    "勾球": ShotType.CROSS_COURT_NET_SHOT,
    "發短球": ShotType.SHORT_SERVICE,
    "發長球": ShotType.LONG_SERVICE,
    **{t.value: t for t in ShotType},
}

_REQUIRED_COLUMNS = frozenset(
    {"rally", "ball_round", "frame_num", "roundscore_A", "roundscore_B", "player", "type"}
)

_SET_FILENAME_RE = re.compile(r"set(\d+)\.csv$", re.IGNORECASE)


class ShuttleSetError(ValueError):
    """A ShuttleSet CSV could not be loaded/used; the message names the file and why."""


class AlignmentError(ValueError):
    """The anchor set does not support a trustworthy frame map; message says why."""


class SideResolutionError(ValueError):
    """near/far could not be derived honestly for a rally — abstain, don't guess."""


class ShuttleStroke(BaseModel):
    """One parsed ShuttleSet stroke row (columns this pipeline consumes; see module doc).

    `frame_num=None` / `type=None` / `player=None` mean the source row was empty or
    untranslatable there — kept as explicit abstentions so downstream code decides
    (and reports) instead of a NaN silently poisoning arithmetic.
    """

    model_config = ConfigDict(extra="forbid")

    set_no: int = Field(ge=1)
    rally: int = Field(ge=1)  # rally serial number within the set (ShuttleSet naming)
    ball_round: int = Field(ge=1)  # stroke order within the rally, 1 = serve
    frame_num: int | None  # frame in ShuttleSet's OWN encode; None = not recorded
    time_str: str | None = None  # ShuttleSet "time" column, for human cross-reference
    type: ShotType | None  # None = untranslatable (e.g. 未知球種); raw kept alongside
    type_raw: str
    player: PlayerRef | None  # A = the match winner (ShuttleSet convention)
    roundscore_a: int | None = Field(default=None, ge=0)  # set score incl. this rally
    roundscore_b: int | None = Field(default=None, ge=0)
    player_location_x: float | None = None  # camera-frame px (see module docstring)
    player_location_y: float | None = None
    getpoint_player: str | None = None  # set on a rally's last stroke


def _opt_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def _opt_str(value: object) -> str | None:
    return None if pd.isna(value) else str(value).strip()


def load_shuttleset_match(
    csv_paths: Sequence[str | Path], set_numbers: Sequence[int] | None = None
) -> list[ShuttleStroke]:
    """Parse one or more ShuttleSet per-set CSVs into strokes, sorted (set, rally, order).

    Set numbers come from the ShuttleSet filename convention (`set1.csv`, `set2.csv`, …)
    unless `set_numbers` supplies them explicitly (same length/order as `csv_paths`).
    Raises ShuttleSetError on unreadable files, missing required columns, or an
    unrecognizable filename — a wrong set number would corrupt serve-side derivation.
    """
    if set_numbers is not None and len(set_numbers) != len(csv_paths):
        raise ShuttleSetError(
            f"set_numbers has {len(set_numbers)} entries for {len(csv_paths)} csv paths"
        )
    strokes: list[ShuttleStroke] = []
    for i, raw_path in enumerate(csv_paths):
        path = Path(raw_path)
        if set_numbers is not None:
            set_no = set_numbers[i]
        else:
            m = _SET_FILENAME_RE.search(path.name)
            if m is None:
                raise ShuttleSetError(
                    f"{path}: cannot infer the set number from the filename (expected "
                    "ShuttleSet's set<N>.csv convention); pass set_numbers explicitly"
                )
            set_no = int(m.group(1))
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise ShuttleSetError(f"ShuttleSet csv not found: {path}") from exc
        except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
            raise ShuttleSetError(f"{path} is not a readable csv: {exc}") from exc
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ShuttleSetError(
                f"{path} is missing expected ShuttleSet columns: {sorted(missing)} "
                f"(got: {list(df.columns)})"
            )
        for row in df.itertuples(index=False):
            type_raw = _opt_str(row.type) or ""
            player = _opt_str(row.player)
            frame = _opt_float(row.frame_num)
            score_a = _opt_float(row.roundscore_A)
            score_b = _opt_float(row.roundscore_B)
            strokes.append(
                ShuttleStroke(
                    set_no=set_no,
                    rally=int(row.rally),
                    ball_round=int(float(row.ball_round)),
                    frame_num=None if frame is None else int(round(frame)),
                    time_str=_opt_str(getattr(row, "time", None)),
                    type=SHOT_TYPE_TRANSLATIONS.get(type_raw),
                    type_raw=type_raw,
                    player=player if player in ("A", "B") else None,
                    roundscore_a=None if score_a is None else int(score_a),
                    roundscore_b=None if score_b is None else int(score_b),
                    player_location_x=_opt_float(getattr(row, "player_location_x", None)),
                    player_location_y=_opt_float(getattr(row, "player_location_y", None)),
                    getpoint_player=_opt_str(getattr(row, "getpoint_player", None)),
                )
            )
    strokes.sort(key=lambda s: (s.set_no, s.rally, s.ball_round))
    return strokes


# --- anchor-based frame alignment ---------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Linear frame map from ShuttleSet's encode to our mezzanine: mezz = scale·ss + offset.

    Both timelines are constant-frame-rate videos of the same broadcast, so the true
    map is exactly linear (scale = fps ratio, offset = trim difference); anchors
    over-determine it and the per-anchor residuals expose any mis-clicked anchor.
    """

    scale: float
    offset: float
    anchors: tuple[tuple[int, int], ...]  # (shuttleset_frame, mezzanine_frame) as given
    residuals: tuple[float, ...]  # predicted − actual mezzanine frame, per anchor

    @property
    def max_residual_frames(self) -> float:
        """Worst absolute anchor residual, in mezzanine frames — the fit's honesty gauge."""
        return max(abs(r) for r in self.residuals)

    def map_frame(self, shuttleset_frame: float) -> int:
        """A ShuttleSet frame number's position on the mezzanine timeline (nearest frame)."""
        return int(round(self.scale * shuttleset_frame + self.offset))

    def require_within(self, tolerance_frames: float = 5.0) -> Alignment:
        """Refuse (AlignmentError) unless every anchor residual is ≤ tolerance.

        5 frames default: docs/golden-set.md gives rally boundaries ±5 frames of slack,
        and hit labels are refined by a human afterwards anyway — but a residual past
        that means at least one anchor (or the linear-map assumption) is wrong, and a
        systematically shifted hit set would poison the F1@±3 metric silently.
        """
        if self.max_residual_frames > tolerance_frames:
            worst = int(np.argmax(np.abs(self.residuals)))
            ss, mz = self.anchors[worst]
            raise AlignmentError(
                f"alignment rejected: anchor {ss}:{mz} is {abs(self.residuals[worst]):.1f} "
                f"mezzanine frames off the fitted line (tolerance {tolerance_frames:g}). "
                "Re-check the anchor correspondences frame-by-frame; if they are all "
                "correct, the two videos do not share a constant-rate linear timeline "
                "and this tool cannot align them."
            )
        return self

    @classmethod
    def fit(
        cls, anchors: Sequence[tuple[int, int]], *, tolerance_frames: float | None = 5.0
    ) -> Alignment:
        """Least-squares fit from ≥2 (shuttleset_frame, mezzanine_frame) anchor pairs.

        With `tolerance_frames` set (the default), the fit is gated through
        `require_within` so an inconsistent anchor refuses loudly instead of silently
        shifting every transformed hit. Pass None to inspect a raw fit.
        """
        if len(anchors) < 2:
            raise AlignmentError(
                f"need at least 2 anchor correspondences to fit scale + offset, got {len(anchors)}"
            )
        ss = np.array([a[0] for a in anchors], dtype=np.float64)
        mz = np.array([a[1] for a in anchors], dtype=np.float64)
        if np.unique(ss).size < 2:
            raise AlignmentError(
                "anchors must reference at least 2 distinct ShuttleSet frames "
                f"(got {sorted({int(x) for x in ss})})"
            )
        design = np.stack([ss, np.ones_like(ss)], axis=1)
        (scale, offset), *_ = np.linalg.lstsq(design, mz, rcond=None)
        if scale <= 0:
            raise AlignmentError(
                f"fitted scale {scale:.4f} is not positive — the anchors run time backwards; "
                "re-check the correspondences (each pair is shuttleset_frame:mezzanine_frame)"
            )
        predicted = design @ np.array([scale, offset])
        fitted = cls(
            scale=float(scale),
            offset=float(offset),
            anchors=tuple((int(a), int(b)) for a, b in anchors),
            residuals=tuple(float(r) for r in predicted - mz),
        )
        return fitted if tolerance_frames is None else fitted.require_within(tolerance_frames)


def fit_alignment(anchors: Sequence[tuple[int, int]]) -> Alignment:
    """Raw least-squares fit (no tolerance gate) — see Alignment.fit for the gated form."""
    return Alignment.fit(anchors, tolerance_frames=None)


# --- near/far side resolution -------------------------------------------------------

SideResolver = Callable[[Sequence[ShuttleStroke]], list[Side]]

# Minimum separation between the two players' mean location-y before location-based
# side resolution trusts itself. The court spans ~370 px vertically in ShuttleSet's
# camera frames (homography.csv: far baseline y≈300, near y≈670), so genuinely
# opposite ends separate by hundreds of px; anything under ~25 px means broken data.
MIN_SIDE_SEPARATION_PX = 25.0


def _check_alternation(rally_strokes: Sequence[ShuttleStroke]) -> None:
    label = f"set {rally_strokes[0].set_no} rally {rally_strokes[0].rally}"
    for stroke in rally_strokes:
        if stroke.player is None:
            raise SideResolutionError(
                f"{label}: stroke {stroke.ball_round} has no player recorded"
            )
    for prev, cur in zip(rally_strokes, rally_strokes[1:], strict=False):
        if prev.player == cur.player:
            raise SideResolutionError(
                f"{label}: strokes {prev.ball_round} and {cur.ball_round} are both by "
                f"player {cur.player} — singles strokes must alternate; the source rows "
                "look corrupted, label this rally by hand"
            )


def resolve_sides_by_location(
    rally_strokes: Sequence[ShuttleStroke], *, min_separation_px: float = MIN_SIDE_SEPARATION_PX
) -> list[Side]:
    """Derive each stroke's near/far side from ShuttleSet's player-location pixels.

    ASSUMPTION made (and why it is honest): player_location_x/y are camera-frame
    pixels of the standard game camera (verified against homography.csv — module
    docstring), and the mezzanine shows the same broadcast — which frame alignment
    already presupposes — so image-down (larger y) is the near end in both. Within a
    rally each player keeps one end, so the hitter's end per stroke follows from one
    bit: whose mean location-y is larger. Per-stroke noise cancels in the means, and
    side switches between games/at 11 are irrelevant because locations are per-stroke
    observations, not derived state.

    Raises SideResolutionError (caller abstains + reports) when players don't
    alternate, either player has no recorded location, or the separation is too small
    to call — never guesses.
    """
    _check_alternation(rally_strokes)
    label = f"set {rally_strokes[0].set_no} rally {rally_strokes[0].rally}"
    ys: dict[PlayerRef, list[float]] = {"A": [], "B": []}
    for stroke in rally_strokes:
        if stroke.player_location_y is not None and stroke.player is not None:
            ys[stroke.player].append(stroke.player_location_y)
    if not ys["A"] or not ys["B"]:
        missing = " and ".join(p for p in ("A", "B") if not ys[p])
        raise SideResolutionError(
            f"{label}: no player-location coordinates for player {missing}; "
            "cannot resolve near/far from locations — use a serve-side mode "
            "or label this rally by hand"
        )
    mean_a = float(np.mean(ys["A"]))
    mean_b = float(np.mean(ys["B"]))
    if abs(mean_a - mean_b) < min_separation_px:
        raise SideResolutionError(
            f"{label}: player mean location-y values are only {abs(mean_a - mean_b):.1f} px "
            f"apart (< {min_separation_px:g}) — too ambiguous to call near/far honestly"
        )
    a_side: Side = "near" if mean_a > mean_b else "far"  # larger y = lower = near camera
    b_side: Side = "far" if a_side == "near" else "near"
    return [a_side if s.player == "A" else b_side for s in rally_strokes]


def make_serve_side_resolver(
    strokes: Sequence[ShuttleStroke], first_server_side: Side
) -> SideResolver:
    """Side resolver seeded by ONE human-supplied fact instead of location data.

    ASSUMPTIONS made (each is exactly a BWF rule plus one user input — spelled out
    because this mode, unlike location mode, derives sides instead of observing them):

        * `first_server_side` states which physical end (mezzanine view) the player
          serving the FIRST rally of the EARLIEST provided set occupies. That server
          is identified from the data: the `player` of ball_round 1.
        * players change ends after every game (so the map flips per set difference),
          and in set 3 — the deciding game — when the leading score first reaches
          `domain.scoring.DECIDING_GAME_SWITCH_SCORE` (11), detected from the
          roundscore columns.
        * within a rally, strokes alternate ends starting from the server's.

    Use when location-based resolution abstains (or to cross-check it). Wrong user
    input flips EVERY side label, so docs/golden-set.md mandates spot-checking hits
    against the video either way.
    """
    by_set: dict[int, dict[int, list[ShuttleStroke]]] = {}
    for stroke in strokes:
        by_set.setdefault(stroke.set_no, {}).setdefault(stroke.rally, []).append(stroke)
    if not by_set:
        raise ShuttleSetError("no strokes to build a serve-side resolver from")

    set_numbers = sorted(by_set)
    first_set = set_numbers[0]
    first_rally = min(by_set[first_set])
    opener = min(by_set[first_set][first_rally], key=lambda s: s.ball_round)
    if opener.player is None:
        raise ShuttleSetError(
            f"set {first_set} rally {first_rally}: the serve stroke has no player recorded; "
            "cannot seed serve-side resolution"
        )
    # Which end player A occupies in the earliest provided set.
    other: Side = "far" if first_server_side == "near" else "near"
    a_side_first_set: Side = first_server_side if opener.player == "A" else other

    side_by_set_rally: dict[tuple[int, int], dict[PlayerRef, Side]] = {}
    for set_no in set_numbers:
        # Ends swap after every game → parity of the set-number gap vs the seeded set.
        flips = (set_no - first_set) % 2
        a_side: Side = a_side_first_set if flips == 0 else ("far" if a_side_first_set == "near" else "near")
        switched = False
        for rally_no in sorted(by_set[set_no]):
            rally_strokes = by_set[set_no][rally_no]
            b_side: Side = "far" if a_side == "near" else "near"
            side_by_set_rally[(set_no, rally_no)] = {"A": a_side, "B": b_side}
            scores = [
                max(s.roundscore_a, s.roundscore_b)
                for s in rally_strokes
                if s.roundscore_a is not None and s.roundscore_b is not None
            ]
            if (
                set_no == 3  # best-of-3 deciding game (ShuttleSet is all best-of-3)
                and not switched
                and scores
                and max(scores) >= DECIDING_GAME_SWITCH_SCORE
            ):
                a_side = "far" if a_side == "near" else "near"
                switched = True

    def resolver(rally_strokes: Sequence[ShuttleStroke]) -> list[Side]:
        _check_alternation(rally_strokes)
        first = rally_strokes[0]
        sides = side_by_set_rally.get((first.set_no, first.rally))
        if sides is None:  # resolver built from a different stroke set
            raise SideResolutionError(
                f"set {first.set_no} rally {first.rally} was not part of the strokes this "
                "serve-side resolver was built from"
            )
        return [sides[s.player] for s in rally_strokes]  # type: ignore[index]

    return resolver


# --- strokes → golden proposals -----------------------------------------------------


@dataclass
class ConversionReport:
    """Everything a human must know before trusting the emitted labels.

    The golden schema cannot mark provenance, so this report is where the honesty
    lives: which rallies were skipped and why, which derived values are approximate,
    and what review work remains (also documented in docs/golden-set.md §7).
    """

    n_strokes: int = 0
    n_rallies: int = 0
    n_emitted: int = 0
    n_hits: int = 0
    side_mode: str = ""
    skipped: list[str] = field(default_factory=list)  # per-rally reasons — human follows up
    warnings: list[str] = field(default_factory=list)  # anomalies in emitted data
    notes: list[str] = field(default_factory=list)  # standing caveats of this tool
    score_timeline: list[GoldenScoreEntry] = field(default_factory=list)  # approximate

    def summary_lines(self) -> list[str]:
        lines = [
            f"strokes: {self.n_strokes} in {self.n_rallies} rallies "
            f"-> {self.n_emitted} rallies / {self.n_hits} hits emitted, "
            f"{len(self.score_timeline)} score entries (side mode: {self.side_mode})",
        ]
        lines += [f"SKIPPED  {s}" for s in self.skipped]
        lines += [f"WARNING  {w}" for w in self.warnings]
        lines += [f"NOTE     {n}" for n in self.notes]
        return lines


_STANDING_NOTES = [
    "rally start/end frames are PROPOSED (hit span padded by --pad-s); refine both "
    "against the video in label_rallies before trusting rally-segmentation metrics",
    "score_timeline frames are the last hit + a fixed delta, NOT the observed scorebug "
    "change; verify each against the broadcast scorebug",
    "spot-check ~10 aligned hits per match against the video before committing "
    "(a systematic anchor error shifts every frame identically)",
]


def _score_warnings(
    ordered: list[tuple[tuple[int, int], list[ShuttleStroke]]],
) -> list[str]:
    """Cross-check roundscore deltas with the scoring rules (plan: scores are the trust
    anchor); anomalies are warnings for the reviewing human, not fatal errors."""
    warnings: list[str] = []
    prev_key: tuple[int, int] | None = None
    prev_score: GameScore | None = None
    for (set_no, rally_no), rally_strokes in ordered:
        last = rally_strokes[-1]
        if last.roundscore_a is None or last.roundscore_b is None:
            warnings.append(f"set {set_no} rally {rally_no}: missing roundscore columns")
            prev_key, prev_score = (set_no, rally_no), None
            continue
        score = GameScore(last.roundscore_a, last.roundscore_b)
        if prev_key is not None and prev_key[0] == set_no and prev_score is not None:
            if not is_valid_transition(prev_score, score):
                warnings.append(
                    f"set {set_no} rally {rally_no}: roundscore "
                    f"{prev_score.a}-{prev_score.b} -> {score.a}-{score.b} is not a legal "
                    "single-rally transition (domain.scoring); verify the score timeline "
                    "by hand"
                )
        elif prev_key is None or prev_key[0] != set_no:
            if (score.a, score.b) not in ((1, 0), (0, 1)):
                warnings.append(
                    f"set {set_no} rally {rally_no}: first provided rally of the set has "
                    f"roundscore {score.a}-{score.b}, not 1-0/0-1 — the csv may be "
                    "truncated; score entries remain per-row facts but review the gap"
                )
        prev_key, prev_score = (set_no, rally_no), score
    return warnings


def strokes_to_golden(
    strokes: Sequence[ShuttleStroke],
    alignment: Alignment,
    *,
    side_resolver: SideResolver | None = None,
    pad_s: float = 1.5,
    fps: float = 30.0,
    max_frame: int | None = None,
    score_delta_s: float = 1.0,
) -> tuple[list[GoldenRally], ConversionReport]:
    """Transform ShuttleSet strokes into proposed golden rallies + an approximate
    score timeline (returned inside the ConversionReport).

    Per rally (grouped by set + rally serial): hits at the alignment-mapped frames
    with sides from `side_resolver` (default: resolve_sides_by_location), and a
    PROPOSED span of first-hit − pad … last-hit + pad — ShuttleSet has no boundary
    labels, so pad_s=1.5 s brackets the serve motion and the shuttle landing for the
    human to tighten in label_rallies (docs/golden-set.md §3.1 conventions). Padding
    never eats a neighbouring rally: colliding spans are trimmed to the midpoint
    between the adjacent hits, and `max_frame` (the mezzanine's last frame index)
    clamps the pads.

    A rally is SKIPPED — with a reason in report.skipped, never guessed at — when any
    stroke lacks a frame number, mapped frames leave [0, max_frame] (wrong video or
    bad alignment), mapped frames fail to strictly increase (degenerate alignment), or
    the side resolver abstains (SideResolutionError).

    Score timeline: one GoldenScoreEntry per EMITTED rally at last-hit +
    `score_delta_s` (scorebugs update ~1 s after the rally) with a/b =
    roundscore_A/B — approximations for a human to verify, as report.notes says.
    """
    resolver = side_resolver if side_resolver is not None else resolve_sides_by_location
    side_mode = (
        "location (ShuttleSet player-location pixels)"
        if side_resolver is None
        else getattr(resolver, "__name__", "custom resolver")
    )

    grouped: dict[tuple[int, int], list[ShuttleStroke]] = {}
    for stroke in strokes:
        grouped.setdefault((stroke.set_no, stroke.rally), []).append(stroke)
    ordered = sorted(grouped.items())
    for _, rally_strokes in ordered:
        rally_strokes.sort(key=lambda s: s.ball_round)

    report = ConversionReport(
        n_strokes=len(strokes),
        n_rallies=len(ordered),
        side_mode=side_mode,
        notes=list(_STANDING_NOTES),
    )
    report.warnings.extend(_score_warnings(ordered))

    pad = int(round(pad_s * fps))
    score_delta = int(round(score_delta_s * fps))

    emitted: list[tuple[list[int], list[Side], ShuttleStroke]] = []
    for (set_no, rally_no), rally_strokes in ordered:
        label = f"set {set_no} rally {rally_no}"
        missing = [s.ball_round for s in rally_strokes if s.frame_num is None]
        if missing:
            report.skipped.append(
                f"{label}: stroke(s) {missing} have no frame_num — label this rally by hand"
            )
            continue
        frames = [alignment.map_frame(s.frame_num) for s in rally_strokes]  # type: ignore[arg-type]
        if frames[0] < 0 or (max_frame is not None and frames[-1] > max_frame):
            report.skipped.append(
                f"{label}: mapped frames {frames[0]}..{frames[-1]} fall outside the "
                f"mezzanine (0..{max_frame if max_frame is not None else '?'}) — wrong "
                "video, or anchors from a different encode"
            )
            continue
        if any(b <= a for a, b in zip(frames, frames[1:], strict=False)):
            report.skipped.append(
                f"{label}: mapped hit frames are not strictly increasing ({frames}) — "
                "alignment scale too small for this stroke spacing; label by hand"
            )
            continue
        try:
            sides = resolver(rally_strokes)
        except SideResolutionError as exc:
            report.skipped.append(str(exc))
            continue
        emitted.append((frames, sides, rally_strokes[-1]))

    # Rallies must land on the mezzanine in (set, rally) order — a later rally mapping
    # to earlier frames means broken source data or a wrong anchor, so it is skipped
    # (abstain), not force-fitted around its neighbour.
    monotonic: list[tuple[list[int], list[Side], ShuttleStroke]] = []
    for frames, sides, last_stroke in emitted:
        if monotonic and frames[0] <= monotonic[-1][0][-1]:
            report.skipped.append(
                f"set {last_stroke.set_no} rally {last_stroke.rally}: mapped hits start at "
                f"frame {frames[0]}, before the previous rally's last hit "
                f"({monotonic[-1][0][-1]}) — out-of-order source data; label by hand"
            )
            continue
        monotonic.append((frames, sides, last_stroke))
    emitted = monotonic

    # Proposed spans, then trim pad collisions between consecutive rallies (the golden
    # schema rejects overlaps; hits themselves are never moved, only padding shrinks).
    spans: list[tuple[int, int]] = []
    for frames, _, _ in emitted:
        start = max(frames[0] - pad, 0)
        end = frames[-1] + pad if max_frame is None else min(frames[-1] + pad, max_frame)
        spans.append((start, end))
    for i in range(len(spans) - 1):
        prev_last_hit = emitted[i][0][-1]
        next_first_hit = emitted[i + 1][0][0]
        if spans[i][1] >= spans[i + 1][0]:
            mid = (prev_last_hit + next_first_hit) // 2
            spans[i] = (spans[i][0], min(spans[i][1], mid))
            spans[i + 1] = (max(spans[i + 1][0], mid + 1), spans[i + 1][1])

    rallies: list[GoldenRally] = []
    seen_score_frames: set[int] = set()
    for (frames, sides, last_stroke), (start, end) in zip(emitted, spans, strict=True):
        rallies.append(
            GoldenRally(
                start_frame=start,
                end_frame=end,
                hits=[GoldenHit(frame=f, side=s) for f, s in zip(frames, sides, strict=True)],
            )
        )
        report.n_hits += len(frames)
        if last_stroke.roundscore_a is None or last_stroke.roundscore_b is None:
            continue  # already warned by _score_warnings
        score_frame = frames[-1] + score_delta
        if max_frame is not None:
            score_frame = min(score_frame, max_frame)
        if score_frame in seen_score_frames:
            report.warnings.append(
                f"score entry for set {last_stroke.set_no} rally {last_stroke.rally} "
                f"collides with an earlier entry at frame {score_frame}; dropped"
            )
            continue
        seen_score_frames.add(score_frame)
        report.score_timeline.append(
            GoldenScoreEntry(
                frame=score_frame, a=last_stroke.roundscore_a, b=last_stroke.roundscore_b
            )
        )
    report.n_emitted = len(rallies)
    return rallies, report
