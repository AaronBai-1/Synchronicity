# Design note: Shot review queue (the label flywheel)

Status: design (targets plan Phases 3–4) · Owner: pipeline + api + web
Contracts referenced: `pipeline/src/synchro_pipeline/schemas/records.py` (ShotRecord),
`pipeline/src/synchro_pipeline/domain/taxonomy.py` (CoarseShotType).

## Problem

The pipeline emits per-shot predictions with explicit uncertainty: every
`ShotRecord` carries `type_conf`, `type_probs`, `hit_det_conf`, and `qa_flags`,
and the contract's rule is *unknown = `None` + a qa flag, never a guess*
(docs/plan.md, "Core design principle"). Coaches will see wrong or abstained
shot labels in the dashboard. Every correction a coach makes is a labeled
training example we currently throw away. Per docs/plan.md, "one-click
correction turns users into the labeling flywheel": the review UI must convert
coach attention into (a) trustworthy match reports and (b) training data.

## Adopted pattern (with credit)

Prior art: BadmintonAnalyzer (Apache-2.0, Dhyey Mavani) implemented a
draft → `review_required` → editable grid → validated import loop across its
`annotation.py` / `structured_logging.py` / `video_lab.py` modules: CV output
becomes draft rows, low-confidence rows are flagged into a review CSV, a
`st.data_editor` grid lets a human edit them, and reviewed rows are imported
into the database as validated `ShotEvent`s. The loop shape is right; we adopt
it and map it onto our architecture:

1. **Enqueue.** A `ShotRecord` enters the review queue when
   `type_conf < threshold` (threshold per-class, tuned on the golden set) or
   `qa_flags` is non-empty. The queue reason is recorded verbatim
   (e.g. `low_type_conf`, or the specific qa flag).
2. **Review grid.** The `web/` dashboard renders an enum-constrained grid —
   the shot-type column is a dropdown over the **coarse-8 vocabulary**
   (`serve, clear, drop, smash, drive, net, lift, push_rush` from
   `CoarseShotType`), never free text — with the rally clip playing alongside
   and seek-to-shot on row focus. This is the same stat→clip loop the product
   is built on (docs/plan.md: "every stat click-through to video clips").
3. **Write-back.** A correction writes to Postgres (the `shots` table is
   updated, provenance preserved — see data model below) **and** appends a
   training delta: a ShuttleSet-schema row (the `ShotRecord` superset we
   already store) tagged with the human label, accumulated for future
   fine-tuning of S6.
4. **Held-out invariant.** Corrections originating from **golden matches are
   never fed back into training.** The golden set is the eval CI
   (docs/plan.md, "Verification approach"); training on it would corrupt the
   only measurement we trust. Enforced at the delta-export layer by match_id
   blocklist, not by reviewer discipline.

## Anti-patterns (defects observed in the predecessor — do not replicate)

- **Never fabricate outcomes.** Their importer force-labelled the last shot of
  every rally as `WINNER` and back-filled `rally_outcome` from that guess.
  Ours: `is_winner` / `is_error` / `lose_reason` stay `None` with a qa flag
  until the scoring state machine or a human supplies them.
- **Never let a proxy masquerade as confidence.** Their `confidence` column
  fell back to a constant (0.6) and mixed detection confidence with
  speed/angle heuristics. Ours: `type_conf` comes only from the classifier's
  calibrated distribution (`type_probs`); trajectory quality and hit
  confidence are separate fields.
- **Reviewed-without-editing must be distinguishable.** Their grid saved the
  whole dataframe back whether or not a human touched a row, so "a coach
  looked at this and agreed" was indistinguishable from "nobody looked".
  Ours: a review action always transitions status — `confirmed` (seen,
  unchanged) is distinct from `corrected` and from `pending`.

## Data model sketch

```sql
CREATE TABLE review_items (
  id               BIGSERIAL PRIMARY KEY,
  shot_id          TEXT NOT NULL REFERENCES shots(id),
  reason           TEXT NOT NULL,           -- 'low_type_conf' | a qa_flag value
  status           TEXT NOT NULL DEFAULT 'pending',
                   -- 'pending' | 'confirmed' | 'corrected'
  corrected_fields JSONB,                   -- {"type_coarse": "smash", ...}; NULL unless corrected
  reviewer         TEXT,                    -- user id; NULL while pending
  ts               TIMESTAMPTZ,             -- review time; NULL while pending
  UNIQUE (shot_id, reason)
);
```

- Original model predictions are never overwritten in place: `shots` keeps the
  predicted fields; the applied correction lives in `corrected_fields` and a
  view resolves human-over-model. That preserves the training pair
  (prediction, human label).
- Aggregates treat `pending` items as low-confidence (excluded + flagged, per
  the trust architecture); `confirmed`/`corrected` items count with full weight.

### API sketch (FastAPI, `api/`)

```
GET  /matches/{match_id}/review-items?status=pending     → paged queue
POST /review-items/{id}/confirm                          → status=confirmed
POST /review-items/{id}/correct   {corrected_fields}     → status=corrected;
                                    fields validated against enums (coarse-8)
GET  /matches/{match_id}/training-deltas                 → export accumulated
                                    ShuttleSet-schema rows (golden matches excluded)
```

Correction payloads are validated server-side against the taxonomy enums —
an invalid shot type is a 422, never a silently stored string.
