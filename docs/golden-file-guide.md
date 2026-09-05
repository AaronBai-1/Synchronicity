# Golden File Guide — preparing one match, step by step

This is the hands-on companion to [golden-set.md](golden-set.md) (which covers *which*
matches to pick and the labeling conventions). This doc covers the mechanics: where
every file goes, how the JSON is created, and exactly what goes in it.

The schema source of truth is
[`pipeline/src/synchro_pipeline/eval/golden.py`](../pipeline/src/synchro_pipeline/eval/golden.py)
— if this doc and that module ever disagree, the module wins.

## 1. Where files go

| File | Location | Committed? |
|---|---|---|
| Original download (mp4/webm/mkv, any name) | `data/source/<match_id>.<ext>` | **never** (size + legal posture) |
| Mezzanine (what you label against) | `data/mezzanine/<match_id>.mp4` | never |
| **The golden JSON** | `data/golden/<match_id>.json` | **yes — the only committed artifact** |
| Pipeline run artifacts | `data/runs/<match_id>/` | never |

`.gitignore` enforces this: everything under `data/` is ignored *except*
`data/golden/*.json`.

**`<match_id>` is a permanent slug** — lowercase, hyphens, chosen once, e.g.
`2023-denmark-open-msf`. It names all three files, keys the JSON, and later keys DB
rows. Never rename it after labeling starts.

## 2. Make the mezzanine

All frame numbers in the golden file index into the **mezzanine**, not the original.
Make it with the pipeline's own S0 code path (drift-proof by construction):

```sh
uv run python -m synchro_pipeline.stages.s0_ingest \
    data/source/<match_id>.webm data/mezzanine/<match_id>.mp4
```

(Equivalent ffmpeg, should you need it by hand: `-vf "scale=-2:720,setsar=1" -r 30
-fps_mode cfr -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an`; ≥ 5.1 for
`-fps_mode`.) The mezzanine has **no audio track** by design — for the audio-assisted
hit labeling below, keep the source file around, or use the `audio.wav` artifact from a
full S0 pipeline run. If S0's parameters ever change, existing golden files stay valid —
their `video_sha256` pins them to the mezzanine they were labelled on — but regenerate
mezzanines for *new* matches with the new settings.

## 3. Create and fill the JSON

**Do not write the JSON by hand.** The first invocation of a labeling tool creates it
(and automatically stamps `video_sha256` from `--video`).

**Shortcut for ShuttleSet matches**: if the match appears in ShuttleSet, don't label
hits manually — align their existing per-stroke labels with 2–4 anchor clicks
(golden-set.md §5.1):

```sh
uv run python pipeline/tools/align_shuttleset.py \
    --csv <shuttleset set csvs...> --golden data/golden/<match_id>.json \
    --video data/mezzanine/<match_id>.mp4 --anchors "ss_frame:mz_frame,..." \
    --match-id <match_id> --video-uri data/mezzanine/<match_id>.mp4 --dry-run
```

then refine the proposed rally boundaries in `label_rallies` and spot-check ~10 hits.
For everything else, manual labeling:

```sh
# rally boundaries + hit frames (do this FIRST — everything hangs off rallies).
# --audio enables audio-onset proposals: o/O jump between candidate hit frames,
# so hit marking becomes confirm-or-reject instead of scrub-and-hunt.
uv run python pipeline/tools/label_rallies.py \
    --video data/mezzanine/<match_id>.mp4 \
    --golden data/golden/<match_id>.json \
    --match-id <match_id> \
    --video-uri data/mezzanine/<match_id>.mp4 \
    --audio data/source/<match_id>.webm

# court keypoints, one frame at a time (~20 frames spread across zoom changes)
uv run python pipeline/tools/label_court.py \
    --video data/mezzanine/<match_id>.mp4 --frame <N> \
    --golden data/golden/<match_id>.json
```

Key bindings are in each tool's `--help` / module docstring. Labeling order:

1. `rallies` start/end (label_rallies: `r` / `e`)
2. `hits` inside each rally (`n` = near player, `f` = far player)
3. `court_labels` on ~20 sampled frames (label_court)
4. `score_timeline` — hand-edit (see below); it's a small list and the state machine
   validates it
5. `shuttle` points — **optional**, only on a few benchmark rallies (most tedious,
   least required)

The hand-edited fields — `source_description`, `broadcaster`, `discipline`,
`first_server`, `a_on_near_side_at_start` (the scoring state machine needs the last
two to derive serve courts and the side-switch schedule) — and the `score_timeline`
are added by opening the JSON in an editor. That is safe: the loader rejects typo'd
keys, overlaps, and contradictions with messages naming the offending frame.
Hits and rallies also accept a `flags` list (`hit_occluded`, `let`, `end_occluded`,
`score_inferred` — conventions in golden-set.md §3) for honest uncertainty marking.

## 4. What goes in the file, precisely

A complete, schema-validated example lives at
[docs/examples/sample-golden.json](examples/sample-golden.json) (kept honest by
`pipeline/tests/test_examples.py`). Annotated walkthrough (field-by-field authority:
`GoldenMatch` in `eval/golden.py`):

```jsonc
{
  "match_id": "2023-denmark-open-msf",          // you (via --match-id); the permanent slug
  "video_uri": "data/mezzanine/2023-denmark-open-msf.mp4",  // you (via --video-uri)
  "video_sha256": "3f5a9c…64 hex chars…",        // AUTOMATIC — stamped by the tools,
                                                 //   verified by tools + benchmark. Never type it.
  "source_description": "yt 'AXELSEN vs …' 1080p webm, downloaded 2026-08",  // you, by hand
  "broadcaster": "bwf-world-tour",               // you, by hand (per-broadcaster metrics)
  "discipline": "MS",                            // you, by hand
  "first_server": "A",                           // you, by hand — from the first rally
  "a_on_near_side_at_start": true,               // you, by hand — state machine anchor

  "rallies": [                                   // label_rallies tool
    {
      "start_frame": 3001,                       // serve motion begins; span is INCLUSIVE
      "end_frame": 3820,                         // shuttle lands / outcome visible
      "hits": [                                  // frame-exact: first frame at/after contact
        { "frame": 3010, "side": "near" },       // side = physical end: "near" | "far"
        { "frame": 3055, "side": "far", "flags": ["hit_occluded"] }  // honest uncertainty
      ],
      "shuttle": [                               // OPTIONAL; omit entirely when unlabelled
        { "frame": 3010, "x": 412.0, "y": 288.5, "visible": true },
        { "frame": 3011, "visible": false }      // human-confirmed not findable (real label)
      ]
    }
  ],

  "court_labels": [                              // label_court tool
    {
      "frame": 3001,
      "keypoints": [ [140.2, 660.1], null, /* …exactly 16 entries… */ ]
      // order = domain/court.py COURT_KEYPOINT_NAMES; null = occluded/not labelable
      // coordinates are mezzanine pixels (1280x720 frame)
    }
  ],

  "score_timeline": [                            // you, by hand
    { "frame": 3825, "a": 1, "b": 0 }            // frame where this score FIRST appears
  ]
}
```

Rules the loader enforces (it will refuse the file, naming the problem):

- unknown/typo'd keys anywhere → error (`extra="forbid"`)
- rallies may not overlap; `end_frame ≥ start_frame`; spans are inclusive
- every hit frame must lie inside its rally's span
- a `visible: true` shuttle point requires `x` and `y`
- `court_labels.keypoints` must have exactly 16 entries; no two labels for the same frame
- `video_sha256` must be 64 lowercase hex (or absent)
- ordering is forgiving — rallies/hits/labels are auto-sorted on load

Distinctions that matter:

- **`shuttle` absent vs `[]`**: absent = "not labelled" (the common case); an empty
  list = "labelled: shuttle never visible". Don't write `[]` unless you mean it.
- **Replays are not rallies**: a rally shown again in a broadcast replay is labelled
  once, at its live occurrence. Lets get a rally entry with no score change.

## 5. Verify, then commit

```sh
# validates schema + hash + runs the fake tracker over your rally spans
uv run python -m synchro_pipeline.eval.benchmark_shuttle \
    --video data/mezzanine/<match_id>.mp4 \
    --golden data/golden/<match_id>.json \
    --tracker fake --out /tmp/report.json

git add data/golden/<match_id>.json && git commit
```

Checklist before commit:

- [ ] every rally's score delta validates against the scoring rules
      (`synchro_pipeline.domain.scoring` — the acceptance script in golden-set.md)
- [ ] `source_description`, `broadcaster`, `discipline`, `first_server`, and
      `a_on_near_side_at_start` filled in
- [ ] ~20 court frames spanning the match's zoom range
- [ ] hits labelled for every rally; shuttle points for the benchmark subset only
- [ ] the benchmark command above exits 0

If the tools ever refuse with a *"not the encode the labels were made against"* error,
stop — you're pointing at a different file than the one labelled. Find the original
mezzanine (the sha256 in the JSON identifies it); only use `--allow-hash-mismatch` if
you deliberately intend to relabel against a new encode.
