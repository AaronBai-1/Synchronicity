# License Outreach Emails + Asset Status

> **DRAFTS — DO NOT SEND WITHOUT REVIEW; sending is a user decision.**
> These are working drafts for the plan's week-1 action ("Email dataset/weight
> authors for commercial-use confirmation in week 1", docs/plan.md Phase 0 /
> risk #3). Review, fill the bracketed placeholders, and send from your own
> account if and when you decide to. Keep replies filed in this directory.

Common context for all four (adapt, don't paste): we are building a commercial
badminton analytics product for players and coaches; match videos are always
sourced/uploaded by our users, never redistributed by us; we are asking about the
*annotations/weights/datasets* specifically, because code licenses (MIT/Apache) do
not necessarily cover them.

---

## Draft 1 — ShuttleSet / CoachAI-Projects authors (NYCU, wywyWang)

**To:** [ShuttleSet corresponding author — see the ShuttleSet paper / wywyWang GitHub profile]
**Subject:** ShuttleSet annotations — commercial-use clarification

Dear [Prof./Dr. name],

I am building a commercial badminton analytics product for players and coaches. We
would like to use the ShuttleSet / ShuttleSet22 stroke-level annotations to train
models (hit detection and shot classification), and to benchmark our pipeline
against them.

The CoachAI-Projects repository is MIT-licensed, but it is not clear to us whether
that license also covers the annotation data itself, as opposed to the code. Could
you clarify:

1. May the stroke annotations be used commercially for model training?
2. Is the annotation data covered by the repository's MIT license, or by separate
   terms? If separate, what are they?

For clarity on our side: we do not redistribute the annotations or any match
footage. Videos are sourced by our users; we would use the annotations internally
for training and evaluation only, with attribution to ShuttleSet and the CoachAI
project in our documentation.

Thank you for making this dataset available.

Best regards,
[Name]
[Product/company, one line]
[Contact]

---

## Draft 2 — TrackNetV3 author (qaz812345)

**To:** [TrackNetV3 maintainer — qaz812345 GitHub profile / paper contact]
**Subject:** TrackNetV3 pretrained weights — provenance and commercial inference

Hello,

I am building a commercial badminton analytics product and evaluating TrackNetV3
for shuttlecock tracking. The code's MIT license is clear; my questions are about
the released pretrained weights:

1. What footage was used to train the released weights (e.g. the NYCU TrackNetV2
   shuttlecock dataset, broadcast recordings), and under what terms was that
   training data used?
2. Is commercial *inference* use of the released weights permitted, in your view as
   the author?

If the weights' provenance makes commercial use unclear, we would retrain the
architecture on our own labeled data instead — so a frank answer either way is
genuinely useful to us.

Thank you for the excellent work and for releasing it openly.

Best regards,
[Name]
[Product/company, one line]
[Contact]

---

## Draft 3 — BFMD authors (Ning-D/BFMD)

**To:** [BFMD corresponding author — Ning-D GitHub / dataset paper contact]
**Subject:** BFMD dataset — availability and license for commercial training

Dear [Dr./Prof. name],

I am building a commercial badminton analytics product and am interested in the
BFMD (Badminton Full Match Dataset) for training rally/non-rally frame
classification and hit-event models.

Could you let me know:

1. Is the dataset currently available for download, and through what channel?
2. What are its license terms — specifically, is use for commercial model training
   permitted?

We would use it for internal training/evaluation only; we do not redistribute
datasets or footage, and our product processes videos our users source themselves.

Thank you.

Best regards,
[Name]
[Product/company, one line]
[Contact]

---

## Draft 4 — NYCU TrackNetV2 Shuttlecock Trajectory Dataset owners

**To:** [NYCU TrackNetV2 dataset maintainers — lab contact from the TrackNetV2 paper/page]
**Subject:** TrackNetV2 Shuttlecock Trajectory Dataset — formal license terms

Dear [Prof./Dr. name],

Your lab's Shuttlecock Trajectory Dataset (distributed via the SharePoint link on
the TrackNetV2 page) is, as far as we can find, published without an explicit
license. I am building a commercial badminton analytics product and would like to
know the terms under which the dataset may be used:

1. Is use for training models that are deployed commercially permitted?
2. Is there a formal license text (or one you would be willing to attach), so that
   downstream users have certainty?

We do not redistribute the dataset or any footage; usage would be internal training
and evaluation, with attribution. If commercial use is not permitted, knowing that
clearly is just as valuable — we would label our own data instead.

Thank you.

Best regards,
[Name]
[Product/company, one line]
[Contact]

---

## Asset / license status table

Status values: **verified-clean** (license text confirmed compatible) ·
**needs-verification** (usable only after outreach/verification above) ·
**banned** (license hard rule in docs/plan.md — never a dependency).

| Asset | License | Status | Fallback |
|---|---|---|---|
| Ultralytics YOLOv8/11 | AGPL-3.0 | **banned** | YOLOX / RT-DETR (Apache-2.0) |
| YOLOv7 | GPL-3.0 | **banned** | YOLOX / RT-DETR (Apache-2.0) |
| MonoTrack (code) | unverified | **blueprint-only** — reimplement ideas, never copy code | Own HitNet reimplementation from the paper's formulation |
| TrackNetV3 code | MIT | verified-clean | — |
| TrackNetV3 pretrained weights | n/a (weight provenance unknown) | needs-verification (Draft 2) | Retrain TrackNetV3 arch on own labels (~4–6 person-weeks per plan) |
| NYCU TrackNetV2 shuttlecock dataset | none published | needs-verification (Draft 4) | Self-label shuttle positions on golden + additional matches |
| ShuttleSet / ShuttleSet22 annotations | repo MIT; data terms unclear | needs-verification (Draft 1) | Regenerate stroke labels through own perception stack |
| CoachAI-Projects code | MIT | verified-clean | — |
| BFMD dataset | unclear | needs-verification (Draft 3) | Self-label rally/non-rally + hit events on own footage |
| TennisCourtDetector (yastrebksv) | code license present; **weights unspecified** | architecture only — **we retrain** on 2–5k self-labeled frames (plan S2) | Already the plan of record; no weight dependency |
| BST (shot-classification repo) | check repo license before use | needs-verification | Reimplement transformer (pose+shuttle+position inputs), retrain |
| WASB-SBDT | MIT | verified-clean (eval cross-check only) | — |
| PySceneDetect | BSD-3-Clause | verified-clean | — |
| TransNetV2 | MIT | verified-clean | — |
| PaddleOCR | Apache-2.0 | verified-clean | — |
| YOLOX | Apache-2.0 | verified-clean | — |
| RT-DETR | Apache-2.0 | verified-clean | — |
| ByteTrack | MIT | verified-clean | — |
| RTMPose (MMPose) | Apache-2.0 | verified-clean | — |
| FFmpeg (as invoked binary) / PyAV | LGPL/GPL-configurable / BSD | verified-clean as separate process (no GPL components enabled) | — |

Maintenance rule: any new model/dataset/weight enters this table **before** it
enters the codebase; "needs-verification" assets may be prototyped against but not
shipped (plan: "Verify before launch"). Update statuses as replies arrive and file
the correspondence in `docs/legal/`.
