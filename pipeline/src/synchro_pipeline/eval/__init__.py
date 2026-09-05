"""Golden-set evaluation: metrics, annotation schemas, benchmark harness.

docs/plan.md "Verification approach": the golden eval set is CI — per-stage metrics
(rally F1, homography reprojection/PCK, shuttle F1@5px, hit F1@±3 frames) are wired
into the repo so every model change answers "did the numbers move".
"""
