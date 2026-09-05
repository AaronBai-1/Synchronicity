"""synchro_pipeline — badminton analytics CV/ML pipeline.

Layout:
    domain/      pure badminton knowledge: court geometry, shot taxonomy, scoring rules
    schemas/     Pydantic contracts between pipeline stages (ShuttleSet-compatible)
    stages/      pipeline stages S0-S7 and the DAG runner
    perception/  model wrappers (TrackNetV3, court keypoint net, ...)
    eval/        golden-set metrics and benchmark harness
"""

__version__ = "0.1.0"
