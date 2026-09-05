"""Stage framework: caching, cache invalidation on version bump, dependency ordering."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel
from synchro_pipeline.stages.base import (
    SOURCE_VIDEO,
    PipelineContext,
    PipelineError,
    Stage,
    run_pipeline,
)

CALLS: list[str] = []  # execution log shared by all fake stages


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


class _Payload(BaseModel):
    text: str


class _FakeStage(Stage):
    """Writes one payload per declared output; logs every actual (non-cached) run.

    Outputs mix in the input hashes (like any real transform of its inputs) so
    upstream content changes cascade; `payload` stands in for the model/logic.
    """

    payload: str = "v1"

    def run(self, ctx: PipelineContext) -> None:
        CALLS.append(self.name)
        upstream = ":".join(ctx.artifacts.hash_of(key)[:12] for key in self.inputs)
        for key in self.outputs:
            ctx.artifacts.save_model(
                self, key, _Payload(text=f"{self.payload}:{key}:{upstream}")
            )


class StageA(_FakeStage):
    name = "fake_a"
    version = "0.1.0"
    inputs = (SOURCE_VIDEO,)
    outputs = ("alpha",)


class StageB(_FakeStage):
    name = "fake_b"
    version = "0.1.0"
    inputs = ("alpha",)
    outputs = ("beta",)


class StageC(_FakeStage):
    name = "fake_c"
    version = "0.1.0"
    inputs = ("beta",)
    outputs = ("gamma",)


class StageBv2(StageB):
    """Version bump whose output content also changes (the usual model upgrade)."""

    version = "0.2.0"
    payload = "v2"


class StageBv2SameOutput(StageB):
    """Version bump that happens to produce byte-identical output."""

    version = "0.2.0"


def make_ctx(tmp_path: Path, content: bytes = b"pretend video bytes") -> PipelineContext:
    source = tmp_path / "source.mp4"
    source.write_bytes(content)
    return PipelineContext.create(
        match_id="match-test", workdir=tmp_path / "work", source_video=source
    )


def chain() -> list[Stage]:
    return [StageA(), StageB(), StageC()]


class TestCaching:
    def test_second_run_skips_everything(self, tmp_path):
        ctx = make_ctx(tmp_path)
        first = run_pipeline(chain(), ctx)
        assert CALLS == ["fake_a", "fake_b", "fake_c"]
        assert [s.cached for s in first.stages] == [False, False, False]

        second = run_pipeline(chain(), ctx)
        assert CALLS == ["fake_a", "fake_b", "fake_c"]  # nothing re-ran
        assert [s.cached for s in second.stages] == [True, True, True]

    def test_cache_survives_process_restart(self, tmp_path):
        run_pipeline(chain(), make_ctx(tmp_path))
        CALLS.clear()
        # A fresh context over the same workdir = a new process resuming the match.
        fresh = PipelineContext.create(
            match_id="match-test",
            workdir=tmp_path / "work",
            source_video=tmp_path / "source.mp4",
        )
        report = run_pipeline(chain(), fresh)
        assert CALLS == []
        assert all(s.cached for s in report.stages)

    def test_source_change_invalidates_whole_chain(self, tmp_path):
        ctx = make_ctx(tmp_path)
        run_pipeline(chain(), ctx)
        CALLS.clear()
        (tmp_path / "source.mp4").write_bytes(b"different video bytes")
        fresh = PipelineContext.create(
            match_id="match-test",
            workdir=tmp_path / "work",
            source_video=tmp_path / "source.mp4",
        )
        run_pipeline(chain(), fresh)
        assert CALLS == ["fake_a", "fake_b", "fake_c"]


class TestVersionBumpInvalidation:
    def test_bump_reruns_stage_and_downstream(self, tmp_path):
        ctx = make_ctx(tmp_path)
        run_pipeline(chain(), ctx)
        CALLS.clear()

        report = run_pipeline([StageA(), StageBv2(), StageC()], ctx)
        # A untouched; B re-runs (new version); C re-runs (B's output hash changed).
        assert CALLS == ["fake_b", "fake_c"]
        assert [s.cached for s in report.stages] == [True, False, False]

    def test_bump_with_identical_output_stops_the_cascade(self, tmp_path):
        ctx = make_ctx(tmp_path)
        run_pipeline(chain(), ctx)
        CALLS.clear()

        report = run_pipeline([StageA(), StageBv2SameOutput(), StageC()], ctx)
        # B re-runs, but its bytes are identical, so C's cache key is unchanged.
        assert CALLS == ["fake_b"]
        assert [s.cached for s in report.stages] == [True, False, True]


class TestOrdering:
    def test_scrambled_input_order_is_fixed_by_dependencies(self, tmp_path):
        ctx = make_ctx(tmp_path)
        run_pipeline([StageC(), StageA(), StageB()], ctx)
        assert CALLS == ["fake_a", "fake_b", "fake_c"]

    def test_report_lists_stages_in_execution_order(self, tmp_path):
        ctx = make_ctx(tmp_path)
        report = run_pipeline([StageB(), StageC(), StageA()], ctx)
        assert [s.stage for s in report.stages] == ["fake_a", "fake_b", "fake_c"]

    def test_missing_input_is_a_hard_error(self, tmp_path):
        ctx = make_ctx(tmp_path)
        with pytest.raises(PipelineError, match="unsatisfied inputs"):
            run_pipeline([StageB()], ctx)  # nothing produces 'alpha'

    def test_duplicate_producers_rejected(self, tmp_path):
        class StageA2(StageA):
            name = "fake_a2"

        ctx = make_ctx(tmp_path)
        with pytest.raises(PipelineError, match="declared by both"):
            run_pipeline([StageA(), StageA2()], ctx)


class TestOutputContract:
    def test_missing_declared_output_rejected(self, tmp_path):
        class Lazy(_FakeStage):
            name = "fake_lazy"
            version = "0.1.0"
            inputs = (SOURCE_VIDEO,)
            outputs = ("alpha", "never_written")

            def run(self, ctx: PipelineContext) -> None:
                ctx.artifacts.save_model(self, "alpha", _Payload(text="only one"))

        with pytest.raises(PipelineError, match="did not produce"):
            run_pipeline([Lazy()], make_ctx(tmp_path))

    def test_undeclared_output_rejected(self, tmp_path):
        class Chatty(_FakeStage):
            name = "fake_chatty"
            version = "0.1.0"
            inputs = (SOURCE_VIDEO,)
            outputs = ("alpha",)

            def run(self, ctx: PipelineContext) -> None:
                ctx.artifacts.save_model(self, "alpha", _Payload(text="ok"))
                ctx.artifacts.save_model(self, "sneaky", _Payload(text="not declared"))

        with pytest.raises(PipelineError, match="undeclared"):
            run_pipeline([Chatty()], make_ctx(tmp_path))


class TestDagIntegrity:
    def test_default_pipeline_orders_cleanly(self, tmp_path):
        """The full S0-S7 DAG (stubs included) must be orderable from source_video."""
        from synchro_pipeline.stages import default_pipeline
        from synchro_pipeline.stages.base import order_stages

        stages = default_pipeline()
        ordered = order_stages(list(reversed(stages)), frozenset({SOURCE_VIDEO}))
        assert [s.name for s in ordered][0] == "s0_ingest"
        assert {s.name for s in ordered} == {s.name for s in stages}

    def test_modal_app_imports_without_modal(self):
        import synchro_pipeline.modal_app as modal_app

        assert hasattr(modal_app, "app")
        group_names = {n for names in modal_app.STAGE_GROUPS.values() for n in names}
        from synchro_pipeline.stages import default_pipeline

        assert group_names == {s.name for s in default_pipeline()}
