import pytest
from pydantic import ValidationError
from synchro_pipeline.domain.taxonomy import CoarseShotType, MergedShotType, ShotType
from synchro_pipeline.schemas.records import XY, RallyRecord, ScorePair, ShotRecord


def make_shot(**overrides) -> ShotRecord:
    base = dict(
        match_id="m1",
        set_no=1,
        rally_id="m1-g1-r1",
        ball_round=1,
        frame_num=1234,
        t_ms=41133.3,
        side="near",
    )
    base.update(overrides)
    return ShotRecord(**base)


def test_shot_record_json_round_trip():
    shot = make_shot(
        player="A",
        type_canonical=ShotType.SHORT_SERVICE,
        type_merged=MergedShotType.SERVE_SHORT,
        type_coarse=CoarseShotType.SERVE,
        type_probs={MergedShotType.SERVE_SHORT: 0.93, MergedShotType.SERVE_LONG: 0.07},
        hit_xy_court=XY(x=0.5, y=-1.2),
        landing_area=6,
        qa_flags=["inpainted_gap"],
    )
    restored = ShotRecord.model_validate_json(shot.model_dump_json())
    assert restored == shot
    assert restored.type_canonical is ShotType.SHORT_SERVICE


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        make_shot(not_a_field=1)


def test_landing_area_bounds():
    with pytest.raises(ValidationError):
        make_shot(landing_area=17)
    with pytest.raises(ValidationError):
        make_shot(landing_area=0)


def test_low_confidence_shot_carries_nulls_not_guesses():
    # the contract allows a fully-abstained classification: all Nones + a flag
    shot = make_shot(type_canonical=None, type_conf=None, qa_flags=["low_traj_quality"])
    assert shot.type_canonical is None
    assert "low_traj_quality" in shot.qa_flags


def test_rally_record_defaults():
    rally = RallyRecord(
        rally_id="m1-g1-r1",
        match_id="m1",
        set_no=1,
        seq=1,
        start_frame=100,
        end_frame=900,
        duration_s=26.7,
        score_before=ScorePair(a=0, b=0),
        score_after=ScorePair(a=1, b=0),
    )
    assert rally.score_verified is False  # verification is opt-in, never assumed
    assert rally.confidence.hits is None
    assert rally.qa_flags == []
