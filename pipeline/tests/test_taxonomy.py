from synchro_pipeline.domain.taxonomy import (
    CANONICAL_TO_COARSE,
    CANONICAL_TO_MERGED,
    MERGED_TO_COARSE,
    CoarseShotType,
    MergedShotType,
    ShotType,
)


def test_layer_sizes():
    assert len(ShotType) == 18
    assert len(MergedShotType) == 11
    assert len(CoarseShotType) == 8


def test_mappings_are_total():
    assert set(CANONICAL_TO_MERGED) == set(ShotType)
    assert set(CANONICAL_TO_COARSE) == set(ShotType)
    assert set(MERGED_TO_COARSE) == set(MergedShotType)


def test_every_merged_and_coarse_class_reachable():
    assert set(CANONICAL_TO_MERGED.values()) == set(MergedShotType)
    assert set(CANONICAL_TO_COARSE.values()) == set(CoarseShotType)


def test_canonical_values_are_shuttleset_strings():
    # enum values must round-trip raw ShuttleSet CSV strings
    assert ShotType("smash") is ShotType.SMASH
    assert ShotType("back-court drive") is ShotType.BACK_COURT_DRIVE
    assert ShotType("cross-court net shot") is ShotType.CROSS_COURT_NET_SHOT


def test_spot_mappings():
    assert CANONICAL_TO_MERGED[ShotType.WRIST_SMASH] is MergedShotType.SMASH
    assert CANONICAL_TO_MERGED[ShotType.PASSIVE_DROP] is MergedShotType.DROP
    assert CANONICAL_TO_COARSE[ShotType.DEFENSIVE_RETURN_DRIVE] is CoarseShotType.DRIVE
    assert CANONICAL_TO_COARSE[ShotType.DEFENSIVE_RETURN_LOB] is CoarseShotType.LIFT
    assert CANONICAL_TO_COARSE[ShotType.LONG_SERVICE] is CoarseShotType.SERVE


def test_merged_to_coarse_consistent_except_documented_defense_loss():
    # For every canonical type, going canonical->merged->coarse must agree with
    # canonical->coarse, except through the documented lossy DEFENSE merge.
    for shot in ShotType:
        via_merged = MERGED_TO_COARSE[CANONICAL_TO_MERGED[shot]]
        direct = CANONICAL_TO_COARSE[shot]
        if CANONICAL_TO_MERGED[shot] is MergedShotType.DEFENSE:
            continue
        assert via_merged is direct, shot
