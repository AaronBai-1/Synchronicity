"""The docs sample golden file must always validate — it is what labellers copy from."""

from pathlib import Path

from synchro_pipeline.eval.golden import load_golden_match

SAMPLE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "sample-golden.json"


def test_sample_golden_file_is_valid():
    match = load_golden_match(SAMPLE)
    assert match.match_id == "2023-denmark-open-msf"
    assert len(match.rallies) == 3
    assert sum(len(r.hits) for r in match.rallies) == 16
    assert len(match.court_labels) == 2
    # rally 1 carries the optional shuttle labels; 2-3 are deliberately unlabelled
    assert match.rallies[0].shuttle is not None
    assert match.rallies[1].shuttle is None
