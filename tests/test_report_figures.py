import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from ops.build_anime_report_figures import paired_differences, paired_gains, summarize


@pytest.fixture
def data():
    path = Path(__file__).resolve().parents[1] / "evidence/anime_report_figure_data_20260926.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_summary_uses_sample_sd_and_keeps_popular_single_run(data):
    synthetic = copy.deepcopy(data)
    rows = synthetic["final"]["models"]["neumf"]["per_seed"]
    for row, value in zip(rows, [.2, .4, .6]):
        row["test_ndcg@10"] = value
    result = summarize(synthetic)
    assert result["neumf"]["test_ndcg@10"] == pytest.approx((.4, .2))
    assert result["popular"]["test_ndcg@10"][1] is None


def test_summary_rejects_duplicate_seed(data):
    data["final"]["models"]["neumf"]["per_seed"][1]["seed"] = 42
    with pytest.raises(ValueError, match="seeds"):
        summarize(data)


def test_component_gains_pair_by_seed_not_input_order(data):
    expected = paired_gains(data, "bpr_ensemble_component")
    data["final"]["models"]["bpr_ensemble_component"]["per_seed"].reverse()
    assert paired_gains(data, "bpr_ensemble_component") == expected
    assert min(expected) > 0
    # The separately selected standalone BPR must not be substituted as component.
    assert expected != paired_gains(data, "bpr")


def test_baseline_difference_uncertainty_uses_paired_observations(data):
    import statistics

    model = data["final"]["models"]["neumf"]["per_seed"]
    baseline = data["final"]["models"]["bpr"]["per_seed"]
    for row, value in zip(baseline, [.3, .5, .7]):
        row["test_ndcg@10"] = value
    for row, value in zip(model, [.32, .54, .69]):
        row["test_ndcg@10"] = value
    baseline.reverse()
    deltas = paired_differences(data, "neumf", "bpr")
    assert deltas == pytest.approx([.02, .04, -.01])
    assert statistics.stdev(deltas) == pytest.approx(statistics.stdev([.02, .04, -.01]))


def test_paired_differences_reject_missing_baseline_seed(data):
    data["final"]["models"]["bpr"]["per_seed"].pop()
    with pytest.raises(ValueError, match="seeds"):
        paired_differences(data, "neumf", "bpr")
