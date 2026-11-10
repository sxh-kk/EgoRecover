import json

import pytest

from egorecover.reference_statistics import summarize_reference_rows


def row(take="A", episode="episode-a", anchor=20, draw=0, delta=2.0, horizon=1, intervention="body_pose", **metadata):
    return {
        "take": take,
        "episode": episode,
        "anchor": anchor,
        "draw": draw,
        "horizon": horizon,
        "intervention": intervention,
        "predicted_mm": 100.0,
        "reference_mm": 100.0 - delta,
        "delta_mm": delta,
        "response_mm": abs(delta),
        "same_shape_predicted_mm": 80.0,
        "same_shape_reference_mm": 80.0 - delta / 2,
        "same_shape_delta_mm": delta / 2,
        "center_predicted_mm": 120.0,
        "center_reference_mm": 120.0 - delta / 4,
        "center_delta_mm": delta / 4,
        **metadata,
    }


def table(rows, **kwargs):
    return summarize_reference_rows(rows, **kwargs)["main"]["body_pose"]["1"]


def test_paired_draws_are_averaged_before_classifying_an_anchor():
    rows = [row(draw=0, delta=10.0), row(draw=1, delta=-8.0)]
    result = table(rows, bootstrap_samples=0)
    assert result["means"]["delta_mm"] == 1.0
    assert result["means"]["reference_mm"] == 99.0
    assert result["fractions"] == {"reference_better": 0.0, "approximately_equal": 1.0, "reference_worse": 0.0}
    assert result["percentages"]["approximately_equal"] == 100.0
    assert result["counts"] == {"raw_draws": 2, "anchor_units": 1, "episodes": 1, "takes": 1}


def test_anchors_then_episodes_then_takes_are_equally_weighted():
    rows = [row(episode="a1", draw=draw, delta=10.0) for draw in range(10)]
    rows += [row(episode="a1", anchor=40, delta=-10.0), row(episode="a2", delta=10.0)]
    rows += [row(take="B", episode="b1", delta=-20.0)]
    result = table(rows, bootstrap_samples=0)
    assert result["counts"] == {"raw_draws": 13, "anchor_units": 4, "episodes": 3, "takes": 2}
    assert result["means"]["delta_mm"] == -7.5
    assert result["means"]["reference_mm"] == 107.5
    assert result["means"]["same_shape_delta_mm"] == -3.75
    assert result["means"]["center_delta_mm"] == -1.875
    assert result["fractions"]["reference_better"] == 0.375
    assert result["fractions"]["reference_worse"] == 0.625
    assert result["per_take"]["A"]["means"]["delta_mm"] == 5.0
    assert result["per_take"]["B"]["means"]["delta_mm"] == -20.0


def test_correlated_draw_replication_cannot_change_take_cluster_ci_or_effective_units():
    original = [row(delta=10.0), row(take="B", episode="b", delta=-10.0)]
    repeated = [{**item, "draw": draw} for item in original for draw in range(40)]
    first = table(original, bootstrap_samples=256, bootstrap_seed=62)
    second = table(repeated, bootstrap_samples=256, bootstrap_seed=62)
    assert first["means"] == second["means"]
    assert first["fractions"] == second["fractions"]
    assert first["ci95"] == second["ci95"]
    assert first["bootstrap"] == second["bootstrap"]
    assert first["ci95"]["delta_mm"] == [-10.0, 10.0]
    assert first["counts"]["takes"] == second["counts"]["takes"] == 2
    assert first["counts"]["anchor_units"] == second["counts"]["anchor_units"] == 2
    assert first["counts"]["raw_draws"] == 2 and second["counts"]["raw_draws"] == 80


def test_delta_direction_and_zero_one_five_mm_tolerance_sensitivity():
    rows = [row(anchor=20 + index, delta=delta) for index, delta in enumerate((0.0, 1.0, 5.0, 6.0, -1.0, -5.0, -6.0))]
    result = table(rows, bootstrap_samples=0)
    assert result["sensitivity"]["0.0"] == pytest.approx(
        {"reference_better": 3 / 7, "approximately_equal": 1 / 7, "reference_worse": 3 / 7}
    )
    assert result["sensitivity"]["1.0"] == pytest.approx(
        {"reference_better": 2 / 7, "approximately_equal": 3 / 7, "reference_worse": 2 / 7}
    )
    assert result["sensitivity"]["5.0"] == pytest.approx(
        {"reference_better": 1 / 7, "approximately_equal": 5 / 7, "reference_worse": 1 / 7}
    )
    assert table([row(delta=3)], bootstrap_samples=0)["fractions"]["reference_better"] == 1
    assert table([row(delta=-3)], bootstrap_samples=0)["fractions"]["reference_worse"] == 1


def test_ci_is_deterministic_under_row_reordering_and_shared_for_identical_take_sets():
    rows = [row(take=name, episode=name, delta=delta) for name, delta in (("A", 5.0), ("B", 10.0), ("C", 15.0))]
    first = summarize_reference_rows(rows, bootstrap_samples=100, bootstrap_seed=-1062)
    assert first == summarize_reference_rows(list(reversed(rows)), bootstrap_samples=100, bootstrap_seed=-1062)
    second = table(
        [{**item, "delta_mm": item["delta_mm"] * 2, "reference_mm": 100 - 2 * item["delta_mm"]} for item in rows],
        bootstrap_samples=100,
        bootstrap_seed=-1062,
    )
    assert first["main"]["body_pose"]["1"]["bootstrap"] == second["bootstrap"]
    assert first["main"]["body_pose"]["1"]["ci95"]["delta_mm"][0] > 0


def test_tables_separate_interventions_horizons_anchors_and_official_parent_tasks():
    rows = [
        row(take="A", episode="a", anchor=20, delta=4, parent_task_id=100, parent_task_name="Sport", task_id=101),
        row(take="A", episode="a", anchor=60, delta=-4, parent_task_id=100, parent_task_name="Sport", task_id=101),
        row(take="B", episode="b", anchor=20, delta=8, parent_task_id=200, parent_task_name="Cooking", task_id=201),
        row(take="A", episode="a", anchor=20, delta=12, horizon=2, parent_task_id=100, parent_task_name="Sport"),
        row(
            take="A",
            episode="a",
            anchor=20,
            delta=16,
            intervention="full_state",
            parent_task_id=100,
            parent_task_name="Sport",
        ),
    ]
    result = summarize_reference_rows(rows, bootstrap_samples=30)
    assert set(result["main"]) == {"body_pose", "full_state"}
    assert set(result["main"]["body_pose"]) == {"1", "2"}
    assert result["main"]["body_pose"]["1"]["means"]["delta_mm"] == 4
    assert result["by_anchor"]["20"]["body_pose"]["1"]["means"]["delta_mm"] == 6
    assert result["by_parent_task"]["id:100"]["parent_task_name"] == "Sport"
    assert result["by_parent_task"]["id:100"]["main"]["body_pose"]["1"]["means"]["delta_mm"] == 0
    assert result["by_anchor"]["20"]["body_pose"]["1"]["ci95"] == {}
    assert result["by_parent_task"]["id:200"]["main"]["body_pose"]["1"]["bootstrap"] is None
    json.dumps(result, allow_nan=False)


def test_optional_current_errors_and_unknown_parent_task_are_supported():
    result = summarize_reference_rows(
        [row(anchor_current_mm=30.0, intervened_current_mm=0.0, seconds=0.1)], bootstrap_samples=0
    )
    assert result["main"]["body_pose"]["1"]["means"]["anchor_current_mm"] == 30.0
    assert result["main"]["body_pose"]["1"]["means"]["intervened_current_mm"] == 0.0
    assert set(result["by_parent_task"]) == {"unknown"}


def test_empty_rows_have_explicit_empty_state():
    result = summarize_reference_rows([])
    assert result["status"] == "empty"
    assert result["main"] == result["by_anchor"] == result["by_parent_task"] == {}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance_mm": -1},
        {"tolerance_mm": float("nan")},
        {"bootstrap_samples": -1},
        {"bootstrap_samples": True},
        {"bootstrap_seed": True},
    ],
)
def test_invalid_statistics_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        summarize_reference_rows([], **kwargs)


def test_duplicate_draw_nonfinite_metric_and_reversed_delta_are_rejected():
    item = row()
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_reference_rows([item, item])
    with pytest.raises(ValueError, match="finite"):
        summarize_reference_rows([{**item, "response_mm": float("inf")}])
    with pytest.raises(ValueError, match="predicted minus reference"):
        summarize_reference_rows([{**item, "delta_mm": -item["delta_mm"]}])


def test_episode_equal_estimate_keeps_take_clusters_and_averages_draws_before_classification():
    rows = [row(episode="a1", delta=10), row(episode="a2", delta=10), row(take="B", episode="b", delta=-20)]
    episode = table(rows, aggregation="episode", bootstrap_samples=128)
    take = table(rows, aggregation="take", bootstrap_samples=128)
    assert episode["means"]["delta_mm"] == 0
    assert take["means"]["delta_mm"] == -5
    assert episode["fractions"]["reference_better"] == pytest.approx(2 / 3)
    assert take["fractions"]["reference_better"] == 0.5
    assert episode["counts"]["takes"] == 2
    repeated = [{**r, "draw": d} for r in rows for d in range(7)]
    again = table(repeated, aggregation="episode", bootstrap_samples=128)
    assert episode["means"] == again["means"]
    assert episode["ci95"] == again["ci95"]
    assert episode["bootstrap"]["index_sha256"] == take["bootstrap"]["index_sha256"]
    with pytest.raises(ValueError, match="aggregation"):
        table(rows, aggregation="draw")
