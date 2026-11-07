import json
from collections import Counter

import pytest

from egorecover.selection import ALGORITHM_VERSION, select_dev_records


def records(counts):
    return [
        {"episode_id": f"{take}/episode{index}", "base_take_name": take}
        for take, count in counts.items()
        for index in range(count)
    ]


def test_small_selection_visits_distinct_takes_before_repeating_any_take():
    rows = records({"A": 20, "B": 2, "C": 3, "D": 1})
    indices, protocol = select_dev_records(rows, count=4)
    assert len({rows[index]["base_take_name"] for index in indices}) == 4
    assert protocol["selected_takes"] == 4
    assert protocol["count"] == 4
    assert protocol["available_count"] == 26


def test_subsets_are_prefix_consistent_and_balanced_across_multiple_rounds():
    rows = records({"A": 5, "B": 5, "C": 5})
    small, _ = select_dev_records(rows, count=4, seed=62)
    large, _ = select_dev_records(rows, count=8, seed=62)
    assert small == large[:4]
    counts = Counter(rows[index]["base_take_name"] for index in large)
    assert sorted(counts.values()) == [2, 3, 3]
    assert len(large) == len(set(large))


def test_exhausted_takes_do_not_block_remaining_episodes_or_create_duplicates():
    rows = records({"A": 1, "B": 2, "C": 5})
    indices, protocol = select_dev_records(rows, count=6)
    counts = Counter(rows[index]["base_take_name"] for index in indices)
    assert counts == {"A": 1, "B": 2, "C": 3}
    assert len(set(protocol["episode_ids"])) == 6


def test_seeded_membership_is_stable_under_input_reordering_and_preserves_source_indices():
    rows = records({"甲": 3, "乙": 4, "丙": 2})
    reversed_rows = list(reversed(rows))
    first, first_protocol = select_dev_records(rows, count=5, seed=-1062)
    second, second_protocol = select_dev_records(reversed_rows, count=5, seed=-1062)
    assert [rows[index]["episode_id"] for index in first] == first_protocol["episode_ids"]
    assert [reversed_rows[index]["episode_id"] for index in second] == first_protocol["episode_ids"]
    assert first_protocol["episode_ids"] == second_protocol["episode_ids"]
    assert first_protocol["population_sha256"] != second_protocol["population_sha256"]
    assert first_protocol["sha256"] != second_protocol["sha256"]
    assert select_dev_records(rows, count=5, seed=-1062) == (first, first_protocol)


def test_full_selection_preserves_official_order_and_count_above_population_is_clipped():
    rows = records({"A": 3, "B": 2})
    indices, protocol = select_dev_records(rows)
    assert indices == list(range(5))
    assert protocol["mode"] == "all_original_order"
    assert protocol["requested_count"] is None
    capped, capped_protocol = select_dev_records(rows, count=100)
    assert sorted(capped) == list(range(5))
    assert capped_protocol["requested_count"] == 100
    assert capped_protocol["count"] == 5


@pytest.mark.parametrize("seed", [0, 1062, -1062, 2**70])
def test_protocol_is_json_serializable_and_binds_selection_population_and_seed(seed):
    rows = records({"A": 3, "B": 3})
    indices, protocol = select_dev_records(rows, count=3, seed=seed)
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["algorithm"] == ALGORITHM_VERSION
    assert protocol["indices"] == indices
    assert protocol["seed"] == seed
    assert len(protocol["sha256"]) == 64
    assert len(protocol["population_sha256"]) == 64
    changed = [dict(row) for row in rows]
    changed[-1]["episode_id"] = "changed-episode"
    assert select_dev_records(changed, count=3, seed=seed)[1]["sha256"] != protocol["sha256"]


def test_empty_and_single_synthetic_record_are_supported_without_mutating_records():
    assert select_dev_records([])[0] == []
    assert select_dev_records([], count=10)[1]["count"] == 0
    row = {"episode_id": "synthetic", "base_take_name": "synthetic", "other_metadata": object()}
    before = dict(row)
    indices, protocol = select_dev_records([row], count=128)
    assert row == before
    assert indices == [0]
    assert protocol["episode_ids"] == ["synthetic"]


@pytest.mark.parametrize("count", [0, -1, 1.5, True, False, "2"])
def test_invalid_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        select_dev_records(records({"A": 2}), count=count)


@pytest.mark.parametrize("seed", [True, False, 1.5, "1062", None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        select_dev_records(records({"A": 2}), seed=seed)


@pytest.mark.parametrize(
    "rows",
    [
        [{}],
        [{"episode_id": "a", "base_take_name": ""}],
        [{"episode_id": 1, "base_take_name": "A"}],
        [{"episode_id": "duplicate", "base_take_name": "A"}, {"episode_id": "duplicate", "base_take_name": "B"}],
    ],
)
def test_missing_identity_or_duplicate_episode_is_rejected(rows):
    with pytest.raises(ValueError, match="nonempty strings|unique"):
        select_dev_records(rows, count=1)
