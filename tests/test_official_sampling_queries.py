from types import SimpleNamespace

import pytest
import torch

from egorecover.official import OfficialWindowData


def sampler():
    data = OfficialWindowData.__new__(OfficialWindowData)
    data.split = "train"
    # Deliberately not alphabetical: retain original take/source insertion order.
    data.source_metadata = {
        "source_zb": {"take": "take_z", "frames": 91},
        "source_a": {"take": "take_a", "frames": 36},
        "source_za": {"take": "take_z", "frames": 180},
        "source_short": {"take": "take_short", "frames": 23},
    }
    data.inventory = {"source_takes": 3}
    data.sampling_inventory = {}
    data._sampling_groups = {}
    return data


def legacy_query_reference(data, generator, count, rollin_frames):
    """The pre-refactor sampling loop, retained as an RNG-compatibility oracle."""
    frame, minimum = 20 + rollin_frames, 22 + rollin_frames
    by_take = {}
    for name, meta in data.source_metadata.items():
        if meta["frames"] >= minimum:
            by_take.setdefault(meta["take"], []).append(name)
    groups = list(by_take.values())
    expected = []
    for _ in range(count):
        names = groups[int(torch.randint(len(groups), (), generator=generator))]
        name = names[int(torch.randint(len(names), (), generator=generator))]
        query = int(torch.randint(frame, data.source_metadata[name]["frames"] - 1, (), generator=generator))
        expected.append((name, query - frame, minimum))
    return expected, frame


def test_planning_does_not_construct_episodes_or_change_source_order(monkeypatch):
    data = sampler()
    monkeypatch.setattr(data, "_episode", lambda *_: pytest.fail("Planning must not load an episode"))
    queries, frame = data.sample_queries(torch.Generator().manual_seed(62), 64, 4)
    assert frame == 24 and len(queries) == 64
    assert data._sampling_groups[4] == [["source_zb", "source_za"], ["source_a"]]
    assert all(minimum == 26 for _, _, minimum in queries)
    assert all(0 <= start <= data.source_metadata[name]["frames"] - minimum for name, start, minimum in queries)
    assert data.sampling_inventory[4] == {
        "required_source_frames": 26,
        "eligible_takes": 2,
        "eligible_sources": 3,
        "skipped_short_sources": 1,
        "skipped_short_only_takes": 1,
    }


def test_planning_matches_legacy_three_draw_loop_and_generator_continuation(monkeypatch):
    data = sampler()
    monkeypatch.setattr(data, "_episode", lambda *_: pytest.fail("No planned query may be loaded"))
    current_rng = torch.Generator().manual_seed(62)
    legacy_rng = torch.Generator().manual_seed(62)
    for count, rollin in ((1, 0), (7, 4), (13, 70), (4, 4), (8, 0)):
        expected = legacy_query_reference(data, legacy_rng, count, rollin)
        actual = data.sample_queries(current_rng, count, rollin)
        assert actual == expected
        assert torch.equal(current_rng.get_state(), legacy_rng.get_state())
    assert torch.equal(torch.rand(5, generator=current_rng), torch.rand(5, generator=legacy_rng))


def test_query_high_remains_t_minus_one_for_shortest_eligible_source(monkeypatch):
    data = sampler()
    data.source_metadata = {"exact": {"take": "only", "frames": 26}}
    data.inventory = {"source_takes": 1}
    monkeypatch.setattr(data, "_episode", lambda *_: pytest.fail("No materialization during planning"))
    queries, frame = data.sample_queries(torch.Generator().manual_seed(62), 3, 4)
    assert frame == 24
    assert queries == [("exact", 0, 26)] * 3


def test_sample_batch_is_the_composition_of_planning_and_materialization(monkeypatch):
    data = sampler()
    calls = []

    def materialize(*query):
        calls.append(query)
        return SimpleNamespace(query=query)

    monkeypatch.setattr(data, "_episode", materialize)
    first_rng = torch.Generator().manual_seed(1062)
    second_rng = torch.Generator().manual_seed(1062)
    queries, frame = data.sample_queries(first_rng, 8, 4)
    assert calls == []
    planned = data.load_training_queries(queries)
    batch, batch_frame = data.sample_batch(second_rng, 8, 4)
    assert batch_frame == frame == 24
    assert [episode.query for episode in batch.episodes] == [episode.query for episode in planned.episodes] == queries
    assert calls == queries + queries
    assert torch.equal(first_rng.get_state(), second_rng.get_state())


def test_materialization_does_not_touch_queries_discarded_after_planning(monkeypatch):
    data, calls = sampler(), []
    monkeypatch.setattr(data, "_episode", lambda *query: calls.append(query) or SimpleNamespace(query=query))
    queries, _ = data.sample_queries(torch.Generator().manual_seed(63), 12, 4)
    assert calls == []
    retained = queries[::3]
    batch = data.load_training_queries(retained)
    assert calls == retained
    assert len(batch) == len(retained)


@pytest.mark.parametrize("count,rollin", [(0, 4), (-1, 4), (True, 4), (2, -1), (2, True)])
def test_planning_retains_argument_validation_without_consuming_rng_or_loading(monkeypatch, count, rollin):
    data = sampler()
    monkeypatch.setattr(data, "_episode", lambda *_: pytest.fail("Invalid plans may not load episodes"))
    generator = torch.Generator().manual_seed(62)
    state = generator.get_state().clone()
    with pytest.raises(ValueError):
        data.sample_queries(generator, count, rollin)
    assert torch.equal(generator.get_state(), state)


def test_validation_split_and_no_eligible_source_are_rejected_before_materialization(monkeypatch):
    data = sampler()
    monkeypatch.setattr(data, "_episode", lambda *_: pytest.fail("Rejected plans may not load episodes"))
    generator = torch.Generator().manual_seed(62)
    state = generator.get_state().clone()
    data.split = "dev"
    with pytest.raises(ValueError, match="training split"):
        data.sample_queries(generator, 1, 4)
    data.split = "train"
    with pytest.raises(ValueError, match="No official sources"):
        data.sample_queries(generator, 1, 1000)
    assert data.sampling_inventory[1000]["skipped_short_sources"] == 4
    assert torch.equal(generator.get_state(), state)
