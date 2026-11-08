import pytest
import torch

from egorecover.sampling import reference_microbatch_plan


class QueryData:
    """Keep the official three scalar randint draws without loading any data."""

    groups = (("a", "b"), ("c",), ("d", "e", "f"))
    lengths = {"a": 40, "b": 80, "c": 65, "d": 97, "e": 42, "f": 119}

    def __init__(self):
        self.calls = []

    def sample_queries(self, generator, count, rollin_frames):
        self.calls.append((count, rollin_frames))
        frame, minimum = 20 + rollin_frames, 22 + rollin_frames
        queries = []
        for _ in range(count):
            names = self.groups[int(torch.randint(len(self.groups), (), generator=generator))]
            name = names[int(torch.randint(len(names), (), generator=generator))]
            query = int(torch.randint(frame, self.lengths[name] - 1, (), generator=generator))
            queries.append((name, query - frame, minimum))
        return queries, frame

    def _episode(self, *args, **kwargs):
        raise AssertionError("A sampling plan must never construct episodes")

    def load(self, *args, **kwargs):
        raise AssertionError("A sampling plan must never load payloads")


def reference_step(data, generator, rank, probability):
    """Independent transcription of the old two-worker, batch32/micro8 loop."""
    result = []
    for offset in range(0, 32, 8):
        queries, frame = data.sample_queries(generator, 8, 4)
        tau = 0.001 + 0.999 * torch.rand(8, generator=generator)
        tau[torch.rand(8, generator=generator) < probability] = 1.0
        result.append({"global_offset": rank * 32 + offset, "queries": queries, "frame": frame, "tau": tau})
    return result


def plan(data, generator, rank=0, world_size=4, **changes):
    options = {
        "rank": rank,
        "world_size": world_size,
        "reference_world_size": 2,
        "batch_size": 64,
        "microbatch_size": 8,
        "rollin_frames": 4,
        "endpoint_probability": 0.1,
        **changes,
    }
    return reference_microbatch_plan(data, generator, **options)


def slots(chunks):
    result = {}
    for chunk in chunks:
        for position, query in enumerate(chunk["queries"]):
            result[chunk["global_offset"] + position] = (query, chunk["frame"], chunk["tau"][position].item())
    return result


@pytest.mark.parametrize("probability", (0.0, 0.1, 1.0))
def test_four_workers_replay_three_old_two_worker_steps_exactly(probability):
    reference_rng = [torch.Generator().manual_seed(63 + rank) for rank in range(2)]
    same_size_rng = [torch.Generator().manual_seed(63 + rank) for rank in range(2)]
    physical_rng = [torch.Generator().manual_seed(63 + rank // 2) for rank in range(4)]
    physical_data = [QueryData() for _ in range(4)]
    for _ in range(3):
        expected = [
            chunk for rank in range(2) for chunk in reference_step(QueryData(), reference_rng[rank], rank, probability)
        ]
        same_size = [
            chunk
            for rank in range(2)
            for chunk in plan(QueryData(), same_size_rng[rank], rank, 2, endpoint_probability=probability)
        ]
        actual = []
        for rank in range(4):
            chunks = plan(physical_data[rank], physical_rng[rank], rank, endpoint_probability=probability)
            assert len(chunks) == 2
            assert [chunk["offset"] for chunk in chunks] == [0, 8]
            assert [chunk["global_offset"] for chunk in chunks] == [rank * 16, rank * 16 + 8]
            assert sum(chunk["count"] for chunk in chunks) == 16
            assert all(chunk["tau"].device.type == "cpu" and chunk["tau"].shape == (8,) for chunk in chunks)
            actual.extend(chunks)
            torch.testing.assert_close(
                physical_rng[rank].get_state(), reference_rng[rank // 2].get_state(), atol=0, rtol=0
            )
        assert len(slots(actual)) == 64
        assert slots(actual) == slots(expected) == slots(same_size)
        for rank in range(2):
            torch.testing.assert_close(same_size_rng[rank].get_state(), reference_rng[rank].get_state(), atol=0, rtol=0)
            torch.testing.assert_close(
                physical_rng[2 * rank].get_state(), physical_rng[2 * rank + 1].get_state(), atol=0, rtol=0
            )
    assert all(data.calls == [(8, 4)] * 12 for data in physical_data)


@pytest.mark.parametrize(
    "changes",
    [
        {"rank": -1},
        {"rank": 4},
        {"rank": True},
        {"world_size": 0},
        {"world_size": 3},
        {"reference_world_size": 0},
        {"reference_world_size": 8},
        {"batch_size": 0},
        {"batch_size": 66},
        {"microbatch_size": 0},
        {"microbatch_size": 6},
        {"microbatch_size": 32},
        {"rollin_frames": -1},
        {"rollin_frames": True},
        {"endpoint_probability": -0.1},
        {"endpoint_probability": 1.1},
        {"endpoint_probability": float("nan")},
        {"endpoint_probability": True},
    ],
)
def test_invalid_plan_is_rejected_before_consuming_rng_or_queries(changes):
    generator, data = torch.Generator().manual_seed(63), QueryData()
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        plan(data, generator, **changes)
    assert data.calls == []
    torch.testing.assert_close(generator.get_state(), before, atol=0, rtol=0)


def test_non_generator_is_rejected_before_sampling():
    data = QueryData()
    with pytest.raises(ValueError, match="CPU torch.Generator"):
        plan(data, None)
    assert data.calls == []
