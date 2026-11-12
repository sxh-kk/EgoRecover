from copy import deepcopy

import pytest
import torch

from scripts.formal_runtime_migration import assert_same_state, migrate_checkpoint


def test_runtime_migration_preserves_optimizer_ema_rng_schedule_step_and_protocol():
    old = {"fk.py": "old", "training.py": "same"}
    new = {**old, "fk.py": "new"}
    saved = {
        "identity": {"code": old, "protocol": {"window": 3}, "batch_size": 16},
        "step": 1000,
        "optimizer": {"state": {0: {"step": torch.tensor(1000), "exp_avg": torch.randn(3)}}},
        "ema": {"weight": torch.randn(4)},
        "rank_rng_states": [{"cpu_rng": torch.get_rng_state()}],
        "scheduler": {"last_epoch": 1000},
    }
    migrated = migrate_checkpoint(saved, original_code=old, reviewed_code=new)
    assert_same_state(saved, migrated)
    assert saved["identity"]["code"] == old
    assert migrated["identity"]["code"] == new
    corrupt = deepcopy(migrated)
    corrupt["optimizer"]["state"][0]["exp_avg"][0] += 1
    with pytest.raises(AssertionError, match="Tensor values changed"):
        assert_same_state(saved, corrupt)
    corrupt = deepcopy(migrated)
    corrupt["identity"]["batch_size"] = 8
    with pytest.raises(AssertionError, match="Value changed"):
        assert_same_state(saved, corrupt)


def test_runtime_migration_rejects_other_loss_training_data_or_sampling_changes():
    old = {"fk.py": "old", "training.py": "same"}
    saved = {"identity": {"code": old}}
    for file in ("losses.py", "training.py", "sampling.py", "new.py"):
        with pytest.raises(ValueError, match="Unreviewed"):
            migrate_checkpoint(saved, original_code=old, reviewed_code={**old, "fk.py": "new", file: "changed"})
    with pytest.raises(ValueError, match="original reviewed"):
        migrate_checkpoint(saved, original_code={"fk.py": "different"}, reviewed_code=old)


def test_packed_runtime_requires_explicit_reviewed_revision():
    old = {"training.py": "old", "sampling.py": "same"}
    new = {**old, "training.py": "packed", "rollin.py": "new"}
    saved = {"identity": {"code": old}, "step": 1000}
    with pytest.raises(ValueError, match="Unreviewed"):
        migrate_checkpoint(saved, original_code=old, reviewed_code=new)
    upgraded = migrate_checkpoint(saved, original_code=old, reviewed_code=new, revision="runtime3")
    assert_same_state(saved, upgraded)
    with pytest.raises(ValueError, match="Unreviewed"):
        migrate_checkpoint(saved, original_code=old, reviewed_code={**new, "sampling.py": "bad"}, revision="runtime3")
