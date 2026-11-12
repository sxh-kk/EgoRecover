"""Strict, explicit checkpoint identity migration after runtime-only review.

The caller archives the original checkpoint and identity before atomically
publishing the migrated files. No training or tensor state is modified here.
"""

from collections.abc import Mapping

import torch

ALLOWED_PACKAGE_CHANGES = frozenset({"fk.py", "flow.py", "state.py", "models/motion.py", "models/layers.py"})
PACKED_PACKAGE_CHANGES = frozenset({"training.py", "rollin.py", "models/motion.py", "losses.py", "system.py"})


def migrate_checkpoint(saved, *, original_code, reviewed_code, revision="runtime2"):
    if saved["identity"]["code"] != original_code:
        raise ValueError("Checkpoint does not match the original reviewed source snapshot.")
    changed = {
        key for key in original_code.keys() | reviewed_code.keys() if original_code.get(key) != reviewed_code.get(key)
    }
    if revision not in ("runtime2", "runtime3"):
        raise ValueError("Unknown reviewed runtime revision.")
    allowed = ALLOWED_PACKAGE_CHANGES if revision == "runtime2" else PACKED_PACKAGE_CHANGES
    if not changed or not changed <= allowed:
        raise ValueError(f"Unreviewed runtime package changes: {sorted(changed)}.")
    return {**saved, "identity": {**saved["identity"], "code": dict(reviewed_code)}}


def assert_same_state(original, migrated, *, path="checkpoint"):
    """Check all values bit-for-bit except the explicitly migrated code hashes."""
    if path == "checkpoint.identity.code":
        return
    if isinstance(original, torch.Tensor):
        if (
            not isinstance(migrated, torch.Tensor)
            or original.dtype != migrated.dtype
            or original.shape != migrated.shape
        ):
            raise AssertionError(f"Tensor metadata changed: {path}.")
        if not torch.equal(original, migrated):
            raise AssertionError(f"Tensor values changed: {path}.")
    elif isinstance(original, Mapping):
        if not isinstance(migrated, Mapping) or original.keys() != migrated.keys():
            raise AssertionError(f"Mapping keys changed: {path}.")
        for key in original:
            assert_same_state(original[key], migrated[key], path=f"{path}.{key}")
    elif isinstance(original, (list, tuple)):
        if type(original) is not type(migrated) or len(original) != len(migrated):
            raise AssertionError(f"Sequence changed: {path}.")
        for index, (a, b) in enumerate(zip(original, migrated)):
            assert_same_state(a, b, path=f"{path}.{index}")
    elif type(original) is not type(migrated) or original != migrated:
        raise AssertionError(f"Value changed: {path}.")
