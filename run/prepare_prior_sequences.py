"""Prepare a small audited GT-body dataset once for independent P experiments."""

import argparse
import json
import os
from pathlib import Path

import torch

from dataset.smpl_utils import get_smpl
from egorecover.annotations import body_states_from_supervision
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.evaluation_protocol import file_sha256, load_fixed_split
from egorecover.evaluation_resume import atomic_json
from egorecover.prior_two_forward import FIELDS
from egorecover.smpl_evaluation import prepare_ground_truth
from run.adapt_prior_on_predictions import save_torch
from run.complete_stages import BOOTSTRAP, ROOT, SIGNAL, SPLIT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new data output directory.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    manifest = json.loads(Path(SPLIT).read_text())
    dataset, _ = open_dataset(signal=Path(SIGNAL), spec_sha256=manifest["dataset_spec_sha256"])
    splits = load_fixed_split(SPLIT, dataset)
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    codec = MotionCodec(stats)
    bootstrap = ModelBootstrapShapes(BOOTSTRAP, allowed_takes=sum(splits.values(), []),
        stats_sha256=file_sha256(stats_path), split_manifest_sha256=file_sha256(SPLIT),
        dataset_spec_sha256=manifest["dataset_spec_sha256"])
    if bootstrap.identity["reference_mode"] != codec.reference_mode:
        raise ValueError("Bootstrap and P reference conventions differ.")
    smpl = get_smpl().to(args.device).eval().requires_grad_(False)
    asset = Path(os.environ.get("SMPLX_MODEL_PATH", ROOT / "body_models/smplx")) / "SMPLX_NEUTRAL.npz"
    records = sorted([r for r in dataset.records if r["variant_name"] == "clean"
                      and r["base_take_name"] in splits["train"] + splits["dev"]], key=lambda r: r["base_take_name"])
    if len(records) != 60 or len({r["base_take_name"] for r in records}) != 60:
        raise ValueError("Expected the fixed 48 train / 12 dev clean episodes.")
    values = {key: [] for key in FIELDS}
    betas, audits = [], {}
    for record in records:
        if record["num_frames"] != 200 or record["bootstrap_frames"] != 20:
            raise ValueError("Expected 200-frame episodes with a 20-frame startup.")
        arrays = dataset.arrays(record["variant_id"], verify_hash=True)
        head = torch.from_numpy(arrays["aria_traj_obs"].copy())
        supervision = dataset.supervision(record["variant_id"])
        startup = bootstrap.startup_for_record(record)
        floor = float(startup["floor_estimate_m"])
        states = body_states_from_supervision(supervision, head, floor_height=floor, contact_floor_height=floor)
        ground_truth = prepare_ground_truth(smpl, supervision, list(range(200)))
        audits[record["base_take_name"]] = ground_truth["asset_audit"]
        for key in FIELDS:
            values[key].append(getattr(states, key))
        betas.append(startup["beta_boot"])
        print("Audited", record["base_take_name"], flush=True)
    identity = {"scope": "audited_prior_train_dev_sequences", "splits": splits, "holdout_used": False,
                "dataset_spec_sha256": manifest["dataset_spec_sha256"], "split_manifest_sha256": file_sha256(SPLIT),
                "stats_sha256": file_sha256(stats_path), "bootstrap_cache_sha256": file_sha256(BOOTSTRAP),
                "bootstrap_identity": bootstrap.identity, "reference_mode": codec.reference_mode,
                "smplx_asset_sha256": file_sha256(asset), "preparation_code_sha256": file_sha256(__file__)}
    path = args.output / "sequences.pt"
    save_torch(path, {"identity": identity, "stats": stats, "take_names": [r["base_take_name"] for r in records],
                     "beta_boot": torch.stack(betas), **{key: torch.stack(value) for key, value in values.items()}})
    atomic_json(args.output / "report.json", {"identity": identity, "audits": audits, "sequences": len(records),
                "cache_sha256": file_sha256(path), "completed": True})


if __name__ == "__main__":
    main()
