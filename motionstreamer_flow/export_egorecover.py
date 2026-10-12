"""Read-only conversion of prepared paired caches into the standalone schema."""

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .data import save_json, save_torch, sha256, validate_episode
from .geometry import PARENTS, fk, head_features, matrix_to_rotation_6d, rotation_6d_to_matrix
from .train import resolve_device


def body_to_pose_root(body, anchor):
    transforms = body["joints"].float()
    global_rot = transforms[..., :3, :3]
    local = torch.stack(
        [
            global_rot[:, parent].transpose(-1, -2) @ global_rot[:, child]
            for child, parent in enumerate(PARENTS[1:], start=1)
        ],
        1,
    )
    pose = torch.cat((matrix_to_rotation_6d(local).flatten(1), body["auxiliary"][:, :26].float()), -1)
    relative = torch.linalg.inv(anchor) @ transforms[:, 0]
    root = torch.cat((matrix_to_rotation_6d(relative[:, :3, :3]), relative[:, :3, 3]), -1)
    joints = (transforms[:, :, :3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
    return pose, root, joints


@torch.no_grad()
def skeleton_from_smplx(layer, beta):
    if tuple(int(x) for x in layer.parents[:22]) != PARENTS:
        raise ValueError("Unexpected SMPL-X body joint order.")
    rest = layer(betas=beta[None], return_verts=False).joints[0, :22]
    offsets = torch.zeros_like(rest)
    for child, parent in enumerate(PARENTS[1:], start=1):
        offsets[child] = rest[child] - rest[parent]
    return offsets, rest[0]


@torch.no_grad()
def audit_fk(layer, beta, offsets, rest_pelvis, pose, root):
    selected = torch.linspace(0, len(pose) - 1, min(3, len(pose))).long().to(pose.device)
    pose, root = pose[selected], root[selected]
    local = rotation_6d_to_matrix(pose[:, :126].reshape(-1, 21, 6))
    reference = layer(
        betas=beta[None].expand(len(pose), -1),
        global_orient=rotation_6d_to_matrix(root[:, :6])[:, None],
        body_pose=local,
        transl=root[:, 6:] - rest_pelvis,
        return_verts=False,
    ).joints[:, :22]
    result, _ = fk(pose, root, offsets)
    error = float((result - reference).abs().max())
    if error > 5e-5:
        raise ValueError(f"SMPL-X skeleton parity failed: {error:.8f}m.")
    return error


def convert_packet(packet, cfg, offsets, anchor=None):
    # Only the predicted startup defines the anchor. GT reference fields are never read.
    anchor = packet["teacher"]["reference"][0].float().clone() if anchor is None else anchor
    pose, root, joints = body_to_pose_root(packet["truth"], anchor)
    startup_pose, startup_root, _ = body_to_pose_root(packet["teacher"], anchor)
    t = len(pose)
    head = torch.eye(4).expand(t, 4, 4).clone()
    head[:, :3, :3] = rotation_6d_to_matrix(packet["head"][:, :6])
    head[:, :3, 3] = packet["head"][:, 6:9]
    head_valid = packet.get("head_valid", torch.ones(t, dtype=torch.bool))
    result = {
        "pose": pose,
        "root": root,
        "joints": joints,
        "images": packet["images"].float(),
        "head": head_features(head, anchor, head_valid),
        "image_valid": packet.get("image_valid", torch.ones(t, dtype=torch.bool)),
        "head_valid": head_valid,
        "valid": torch.ones(t, dtype=torch.bool),
        "beta": packet["beta_boot"].float(),
        "offsets": offsets.cpu().float(),
        "anchor": anchor,
        "startup_pose": startup_pose[: cfg.bootstrap],
        "startup_root": startup_root[: cfg.bootstrap],
        "startup_source": "predicted",
    }
    validate_episode(result, cfg)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Existing prepared data directory containing manifest.json and episodes/."
    )
    parser.add_argument("--output", required=True, help="Fresh separate dataset directory; source is never written.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smplx-model-path", required=True, help="Directory containing licensed SMPLX_NEUTRAL.npz.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-per-split", type=int, help="Debug export only; omit for formal comparison.")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    device = resolve_device(args.device, "fp32")
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the existing source tree.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Export requires a fresh output directory.")
    model_path = Path(args.smplx_model_path) / "SMPLX_NEUTRAL.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing licensed neutral model: {model_path}")
    import smplx  # Optional dependency only at offline conversion time.

    layer = (
        smplx.SMPLXLayer(str(model_path.parent), ext="npz", gender="NEUTRAL", use_pca=True, num_pca_comps=12)
        .to(device)
        .eval()
    )
    source_manifest = source / "manifest.json"
    frozen = json.loads(source_manifest.read_text())
    source_digest = sha256(source_manifest)
    records, counts, maximum_error = [], {}, 0.0
    for record in frozen["records"]:
        split, eid = record["group"], record["episode_id"]
        if split not in ("train", "dev", "test"):
            continue
        if Path(eid).name != eid:
            raise ValueError("Unsafe source episode id.")
        if args.limit_per_split is not None and counts.get(split, 0) >= args.limit_per_split:
            continue
        path = source / "episodes" / f"{eid}.pt"
        marker_path = path.with_suffix(".json")
        if not path.is_file() or not marker_path.is_file():
            raise FileNotFoundError(f"Prepared episode or marker missing: {path}")
        marker = json.loads(marker_path.read_text())
        digest = sha256(path)
        if not marker.get("completed") or marker["manifest_sha256"] != source_digest or marker["sha256"] != digest:
            raise ValueError("Incomplete or changed source cache.")
        packet = torch.load(path, map_location="cpu", weights_only=True)
        beta = packet["beta_boot"].to(device).float()
        offsets, rest_pelvis = skeleton_from_smplx(layer, beta)
        result = convert_packet(packet, cfg, offsets)
        error = audit_fk(layer, beta, offsets, rest_pelvis, result["pose"].to(device), result["root"].to(device))
        startup_error = audit_fk(
            layer, beta, offsets, rest_pelvis, result["startup_pose"].to(device), result["startup_root"].to(device)
        )
        maximum_error = max(maximum_error, error, startup_error)
        result["source_sha256"] = digest
        result["fk_parity_max_abs_m"] = max(error, startup_error)
        destination = output / "episodes" / f"{eid}.pt"
        save_torch(destination, result)
        records.append(
            {
                "id": eid,
                "take": record["base_take_name"],
                "split": split,
                "path": f"episodes/{eid}.pt",
                "sha256": sha256(destination),
            }
        )
        counts[split] = counts.get(split, 0) + 1
        print(
            json.dumps({"episode": eid, "split": split, "fk_parity_max_abs_m": result["fk_parity_max_abs_m"]}),
            flush=True,
        )
    if set(counts) != {"train", "dev", "test"}:
        raise ValueError("Export must contain train/dev/test. Partial files exist but no valid manifest was published.")
    save_json(
        output / "manifest.json",
        {
            "version": 1,
            "fps": cfg.fps,
            "backend": "smplx",
            "records": records,
            "source_manifest_sha256": source_digest,
            "smplx_asset_sha256": sha256(model_path),
            "fk_parity_max_abs_m": maximum_error,
            "limited": args.limit_per_split is not None,
            "coordinates": "fixed predicted startup heading/XY anchor; Z is supplied source floor",
            "floor_source": "inherited offline cache floor; not estimated from future frames by this exporter",
        },
    )


if __name__ == "__main__":
    main()
