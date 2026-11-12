"""Matched real-data K8 microbatch benchmark; not a DDP throughput estimate.

Run the same script with PYTHONPATH pointing to each frozen source revision.
All weights, episodes, source noise, Flow times and dropout seeds are fixed.
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from egorecover.bootstrap import HeadAlignedBootstrap
from egorecover.codec import MotionCodec
from egorecover.config import ModelConfig, WindowConfig
from egorecover.data import TrainingBatch
from egorecover.fk import FixedShapeFK, load_smpl
from egorecover.io import code_identity, save_checkpoint, save_json
from egorecover.losses import multi_step_objective
from egorecover.models import MotionModel
from egorecover.official import OfficialWindowData
from egorecover.system import WindowReconstructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--anchors", type=int, choices=(2, 4), default=2)
    parser.add_argument("--packed-rollin", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(62)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    cfg = WindowConfig(**saved["protocol"])
    model = MotionModel(cfg, ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    if args.eval_compile:
        model.enable_eval_compile()
    if args.bf16:
        original_forward = model.forward

        def mixed_forward(*inputs, **kwargs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return original_forward(*inputs, **kwargs).float()

        model.forward = mixed_forward
    system = WindowReconstructor(MotionCodec(saved["stats"]).to(device), None, cfg)
    smpl = load_smpl(device)
    data = OfficialWindowData(args.data_root, "val", device=device, bootstrap=HeadAlignedBootstrap.from_smpl(smpl))
    selection = json.loads(args.selection.read_text())
    episodes = tuple(data.load(row["record_index"]) for row in selection["records"][: args.anchors])
    batches = [TrainingBatch(episodes[i : i + 2]) for i in range(0, len(episodes), 2)]
    frame = 40
    prepared = []
    for index, batch in enumerate(batches):
        prepared.append(
            (
                [batch.observations(frame + i) for i in range(8)],
                [batch.target(frame + i, cfg.window) for i in range(8)],
                torch.stack(
                    [batch.noise(1063, frame + i, cfg.window, tag="benchmark", offset=index * 2) for i in range(8)]
                ),
                torch.full((2,), 0.5, device=device),
                torch.stack(
                    [e.timestamps[frame + 1 : frame + 8] - e.timestamps[frame : frame + 7] for e in batch.episodes]
                ),
                FixedShapeFK(smpl, batch.beta, shape_mode="predicted"),
            )
        )

    def run():
        model.zero_grad(set_to_none=True)
        caches = None
        if args.packed_rollin:
            from egorecover.rollin import RollinChunk, packed_rollin_caches

            caches = packed_rollin_caches(
                system, model, [RollinChunk(b, frame, i * 2, 20) for i, b in enumerate(batches)], seed=1063
            )
        rows, codes = [], []
        for index, batch in enumerate(batches):
            model.eval()
            cache = batch.cache(20) if caches is None else caches[index]
            with torch.autocast("cuda", enabled=False) if args.bf16 else nullcontext():
                if caches is None:
                    for t in range(20, frame):
                        cache = system.online(
                            model,
                            cache,
                            batch.observations(t),
                            batch.noise(1063, t, cfg.window, tag="rollin", offset=index * 2),
                        ).cache
            model.train()
            observations, targets, noise, tau, dt, fk = prepared[index]
            with torch.autocast("cuda", enabled=False) if args.bf16 else nullcontext():
                result = multi_step_objective(
                    system,
                    model,
                    None,
                    cache.detach(),
                    observations,
                    targets,
                    noise,
                    tau,
                    fk,
                    dt=dt,
                    velocity_weight=0.013409926806430328,
                    slow_weight=0,
                    forward_seed=123 + index,
                )
                (result["loss"] / len(batches)).backward()
            rows.append({key: result[key].detach() for key in ("loss", "flow", "current", "velocity", "slow")})
            codes.append(result["result"].codes.detach())
        return {**{key: torch.stack([row[key] for row in rows]).mean() for key in rows[0]}, "codes": torch.cat(codes)}

    run()
    timings = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = run()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    scalars = {key: float(result[key].detach()) for key in ("loss", "flow", "current", "velocity", "slow")}
    save_checkpoint(
        args.output.with_suffix(".pt"),
        {
            "scalars": scalars,
            "codes": result["codes"].cpu(),
            "gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None},
        },
    )
    output = {
        "scope": "Actual formal1000 model, W%d mature20 rollin plus K8 forward and backward; no optimizer or DDP collectives. GPU may be shared; inspect experiment record."
        % cfg.window,
        "checkpoint_step": saved["step"],
        "backend": getattr(prepared[0][-1], "backend", "full"),
        "anchors": args.anchors,
        "packed_rollin": args.packed_rollin,
        "eval_compile": args.eval_compile,
        "bf16": args.bf16,
        "seconds": timings,
        "median_seconds": sorted(timings)[2],
        "scalars": scalars,
        "code": code_identity(),
    }
    save_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "code"}), flush=True)


if __name__ == "__main__":
    main()
