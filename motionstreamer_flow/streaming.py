import torch

from .geometry import fk, root_to_world, to_world, validate_rigid_transform
from .objective import temporarily_eval

ACTION_BITS = {"a11": (True, True), "a10": (True, False), "a01": (False, True), "a00": (False, False)}


class StreamingSession:
    """One-frame causal generation. GT is deliberately absent from this API."""

    @torch.no_grad()
    def __init__(
        self,
        flow,
        tae,
        normalizer,
        *,
        startup_pose,
        startup_root,
        images,
        head,
        image_valid,
        head_valid,
        beta,
        offsets,
        anchor,
        original_world_from_anchor,
    ):
        self.flow, self.tae, self.norm = flow, tae, normalizer
        self.cfg = flow.cfg
        device = next(flow.parameters()).device
        self.beta, self.offsets, self.anchor = (x.to(device).float() for x in (beta, offsets, anchor))
        validate_rigid_transform(self.anchor, "anchor")
        validate_rigid_transform(original_world_from_anchor, "original_world_from_anchor")
        self.original_world_from_anchor = original_world_from_anchor.to(device).float()
        n = len(startup_pose)
        if (
            n != self.cfg.bootstrap
            or len(startup_root) != n
            or any(len(x) != n for x in (images, head, image_valid, head_valid))
        ):
            raise ValueError("Initialize with exactly the predicted startup prefix and corresponding observations.")
        with temporarily_eval(tae):
            latent, _ = tae.encode(self.norm.normalize("pose", startup_pose.to(device))[None], self.beta[None])
        self.states = torch.cat(
            (self.norm.normalize("latent", latent[0]), self.norm.normalize("root", startup_root.to(device))), -1
        )
        self.images, self.head = images.to(device).clone(), head.to(device).clone()
        self.image_valid, self.head_valid = image_valid.to(device).clone(), head_valid.to(device).clone()
        self.frames = n
        if not torch.isfinite(self.states).all():
            raise ValueError("Non-finite startup state.")

    @torch.no_grad()
    def step(self, image, head, image_valid=True, head_valid=True, *, action="a11", noise=None):
        if action not in ACTION_BITS:
            raise ValueError("Unknown observation action.")
        cfg, device = self.cfg, self.states.device
        # Keep enough latent prefix for the decoder as well as the history Transformer.
        keep = max(cfg.history, cfg.receptive_field - 1)
        past = self.states[-keep:]
        states = torch.cat((past, past.new_zeros(1, cfg.state_dim)), 0)[None]
        images = torch.cat((self.images[-keep:], image.to(device)[None]), 0)
        heads = torch.cat((self.head[-keep:], head.to(device)[None]), 0)
        iv = torch.cat((self.image_valid[-keep:], torch.tensor([image_valid], device=device, dtype=torch.bool)))
        hv = torch.cat((self.head_valid[-keep:], torch.tensor([head_valid], device=device, dtype=torch.bool)))
        actions = torch.tensor(ACTION_BITS[action], device=device).expand(1, len(images), 2)
        if (image_valid and not torch.isfinite(image).all()) or (head_valid and not torch.isfinite(head).all()):
            raise ValueError("Non-finite available observation.")
        amp = device.type == "cuda" and cfg.precision != "fp32"
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
        with temporarily_eval(self.flow), temporarily_eval(self.tae), torch.autocast(
            device.type, dtype=dtype, enabled=amp
        ):
            condition = self.flow.conditioner(
                states,
                images[None],
                self.norm.normalize("head", heads)[None],
                iv[None],
                hv[None],
                torch.ones(states.shape[:2], device=device, dtype=torch.bool),
                self.beta[None],
                actions,
                last_only=True,
            )[:, -1]
            epsilon = (
                torch.randn(1, cfg.state_dim, device=device)
                if noise is None
                else noise.to(device).reshape(1, cfg.state_dim)
            )
            state = self.norm.project_state(self.flow.sample(condition, epsilon), cfg.latent_dim)[0]
            context = torch.cat((past[-(cfg.receptive_field - 1) :, : cfg.latent_dim], state[None, : cfg.latent_dim]))
            pose = self.norm.restore(
                "pose", self.tae.decode(self.norm.restore("latent", context)[None], self.beta[None])[0, -1]
            )
            root = self.norm.restore("root", state[cfg.latent_dim :])
            joints, rotations = fk(pose, root, self.offsets)
            if not all(torch.isfinite(x).all() for x in (state, pose, root, joints, rotations)):
                raise FloatingPointError("Invalid generation; stream state was not committed.")
        pose = torch.cat((pose[:150], pose[150:].clamp(0, 1)))
        # Commit only after all predictions pass validation. Never re-encode generated bodies.
        self.states = torch.cat((past, state[None]))[-keep:]
        self.images, self.head = images[-keep:].clone(), heads[-keep:].clone()
        self.image_valid, self.head_valid = iv[-keep:].clone(), hv[-keep:].clone()
        self.frames += 1
        return {
            "state": state.clone(),
            "pose": pose,
            "root": root,
            "joints": joints,
            "rotations": rotations,
            "root_floor_world": root_to_world(root, self.anchor),
            "joints_floor_world": to_world(joints, self.anchor),
            "root_world": root_to_world(root, self.original_world_from_anchor),
            "joints_world": to_world(joints, self.original_world_from_anchor),
        }


@torch.no_grad()
def rollout(flow, tae, normalizer, inputs, *, action="a11", seed=62):
    cfg = flow.cfg
    n = cfg.bootstrap
    session = StreamingSession(
        flow,
        tae,
        normalizer,
        **{
            key: value[:n] if key in ("images", "head", "image_valid", "head_valid") else value
            for key, value in inputs.items()
        },
    )
    generator = torch.Generator(device=session.states.device).manual_seed(seed)
    outputs = []
    for j in range(n, len(inputs["images"])):
        noise = torch.randn(cfg.state_dim, device=session.states.device, generator=generator)
        outputs.append(
            session.step(
                inputs["images"][j],
                inputs["head"][j],
                bool(inputs["image_valid"][j]),
                bool(inputs["head_valid"][j]),
                action=action,
                noise=noise,
            )
        )
    return {key: torch.stack([result[key].cpu() for result in outputs]) for key in outputs[0]}
