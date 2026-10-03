"""Current-frame source construction around the unchanged E7 Flow solver."""

import math

import torch

from egorecover.conditioning import finite_payload, floating_tensor, prepare_conditioning
from mydiffusion.flow_matching import FlowMatching


class HistoryFlow:
    def __init__(self, *, source_mode="history", sigma=1.0, flow=None):
        if source_mode not in ("history", "gaussian"):
            raise ValueError("source_mode must be 'history' or 'gaussian'.")
        if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive.")
        self.source_mode, self.sigma = source_mode, float(sigma)
        self.flow = FlowMatching() if flow is None else flow
        if self.flow.repaint_enabled:
            raise ValueError("History is external to Flow; repaint must be disabled.")

    def build_source(self, prior_mu, *, epsilon=None, generator=None):
        """epsilon is standard noise, never an already shifted/scaled source.

        Pass the same epsilon for paired source/action comparisons. Both
        source modes still pass the same prior_mu condition to the network.
        """
        if not isinstance(prior_mu, torch.Tensor) or prior_mu.ndim != 3 or prior_mu.shape[1:] != (1, 243):
            raise ValueError("prior_mu must have shape [B,1,243].")
        if prior_mu.shape[0] < 1 or not prior_mu.is_floating_point() or not bool(torch.isfinite(prior_mu).all()):
            raise ValueError("prior_mu must be nonempty, floating point and finite.")
        if epsilon is not None and generator is not None:
            raise ValueError("Pass epsilon or generator, not both.")
        if epsilon is None:
            epsilon = torch.randn(prior_mu.shape, dtype=prior_mu.dtype, device=prior_mu.device, generator=generator)
        else:
            epsilon = floating_tensor(epsilon, prior_mu.shape, prior_mu, "epsilon")
            if not bool(torch.isfinite(epsilon).all()):
                raise ValueError("epsilon must be finite.")
        source = self.sigma * epsilon
        return source + prior_mu if self.source_mode == "history" else source

    def training_losses(
        self, model, target_current, y, *, epsilon=None, generator=None, t=None, return_diagnostics=False
    ):
        y = prepare_conditioning(y, history_length=getattr(model, "history_length", None))
        target = floating_tensor(target_current, y["prior_mu"].shape, y["history_motion"], "target_current")
        # Unlabelled/padded targets may be absent. They must not introduce NaN
        # through interpolation or through multiplication by a zero loss mask.
        target = finite_payload(target, y.get("loss_mask", y["valid_frames"]), "target_current")
        source = self.build_source(y["prior_mu"], epsilon=epsilon, generator=generator)
        return self.flow.training_losses(
            model,
            target,
            model_kwargs={"y": y},
            noise=source,
            t=t,
            return_diagnostics=return_diagnostics,
        )

    @torch.no_grad()
    def sample(
        self, model, y, *, epsilon=None, generator=None, num_steps=None, progress=False, return_all_pred_xstart=False
    ):
        if model.training:
            raise ValueError("Call model.eval() before sampling so dropout cannot change paired comparisons.")
        y = prepare_conditioning(y, history_length=getattr(model, "history_length", None))
        source = self.build_source(y["prior_mu"], epsilon=epsilon, generator=generator)
        return self.flow.sample_loop(
            model,
            source.shape,
            model_kwargs={"y": y},
            noise=source,
            num_steps=num_steps,
            device=source.device,
            progress=progress,
            return_all_pred_xstart=return_all_pred_xstart,
            repaint_enabled=False,
        )
