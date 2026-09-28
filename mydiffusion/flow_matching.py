"""E7 x0 Flow Matching. Optional history projection follows the noisy flow path.

t=1 is noise; t=0 is data. Known coordinates follow (1-t)*value+t*noise.
This inference-only constraint changes no learned parameters or training target.
"""

from collections.abc import Mapping
import torch


def _append_dims(value, ndim):
    if value.ndim > ndim:
        raise ValueError("Cannot broadcast to fewer dimensions.")
    return value.reshape(*value.shape, *((1,) * (ndim - value.ndim)))


class FlowMatching:
    def __init__(
        self,
        num_steps=10,
        solver="euler",
        beta_alpha=1.5,
        beta_beta=1.0,
        t_min=0.001,
        prediction_type="x0",
        global_weight=8.0,
        global_feature_start=198,
        global_feature_end=207,
        repaint_enabled=False,
    ):
        self._check_steps(num_steps)
        if solver != "euler" or prediction_type != "x0":
            raise ValueError("E7 requires x0 prediction with Euler sampling.")
        if beta_alpha <= 0 or beta_beta <= 0 or not 0 <= t_min < 1:
            raise ValueError("Invalid Beta schedule.")
        if global_weight <= 0 or not 0 <= global_feature_start < global_feature_end:
            raise ValueError("Invalid global weight or feature slice.")
        if not isinstance(repaint_enabled, bool):
            raise TypeError("repaint_enabled must be a bool.")
        self.num_steps, self.solver = num_steps, solver
        self.prediction_type = prediction_type
        self.beta_alpha, self.beta_beta = float(beta_alpha), float(beta_beta)
        self.t_min, self.global_weight = float(t_min), float(global_weight)
        self.global_feature_start, self.global_feature_end = global_feature_start, global_feature_end
        self.repaint_enabled = repaint_enabled

    @staticmethod
    def _check_steps(steps):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("num_steps must be a positive integer.")

    @staticmethod
    def _normalize_t(t, batch_size, device):
        t = torch.as_tensor(t, device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,) or not bool(torch.all((t > 0) & (t <= 1))):
            raise ValueError("E7 requires one finite time in (0, 1] per sample.")
        return t

    def sample_timesteps(self, batch_size, device):
        alpha = torch.tensor(self.beta_alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(self.beta_beta, device=device, dtype=torch.float32)
        samples = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return self.t_min + (1.0 - self.t_min) * samples

    @staticmethod
    def interpolate(x_start, noise, t):
        if x_start.shape != noise.shape:
            raise ValueError("noise must have the same shape as x_start.")
        t_view = _append_dims(t.to(device=x_start.device, dtype=x_start.dtype), x_start.ndim)
        return (1.0 - t_view) * x_start + t_view * noise

    @staticmethod
    def target_to_velocity(x_t, t, pred_xstart):
        if x_t.shape != pred_xstart.shape or not bool(torch.all(t > 0)):
            raise ValueError("Conversion requires matching shapes and t > 0.")
        return (x_t - pred_xstart) / _append_dims(t.to(x_t), x_t.ndim)

    @staticmethod
    def _split_constraints(model_kwargs):
        kwargs = dict(model_kwargs or {})
        y = kwargs.get("y", {})
        if not isinstance(y, Mapping):
            raise TypeError("model_kwargs['y'] must be a mapping.")
        y = dict(y)
        mask, value = y.pop("repaint_mask", None), y.pop("repaint_value", None)
        kwargs["y"] = y
        return kwargs, mask, value

    def _constraints(self, mask, value, x, override):
        enabled = self.repaint_enabled if override is None else override
        if not isinstance(enabled, bool):
            raise TypeError("repaint_enabled must be a bool or None.")
        if not enabled or (mask is None and value is None):
            return None, None
        if mask is None or value is None:
            raise ValueError("Enabled repaint requires both repaint_mask and repaint_value.")
        mask = torch.as_tensor(mask, device=x.device)
        value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        if mask.shape != x.shape or value.shape != x.shape:
            raise ValueError(f"Repaint mask/value must have exact motion shape {tuple(x.shape)}.")
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("repaint_mask must be binary (0/1 or bool).")
        mask = mask.bool()
        if not bool(torch.isfinite(value[mask]).all()):
            raise ValueError("Masked repaint_value coordinates must be finite.")
        if not bool(mask.any()):
            return None, None
        return mask, torch.where(mask, value, torch.zeros_like(value))

    def _valid_frame_mse(self, error, model_kwargs):
        y = model_kwargs["y"]
        valid_frames = y.get("loss_mask", y["valid_frames"])
        mask = torch.as_tensor(valid_frames, device=error.device)
        if mask.shape != error.shape[:-1]:
            raise ValueError("valid_frames/loss_mask must match motion batch/time dimensions.")
        mask = _append_dims(mask, error.ndim).expand_as(error).float()
        feature_dim = error.shape[-1]
        if self.global_feature_end > feature_dim:
            raise ValueError("Global feature slice exceeds motion dimensions.")
        weights = torch.ones(feature_dim, device=error.device, dtype=torch.float32)
        weights[self.global_feature_start : self.global_feature_end] = self.global_weight
        weights = weights.reshape(*((1,) * (error.ndim - 1)), feature_dim)
        loss_weights = mask * weights
        dims = tuple(range(1, error.ndim))
        numerator = (error.float().square() * loss_weights).sum(dim=dims)
        denominator = loss_weights.sum(dim=dims)
        loss = torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), torch.zeros_like(numerator))
        squared_error = error.float().square() * mask
        global_error = squared_error[..., self.global_feature_start : self.global_feature_end]
        global_mask = mask[..., self.global_feature_start : self.global_feature_end]
        global_sum, global_count = global_error.sum(dim=dims), global_mask.sum(dim=dims)
        local_sum, local_count = squared_error.sum(dim=dims) - global_sum, mask.sum(dim=dims) - global_count
        local = torch.where(local_count > 0, local_sum / local_count.clamp_min(1.0), torch.zeros_like(local_sum))
        global_mse = torch.where(
            global_count > 0, global_sum / global_count.clamp_min(1.0), torch.zeros_like(global_sum)
        )
        return loss, local, global_mse

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, t=None, return_diagnostics=False):
        kwargs, mask, value = self._split_constraints(model_kwargs)
        if self.repaint_enabled and (mask is not None or value is not None):
            raise ValueError("Repaint is inference-only; do not constrain E7 training targets.")
        noise = torch.randn_like(x_start) if noise is None else noise.to(x_start)
        t = (
            self.sample_timesteps(x_start.shape[0], x_start.device)
            if t is None
            else self._normalize_t(t, x_start.shape[0], x_start.device)
        )
        x_t = self.interpolate(x_start, noise, t)
        pred = model(x_t, t, **kwargs)
        if pred.shape != x_start.shape:
            raise ValueError("Model output must match motion shape.")
        loss, local, global_mse = self._valid_frame_mse(pred - x_start, kwargs)
        result = dict(loss=loss, mse=loss, t=t, local_mse=local, global_mse=global_mse)
        if return_diagnostics:
            result.update(
                x_t=x_t,
                model_output=pred,
                pred_xstart=pred,
                pred_velocity=self.target_to_velocity(x_t, t, pred),
                target_velocity=noise - x_start,
            )
        return result

    @torch.no_grad()
    def sample_loop_progressive(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        num_steps=None,
        solver=None,
        device=None,
        progress=False,
        repaint_enabled=None,
    ):
        """Known values must be normalized motion in the current window's coordinates.

        No constraints or an all-zero mask exactly preserves original E7 Euler.
        The caller owns physical coordinate transforms and history construction.
        """
        kwargs, mask, value = self._split_constraints(model_kwargs)
        steps = self.num_steps if num_steps is None else num_steps
        self._check_steps(steps)
        if solver is not None and solver != "euler":
            raise ValueError("E7 supports Euler only.")
        shape = tuple(shape)
        if len(shape) != 3 or any(n <= 0 for n in shape):
            raise ValueError("shape must be positive (batch, frames, features).")
        if device is None:
            if noise is not None:
                device = noise.device
            else:
                try:
                    device = next(model.parameters()).device
                except (AttributeError, StopIteration):
                    device = torch.device("cpu")
        if noise is None:
            x = torch.randn(*shape, device=device)
        else:
            if tuple(noise.shape) != shape:
                raise ValueError("noise must have the exact sample shape.")
            x = noise.to(device=device)
        mask, known = self._constraints(mask, value, x, repaint_enabled)
        initial_noise = x.clone() if mask is not None else None
        times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        indices = range(steps)
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for index in indices:
            t_scalar, next_t_scalar = times[index], times[index + 1]
            t = t_scalar.expand(shape[0])
            if mask is not None:
                x = torch.where(mask, self.interpolate(known, initial_noise, t), x)
            pred = model(x, t, **kwargs)
            if pred.shape != x.shape:
                raise ValueError("Model output must match motion shape.")
            if mask is not None:
                pred = torch.where(mask, known, pred)
            velocity = self.target_to_velocity(x, t, pred)
            dt = (next_t_scalar - t_scalar).to(dtype=x.dtype)
            next_x = x + dt * velocity
            if mask is not None:
                next_path = self.interpolate(known, initial_noise, next_t_scalar.expand(shape[0]))
                next_x = torch.where(mask, next_path, next_x)
            yield dict(sample=next_x, pred_xstart=pred, t=t, next_t=next_t_scalar.expand(shape[0]))
            x = next_x

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        num_steps=None,
        solver=None,
        device=None,
        progress=False,
        return_all_pred_xstart=False,
        repaint_enabled=None,
    ):
        predictions, final = [], None
        for output in self.sample_loop_progressive(
            model, shape, model_kwargs, noise, num_steps, solver, device, progress, repaint_enabled
        ):
            final = output["sample"]
            if return_all_pred_xstart:
                predictions.append(output["pred_xstart"])
        return (final, predictions) if return_all_pred_xstart else final
