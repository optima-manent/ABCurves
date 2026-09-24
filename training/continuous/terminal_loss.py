"""Domain-sensitive training rewards with no new runtime observations."""
import torch
from . import losses as original

MODES = {"all_time", "static_terminal", "static_imitation"}


def feedback(prediction, reference, initial_position, targets, *, domain, mode,
             observed_remaining=None, control_weight=1., acceleration_weight=.25, control_valid=None):
    if mode not in MODES or domain not in {"tracking", "early", "late"}:
        raise ValueError("Unknown declared domain/objective")
    kwargs = dict(control_weight=control_weight, acceleration_weight=acceleration_weight,
                  control_valid=control_valid)
    if domain == "tracking" or mode == "all_time":
        return original.feedback(prediction, reference, initial_position, targets, **kwargs)
    h = prediction.shape[1]
    if not isinstance(observed_remaining, int) or observed_remaining < h:
        raise ValueError("Static source must retain its genuine remaining length")
    base, parts = original.feedback(prediction, reference, initial_position, targets,
                                   **dict(kwargs, control_weight=0.))
    trajectory_control = parts["control"]
    terminal = prediction.sum() * 0.
    ended = observed_remaining == h
    if mode == "static_terminal" and ended:
        valid = torch.ones((), dtype=torch.bool, device=prediction.device) if control_valid is None else control_valid[-1]
        error = torch.where(valid, initial_position[None] + prediction.sum(1) - targets[-1], 0.)
        terminal = (torch.sqrt(1 + (error / original.POSITION_SCALE).square().sum(-1)) - 1).mean()
    # Keep components scalar and source-local for the unchanged training logger.
    parts.update(control=terminal, control_trajectory=trajectory_control,
                 terminal_observed=prediction.new_tensor(float(ended)))
    return base + control_weight * terminal, parts
