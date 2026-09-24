"""Selected terminal-tail continuation objective; observed and synthetic support stay distinct."""
import numpy as np
import torch
from torch.nn import functional as F
from abcurves import continuous_model as policy
from . import losses
from . import terminal_loss as objective_losses

def rollout(net, pack, seed, draws=4):
    device = next(net.parameters()).device
    h = len(pack['human_delta'])
    inputs = {}
    for key in ('history', 'position', 'target', 'available', 'valid', 'motion_known', 'cut_us'):
        value = torch.as_tensor(np.asarray(pack[key]), device=device)
        if value.dtype.is_floating_point:
            value = value.double()
        inputs[key] = value.unsqueeze(0).repeat_interleave(draws, 0)
    history, position, known = (inputs['history'], inputs['position'], inputs['motion_known'])
    rng = torch.Generator(device=device).manual_seed(seed)
    pieces, probabilities, values, quiet = ([], [], [], [])
    for at in range(0, h, net.config.commit):
        c, f = policy.features(history, position, inputs['target'][:, at:at + 640], inputs['available'][:, at:at + 640], inputs['valid'][:, at:at + 640], known, inputs['cut_us'] + at * 1000, net.config)
        forecast, _ = net(c, f, history[:, -1])
        if not torch.isfinite(forecast).all() or (forecast.abs() > 1000000.0).any():
            raise FloatingPointError('Nonfinite/runaway complete forecast')
        selected, sample, log_prob = net.draw(forecast, rng)
        take = min(net.config.commit, h - at)
        delta = selected[:, :take]
        pieces.append(delta)
        if at + take < h:
            history = torch.cat((history[:, take:], delta), 1)
            position = position + delta.sum(1)
            known = torch.cat((known[:, take:], torch.ones_like(known[:, :take])), 1)
    extra = dict(log_prob=torch.stack(probabilities, 1), value=torch.stack(values, 1), quiet=torch.stack(quiet, 1)) if net.gated else {}
    return (torch.cat(pieces, 1), extra)

def quiet_supervision(prediction, reference, lengths):
    complete = lengths >= 32
    whole = complete & (reference[:, :32] == 0).all(dim=(1, 2))
    tail = complete & ~whole & (reference[:, 24:32] == 0).all(dim=(1, 2))
    a = (prediction[:, :, :32] / 0.25).square().mean(dim=(1, 2, 3))
    b = (prediction[:, :, 24:32] / 0.25).square().mean(dim=(1, 2, 3))
    return ((a * whole + b * tail).mean(), whole, complete)

def surrogate(objective, trace, *, individual=False):
    if not trace:
        return (objective.mean() if individual else objective, {})
    target = objective.detach()[:, None] if individual else objective.detach().expand_as(trace['value'])
    advantage = target - trace['value'].detach()
    terms = advantage * trace['log_prob']
    actor = terms.sum(1).mean() if individual else terms.sum()
    critic = F.smooth_l1_loss(trace['value'], target.expand_as(trace['value']))
    value = objective.mean() if individual else objective
    return (value + actor + 0.1 * critic, dict(actor=float(actor.detach()), critic=float(critic.detach()), sampled_quiet=float(trace['quiet'].float().mean()), mean_value=float(trace['value'].detach().mean())))

def static_tail(entry, duration=512):
    """Explicit synthetic extension; never treats a cropped window as resolution."""
    pack = entry['pack']
    h = len(pack['human_delta'])
    if entry['domain'] == 'tracking' or entry['ref'].get('observed_remaining') != h:
        return (pack, 0)
    if not bool(entry['control_valid'][-1]):
        return (pack, 0)
    extended = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}
    for key in ('target', 'available', 'valid'):
        extended[key] = np.concatenate((extended[key], np.repeat(extended[key][-1:], duration, axis=0)))
    extended['human_delta'] = np.concatenate((extended['human_delta'], np.zeros((duration, 2))))
    return (extended, duration)

def make_optimizer(net):
    groups = []
    for label, prefix, lr in [('movement', None, 2e-05), ('gate', 'quiet_gate.', 0.0002), ('critic', 'critic.', 0.0005)]:
        selected = [p for name, p in net.named_parameters() if p.requires_grad and (not name.startswith(('quiet_gate.', 'critic.')) if prefix is None else name.startswith(prefix))]
        if selected:
            groups.append(dict(params=selected, lr=lr, name=label))
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

def update(net, opt, dataset, step, seed):
    opt.zero_grad(set_to_none=True)
    batch = dataset.teacher(23000 + step, seed, 'base', rotation=True, augmentation=True)
    tensors = {k: torch.as_tensor(batch[k], device=next(net.parameters()).device) for k in ('coarse', 'fine', 'incoming', 'delta', 'length')}
    c, f = (tensors['coarse'].float(), tensors['fine'].float())
    reference = tensors['delta'][:, :128].double()
    lengths = tensors['length'].clamp_max(128).long()
    prediction, _ = net(c, f, tensors['incoming'].double())
    teacher, _ = losses.teacher(prediction, reference, lengths, 0.05, 0.25)
    quiet_loss, quiet_label, complete = quiet_supervision(prediction, reference, lengths)
    total = teacher
    record = dict(step=step, teacher=float(teacher.detach()), quiet_teacher=float(quiet_loss.detach()))
    if total.requires_grad:
        total.backward()
    del prediction, total, teacher, quiet_loss, tensors, c, f, reference, lengths
    entries = dataset.sequences(3000 + step, seed, 'base', rotation=True, augmentation=True)
    sequence_logs = []
    for ordinal, entry in enumerate(entries):
        pack = entry['pack']
        training_pack, added = static_tail(entry) if net.arm == 'S1' else (pack, 0)
        pred, trace = rollout(net, training_pack, seed + step * 1009 + ordinal * 53)
        authentic_h = len(pack['human_delta'])
        actual = torch.as_tensor(pack['human_delta'], device=next(net.parameters()).device).double()
        initial = torch.as_tensor(pack['position'], device=next(net.parameters()).device).double()
        target = torch.as_tensor(entry['control_target'], device=next(net.parameters()).device).double()
        valid = torch.as_tensor(entry['control_valid'], device=next(net.parameters()).device).bool()
        objective, _ = objective_losses.feedback(pred[:, :authentic_h], actual, initial, target, domain=entry['domain'], mode='static_terminal', observed_remaining=entry['ref'].get('observed_remaining'), control_weight=4.0, acceleration_weight=0.25, control_valid=valid)
        tail_value = None
        if added:
            tail = pred[:, authentic_h:]
            tail_initial = initial[None] + pred[:, :authentic_h].sum(1)
            zero = torch.zeros_like(tail[0])
            tail_imitation, _ = losses.feedback(tail, zero, torch.zeros_like(initial), torch.zeros_like(zero), control_weight=0.0, acceleration_weight=0.25)
            errors = (tail_initial[:, None] + tail.cumsum(1) - target[-1]) / 64.0
            control = (torch.sqrt(1 + errors.square().sum(-1)) - 1).mean()
            tail_value = tail_imitation + 4 * control
            objective = objective + tail_value
        loss, diagnostics = surrogate(objective, trace)
        if loss.requires_grad:
            (loss / len(entries)).backward()
        sequence_logs.append(dict(domain=entry['domain'], loss=float(objective.detach()), synthetic_tail_ms=added, tail_loss=None if tail_value is None else float(tail_value.detach()), **diagnostics))
        del pred, trace, loss, objective
    record['sequences'] = sequence_logs
    norms = {}
    for group in opt.param_groups:
        parameters = group['params']
        if any((p.grad is None for p in parameters)):
            raise FloatingPointError('Missing gradient: ' + group['name'])
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0 if group['name'] == 'gate' else 5.0, error_if_nonfinite=True)
        norms[group['name']] = float(norm)
    opt.step()
    if not all((torch.isfinite(p).all() for p in net.parameters())):
        raise FloatingPointError('Nonfinite updated parameter')
    record['gradient_norms'] = norms
    torch.cuda.synchronize() if next(net.parameters()).is_cuda else None
    return record
