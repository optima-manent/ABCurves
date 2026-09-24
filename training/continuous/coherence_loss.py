"""Movement-prior and generated-history objectives for the selected motor."""
import numpy as np
import torch
from torch.nn import functional as F
from abcurves import continuous_model as base
from . import losses as old
from . import coherence_data as dataset

def tensor(x, device):
    return torch.as_tensor(np.asarray(x), device=device)

def rwta(distance):
    winner = distance.detach().argmin(1)
    best = distance.gather(1, winner[:, None])[:, 0]
    return ((0.95 * best + 0.05 * (distance.sum(1) - best) / 15).mean(), winner)

def direction_turn(pred, reference, lengths):
    p = pred[:, :, :32].sum(2)
    r = reference[:, :32].sum(1)
    valid = (lengths >= 32) & (r.norm(dim=-1) > 4.0)
    direction = (1 - F.cosine_similarity(p, r[:, None], dim=-1, eps=1e-06)) * valid[:, None]

    def turns(x):
        n = x.shape[-2] // 8
        v = x[..., :n * 8, :].reshape(*x.shape[:-2], n, 8, 2).mean(-2)
        a, b = (v[..., :-1, :], v[..., 1:, :])
        support = (a.norm(dim=-1) > 0.04) & (b.norm(dim=-1) > 0.04)
        return (1 - F.cosine_similarity(a, b, dim=-1, eps=1e-06)) * support
    pt, rt = (turns(pred), turns(reference))
    mask = torch.arange(pt.shape[-1], device=pred.device)[None] < (lengths // 8 - 1).clamp_min(0)[:, None]
    excess = ((pt - rt[:, None] - 0.05).clamp_min(0) * mask[:, None]).sum(-1) / mask.sum(-1).clamp_min(1)[:, None]
    return 0.05 * direction + 0.03 * excess

def teacher(net, batch):
    device = next(net.parameters()).device
    c = tensor(batch['coarse'], device).float()
    f = tensor(batch['fine'], device).float()
    incoming = tensor(batch['incoming'], device).double()
    reference = tensor(batch['delta'], device).double()
    lengths = tensor(batch['length'], device).long()
    pred, _ = net(c, f, incoming)
    reference = reference[:, :128]
    lengths = lengths.clamp_max(128)
    distance, _ = old.masked_distance(pred, reference, lengths)
    distance += direction_turn(pred, reference, lengths)
    extra = pred.sum() * 0.0
    loss, winner = rwta(distance)
    return (loss + extra, dict(teacher=float(loss.detach()), winner_counts=torch.bincount(winner, minlength=16).tolist()))

def synthetic_quiet(net, step, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 113, step]))
    pack = dataset.quiet_pack(16, rng)
    c, f = dataset.old.physical_features(pack)
    c, f = dataset.old.materialize_features(c, f)
    device = next(net.parameters()).device
    c = tensor(c, device).float()
    f = tensor(f, device).float()
    incoming = torch.zeros((16, 2), device=device, dtype=torch.float64)
    pred, _ = net(c, f, incoming)
    categorical = pred.sum() * 0.0
    return 0.1 * (pred / 0.25).square().mean() + 0.03 * categorical

def movement_features(delta):
    return torch.cat((delta.cumsum(-2)[..., 7:32:8, :].flatten(-2) / 64 / 8 ** 0.5, delta[..., :32, :].reshape(*delta.shape[:-2], 4, 8, 2).mean(-2).flatten(-2) / 4 / 8 ** 0.5, old.acceleration(delta[..., :32, :]).flatten(-2) / 0.25 / 14 ** 0.5 * 0.5), -1).float()

def conditional_style(net, prior, c, f, incoming, info):
    with torch.no_grad():
        human = prior(c.detach(), f.detach(), incoming.detach())[0][:, :, :32]
    options = info['forecasts'][:, :, :32]
    weights = torch.full(options.shape[:2], 1 / 16, device=c.device)
    x, y = (movement_features(options), movement_features(human))
    cross = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    xx = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    yy = torch.cdist(y, y, compute_mode='donot_use_mm_for_euclid_dist')
    return 2 * (cross.mean(-1) * weights).sum(-1) - (xx * weights[:, :, None] * weights[:, None, :]).sum((1, 2)) - yy.mean((1, 2))

def rollout(net, record, seed, prior=None, draws=4, trace=False):
    device = next(net.parameters()).device
    pack = record['pack']
    h = record['H']
    inputs = {k: tensor(pack[k], device).unsqueeze(0).repeat_interleave(draws, 0) for k in ('history', 'position', 'target', 'available', 'valid', 'motion_known', 'cut_us')}
    history = inputs['history'].double()
    position = inputs['position'].double()
    known = inputs['motion_known']
    rng = torch.Generator(device=device).manual_seed(seed)
    state = None
    pieces = []
    styles = []
    logps = []
    decisions = []
    for at in range(0, h, 32):
        c, f = base.features(history, position, inputs['target'][:, at:at + 640], inputs['available'][:, at:at + 640], inputs['valid'][:, at:at + 640], known, inputs['cut_us'] + at * 1000, net.config)
        action, state, info = net.act(c, f, history[:, -1], position, state, rng)
        if not torch.isfinite(action).all() or (action.abs() > 100000.0).any():
            raise FloatingPointError('Nonfinite/runaway action')
        if prior is not None:
            styles.append(conditional_style(net, prior, c, f, history[:, -1], info))
        if info['log_prob'] is not None:
            logps.append(info['log_prob'])
        if trace:
            decisions.append({k: v.detach().cpu().tolist() for k, v in info.items() if k in ('head', 'revised', 'stopped', 'duration')})
        take = min(32, h - at)
        delta = action[:, :take]
        pieces.append(delta)
        history = torch.cat((history[:, take:], delta), 1)
        position = position + delta.sum(1)
        known = torch.cat((known[:, take:], torch.ones_like(known[:, :take])), 1)
    prediction = torch.cat(pieces, 1)
    return (prediction, dict(style=torch.stack(styles, 1) if styles else None, log_prob=torch.stack(logps, 1) if logps else None, decisions=decisions, rng=rng.get_state()))

def feedback(net, prior, record, seed):
    pred, info = rollout(net, record, seed, prior)
    device = pred.device
    h = record['H']
    pack = record['pack']
    target = tensor(record['control_target'], device).double()
    valid = tensor(record['control_valid'], device).bool()
    error = (tensor(pack['position'], device).double()[None, None] + pred.cumsum(1) - target[None]).norm(dim=-1)
    excess = (error - 10).clamp_min(0) / 64
    task = torch.sqrt(1 + excess.square()) - 1
    if record['domain'] in ('early', 'late'):
        eligible = int(record['ref']['observed_remaining']) <= h
        endpoint = int(record['ref'].get('authentic_horizon', h)) - 1
        control = (task[:, endpoint:] * valid[endpoint:]).mean(1) * eligible
    else:
        control = (task * valid).sum(1) / valid.sum().clamp_min(1)
    style = info['style'].mean(1)
    incoming = tensor(pack['history'][-1], device).double()
    joined = torch.cat((incoming[None, None].expand(len(pred), 1, 2), pred), 1)
    change = joined[:, 1:] - joined[:, :-1]
    effort = (pred / 0.25).square().mean((1, 2))
    join = (change / 0.25).square().mean((1, 2))
    costs = 4 * control + 0.5 * style
    if record['domain'] == 'functional':
        costs = costs + 0.003 * effort + 0.002 * join
    loss = costs.mean()
    if info['log_prob'] is not None:
        credit = costs.detach()
        baseline = (credit.sum() - credit) / (len(credit) - 1)
        loss = loss + ((credit - baseline) * info['log_prob'].sum(1)).mean()
    return (loss, dict(domain=record['domain'], task=float(control.detach().mean()), style=float(style.detach().mean()), effort=float(effort.detach().mean()), join=float(join.detach().mean()), cost=float(costs.detach().mean())))
