"""Physical forecast/feedback objectives; no actor receives future labels."""
from __future__ import annotations
import torch
from torch.nn import functional as F

POSITION_SCALE = 64.
VELOCITY_SCALE = 4.
ACCELERATION_SCALE = .25


def acceleration(delta):
    """4ms mean velocities, differenced over4ms; no invented boundary sample."""
    n = delta.shape[-2] // 4
    v = delta[..., :n*4, :].reshape(*delta.shape[:-2], n, 4, 2).mean(-2)
    return (v[..., 1:, :] - v[..., :-1, :]) / 4.


def masked_distance(prediction, reference, lengths, acceleration_weight=.25):
    """[B,K,H,2] vs [B,H,2], full-head masked whole-forecast distances."""
    if prediction.ndim != 4 or reference.shape != (prediction.shape[0], *prediction.shape[2:]):
        raise ValueError('Expected B,K,H,2 and B,H,2')
    h = prediction.shape[2]
    if lengths.shape != (len(prediction),) or bool(((lengths < 1) | (lengths > h)).any()):
        raise ValueError('Observed length outside forecast')
    mask = torch.arange(h, device=prediction.device)[None] < lengths[:, None]
    def mean_distance(a, b, scale, selected):
        d = F.smooth_l1_loss(a/scale, b[:, None].expand_as(a)/scale, reduction='none').mean(-1)
        return (d * selected[:, None]).sum(-1) / selected.sum(-1).clamp_min(1)[:, None]
    pos = mean_distance(prediction.cumsum(2), reference.cumsum(1), POSITION_SCALE, mask)
    vel = mean_distance(prediction, reference, VELOCITY_SCALE, mask)
    n = h//4 - 1
    if n > 0:
        amask = torch.arange(n, device=prediction.device)[None] < (lengths//4-1).clamp_min(0)[:, None]
        acc = mean_distance(acceleration(prediction), acceleration(reference), ACCELERATION_SCALE, amask)
    else:
        acc = torch.zeros_like(pos)
    total = (pos + vel + acceleration_weight*acc)/(2+acceleration_weight)
    return total, dict(position=pos, velocity=vel, acceleration=acc)


def teacher(prediction, reference, lengths, epsilon=.05, acceleration_weight=.25):
    distance, parts = masked_distance(prediction, reference, lengths, acceleration_weight)
    k = distance.shape[1]
    if k < 2 or not 0 <= epsilon < 1:
        raise ValueError('Need at least two hypotheses and valid RWTA epsilon')
    winners = distance.detach().argmin(1)
    best = distance.gather(1, winners[:, None])[:, 0]
    per_row = (1-epsilon)*best + epsilon*(distance.sum(1)-best)/(k-1)
    return per_row.mean(), dict(per_row=per_row, winners=winners, distance=distance,
                               best=best, uniform=distance.mean(1), **parts)


def energy(samples, reference):
    """Unbiased finite-ensemble energy score, K independent uniform schedules."""
    if samples.shape[0] < 2 or samples.shape[1:] != reference.shape:
        raise ValueError('K>=2 samples and one corresponding physical reference')
    k = len(samples)
    x, y = samples.reshape(k, -1), reference.reshape(1, -1)
    first = torch.linalg.vector_norm(x-y, dim=-1).mean()
    # compute_mode avoids cdist's large-matrix algebra cancellation near zeros.
    pair = torch.cdist(x, x, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    return first - pair.sum()/(2*k*(k-1))


def temporal_score(prediction, reference):
    n = prediction.shape[1]//8
    if n < 3:
        return prediction.sum()*0.
    g = prediction[:, :n*8].reshape(len(prediction), n, 8, 2).mean(2)/VELOCITY_SCALE
    h = reference[:n*8].reshape(n, 8, 2).mean(1)/VELOCITY_SCALE
    gs, hs = torch.linalg.vector_norm(g, dim=-1), torch.linalg.vector_norm(h, dim=-1)
    terms=[]
    k=len(g)
    for lag_ms in (16,32,64,128,256,512):
        lag=lag_ms//8
        if lag >= n:
            continue
        transforms=((torch.linalg.vector_norm(g[:,lag:]-g[:,:-lag],dim=-1),
                     torch.linalg.vector_norm(h[lag:]-h[:-lag],dim=-1)),
                    ((gs[:,lag:]-gs[:,:-lag]).abs(),(hs[lag:]-hs[:-lag]).abs()))
        for a,b in transforms:
            mean=a.mean(0)
            variance_of_mean=(a-mean).square().sum(0)/(k*(k-1))
            terms.append(((mean-b).square()-variance_of_mean).mean())
    return torch.stack(terms).mean()


def feedback(prediction, reference, initial_position, targets, *, control_weight=1., acceleration_weight=.25, control_valid=None):
    """One physical source, K,H,2; only genuine H enters, actor independent."""
    h=prediction.shape[1]
    if reference.shape != (h,2) or targets.shape != (h,2):
        raise ValueError('Feedback uses exactly the observed source horizon')
    p=prediction.cumsum(1)
    y=reference.cumsum(0)
    # Complete temporal support at fixed physical normalization; no quiet RMS divisor.
    position=energy(p/(POSITION_SCALE*h**.5),y/(POSITION_SCALE*h**.5))
    velocity=energy(prediction/(VELOCITY_SCALE*h**.5),reference/(VELOCITY_SCALE*h**.5))
    if h >= 8:
        a,b=acceleration(prediction),acceleration(reference)
        acc=energy(a/(ACCELERATION_SCALE*len(b)**.5),b/(ACCELERATION_SCALE*len(b)**.5))
    else:
        acc=prediction.sum()*0.
    imitation=(position+velocity+acceleration_weight*acc)/(2+acceleration_weight)
    temporal=temporal_score(prediction,reference)
    valid=torch.ones(h,dtype=torch.bool,device=prediction.device) if control_valid is None else control_valid
    if valid.shape!=(h,) or valid.dtype!=torch.bool:raise ValueError('Expected physical target-known mask[H]')
    error=torch.where(valid[None,:,None],initial_position[None,None]+p-targets[None],0.)
    squared=(error/POSITION_SCALE).square().sum(-1)
    control=(torch.sqrt(1+squared)-1).sum()/(len(prediction)*valid.sum().clamp_min(1))
    total=imitation+temporal+control_weight*control
    return total,dict(imitation=imitation,position=position,velocity=velocity,
                      acceleration=acc,temporal=temporal,control=control)
