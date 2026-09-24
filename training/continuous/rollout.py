"""Differentiable causal commitments for precursor motor training."""
import numpy as np
import torch
from abcurves import continuous_model as policy

def tensor(value,device):
    return torch.as_tensor(np.asarray(value),device=device)

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [cpu_tree(v) for v in value]
    return value

def rollout(net,pack,seed,draws=4,trace=None):
    """All parameter dependence through generated state remains in the graph."""
    device=next(net.parameters()).device
    h=len(pack['human_delta'])
    if np.asarray(pack['human_delta']).shape != (h,2) or not 1 <= h <= 1024:
        raise ValueError('Expected one unbatched observed sequence')
    trace={} if trace is None else trace
    inputs={}
    for key in ('history','position','target','available','valid','motion_known','cut_us'):
        value=tensor(pack[key],device)
        if value.dtype.is_floating_point:value=value.double()
        inputs[key]=value.unsqueeze(0).repeat_interleave(draws,0)
    history,position,known=inputs['history'],inputs['position'],inputs['motion_known']
    rng=torch.Generator(device=device).manual_seed(seed)
    parts=[];selected=None;memory=None;head_log=[];forecast_log=[]
    trace.update(heads=head_log,forecasts=forecast_log,committed_horizon=0,initial_rng=rng.get_state())
    for ordinal,at in enumerate(range(0,h,net.config.commit)):
        trace['attempted_cut']=at
        c,f=policy.features(history,position,inputs['target'][:,at:at+640],
            inputs['available'][:,at:at+640],inputs['valid'][:,at:at+640],known,
            inputs['cut_us']+at*1000,net.config)
        forecast,memory=net(c,f,history[:,-1],memory)
        if not bool(torch.isfinite(forecast).all()) or bool((forecast.abs()>1e6).any()):
            trace['fault_forecast']=forecast.detach().cpu()
            trace['fault_inputs']=cpu_tree(dict(coarse=c,fine=f,incoming=history[:,-1],memory=memory))
            raise FloatingPointError('Nonfinite/runaway full forecast before commitment')
        if ordinal % net.config.head_hold == 0:
            selected=torch.randint(16,(draws,),device=device,generator=rng)
        full=forecast[torch.arange(draws,device=device),selected]
        take=min(net.config.commit,h-at)
        delta=full[:,:take]
        parts.append(delta)
        # Retain this small current source for faults / first-update evidence only.
        head_log.append(selected.detach().cpu())
        forecast_log.append(full.detach().cpu())
        trace['committed_horizon']=at+take
        if at+take<h:
            history=torch.cat((history[:,take:],delta),1)
            position=position+delta.sum(1)
            known=torch.cat((known[:,take:],torch.ones_like(known[:,:take])),1)
    trace.update(heads=torch.stack(head_log),forecasts=torch.stack(forecast_log),final_rng=rng.get_state())
    return torch.cat(parts,1),trace
