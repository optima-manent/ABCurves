"""Precursor coefficient and final observed-path brake objectives."""
import torch
from torch import nn
from .brake_initial_geometry import tensor_path
from .brake_geometry import tensor_path as c2_tensor_path

def initial_loss(net,x,v,coeff,duration):
    t,q,logits,_=net(x)
    s=torch.linspace(0,1,17,device=x.device)
    pred=tensor_path(v,t,q,s)
    reference=tensor_path(v,duration[:,None],coeff[:,None],s)[:,0]
    distance=((pred-reference[:,None])/32.).square().mean((2,3))
    distance=distance+.15*(t.log()-duration[:,None].log()).square()
    winner=distance.detach().argmin(1)
    # Small shared mass keeps unused shapes trainable. Frequency is fitted to
    # observed assignment, not forced uniform or inflated with noise.
    best=distance.gather(1,winner[:,None])[:,0]
    reconstruction=(.97*best+.03*distance.mean(1)).mean()
    likelihood=nn.functional.cross_entropy(logits,winner)
    return reconstruction+.08*likelihood,dict(reconstruction=reconstruction.detach(),frequency=likelihood.detach(),winner=winner)

def brake_loss(net,x,v,acceleration,truth,duration):
    t,q,logits,_=net(x);s=torch.linspace(0,1,17,device=x.device)
    pred=c2_tensor_path(v,acceleration,t,q,s)
    distance=((pred-truth[:,None])/32.).square().mean((2,3))+.15*(t.log()-duration[:,None].log()).square()
    winner=distance.detach().argmin(1)
    reconstruction=.97*distance.gather(1,winner[:,None])[:,0]+.03*distance.mean(1)
    frequency=torch.nn.functional.cross_entropy(logits,winner,reduction='none')
    return reconstruction+.08*frequency,dict(reconstruction=reconstruction.detach(),frequency=frequency.detach(),winner=winner)
