"""Finite paths with fixed incoming velocity and measured incoming acceleration."""
import numpy as np
import torch


def bases(s):
    z=2*s-1
    envelope=64*s**3*(1-s)**3
    residual=np.stack((envelope,envelope*z,envelope*(3*z*z-1)/2,envelope*(5*z*z*z-3*z)/2),-1)
    return s-6*s**3+8*s**4-3*s**5,.5*s*s*(1-s)**3,10*s**3-15*s**4+6*s**5,residual


def path(v0,a0,duration,coeff,times):
    s=np.clip(np.asarray(times)/duration,0.,1.)
    carry,acc,goal,residual=bases(s)
    return carry[...,None]*duration*v0+acc[...,None]*duration**2*a0+goal[...,None]*coeff[0]+residual@coeff[1:]


def fit_path(delta,v0,a0):
    n=len(delta);xy=np.asarray(delta).cumsum(0);s=np.arange(1,n+1)/n
    carry,acc,goal,residual=bases(s);endpoint=xy[-1]
    remain=xy-carry[:,None]*n*v0-acc[:,None]*n*n*a0-goal[:,None]*endpoint
    q=np.linalg.solve(residual.T@residual+np.eye(4)*.001,residual.T@remain)
    coeff=np.vstack((endpoint,q))
    return coeff,path(v0,a0,n,coeff,np.arange(1,n+1))


def tensor_path(v0,a0,duration,coeff,s):
    z=2*s-1;e=64*s**3*(1-s)**3
    res=torch.stack((e,e*z,e*(3*z*z-1)/2,e*(5*z*z*z-3*z)/2),-1)
    carry=s-6*s**3+8*s**4-3*s**5;acc=.5*s*s*(1-s)**3;goal=10*s**3-15*s**4+6*s**5
    return (carry[None,None,:,None]*duration[:,:,None,None]*v0[:,None,None,:]
            +acc[None,None,:,None]*duration[:,:,None,None].square()*a0[:,None,None,:]
            +goal[None,None,:,None]*coeff[:,:,None,0,:]+torch.einsum('tk,bhkd->bhtd',res,coeff[:,:,1:]))
