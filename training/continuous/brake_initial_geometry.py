"""Finite C1 position family with independent terminal position and timing."""
import numpy as np
import torch

def bases(s):
    z=2*s-1
    envelope=16*s*s*(1-s)*(1-s)
    values=[envelope,envelope*z,envelope*(3*z*z-1)/2,envelope*(5*z*z*z-3*z)/2]
    return s*(1-s)**2,s*s*(3-2*s),np.stack(values,-1)

def path(v0,duration,coeff,times):
    s=np.clip(np.asarray(times)/duration,0.,1.)
    carry,goal,res=bases(s)
    return carry[...,None]*duration*v0+goal[...,None]*coeff[0]+res@coeff[1:]

def fit_path(delta,v0):
    n=len(delta);xy=np.asarray(delta).cumsum(0);s=np.arange(1,n+1)/n
    carry,goal,res=bases(s);endpoint=xy[-1]
    remaining=xy-carry[:,None]*n*v0-goal[:,None]*endpoint
    # Small ridge guards very short traces without fitting measurement noise.
    q=np.linalg.solve(res.T@res+np.eye(4)*.001,res.T@remaining)
    coeff=np.vstack((endpoint,q))
    pred=path(v0,n,coeff,np.arange(1,n+1))
    return coeff,pred

def tensor_path(v0,duration,coeff,s):
    # B,K,T,2; all time-normalized shapes share exactly the endpoint constraints.
    z=2*s-1;e=16*s*s*(1-s)**2
    res=torch.stack((e,e*z,e*(3*z*z-1)/2,e*(5*z*z*z-3*z)/2),-1)
    carry=s*(1-s)**2;goal=s*s*(3-2*s)
    return (carry[None,None,:,None]*duration[:,:,None,None]*v0[:,None,None,:]
            +goal[None,None,:,None]*coeff[:,:,None,0,:]
            +torch.einsum('tk,bhkd->bhtd',res,coeff[:,:,1:]))

def local_basis(history,position,target):
    error=target-position;norm=np.linalg.norm(error,axis=-1)
    v=history[:,-8:].mean(1);vn=np.linalg.norm(v,axis=-1)
    axis=np.where((norm>1e-6)[:,None],error/np.maximum(norm[:,None],1e-9),v/np.maximum(vn[:,None],1e-9))
    axis=np.where(((norm<=1e-6)&(vn<=1e-6))[:,None],np.array([1.,0.]),axis)
    return np.stack((axis,np.stack((-axis[:,1],axis[:,0]),-1)),-1)
