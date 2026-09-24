import numpy as np
from .geometry import local_basis

FEATURES=['distance','speed1','speed8','speed32','speed_ratio8_32','target_speed32',
 'target_speed128','closing_speed','lateral_speed','error_growth32','deceleration',
 'target_excursion128','target_unchanged_ms','known_target_fraction','target_known',
 'history_known','state_age_ms','hold_target_excursion','hold_position_excursion',
 'target_speed8','cursor_target_alignment','initial_hold_error']


SCALES=np.array([10,.1,.1,.1,1,.1,.1,.1,.1,.1,.01,10,128,1,1,1,128,10,10,.1,1,10])


def physical_features(history,position,target,available,valid,cut_us,*,hold_age=None,
                      hold_target=None,hold_position=None,initial_error=None,motion_known=None):
    """One implementation for human examples and runtime; no future indexing."""
    h=np.asarray(history,np.float64);p=np.asarray(position,np.float64);g=np.asarray(target,np.float64)
    b,n=h.shape[:2];times=np.asarray(cut_us)[:,None]+np.arange(1-n,1)[None]*1000
    known=np.asarray(valid,bool)&(np.asarray(available)<=times)&np.isfinite(g).all(-1)
    latest=known[:,-1];goal=np.where(latest[:,None],g[:,-1],p)
    error=goal-p;d=np.linalg.norm(error,axis=-1);axis=error/np.maximum(d[:,None],1e-8)
    v1=h[:,-1];v8=h[:,-8:].mean(1);v32=h[:,-32:].mean(1)
    speed=lambda x:np.linalg.norm(x,axis=-1)
    tv={k:np.where((latest&known[:,-1-k])[:,None],(goal-g[:,-1-k])/k,0.) for k in (8,32,128)}
    previous_p=p-h[:,-32:].sum(1);previous_error=speed(g[:,-33]-previous_p)
    growth=np.where(latest&known[:,-33],(d-previous_error)/32,0.)
    acceleration=(v8-h[:,-16:-8].mean(1))/8
    changed=(speed(g[:,-128:]-goal[:,None])>1e-8)|~known[:,-128:]
    last_change=np.max(np.where(changed,np.arange(128)[None],-1),axis=1)
    age=127-last_change
    excursion=np.max(np.where(known[:,-128:],speed(g[:,-128:]-goal[:,None]),0.),axis=1)
    zeros=np.zeros(b)
    hold_age=zeros if hold_age is None else np.asarray(hold_age)
    target_exc=zeros if hold_target is None else speed(goal-np.asarray(hold_target))
    position_exc=zeros if hold_position is None else speed(p-np.asarray(hold_position))
    initial_error=zeros if initial_error is None else np.asarray(initial_error)
    values=np.stack([d,speed(v1),speed(v8),speed(v32),speed(v8)/np.maximum(speed(v32),.005),
        speed(tv[32]),speed(tv[128]),(v8*axis).sum(-1),np.abs(v8[:,0]*axis[:,1]-v8[:,1]*axis[:,0]),
        growth,-(acceleration*v8).sum(-1)/np.maximum(speed(v8),.005),excursion,age,
        known[:,-128:].mean(1),latest.astype(float),
        np.ones(b) if motion_known is None else np.asarray(motion_known).mean(1),hold_age,
        target_exc,position_exc,speed(tv[8]),(v8*tv[32]).sum(-1)/np.maximum(speed(v8)*speed(tv[32]),1e-8),initial_error],1)
    return values.astype(np.float32)


def transform(x):return np.arcsinh(np.asarray(x)/SCALES).astype(np.float32)

def context(raw,h,p,g,available,valid,cut,innovation_age=None):
    raw=np.array(raw,copy=True);raw[:,16]=np.minimum(raw[:,16],256.)
    times=np.asarray(cut)[:,None]+np.arange(1-h.shape[1],1)[None]*1000
    known=np.asarray(valid)&(available<=times)
    goal=np.where(known[:,-1,None],g[:,-1],p)
    basis=local_basis(h,p,goal)
    def local(x):return np.einsum('bi,bij->bj',x,basis)
    v1=h[:,-1];v8=h[:,-8:].mean(1);acc=(v8-h[:,-16:-8].mean(1))/8
    tv=[np.where((known[:,-1]&known[:,-1-k])[:,None],(goal-g[:,-1-k])/k,0.) for k in (32,128)]
    extra=np.column_stack((local(v1),local(acc)*8,local(tv[0]),local(tv[1]),local(v8)[:,1],
        np.minimum(2048.,np.zeros(len(h)) if innovation_age is None else innovation_age)/128.))
    return np.concatenate((transform(raw),np.arcsinh(extra)),1).astype(np.float32),basis
