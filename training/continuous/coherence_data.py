"""Later causal tracking examples and declared synthetic functional support."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import base_data as old
from .base_data import read, sha

class Data:
    def __init__(self, root):
        self.root=Path(root)
        self.new=self.root/"new_tracking"
        NEW=self.new
        self.original=old.UnifiedData(root,verify=True)
        receipt=read(NEW/'receipt.json')
        assert receipt['status']=='COMPLETE'
        assert sha(NEW/'episodes.json')==receipt['episodes_sha256']
        assert sha(NEW/'roles.json')==receipt['roles_sha256']
        assert sha(NEW/'windows.npz')==receipt['selection']['windows_sha256']
        self.rows=read(NEW/'episodes.json')
        with np.load(NEW/'windows.npz') as z:self.windows={k:z[k] for k in z.files}
        assert all(self.rows[int(i)]['split']=='train' for i in np.unique(self.windows['episode']))

    @lru_cache(maxsize=256)
    def episode(self,index):
        row=self.rows[index]
        if row['split']!='train':raise ValueError('Non-TRAIN requested')
        if sha(self.new / row['file'])!=row['sha256']:raise ValueError('Changed episode')
        with np.load(self.new / row['file'],allow_pickle=False) as z:
            return {k:z[k] for k in ('xy','initial','delta','target','available','valid','phase_active')}

    def new_pack(self,indices,horizon):
        b=len(indices);n=640+horizon
        out=dict(history=np.zeros((b,640,2)),position=np.zeros((b,2)),target=np.zeros((b,n,2)),
            available=np.zeros((b,n),np.int64),valid=np.zeros((b,n),bool),motion_known=np.zeros((b,640),bool),
            cut_us=np.zeros(b,np.int64),human_delta=np.zeros((b,horizon,2)))
        lengths=np.empty(b,np.int64);active=np.zeros((b,horizon),bool)
        for k,index in enumerate(indices):
            ep=self.episode(int(self.windows['episode'][index]));c=int(self.windows['cut'][index]);take=min(c,640)
            h=min(horizon,len(ep['delta'])-c);offset=640-take;start=c-take
            out['history'][k,offset:]=ep['delta'][start:c];out['motion_known'][k,offset:]=True
            out['position'][k]=ep['xy'][c-1];out['cut_us'][k]=c*1000;out['human_delta'][k,:h]=ep['delta'][c:c+h]
            for key in ('target','available','valid'):out[key][k,offset:640+h]=ep[key][start:c+h]
            lengths[k]=h;active[k,:h]=ep['phase_active'][c:c+h]
        return out,lengths,active

    def teacher(self,step,seed):
        original=self.original.teacher(step,seed,'base',rotation=True,augmentation=True)
        rng=np.random.default_rng(np.random.SeedSequence([seed,101,step]))
        indices=rng.choice(len(self.windows['cut']),64,p=self.windows['weights'])
        pack,lengths,_=self.new_pack(indices,256)
        old.augment_pack(pack,rng)
        angle=float(rng.uniform(-np.pi,np.pi));old.rotate_pack(pack,angle)
        c,f=old.physical_features(pack);c,f=old.materialize_features(c,f)
        result={k:original[k] for k in ('coarse','fine','incoming','delta','length','domain')}
        appended=dict(coarse=c,fine=f,incoming=pack['history'][:,-1].astype(np.float32),
            delta=pack['human_delta'].astype(np.float32),length=lengths,domain=np.full(64,2,np.int64))
        return {k:np.concatenate((v,appended[k])) for k,v in result.items()}

    def sequences(self,step,seed,horizon=512):
        original=self.original.sequences(step,seed,'base',rotation=True,augmentation=True)
        records=[original[step%2],original[2+step%2]]
        if step%3==0:
            rng=np.random.default_rng(np.random.SeedSequence([seed,103,step]))
            index=int(rng.choice(len(self.windows['cut']),p=self.windows['weights']))
            pack,lengths,active=self.new_pack([index],horizon)
            h=int(lengths[0]);angle=float(rng.uniform(-np.pi,np.pi))
            target=pack['target'][0,640:640+h].copy();valid=pack['valid'][0,640:640+h]&active[0,:h]
            old.augment_pack(pack,rng);old.rotate_pack(pack,angle)
            records[0]=dict(pack={k:v[0] for k,v in pack.items()},H=h,domain='tracking',
                control_target=old.rotate_vectors(target,angle),control_valid=valid,
                ref=dict(new=True,episode=int(self.windows['episode'][index]),cut=int(self.windows['cut'][index])))
        for record in records:
            n=min(horizon,record['H']);pack=record['pack']
            pack['human_delta']=pack['human_delta'][:n]
            for k in ('target','valid','available'):pack[k]=pack[k][:640+n]
            record['H']=n;record['control_target']=record['control_target'][:n];record['control_valid']=record['control_valid'][:n]
            if record['domain'] in ('early','late') and record['ref']['observed_remaining']==n:
                # Append 256 ms of synthetic settling support after the authentic endpoint.
                record['ref']['synthetic_tail_ms']=256;record['ref']['authentic_horizon']=n
                pack['human_delta']=np.concatenate((pack['human_delta'],np.zeros((256,2))))
                for k in ('target','valid','available'):
                    pack[k]=np.concatenate((pack[k],np.repeat(pack[k][-1:],256,axis=0)))
                record['control_target']=np.concatenate((record['control_target'],np.repeat(record['control_target'][-1:],256,axis=0)))
                record['control_valid']=np.concatenate((record['control_valid'],np.repeat(record['control_valid'][-1:],256)))
                record['H']=n+256
        return records

def quiet_pack(count,rng,horizon=256):
    # Synthetic quiet-state support.
    n=640+horizon
    angle=rng.uniform(-np.pi,np.pi,size=count);radius=rng.uniform(0.,10.,size=count)
    error=np.c_[np.cos(angle),np.sin(angle)]*radius[:,None]
    return dict(history=np.zeros((count,640,2)),position=np.zeros((count,2)),
        target=np.broadcast_to(error[:,None],(count,n,2)).copy(),available=np.full((count,n),-1000000,np.int64),
        valid=np.ones((count,n),bool),motion_known=np.ones((count,640),bool),cut_us=np.zeros(count,np.int64),
        human_delta=np.zeros((count,horizon,2)))

def functional(step,seed,horizon=1024):
    rng=np.random.default_rng(np.random.SeedSequence([seed,107,step]))
    angle=float(rng.uniform(-np.pi,np.pi));axis=np.array([np.cos(angle),np.sin(angle)]);normal=np.array([-axis[1],axis[0]])
    kind=('hold','on_target','drift','stop_resume','curve','reversal','small_steps','hold')[step%8]
    distance=float(rng.uniform(40,800)) if kind not in ('on_target','drift') else float(rng.uniform(0,8))
    goal=axis*distance
    t=np.arange(horizon+1,dtype=float)
    truth=np.tile(goal,(horizon+1,1))
    speed=float(rng.choice([.009,.025,.06,.15,.4,1.2]))
    if kind=='drift':truth+=t[:,None]*speed*normal
    elif kind=='stop_resume':truth+=np.maximum(t-horizon*.6,0.)[:,None]*max(.15,speed)*normal
    elif kind=='reversal':truth+=(np.minimum(t,horizon*.5)-np.maximum(t-horizon*.5,0.))[:,None]*max(.15,speed)*normal
    elif kind=='small_steps':truth+=np.floor(t/200)[:,None]*2*normal
    elif kind=='curve':truth+=60*np.sin(t[:,None]*2*np.pi/1700)*normal+40*np.sin(t[:,None]*4*np.pi/1700)*axis
    incoming=np.zeros(2)
    if step%11==0:incoming=rng.normal(0,.25,2)
    pack=dict(history=np.tile(incoming,(640,1)),position=np.zeros(2),target=np.zeros((640+horizon,2)),
        available=np.zeros(640+horizon,np.int64),valid=np.zeros(640+horizon,bool),
        motion_known=np.ones(640,bool),cut_us=np.array(0,np.int64),human_delta=np.zeros((horizon,2)))
    coverage=int(rng.choice([160,640]));pack['motion_known'][:640-coverage]=False;pack['history'][:640-coverage]=0.
    cadence=int(rng.choice([1,16,33,66,100]));receipts=np.arange(horizon+1)//cadence*cadence
    if kind in ('hold','on_target'):receipts[:]=0
    pack['target'][639:]=truth[receipts];pack['available'][639:]=receipts*1000;pack['valid'][639:]=True
    return dict(pack=pack,H=horizon,domain='functional',control_target=truth[1:],
        control_valid=np.ones(horizon,bool),ref=dict(kind=kind,speed=speed,distance=distance,cadence=cadence))
