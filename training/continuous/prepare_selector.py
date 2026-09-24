"""Prepare authentic connected MOVE sequences for coherent choice fitting."""
from collections import Counter
from pathlib import Path
import argparse, hashlib, time
import numpy as np
import torch
from abcurves import continuous_model as motor
from abcurves.continuous_model import head_geometry as geometry
from .common import motor as load_motor, sha, write
from .base_data import read
from .patterns import source
from .patterns import prepare as packing

def order(text): return hashlib.sha256(text.encode()).hexdigest()

def weighted_median(x,w):
    ix=np.argsort(x); return float(x[ix[np.searchsorted(np.cumsum(w[ix]),w.sum()/2)]])

def prepare(out, components, roots, motor_path, device="cuda", limit=None):
    torch.set_num_threads(1); out=out.resolve(); out.mkdir(parents=True,exist_ok=False); started=time.time()
    original=components/'teacher_metadata_filtered_v1.npz'; episodes=read(components/'episodes.json')
    byid={r['id']:i for i,r in enumerate(episodes)}
    with np.load(original) as z: teacher={k:z[k] for k in ('episode','cut','role','weight')}
    net,_=load_motor(motor_path,device=device); arrays=[]; roster=[]; excluded=Counter(); source_bindings={}
    for row,ep in source.sources(roots):
        if row['id'] not in byid: continue
        eid=byid[row['id']]; ti=np.flatnonzero(teacher['episode']==eid)
        if not len(ti): continue
        roles=np.unique(teacher['role'][ti]); assert len(roles)==1
        role=int(roles[0])
        if role==2: continue  # External validation is not used even for bandwidth.
        if ep is None:
            assert sha(row['file'])==row['sha256']
            with np.load(row['file']) as z: ep={k:z[k] for k in z.files}
        source_bindings[row['file']]=sha(row['file'])
        n=len(ep['delta']); authentic=int(ep.get('authentic_length',n))
        approved=np.zeros(n,bool)
        for cut in teacher['cut'][ti]: approved[int(cut):min(n,int(cut)+128)]=True
        prior,_=source.transition_rows(ep,row)
        eligible=np.zeros(n,bool)
        if prior is not None:
            for cut,mode,brake,synth in zip(prior['cut'],prior['mode'],prior['y'][:,0],prior['synthetic']):
                if cut+32<=authentic and not mode and not brake and not synth and approved[cut:cut+32].all():
                    eligible[cut]=True
        starts=teacher['cut'][ti]; starts=starts[starts<authentic-31]
        starts=[int(c) for c in starts if eligible[int(c)]]
        # Hash sampling is independent of trajectory geometry/quality. Each source
        # retains its prior mass after sampling, conditional on authentic MOVE.
        cap=8 if row['domain']=='static' else 20
        starts=sorted(set(starts),key=lambda c:order(row['id']+'|'+str(c)))[:cap]
        if not starts: excluded[row['domain']]+=1; continue
        cuts=np.asarray(starts)[:,None]+np.arange(16)[None]*32
        mask=(cuts+32<=authentic)&eligible[np.minimum(cuts,n-1)]
        effective=np.minimum(cuts,authentic-32)
        packet=packing.pack(ep,effective.ravel(),horizon=32)
        tensors={k:torch.as_tensor(v,device=device) for k,v in packet.items()}
        c,f=motor.features(tensors['history'],tensors['position'],tensors['target'][:,:640],
            tensors['available'][:,:640],tensors['valid'][:,:640],tensors['motion_known'],tensors['cut_us'],net.config)
        with torch.no_grad():
            forecasts,_=net(c,f,tensors['history'][:,-1]); head=geometry(forecasts)
            enc=net.last_diagnostics['encoded'].detach()
        b=len(starts); positions=packet['position'].reshape(b,16,2)
        step=np.concatenate((np.zeros_like(positions[:,:1]),np.diff(positions,axis=1)),axis=1)
        truth=packet['human_delta'].reshape(b,16,8,4,2).mean(3).astype(np.float32)
        block=dict(encoded=enc.cpu().numpy().reshape(b,16,96),heads=head.cpu().numpy().reshape(b,16,16,16,2),
            displacement=step.astype(np.float32),truth=truth,mask=mask,
            role=np.full(b,role,np.int64),episode=np.full(b,eid,np.int64),cut=np.asarray(starts,np.int64),
            weight=np.full(b,float(teacher['weight'][ti].sum())/b,np.float64))
        arrays.append(block); roster.append(dict(episode=eid,id=row['id'],domain=row['domain'],role=role,
            person=row['person'],group=row['group'],sequences=b,valid_frames=int(mask.sum()),
            candidate_frames=int(mask.size),authentic_length=authentic,prior_mass=float(teacher['weight'][ti].sum())))
        if len(roster)%100==0: print('sequence sources',len(roster),'seconds',round(time.time()-started),flush=True)
        if limit is not None and len(roster)>=limit: break
    data={k:np.concatenate([r[k] for r in arrays]) for k in arrays[0]}
    # Fixed Gaussian observation bandwidth from TRAIN best-head residual only.
    # This is a smoothed emission model, not output noise or a turn penalty.
    squared=((data['heads'][...,:8,:]-data['truth'][:,:,None])**2).sum((-1,-2))
    train=data['role']==0
    weights=(data['weight'][:,None]*data['mask']/np.maximum(data['mask'].sum(1),1)[:,None])[train]
    best_rms=np.sqrt(squared[train].min(2)/16)
    sigma=max(.05,weighted_median(best_rms.ravel(),weights.ravel()))
    data['emission']=(-.5*squared/sigma**2-16*np.log(sigma*np.sqrt(2*np.pi))).astype(np.float32)
    np.savez(out/'sequences.npz',**data)
    write(out/'roster.json',roster)
    # Domain/person/source weighting is inherited, but conditioning on MOVE and
    # removing sources with no observed eligible MOVE changes the target measure.
    totals={}
    for role in (0,1):
        mask=data['role']==role
        totals[str(role)]=dict(sequences=int(mask.sum()),frames=int(data['mask'][mask].sum()),sources=len(set(data['episode'][mask].tolist())),mass=float(data['weight'][mask].sum()))
    group_roles={}
    for r in roster:
        key=r['domain']+'|'+r['group'];group_roles.setdefault(key,set()).add(r['role'])
    assert all(len(v)==1 for v in group_roles.values()),'Grouped holdout leakage'
    write(out/'receipt.json',dict(status='COMPLETE',totals=totals,sigma=sigma,
        scope='smoke' if limit else 'selected_recipe',
        files={p.name:sha(p) for p in [out/'sequences.npz',out/'roster.json']},
        motor_sha256=sha(motor_path),source_metadata_sha256=sha(original),
        source_bindings=source_bindings,elapsed_seconds=time.time()-started,
        emission='16-dimensional Gaussian over eight4ms mean velocities; TRAIN weighted-median best-head RMS, floor.05; no inference noise.'))
    print(dict(totals=totals,sigma=sigma),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Complete motor physical-data root')
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--motor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,help='Explicit source-limited smoke test')
    a=p.parse_args();d=a.data.resolve()
    roots=source.SourceRoots(d,d/'tracking/development.json',d/'new_tracking/episodes.json')
    prepare(a.output,a.components,roots,a.motor,a.device,a.limit)
