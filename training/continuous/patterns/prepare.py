"""Human transition/brake supervision and coherent-choice source metadata."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import time
import numpy as np
from . import source
from .support import read,write,sha
from .features import context
from .geometry import fit_path

def hashorder(x):return hashlib.sha256(str(x).encode()).hexdigest()


def pack(ep,cuts,horizon=128):
    b=len(cuts);n=len(ep['delta']);size=640+horizon
    o=dict(history=np.zeros((b,640,2)),position=ep['xy'][cuts-1].copy(),target=np.zeros((b,size,2)),
        available=np.zeros((b,size),np.int64),valid=np.zeros((b,size),bool),motion_known=np.zeros((b,640),bool),
        cut_us=cuts*1000,human_delta=np.zeros((b,horizon,2)))
    for k,c in enumerate(cuts):
        take=min(640,c);lo=c-take;off=640-take;end=min(n,c+horizon)
        o['history'][k,off:]=ep['delta'][lo:c];o['motion_known'][k,off:]=True
        for key in ('target','available','valid'):o[key][k,off:640+end-c]=ep[key][lo:end]
        o['human_delta'][k,:end-c]=ep['delta'][c:end]
    return o


def prepare_components(root,PRIOR_ROOT,DATA):
    DATA.mkdir(parents=True,exist_ok=False);started=time.time()
    prior=np.load(PRIOR_ROOT/'frames.npz');rows=read(PRIOR_ROOT/'episodes.json')
    byid={r['id']:i for i,r in enumerate(rows)};seen=[];dec=[];brakes=[];teachers=[];oracle=[]
    for row,ep in source.sources(root):
        if row['id'] not in byid:continue
        eid=byid[row['id']];ix=np.flatnonzero(prior['episode']==eid)
        if ep is None:
            assert sha(row['file'])==row['sha256']
            with np.load(row['file']) as z:ep={k:z[k] for k in z.files}
        seen.append(row['id']);h=ep['delta'];n=len(h);cuts=prior['cut'][ix];mode=prior['mode'][ix];raw=prior['raw'][ix].copy()
        raw[:,15]=np.minimum(cuts,640)/640.
        qstart,qend=source.runs(np.linalg.norm(h,axis=1)<=.01);keep=qend-qstart>=32;qstart,qend=qstart[keep],qend[keep]
        smooth=np.stack([np.convolve(np.r_[np.zeros(7),h[:,j]],np.ones(8)/8,'valid') for j in range(2)],1)
        speed=np.linalg.norm(smooth,axis=1);peaks=[]
        for q,a in enumerate(qstart):
            lo=max(0,int(a)-128,int(qend[q-1]) if q else 0);part=speed[lo:a]
            peaks.append(lo+int(np.flatnonzero(part>=part.max()-1e-12)[-1])+1 if len(part) else int(a))
        labels=np.zeros((len(ix),2),np.float32);labels[:,1]=prior['y'][ix,1]
        risk=np.ones(len(ix),bool);innov_age=np.zeros(len(ix));risk_events=0
        for a,peak in zip(qstart,peaks):
            exposed=np.flatnonzero((mode==0)&(cuts>=peak)&(cuts<a))
            if len(exposed):
                first=int(exposed[0]);risk[exposed[1:]]=False
                if prior['y'][ix[first],0]>.5:labels[first,0]=1;risk_events+=1
                else:risk[first]=False
        for j,c in enumerate(cuts):
            if not mode[j]:continue
            k=np.searchsorted(qstart,c,side='right')-1
            a=int(qstart[k]);anchor=ep['target'][max(0,a-1)]
            valid=ep['valid'][a:c]&(ep['available'][a:c]<=np.arange(a+1,c+1)*1000)
            changed=np.flatnonzero(valid&(np.linalg.norm(ep['target'][a:c]-anchor,axis=1)>1e-7))
            if len(changed):innov_age[j]=c-(a+int(changed[0]))
        hist=h[cuts[:,None]+np.arange(-160,0)];target=ep['target'][cuts[:,None]+np.arange(-160,0)]
        av=ep['available'][cuts[:,None]+np.arange(-160,0)];valid=ep['valid'][cuts[:,None]+np.arange(-160,0)]
        x,basis=context(raw,hist,ep['xy'][cuts-1],target,av,valid,cuts*1000,innov_age)
        block={k:prior[k][ix] for k in ('mode','role','weight','episode','cut','synthetic')}
        block.update(x=x,y=labels,risk=risk);dec.append(block)
        indices=np.flatnonzero((mode==0)&(prior['y'][ix,0]>.5))
        for j in indices:
            c=int(cuts[j]);q=int(qstart[np.searchsorted(qstart,c)]);duration=q-c
            if not 4<=duration<=192:continue
            v=h[c-1]@basis[j];delta=h[c:q]@basis[j];coeff,pred=fit_path(delta,v)
            brakes.append(dict(x=x[j],v=v.astype(np.float32),coeff=coeff.astype(np.float32),duration=np.float32(duration),
                role=block['role'][j],weight=block['weight'][j],episode=eid,cut=c,synthetic=block['synthetic'][j]))
            oracle.append(dict(episode=eid,role=int(block['role'][j]),synthetic=bool(block['synthetic'][j]),
                duration=duration,speed=float(np.linalg.norm(v)),max_error=float(np.linalg.norm(pred-delta.cumsum(0),axis=1).max()),
                original_endpoint_error=float(np.linalg.norm(v*duration/2-delta.sum(0)))))
        # Natural exposure, no selection on cursor accuracy. Hash-selected cuts
        # cap each source; future must be observed and inside the active phase.
        active=ep.get('phase_active',np.ones(n,bool))
        eligible=[int(j) for j,c in enumerate(cuts) if c+128<=n and active[c:c+128].all()]
        chosen=np.array(sorted(eligible,key=lambda j:hashorder(row['id']+'|'+str(cuts[j])))[:96],np.int64)
        if len(chosen):
            # Only metadata enters the selected coherent-choice enrollment.
            # Preserve the exact original subsampling and natural-source mass.
            w=block['weight'][chosen]*len(eligible)/len(chosen)
            teachers.append(dict(weight=w,role=block['role'][chosen],episode=block['episode'][chosen],cut=cuts[chosen]))
        if len(seen)%100==0:print('sources',len(seen),'brakes',len(brakes),'seconds',round(time.time()-started),flush=True)
    assert set(seen)==set(byid),len(set(byid)-set(seen))
    for name,blocks in [('decisions',dec),('teacher_metadata',teachers)]:
        np.savez(DATA/(name+'.npz'),**{k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]})
    np.savez(DATA/'brakes.npz',**{k:np.asarray([b[k] for b in brakes]) for k in brakes[0]})
    write(DATA/'episodes.json',rows);write(DATA/'brake_oracle_rows.json',oracle)
    summary={}
    for role in (0,1,2):
        selected=[r for r in oracle if r['role']==role and not r['synthetic']]
        summary[str(role)]=dict(n=len(selected),sources=len({r['episode'] for r in selected}),
            max_path_error_p95=float(np.quantile([r['max_error'] for r in selected],.95)),
            original_endpoint_error_p95=float(np.quantile([r['original_endpoint_error'] for r in selected],.95)))
    paths=[DATA/(n+'.npz') for n in ('decisions','teacher_metadata','brakes')]
    write(DATA/'receipt.json',dict(status='COMPLETE',sources=len(rows),domains=dict(Counter(r['domain'] for r in rows)),
        brake_oracle=summary,elapsed_seconds=time.time()-started,
        files={p.name:sha(p) for p in paths+[DATA/'episodes.json']},
        transition_receipt_sha256=sha(PRIOR_ROOT/'receipt.json'),
        roles='0 fit, 1 grouped TRAIN holdout, 2 existing external validation; same roles as source-sealed settling study',
        weighting='Prior domain/person/family/source weights; subsample inclusion reweighting; synthetic downweight already retained'))
    print(summary,flush=True)


def filter_teacher_metadata(root,DATA):
    """Selected prior window eligibility applies to motor metadata only."""
    from collections import defaultdict
    allowed=defaultdict(set)
    oldrows=read(root.static/'tracking_rows.json')
    with np.load(root.static/'enrollment.npz') as z:
        for i in z['tracking_base']:
            r=oldrows[int(i)];allowed[r['source_id']].add(int(r['cut_ms']))
    newrows=read(root.new_index)
    with np.load(root.new_index.parent/'windows.npz') as z:
        for i,c in zip(z['episode'],z['cut']):allowed[newrows[int(i)]['id']].add(int(c))
    rows=read(DATA/'episodes.json')
    with np.load(DATA/'teacher_metadata.npz') as z:a={k:z[k] for k in z.files}
    cuts=a['cut']
    keep=np.array([rows[int(e)]['domain']=='static' or rows[int(e)]['split']=='validation' or int(c) in allowed[rows[int(e)]['id']] for e,c in zip(a['episode'],cuts)])
    filtered={k:v[keep].copy() for k,v in a.items()}
    for eid in np.unique(filtered['episode']):
        mask=filtered['episode']==eid;filtered['weight'][mask]*=a['weight'][a['episode']==eid].sum()/filtered['weight'][mask].sum()
    np.savez(DATA/'teacher_metadata_filtered_v1.npz',**filtered)
    write(DATA/'motor_filter_v1.json',dict(status='COMPLETE',before=len(keep),after=int(keep.sum()),
        role_counts={str(i):int((filtered['role']==i).sum()) for i in (0,1,2)},
        sources_before=len(np.unique(a['episode'])),sources_after=len(np.unique(filtered['episode'])),
        reason='Frozen approved motor cuts; decision/brake natural pause exposure remains unchanged.'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static',required=True,type=Path)
    p.add_argument('--old-index',required=True,type=Path)
    p.add_argument('--new-index',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();root=source.SourceRoots(a.static.resolve(),a.old_index.resolve(),a.new_index.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    source.prepare_transitions(root,a.output/'transitions')
    prepare_components(root,a.output/'transitions',a.output/'components')
    filter_teacher_metadata(root,a.output/'components')

if __name__=='__main__':main()
