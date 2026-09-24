"""Frozen selected source enrollment and causal transition labels."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .support import read,write,sha
from .features import physical_features,transform,FEATURES

@dataclass(frozen=True)
class SourceRoots:
    static: Path
    old_index: Path
    new_index: Path

class StaticSource:
    def __init__(self,root):
        self.root=Path(root);self.parents=read(self.root/'static_parents.json')
        self.sessions=read(self.root/'static_sessions.json');self.handles={}
    def native(self,key):
        if key not in self.handles:
            self.handles[key]=np.load(self.root/self.sessions[key]['cache_path'],mmap_mode='r',allow_pickle=False)
        return self.handles[key]

def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8))
    return np.flatnonzero(d==1),np.flatnonzero(d==-1)


def transition_rows(ep,row):
    h=ep['delta'];n=len(h);active=ep.get('phase_active',np.ones(n,bool))
    known=ep['valid']&(ep['available']<=np.arange(1,n+1)*1000)
    quiet=np.linalg.norm(h,axis=1)<=.01
    starts,ends=runs(quiet);keep=ends-starts>=32;starts=starts[keep];ends=ends[keep]
    # Retrospective braking boundary: last maximum of trailing8ms speed in
    # the128ms before an observed quiet interval. It is a label, never input.
    smoothed=np.stack([np.convolve(np.r_[np.zeros(7),h[:,j]],np.ones(8)/8,'valid') for j in range(2)],1)
    speed8=np.linalg.norm(smoothed,axis=1)
    braking=[]
    for q,a in enumerate(starts):
        lo=max(0,int(a)-128,int(ends[q-1]) if q else 0)
        segment=speed8[lo:a]
        peak=lo+int(np.flatnonzero(segment>=segment.max()-1e-12)[-1])+1 if len(segment) else int(a)
        braking.append(peak)
    cuts=np.arange(160,n-32,32,dtype=np.int64)
    # No countdown cue is supplied to the model. Never teach its externally
    # instructed waiting time as spontaneous pursuit pause/resumption.
    ok=np.array([bool(active[c-32:c+32].all() and known[c-32:c+32].all()) for c in cuts])
    cuts=cuts[ok]
    if not len(cuts):return None,[]
    previous=np.searchsorted(starts,cuts,side='right')-1
    inside=(previous>=0)&(cuts<ends[np.maximum(previous,0)]) if len(starts) else np.zeros(len(cuts),bool)
    mode=inside.astype(np.int64)
    age=np.zeros(len(cuts));goal=np.zeros((len(cuts),2));position=np.zeros_like(goal);initial_error=np.zeros(len(cuts))
    brake=np.zeros(len(cuts));resume=np.zeros(len(cuts));duration=np.ones(len(cuts))*64
    synthetic=np.zeros(len(cuts),bool)
    authentic=int(ep.get('authentic_length',n))
    for i,c in enumerate(cuts):
        if inside[i]:
            a,b=int(starts[previous[i]]),int(ends[previous[i]])
            age[i]=min(4096,c-a);goal[i]=ep['target'][a];position[i]=ep['xy'][max(0,a-1)]
            initial_error[i]=np.linalg.norm(ep['target'][a]-position[i])
            # A right-censored pause contributes survival, never a fake resume.
            resume[i]=b<n and b<=c+32
            synthetic[i]=c>=authentic or b>authentic
        else:
            previous_end=ends[ends<=c]
            age[i]=min(4096,c-int(previous_end[-1])) if len(previous_end) else min(c,4096)
            q=np.searchsorted(starts,c)
            if q<len(starts) and starts[q]-c<=128 and c>=braking[q]:
                a,b=int(starts[q]),int(ends[q])
                if active[c:min(b,c+160)].all() and known[c:min(b,c+160)].all():
                    brake[i]=1.;duration[i]=max(16,a-c)
                    synthetic[i]=a+32>authentic
    x=[]
    for lo in range(0,len(cuts),4096):
        c=cuts[lo:lo+4096];ix=c[:,None]+np.arange(-160,0)[None]
        x.append(physical_features(h[ix],ep['xy'][c-1],ep['target'][ix],ep['available'][ix],ep['valid'][ix],c*1000,
            hold_age=age[lo:lo+len(c)],hold_target=goal[lo:lo+len(c)],hold_position=position[lo:lo+len(c)],
            initial_error=initial_error[lo:lo+len(c)]))
    x=np.concatenate(x)
    # Hold-only state fields carry no meaning while moving. Use identical zeros
    # in fitting and runtime; otherwise the model could distinguish them cheaply.
    x[mode==0,17:19]=0.;x[mode==0,21]=0.
    events=[]
    for a,b in zip(starts,ends):
        if a<160 or not known[a:b].all() or not active[a:b].all():continue
        events.append(dict(episode=row['id'],start=int(a),end=int(b),duration=int(b-a),
            error=float(np.linalg.norm(ep['xy'][a]-ep['target'][a])),
            target_excursion=float(np.linalg.norm(ep['target'][a:b]-ep['target'][a],axis=-1).max()),
            left_censored=bool(a==0),right_censored=bool(b==n),synthetic=bool(b>authentic)))
    return dict(raw=x,x=transform(x),mode=mode,y=np.stack((brake,resume),1).astype(np.float32),
        duration=duration.astype(np.float32),synthetic=synthetic,cut=cuts),events


def sources(root):
    OLDINDEX=root.old_index;NEWINDEX=root.new_index
    old=read(OLDINDEX)
    for r in old['episodes']:
        if r['split'] not in ('train','validation'):continue
        # The frozen motor index has already replaced58 Person3 active episodes.
        # Those acq_ replacements are deliberately absent from this HP cohort.
        if 'acq_' in r['id']:continue
        row=dict(r,file=str((OLDINDEX.parent/r['file']).resolve()),domain='old_tracking')
        yield row,None
    for r in read(NEWINDEX):
        if r['split'] not in ('train','validation'):continue
        yield dict(r,file=str((NEWINDEX.parent/r['file']).resolve()),domain='new_tracking'),None
    base=root.static
    source=StaticSource(base);parents=source.parents
    with np.load(base/'enrollment.npz') as z:allowed=z['static_base']
    # Fixed metadata-only subsample of the existing strict TRAIN enrollment;
    # no new quality selection. 1,024 full movements, not just late windows.
    selected=sorted(allowed,key=lambda i:sha_text(parents[int(i)]['source_id']))[:1024]
    for index in selected:
        r=parents[int(index)];native=source.native(r['session_key'])
        start=int(r['presentation_index'])-160;end=int(r['end']);factor=r['factor']
        if start<4 or end-start<192:continue
        raw=np.array(native[start-4:end],np.float64)
        filtered=(raw[:-4]+2*raw[1:-3]+3*raw[2:-2]+2*raw[3:-1]+raw[4:])/9*factor
        b80=int(r['b80'])
        offset=(np.array(native[b80:start],np.float64).sum(0) if start>=b80 else -np.array(native[start:b80],np.float64).sum(0))
        initial=(offset-(8*raw[3]+6*raw[2]+3*raw[1]+raw[0])/9)*factor
        # Synthetic native-zero extension receives the same causal W5 filtering.
        extended=np.vstack((raw,np.zeros((512,2))))
        delta=(extended[:-4]+2*extended[1:-3]+3*extended[2:-2]+2*extended[3:-1]+extended[4:])/9*factor
        n=len(delta);available=np.full(n,int(r['presentation_available_us'])+160000,np.int64)
        ep=dict(delta=delta,xy=initial+delta.cumsum(0),initial=initial,
            target=np.tile(r['target_B_common'],(n,1)),available=available,valid=np.ones(n,bool),
            phase_active=np.arange(1,n+1)*1000>=available,authentic_length=end-start)
        row=dict(id='static_'+str(index),person=r['collection_key'],family='static',guide=False,
            parent=r['source_id'],group=r['session_key'],split='train',domain='static',
            static_parent=int(index),file=str((base/source.sessions[r['session_key']]['cache_path']).resolve()),
            source_id=r['source_id'],synthetic_tail_ms=512)
        yield row,ep


def sha_text(s):
    import hashlib
    return hashlib.sha256(str(s).encode()).hexdigest()


def prepare_transitions(root,OUT):
    OUT.mkdir(parents=True,exist_ok=False)
    blocks=[];rows=[];events=[]
    for row,ep in sources(root):
        if ep is None:
            if sha(row['file'])!=row['sha256']:raise ValueError('Source hash mismatch: '+row['id'])
            with np.load(row['file']) as z:ep={k:z[k] for k in z.files}
        pack,found=transition_rows(ep,row)
        if pack is None:continue
        ordinal=len(rows);row['rows']=len(pack['mode']);rows.append(row)
        pack['episode']=np.full(row['rows'],ordinal,np.int64)
        internal=int(sha_text(row['domain']+'|'+row['group'])[:8],16)%5==0
        role=2 if row['split']=='validation' else int(internal)
        pack['role']=np.full(row['rows'],role,np.int64)
        blocks.append(pack);events.extend(found)
        if len(rows)%100==0:print('prepared',len(rows),'sources',flush=True)
    arrays={k:np.concatenate([p[k] for p in blocks]) for k in blocks[0]}
    # Equal participant/family/group contribution within domain, with natural
    # frame prevalence within each group. No rare-class oversampling prior shift.
    weights=np.zeros(len(arrays['mode']),np.float64)
    for role in (0,1,2):
        for domain,mass in [('old_tracking',.35),('new_tracking',.45),('static',.20)]:
            eligible=[(i,r) for i,r in enumerate(rows) if r['domain']==domain and
                np.any((arrays['episode']==i)&(arrays['role']==role))]
            if not eligible:continue
            people={r['person'] for _,r in eligible}
            for person in people:
                pr=[(i,r) for i,r in eligible if r['person']==person]
                families={r['family'] for _,r in pr}
                for family in families:
                    fr=[(i,r) for i,r in pr if r['family']==family]
                    for i,r in fr:
                        mask=(arrays['episode']==i)&(arrays['role']==role)
                        weights[mask]=mass/len(people)/len(families)/len(fr)/mask.sum()
    weights[arrays['synthetic']]*=.15
    arrays['weight']=weights.astype(np.float32)
    np.savez_compressed(OUT/'frames.npz',**arrays)
    write(OUT/'episodes.json',rows);write(OUT/'events.json',events)
    summary={}
    for role in (0,1,2):
        for mode in (0,1):
            mask=(arrays['role']==role)&(arrays['mode']==mode)
            summary[f'role{role}_mode{mode}']=dict(rows=int(mask.sum()),positives=int(arrays['y'][mask,mode].sum()),
                mass=float(weights[mask].sum()),synthetic=int(arrays['synthetic'][mask].sum()))
    write(OUT/'receipt.json',dict(status='COMPLETE',sources=len(rows),rows=len(weights),features=FEATURES,
        source_domains=dict(Counter(r['domain'] for r in rows)),summary=summary,
        roles='0 fit, 1 internal grouped TRAIN holdout, 2 existing external validation; test/calibration untouched',
        labels='MOVE: currently after the last trailing8ms speed peak before an observed >=32ms quiet interval, ending within128ms. Duration is time to quiet. HOLD: observed movement resumes within32ms; censored tails are survival only.',
        scope='Human transition supervision only; no target-distance loss. Static512ms tails are synthetic, weight .15; countdown waits excluded.',
        bindings={str(p):sha(p) for p in (root.old_index,root.new_index,OUT/'frames.npz',OUT/'episodes.json')}))
    print(summary,flush=True)
