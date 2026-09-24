"""New3 acquisition segmentation and exact report-identity origin anchoring."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
from . import canonical as source_io
from . import canonical as acquisition
from .canonical import smooth, COMMON
from .support import read, write, sha, load

def episodes(root, ident, role):
    folder=root/ident; meta=read(folder/'metadata.json')['manifest']; info=read(folder/'tracking.json')
    frames, witness, usb=[load(folder/(name+'.npz')) for name in ('frames','witness','usb')]
    factor=float(meta['protocol_plan']['effective_radians_per_count'])/COMMON
    assert np.isfinite(factor) and factor>0 and meta['qpc_frequency']==10_000_000
    times=usb['estimated_capture_qpc']; count=usb['canonical_dxdy'].astype(np.int64)
    summed=np.vstack((np.zeros((1,2),np.int64),count.cumsum(0)))
    flag_sum=np.r_[0,(usb['quality_flags']!=0).astype(np.int64).cumsum()]
    starts={}; result=[]; excluded=[]
    for b in info['boundaries']:
        key=(b['scenario_id'],b['segment_id'])
        if b['kind'] in ('scenario_countdown','resume_countdown'):starts[key]=b
    for active,end in source_io.active_intervals(info['boundaries']):
        sid, segment=int(active['scenario_id']),int(active['segment_id'])
        identity=f'{ident}:{sid}'; r=role[identity]
        if r['split']=='excluded_old_stimulus':continue
        beginning=starts.get((sid,segment),active)
        if any(b['kind'] in ('manual_pause','focus_lost','capture_error') and beginning['qpc']<b['qpc']<active['qpc'] for b in info['boundaries']):
            beginning=active
        ix=np.flatnonzero((frames['scenario_id']==sid)&(frames['segment_id']==segment))
        local={k:v[ix] for k,v in frames.items()}
        dense,authority=acquisition.dense_from_boundary(witness,local,beginning,end)
        if dense is None:continue
        origin=dense['origin']; n=len(dense['raw'])
        endpoints=origin+np.arange(n+1,dtype=np.int64)*10000
        usbix=np.searchsorted(times,endpoints,side='right')
        valid=dense['valid'] & (flag_sum[usbix[1:]]==flag_sum[usbix[:-1]])
        valid &= (endpoints[1:]<=times[-1]) & (endpoints[:-1]>=times[0])
        guide=local['guide_enabled'][dense['selected']].astype(bool)
        changes=np.flatnonzero(guide[1:]!=guide[:-1])+1
        for fragment,(lo,hi) in enumerate(source_io.contiguous_runs(valid)):
            cuts=np.r_[lo,changes[(changes>lo)&(changes<hi)],hi]
            for part,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
                if b-a<192:
                    excluded.append(dict(parent=identity,reason='short',ticks=int(b-a)));continue
                start=origin+int(a)*10000
                raw=np.diff(summed[usbix[a:b+1]],axis=0)*factor
                initial=(dense['initial'] if a==0 else dense['cursor'][a-1])*factor
                xy=smooth(initial+raw.cumsum(0),initial)
                selected=dense['selected'][a:b]
                target=np.column_stack((local['target_x_counts'][selected],local['target_y_counts'][selected]))*factor
                radius=np.column_stack((local['radius_x_counts'][selected],local['radius_y_counts'][selected]))*factor
                available=((local['present_return_qpc'][selected]-start+9)//10).astype(np.int64)
                phase_active=endpoints[a+1:b+1]>int(active['qpc'])
                assert (available<=np.arange(1,len(xy)+1)*1000).all()
                delta=np.diff(np.vstack((initial,xy)),axis=0)
                eid=f'{ident[:8]}_{sid:03d}_{segment:03d}_{fragment:02d}_{part:02d}'
                path=root/'episodes'/(eid+'.npz')
                np.savez_compressed(path,xy=xy,initial=initial,delta=delta,target=target,radius=radius,
                    available=available,valid=np.ones(len(xy),bool),phase_active=phase_active,raw=raw)
                result.append(dict(id=eid,person=meta['user_id'],session=meta['session_id'],parent=identity,
                    group=r['group'],split=r['split'],family=r['family'],guide=bool(guide[a]),ticks=len(xy),
                    file=str(path),sha256=sha(path),factor=factor,initial_authority=authority,
                    acquisition_ticks=int((~phase_active).sum()),start_qpc=int(start)))
    return result,excluded

def select_windows(root, rows):
    candidates=[]
    for ei,row in enumerate(rows):
        if row['split']!='train':continue
        ep=load(row['file']); n=row['ticks']
        err=np.linalg.norm(ep['xy']-ep['target'],axis=1)
        speed=np.linalg.norm(ep['delta'],axis=1)
        for c in range(32,n-32,64):
            stop=min(c+256,n); quiet=np.linalg.norm(ep['delta'][c:min(c+32,n)],axis=1).max()<.01
            was_quiet=np.linalg.norm(ep['delta'][max(0,c-32):c],axis=1).max()<.01
            # Protect acquisition, entry/exit of real quiet runs and braking into
            # an observed quiet tail; accuracy filtering only removes ordinary windows.
            protected=(not ep['phase_active'][c]) or quiet or was_quiet or (speed[c:stop][-16:].max()<.01)
            candidates.append(dict(episode=ei,cut=c,quality=float(np.maximum(err[max(0,c-160):stop]-10.,0.).mean()),
                protected=bool(protected),person=row['person'],family=row['family'],guide=row['guide'],parent=row['parent']))
    by=defaultdict(list)
    for i,r in enumerate(candidates):
        if not r['protected']:by[(r['person'],r['family'],r['guide'])].append(i)
    removed=set()
    for indices in by.values():
        order=sorted(indices,key=lambda i:(candidates[i]['quality'],i),reverse=True)
        removed.update(order[:int(len(order)*.1)])
    retained=[r for i,r in enumerate(candidates) if i not in removed]
    # Equal person/family/guide/parent mass; windows within a parent share its mass.
    keys=[(r['person'],r['family'],r['guide'],r['parent']) for r in retained]
    children=defaultdict(set);counts=Counter(keys)
    for key in keys:
        for depth in range(4):children[key[:depth]].add(key[depth])
    weights=np.array([1./(np.prod([len(children[k[:d]]) for d in range(4)])*counts[k]) for k in keys])
    weights/=weights.sum()
    np.savez_compressed(root/'windows.npz',episode=np.array([r['episode'] for r in retained],np.int64),
        cut=np.array([r['cut'] for r in retained],np.int64),weights=weights,
        protected=np.array([r['protected'] for r in retained],bool))
    return dict(candidate_windows=len(candidates),retained=len(retained),removed=len(removed),
        protected=sum(r['protected'] for r in retained),filter='Worst 10% ordinary TRAIN windows per person/family/guide; protected transitions retained.',
        strata={str(k):len(v) for k,v in by.items()},windows_sha256=sha(root/'windows.npz'))

def anchor(source, destination):
    if (destination/'receipt.json').exists():raise FileExistsError('Anchored data already finalized')
    (destination/'episodes').mkdir(parents=True,exist_ok=True)
    rows=read(source/'episodes.json');output=[];audit=[]
    for session in sorted(set(r['session'] for r in rows)):
        folder=source/session[2:]
        with np.load(folder/'usb.npz') as z:u={k:z[k] for k in z.files}
        with np.load(folder/'witness.npz') as z:w={k:z[k] for k in z.files}
        nonzero=np.flatnonzero((u['canonical_dxdy']!=0).any(1))
        movement=u['canonical_dxdy'][nonzero];ut=u['estimated_capture_qpc'][nonzero]
        summed=np.r_[np.zeros((1,2),np.int64),u['canonical_dxdy'].cumsum(0)]
        wm=np.c_[w['dx_counts'],w['dy_counts']]
        for row in [r for r in rows if r['session']==session]:
            origin=row['start_qpc'];end=origin+row['ticks']*10000
            indices=np.flatnonzero((w['receipt_qpc']>origin)&(w['receipt_qpc']<end)&(wm!=0).any(1)&(w['gameplay_applied']==1))
            begin_usb=np.searchsorted(u['estimated_capture_qpc'],origin,side='right')
            anchors=[]
            # Multiple witnesses verify this is a constant origin correction.
            candidates=np.unique(np.linspace(0,max(0,len(indices)-32),min(20,max(0,len(indices)-31)),dtype=int))
            for at in candidates:
                sample=indices[at:at+32]
                if len(sample)!=32:continue
                ticks=w['receipt_qpc'][sample]
                lo=np.searchsorted(ut,ticks[0]-500000);hi=np.searchsorted(ut,ticks[0]+500000)
                matches=[]
                for j in range(lo,min(hi,len(movement)-31)):
                    if np.array_equal(movement[j:j+32],wm[sample]):matches.append(j)
                if len(matches)!=1:continue
                j=matches[0];native=int(nonzero[j]);wmfirst=int(sample[0])
                before=np.array([w['cursor_before_x_counts'][wmfirst],w['cursor_before_y_counts'][wmfirst]])
                initial=before-(summed[native]-summed[begin_usb])
                anchors.append(dict(initial=initial.tolist(),wm_index=wmfirst,usb_index=native,
                    receipt_minus_capture_us=float((ticks[0]-ut[j])/10),elapsed_ms=float((ticks[0]-origin)/10000)))
            good=len(anchors)>=2 and all(a['initial']==anchors[0]['initial'] for a in anchors)
            record=dict(id=row['id'],anchors=anchors,status='PASS' if good else 'EXCLUDED_AMBIGUOUS_OR_INCONSISTENT_ORIGIN')
            if good:
                with np.load(row['file']) as z:ep={k:z[k] for k in z.files}
                initial=np.array(anchors[0]['initial'])*row['factor'];offset=initial-ep['initial']
                ep['xy']=ep['xy']+offset;ep['initial']=initial
                # A pure translation leaves all native deltas and timings intact.
                record['correction_common']=offset.tolist()
                path=destination/'episodes'/(row['id']+'.npz');np.savez_compressed(path,**ep)
                output.append(dict(row,file=str(path),sha256=sha(path),original_episode_sha256=row['sha256'],
                    initial_authority='exact USB/WM report subsequences, consistent early/late integer origin'))
            audit.append(record)
    if len(output)<.8*len(rows):raise ValueError('Unexpected report correspondence loss; inspect before training')
    write(destination/'episodes.json',output)
    write(destination/'anchor_audit.json',audit)
    return output, audit
