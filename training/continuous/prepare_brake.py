"""Rebuild C2 brake supervision from actual paths on the frozen component roster."""
import argparse
from pathlib import Path
import numpy as np
from .base_data import read
from .common import sha, write
from .patterns import source
from .patterns.geometry import local_basis, fit_path


def prepare(data, components, output):
    output.mkdir(parents=True,exist_ok=False)
    receipt=read(data/'receipt.json')
    if receipt.get('status')!='COMPLETE_PHYSICAL_DATA':
        raise ValueError('Complete authenticated physical preparation is required')
    for name,digest in {**receipt['metadata_sha256'],**receipt['source_sha256']}.items():
        if sha(data/name)!=digest:
            raise ValueError('Changed physical source: '+name)
    component_receipt=read(components/'receipt.json')
    for name in ('brakes.npz','episodes.json'):
        if sha(components/name)!=component_receipt['files'][name]:
            raise ValueError('Changed component source: '+name)
    roots=source.SourceRoots(data,data/'tracking/development.json',data/'new_tracking/episodes.json')
    with np.load(components/'brakes.npz',allow_pickle=False) as z:
        original={k:z[k] for k in z.files}
    rows=read(components/'episodes.json');by_id={r['id']:i for i,r in enumerate(rows)}
    acceleration=np.zeros((len(original['x']),2),np.float32)
    truth=np.zeros((len(original['x']),17,2),np.float32)
    done=np.zeros(len(truth),bool)
    for row,ep in source.sources(roots):
        eid=by_id.get(row['id'])
        indices=np.flatnonzero(original['episode']==eid) if eid is not None else []
        if not len(indices):
            continue
        if ep is None:
            if sha(row['file'])!=row['sha256']:
                raise ValueError('Changed authentic source episode')
            with np.load(row['file'],allow_pickle=False) as z:
                ep={k:z[k] for k in z.files}
        for index in indices:
            cut=int(original['cut'][index]);duration=int(original['duration'][index])
            h=ep['delta'];history=h[cut-160:cut][None];position=ep['xy'][cut-1][None]
            known=ep['valid'][cut-1] and ep['available'][cut-1]<=cut*1000
            goal=(ep['target'][cut-1] if known else position[0])[None]
            basis=local_basis(history,position,goal)[0]
            v=h[cut-1]@basis
            a=(h[cut-8:cut].mean(0)-h[cut-16:cut-8].mean(0))/8@basis
            delta=h[cut:cut+duration]@basis
            coefficients,_=fit_path(delta,v)
            if not np.allclose(v,original['v'][index],atol=2e-6) or not np.allclose(coefficients,original['coeff'][index],atol=3e-4,rtol=2e-6):
                raise ValueError('Brake row does not match its original source/clock')
            actual=delta.cumsum(0)
            acceleration[index]=a
            truth[index]=np.column_stack([np.interp(np.linspace(0,duration,17),np.arange(duration+1),
                                         np.r_[0,actual[:,axis]]) for axis in range(2)])
            done[index]=True
    if not done.all():
        raise ValueError('Missing authentic source for a frozen brake row')
    np.savez(output/'brakes.npz',**original,acceleration=acceleration,truth=truth)
    write(output/'receipt.json',dict(status='COMPLETE',rows=len(truth),
          files={'brakes.npz':sha(output/'brakes.npz')},
          initial_brakes_sha256=sha(components/'brakes.npz'),
          data_receipt_sha256=sha(data/'receipt.json'),
          components_receipt_sha256=sha(components/'receipt.json'),
          episodes_sha256=sha(components/'episodes.json'),
          physical_contract='17 actual normalized path positions; incoming acceleration from two preceding8ms mean velocities. Same rows, roles and weights as precursor components.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.data.resolve(),a.components.resolve(),a.output.resolve())
