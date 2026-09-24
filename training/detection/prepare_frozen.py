"""Reconstruct frozen Static B80 study inputs from authenticated Capture exports.

This stage extracts and verifies inputs only. It runs no statistical judge and
does not change any model, study membership, head choice, or renderer draw.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from abcurves.smoothing import smooth_dxdy
from abcurves.capture_preprocess import CausalBConfig, edge_progress_decision
from abcurves.judges import full_system_features
from training.capture_sources import load_frozen_session, sha, write
from .causal_context import causal_context_matrix
from .causal_ema import causal_ema_dxdy

def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path(__file__).resolve().parents[2]/'recipes/detection')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    recipe=json.loads((a.recipe/'recipe.json').read_text())
    for name,record in recipe['files'].items():
        if sha(a.recipe/name)!=record['sha256']:raise ValueError('Recipe differs: '+name)
    a.output.mkdir(exist_ok=False)
    started=time.perf_counter()
    write(a.output/'receipt.json',{'status':'BUILDING'})
    exports={}
    for path in a.exports.rglob('export_manifest.json'):
        sid=json.loads(path.read_text())['source_session']['session_id']
        if sid in exports: raise ValueError('Duplicate export: '+sid)
        exports[sid]=path.parent
    sessions=json.loads((a.recipe/'sessions.json').read_text())['sessions']
    raw={};bindings=[]
    for record in sessions:
        sid=record['session_id']
        if sid not in exports:raise ValueError('Missing frozen source export: '+sid)
        motion,binding=load_frozen_session(exports[sid],record)
        raw[sid]=motion.dxdy;bindings.append(binding)
        print('Verified '+sid,flush=True)
    receipt={'schema':'abcurves.frozen_static_b80_input_reconstruction.v1','status':'BUILDING',
        'recipe_sha256':sha(a.recipe/'recipe.json'),'sources':bindings,'roles':{}}
    datasets={}
    for role,expected in recipe['roles'].items():
        data=load(a.recipe/f'{role}_cuts.npz');n=expected['rows']
        data.update(prefix_raw_dxdy=np.zeros((n,160,2),np.float32),
            prefix_causal_smooth_dxdy=np.zeros((n,160,2),np.float32),
            prefix_mask=np.zeros((n,160),np.float32),
            future_raw_dxdy=np.zeros((n,1000,2),np.float32),
            future_smooth_dxdy=np.zeros((n,1000,2),np.float32),
            future_mask=np.zeros((n,1000),np.float32))
        for i in range(n):
            sid=str(data['session_id'][i]);begin=int(data['dense_model_start_index'][i]);end=int(data['dense_stop_index'][i])
            b=int(data['b_index'][i]);duration=int(data['future_duration_ms'][i])
            if not (0<=begin<end<=len(raw[sid])):raise ValueError('Frozen event outside session')
            event=raw[sid][begin:end]
            if len(event)!=b+duration:raise ValueError('Frozen event duration differs')
            target=np.array([data['target_rel_x_at_A'][i],data['target_rel_y_at_A'][i]])
            seam=edge_progress_decision(event,target,float(data['target_radius'][i]),b_config=CausalBConfig(threshold=.8),min_prefix_ms=24,min_future_ms=12)
            if not seam.training_eligible or seam.seam is None or seam.seam.split_index!=b:
                raise ValueError('Historical edge80 seam differs: '+str(data['source_trial_id'][i]))
            derived=target-event[:b].sum(axis=0,dtype=np.float64)
            if not np.array_equal(derived,np.array([data['target_rel_x_at_B'][i],data['target_rel_y_at_B'][i]])):
                raise ValueError('Historical B target differs')
            take=min(160,b)
            data['prefix_raw_dxdy'][i,-take:]=event[b-take:b]
            data['prefix_causal_smooth_dxdy'][i,-take:]=causal_ema_dxdy(event[:b])[-take:]
            data['prefix_mask'][i,-take:]=1
            data['future_raw_dxdy'][i,:duration]=event[b:]
            data['future_smooth_dxdy'][i,:duration]=smooth_dxdy(event,'triangular_moving_average_path:window=5')[b:]
            data['future_mask'][i,:duration]=1
        checks={k:array_hash(data[k])==record['data_sha256'] for k,record in expected['original_arrays'].items()}
        if not all(checks.values()):raise ValueError('Frozen array differs: '+str([k for k,v in checks.items() if not v]))
        context=causal_context_matrix(data).astype(np.float32)
        context_sha=array_hash(context)
        if context_sha!=expected['causal_context_float32_sha256']:raise ValueError('Frozen causal context differs')
        features=np.empty((n,49),np.float32)
        targets=np.column_stack([data['target_rel_x_at_B'],data['target_rel_y_at_B']])
        for start in range(0,n,256):
            stop=min(start+256,n);take=slice(start,stop)
            features[take]=full_system_features(data['future_raw_dxdy'][take],data['future_mask'][take],
                targets[take],data['target_radius'][take],prefix_dxdy=data['prefix_raw_dxdy'][take],
                prefix_mask=data['prefix_mask'][take],seam_window=8,stop_window=32)
        feature_sha=array_hash(features)
        if feature_sha!=expected['full49_float32_sha256']:raise ValueError('Frozen human Full49 differs')
        np.savez_compressed(a.output/f'{role}.npz',**data,causal_context=context,features=features)
        datasets[role]=data
        receipt['roles'][role]={'rows':n,'exact_original_arrays':checks,
            'all_edge80_cuts_eligible_and_exact':True,'causal_context_float32_sha256':context_sha,
            'full49_float32_sha256':feature_sha,'output_sha256':sha(a.output/f'{role}.npz')}
        print('Exact '+role+' arrays, context and human descriptors: '+str(n),flush=True)
    panel=load(a.recipe/'panel.npz');data=datasets['held']
    contexts=np.empty((320,256,2),np.float32)
    for i,row in enumerate(panel['development_row_index']):
        stop=int(data['dense_model_start_index'][row]+data['b_index'][row]);begin=stop-256
        if begin<0:raise ValueError('Panel lacks genuine preroll')
        contexts[i]=raw[str(data['session_id'][row])][begin:stop]
        valid=data['prefix_raw_dxdy'][row][data['prefix_mask'][row]>.5]
        if not np.array_equal(contexts[i,-len(valid):],valid):raise ValueError('Panel context/prefix mismatch')
    if array_hash(contexts)!=recipe['renderer_context_logical_float32_sha256']:
        raise ValueError('Historical complete 256-report contexts differ')
    np.savez_compressed(a.output/'panel_context.npz',**panel,renderer_context_raw_dxdy=contexts)
    receipt['panel_context']={'shape':list(contexts.shape),'logical_float32_sha256':array_hash(contexts),'all_suffixes_exact':True}
    receipt.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started)
    write(a.output/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds'],'context':receipt['panel_context']}))

if __name__=='__main__':main()
