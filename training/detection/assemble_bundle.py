"""Assemble an exact archived descriptor bundle; never run a statistical judge."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from abcurves.judges import full_system_features
from evaluation.bundle import DescriptorBundle, write_descriptor_bundle

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def ah(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('pipeline','oracle'),required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path(__file__).resolve().parents[2]/'recipes/detection')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();recipe=json.loads((a.recipe/'recipe.json').read_text())
    for name,record in recipe['files'].items():
        if sha(a.recipe/name)!=record['sha256']:raise ValueError('Frozen recipe differs: '+name)
    data={role:load(a.prepared/f'{role}.npz') for role in ('reference','held')}
    for role,arrays in data.items():
        record=recipe['roles'][role]
        for field,expected in record['original_arrays'].items():
            if ah(arrays[field])!=expected['data_sha256']:raise ValueError('Frozen human array differs')
        if ah(arrays['causal_context'])!=record['causal_context_float32_sha256'] or ah(arrays['features'])!=record['full49_float32_sha256']:
            raise ValueError('Frozen human context or descriptor differs')
    panel=load(a.recipe/'panel.npz');indices=panel['development_row_index']
    generated=load(a.generated);hashes=load(a.recipe/'generated_stream_hashes.npz')
    if not np.array_equal(generated['source_trial_id'],panel['source_trial_id']):raise ValueError('Generated panel order differs')
    if 'selected_panel_rows' in generated and not np.array_equal(generated['selected_panel_rows'],np.arange(320)):
        raise ValueError('A subset is not the complete study')
    cells=recipe['bundles'][a.kind]['generator_cells']
    if a.kind=='pipeline':
        raw=np.asarray(generated['generated_raw_dxdy']);masks=np.asarray(generated['generated_mask'])
        if raw.shape!=(2,2,320,1000,2) or raw.dtype!=np.int16 or masks.shape!=(2,320,1000):raise ValueError('Pipeline shape/dtype differs')
        streams=[]
        for model in range(2):
            for draw in range(2):
                for row in range(320):
                    if ah(raw[model,draw,row])!=str(hashes['pipeline_raw'][model,draw,row]) or ah(masks[model,row])!=str(hashes['pipeline_mask'][model,row]):
                        raise ValueError('Archived generated pipeline stream differs')
                streams.append((raw[model,draw],masks[model]))
    else:
        raw=np.asarray(generated['generated_raw_dxdy']);masks=np.asarray(generated['generated_mask'])
        if raw.shape!=(320,2,1000,2) or raw.dtype!=np.int32 or masks.shape!=(320,2,1000):raise ValueError('Oracle shape/dtype differs')
        for row in range(320):
            for draw in range(2):
                if ah(raw[row,draw])!=str(hashes['oracle_raw'][row,draw]) or ah(masks[row,draw])!=str(hashes['oracle_mask'][row,draw]):
                    raise ValueError('Archived generated oracle stream differs')
        streams=[(raw[:,draw],masks[:,draw]) for draw in range(2)]
    ref,held=data['reference'],data['held'];nref=len(ref['source_trial_id']);nheld=len(held['source_trial_id']);human_n=nref+nheld
    targets=np.column_stack([held['target_rel_x_at_B'][indices],held['target_rel_y_at_B'][indices]])
    features=[ref['features'],held['features']]
    for raw,mask in streams:
        features.append(full_system_features(raw,mask,targets,held['target_radius'][indices],
            prefix_dxdy=held['prefix_raw_dxdy'][indices],prefix_mask=held['prefix_mask'][indices],seam_window=8,stop_window=32).astype(np.float32))
    def joined(field):
        return np.concatenate([ref[field],held[field],*[held[field][indices] for _ in cells]])
    human_order=[np.arange(nref,dtype=np.int64),np.arange(nheld,dtype=np.int64)]
    dev_audit=np.zeros(nheld,bool);dev_audit[indices]=True
    dev_audit_order=np.full(nheld,-1,np.int64);dev_audit_order[indices]=panel['audit_order']
    bundle=DescriptorBundle(features=np.concatenate(features),
        origin=np.concatenate([np.full(human_n,'human'),*[np.full(320,'generated') for _ in cells]]),
        installation_key=joined('user_id'),session_id=joined('session_id'),source_id=joined('source_trial_id'),
        order=np.concatenate([*human_order,*[human_order[1][indices] for _ in cells]]),task=joined('task_type'),
        feature_names=tuple(recipe['descriptor']['feature_names']),panel_slices={k:tuple(recipe['descriptor']['panel_slices'][k]) for k in ('trajectory','texture','full')},
        population_role=np.concatenate([np.full(nref,'reference'),np.full(nheld,'held'),*[np.full(320,'held') for _ in cells]]),
        generator_cell=np.concatenate([np.full(human_n,'human'),*[np.full(320,cell) for cell in cells]]),
        target_role=joined('target_role'),causal_context=joined('causal_context'),block_order=joined('block_ordinal'),
        audit_panel=np.concatenate([np.zeros(nref,bool),dev_audit,*[np.ones(320,bool) for _ in cells]]),
        audit_order=np.concatenate([np.full(nref,-1,np.int64),dev_audit_order,*[panel['audit_order'] for _ in cells]]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    metadata={'schema':'abcurves.frozen_detection_bundle_reproduction.v1','kind':a.kind,
        'runtime_commit':recipe['historical_runtime']['commit'],'recipe_sha256':sha(a.recipe/'recipe.json'),
        'generated_cache_sha256':sha(a.generated),'original_bundle_sha256':recipe['bundles'][a.kind]['original_sha256'],
        'scope':recipe['scope'],'judges_run':False}
    write_descriptor_bundle(a.output,bundle,metadata=metadata)
    rebuilt=load(a.output)
    checks={k:ah(rebuilt[k])==v for k,v in recipe['bundles'][a.kind]['array_data_sha256'].items()}
    if not all(checks.values()):raise ValueError('Historical bundle data differs: '+str([k for k,v in checks.items() if not v]))
    receipt={**metadata,'status':'EXACT_ARCHIVED_BUNDLE_ARRAYS','rows':len(rebuilt['features']),
        'array_checks':checks,'metadata_json_changed_to_public_provenance':True,'output_sha256':sha(a.output)}
    a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
