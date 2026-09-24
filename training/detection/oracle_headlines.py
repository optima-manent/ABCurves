"""Run the original Renderer-only headline protocol on frozen raw inputs.

Use --verify-inputs-only to authenticate and derive its exact float64 feature
matrices without running the statistical judges or producing new measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from .generate_frozen import load, sha, ah
from .oracle_headline_ops import (derive_feature_panels_from_arrays, reference_std,
    standardized_w1_report, paired_grouped_c2st)

def panel_hash(array):
    a=np.ascontiguousarray(array);h=hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'));h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path(__file__).resolve().parents[2]/'recipes/detection')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-inputs-only',action='store_true')
    a=p.parse_args();recipe=json.loads((a.recipe/'recipe.json').read_text())
    for name,r in recipe['files'].items():
        if sha(a.recipe/name)!=r['sha256']:raise ValueError('Frozen recipe differs')
    human=load(a.prepared/'held.npz');panel=load(a.recipe/'panel.npz');rows=panel['development_row_index']
    for k,r in recipe['roles']['held']['original_arrays'].items():
        if ah(human[k])!=r['data_sha256']:raise ValueError('Frozen human input differs')
    cache=load(a.generated);expected=load(a.recipe/'generated_stream_hashes.npz')
    if not np.array_equal(cache['source_trial_id'],panel['source_trial_id']):raise ValueError('Generated panel differs')
    generated=cache['generated_raw_dxdy'];masks=cache['generated_mask']
    if generated.shape!=(320,2,1000,2) or generated.dtype!=np.int32 or masks.shape!=(320,2,1000):raise ValueError('Oracle shape/dtype differs')
    for i in range(320):
        for draw in range(2):
            if ah(generated[i,draw])!=str(expected['oracle_raw'][i,draw]) or ah(masks[i,draw])!=str(expected['oracle_mask'][i,draw]):
                raise ValueError('Archived oracle reports differ')
    targets=np.column_stack([human['target_rel_x_at_B'][rows],human['target_rel_y_at_B'][rows]])
    kwargs={'prefix_dxdy':human['prefix_raw_dxdy'][rows],'prefix_mask':human['prefix_mask'][rows]}
    human_panels=derive_feature_panels_from_arrays(human['future_raw_dxdy'][rows],human['future_mask'][rows],targets,human['target_radius'][rows],**kwargs)
    candidates=[derive_feature_panels_from_arrays(generated[:,d],masks[:,d],targets,human['target_radius'][rows],**kwargs) for d in range(2)]
    spec=recipe['oracle']['headline_protocol']
    for name,values in human_panels.items():
        if panel_hash(values)!=spec['human_panel_sha256'][name]:raise ValueError('Original human float64 descriptors differ')
    for d,panels in enumerate(candidates):
        for name,values in panels.items():
            if panel_hash(values)!=spec['candidate_panel_sha256'][str(d)][name]:raise ValueError('Original generated float64 descriptors differ')
    result={'schema':'abcurves.archived_oracle_headlines.v1','recipe_sha256':sha(a.recipe/'recipe.json'),
        'rows':320,'draws':2,'feature_inputs_exact':True,'judges_run':not a.verify_inputs_only,'protocol':spec}
    if not a.verify_inputs_only:
        sessions=human['session_id'][rows]
        groups=np.array([f'user={u}|session={s}|trial={t}' for u,s,t in zip(human['user_id'][rows],sessions,human['source_trial_id'][rows])])
        parts={};metrics={}
        for key,lo,hi in [('trajectory14',0,14),('texture19',14,33),('full49',0,49)]:
            real=human_panels[key];scale=reference_std(real);names=recipe['descriptor']['feature_names'][lo:hi];estimates=[]
            for d,fake_panels in enumerate(candidates):
                fake=fake_panels[key]
                pooled=standardized_w1_report(real,fake,names,scale_reference=real)['mean']
                session_values=[standardized_w1_report(real[sessions==s],fake[sessions==s],names,scale_reference=scale)['mean'] for s in sorted(np.unique(sessions))]
                c2st=paired_grouped_c2st(real,fake,groups,seed=7,folds=5,repeats=3,bootstrap=200)
                estimates.append({'pooled_w1':pooled,'equal_session_w1':float(np.mean(session_values)),
                    'auc_orientation_free':c2st['auc_orientation_free']})
            parts[key]=estimates;metrics[key]={k:float(np.mean([v[k] for v in estimates])) for k in estimates[0]}
        result.update(draw_point_estimates=parts,metrics=metrics)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    with a.output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='protocol'}))

if __name__=='__main__':main()
