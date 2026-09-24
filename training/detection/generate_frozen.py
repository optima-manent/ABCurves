"""Generate the frozen archived study using the authenticated historical runtime.

No detector or statistical judge is run. --rows selects a qualification subset;
omit it to regenerate the complete historical pipeline and oracle raw caches.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def ah(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path(__file__).resolve().parents[2]/'recipes/detection')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',help='comma-separated frozen panel row indices; default all 320')
    a=p.parse_args();start=time.perf_counter()
    recipe=json.loads((a.recipe/'recipe.json').read_text());runtime=recipe['historical_runtime']
    for name,record in recipe['files'].items():
        if sha(a.recipe/name)!=record['sha256']:raise ValueError('Frozen recipe differs: '+name)
    for name,digest in {**runtime['source_sha256'],**runtime['artifact_sha256'],
                        runtime['native_library_path']:runtime['native_library_sha256']}.items():
        if sha(a.runtime_root/name)!=digest:raise ValueError('Historical runtime differs: '+name)
    sys.path.insert(0,str(a.runtime_root.resolve()))
    from abcurves.pipeline import Pipeline
    from abcurves.portable_renderer import PortableRendererModel
    import abcurves
    if Path(abcurves.__file__).resolve().parent!=(a.runtime_root/'abcurves').resolve():
        raise ValueError('A different ABCurves import was already loaded')
    selected=np.arange(320,dtype=np.int64) if a.rows is None else np.array([int(x) for x in a.rows.split(',')],np.int64)
    if len(selected)==0 or len(np.unique(selected))!=len(selected) or np.any((selected<0)|(selected>=320)):
        raise ValueError('Invalid frozen row subset')
    held=load(a.prepared/'held.npz');panel=load(a.recipe/'panel.npz')
    context=load(a.prepared/'panel_context.npz')['renderer_context_raw_dxdy']
    if ah(context)!=recipe['renderer_context_logical_float32_sha256']:raise ValueError('Contexts differ')
    for name,record in recipe['roles']['held']['original_arrays'].items():
        if ah(held[name])!=record['data_sha256']:raise ValueError('Frozen human input differs: '+name)
    expected=load(a.recipe/'generated_stream_hashes.npz');oracle_schedule=load(a.recipe/'oracle_schedule.npz')
    n=len(selected);generated=np.zeros((2,2,n,1000,2),np.int16)
    smooth=np.zeros((2,n,1000,2),np.float32);mask=np.zeros((2,n,1000),np.float32)
    for model_axis,seed in enumerate((7,23)):
        with Pipeline(model_seed=seed,model_dir=a.runtime_root/'models',verify_models=True,prewarm=False,torch_threads=1) as pipeline:
            if pipeline.renderer_receipt['native_library_sha256']!=runtime['native_library_sha256']:
                raise ValueError('Historical native runtime was overridden')
            for draw_axis,draw in enumerate((47007,47023)):
                for j,panel_row in enumerate(selected):
                    row=int(panel['development_row_index'][panel_row]);event_seed=int(panel[f'renderer_seed_s{seed}_d{draw}'][panel_row])
                    prefix=held['prefix_raw_dxdy'][row][held['prefix_mask'][row]>.5]
                    event=pipeline.prepare(prefix,renderer_context_raw_dxdy=context[panel_row],
                        target_rel_at_B=(float(held['target_rel_x_at_B'][row]),float(held['target_rel_y_at_B'][row])),
                        target_radius=float(held['target_radius'][row]),progress_center=float(held['progress'][row]),
                        planner_seed=event_seed,renderer_event_seed_u64=event_seed,
                        planner_head=int(panel[f'head_seed{seed}'][panel_row]),b_index_ms=float(held['b_index'][row]))
                    intent=event.planned.intent;raw=event.render_remaining()
                    generated[model_axis,draw_axis,j,:len(raw)]=raw
                    if ah(generated[model_axis,draw_axis,j])!=str(expected['pipeline_raw'][model_axis,draw_axis,panel_row]):
                        raise ValueError(f'Archived pipeline output differs: seed={seed}, draw={draw}, row={panel_row}')
                    if ah(intent.smooth_dxdy)!=str(expected['pipeline_smooth'][model_axis,panel_row]) or ah(intent.mask)!=str(expected['pipeline_mask'][model_axis,panel_row]):
                        raise ValueError('Archived planner intent or mask differs')
                    if draw_axis==0:smooth[model_axis,j]=intent.smooth_dxdy;mask[model_axis,j]=intent.mask
                    elif not np.array_equal(smooth[model_axis,j],intent.smooth_dxdy) or not np.array_equal(mask[model_axis,j],intent.mask):
                        raise ValueError('Planner changed across renderer draws')
            print('Exact archived pipeline model seed '+str(seed),flush=True)
    oracle=np.zeros((n,2,1000,2),np.int32);oracle_mask=np.zeros((n,2,1000),np.float32)
    model=PortableRendererModel(a.runtime_root/'models/renderer_global_h80.bin')
    for j,panel_row in enumerate(selected):
        row=int(panel['development_row_index'][panel_row])
        for draw_axis,draw in enumerate((0,1)):
            event=model.prepare_context(context[panel_row]).begin(held['future_smooth_dxdy'][row],held['future_mask'][row],
                event_seed=int(oracle_schedule['renderer_seed'][panel_row,draw_axis]))
            raw=event.render_remaining();oracle[j,draw_axis,:len(raw)]=raw;oracle_mask[j,draw_axis]=held['future_mask'][row]
            if ah(oracle[j,draw_axis])!=str(expected['oracle_raw'][panel_row,draw_axis]) or ah(oracle_mask[j,draw_axis])!=str(expected['oracle_mask'][panel_row,draw_axis]):
                raise ValueError(f'Archived oracle output differs: draw={draw}, row={panel_row}')
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'pipeline.npz',schema=np.array('abcurves.frozen_detection_pipeline_generated.v1'),
        selected_panel_rows=selected,source_trial_id=panel['source_trial_id'][selected],
        model_seed=np.array([7,23]),renderer_draw_label=np.array([47007,47023]),
        generated_raw_dxdy=generated,planner_smooth=smooth,generated_mask=mask)
    np.savez_compressed(a.output/'oracle.npz',schema=np.array('abcurves.frozen_detection_oracle_generated.v1'),
        selected_panel_rows=selected,source_trial_id=panel['source_trial_id'][selected],
        renderer_draw_label=np.array([0,1]),generated_raw_dxdy=oracle,generated_mask=oracle_mask)
    receipt={'schema':'abcurves.frozen_detection_generation_receipt.v1','status':'EXACT_ARCHIVED_OUTPUTS',
        'runtime_commit':runtime['commit'],'runtime_native_sha256':runtime['native_library_sha256'],
        'recipe_sha256':sha(a.recipe/'recipe.json'),'selected_panel_rows':selected.tolist(),
        'complete_panel':np.array_equal(selected,np.arange(320)),
        'pipeline_streams':4*n,'oracle_streams':2*n,'planner_smooth_and_masks_exact':True,
        'judges_run':False,'elapsed_seconds':time.perf_counter()-start,
        'outputs':{p.name:sha(p) for p in a.output.glob('*.npz')}}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
