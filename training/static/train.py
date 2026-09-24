"""Train a Static Planner from the complete, hash-verified selected preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from abcurves.features import build_feature_table
from abcurves.planner import PlannerConfig
from abcurves.prodmp import ProDMP, ProDMPConfig
from training import train_planner as core
from training.capture_sources import sha, write


def train(data, output, seed=7, epochs=260, device='cuda', recipe_path=None):
    recipe_path = recipe_path or Path(__file__).resolve().parents[2]/'recipes/static/recipe.json'
    recipe = json.loads(recipe_path.read_text())
    receipt = json.loads((data/'receipt.json').read_text())
    if receipt.get('status') != 'COMPLETE' or receipt['recipe_sha256'] != sha(recipe_path):
        raise ValueError('Complete selected preparation matching this recipe is required')
    for name, digest in receipt['train_files'].items():
        if sha(data/'train'/(name+'.npy')) != digest or digest != recipe['materialized_array_file_sha256'][name]:
            raise ValueError('Selected training data differs: '+name)
    for name in ('normalizer','validation'):
        filename = name+('.json' if name=='normalizer' else '.npz')
        if sha(data/filename) != receipt[name+'_sha256']:
            raise ValueError('Prepared '+name+' differs')
    if seed not in recipe['model_seeds'] or not 1 <= epochs <= 260:
        raise ValueError('Selected recipe uses model seeds 7/23 and at most 260 epochs')
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    arrays = {p.stem:np.load(p,mmap_mode='r') for p in (data/'train').glob('*.npy')}
    arrays['edge_trigger_progress'] = arrays['threshold']
    source_summary = core.source_trial_weight_summary(arrays)
    normalizer = json.loads((data/'normalizer.json').read_text())
    stats = normalizer['statistics']
    config = dict(recipe['selected_planner_config'])
    config.pop('patience', None)  # The selected run always used its full terminal budget.
    cfg = PlannerConfig(**{**config,'seed':seed,'epochs':epochs,
                          'turn_hinge_thresholds':tuple(stats['turn_hinge_thresholds'])})
    sm, ss = np.asarray(stats['summary_mean']), np.asarray(stats['summary_std'])
    ym, ys = np.asarray(stats['y_mean']), np.asarray(stats['y_std'])
    pm, ps = np.asarray(stats['prefix_mean'],np.float32), np.asarray(stats['prefix_std'],np.float32)
    def tensor(value):
        return torch.as_tensor(np.array(value,dtype=np.float32,order='C',copy=True),device=device)
    def summary(values):
        x = (np.asarray(values,dtype=np.float64)-sm)/ss
        x[~np.isfinite(x)] = 0
        return tensor(x)
    # Training prefix normalization was performed on the accelerator in float32.
    # Preserve that operation order while keeping host memory bounded.
    n = len(arrays['tau'])
    prefix = torch.empty((n,160,3),device=device)
    mask = torch.empty((n,160),device=device)
    for start in range(0,n,8192):
        raw = tensor(arrays['prefix_raw_dxdy'][start:start+8192])
        m = tensor(arrays['prefix_mask'][start:start+8192])
        normalized = (raw-tensor(pm).reshape(1,1,2))/tensor(ps).reshape(1,1,2)
        normalized[m<.5] = 0
        prefix[start:start+len(m),:,:2] = normalized
        prefix[start:start+len(m),:,2] = m
        mask[start:start+len(m)] = m
    tr = dict(prefix=prefix,mask=mask,summary=summary(arrays['summary_features']),
              target=tensor((np.asarray(arrays['y_raw'],dtype=np.float64)-ym)/ys),
              path=tensor(arrays['grid_path']),tau=tensor(arrays['tau']),ydot=tensor(arrays['ydot']))
    with np.load(data/'validation.npz') as z: val = dict(z)
    val = core.planner_input_arrays(val,prefix_len=cfg.prefix_len)
    features = build_feature_table(val,profile_bins=cfg.profile_bins)
    if features['feature_names'] != normalizer['feature_names']:
        raise ValueError('Development feature order differs')
    prodmp = ProDMP(ProDMPConfig(n_basis=cfg.n_basis,alpha=cfg.alpha,alpha_phase=cfg.alpha_phase,ridge=cfg.weight_ridge))
    labels = core.fit_weight_labels(prodmp,val,eps_scale=cfg.eps_scale,ridge=cfg.weight_ridge)
    val_y = core.build_targets(labels)
    vp, vm = core.prefix_tensor(val,pm,ps,cfg.prefix_len)
    grid = np.arange(1,cfg.grid_size+1,dtype=np.float64)/cfg.grid_size
    phi,_ = prodmp._basis_at(grid)
    path,tau,ydot = core.grid_targets(val,grid)
    va = dict(prefix=tensor(vp),mask=tensor(vm),summary=summary(features['features']),
              target=tensor((val_y-ym)/ys),path=tensor(path),tau=tensor(tau),
              ydot=tensor(ydot),weight=tensor(core.event_weights(val)))
    aux = dict(s_grid=tensor(grid),H=tensor(phi),half_alpha=prodmp.alpha/2,
               n_weights=prodmp.n_weights,horizon=1000,y_mean=tensor(ym),y_std=tensor(ys),
               dir_index=int(np.clip(round(cfg.dir_s*cfg.grid_size)-1,0,cfg.grid_size-1)))
    started = time.perf_counter()
    model, metric = core.optimize(arrays,tr,va,cfg,aux,device)
    meta = dict(feature_names=normalizer['feature_names'],summ_mean=sm,summ_std=ss,
        pmean=pm,pstd=ps,y_mean=ym[None],y_std=ys[None],horizon=1000,
        thresholds=stats['turn_hinge_thresholds'],best_val=metric,
        train_event_weight_sum=recipe['training']['physical_events'],
        val_event_weight_sum=float(core.event_weights(val).sum()),seam_contract=recipe['seam_contract'],
        cut_sampling='one_per_source_per_epoch',cut_schedule=core.SOURCE_BALANCED_CUT_SCHEDULE,
        model_selection_interval=epochs,complete_source_grid=None,train_source_trial_summary=source_summary,
        optimizer_examples_per_epoch=source_summary['source_trials'],
        train_dataset='selected-preparation/receipt.json',train_dataset_sha256=sha(data/'receipt.json'),
        val_dataset='selected-preparation/validation.npz',val_dataset_sha256=receipt['validation_sha256'])
    core.export(model,cfg,meta,output)
    write(output.with_suffix('.json'),dict(schema='abcurves.static_training.v1',
        scope='selected_recipe' if epochs==260 else 'smoke',seed=seed,epochs=epochs,
        recipe_sha256=sha(recipe_path),data_receipt_sha256=sha(data/'receipt.json'),
        checkpoint_sha256=sha(output),terminal_validation=metric,training_seconds=time.perf_counter()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[7,23],default=7)
    p.add_argument('--epochs',type=int,default=260,help='Lower values are explicitly labeled smoke fits')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    train(a.data,a.output,a.seed,a.epochs,a.device)


if __name__ == '__main__': main()
