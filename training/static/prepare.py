"""Rebuild the selected Static Planner rows from authenticated Capture exports."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from abcurves.smoothing import smooth_dxdy
from training.capture_sources import load_frozen_session, logical, sha, write
from .selected_arrays import SelectedTargets
from .selected_normalizer import fit

SPECS = {
    'prefix_raw_dxdy': ('f4', (160, 2)), 'prefix_mask': ('u1', (160,)),
    'summary_features': ('f8', (62,)), 'y_raw': ('f4', (43,)),
    'grid_path': ('f4', (48, 2)), 'tau': ('f4', ()), 'ydot': ('f4', (2,)),
    'event_weight': ('f4', ()), 'source_trial_id': ('U48', ()),
    'cut_id': ('U48', ()), 'threshold': ('f8', ()), 'target_radius': ('f4', ()),
    'target_rel_at_b': ('f4', (2,)), 'progress': ('f4', ()),
    'b_index': ('i4', ()), 'future_duration_ms': ('i2', ()),
    'synthetic': ('u1', ()), 'user_id': ('U40', ()), 'session_id': ('U40', ()),
    'corpus': ('U24', ()), 'variant_id': ('U24', ()), 'outcome': ('U20', ()),
}


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recipe', type=Path, default=Path(__file__).resolve().parents[2]/'recipes/static')
    a = p.parse_args(argv)
    recipe = json.loads((a.recipe/'recipe.json').read_text())
    for name, record in recipe['files'].items():
        if sha(a.recipe/name) != record['sha256']:
            raise ValueError('Frozen recipe file differs: '+name)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(a.output/'receipt.json', {'status': 'BUILDING'})
    sources = rows(a.recipe/'train_sources.csv.gz')
    validation = rows(a.recipe/'validation_sources.csv.gz')
    sessions = json.loads((a.recipe/'sessions.json').read_text())['sessions']
    native = json.loads((a.recipe/'native_motion.json').read_text())['sessions']
    exports = {}
    for path in a.exports.rglob('export_manifest.json'):
        identity = json.loads(path.read_text())['source_session']['session_id']
        if identity in exports:
            raise ValueError('Duplicate export identity: '+identity)
        exports[identity] = path.parent
    # The known quarantined raw session has no selected rows or exported stream.
    needed = {r['session_id'] for r in sources + validation}
    missing = sorted(needed - exports.keys())
    if missing:
        raise ValueError('Missing selected exports: '+', '.join(missing))
    motions, bindings = {}, []
    (a.output/'native').mkdir()
    for record in sessions:
        sid = record['session_id']
        if sid not in needed:
            continue
        expected = {**record, 'source_protocol_id': 'abcurves.capture-trainer.protocol-v3',
            'ticks': native[sid]['shape'][0],
            'native_count_logical_float32_sha256': native[sid]['logical_sha256'],
            'original_mouse_csv_sha256': next(r['sha256'] for r in record['original_export_artifacts'] if r['relative_path']=='mouse_1ms.csv')}
        # Protocol identity is frozen by the original source manifest and raw hash.
        session, binding = load_frozen_session(exports[sid], expected)
        path = a.output/'native'/(sid+'.npy')
        np.save(path, session.dxdy)
        motions[sid] = np.load(path, mmap_mode='r')
        bindings.append(binding)
        print('Verified native source '+sid, flush=True)
    normalizer = json.loads((a.recipe/'normalizer.json').read_text())
    builder = SelectedTargets(recipe['selected_planner_config'], normalizer['feature_names'])
    with np.load(a.recipe/'train_cuts.npz') as z:
        cuts = dict(z)
    n = len(cuts['source_index'])
    directory = a.output/'train'; directory.mkdir()
    arrays = {k: np.lib.format.open_memmap(directory/(k+'.npy'), mode='w+', dtype=dtype, shape=(n,)+shape)
              for k, (dtype, shape) in SPECS.items()}
    previous = None
    row_hash = hashlib.sha256()
    for i, source_index in enumerate(cuts['source_index']):
        source = sources[int(source_index)]
        if source_index != previous:
            raw = np.asarray(motions[source['session_id']][int(source['dense_a']):int(source['dense_c'])], dtype=np.float32)
            smooth = smooth_dxdy(raw, 'triangular_moving_average_path:window=5')
            previous = source_index
        cut = {k: v[i] for k, v in cuts.items()}
        split = int(cut['split_index'])
        target = np.array([float(source['target_rel_a_x']), float(source['target_rel_a_y'])])-raw[:split].sum(axis=0,dtype=np.float64)
        if not np.allclose(target, [cut['target_b_x'], cut['target_b_y']], rtol=0, atol=1e-5):
            raise ValueError('Source target geometry differs')
        for k, value in builder.row(raw, cut, event_smooth=smooth).items():
            arrays[k][i] = value
        metadata = {k: source[k] for k in ('corpus','source_trial_id','session_id','user_id')}
        metadata.update(cut_id=cut['cut_id'], variant_id=cut['variant_id'], synthetic=cut['synthetic'],
            event_weight=cut['example_weight'], threshold=cut['threshold'], target_radius=cut['target_radius'],
            target_rel_at_b=target, progress=cut['center_progress'], b_index=split,
            future_duration_ms=cut['future_ms'], outcome=source['natural_outcome'])
        for k, value in metadata.items(): arrays[k][i] = value
        identity = json.dumps([source['corpus'], source['source_trial_id'], str(cut['variant_id']),
            int(cut['requested_index']), split], separators=(',',':')).encode()
        row_hash.update(len(identity).to_bytes(8,'little')); row_hash.update(identity)
        if i % 20000 == 0: print(f'Training rows: {i}/{n}', flush=True)
    if row_hash.hexdigest() != recipe['training']['row_order_sha256']:
        raise ValueError('Selected row order differs')
    for value in arrays.values(): value.flush()
    train_hashes = {k: sha(directory/(k+'.npy')) for k in arrays}
    for k, digest in train_hashes.items():
        if digest != recipe['materialized_array_file_sha256'][k]:
            raise ValueError('Selected training array differs: '+k)
    stats = fit(arrays)
    stat_sha = hashlib.sha256(json.dumps(stats, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    if stat_sha != normalizer['statistics_sha256']:
        raise ValueError('Selected normalization differs')
    write(a.output/'normalizer.json', {**normalizer, 'statistics': stats})

    # Keep the small historical development panel in its original full form;
    # its labels are fitted by the same public routine used during training.
    with np.load(a.recipe/'validation_cuts.npz') as z: val = dict(z)
    m = len(validation)
    val.update(prefix_raw_dxdy=np.zeros((m,160,2),np.float32), prefix_mask=np.zeros((m,160),np.float32),
        future_raw_dxdy=np.zeros((m,1000,2),np.float32), future_smooth_dxdy=np.zeros((m,1000,2),np.float32),
        future_mask=np.zeros((m,1000),np.float32))
    for i, source in enumerate(validation):
        raw = np.asarray(motions[source['session_id']][int(source['dense_a']):int(source['dense_c'])],dtype=np.float32)
        split, duration = int(val['b_index'][i]), int(val['future_duration_ms'][i])
        if duration != len(raw)-split: raise ValueError('Development duration differs')
        take = min(160,split)
        val['prefix_raw_dxdy'][i,-take:] = raw[split-take:split]
        val['prefix_mask'][i,-take:] = 1
        val['future_raw_dxdy'][i,:duration] = raw[split:]
        val['future_smooth_dxdy'][i,:duration] = smooth_dxdy(raw,'triangular_moving_average_path:window=5')[split:]
        val['future_mask'][i,:duration] = 1
    for k, digest in recipe['validation']['logical_array_sha256'].items():
        if logical(val[k]) != digest: raise ValueError('Frozen development array differs: '+k)
    for k, value in recipe['validation']['contract'].items(): val[k] = np.array([value])
    np.savez(a.output/'validation.npz', **val)
    write(a.output/'receipt.json', {'schema':'abcurves.static_selected_preparation.v1','status':'COMPLETE',
        'recipe_sha256':sha(a.recipe/'recipe.json'), 'train_rows':n,'validation_rows':m,
        'sources':bindings, 'train_files':train_hashes, 'validation_sha256':sha(a.output/'validation.npz'),
        'normalizer_sha256':sha(a.output/'normalizer.json'), 'statistics_sha256':stat_sha,
        'elapsed_seconds':time.perf_counter()-started})


if __name__ == '__main__': main()
