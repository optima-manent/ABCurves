"""Rebuild the selected renderer windows from validated public Capture exports.

The frozen recipe supplies source identity and order. Native report evidence
is written in the global_data / train_renderer format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from abcurves import capture_exports
from abcurves.global_data import (
    FullSession, FullSessionCollection, GlobalRendererPreparationConfig,
    assign_user_splits, save_global_renderer_dataset, write_portable_full_sessions,
)


from training.capture_sources import sha, logical, write, load_frozen_session


def materialize_frozen_renderer(exports: str | Path, recipe_path: str | Path, output: str | Path) -> dict:
    """Build frozen train/val arrays, adapter row IDs and authenticated receipts."""
    exports, recipe_path, output = Path(exports), Path(recipe_path), Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {output}')
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('schema') != 'abcurves.renderer_frozen_cohort.v1':
        raise ValueError('unsupported frozen renderer recipe')
    started = time.perf_counter()
    records = [(split, source) for split in ('train','val') for source in recipe['splits'][split]['sources']]
    # Locate exports by their manifest identities, not a historical directory naming convention.
    directories = {}
    for manifest_path in sorted(exports.rglob('export_manifest.json')):
        record = json.loads(manifest_path.read_text())
        sid = record.get('source_session',{}).get('session_id')
        if sid in directories:
            raise ValueError(f'duplicate export identity: {sid}')
        directories[sid] = manifest_path.parent
    missing = [r['session_id'] for _,r in records if r['session_id'] not in directories]
    if missing:
        raise ValueError(f'missing selected exports: {missing}')
    sessions, source_receipts = [], []
    for i, (split, expected) in enumerate(records):
        session, receipt = load_frozen_session(directories[expected['session_id']],expected)
        sessions.append(session)
        source_receipts.append({'split':split, **receipt})
        if (i+1)%5 == 0 or i+1 == len(records):
            print(f'Native sources: {i+1}/{len(records)} exact; {time.perf_counter()-started:.1f}s',flush=True)
    config = GlobalRendererPreparationConfig()
    inferred = assign_user_splits(sessions, validation_fraction=config.validation_fraction,
        split_seed=config.split_seed)
    # Verify the frozen cohort's train/validation assignment.
    for split, source in records:
        if inferred[source['user_id']] != split:
            raise ValueError('public split function does not reproduce frozen roles')
    collection = FullSessionCollection(tuple(sessions),'frozen_research_export_directories',
        recipe_path.name,sha(recipe_path),True)
    build = save_global_renderer_dataset(output,collection,config)
    split_receipts = {}
    compact = {'schema':'abcurves.renderer_frozen_window_index.v1',
        'recipe_sha256':sha(recipe_path),'native_clock':'begin-inclusive/end-exclusive 1ms bins',
        'splits':{}}
    for split in ('train','val'):
        expected = recipe['splits'][split]
        directory = output/f'renderer_{split}'
        metadata = json.loads((directory/'meta.json').read_text())
        prefix = np.load(directory/'prefix_raw_dxdy.npy',mmap_mode='r')
        future = np.load(directory/'future_raw_dxdy.npy',mmap_mode='r')
        hashes = {'prefix':logical(prefix),'future':logical(future)}
        if hashes != expected['logical_float32_sha256'] or len(prefix) != expected['windows']:
            raise ValueError(f'{split}: selected array content differs')
        ids = np.empty(len(prefix),dtype='<i8')
        row = 0
        ranges = []
        for source in expected['sources']:
            n = source['windows']; stop = row+n
            if source['prepared_row_start'] != row:
                raise ValueError('recipe compact row order is not contiguous')
            if (metadata['session_id'][row:stop] != [source['session_id']]*n or
                metadata['user_id'][row:stop] != [source['user_id']]*n or
                metadata['window_start_tick'][row:stop] != list(range(0,n*1056,1056))):
                raise ValueError(f'{split}: window identity/order differs')
            original = int(source['outer90_row_start'])
            ids[row:stop] = np.arange(original,original+n,dtype='<i8')
            ranges.append({k:source[k] for k in ['session_id','user_id','ticks','windows',
                'prepared_row_start','outer90_row_start','dropped_tail_ticks']})
            row = stop
        ids_sha = logical(ids,dtype='<i8')
        if row != len(prefix) or ids_sha != expected['outer_row_ids_raw_i64_sha256']:
            raise ValueError(f'{split}: original source row IDs differ')
        ids_path = directory/'source_row_ids.npy'
        np.save(ids_path,ids,allow_pickle=False)
        file_equal = {name:sha(directory/f'{name}_raw_dxdy.npy')==expected['original_npy_sha256'][name]
            for name in ('prefix','future')}
        split_receipts[split] = {'windows':len(prefix),'sessions':len(ranges),
            'installation_keys':len({r['user_id'] for r in ranges}),
            'logical_float32_sha256':hashes,'original_npy_bytes_equal':file_equal,
            'original_row_ids_logical_i64_sha256':ids_sha,'source_row_ids_file_sha256':sha(ids_path)}
        compact['splits'][split] = {'rows':len(prefix),'ranges':ranges,
            'source_row_ids_sha256':sha(ids_path),'source_row_ids_logical_i64_sha256':ids_sha}
    train_ids=np.load(output/'renderer_train/source_row_ids.npy',mmap_mode='r').astype(np.uint64)
    dev=(((train_ids*np.uint64(2654435761))&np.uint64(0xffffffff))%10)==0
    adapter_counts={'train':int((~dev).sum()),'development':int(dev.sum())}
    if adapter_counts != recipe['adapter_split_counts']:
        raise ValueError('selected adapter development membership differs')
    compact_path=output/'frozen_window_index.json';write(compact_path,compact)
    selection_manifest = write_portable_full_sessions(
        output/'selection_sessions',
        [session for session, (split, _) in zip(sessions, records) if split == 'val'])
    report={'schema':'abcurves.renderer_frozen_materialization.v1','status':'PASS',
        'recipe_sha256':sha(recipe_path),'sources':source_receipts,'splits':split_receipts,
        'selection_sessions_manifest_sha256':sha(selection_manifest),
        'selection_sessions_count':len(recipe['splits']['val']['sources']),
        'adapter_split_counts':adapter_counts,'frozen_window_index_sha256':sha(compact_path),
        'preparation_source_index_sha256':sha(output/'source_index.json'),
        'preparation_build_report_sha256':sha(output/'build_report.json'),
        'teacher_views':'generated by the public renderer training code from the exact raw windows; not serialized here',
        'seconds':time.perf_counter()-started}
    write(output/'frozen_materialization_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports',type=Path)
    parser.add_argument('recipe',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=materialize_frozen_renderer(args.exports,args.recipe,args.output)
    print(json.dumps({k:report[k] for k in ['status','splits','adapter_split_counts','seconds']},indent=2))


if __name__=='__main__':main()
