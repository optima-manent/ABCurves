"""Rebuild the selected motor's physical caches from public Capture preparations."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from abcurves.capture_exports import load_export_manifest
from .base_data import StaticSource, TrackingSource, physical_features, CACHE_SEEDS, DOMAINS
from .common import sha, write


def native_session(export, record, destination):
    """Stream Capture's authenticated half-open canonical native bins once."""
    manifest = load_export_manifest(export)
    if manifest['source_session']['session_id'] != record['session_id']:
        raise ValueError('Export session identity differs from frozen source')
    if sha(export/'export_manifest.json') != record['export_manifest_sha256']:
        raise ValueError('Export manifest differs from frozen source')
    grid = manifest['dense_grid']
    if grid['period_ns'] != 1000000 or int(grid['range']['grid_origin_unix_ns']) != int(record['grid_origin_unix_ns']):
        raise ValueError('Native clock/grid origin differs from frozen source')
    csv = export/'mouse_1ms.csv'
    expected = next(x['sha256'] for x in manifest['artifacts'] if x['relative_path']=='mouse_1ms.csv')
    if expected != record['source_mouse_csv_sha256'] or sha(csv) != expected:
        raise ValueError('Export native table differs from its checksum inventory')
    count = int(manifest['counts']['dense_millisecond_bins'])
    if count != record['native_ticks']:
        raise ValueError('Native session length differs from frozen source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(destination,mode='w+',dtype=np.int16,shape=(count,2))
    offset=0
    expected_begin=int(record['grid_origin_unix_ns'])
    for chunk in pd.read_csv(csv,usecols=['begin_unix_ns','end_unix_ns','canonical_dx','canonical_dy'],chunksize=131072):
        begin=chunk['begin_unix_ns'].to_numpy(np.int64)
        end=chunk['end_unix_ns'].to_numpy(np.int64)
        values=chunk[['canonical_dx','canonical_dy']].to_numpy()
        if not np.array_equal(begin,expected_begin+np.arange(len(chunk),dtype=np.int64)*1000000) or not np.array_equal(end,begin+1000000):
            raise ValueError('Native bins are not consecutive half-open milliseconds')
        if not np.isfinite(values).all() or not np.equal(values,np.rint(values)).all() or np.any((values < -32768)|(values > 32767)):
            raise ValueError('Invalid signed native counts')
        mapped[offset:offset+len(chunk)]=values
        offset+=len(chunk);expected_begin+=len(chunk)*1000000
    if offset != count:
        raise ValueError('Native table ended before its declared range')
    mapped.flush();del mapped
    if sha(destination) != record['cache_sha256']:
        raise ValueError('Rebuilt native bins differ from the selected model source')
    return dict(export_manifest_sha256=sha(export/'export_manifest.json'),
                mouse_csv_sha256=expected,native_sha256=sha(destination),ticks=count)


def bound_source_files(root):
    """Bind indices as well as episodes: target timing and role metadata matter."""
    return sorted(list((root/'native').glob('*.npy'))+
                  [p for name in ('tracking','new_tracking') for p in (root/name).rglob('*')
                   if p.is_file() and p.suffix in ('.npz','.json','.npy')])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports',type=Path,required=True,help='Validated static Capture export root')
    parser.add_argument('--original-tracking',type=Path,required=True,help='Assembled original225-episode cohort')
    parser.add_argument('--new-tracking',type=Path,required=True,help='Assembled later282-episode cohort')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,default=Path(__file__).resolve().parents[2]/'recipes/continuous')
    args=parser.parse_args(argv)
    root=args.output.resolve();recipe=args.recipe.resolve()
    root.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    write(root/'receipt.json',dict(status='BUILDING_PHYSICAL_DATA'))
    for filename in ('static_parents.json','tracking_rows.json','late_cuts.npy','early_cuts.npy'):
        with gzip.open(recipe/(filename+'.gz'),'rb') as src,(root/filename).open('xb') as dst:
            shutil.copyfileobj(src,dst)
    for filename in ('static_sessions.json','enrollment.npz','physical_contract.json'):
        shutil.copy2(recipe/filename,root/filename)
    sessions=json.loads((root/'static_sessions.json').read_text())
    exports={}
    for path in args.exports.rglob('export_manifest.json'):
        directory=path.parent
        manifest = load_export_manifest(directory)
        identity = manifest['source_session']['session_id']
        if identity in exports:
            raise ValueError('Duplicate static export identity')
        exports[identity]=directory
    missing=sorted({r['session_id'] for r in sessions.values()}-set(exports))
    if missing:
        raise ValueError('Missing selected source sessions: '+', '.join(missing))
    native_receipts={}
    for record in sessions.values():
        identity=record['session_id']
        if identity not in native_receipts:
            native_receipts[identity]=native_session(exports[identity],record,root/record['cache_path'])
            print('Verified native source '+identity,flush=True)
    write(root/'native_sources.json',native_receipts)
    original=json.loads((args.original_tracking/'receipt.json').read_text())
    new=json.loads((args.new_tracking/'receipt.json').read_text())
    if original['status']!='COMPLETE_ORIGINAL_MOTOR_COHORT' or original['episodes']!=225 or new['status']!='COMPLETE' or new['episodes']!=282:
        raise ValueError('Tracking preparations are not the complete selected cohorts')
    shutil.copytree(args.original_tracking,root/'tracking')
    shutil.copytree(args.new_tracking,root/'new_tracking')
    tracking_rows=json.loads((root/'tracking_rows.json').read_text())
    for file,digest in {(r['source_file'],r['source_sha256']) for r in tracking_rows}:
        if sha(root/'tracking'/file)!=digest:
            raise ValueError('Tracking episode differs from the selected source: '+file)
    source=StaticSource(root);tracking=TrackingSource(root)
    with np.load(root/'enrollment.npz') as z:
        tracking_indices=z['tracking_global']
    expected=json.loads((recipe/'expected_caches.json').read_text())
    cache={}
    for domain in DOMAINS:
        dest=root/domain
        dest.mkdir(exist_ok=(domain=='tracking'))
        specs=expected[domain]
        arrays={key:np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=spec['dtype'],shape=tuple(spec['shape'])) for key,spec in specs.items()}
        count=specs['length']['shape'][0]
        rng=np.random.default_rng(CACHE_SEEDS[domain])
        for view in (0,1):
            for start in range(0,count,1024):
                ix=np.arange(start,min(start+1024,count),dtype=np.int64)
                if domain=='tracking':
                    pack=tracking.pack(tracking_indices[ix],future_ms=256,rng=rng,augment=bool(view))
                    lengths=np.full(len(ix),256,np.int64)
                else:
                    pack,lengths=source.pack(domain,ix,256,rng=rng,augment=bool(view))
                coarse,fine=physical_features(pack)
                if not all(np.isfinite(v).all() for v in (coarse,fine,pack['human_delta'])):
                    raise ValueError('Nonfinite physical feature')
                arrays['coarse_linear'][view,ix]=coarse.astype(np.float32)
                arrays['fine_linear'][view,ix]=fine.astype(np.float32)
                incoming=pack['history'][:,-1].astype(np.float32)
                delta=pack['human_delta'].astype(np.float32)
                if view==0:
                    arrays['incoming'][ix]=incoming;arrays['delta'][ix]=delta;arrays['length'][ix]=lengths
                elif not all(np.array_equal(arrays[k][ix],v) for k,v in [('incoming',incoming),('delta',delta),('length',lengths)]):
                    raise ValueError('Visibility augmentation changed movement labels')
                if start%32768==0:
                    print(f'{domain}: view{view} {start+len(ix)}/{count}',flush=True)
        for a in arrays.values():
            a.flush()
        del arrays
        cache[domain]={}
        for key,spec in specs.items():
            path=dest/(key+'.npy');digest=sha(path)
            if digest!=spec['sha256'] or path.stat().st_size!=spec['file_bytes']:
                raise ValueError('Physical cache differs from selected source: '+domain+'/'+key)
            cache[domain][key]=spec
    metadata=['static_parents.json','static_sessions.json','tracking_rows.json','late_cuts.npy','early_cuts.npy','enrollment.npz']
    source_files=bound_source_files(root)
    write(root/'receipt.json',dict(status='COMPLETE_PHYSICAL_DATA',schema='abcurves.continuous_preparation.v1',
        cache=cache,metadata_sha256={name:sha(root/name) for name in metadata},
        source_sha256={p.relative_to(root).as_posix():sha(p) for p in source_files},
        all_selected_caches_exact=True,elapsed_seconds=time.perf_counter()-began))


if __name__=='__main__':
    main()
