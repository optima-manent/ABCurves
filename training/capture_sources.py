"""Authenticate frozen Capture source identities and canonical native evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
from abcurves import capture_exports
from abcurves.global_data import FullSession

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def logical(array, *, dtype='<f4'):
    digest = hashlib.sha256()
    for start in range(0, len(array), 1024):
        digest.update(np.ascontiguousarray(array[start:start+1024], dtype=dtype).tobytes())
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def artifact_map(items):
    result = {}
    for item in items:
        name = item['relative_path']
        if name in result:
            raise ValueError(f'duplicate artifact: {name}')
        result[name] = (item['sha256'], int(item['size_bytes']))
    return result


def load_frozen_session(directory: Path, expected: dict):
    sid = expected['session_id']
    manifest_path = directory/'export_manifest.json'
    manifest_sha = sha(manifest_path)
    manifest = capture_exports.load_export_manifest(directory)
    if manifest['source_session']['session_id'] != sid:
        raise ValueError(f'{sid}: source session identity differs')
    if manifest['source_session']['protocol_id'] != expected['source_protocol_id']:
        raise ValueError(f'{sid}: recorded protocol differs')
    if manifest['dense_grid'] != expected['original_dense_grid']:
        raise ValueError(f'{sid}: native grid clock or interval contract differs')
    if artifact_map(manifest['source_artifacts']) != artifact_map(expected['raw_source_artifacts']):
        raise ValueError(f'{sid}: raw source artifact identities differ')

    # Authenticate every actual export artifact, including the copied source
    # evidence, before reading the public adapter's tables.
    declared = artifact_map(manifest['artifacts'])
    for name, (digest, size) in declared.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe export artifact reference')
        path = directory/relative
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f'{sid}: export artifact checksum differs: {name}')
    _, events, mouse = capture_exports.load_export_tables(directory)
    users = sorted({str(value) for value in events['user_id'].dropna()})
    ids = sorted({str(value) for value in events['session_id'].dropna()})
    if users != [expected['user_id']] or ids != [sid]:
        raise ValueError(f'{sid}: event identity differs')
    grid = manifest['dense_grid']['range']
    if len(mouse) != expected['ticks'] or len(mouse) != int(grid['bin_count']):
        raise ValueError(f'{sid}: dense length differs')
    if (int(mouse['begin_unix_ns'].iloc[0]) != int(grid['first_begin_unix_ns']) or
        int(mouse['end_unix_ns'].iloc[-1]) != int(grid['last_end_unix_ns']) or
        (int(mouse['begin_unix_ns'].iloc[0])-int(grid['grid_origin_unix_ns']))//1_000_000 != int(grid['first_bin_index']) or
        (int(mouse['end_unix_ns'].iloc[-1])-int(grid['grid_origin_unix_ns']))//1_000_000 != int(grid['range_end_exclusive'])):
        raise ValueError(f'{sid}: actual absolute clock endpoints differ')
    raw = mouse[['canonical_dx', 'canonical_dy']].to_numpy(dtype=np.float32, copy=True)
    native_sha = logical(raw)
    if native_sha != expected['native_count_logical_float32_sha256']:
        raise ValueError(f'{sid}: complete native count stream differs')
    # A changing file cannot inherit the pre-read verification.
    for name, (digest, _) in declared.items():
        if sha(directory/name) != digest:
            raise ValueError(f'{sid}: export changed while reading: {name}')
    if sha(manifest_path) != manifest_sha:
        raise ValueError(f'{sid}: manifest changed while reading')
    hashes = {name:digest for name,(digest,_) in declared.items()}
    hashes['export_manifest.json'] = manifest_sha
    session = FullSession(user_id=users[0], session_id=sid, dxdy=raw,
        source_ref=directory.name, source_hashes=hashes)
    receipt = {'session_id':sid,'user_id':users[0],'ticks':len(raw),
        'raw_source_artifacts_exact':True,'source_artifacts':len(manifest['source_artifacts']),
        'actual_export_artifacts_verified':len(declared),'native_grid_exact':True,
        'native_logical_float32_sha256':native_sha,'export_manifest_sha256':manifest_sha,
        'original_export_manifest_bytes_equal':manifest_sha==expected['original_export_manifest_sha256'],
        'mouse_csv_sha256':declared['mouse_1ms.csv'][0],
        'original_mouse_csv_bytes_equal':declared['mouse_1ms.csv'][0]==expected['original_mouse_csv_sha256']}
    return session, receipt
