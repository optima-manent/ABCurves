"""Build the self-contained ABCurves showcase from portable frozen inputs."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import zlib

ROOT = Path(__file__).resolve().parent
TYPES = {'<i2': ('<h', 2), '<f4': ('<f', 4), '<f8': ('<d', 8)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def validate(catalog, motions, provenance):
    """Reject missing, reordered, corrupt or truncated approved motion inputs."""
    ids = [meta['id'] for meta in catalog]
    expected = [entry['id'] for entry in provenance['examples']]
    if ids != expected or list(motions) != expected or len(set(ids)) != 50:
        raise ValueError('The catalog and motion must retain all 50 approved examples in order')
    if Counter(meta['section'] for meta in catalog) != {'static': 24, 'continuous': 18, 'renderer': 8}:
        raise ValueError('Expected 24 static, 18 continuous and eight renderer examples')
    for entry in provenance['examples']:
        identifier = entry['id']
        payload = motions[identifier]
        if sha(json.dumps(payload, separators=(',', ':')).encode('utf8')) != entry['payload_json_sha256']:
            raise ValueError(f'{identifier}: approved motion layout or payload changed')
        meta = next(meta for meta in catalog if meta['id'] == identifier)
        motion_keys = ['id', 'section', 'start', 'factor', 'humanFactor', 'duration', 'prefixDuration', 'human']
        metadata = {key: meta[key] for key in motion_keys}
        if sha(json.dumps(metadata, separators=(',', ':')).encode('utf8')) != entry['motion_metadata_sha256']:
            raise ValueError(f'{identifier}: approved motion scale, anchor or timing changed')
        compressed = base64.b64decode(payload['blob'], validate=True)
        if sha(compressed) != entry['compressed_motion_sha256']:
            raise ValueError(f'{identifier}: approved compressed motion changed')
        raw = zlib.decompress(compressed)
        if len(raw) != payload['bytes'] or len(raw) != entry['packed_motion_bytes']:
            raise ValueError(f'{identifier}: incorrect motion byte count')
        if sha(raw) != payload['sha256'] or sha(raw) != entry['packed_motion_sha256']:
            raise ValueError(f'{identifier}: motion checksum mismatch')
        intervals = []
        for name, spec in payload['layout'].items():
            if spec['type'] not in TYPES or not spec['shape'] or any(
                    type(dimension) is not int or dimension < 0 for dimension in spec['shape']):
                raise ValueError(f'{identifier}/{name}: invalid array type or shape')
            format_, width = TYPES[spec['type']]
            offset = spec['offset']
            end = offset + math.prod(spec['shape']) * width
            if type(offset) is not int or offset < 0 or offset % 8 or end > len(raw):
                raise ValueError(f'{identifier}/{name}: invalid array interval')
            if any(not math.isfinite(value) for (value,) in struct.iter_unpack(format_, raw[offset:end])):
                raise ValueError(f'{identifier}/{name}: nonfinite motion values')
            intervals.append((offset, end))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f'{identifier}: overlapping arrays')
    for asset in provenance['assets']:
        data = (ROOT / asset['file']).read_bytes()
        if len(data) != asset['bytes'] or sha(data) != asset['sha256']:
            raise ValueError(f"Approved asset changed: {asset['file']}")


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'docs/index.html')
    parser.add_argument('--verify-approved', action='store_true',
                        help='also require the exact approved HTML bytes, before editorial changes')
    parser.add_argument('--dump-motion', type=Path,
                        help='write readable motion JSON for inspection instead of building')
    args = parser.parse_args()
    with gzip.open(ROOT / 'data/motion.json.gz', 'rt', encoding='utf8') as archive:
        data = json.load(archive)
    if data['schema'] != 'abcurves.showcase.motion.v1':
        raise ValueError('Unsupported motion data schema')
    catalog = read_json(ROOT / 'data/catalog.json')
    provenance = read_json(ROOT / 'data/provenance.json')
    validate(catalog, data['motions'], provenance)
    if args.dump_motion is not None:
        args.dump_motion.parent.mkdir(parents=True, exist_ok=True)
        args.dump_motion.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n',
                                   encoding='utf8', newline='\n')
        print(f'Wrote {args.dump_motion}')
        return
    blocks = [f'<script type="application/json" id="motion-{meta["id"]}">'
              + json.dumps(data['motions'][meta['id']], separators=(',', ':')) + '</script>'
              for meta in catalog]
    substitutions = {
        '__STYLES__': (ROOT / 'src/styles.css').read_text(encoding='utf8'),
        '__APP__': (ROOT / 'src/app.js').read_text(encoding='utf8'),
        '__CATALOG__': json.dumps(catalog, separators=(',', ':'), ensure_ascii=False),
        '__MOTION_DATA__': '\n'.join(blocks),
    }
    template = (ROOT / 'src/index.html').read_text(encoding='utf8')
    for key, value in substitutions.items():
        if template.count(key) != 1:
            raise ValueError(f'Expected exactly one template placeholder: {key}')
        template = template.replace(key, value)
    if re.search(r'__[A-Z_]+__', template):
        raise ValueError('Unexpanded template placeholder')
    output = template.encode('utf8')
    approved = sha(output) == provenance['approved_html']['sha256']
    if args.verify_approved and not approved:
        raise ValueError('Rebuilt HTML differs from the approved deliverable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    (args.output.parent / '.nojekyll').touch()
    print(json.dumps({'output': str(args.output), 'examples': len(catalog), 'bytes': len(output),
                      'sha256': sha(output), 'matches_approved_html': approved}, indent=2))


if __name__ == '__main__':
    build()
