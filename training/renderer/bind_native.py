"""Bind one exported Renderer artifact to an explicit, separate native build."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def write_binding(artifact: Path, receipt_path: Path, output: Path):
    blob=artifact.read_bytes()
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='abcurves.renderer_export.v1' or receipt.get('mode') not in ('frozen','candidate'):
        raise ValueError('expected an explicit renderer export receipt')
    digest=hashlib.sha256(blob).hexdigest()
    if len(blob)!=44484 or receipt['combined_bytes']!=44484 or digest!=receipt['combined_sha256']:
        raise ValueError('export receipt does not authenticate this artifact')
    if blob[:8]!=b'ABCFIX1\0' or struct.unpack_from('<6H',blob,8)!=(1,256,80,20,5,2):
        raise ValueError('unsupported fixed H80/20-feature/R5 strategy2 header')
    if struct.unpack_from('<2I',blob,20)!=(15,39512):
        raise ValueError('unsupported fixed header flags or size')
    if blob[212:228]!=hashlib.sha256(b'ABCFIX1-retained-H80-fixed-schema-v1').digest()[:16] or any(blob[228:256]):
        raise ValueError('unsupported fixed feature schema or header padding')
    source=blob[180:212].hex()
    if source!=receipt['embedded_source_container_sha256']:
        raise ValueError('embedded float identity differs')
    body_crc=struct.unpack_from('<I',blob,84)[0]
    if body_crc!=(zlib.crc32(blob[256:39512])&0xffffffff):
        raise ValueError('fixed body CRC differs')
    if hashlib.sha256(blob[:39512]).hexdigest()!=receipt['base_sha256']:
        raise ValueError('fixed body identity differs')
    adapter=blob[39512:]
    if (adapter[:8]!=b'OHV1R16\0' or struct.unpack_from('<4I',adapter,8)!=(1,145,16,80)
        or hashlib.sha256(adapter).hexdigest()!=receipt['adapter_sha256']):
        raise ValueError('adapter identity or layout differs')
    for offset,count,dtype in ((24,145,'<f2'),(314,145,'<f2'),(604,16,'<f4'),
                              (668,16,'<f4'),(3052,80,'<f4'),(3372,80,'<f4')):
        if not np.isfinite(np.frombuffer(adapter,dtype=dtype,count=count,offset=offset)).all():
            raise ValueError('non-finite packed adapter value')
    config=struct.unpack_from('<11i',blob,136)
    adapter_crc=zlib.crc32(adapter)&0xffffffff
    record={'schema':'abcurves.renderer_native_binding.v1','mode':receipt['mode'],
        'is_selected_release':digest=='8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b','artifact_sha256':digest,
        'artifact_bytes':44484,'export_receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'source_container_sha256':source,'fixed_body_crc32':f'{body_crc:08x}',
        'adapter_crc32':f'{adapter_crc:08x}','quantized_sampling_config':list(config),
        'scope':'one specific exported H80/20-feature/R5/rank16 artifact; no change to default release anchors'}
    header='''/* Generated per-export binding. Use only in its separate native build. */
#ifndef ABC_MODEL_BINDING_H
#define ABC_MODEL_BINDING_H
#include <stdint.h>
'''
    header+=f'#define ABC_BOUND_BODY_CRC32 UINT32_C(0x{body_crc:08x})\n'
    header+=f'#define ABC_BOUND_ADAPTER_CRC32 UINT32_C(0x{adapter_crc:08x})\n'
    header+='#define ABC_BOUND_SOURCE_SHA256 {'+', '.join(f'0x{x:02x}' for x in bytes.fromhex(source))+'}\n'
    header+='#define ABC_BOUND_CONFIG {'+', '.join(str(x) for x in config)+'}\n'
    header+=f'#define ABC_BOUND_ARTIFACT_SHA256 "{digest}"\n'
    header+=f'#define ABC_BOUND_IS_SELECTED_RELEASE {int(record["is_selected_release"])}\n'
    header+='#endif\n'
    output.mkdir(parents=True,exist_ok=False)
    (output/'abc_model_binding.h').write_text(header)
    (output/'binding.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(write_binding(a.artifact,a.receipt,a.output),indent=2))


if __name__=='__main__':main()
