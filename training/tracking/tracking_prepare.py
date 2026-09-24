"""Rebuild selected tracking episodes from complete raw Capture session archives.

Original person1/person2 and later acquisition/anchoring are explicit modes;
their historical preprocessing choices must not be silently interchanged.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import subprocess
import shutil
import tempfile
import time
import numpy as np

from . import canonical
from . import new_sources
from . import original_second
from . import person3_sources
from . import original_selection
from .support import read, write, sha, load, array_inventory


def validate_source(src, validator, scratch):
    """Keep noncanonical ZIP wrappers intact; validate their identical sealed files."""
    result=subprocess.run([str(validator),'validate',str(src.path)],text=True,capture_output=True)
    if result.returncode==0:
        return dict(mode='archive',stdout=result.stdout)
    message=result.stdout+result.stderr
    if 'ZIP central member is not canonical' not in message:
        raise ValueError('Capture validation failed: '+message.strip())
    scratch.mkdir(parents=True,exist_ok=True)
    # DirectoryAwareVerifiedZip has already checked safe inventory membership.
    # Every extracted artifact is hashed before the public semantic validator runs.
    with tempfile.TemporaryDirectory(prefix='sealed-validation-',dir=scratch) as temporary:
        base=Path(temporary).resolve()
        if not base.is_relative_to(scratch.resolve()):
            raise ValueError('Validation scratch escaped its output root')
        sealed=base/Path(src.root.rstrip('/')).name
        for name in sorted(set(src.expected)|{'checksums.sha256','COMPLETE'}):
            path=(sealed/name).resolve()
            if not path.is_relative_to(base):raise ValueError('Unsafe extraction path')
            path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with src.z.open(src.root+name) as source,path.open('xb') as target:
                for block in iter(lambda:source.read(1<<20),b''):
                    digest.update(block);target.write(block)
            if name in src.expected:
                if digest.hexdigest()!=src.expected[name]:raise ValueError('Extracted artifact checksum differs')
                src.verified.add(name)
        check=subprocess.run([str(validator),'validate',str(sealed)],text=True,capture_output=True)
        if check.returncode:
            raise ValueError('Capture directory validation failed: '+(check.stdout+check.stderr).strip())
        return dict(mode='byte_identical_sealed_directory',archive_container_rejection=message.strip(),
            stdout=check.stdout,raw_archive_unchanged=True)


def project(archive, root, expected_hash, validator):
    """Canonical journals/native decoder, with current public Capture validation."""
    digest = sha(archive)
    if digest != expected_hash:
        raise ValueError('Archive differs from the frozen source identity')
    src = canonical.VerifiedZip(archive)
    try:
        validation=validate_source(src,validator,root)
        manifest = src.manifest
        if manifest['schema'] != 'abcurves.capture.session.v2':
            raise ValueError('Unsupported Capture envelope')
        if manifest['qpc_frequency'] != 10_000_000:
            raise ValueError('The selected data uses the exact 10MHz clock adapter')
        ident = manifest['session_id'].removeprefix('s-')
        out = root / ident
        out.mkdir(parents=True, exist_ok=False)
        if sum(info.file_size for info in src.z.infolist()) > 20_000_000_000:
            raise ValueError('Source exceeds the selected preparation size contract')
        write(out/'metadata.json', dict(archive=archive.name, archive_sha256=digest,
            manifest=manifest))
        journals = canonical.small_journals(src)
        native, blocks = canonical.parse_usb(src)
        anchors = journals['clocks/anchors.jsonl']
        ns_tick = 1_000_000_000 // manifest['qpc_frequency']
        qpc = np.asarray([a['qpc_midpoint'] for a in anchors], np.int64)
        utc = np.asarray([int(a['utc_unix_ns']) for a in anchors], np.int64)
        offsets = utc - qpc * ns_tick
        offset = int(np.sort(offsets)[len(offsets)//2])
        native['estimated_capture_qpc'] = (native['capture_unix_ns']-offset+ns_tick//2)//ns_tick
        if np.any(np.diff(native['estimated_capture_qpc']) < 0):
            raise ValueError('Regressed authoritative capture clock')
        # Older native derivation uses the same clock metadata for its strict binner.
        native.update(anchor_qpc=qpc, anchor_utc_unix_ns=utc,
            clock_nominal_offset_ns=np.array(offset,np.int64),
            schema=np.array('abcurves.authoritative_usb_report_audit.v1'),
            qpc_frequency=np.array(manifest['qpc_frequency'],np.int64))
        np.savez_compressed(out/'usb.npz', **native)
        del native
        print(json.dumps(dict(stage='native', source=ident)), flush=True)
        witness, witness_stats = canonical.read_witness(src)
        np.savez_compressed(out/'witness.npz', **witness)
        del witness
        print(json.dumps(dict(stage='witness', source=ident)), flush=True)
        frames, scenarios, boundaries, census = canonical.read_tracking(src)
        np.savez_compressed(out/'frames.npz', **frames)
        write(out/'tracking.json', dict(scenarios=scenarios, boundaries=boundaries, census=census))
        src.finish_verification()
        receipt = dict(status='VERIFIED', archive_sha256=digest, artifacts=src.expected,
            metadata_sha256=sha(out/'metadata.json'),
            usb_blocks=blocks, witness=witness_stats, frame_rows=len(frames['sample_qpc']),
            clock_anchor_residual_max_ns=int(np.abs(offsets-offset).max()),
            capture_validation=validation,
            projections={name:sha(out/name) for name in ('usb.npz','witness.npz','frames.npz','tracking.json')})
        write(out/'verified.json', receipt)
        print(json.dumps(dict(stage='projected', source=ident)), flush=True)
        return ident, out
    finally:
        src.z.close()


def original_episodes(projection, output, person, contract, archive):
    """Frozen original segmentation -> native right-closed bins -> causal W5."""
    f, w, usb = (load(projection/(name+'.npz')) for name in ('frames','witness','usb'))
    tracking = read(projection/'tracking.json')
    scenarios = {int(key):value for key,value in tracking['scenarios'].items()}
    manifest = read(projection/'metadata.json')['manifest']
    if person == 'person1':
        source, exclusions = canonical.make_dataset(w,f,scenarios,tracking['boundaries'],10_000_000,20260913)
    elif person == 'person2':
        source, exclusions = original_second.make_unassigned_dataset(w,f,scenarios,
            tracking['boundaries'],10_000_000,dict(manifest,__source_path=str(archive)))
    else:
        raise ValueError('Only original person1/person2 use this preparation contract')
    role_rows = {int(r['scenario_id']):r for r in contract['old_roles'] if r['source_id']==person}
    allowed = {(person,sid):row['role'] for sid,row in role_rows.items()}
    calibration = contract['old_calibrations'][person]
    factor = float(calibration['radians_per_count'])/canonical.COMMON
    if factor != float(manifest['protocol_plan']['effective_radians_per_count'])/canonical.COMMON:
        raise ValueError('Recorded sensitivity differs from frozen source calibration')
    order = np.argsort(usb['estimated_capture_qpc'],kind='stable')
    native_fields = {name:usb[name] for name in ('estimated_capture_qpc','capture_unix_ns','observed_qpc',
        'canonical_dxdy','raw_hid_dxdy','capture_sequence','pcap_sequence','reports_in_transfer',
        'report_index_in_transfer','quality_flags')}
    native = SimpleNamespace(fields=native_fields,order=order,times=usb['estimated_capture_qpc'][order],
        source_id=person,frequency=10_000_000,step=10000)
    (output/'episodes').mkdir(parents=True,exist_ok=False)
    result = []
    for e, source_id in enumerate(source['episode_id']):
        sid = int(source['scenario_id'][e]); role = role_rows[sid]
        if role['role'] not in ('train','validation'):
            continue
        lo,hi=map(int,source['episode_offsets'][e:e+2]); n=hi-lo
        row=dict(participant_id=person,scenario_id=sid,split=role['role'],
            start_qpc=int(source['start_qpc'][e]),ticks=n)
        binned=canonical.bin_development_episode(native,row,allowed)
        initial=source['start_cursor_xy'][e].astype(np.float64)*np.full(2,factor,np.float64)
        raw,cursor=canonical.derive_position(binned['native'],np.full(2,factor,np.float64),initial)
        xy=canonical.smooth(cursor,initial)
        # The selected tracking_control writer recomputes delta from W5 positions.
        arrays=dict(xy=xy,initial=initial,
            target=source['target_xy'][lo:hi].astype(np.float64)*np.full(2,factor,np.float64),
            radius=source['target_radius_xy'][lo:hi].astype(np.float64)*np.full(2,factor,np.float64),
            available=source['target_available_us'][lo:hi].astype(np.int64),
            valid=source['target_valid'][lo:hi].astype(bool))
        # Preserve the selected writer's NPZ member order as well as array values.
        arrays['delta']=np.diff(np.vstack((initial,xy)),axis=0)
        arrays['raw']=raw
        ident=f'old_{person}:{source_id}'
        name=ident.replace(':','_')+'.npz'; path=output/'episodes'/name
        np.savez_compressed(path,**arrays)
        result.append(dict(id=ident,person=person,parent=f'{person}:scenario:{sid:03d}',
            group=role['stimulus_group_id'],split=role['role'],family=str(source['family'][e]),
            guide=bool(source['guide_enabled'][e]),scenario_id=sid,source_to_common=factor,
            source_counts_per_pixel=calibration['counts_per_pixel_xy'],
            source='authenticated_old_development',ticks=n,file='episodes/'+name,sha256=sha(path)))
    write(output/'episodes.json',result)
    return result, exclusions


def prepare(args):
    if args.contract is None:
        args.contract=Path(__file__).with_name("person3_source.json" if args.source=="person3" else "frozen_sources.json")
    contract=read(args.contract)
    expected_schema='abcurves.selected_person3_source.v1' if args.source=='person3' else 'abcurves.selected_tracking_sources.v1'
    if contract['schema']!=expected_schema:
        raise ValueError('Unknown frozen source manifest')
    if args.source=='person3':
        expected=contract['archive_sha256']
    elif args.source in ('person1','person2'):
        expected=contract['old_calibrations'][args.source]['source_zip_sha256']
    elif args.source in contract['new_archive_sha256']:
        expected=contract['new_archive_sha256'][args.source]
    else:
        raise ValueError('Unknown selected source identity')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    if args.reuse_projection is None:
        ident,projection=project(args.archive,args.output/'projection',expected,args.validator)
    else:
        projection=args.reuse_projection
        receipt=read(projection/'verified.json')
        if receipt['status']!='VERIFIED' or receipt['archive_sha256']!=expected or sha(args.archive)!=expected:
            raise ValueError('Cached projection has a different raw source')
        for name,digest in receipt['projections'].items():
            if sha(projection/name)!=digest:raise ValueError('Changed projection: '+name)
        src=canonical.VerifiedZip(args.archive)
        try:
            if read(projection/'metadata.json')['manifest']!=src.manifest:
                raise ValueError('Cached manifest differs from authenticated raw source')
            ident=src.manifest['session_id'].removeprefix('s-')
        finally:
            src.z.close()
    if args.source=='person3':
        original,excluded=person3_sources.prepare_active(projection,args.output/'original-active',contract)
        write(args.output/'original-active'/'episodes.json',original)
        acquisition,bad,checks=person3_sources.prepare_acquisition(projection,args.output/'acquisition',contract,original)
        write(args.output/'acquisition'/'episodes.json',acquisition)
        write(args.output/'acquisition'/'checks.json',checks)
        replacements={row['replacement_for']:row for row in acquisition}
        if len(replacements)!=contract['acquisition_replacement_count']:
            raise ValueError('Unexpected Person3 acquisition replacement census')
        prepared=args.output/'prepared';(prepared/'episodes').mkdir(parents=True,exist_ok=False)
        result=[]
        for previous in original:
            if previous['id'] in replacements:
                row=replacements[previous['id']];source=args.output/'acquisition'
                if row['split']!='train' or any(row[k]!=previous[k] for k in ('parent','group','split','guide','family','scenario_id')):
                    raise ValueError('Acquisition replacement altered frozen custody')
            else:
                row=previous;source=args.output/'original-active'
            src=source/row['file']
            if sha(src)!=row['sha256']:raise ValueError('Prepared Person3 source changed')
            shutil.copy2(src,prepared/row['file']);result.append(row)
        write(prepared/'episodes.json',result)
        excluded+=bad
        mode='original Person3 validation, with exactly58 TRAIN acquisition replacements'
    elif args.source in ('person1','person2'):
        result,excluded=original_episodes(projection,args.output/'prepared',args.source,contract,args.archive)
        mode='original active segments; once-anchored USB virtual cursor'
    else:
        if ident!=args.source:
            raise ValueError('Source session identity differs from requested frozen source')
        if args.reuse_projection is not None:
            destination=args.output/'projection'/ident
            destination.mkdir(parents=True,exist_ok=False)
            for name in ('metadata.json','verified.json','usb.npz','witness.npz','frames.npz','tracking.json'):
                shutil.copy2(projection/name,destination/name)
            projection=destination
        root=projection.parent
        (root/'episodes').mkdir(exist_ok=False)
        rows,excluded=new_sources.episodes(root,ident,contract['new_roles']['rows'])
        write(root/'episodes.json',rows)
        result,audit=new_sources.anchor(root,args.output/'prepared')
        excluded+= [r for r in audit if r['status']!='PASS']
        for row in result:
            row['file']='episodes/'+Path(row['file']).name
        write(args.output/'prepared'/'episodes.json',result)
        mode='countdown/acquisition plus active segments; exact report-identity anchored native origin'
    receipt=dict(schema='abcurves.selected_tracking_reproduction.v1',status='COMPLETE_SOURCE_EPISODES',
        source=args.source,source_archive_sha256=expected,source_contract_sha256=sha(args.contract),
        mode=mode,episodes=len(result),counts=dict(Counter(r['split'] for r in result)),
        excluded=excluded,elapsed_seconds=time.perf_counter()-start,
        limitation='Single-source episodes only. Full-cohort filtering/weights must be run after assembling every selected source; no substitute cohort is generated.',
        numerical_arrays={r['id']:array_inventory(args.output/'prepared'/r['file']) for r in result})
    write(args.output/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','source','episodes','counts','elapsed_seconds')}),flush=True)


def compare(args):
    actual=read(args.prepared/'episodes.json')
    reference=read(args.reference)
    if isinstance(reference,dict):
        reference=reference['episodes']
    indexed={row['id']:row for row in reference}
    differences=[]; compared=0; fields=0
    if not actual:raise ValueError('Empty prepared roster is not a comparison')
    sessions={r['session'] for r in actual if 'session' in r}
    persons={r['person'] for r in actual}
    expected_ids={r['id'] for r in reference if (r.get('session') in sessions if sessions else r['person'] in persons)}
    actual_ids={r['id'] for r in actual}
    for missing in sorted(expected_ids-actual_ids):
        differences.append(dict(id=missing,reason='missing_prepared_episode'))
    for row in actual:
        if row['id'] not in indexed:
            differences.append(dict(id=row['id'],reason='missing_reference')); continue
        ref=indexed[row['id']]
        path=Path(ref['file'])
        if not path.is_absolute():path=args.reference.parent/path
        a,b=load(args.prepared/row['file']),load(path)
        for name in sorted(set(a)|set(b)):
            fields+=1
            if name not in a or name not in b:
                differences.append(dict(id=row['id'],field=name,reason='missing_field'));continue
            x,y=a[name],b[name]
            equal=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            if not equal:
                difference=dict(id=row['id'],field=name,actual_shape=list(x.shape),reference_shape=list(y.shape),
                    actual_dtype=str(x.dtype),reference_dtype=str(y.dtype))
                if x.shape==y.shape and x.dtype.kind in 'fiu':
                    difference['max_abs']=float(np.max(np.abs(x.astype(float)-y.astype(float))))
                differences.append(difference)
        for key in ('split','group','family','guide','ticks'):
            if row[key]!=ref[key]:differences.append(dict(id=row['id'],field=key,reason='metadata_mismatch'))
        compared+=1
    result=dict(status='PASS' if not differences else 'FAIL',episodes=compared,fields=fields,
        comparison='exact dtype,shape,C-order array bytes plus selected metadata',differences=differences)
    write(args.output,result);print(json.dumps(result),flush=True)
    if differences:raise SystemExit(1)


def assemble_new(args):
    """Apply the selected filter only after the complete new3 cohort is present."""
    contract=read(args.contract);expected=contract['new_archive_sha256'];sources={}
    for folder in args.inputs:
        receipt=read(folder/'receipt.json');source=receipt['source']
        if source in sources or source not in expected:
            raise ValueError('Duplicate or unexpected source in new3 assembly')
        if receipt['source_archive_sha256']!=expected[source] or receipt['source_contract_sha256']!=sha(args.contract):
            raise ValueError('Source archive or frozen role contract differs')
        sources[source]=folder
    if set(sources)!=set(expected):
        raise ValueError('All three frozen sources are required before selected window weighting')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'episodes').mkdir()
    rows=[]
    for source,folder in sorted(sources.items()):
        for row in read(folder/'prepared'/'episodes.json'):
            path=folder/'prepared'/row['file']
            if sha(path)!=row['sha256']:raise ValueError('Episode changed: '+row['id'])
            destination=args.output/'episodes'/path.name
            shutil.copy2(path,destination)
            rows.append(dict(row,file=str(destination)))
    selection=new_sources.select_windows(args.output,rows)
    for row in rows:row['file']='episodes/'+Path(row['file']).name
    write(args.output/'episodes.json',rows)
    write(args.output/'roles.json',contract['new_roles'])
    write(args.output/'receipt.json',dict(status='COMPLETE',stage='new3_cohort',sources=expected,
        episodes=len(rows),counts=dict(Counter(r['split'] for r in rows)),selection=selection,
        roles_sha256=sha(args.output/'roles.json'),episodes_sha256=sha(args.output/'episodes.json')))
    print(json.dumps(dict(episodes=len(rows),selection=selection)),flush=True)


def assemble_original(args):
    """Preserve the225-episode motor cohort and its exact phase-preserving95 law."""
    original_contract=read(Path(__file__).with_name('frozen_sources.json'))
    person3_contract=read(Path(__file__).with_name('person3_source.json'))
    expected={name:value['source_zip_sha256'] for name,value in original_contract['old_calibrations'].items()}
    expected['person3']=person3_contract['archive_sha256'];sources={}
    for folder in args.inputs:
        receipt=read(folder/'receipt.json');name=receipt['source']
        if name in sources or name not in expected or receipt['source_archive_sha256']!=expected[name]:
            raise ValueError('Original source roster or raw identity differs')
        sources[name]=folder
    if set(sources)!=set(expected):raise ValueError('All three original sources are required')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'episodes').mkdir()
    rows=[]
    for name,folder in sorted(sources.items()):
        for row in read(folder/'prepared'/'episodes.json'):
            source=folder/'prepared'/row['file'];destination=args.output/'episodes'/source.name
            if sha(source)!=row['sha256']:raise ValueError('Changed source episode')
            shutil.copy2(source,destination)
            rows.append(dict(row,file='episodes/'+source.name))
    if len(rows)!=225 or sum('replacement_for' in r for r in rows)!=58:
        raise ValueError('Frozen original cohort census differs')
    roles=dict(original=original_contract['old_roles'],person3=person3_contract['roles']['person3'],
        source_archive_sha256=expected)
    write(args.output/'roles.json',roles)
    write(args.output/'development.json',dict(schema='abcurves.tracking_control.episodes.v1',
        roles_sha256=sha(args.output/'roles.json'),episodes=rows,excluded=[],
        counts=dict(Counter(r['person']+'|'+r['split'] for r in rows)),
        acquisition='Exactly58 Person3 TRAIN episodes replace their original active-only records; validation remains unchanged.'))
    data=original_selection.TrainingData(args.output/'development.json').build()
    indices,weights,receipt=original_selection.select_phase_preserving(data,.95,False)
    if receipt['indices_sha256']!='246240fc935be731981c7edb2be3a4a4aa2ef877be320ff0adfab2c6df734689' or receipt['weights_sha256']!='44c7a9e991ee40550d973a4c7db0b627a7e913cfaaed00e4c6a0203a1d24a854':
        raise ValueError('Rebuilt phase-preserving95 enrollment differs from the selected source')
    selection=args.output/'phase_preserving95';selection.mkdir()
    np.savez_compressed(selection/'selections.npz',default_indices=indices,default_weights=weights)
    write(selection/'default_receipt.json',receipt)
    np.savez_compressed(selection/'windows.npz',**data.windows)
    write(selection/'scales.json',data.scales)
    write(args.output/'receipt.json',dict(status='COMPLETE_ORIGINAL_MOTOR_COHORT',sources=expected,
        episodes=len(rows),acquisition_replacements=58,windows=len(indices),
        acquisition_windows=receipt['acquisition_windows'],indices_sha256=receipt['indices_sha256'],
        weights_sha256=receipt['weights_sha256']))
    print(json.dumps(dict(episodes=len(rows),windows=len(indices),acquisition_windows=receipt['acquisition_windows'],
        indices_sha256=receipt['indices_sha256'],weights_sha256=receipt['weights_sha256'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validator',type=Path,required=True,help='Public Capture abct_session_tool executable')
    p.add_argument('--reuse-projection',type=Path,help='Previously verified immutable projection; skips streaming again')
    p.add_argument('--contract',type=Path)
    p.set_defaults(run=prepare)
    p=commands.add_parser('compare')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.set_defaults(run=compare)
    p=commands.add_parser('assemble-new')
    p.add_argument('--inputs',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contract',type=Path,default=Path(__file__).with_name('frozen_sources.json'))
    p.set_defaults(run=assemble_new)
    p=commands.add_parser('assemble-original')
    p.add_argument('--inputs',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.set_defaults(run=assemble_original)
    args=parser.parse_args()
    for key in ('archive','output','validator','contract','prepared','reference','reuse_projection'):
        if hasattr(args,key) and getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve())
    if hasattr(args,'inputs'):args.inputs=[path.resolve() for path in args.inputs]
    args.run(args)


if __name__=='__main__':
    main()
