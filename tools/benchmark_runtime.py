"""Measure the public runtime in fresh processes, without device pacing.

Run from a source checkout with --output DIRECTORY. Each worker measures only
that checkout. Samples, receipts and dedicated Numba caches go beneath output.
The default full campaign preserves the selected release measurement protocol.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, importlib.util, json, os, platform, subprocess, sys, time
from pathlib import Path

RELEASE: Path
OUTPUT: Path
FIXTURES: Path
PROFILE: Path
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def bindings():
    paths = sorted((RELEASE/'abcurves').rglob('*.py'))
    paths += sorted(p for p in (RELEASE/'abcurves/_native').iterdir() if p.suffix in {'.dll', '.so', '.dylib'})
    paths += [RELEASE/'models'/s for s in ('planner_seed7.pt', 'planner_seed23.pt', 'renderer_global_h80.bin')]
    paths += sorted((RELEASE/'models/continuous').glob('*'))
    return {p.relative_to(RELEASE).as_posix(): sha(p) for p in paths if p.is_file()}

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')

def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if not len(a): return {'n': 0}
    return dict(n=len(a), median_us=float(np.median(a)/1000), p95_us=float(np.percentile(a,95)/1000),
                max_us=float(a.max()/1000), mean_us=float(a.mean()/1000), sum_seconds=float(a.sum()/1e9))

def metadata():
    import psutil
    import threadpoolctl
    versions = {}
    for name in ('numpy','numba','llvmlite','torch','onnxruntime','abcurves'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    import abcurves
    pools = threadpoolctl.threadpool_info()
    for pool in pools:
        if 'filepath' in pool: pool['filepath'] = Path(pool['filepath']).name
    return dict(python=sys.version, executable=Path(sys.executable).name, platform=platform.platform(),
        imported_version=abcurves.__version__, import_scope='explicit source checkout',
        cpu=platform.processor(), physical_cores=psutil.cpu_count(logical=False), logical_cores=os.cpu_count(),
        process_affinity=psutil.Process().cpu_affinity() if hasattr(psutil.Process(), 'cpu_affinity') else None, versions=versions,
        env={k:os.environ.get(k) for k in THREAD_ENV},
        numba_cache='dedicated beneath output', thread_pools=pools, priority=str(psutil.Process().nice()),
        gc='enabled; no timing samples discarded', clock=time.get_clock_info('perf_counter')._asdict()
        if hasattr(time.get_clock_info('perf_counter'),'_asdict') else str(time.get_clock_info('perf_counter')))

def initial_object(kind, *, prewarm):
    import numpy as np
    if kind=='cp':
        from abcurves import ContinuousPlanner
        return ContinuousPlanner(seed=7, prewarm=prewarm)
    from abcurves import ContinuousPipeline, CountTransform
    with np.load(PROFILE, allow_pickle=False) as f:
        profile=np.array(f['profile_hardware']); radians=float(f['radians_per_count'])
    return ContinuousPipeline(profile, transform=CountTransform(radians, y_down=True),
                              seed=7, renderer_seed=47007, prewarm=prewarm)

def movement(obj, kind):
    return obj if kind=='cp' else obj.movement

def startup_worker(kind, model_seed):
    started=time.perf_counter_ns()
    if kind=='static':
        from abcurves import StaticPipeline
        import torch
        torch.set_num_interop_threads(1)
    elif kind=='cp':
        from abcurves import ContinuousPlanner
    else:
        from abcurves import ContinuousPipeline, CountTransform
    imported=time.perf_counter_ns()
    if kind=='static':
        obj=StaticPipeline(model_seed=model_seed, prewarm=True, torch_threads=1)
        loaded=time.perf_counter_ns()
        import numpy as np
        fixture=np.load(FIXTURES/'static_event.npz', allow_pickle=False)
        t=time.perf_counter_ns(); profile=obj.prepare_renderer_profile(fixture['profile_raw_dxdy']); profile_ns=time.perf_counter_ns()-t
        b=int(fixture['b_index']); args=dict(target_rel_at_B=fixture['target_rel_b'],target_radius=float(fixture['target_radius']),
            progress_center=float(fixture['progress_center']),planner_seed=7,renderer_event_seed_u64=47007,b_index_ms=b)
        t=time.perf_counter_ns(); stream=obj.prepare(fixture['raw_dxdy'][:b],renderer_profile=profile,**args); first_ns=time.perf_counter_ns()-t
        result=dict(import_ns=imported-started, constructor_with_prewarm_ns=loaded-imported,
                    prepared_profile_ns=profile_ns, first_real_handoff_ns=first_ns, first_duration_ms=stream.duration_ms)
        obj.close()
    else:
        obj=initial_object(kind,prewarm=False)
        loaded=time.perf_counter_ns()
        movement(obj,kind).planner.prewarm()
        warmed=time.perf_counter_ns()
        # No live stream state was consumed by policy prewarm.
        obj.update_target((180.,100.),0)
        t=time.perf_counter_ns(); output=obj.advance(1000); first_ns=time.perf_counter_ns()-t
        assert output['xy'].shape==(1,2)
        result=dict(import_ns=imported-started, load_verify_without_prewarm_ns=loaded-imported,
            explicit_policy_prewarm_ns=warmed-loaded, load_and_prewarm_ns=warmed-imported,
            first_fresh_output_ns=first_ns, first_motor_evaluations=movement(obj,kind).planner.motor_evaluations)
    result.update(kind=kind, model_seed=model_seed if kind=='static' else None, metadata=metadata())
    return result

def target_receipts():
    import math
    events={0:(180.,100.)}
    for t in range(1600,5201,16):
        phase=(t-1600)/1000.
        events[t]=(180.+160.*math.sin(phase*1.6),100.+110.*math.sin(phase*2.1))
    events[6400]=(-160.,140.)
    return events

def warm_continuous(kind, repeats):
    import numpy as np
    obj=initial_object(kind,prewarm=True)
    policy=movement(obj,kind).planner
    receipts=target_receipts()
    all_times=[]; all_fresh=[]; all_motor=[]; all_mode=[]; all_reps=[]
    receipt_times=[]; reset_times=[]; runs=[]; output_digest=hashlib.sha256()
    warm_started=time.perf_counter_ns()
    # One complete discarded trace exercises moving targets, brake/hold/restart.
    for rep in range(-1,repeats):
        seed=7+rep if rep>=0 else 999
        t=time.perf_counter_ns()
        if kind=='cp': obj.reset(seed=seed)
        else: obj.reset(seed=seed,renderer_seed=47000+seed)
        reset_ns=time.perf_counter_ns()-t
        if rep>=0: reset_times.append(reset_ns)
        times=[]; fresh=[]; motor=[]; modes=[]; per_receipt=[]; points=[]; reports=[]
        run_started=time.perf_counter_ns()
        for tick in range(8000):
            if tick in receipts:
                t=time.perf_counter_ns(); obj.update_target(receipts[tick],tick*1000); elapsed=time.perf_counter_ns()-t
                if rep>=0: per_receipt.append(elapsed)
            before=movement(obj,kind).fresh_decisions
            before_motor=policy.motor_evaluations
            t=time.perf_counter_ns(); out=obj.advance((tick+1)*1000); elapsed=time.perf_counter_ns()-t
            assert out['xy'].shape==(1,2) and out['time_us'][0]==(tick+1)*1000
            assert np.isfinite(out['xy']).all()
            if rep>=0:
                times.append(elapsed); fresh.append(movement(obj,kind).fresh_decisions>before)
                motor.append(policy.motor_evaluations>before_motor); modes.append(policy.mode)
                points.append(out['xy'])
                if kind!='cp': reports.append(out['reports'])
        run_wall=time.perf_counter_ns()-run_started
        if rep<0:
            warmup_ns=time.perf_counter_ns()-warm_started
            continue
        a=np.asarray(times,np.int64); f=np.asarray(fresh,bool); m=np.asarray(motor,bool)
        receipt_times.extend(per_receipt); all_times.extend(times); all_fresh.extend(fresh); all_motor.extend(motor)
        all_mode.extend(modes); all_reps.extend([rep]*len(times))
        points=np.concatenate(points)
        output_digest.update(points.tobytes())
        if reports: output_digest.update(np.concatenate(reports).tobytes())
        runs.append(dict(repetition=rep,planner_seed=seed,renderer_seed=None if kind=='cp' else 47000+seed,
            simulated_ms=8000, wall_including_instrumentation_ms=run_wall/1e6,
            fresh=stats(a[f]),buffered=stats(a[~f]),fresh_with_motor=stats(a[f&m]),
            fresh_without_motor=stats(a[f&~m]), motor_evaluations=policy.motor_evaluations,
            modes={str(k):int(np.sum(np.asarray(modes)==k)) for k in sorted(set(modes))}))
    a=np.asarray(all_times,np.int64); f=np.asarray(all_fresh,bool); m=np.asarray(all_motor,bool)
    np.savez_compressed(OUTPUT/f'{kind}_warm_samples.npz',elapsed_ns=a,fresh=f,motor=m,
                        mode=np.asarray(all_mode,np.int8),repetition=np.asarray(all_reps,np.int8),
                        receipt_ns=np.asarray(receipt_times,np.int64),reset_ns=np.asarray(reset_times,np.int64))
    return dict(kind=kind,repeats=repeats,numerical_ms_per_run=8000,discarded_warmup_ms=warmup_ns/1e6,
        fresh=stats(a[f]), buffered=stats(a[~f]), fresh_with_motor=stats(a[f&m]),
        fresh_without_motor=stats(a[f&~m]), every_1ms_advance=stats(a),
        target_receipt=stats(receipt_times),reset=stats(reset_times),runs=runs,
        output_digest=output_digest.hexdigest(),profile_sha256=sha(PROFILE) if kind!='cp' else None,
        initialization='independent, initial_xy=(0,0), no supplied human history',
        target_schedule=dict(units='common angular counts, x right / y up',sample_ms=1,commit_ms=32,
            receipts=[dict(at_ms=t,target=list(xy)) for t,xy in receipts.items()]),
        renderer_receipt=None if kind=='cp' else vars(obj.renderer_model.receipt))

def warm_static(repeats):
    import numpy as np
    from abcurves import StaticPipeline
    import torch
    torch.set_num_interop_threads(1)
    fixture_info=json.loads((FIXTURES/'benchmark_static.json').read_text(encoding='utf8'))
    fixtures=[{k: v for k,v in np.load(FIXTURES/d['fixture'], allow_pickle=False).items()} for d in fixture_info]
    results={}
    for seed in (7,23):
        p=StaticPipeline(model_seed=seed,prewarm=True,torch_threads=1)
        profile_times=[]; profiles=[]
        for f in fixtures:
            for _ in range(40):
                t=time.perf_counter_ns(); profile=p.prepare_renderer_profile(f['profile_raw_dxdy']); profile_times.append(time.perf_counter_ns()-t)
            profiles.append(profile)
        records=[]; step_times=[]; output_hash=hashlib.sha256()
        for rep in range(-1,repeats):
            for case,(f,profile) in enumerate(zip(fixtures,profiles)):
                b=int(f['b_index']); prefix=f['raw_dxdy'][:b]
                common=dict(target_rel_at_B=f['target_rel_b'],target_radius=float(f['target_radius']),
                            progress_center=float(f['progress_center']),b_index_ms=b)
                for head in range(16):
                    runtime_seed=47000+max(rep,0)*100+case*16+head
                    t=time.perf_counter_ns()
                    planned=p.planner.plan(prefix,seed=runtime_seed,head=head,**common)
                    plan_ns=time.perf_counter_ns()-t
                    t=time.perf_counter_ns()
                    stream=p.prepare(prefix,renderer_profile=profile,planner_seed=runtime_seed,
                        planner_head=head,renderer_event_seed_u64=runtime_seed,**common)
                    handoff_done=time.perf_counter_ns()
                    output=stream.render_remaining()
                    render_done=time.perf_counter_ns()
                    assert len(output)==stream.duration_ms and len(output)==planned.intent.duration_ms
                    # A second independent event measures ordinary one-tick Python output.
                    step_stream=p.prepare(prefix,renderer_profile=profile,planner_seed=runtime_seed,
                        planner_head=head,renderer_event_seed_u64=runtime_seed,**common)
                    ticks=[]; times=[]
                    while not step_stream.complete:
                        ts=time.perf_counter_ns(); tick=step_stream.step(); spent=time.perf_counter_ns()-ts
                        ticks.append(tick)
                        if rep>=0: times.append(spent)
                    assert np.array_equal(output,np.asarray(ticks))
                    if rep>=0:
                        output_hash.update(output.tobytes()); step_times.extend(times)
                        records.append(dict(rep=rep,case=case,head=head,runtime_seed=runtime_seed,
                            duration_ms=stream.duration_ms,plan_ns=plan_ns,handoff_ns=handoff_done-t,
                            render_all_ns=render_done-handoff_done,combined_ns=render_done-t,
                            encode_ns=planned.timing.encode_us*1000,forward_ns=planned.timing.forward_us*1000,
                            decode_ns=planned.timing.decode_us*1000))
        fields=('plan_ns','handoff_ns','render_all_ns','combined_ns','encode_ns','forward_ns','decode_ns')
        results[str(seed)]=dict(model_selection_seed=seed,repetitions=repeats,events=len(records),
            profiles_prepare=stats(profile_times), per_tick_render=stats(step_times),
            measurements={k:stats([r[k] for r in records]) for k in fields},
            duration_ms=dict(min=min(r['duration_ms'] for r in records),median=float(np.median([r['duration_ms'] for r in records])),
                             max=max(r['duration_ms'] for r in records)),
            per_case=[dict(case=i,source_trial_id=fixture_info[i]['source_trial_id'],
                measurements={k:stats([r[k] for r in records if r['case']==i]) for k in fields}) for i in range(len(fixtures))],
            per_repetition=[dict(rep=i,measurements={k:stats([r[k] for r in records if r['rep']==i]) for k in fields}) for i in range(repeats)],
            renderer_receipt=p.renderer_receipt,output_sha256=output_hash.hexdigest())
        np.savez_compressed(OUTPUT/f'static_{seed}_samples.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]},
                            profile_ns=np.asarray(profile_times),step_ns=np.asarray(step_times))
        p.close()
    return dict(kind='static',models=results,fixtures=fixture_info,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())

def warm_static_sampled():
    """Ordinary runtime head sampling, separate from the controlled head sweep."""
    import numpy as np
    import torch
    from abcurves import StaticPipeline
    torch.set_num_interop_threads(1)
    info=json.loads((FIXTURES/'benchmark_static.json').read_text(encoding='utf8'))
    fixtures=[{k:v for k,v in np.load(FIXTURES/r['fixture'], allow_pickle=False).items()} for r in info]
    results={}
    for model_seed in (7,23):
        p=StaticPipeline(model_seed=model_seed,prewarm=True,torch_threads=1)
        profiles=[p.prepare_renderer_profile(f['profile_raw_dxdy']) for f in fixtures]
        records=[]
        for rep in range(-1,5):
            for case,(f,profile) in enumerate(zip(fixtures,profiles)):
                b=int(f['b_index']); prefix=f['raw_dxdy'][:b]
                args=dict(target_rel_at_B=f['target_rel_b'],target_radius=float(f['target_radius']),
                    progress_center=float(f['progress_center']),b_index_ms=b)
                for draw in range(16):
                    runtime_seed=47000+max(rep,0)*100+case*16+draw
                    t=time.perf_counter_ns(); planned=p.planner.plan(prefix,seed=runtime_seed,**args)
                    planned_ns=time.perf_counter_ns()-t
                    t=time.perf_counter_ns()
                    stream=p.prepare(prefix,renderer_profile=profile,planner_seed=runtime_seed,
                        renderer_event_seed_u64=runtime_seed,**args)
                    handoff=time.perf_counter_ns()
                    out=stream.render_remaining(); rendered=time.perf_counter_ns()
                    assert stream.planned.intent.head==planned.intent.head
                    assert len(out)==planned.intent.duration_ms
                    if rep>=0:
                        records.append(dict(rep=rep,case=case,runtime_seed=runtime_seed,head=planned.intent.head,
                            duration_ms=len(out),plan_ns=planned_ns,handoff_ns=handoff-t,
                            render_all_ns=rendered-handoff,combined_ns=rendered-t))
        fields=('plan_ns','handoff_ns','render_all_ns','combined_ns')
        results[str(model_seed)]=dict(events=len(records),model_selection_seed=model_seed,
            runtime_sampling_seed='47000 + repetition*100 + case*16 + draw; model_seed kept separate',
            measurements={k:stats([r[k] for r in records]) for k in fields},
            per_repetition=[dict(rep=i,measurements={k:stats([r[k] for r in records if r['rep']==i]) for k in fields}) for i in range(5)],
            sampled_head_counts={str(h):sum(r['head']==h for r in records) for h in range(16)},
            duration_ms=dict(min=min(r['duration_ms'] for r in records),median=float(np.median([r['duration_ms'] for r in records])),
                             max=max(r['duration_ms'] for r in records)))
        np.savez_compressed(OUTPUT/f'static_{model_seed}_sampled_head_samples.npz',**{k:np.array([r[k] for r in records]) for k in records[0]})
        p.close()
    return dict(schema='abcurves.static_public_defaults_performance.v1', models=results,
        script_sha256=sha(__file__), torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads())

def render_summary(report):
    """Summarize this run without introducing machine-independent speed claims."""
    lines=['# Runtime measurements', '', report['scope'], '',
        'Source bindings remained unchanged: '+str(report['all_sources_unchanged'])+'.',
        'All workers used the same source bindings: '+str(report['sources_consistent_between_workers'])+'.', '',
        '## Warm calls', '', '| Operation | n | Median us | p95 us | Maximum us |',
        '|---|---:|---:|---:|---:|']
    def row(label, item, scale=1):
        return '| '+label+' | '+str(item['n'])+' | '+ ' | '.join(
            format(item[key]/scale,'.3f') for key in ('median_us','p95_us','max_us'))+' |'
    for item in report['warmed']:
        if item['kind']=='static':
            for seed, model in item['models'].items():
                for key, value in model['measurements'].items(): lines.append(row('Static '+seed+' controlled: '+key,value))
                lines.append(row('Static '+seed+' profile',model['profiles_prepare']))
                lines.append(row('Static '+seed+' one tick',model['per_tick_render']))
        else:
            for key in ('fresh','fresh_with_motor','fresh_without_motor','buffered','every_1ms_advance','target_receipt','reset'):
                lines.append(row(item['kind']+': '+key,item[key]))
    for seed, model in report['sampled_static']['models'].items():
        for key, value in model['measurements'].items(): lines.append(row('Static '+seed+' sampled: '+key,value))
    lines+=['', '## Fresh-process startup with populated Numba cache', '',
        '| Stage | n | Median ms | p95 ms | Maximum ms |', '|---|---:|---:|---:|---:|']
    for item in report['startup']:
        label=item['kind']+(' '+str(item['model_seed']) if item['model_seed'] is not None else '')
        for key, value in item['warm_cache_fresh_process'].items():
            if key!='parent_process_wall_ns': lines.append(row(label+': '+key,value,1000))
    lines+=['', 'Empty-cache observations, source hashes, phase counters, numerical schedules, thread settings and per-repetition summaries are in report.json.',
        'One empty-cache run per kind/seed is not a percentile estimate. Provenance hashing warmed artifact file caches before startup timing.',
        'Parent process totals include instrumentation. These synchronous measurements do not establish device delivery or operating-system scheduling guarantees.']
    (OUTPUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def worker(args):
    sys.path.insert(0,str(RELEASE))
    before=bindings()
    import psutil
    idle=psutil.cpu_percent(interval=.5)
    if args.worker=='startup': result=startup_worker(args.kind,args.model_seed)
    elif args.worker=='sampled-static': result=warm_static_sampled()
    elif args.kind=='static': result=warm_static(args.repeats)
    else: result=warm_continuous(args.kind,args.repeats)
    after=bindings()
    result.update(cpu_busy_percent_before=idle,source_before=before,source_after=after,
                  code_unchanged=(before==after),metadata=metadata())
    write(args.result,result)
    print(json.dumps(dict(output=str(args.result),kind=args.kind,unchanged=(before==after))))

def launch(worker_type,kind,output,cache,**options):
    env=os.environ.copy()
    for k in THREAD_ENV: env[k]='1'
    env['NUMBA_CACHE_DIR']=str(cache)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',worker_type,'--kind',kind,'--output',str(OUTPUT),'--result',str(output),
        '--release-root',str(RELEASE),'--fixtures',str(FIXTURES)]
    for k,v in options.items(): command.extend(['--'+k.replace('_','-'),str(v)])
    t=time.perf_counter_ns()
    result=subprocess.run(command,env=env,text=True,capture_output=True,check=False)
    total=time.perf_counter_ns()-t
    if result.returncode:
        raise RuntimeError(' '.join(command)+'\n'+result.stdout+'\n'+result.stderr)
    print(result.stdout.strip(),flush=True)
    value=json.loads(output.read_text())
    value['parent_process_wall_ns']=total
    value['worker_options']=dict(worker=worker_type,kind=kind,**options)
    if result.stderr: value['stderr']=result.stderr
    write(output,value)
    return value

def main(args):
    stamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    records=[]
    kinds=('cp','composed','static')
    for kind in kinds:
        seeds=(7,23) if kind=='static' else (7,)
        for seed in seeds:
            # One clean disk-cache run documents initial compilation separately.
            cache=OUTPUT/'numba-cache'/f'{kind}-{seed}-{time.time_ns()}'
            for rep in range(6):
                result=launch('startup',kind,OUTPUT/f'startup_{kind}_{seed}_{rep}.json',cache,model_seed=seed)
                result['cache_status']='empty' if rep==0 else 'warm_disk_cache_fresh_process'
                result['repeat']=rep
                records.append(result)
    startup=[]
    for kind in kinds:
        for seed in ((7,23) if kind=='static' else (None,)):
            group=[r for r in records if r['kind']==kind and r['model_seed']==seed]
            fields=[k for k in group[0] if k.endswith('_ns')]
            startup.append(dict(kind=kind,model_seed=seed,cold_first={k:group[0][k]/1e6 for k in fields},
                warm_cache_fresh_process={k:stats([r[k] for r in group[1:]]) for k in fields}))
    warmed=[]
    warm_cache=OUTPUT/'numba-cache'/'warm-runtime'
    for kind in kinds:
        warmed.append(launch('warm',kind,OUTPUT/f'warm_{kind}.json',warm_cache,repeats=5 if kind=='static' else 7))
    report=dict(schema='abcurves.combined_runtime_performance.v1',started_at=stamp,
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),benchmark_sha256=sha(__file__),
        machine=platform.platform()+'; '+platform.processor()+'; CPU inference only',
        scope='combined current release; local synchronous calls, no device output, no scheduler deadlines, no summed speedups',
        cold_note='one first-compilation observation per kind/seed, not a cold-start percentile estimate; warm-start n=5',
        timing_note='ns sample clock; aggregate median/p95/max in microseconds; startup cold_first fields in milliseconds',
        thread_policy={k:'1' for k in THREAD_ENV},startup=startup,warmed=warmed,
        all_sources_unchanged=all(r['code_unchanged'] for r in records+warmed),
        sources_consistent_between_workers=all(r['source_before']==records[0]['source_before'] for r in records+warmed))
    sampled=launch('sampled-static','static',OUTPUT/'static_public_defaults.json',warm_cache)
    report['sampled_static']=sampled
    report['all_sources_unchanged'] &= sampled['code_unchanged']
    report['sources_consistent_between_workers'] &= sampled['source_before']==records[0]['source_before']
    report['fixture_bindings']=fixture_bindings()
    report['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write(OUTPUT/'report.json',report)
    render_summary(report)
    print(json.dumps({'report':str(OUTPUT/'report.json'),'sources_consistent':report['sources_consistent_between_workers']}),flush=True)

def fixture_bindings():
    info=json.loads((FIXTURES/'benchmark_static.json').read_text(encoding='utf8'))
    if len(info)!=3 or {r['fixture'] for r in info} != {'static_event.npz', 'static_event_short.npz', 'static_event_long.npz'}:
        raise ValueError('The measurement protocol requires the three authenticated static fixtures')
    records=info+[json.loads((FIXTURES/'human_start.json').read_text(encoding='utf8')) | {'fixture':'human_start.npz'}]
    result={}
    for record in records:
        path=(FIXTURES/record['fixture']).resolve()
        if path.parent!=FIXTURES or path.suffix!='.npz':
            raise ValueError('Fixture must name one NPZ inside the fixture directory')
        digest=sha(path)
        if digest!=record['fixture_sha256']:
            raise ValueError('Fixture SHA-256 mismatch: '+path.name)
        result[path.name]=digest
    return result

def configure(args):
    global RELEASE, OUTPUT, FIXTURES, PROFILE
    root=args.release_root
    if root is None:
        candidate=Path(__file__).resolve().parents[1]
        if (candidate/'abcurves/__init__.py').is_file():
            root=candidate
        else:
            spec=importlib.util.find_spec('abcurves')
            root=Path(spec.origin).resolve().parents[1] if spec and spec.origin else candidate
    RELEASE=root.resolve()
    if not (RELEASE/'abcurves/__init__.py').is_file() or not (RELEASE/'models/manifest.json').is_file():
        raise ValueError('Use a source checkout with models; supply --release-root if needed')
    OUTPUT=args.output.expanduser().resolve()
    if not args.worker and not args.check and OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError('Use a fresh benchmark output directory; old reports must not mask an incomplete run')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    FIXTURES=(args.fixtures or RELEASE/'examples/data').resolve()
    PROFILE=FIXTURES/'human_start.npz'
    fixture_bindings()
    sys.path.insert(0,str(RELEASE))
    if args.worker and (args.result is None or args.result.resolve().parent!=OUTPUT):
        raise ValueError('Internal worker result must be directly beneath output')

def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Directory for samples, JSON and compilation caches')
    parser.add_argument('--release-root',type=Path,help='Source checkout; derived from this script or the importable package by default')
    parser.add_argument('--fixtures',type=Path,help='Recorded fixtures; default: <release-root>/examples/data')
    parser.add_argument('--check',action='store_true',help='Validate inputs and write their hashes without timing inference')
    parser.add_argument('--worker',choices=['startup','warm','sampled-static'],help=argparse.SUPPRESS)
    parser.add_argument('--kind',choices=['cp','composed','static'],help=argparse.SUPPRESS)
    parser.add_argument('--model-seed',type=int,default=7,help=argparse.SUPPRESS)
    parser.add_argument('--repeats',type=int,default=7,help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    configure(args)
    if args.check:
        write(OUTPUT/'inputs.json',dict(fixtures=fixture_bindings(),runtime=bindings(),benchmark_sha256=sha(__file__)))
        print('Verified fixture and source bindings; no inference was timed.')
    elif args.worker:
        worker(args)
    else:
        main(args)

if __name__=='__main__':
    cli()
