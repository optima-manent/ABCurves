"""Public runtime regression tests using bundled models and one raw-derived fixture."""
import numpy as np
import pytest
from abcurves import ContinuousPlanner, CountTransform, prepare_history, BTrigger, BFire, BReject
from abcurves.continuous import COMMON_RADIANS_PER_COUNT
from abcurves.continuous_pipeline import ContinuousPipelineFailure


def join(chunks,key):
    return np.concatenate([chunk[key] for chunk in chunks],axis=0)


def command(stream):
    stream.update_target((80.,-35.),0)
    stream.update_target((-30.,65.),15500)
    stream.update_target((110.,20.),66001)


def test_human_history_matches_authenticated_training_representation(human_start):
    h=human_start
    start=prepare_history(h['raw_common'],current_xy=h['observed_xy'])
    np.testing.assert_allclose(start.history,h['filtered_history'],rtol=0,atol=2e-13)
    np.testing.assert_allclose(start.initial_xy,h['filtered_xy'],rtol=0,atol=2e-12)
    np.testing.assert_array_equal(start.observed_xy,h['observed_xy'])
    assert not np.array_equal(start.observed_xy,start.initial_xy)
    raw=h['raw_common'].copy();current=h['observed_xy'].copy()
    copied=prepare_history(raw,current_xy=current)
    raw[:]=999;current[:]=999
    np.testing.assert_allclose(copied.history,start.history,rtol=0,atol=0)
    np.testing.assert_array_equal(copied.observed_xy,start.observed_xy)


def test_history_filter_preserves_motion_at_large_origins(human_start):
    raw=human_start['raw_common']
    near=prepare_history(raw,current_xy=(0.,0.))
    far=prepare_history(raw,current_xy=(1e14,-1e14))
    np.testing.assert_array_equal(far.history,near.history)


@pytest.mark.parametrize('raw,current',[(np.zeros((160,2)),(0,0)),(np.zeros((165,2)),(0,0)),
    (np.full((164,2),np.nan),(0,0)),(np.zeros((164,2)),(np.inf,0)),
    (np.full((164,2),1e308),(0,0))])
def test_history_rejects_missing_support_or_nonfinite_values(raw,current):
    with pytest.raises(ValueError):prepare_history(raw,current_xy=current)


def test_count_transform_roundtrip_anisotropic_axes_and_ownership():
    scales=np.array([2.,3.])*COMMON_RADIANS_PER_COUNT
    transform=CountTransform(scales,y_down=True);scales[:]=999
    raw=np.array([[2.,-4.],[-3.,5.]])
    common=transform.to_common(raw)
    np.testing.assert_allclose(common,[[4,12],[-6,-15]],rtol=0,atol=1e-14)
    np.testing.assert_allclose(transform.to_native(common),raw,rtol=0,atol=1e-14)
    up=CountTransform((2*COMMON_RADIANS_PER_COUNT,3*COMMON_RADIANS_PER_COUNT),y_down=False)
    np.testing.assert_allclose(up.to_common(raw),[[4,-12],[-6,15]],rtol=0,atol=1e-14)


@pytest.mark.parametrize('scale',[0,-1,np.nan,np.inf,(),(1,2,3),(1,0),1e308])
def test_count_transform_rejects_invalid_scales(scale):
    with pytest.raises(ValueError):CountTransform(scale)


def test_public_planner_receipts_are_causal_and_do_not_revise_commitment(planner,monkeypatch):
    calls=[];original=planner.planner.plan
    def record(history,position,target,available,valid,motion_known,now_us):
        calls.append(dict(time=now_us,target=target.copy(),available=available.copy(),valid=valid.copy()))
        return original(history,position,target,available,valid,motion_known,now_us)
    monkeypatch.setattr(planner.planner,'plan',record)
    planner.update_target((80.,-35.),0)
    first=planner.advance(5000)
    planner.update_target((-30.,65.),15500)
    rest=planner.advance(32000)
    reference=ContinuousPlanner(seed=71,prewarm=False)
    reference.update_target((80.,-35.),0)
    np.testing.assert_array_equal(join([first,rest],'xy'),reference.advance(32000)['xy'])
    assert len(calls)==1 and calls[0]['time']==0
    np.testing.assert_array_equal(calls[0]['target'][calls[0]['valid']],[[80.,-35.]])
    planner.advance(33000)
    assert calls[1]['time']==32000
    np.testing.assert_array_equal(calls[1]['target'][-1],[-30.,65.])
    assert calls[1]['available'][-1]==15500
    assert np.all(calls[1]['available'][calls[1]['valid']]<=32000)


def test_future_first_receipt_preserves_position_and_policy_rng(planner):
    planner.reset(initial_xy=(3.,-4.))
    planner.update_target((80.,-35.),15500)
    quiet=planner.advance(16000)
    np.testing.assert_array_equal(quiet['xy'],np.tile((3.,-4.),(16,1)))
    assert planner.decisions==0 and planner.needs_plan
    planner.advance(17000)
    assert planner.decisions==1


def test_chunking_reset_and_independent_streams_use_real_models(planner):
    command(planner)
    chunks=[planner.advance(t) for t in (500,1500,32000,32100,97000,160000)]
    assert chunks[0]['xy'].shape==(0,2)
    assert chunks[1]['time_us'].tolist()==[1000]
    reference=ContinuousPlanner(seed=71,prewarm=False);command(reference)
    whole=reference.advance(160000)
    for key in ('time_us','xy'):np.testing.assert_array_equal(join(chunks,key),whole[key])
    planner.reset();command(planner)
    replay=planner.advance(160000)
    for key in whole:np.testing.assert_array_equal(replay[key],whole[key])
    peer=ContinuousPlanner(seed=71,prewarm=False);command(peer)
    peer.advance(18000);planner.advance(210000)
    np.testing.assert_array_equal(peer.advance(160000)['xy'],whole['xy'][18:])


def test_public_planner_failed_validation_does_not_consume_output(planner):
    control=ContinuousPlanner(seed=71,prewarm=False)
    for stream in (planner,control):
        stream.update_target((100.,30.),0);stream.advance(1500)
    for value in (True,-1,1.5,np.iinfo(np.int64).max+1):
        with pytest.raises(ValueError):planner.advance(value)
    with pytest.raises(ValueError):planner.update_target((1,2),999)
    with pytest.raises(ValueError):planner.update_target((np.nan,2),2000)
    with pytest.raises(ValueError):planner.reset(seed=True)
    expected=control.advance(65000);actual=planner.advance(65000)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_pipeline_is_same_persistent_renderer_as_manual_composition(make_pipeline,human_start):
    pipeline=make_pipeline();command(pipeline)
    output=pipeline.advance(160000)
    renderer=pipeline.profile.begin_stream(event_seed=29)
    previous=np.zeros(2);reports=[]
    # Independent expression: hardware transform then canonicalize its Y axis.
    for point in output['xy']:
        hardware=pipeline.transform.to_native(point-previous)
        canonical=hardware*np.array([1.,-1.])
        emitted=renderer.step(canonical);emitted[1]*=-1
        reports.append(emitted);previous=point
    np.testing.assert_array_equal(output['reports'],reports)
    expected=np.cumsum(pipeline.transform.to_common(output['reports']),axis=0)
    np.testing.assert_allclose(output['rendered_xy'],expected,rtol=0,atol=2e-13)
    assert pipeline.renderer.ticks==160
    assert output['reports'].dtype==np.int16


def test_pipeline_axis_encoding_does_not_change_learned_texture(make_pipeline,human_start):
    down=make_pipeline()
    canonical=human_start['profile_hardware'].copy();canonical[:,1]*=-1
    up=make_pipeline(profile=canonical,transform=CountTransform(float(human_start['radians_per_count']),y_down=False))
    for stream in (up,down):command(stream)
    a=up.advance(128000);b=down.advance(128000)
    np.testing.assert_array_equal(a['xy'],b['xy'])
    np.testing.assert_array_equal(a['reports'][:,0],b['reports'][:,0])
    np.testing.assert_array_equal(a['reports'][:,1],-b['reports'][:,1])
    np.testing.assert_array_equal(a['rendered_xy'],b['rendered_xy'])


def test_assisted_pipeline_uses_filtered_model_anchor_and_observed_physical_anchor(make_pipeline,human_start):
    start=prepare_history(human_start['raw_common'],current_xy=human_start['observed_xy'])
    p=make_pipeline(initial_xy=start.initial_xy,history=start.history,observed_xy=start.observed_xy)
    np.testing.assert_array_equal(p.current_xy,start.initial_xy)
    np.testing.assert_allclose(p.rendered_xy,start.observed_xy,rtol=0,atol=1e-13)
    p.update_target(human_start['target_xy'],0);out=p.advance(96000)
    expected=start.observed_xy+np.cumsum(p.transform.to_common(out['reports']),axis=0)
    np.testing.assert_allclose(out['rendered_xy'],expected,rtol=0,atol=3e-12)
    p.reset();p.update_target(human_start['target_xy'],0);again=p.advance(96000)
    for key in out:np.testing.assert_array_equal(out[key],again[key])


def test_pipeline_chunking_and_reset_leave_peer_isolated(make_pipeline):
    a=make_pipeline();b=make_pipeline()
    for stream in (a,b):command(stream)
    whole=a.advance(128000)
    pieces=[b.advance(t) for t in (7000,32000,68000,128000)]
    for key in whole:np.testing.assert_array_equal(join(pieces,key),whole[key])
    a.reset();command(a)
    for key,value in a.advance(128000).items():np.testing.assert_array_equal(value,whole[key])
    before=b.rendered_xy.copy();copy=b.rendered_xy;copy[:]=999
    np.testing.assert_array_equal(b.rendered_xy,before)


def test_invalid_pipeline_reset_does_not_reset_either_state_machine(make_pipeline):
    a=make_pipeline();b=make_pipeline()
    for stream in (a,b):command(stream);stream.advance(17000)
    for kwargs in (dict(renderer_seed=-1),dict(initial_xy=(np.nan,0)),dict(history=np.zeros((159,2))),dict(observed_xy=(0,np.inf))):
        with pytest.raises((ValueError,RuntimeError)):a.reset(**kwargs)
    expected=b.advance(100000);actual=a.advance(100000)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_planner_failure_retains_complete_rendered_prefix_and_latches(make_pipeline,monkeypatch):
    p=make_pipeline();control=make_pipeline()
    for stream in (p,control):command(stream)
    original=p.movement.planner.plan;calls=0
    def fail_second(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise FloatingPointError('deliberate second-plan failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(p.movement.planner,'plan',fail_second)
    with pytest.raises(ContinuousPipelineFailure) as error:p.advance(80000)
    expected=control.advance(32000)
    for key in expected:np.testing.assert_array_equal(error.value.partial[key],expected[key])
    assert p.failed and p.movement.failed and p.movement.now_us==32000
    with pytest.raises(RuntimeError):p.advance(81000)
    with pytest.raises(RuntimeError):p.update_target((0,0),81000)
    monkeypatch.setattr(p.movement.planner,'plan',original)
    p.reset();command(p)
    for key,value in p.advance(32000).items():np.testing.assert_array_equal(value,expected[key])


def test_renderer_failure_returns_only_successfully_emitted_reports(make_pipeline):
    p=make_pipeline();control=make_pipeline()
    for stream in (p,control):command(stream)
    original=p.renderer
    class FailureAtSeventeen:
        def __init__(self):self.calls=0
        def step(self,delta):
            self.calls+=1
            if self.calls==17:raise RuntimeError('deliberate renderer failure')
            return original.step(delta)
    p.renderer=FailureAtSeventeen()
    with pytest.raises(ContinuousPipelineFailure) as error:p.advance(80000)
    expected=control.advance(16000)
    for key in expected:np.testing.assert_array_equal(error.value.partial[key],expected[key])
    assert p.failed and p.movement.now_us==80000
    np.testing.assert_array_equal(p.rendered_xy,expected['rendered_xy'][-1])
    with pytest.raises(RuntimeError):p.advance(81000)
    p.reset();command(p)
    for key,value in p.advance(16000).items():np.testing.assert_array_equal(value,expected[key])


def walk(trigger,distance,radius,steps):
    trigger.arm((distance,0),radius);position=0.
    for dx in steps:
        position+=dx
        result=trigger.push_tick(dx,0,target_rel_now=(distance-position,0),target_radius_now=radius)
        if result is not None:return result
    return None


def test_recommended_static_seam_is_edge_referenced_and_preserves_legacy_contract():
    recommended=BTrigger.recommended();legacy=BTrigger()
    event=walk(recommended,110,10,np.ones(100))
    assert isinstance(event,BFire) and event.t_ms==90
    assert event.progress_edge==pytest.approx(.9)
    assert event.progress_center==pytest.approx(90/110)
    assert event.remaining_counts==pytest.approx(20)
    old=walk(legacy,110,10,np.ones(100))
    assert isinstance(old,BFire) and old.t_ms==80
    assert not recommended.armed
    assert recommended.push_tick(1,0,target_rel_now=(19,0),target_radius_now=10) is None


@pytest.mark.parametrize('distance,radius,steps,reason',[
    (110,10,np.full(30,5.),'min_prefix_ms'),
    (60,10,np.ones(60),'min_edge_margin_counts'),
    (110,10,np.r_[np.zeros(24),105.],'max_center_progress'),
    (110,10,np.zeros(1501),'max_ab_ms'),
    (110,10,np.r_[np.full(30,2.),-25.],'progress_regression'),
])
def test_recommended_static_rejects_unsupported_first_crossings(distance,radius,steps,reason):
    trigger=BTrigger.recommended();result=walk(trigger,distance,radius,steps)
    assert isinstance(result,BReject) and result.reason==reason
    assert not trigger.armed
    assert trigger.push_tick(1,0,target_rel_now=(100,0),target_radius_now=10) is None
    trigger.arm((110,0),10)
    assert trigger.armed and np.array_equal(trigger.movement_counts,[0,0])


def test_static_trigger_invalid_input_does_not_consume_a_tick():
    trigger=BTrigger.recommended();trigger.arm((110,0),10)
    with pytest.raises(ValueError):trigger.push_tick(np.nan,0,target_rel_now=(110,0),target_radius_now=10)
    result=None
    for i in range(90):result=trigger.push_tick(1,0,target_rel_now=(109-i,0),target_radius_now=10)
    assert isinstance(result,BFire) and result.t_ms==90


def test_assisted_public_initialization_keeps_unknown_older_motion_and_target_history(human_start,monkeypatch):
    start=prepare_history(human_start['raw_common'],current_xy=human_start['observed_xy'])
    p=ContinuousPlanner(seed=71,prewarm=False,initial_xy=start.initial_xy,history=start.history)
    original=p.planner.plan;seen=[]
    def record(history,position,target,available,valid,motion_known,now_us):
        seen.append((history.copy(),position.copy(),target.copy(),valid.copy(),motion_known.copy()))
        return original(history,position,target,available,valid,motion_known,now_us)
    monkeypatch.setattr(p.planner,'plan',record)
    p.update_target(human_start['target_xy'],0);p.advance(1000)
    history,position,target,valid,known=seen[0]
    np.testing.assert_array_equal(history[-160:],start.history)
    np.testing.assert_array_equal(position,start.initial_xy)
    assert not known[:-160].any() and known[-160:].all()
    assert not valid[:-1].any() and valid[-1]
    np.testing.assert_array_equal(target[-1],human_start['target_xy'])


def test_renderer_state_survives_committed_move_hold_and_restart(make_pipeline,monkeypatch):
    p=make_pipeline();p.update_target((50.,40.),0)
    # Controlled intent isolates the composition contract from learned decisions.
    actions=[np.tile((.7,-.2),(32,1)),np.zeros((32,2)),np.tile((-.4,.6),(32,1))]
    def plan(*args):return actions[args[-1]//32000].copy()
    monkeypatch.setattr(p.movement.planner,'plan',plan)
    renderer=p.profile.begin_stream(event_seed=29)
    chunks=[p.advance(t) for t in (11000,32000,50000,64000,96000)]
    points=join(chunks,'xy');reports=join(chunks,'reports')
    previous=np.zeros(2);expected=[]
    for point in points:
        delta=p.transform.to_native(point-previous)*np.array([1.,-1.])
        report=renderer.step(delta);report[1]*=-1;expected.append(report);previous=point
    np.testing.assert_array_equal(reports,expected)
    np.testing.assert_array_equal(points[32:64],np.tile(points[31],(32,1)))
    assert not np.array_equal(points[-1],points[63])
    assert p.renderer.ticks==96 and p.movement.decisions==3


@pytest.mark.parametrize('profile',[np.zeros((255,2)),np.zeros((257,2)),np.full((256,2),.5),np.full((256,2),np.inf)])
def test_pipeline_requires_real_integer_profile_support(make_pipeline,profile):
    with pytest.raises((ValueError,RuntimeError)):make_pipeline(profile=profile)
