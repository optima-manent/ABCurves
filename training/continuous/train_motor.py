"""Train the selected motor chain, with separate measured precursor objectives.

Run as python -m training.continuous.train_motor. --steps selects a shorter test run.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from abcurves.continuous_model import Config, MotorBase, sample_heads
from .base_data import UnifiedData
from .coherence_data import Data, functional
from . import common, losses, terminal_loss, coherence_loss, settling
from .rollout import rollout

STAGES = {
    'teacher': (20000, 2e-4, 1000),
    'feedback': (3000, 5e-5, 250),
    'settling': (3500, 2e-5, 250),
    'prior': (2500, 5e-5, 100),
    'motor': (600, 3e-5, 100),
}


def precursor(net, data, step, seed, phase, budget):
    device = next(net.parameters()).device
    data_step = step if phase == 'teacher' else 20000 + step
    batch = data.teacher(data_step, seed, 'base', rotation=True, augmentation=True)
    value = {k:torch.as_tensor(v, device=device) for k,v in batch.items() if isinstance(v,np.ndarray)}
    prediction, _ = net(value['coarse'].float(), value['fine'].float(), value['incoming'].double())
    epsilon = .05 if phase == 'feedback' else .2-.15*min(1., step/(budget/2))
    objective, detail = losses.teacher(prediction, value['delta'][:,:128].double(),
                                      value['length'].clamp_max(128).long(), epsilon, .25)
    objective.backward()
    record = dict(teacher=float(objective.detach()), epsilon=epsilon,
                  winner_counts=torch.bincount(detail['winners'], minlength=16).tolist())
    if phase == 'feedback':
        entries = data.sequences(step, seed, 'base', rotation=True, augmentation=True)
        record['sequences'] = []
        for ordinal, entry in enumerate(entries):
            pack = entry['pack']
            prediction, _ = rollout(net, pack, seed+step*1009+ordinal*53)
            objective, parts = terminal_loss.feedback(
                prediction, torch.as_tensor(pack['human_delta'], device=device).double(),
                torch.as_tensor(pack['position'], device=device).double(),
                torch.as_tensor(entry['control_target'], device=device).double(),
                domain=entry['domain'], mode='static_terminal',
                observed_remaining=entry['ref'].get('observed_remaining'),
                control_weight=4., acceleration_weight=.25,
                control_valid=torch.as_tensor(entry['control_valid'], device=device).bool())
            (objective/len(entries)).backward()
            record['sequences'].append(dict(domain=entry['domain'], loss=float(objective.detach())))
    return record


def clip_precursor(net):
    # Preserve the original double-precision global norm before float32 scaling.
    parameters = list(net.parameters())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError('Missing or nonfinite motor gradient')
    norm = torch.stack([p.grad.double().square().sum() for p in parameters]).sum().sqrt()
    scale = torch.where(norm > 5, 5/norm, torch.ones_like(norm))
    for p in parameters:
        p.grad.mul_(scale.to(p.grad.dtype))
    return float(norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--resume', type=Path, help='Saved optimizer/RNG state for this same stage')
    parser.add_argument('--steps', type=int, help='Explicit shortened smoke budget')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    budget, lr, interval = STAGES[args.stage]
    total = budget if args.steps is None else args.steps
    if not 1 <= total <= budget:
        parser.error('--steps must be positive and no greater than the frozen stage budget')
    if args.stage != 'teacher' and args.initial is None:
        parser.error('This stage requires its preceding --initial checkpoint')
    common.configure(7)
    args.output.mkdir(parents=True, exist_ok=False)
    newer = args.stage in ('prior','motor')
    data = Data(args.data) if newer else UnifiedData(args.data)
    if args.stage == 'teacher':
        net = MotorBase(Config(commit=64)).to(args.device)
    else:
        net, _ = common.motor(args.initial, coherent=newer, device=args.device, commit=32)
    prior = None
    if args.stage == 'motor':
        prior, _ = common.motor(args.initial, coherent=True, device=args.device)
        prior.eval().requires_grad_(False)
    if args.stage == 'settling':
        # The selected S1 precursor shares every tensor with the base motor.
        net.arm, net.gated = 'S1', False
        def draw(forecast, generator):
            head = sample_heads(generator, len(forecast), forecast.device)
            return forecast[torch.arange(len(forecast), device=forecast.device), head], {'head':head}, None
        net.draw = draw
        optimizer = settling.make_optimizer(net)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        resume_model = args.resume.parent / saved['checkpoint']
        if (saved['stage'] != args.stage or saved['checkpoint_sha256'] != common.sha(resume_model)
                or saved['initial_sha256'] != (common.sha(args.initial) if args.initial else None)
                or saved['data_receipt_sha256'] != common.sha(args.data/'receipt.json')):
            raise ValueError('Resume state does not bind this stage, source data and original donor')
        restored = torch.load(resume_model, map_location='cpu', weights_only=True)
        net.load_state_dict(restored['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'])
        start = saved['step']
    if start >= total:
        raise ValueError('Resume endpoint is not before requested budget')
    recipe = dict(stage=args.stage, seed=7, steps=total, selected_budget=budget,
                  scope='full_stage' if total==budget else 'smoke',
                  initial_sha256=common.sha(args.initial) if args.initial else None,
                  data_receipt_sha256=common.sha(args.data/'receipt.json'))
    common.write(args.output/'recipe.json', recipe)
    began = time.perf_counter()
    net.train()
    with (args.output/'steps.jsonl').open('x') as log:
        for step in range(start+1, total+1):
            optimizer.zero_grad(set_to_none=True)
            if args.stage == 'settling':
                record = settling.update(net, optimizer, data, step, 7)
            elif newer:
                data_step = step + (20000 if args.stage=='motor' else 0)
                batch = data.teacher(data_step,7)
                rng = np.random.default_rng(np.random.SeedSequence([7,127,data_step]))
                ix = rng.choice(len(batch['coarse']),160,replace=False)
                objective, record = coherence_loss.teacher(net,{k:v[ix] for k,v in batch.items()})
                quiet = coherence_loss.synthetic_quiet(net,data_step,7)
                (objective+quiet).backward()
                record['quiet'] = float(quiet.detach())
                if prior is not None:
                    entries = data.sequences(data_step,7,horizon=512)
                    entries.append(functional(data_step,7,horizon=1024))
                    record['sequences'] = []
                    for ordinal, entry in enumerate(entries):
                        value, detail = coherence_loss.feedback(net,prior,entry,7+1009*data_step+53*ordinal)
                        if not bool(torch.isfinite(value)):
                            raise FloatingPointError('Nonfinite generated-history objective')
                        (value/len(entries)).backward()
                        record['sequences'].append(detail)
                record['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True))
                optimizer.step()
            else:
                record = precursor(net,data,step,7,args.stage,budget)
                record['gradient_norm'] = clip_precursor(net)
                optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
                raise FloatingPointError('Nonfinite updated motor')
            record.update(step=step, elapsed_seconds=time.perf_counter()-began)
            import json
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if step==start+1 or step%25==0 or step==total:
                print(f'{args.stage} {step}/{total}: {record["teacher"]:.6g}', flush=True)
            if step%interval==0 or step==total:
                checkpoint=args.output/f'step_{step:05d}.pt'
                common.save(checkpoint,net,'motor',dict(recipe=recipe,step=step))
                torch.save(dict(stage=args.stage,step=step,checkpoint=checkpoint.name,
                                checkpoint_sha256=common.sha(checkpoint),initial_sha256=recipe['initial_sha256'],
                                data_receipt_sha256=recipe['data_receipt_sha256'],
                                optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state() if str(args.device).startswith('cuda') else None),
                           args.output/f'resume_{step:05d}.pt')
    common.write(args.output/'completion.json',dict(recipe=recipe,step=total,
                 checkpoint=checkpoint.name,checkpoint_sha256=common.sha(checkpoint),
                 elapsed_seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
