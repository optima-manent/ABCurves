"""Fit the selected coherent choice model or C2 brake, with grouped holdout selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from abcurves.continuous_model import Components, Selector
from .brake_losses import brake_loss
from .common import configure, save, sha, write


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['selector','brake'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-events',type=Path)
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if not 1<=args.steps<=6000:
        parser.error('--steps must be 1..6000')
    if args.stage=='brake' and args.initial_events is None:
        parser.error('The C2 brake requires the selected precursor --initial-events')
    configure(23)
    args.output.mkdir(parents=True,exist_ok=False)
    choice=args.stage=='selector'
    filename='sequences.npz' if choice else 'brakes.npz'
    receipt=json.loads((args.data/'receipt.json').read_text())
    if sha(args.data/filename)!=receipt['files'][filename]:
        raise ValueError('Prepared input differs from its receipt')
    keys=('encoded','heads','displacement','emission','mask') if choice else ('x','v','acceleration','truth','duration')
    with np.load(args.data/filename,allow_pickle=False) as z:
        data={k:torch.as_tensor(z[k],device=args.device) for k in keys}
        roles=z['role'];weight=torch.as_tensor(z['weight'],device=args.device,dtype=torch.float64)
    if not bool(torch.isfinite(weight).all()) or bool((weight<0).any()):
        raise ValueError('Invalid source weights')
    fit=torch.as_tensor(np.flatnonzero(roles==0),device=args.device)
    val=torch.as_tensor(np.flatnonzero(roles==1),device=args.device)
    if len(fit)==0 or len(val)==0:
        raise ValueError('Both fit and grouped holdout sources are required')
    fit_weight=weight[fit];val_weight=weight[val];val_weight=val_weight/val_weight.sum()
    if choice:
        net=Selector(coherent=True).to(args.device)
        frozen=None
    else:
        payload=torch.load(args.initial_events,map_location='cpu',weights_only=True)
        if payload.get('schema')!='abcurves.continuous_component.v1' or payload['role']!='events':
            raise ValueError('Expected portable precursor event components')
        net=Components().to(args.device)
        net.load_state_dict(payload['state_dict'],strict=True)
        net.hazard.requires_grad_(False)
        frozen={k:v.detach().clone() for k,v in net.hazard.state_dict().items()}
    batch_size=128 if choice else 512
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                lr=.001 if choice else .0005,weight_decay=.0001 if choice else .0005)
    rng=torch.Generator(device=args.device).manual_seed(23^(0x4198 if choice else 0x3819))
    def objective(indices):
        batch={k:data[k][indices] for k in keys}
        return net.sequence_loss(**batch) if choice else brake_loss(net,**batch)[0]
    def validation():
        net.eval()
        with torch.no_grad():
            losses=[objective(val[i:i+(256 if choice else 512)]) for i in range(0,len(val),256 if choice else 512)]
            result=float((torch.cat(losses).double()*val_weight).sum())
        net.train()
        if not np.isfinite(result):
            raise FloatingPointError('Nonfinite grouped holdout objective')
        return result
    recipe=dict(stage=args.stage,seed=23,maximum_steps=6000,minimum_steps=1600,
                patience_steps=1000,validation_every=200,min_improvement=.0001,
                scope='selected_recipe' if args.steps==6000 else 'smoke',
                data_receipt_sha256=sha(args.data/'receipt.json'),
                initial_events_sha256=sha(args.initial_events) if args.initial_events else None)
    def checkpoint(step,score):
        if frozen is not None and not all(torch.equal(v,net.hazard.state_dict()[k]) for k,v in frozen.items()):
            raise ArithmeticError('Frozen event hazards changed during brake fitting')
        path=args.output/f'step_{step:05d}.pt'
        save(path,net,'selector' if choice else 'events',dict(recipe=recipe,step=step,holdout=score))
        return path
    began=time.perf_counter();best=validation();best_step=0;best_path=checkpoint(0,best)
    history=[dict(step=0,holdout=best)]
    write(args.output/'recipe.json',recipe)
    for step in range(1,args.steps+1):
        indices=fit[torch.multinomial(fit_weight,batch_size,replacement=True,generator=rng)]
        loss=objective(indices).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite selected-component objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        if step%200==0 or step==args.steps:
            score=validation();path=checkpoint(step,score)
            history.append(dict(step=step,holdout=score,training=float(loss.detach())))
            if score<best-.0001:
                best,best_step,best_path=score,step,path
            print(f'{args.stage} {step}: holdout={score:.7g}, selected={best_step}',flush=True)
            if step>=1600 and step-best_step>=1000:
                break
    write(args.output/'selection.json',dict(status='COMPLETE',recipe=recipe,
          completed_steps=step,best_step=best_step,best_loss=best,checkpoint=best_path.name,
          checkpoint_sha256=sha(best_path),history=history,elapsed_seconds=time.perf_counter()-began))


if __name__=='__main__':
    main()
