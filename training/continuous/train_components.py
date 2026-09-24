"""Fit the hazard/brake precursor, selecting minimum held-out joint objective."""
import argparse, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from abcurves.continuous_model import Components
from .common import sha, write, save, configure
from .base_data import read
from .brake_losses import initial_loss as brake_loss

def save_component(path,net,metadata):
    save(path,net,'events',metadata)

def run(out,seed,steps,data,device="cuda",engineering=False):
    DATA=Path(data)
    configure(seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    receipt=read(DATA/'receipt.json')
    for p,d in receipt['files'].items():
        if sha(DATA/p)!=d:raise ValueError('Changed component input: '+p)
    bound={str(DATA/p):d for p,d in receipt['files'].items()};net=Components().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.0005)
    cache={}
    for name in ('brakes','decisions'):
        with np.load(DATA/(name+'.npz')) as z:cache[name]={k:torch.tensor(z[k],device=device) for k in z.files}
    br,de=cache['brakes'],cache['decisions']
    def weights(a,role,mode=None):
        w=a['weight'].double().clone();w[a['role']!=role]=0
        if 'risk' in a:w[~a['risk']]=0
        if mode is not None:w[a['mode']!=mode]=0
        return w/w.sum()
    bw=weights(br,0);dw=[weights(de,0,m) for m in (0,1)]
    rng=torch.Generator(device=device).manual_seed(seed+971)
    tests={}
    for name,a in cache.items():
        w=weights(a,1);tests[name]=torch.multinomial(w,min(8192,int((w>0).sum())),replacement=True,generator=torch.Generator(device=device).manual_seed(824))
    best=float('inf');beststep=0
    write(out/'launch.json',dict(seed=seed,steps=steps,pid=os.getpid(),started_epoch=start,engineering_only=engineering,bindings=bound))
    try:
        with (out/'steps.jsonl').open('x') as log:
            for step in range(1,steps+1):
                tick=time.time();net.train();opt.zero_grad(set_to_none=True)
                i=torch.multinomial(bw,512,True,generator=rng)
                bl,detail=brake_loss(net,br['x'][i],br['v'][i],br['coeff'][i],br['duration'][i])
                hl=0.
                for m in (0,1):
                    j=torch.multinomial(dw[m],2048,True,generator=rng)
                    logits=net.hazard(de['x'][j])[:,m]
                    hl=hl+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits,de['y'][j,m])
                value=bl+hl
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite component loss')
                value.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
                row=dict(step=step,loss=float(value.detach()),brake=float(bl.detach()),hazard=float(hl.detach()),gradient_norm=float(norm),seconds=time.time()-tick)
                if step==1 or step%250==0 or step==steps:
                    net.eval()
                    with torch.no_grad():
                        j=tests['brakes'];vb,_=brake_loss(net,br['x'][j],br['v'][j],br['coeff'][j],br['duration'][j])
                        vh=0.
                        for m in (0,1):
                            w=weights(de,1,m);j=torch.multinomial(w,8192,True,generator=torch.Generator(device=device).manual_seed(924+m))
                            vh=vh+.5*torch.nn.functional.binary_cross_entropy_with_logits(net.hazard(de['x'][j])[:,m],de['y'][j,m])
                        score=float(vb+vh);row['holdout']=dict(brake=float(vb),hazard=float(vh),score=score)
                        if score<best:
                            best=score;beststep=step
                            save_component(out/'components.pt',net,dict(seed=seed,step=step,holdout=row['holdout'],data_sha256=sha(DATA/'receipt.json'),engineering_only=engineering))
                    write(out/'progress.json',dict(**row,elapsed_seconds=time.time()-start));print(row,flush=True)
                log.write(__import__('json').dumps(row)+'\n');log.flush()
        for p,d in bound.items():assert sha(p)==d,p
        write(out/'completion.json',dict(status='COMPLETE',seed=seed,completed_steps=steps,best_step=beststep,holdout_score=best,
            checkpoint=str(out/'components.pt'),sha256=sha(out/'components.pt'),elapsed_seconds=time.time()-start,engineering_only=engineering))
    except BaseException:
        write(out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),elapsed_seconds=time.time()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if not 1<=a.steps<=6000:p.error('--steps must be 1..6000')
    run(a.output,7,a.steps,a.data,a.device,engineering=a.steps!=6000)
