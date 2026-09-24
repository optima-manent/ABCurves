"""Small, explicit checkpoint and run receipts shared by training commands."""
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch
from abcurves.continuous_model import Config, Motor, MotorBase


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configure(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save(path, net, role, metadata):
    payload = dict(schema='abcurves.continuous_component.v1', role=role,
                   state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},
                   metadata=metadata)
    if hasattr(net, 'config'):
        payload['config'] = net.config.to_dict()
    temporary = Path(path).with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def motor(path, *, coherent=True, device='cpu', commit=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'abcurves.continuous_component.v1' or payload['role'] != 'motor':
        raise ValueError('Expected a portable Continuous Planner motor checkpoint')
    config = dict(payload['config'])
    if commit is not None:
        config['commit'] = commit
    net = (Motor if coherent else MotorBase)(Config(**config))
    net.load_state_dict(payload['state_dict'], strict=True)
    return net.to(device), payload
