"""Small path-independent storage helpers; arrays never require pickle."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def array_inventory(path):
    return {key: dict(dtype=value.dtype.str, shape=list(value.shape),
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        for key, value in load(path).items()}
