from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from abcurves import ContinuousPlanner, ContinuousPipeline, CountTransform

@pytest.fixture(scope='session')
def human_start():
    root=Path(__file__).parent/'fixtures'
    meta=json.loads((root/'human_start.json').read_text(encoding='utf8'))
    path=root/'human_start.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

@pytest.fixture
def planner():
    return ContinuousPlanner(seed=71,prewarm=False)

@pytest.fixture
def make_pipeline(human_start):
    def make(**options):
        defaults=dict(seed=71,renderer_seed=29,prewarm=False)
        defaults.update(options)
        transform=defaults.pop('transform',CountTransform(float(human_start['radians_per_count'])))
        profile=defaults.pop('profile',human_start['profile_hardware'])
        return ContinuousPipeline(profile,transform=transform,**defaults)
    return make
