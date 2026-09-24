"""Source-balanced physical data for the selected Continuous Planner."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import hashlib, json, math
import numpy as np
VELOCITY_SCALE = 1.1982087601807259
POSITION_SCALE = 81.73350722609987
COMMON = .0003509487083280618
FIELDS = ('history','position','target','available','valid','motion_known','cut_us','human_delta')
DOMAINS = ('late','early','tracking')
CACHE_SEEDS = dict(late=2026091561,early=2026091562,tracking=2026091563)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def augment_pack(out, rng):
    """Exact inherited per-occurrence thinning then prefix-truncation law."""
    n = out['target'].shape[1]
    for k in range(len(out['history'])):
        if rng.random() < .5:
            period = int(rng.choice([8, 16, 33, 50, 66, 100])); phase = int(rng.integers(period))
            source = np.maximum(0, ((np.arange(n) + phase) // period) * period - phase)
            for key in ('target', 'available', 'valid'): out[key][k] = out[key][k, source]
        if rng.random() < .5:
            retain = int(rng.choice([16, 64, 160, 320])); unknown = 640 - retain
            out['history'][k, :unknown] = 0.; out['motion_known'][k, :unknown] = False
            out['target'][k, :unknown] = 0.; out['valid'][k, :unknown] = False
    return out

def physical_features(pack):
    """NumPy copy of temporal_core.features before its two asinh operations.

    Only target/receipt indices0..639 enter; future transport is never read.
    Computation is float64; cache quantization happens in the writer.
    """
    history = np.asarray(pack['history'], np.float64)
    position = np.asarray(pack['position'], np.float64)
    target = np.asarray(pack['target'][:, :640], np.float64)
    available = pack['available'][:, :640]; valid = pack['valid'][:, :640]
    mk = pack['motion_known']; b = len(history)
    endpoint = pack['cut_us'][:, None] + np.arange(-639, 1)[None] * 1000
    known = valid & (available <= endpoint)
    motion = np.where(mk[..., None], history, 0.)
    goal = np.where(known[..., None], target, 0.)
    old = np.maximum(0, np.arange(640) - 32)
    pair = known & known[:, old] & (np.arange(640)[None] >= 32)
    tv = np.where(pair[..., None], (goal - goal[:, old]) / 32, 0.)
    points = position[:, None] + motion.cumsum(1) - motion.sum(1)[:, None]
    relative = np.where((known & mk)[..., None], (goal - points) / POSITION_SCALE, 0.)
    index = np.r_[np.arange(15, 480, 16), np.arange(483, 640, 4)]
    mean_v = np.concatenate((motion[:, :480].reshape(b, 30, 16, 2).mean(2),
                             motion[:, 480:].reshape(b, 40, 4, 2).mean(2)), 1)
    coverage = np.concatenate((mk[:, :480].reshape(b, 30, 16).mean(2),
                               mk[:, 480:].reshape(b, 40, 4).mean(2)), 1)
    widths = np.broadcast_to(np.r_[np.ones(30), np.full(40, .25)][None, :, None], (b, 70, 1))
    coarse = np.concatenate((mean_v / VELOCITY_SCALE, relative[:, index], tv[:, index] / VELOCITY_SCALE,
                              known[:, index, None], coverage[..., None], widths), -1)
    current = np.where(known[:, -1, None], (goal[:, -1] - position) / POSITION_SCALE, 0.)
    fine = np.concatenate((motion[:, -16:].reshape(b, 32) / VELOCITY_SCALE, mk[:, -16:], current,
                            tv[:, -1] / VELOCITY_SCALE, known[:, -1, None], mk.mean(1)[:, None]), -1)
    return coarse, fine

def rotate_vectors(values, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(values) @ np.array([[c, s], [-s, c]])

def materialize_features(coarse, fine, angle=0.):
    coarse = np.array(coarse, np.float64, copy=True); fine = np.array(fine, np.float64, copy=True)
    if angle:
        for start in (0, 2, 4): coarse[..., start:start+2] = rotate_vectors(coarse[..., start:start+2], angle)
        fine[:, :32] = rotate_vectors(fine[:, :32].reshape(-1, 16, 2), angle).reshape(-1, 32)
        for start in (48, 50): fine[:, start:start+2] = rotate_vectors(fine[:, start:start+2], angle)
    coarse[..., 2:4] = np.arcsinh(coarse[..., 2:4]); fine[:, 48:50] = np.arcsinh(fine[:, 48:50])
    return coarse.astype(np.float32), fine.astype(np.float32)

def rotate_pack(pack, angle):
    if angle:
        for key in ('history', 'position', 'target', 'human_delta'):
            pack[key] = rotate_vectors(pack[key], angle)
    return pack

def hierarchical_weights(rows):
    keys = [(r['person'], r['family'], r['guide'], r['parent']) for r in rows]
    children = defaultdict(set); counts = Counter(keys)
    for key in keys:
        for d in range(len(key)): children[key[:d]].add(key[d])
    w = np.array([1. / (math.prod(len(children[key[:d]]) for d in range(len(key))) * counts[key]) for key in keys])
    return w / w.sum()

class StaticSource:
    """Mmap native session transport; full physical source history, causal masks."""
    def __init__(self, root):
        self.root = Path(root); self.parents = read(self.root / 'static_parents.json')
        self.sessions = read(self.root / 'static_sessions.json')
        self.maps = {d: np.load(self.root / (d + '_cuts.npy'), mmap_mode='r') for d in ('late', 'early')}
        self.handles = {}; self.sums = {}

    def native(self, key):
        if key not in self.handles:
            self.handles[key] = np.load(self.root / self.sessions[key]['cache_path'], mmap_mode='r', allow_pickle=False)
        return self.handles[key]

    def cumulative(self, key):
        # Integer native sums are exact and retained once per touched session.
        if key not in self.sums:
            native = self.native(key)
            self.sums[key] = np.concatenate((np.zeros((1, 2), np.int64), native.cumsum(0, dtype=np.int64)))
        return self.sums[key]

    def pack(self, domain, indices, max_h=256, *, rng=None, augment=False):
        indices = np.asarray(indices, np.int64).reshape(-1)
        if domain not in self.maps or not len(indices): raise ValueError('Nonempty static domain indices required')
        mapping = self.maps[domain][indices]; b = len(indices); n = 640 + max_h
        out = dict(history=np.empty((b, 640, 2)), position=np.empty((b, 2)),
                   target=np.zeros((b, n, 2)), available=np.empty((b, n), np.int64),
                   valid=np.zeros((b, n), bool), motion_known=np.empty((b, 640), bool),
                   cut_us=np.empty(b, np.int64), human_delta=np.zeros((b, max_h, 2)))
        length = np.empty(b, np.int64)
        groups = defaultdict(list)
        for k, (parent, cut) in enumerate(mapping): groups[self.parents[int(parent)]['session_key']].append(k)
        for key, slots_list in groups.items():
            slots = np.asarray(slots_list); native = self.native(key); cumulative = self.cumulative(key)
            parent_rows = [self.parents[int(mapping[k, 0])] for k in slots]
            cuts = mapping[slots, 1]; ends = np.array([r['end'] for r in parent_rows])
            # Read no post-resolution row, including for storage-only padded labels.
            native_indices = np.minimum(cuts[:, None] + np.arange(-644, max_h)[None], ends[:, None] - 1)
            assert native_indices.min() >= 0 and native_indices.max() < len(native)
            raw = native[native_indices].astype(np.float64)
            filtered = (raw[:, :-4] + 2*raw[:, 1:-3] + 3*raw[:, 2:-2] + 2*raw[:, 3:-1] + raw[:, 4:]) / 9
            factors = np.array([r['factor'] for r in parent_rows])
            filtered *= factors[:, None, None]
            start = np.array([r['presentation_index'] for r in parent_rows]); b80 = np.array([r['b80'] for r in parent_rows])
            count = np.full(len(slots), 640) if domain == 'late' else np.minimum(640, 160 + cuts - start)
            mk = np.arange(640)[None] >= 640 - count[:, None]
            out['history'][slots] = np.where(mk[..., None], filtered[:, :640], 0.)
            out['motion_known'][slots] = mk
            lag = -(8*raw[:, 643] + 6*raw[:, 642] + 3*raw[:, 641] + raw[:, 640]) / 9
            out['position'][slots] = (cumulative[cuts] - cumulative[b80] + lag) * factors[:, None]
            out['cut_us'][slots] = (cuts - start) * 1000
            available = np.array([r['presentation_available_us'] for r in parent_rows])
            clocks = out['cut_us'][slots, None] + np.arange(-639, max_h+1)[None] * 1000
            valid = (clocks >= available[:, None]) & (clocks <= (ends-start)[:, None]*1000)
            targets = np.array([r['target_B_common'] for r in parent_rows])
            out['target'][slots] = np.where(valid[..., None], targets[:, None], 0.)
            out['valid'][slots] = valid; out['available'][slots] = available[:, None]
            length[slots] = np.minimum(max_h, ends-cuts)
            observed = np.arange(max_h)[None] < length[slots, None]
            out['human_delta'][slots] = np.where(observed[..., None], filtered[:, 640:], 0.)
        if np.any(length <= 0): raise ValueError('No authentic future')
        if augment: augment_pack(out, rng)
        return out, length

class UnifiedData:
    """Stateless step/seed plans, shared across comparable architectures.

    step is one-based independently within teacher or feedback. Arrays returned
    by teacher are ordinary NumPy values, suitable for torch.as_tensor.
    """
    def __init__(self, root=None, verify=True):
        root = Path(root)
        self.root = root if (root / 'receipt.json').is_file() else root / 'data_v1'
        self.receipt = read(self.root / 'receipt.json')
        if self.receipt['status'] != 'COMPLETE_PHYSICAL_DATA': raise ValueError('Data preparation is incomplete')
        if verify:
            for path, digest in self.receipt['source_sha256'].items():
                if sha(self.root / path) != digest: raise ValueError('Changed data source: ' + path)
            for name, digest in self.receipt['metadata_sha256'].items():
                if sha(self.root / name) != digest: raise ValueError('Changed enrollment: ' + name)
            # Large immutable caches were content-hashed by preparation; loaders
            # check size/header, while the campaign binds their saved hashes.
        self.enrollment = np.load(self.root / 'enrollment.npz', allow_pickle=False)
        self.parents = read(self.root / 'static_parents.json')
        self.tracking_rows = read(self.root / 'tracking_rows.json')
        self.cache = {}
        for domain in DOMAINS:
            self.cache[domain] = {key: np.load(self.root / domain / (key+'.npy'), mmap_mode='r', allow_pickle=False)
                                  for key in ('coarse_linear', 'fine_linear', 'incoming', 'delta', 'length')}
            for key, a in self.cache[domain].items():
                spec = self.receipt['cache'][domain][key]
                if list(a.shape) != spec['shape'] or str(a.dtype) != spec['dtype'] or (self.root/domain/(key+'.npy')).stat().st_size != spec['file_bytes']:
                    raise ValueError('Cache header/size changed')
        self._static = None; self._tracking = None
        self.maps = {d: np.load(self.root / (d+'_cuts.npy'), mmap_mode='r') for d in ('late', 'early')}
        self._ranges = {}
        for d in ('late', 'early'):
            ids = self.maps[d][:, 0]; counts = np.bincount(ids, minlength=len(self.parents))
            self._ranges[d] = (np.r_[0, counts.cumsum()[:-1]], counts)

    def _variant(self, variant, step, sequence=False):
        if variant not in ('base',): raise ValueError('Unknown data variant: '+str(variant))
        return ('base' if sequence or step > 10000 else 'quality80') if variant == 'curriculum' else variant

    @lru_cache(maxsize=48)
    def _epoch(self, seed, domain, variant, epoch):
        allowed = self.enrollment['static_half' if variant == 'half' else 'static_old' if variant == 'static_old' else 'static_base']
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 11, DOMAINS.index(domain), int(epoch)]))
        return rng.permutation(allowed)

    @lru_cache(maxsize=16)
    def _initial_offsets(self, seed, domain):
        # Same full parent order across variants; independent of permutation,
        # augmentation and rotation streams. Each local cut can start a cycle.
        _, sizes = self._ranges[domain]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 12, DOMAINS.index(domain)]))
        return rng.integers(0, sizes, dtype=np.int64)

    def _static_indices(self, domain, step, seed, variant, count):
        allowed = self.enrollment['static_half' if variant == 'half' else 'static_old' if variant == 'static_old' else 'static_base']
        starts, sizes = self._ranges[domain]; initial = self._initial_offsets(seed, domain); result = []
        for occurrence in range((step-1)*count, step*count):
            epoch, within = divmod(occurrence, len(allowed))
            parent = int(self._epoch(seed, domain, variant, epoch)[within])
            # Uniform initial phase, then every local cut before any repeat.
            result.append(int(starts[parent] + (initial[parent] + epoch) % sizes[parent]))
        return np.asarray(result, np.int64)

    def _tracking_indices(self, count, rng, variant):
        key = variant if variant in ('half', 'quality80', 'no_person1', 'no_person2', 'no_person3') else 'base'
        return rng.choice(self.enrollment['tracking_'+key], size=count, replace=True,
                          p=self.enrollment['tracking_'+key+'_weights'])

    def teacher(self, step, seed, variant='base', *, rotation=True, augmentation=True):
        if step < 1: raise ValueError('One-based positive step required')
        variant = self._variant(variant, step)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 21, int(step)]))
        choices = [] if variant == 'tracking_only' else [(d, self._static_indices(d, step, seed, variant, 64)) for d in ('late', 'early')]
        choices.append(('tracking', self._tracking_indices(256 if variant == 'tracking_only' else 128, rng, variant)))
        # Rotation and view-choice streams are independent of selection and one another.
        rot_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 23, int(step)]))
        angle = float(rot_rng.uniform(-np.pi, np.pi)) if rotation and rot_rng.random() < .5 else 0.
        accum = defaultdict(list); identities = []
        for domain, indices in choices:
            cache = self.cache[domain]
            # Every prepared augmented view already applies both50% mechanisms;
            # it is the training view when augmentation=True, not another50% gate.
            views = np.full(len(indices), int(bool(augmentation)), np.int64)
            for key in ('coarse_linear', 'fine_linear'):
                accum[key].append(np.asarray(cache[key][views, indices]))
            for key in ('incoming', 'delta', 'length'): accum[key].append(np.asarray(cache[key][indices]))
            accum['domain'].append(np.full(len(indices), DOMAINS.index(domain), np.int64))
            identities.extend(dict(domain=domain, row=int(i), augmented=bool(v)) for i, v in zip(indices, views))
        coarse, fine = materialize_features(np.concatenate(accum['coarse_linear']), np.concatenate(accum['fine_linear']), angle)
        return dict(coarse=coarse, fine=fine,
                    incoming=rotate_vectors(np.concatenate(accum['incoming']), angle).astype(np.float32),
                    delta=rotate_vectors(np.concatenate(accum['delta']), angle).astype(np.float32),
                    length=np.concatenate(accum['length']).astype(np.int64), domain=np.concatenate(accum['domain']),
                    ids=identities, rotation_radians=angle)

    def sequences(self, step, seed, variant='base', *, rotation=True, augmentation=True):
        if step < 1: raise ValueError('One-based positive step required')
        variant = self._variant(variant, step, sequence=True)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 31, int(step)]))
        count = 4 if variant == 'tracking_only' else 2
        picks = [('tracking', int(i)) for i in self._tracking_indices(count, rng, variant)]
        if count == 2:
            picks += [(d, int(self._static_indices(d, step, seed+100000, variant, 1)[0])) for d in ('early', 'late')]
        aug_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 32, int(step)]))
        rot_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 33, int(step)]))
        angle = float(rot_rng.uniform(-np.pi, np.pi)) if rotation and rot_rng.random() < .5 else 0.
        records = []
        for domain, row in picks:
            if domain == 'tracking':
                if self._tracking is None: self._tracking = TrackingSource(self.root)
                global_id = int(self.enrollment['tracking_global'][row])
                pack = self._tracking.pack([global_id], future_ms=1024, augment=False)
                h = 1024; ref = dict(self.tracking_rows[row])
            else:
                if self._static is None: self._static = StaticSource(self.root)
                parent, cut = map(int, self.maps[domain][row]); source = self.parents[parent]
                h = min(1024, source['end']-cut)
                pack, lengths = self._static.pack(domain, [row], h, augment=False)
                assert int(lengths[0]) == h
                ref = dict(source_id=source['source_id'], corpus=source['corpus'], parent=parent, cut=cut,
                           session_key=source['session_key'], observed_remaining=source['end']-cut)
            # Physical control truth is copied before actor visibility changes.
            # It is loss-only metadata, never supplied to the policy feature call.
            control_target = pack['target'][:, 640:].copy()
            future_clocks = pack['cut_us'][:, None] + np.arange(1, h+1)[None] * 1000
            control_valid = (pack['valid'][:, 640:] & (pack['available'][:, 640:] <= future_clocks)).copy()
            authentic_delta = pack['human_delta'].copy()
            authentic_incoming = pack['history'][:, -16:].copy()
            if augmentation: augment_pack(pack, aug_rng)
            assert np.array_equal(pack['human_delta'], authentic_delta)
            assert np.array_equal(pack['history'][:, -16:], authentic_incoming)
            rotate_pack(pack, angle)
            control_target = rotate_vectors(control_target, angle)
            assert set(pack) == set(FIELDS) and pack['human_delta'].shape == (1, h, 2)
            records.append(dict(pack={key:value[0] for key,value in pack.items()}, H=h, domain=domain,
                                control_target=control_target[0], control_valid=control_valid[0],
                                ref=dict(ref, row=row, augmented=augmentation, rotation_radians=angle)))
        return records

class TrackingSource:
    def __init__(self, root):
        self.root = Path(root)
        rows = read(self.root / 'tracking_rows.json')
        self.rows = rows
        size = max(r['global_id'] for r in rows) + 1
        self.windows = {'episode_index':np.full(size,-1,np.int64), 'cut_ms':np.zeros(size,np.int64)}
        self.sources = {}
        for r in rows:
            self.windows['episode_index'][r['global_id']] = r['episode_index']
            self.windows['cut_ms'][r['global_id']] = r['cut_ms']
            self.sources[r['episode_index']] = r
        self.cache = {}

    def episode(self, index):
        if index not in self.cache:
            r = self.sources[index]
            path = self.root / 'tracking' / r['source_file']
            # Physical array hashes are checked during public episode preparation.
            with np.load(path, allow_pickle=False) as z:
                self.cache[index] = {k:z[k] for k in ('xy','delta','target','available','valid')}
        return self.cache[index]


    def pack(self, indices, future_ms=64, *, rng=None, augment=False):
        b = len(indices)
        h = 640
        n = h + future_ms
        out = dict(history=np.zeros((b, h, 2), np.float64), position=np.empty((b, 2), np.float64), target=np.zeros((b, n, 2), np.float64), available=np.zeros((b, n), np.int64), valid=np.zeros((b, n), bool), motion_known=np.zeros((b, h), bool), cut_us=np.empty(b, np.int64), human_delta=np.empty((b, future_ms, 2), np.float64))
        for k, index in enumerate(indices):
            e = int(self.windows['episode_index'][index])
            c = int(self.windows['cut_ms'][index])
            ep = self.episode(e)
            take = min(c, h)
            start = c - take
            offset = h - take
            out['history'][k, offset:] = ep['delta'][start:c]
            out['motion_known'][k, offset:] = True
            out['position'][k] = ep['xy'][c - 1]
            out['cut_us'][k] = c * 1000
            out['human_delta'][k] = ep['delta'][c:c + future_ms]
            for key in ('target', 'available', 'valid'):
                out[key][k, offset:] = ep[key][start:c + future_ms]
            if augment:
                if rng.random() < 0.5:
                    period = int(rng.choice([8, 16, 33, 50, 66, 100]))
                    phase = int(rng.integers(period))
                    source = np.maximum(0, (np.arange(n) + phase) // period * period - phase)
                    for key in ('target', 'available', 'valid'):
                        out[key][k] = out[key][k, source]
                if rng.random() < 0.5:
                    retain = int(rng.choice([16, 64, 160, 320]))
                    unknown = h - retain
                    out['history'][k, :unknown] = 0.0
                    out['motion_known'][k, :unknown] = False
                    out['target'][k, :unknown] = 0.0
                    out['valid'][k, :unknown] = False
        return out
