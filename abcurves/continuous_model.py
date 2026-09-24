"""Torch training/reference networks for the selected Continuous Planner.

Production inference is implemented in _continuous. These networks preserve the
selected stage-dependent motor dynamics, ProDMP basis and learned components.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import prodmp

VELOCITY_SCALE = 1.1982087601807259
POSITION_SCALE = 81.73350722609987


def _reference():
    return prodmp

@dataclass(frozen=True)
class Config:
    enc: str = "gru"
    decoder: str = "prodmp"
    forecast: int = 128
    commit: int = 64
    history: int = 640
    head_hold: int = 1
    memory: bool = False
    dynamics: bool = True
    velocity_scale: float = VELOCITY_SCALE
    position_scale: float = POSITION_SCALE
    target_secant_ms: int = 32

    def __post_init__(self):
        if self.enc != "gru" or self.decoder != "prodmp":
            raise ValueError("Unknown campaign encoder/decoder")
        if self.forecast not in (128, 256) or self.commit not in (32, 64):
            raise ValueError("Unsupported forecast/commit")
        if self.history not in (160, 640) or self.head_hold not in (1, 4, 16):
            raise ValueError("Unsupported history/head persistence")
        if type(self.memory) is not bool or type(self.dynamics) is not bool:
            raise TypeError("memory and dynamics must be bool")
        if (self.velocity_scale, self.position_scale, self.target_secant_ms) != (
                VELOCITY_SCALE, POSITION_SCALE, 32):
            raise ValueError("Campaign physical feature scales are fixed")

    def to_dict(self):
        return asdict(self)


def _config(config=None):
    return Config() if config is None else (Config(**config) if isinstance(config, dict) else config)


def features(history, position, target, available, valid, motion_known, cut_us, config=None):
    """The source-bound temporal70x9/fine54 kernel, with an optional160 censor.

    Inputs contain640 already observed motion/target slots; availability is
    tested against each slot's physical clock. No future transport is accepted.
    """
    config = _config(config)
    b, h, _ = history.shape
    if h != 640 or target.shape != (b, h, 2) or motion_known.shape != (b, h):
        raise ValueError("Expected640ms causal state")
    if config.history == 160:
        keep = torch.arange(h, device=history.device)[None] >= 480
        motion_known = motion_known & keep
        valid = valid & keep
    endpoint = cut_us[:, None] - torch.arange(h-1, -1, -1, device=history.device)[None]*1000
    known = valid & (available <= endpoint)
    motion = torch.where(motion_known[:, :, None], history, 0.)
    goal = torch.where(known[:, :, None], target, 0.)
    old = (torch.arange(h, device=history.device)-32).clamp_min(0)
    pair = known & known[:, old] & (torch.arange(h, device=history.device)[None] >= 32)
    tv = torch.where(pair[:, :, None], (goal-goal[:, old])/32., 0.)
    points = position[:, None] + motion.cumsum(1) - motion.sum(1)[:, None]
    rel = torch.where((known & motion_known)[:, :, None], torch.asinh((goal-points)/config.position_scale), 0.)
    index = torch.cat((torch.arange(15, 480, 16, device=history.device), torch.arange(483, 640, 4, device=history.device)))
    mean_v = torch.cat((motion[:, :480].reshape(b, 30, 16, 2).mean(2), motion[:, 480:].reshape(b, 40, 4, 2).mean(2)), 1)
    coverage = torch.cat((motion_known[:, :480].float().reshape(b, 30, 16).mean(2), motion_known[:, 480:].float().reshape(b, 40, 4).mean(2)), 1)
    widths = torch.cat((torch.full((30,), 1., device=history.device), torch.full((40,), .25, device=history.device)))
    coarse = torch.cat((mean_v/config.velocity_scale, rel[:, index], tv[:, index]/config.velocity_scale,
        known[:, index, None].to(history.dtype), coverage[:, :, None], widths[None, :, None].expand(b, -1, -1)), 2)
    current_error = torch.where(known[:, -1, None], torch.asinh((goal[:, -1]-position)/config.position_scale), 0.)
    fine = torch.cat(((motion[:, -16:]/config.velocity_scale).flatten(1), motion_known[:, -16:].to(history.dtype),
        current_error, tv[:, -1]/config.velocity_scale, known[:, -1, None].to(history.dtype), motion_known.float().mean(1, keepdim=True)), 1)
    return coarse.float(), fine.float()


@lru_cache(maxsize=4)
def geometry(decoder, forecast):
    """Full-rank21D physical QR; return independent NumPy arrays by convention.

    QR rows are position/64counts and sampled velocity/4counts per ms.
    It conditions the complete forecast; it never fits/inverts a short prefix.
    """
    if decoder not in ("prodmp", "knots") or forecast not in (128, 256):
        raise ValueError("Unsupported geometry")
    t = np.arange(1, forecast+1, dtype=np.float64)
    if decoder == "prodmp":
        reference = _reference()
        primitive = reference.ProDMP(reference.ProDMPConfig(n_basis=20, alpha=25., alpha_phase=3., grid_points=2000))
        _, carry_position, position = primitive._components(t, float(forecast), 0.)
        velocity = np.diff(position, axis=0, prepend=np.zeros((1, 21)))
        carry_velocity = np.diff(carry_position, prepend=0.)
    else:
        anchors = np.linspace(0., float(forecast), 22)
        left = np.searchsorted(anchors, t, side="left")-1
        fraction = (t-anchors[left])/(anchors[left+1]-anchors[left])
        interpolation = np.zeros((forecast, 22), dtype=np.float64)
        interpolation[np.arange(forecast), left] = 1.-fraction
        interpolation[np.arange(forecast), left+1] = fraction
        velocity, carry_velocity = interpolation[:, 1:], interpolation[:, 0]
        position, carry_position = velocity.cumsum(0), carry_velocity.cumsum(0)
    physical = np.concatenate((position/64., velocity/4.), axis=0)
    singular = np.linalg.svd(physical, compute_uv=False)
    rank = int(np.sum(singular > max(physical.shape)*np.finfo(np.float64).eps*singular[0]))
    if rank != 21:
        raise ArithmeticError(f"Full physical basis lost rank: {rank}/21")
    _, r = np.linalg.qr(physical, mode="reduced")
    signs = np.where(np.diag(r) < 0., -1., 1.)
    r = signs[:, None]*r
    transform = np.linalg.solve(r, np.eye(21))
    result = dict(position_basis=position@transform, velocity_basis=velocity@transform,
        carry_position=carry_position, carry_velocity=carry_velocity,
        physical_transform=transform, qr_r=r, singular_values=singular,
        original_position_basis=position, original_velocity_basis=velocity)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ArithmeticError("Nonfinite physical geometry")
    return result


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(F.pad(x, (2*self.dilation, 0))).transpose(1, 2)
        return x + F.silu(self.norm(y)).transpose(1, 2)


class MotorBase(nn.Module):
    heads = 16

    def __init__(self, config=None):
        super().__init__()
        self.config = _config(config)
        if not isinstance(self.config, Config):
            raise TypeError("Expected Config or its dictionary")
        if self.config.enc == "gru":
            self.input = nn.Sequential(nn.Linear(9, 96), nn.SiLU())
            self.encoder = nn.GRU(96, 96, batch_first=True)
            encoded_size = 96
        else:
            self.input = nn.Conv1d(9, 56, 1)
            self.encoder = nn.Sequential(*(CausalBlock(56, d) for d in (1, 2, 4, 8, 16, 32)))
            encoded_size = 56
        self.trunk = nn.Sequential(nn.Linear(encoded_size+54+(8 if self.config.dynamics else 0), 96), nn.SiLU())
        self.decision_memory = nn.GRUCell(96, 96) if self.config.memory else None
        self.output = nn.Linear(96, self.heads*21*2)
        arrays = geometry(self.config.decoder, self.config.forecast)
        self.decoder_fingerprint = _fingerprint(arrays)
        for name, value in arrays.items():
            self.register_buffer(name, torch.from_numpy(value.copy()))
        self.last_diagnostics = None

    def _view(self, coarse, fine):
        if coarse.ndim != 3 or coarse.shape[1:] != (70, 9) or fine.shape != (len(coarse), 54):
            raise ValueError("Expected coarse[B,70,9] and fine[B,54]")
        if self.config.history == 160:
            # First eight retained target secants reach before the160ms view.
            coarse = coarse[:, 30:].clone()
            coarse[:, :8, 4:6] = 0.
            fine = torch.cat((fine[:, :53], coarse[:, :, 7].mean(1, keepdim=True)), 1)
        return coarse.float(), fine.float()

    def _dynamics(self, fine):
        velocity = fine[:, :32].reshape(-1, 16, 2)
        known = fine[:, 32:48] > .5
        current_v = velocity[:, -1]
        acceleration = (velocity[:, -4:].mean(1)-velocity[:, -8:-4].mean(1))*self.config.velocity_scale/4.
        acceleration = torch.where(known[:, -8:].all(1)[:, None], acceleration, 0.)/.25
        error = torch.sinh(fine[:, 48:50].double())*self.config.position_scale
        length = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
        direction = (error/torch.where(length > 0., length, 1.)).float()
        direction = torch.where((fine[:, 52] > .5)[:, None], direction, 0.)
        normal = torch.stack((-direction[:, 1], direction[:, 0]), -1)
        relative_v = current_v-fine[:, 50:52]
        return torch.cat((current_v, acceleration,
            (relative_v*direction).sum(1, keepdim=True), (relative_v*normal).sum(1, keepdim=True),
            (acceleration*direction).sum(1, keepdim=True), (acceleration*normal).sum(1, keepdim=True)), 1)

    def encode(self, coarse, fine, memory=None):
        coarse, fine = self._view(coarse, fine)
        if self.output.weight.dtype != torch.float32:
            raise TypeError("Campaign neural parameters must remain float32")
        if self.config.enc == "gru":
            _, h = self.encoder(self.input(coarse))
            hidden = h[-1]
        else:
            hidden = self.encoder(self.input(coarse.transpose(1, 2)))[:, :, -1]
        parts = [hidden, fine]
        if self.config.dynamics:
            parts.append(self._dynamics(fine))
        observation = self.trunk(torch.cat(parts, 1))
        if self.decision_memory is not None:
            if memory is None:
                memory = torch.zeros_like(observation)
            if memory.shape != observation.shape:
                raise ValueError("Decision memory must have shape[B,96]")
            new_memory = self.decision_memory(observation, memory)
            return new_memory, new_memory
        if memory is not None:
            raise ValueError("Memory supplied to a memory-free configuration")
        return observation, None

    def decode(self, coefficients, incoming):
        if coefficients.shape[-2:] != (21, 2) or coefficients.ndim != 4:
            raise ValueError("Expected QR coefficients[B,K,21,2]")
        if incoming.shape != (len(coefficients), 2):
            raise ValueError("Expected actual incoming[B,2]")
        q = coefficients.double()
        delta = torch.einsum("hw,bkwd->bkhd", self.velocity_basis, q)
        delta = delta+self.carry_velocity[None, None, :, None]*incoming.double()[:, None, None, :]
        position = delta.cumsum(2)
        raw = torch.einsum("uv,bkvd->bkud", self.physical_transform, q)
        return delta, position, raw

    def forward(self, coarse, fine, incoming, memory=None):
        encoded, new_memory = self.encode(coarse, fine, memory)
        coefficients = self.output(encoded).reshape(-1, self.heads, 21, 2)
        delta, position, raw = self.decode(coefficients, incoming)
        self.last_diagnostics = dict(coefficients=coefficients, physical_coefficients=raw,
            positions=position, encoded=encoded)
        return delta, new_memory

    def sample_heads(self, generator, batch):
        return sample_heads(generator, batch, self.output.weight.device)


def sample_heads(generator, batch, device="cpu"):
    """Uniform parameter-independent heads; the caller owns the only RNG."""
    if generator is None:
        raise ValueError("An explicit caller generator is required")
    return torch.randint(16, (batch,), generator=generator, device=generator.device).to(device)


class Motor(MotorBase):
    arm = 'A'

    def _dynamics(self, fine):
        velocity=fine[:,:32].reshape(-1,16,2); known=fine[:,32:48]>.5
        current=velocity[:,-1]
        acc=(velocity[:,-4:].mean(1)-velocity[:,-8:-4].mean(1))*self.config.velocity_scale/4.
        acc=torch.where(known[:,-8:].all(1)[:,None],acc,0.)/.25
        error=torch.sinh(fine[:,48:50].double())*self.config.position_scale
        # Smooth feature normalization, not a movement cutoff or runtime deadband.
        direction=(error/torch.sqrt(error.square().sum(-1,keepdim=True)+100.)).float()
        direction=torch.where((fine[:,52]>.5)[:,None],direction,0.)
        normal=torch.stack((-direction[:,1],direction[:,0]),-1)
        relative=current-fine[:,50:52]
        return torch.cat((current,acc,(relative*direction).sum(1,keepdim=True),
            (relative*normal).sum(1,keepdim=True),(acc*direction).sum(1,keepdim=True),
            (acc*normal).sum(1,keepdim=True)),1)

    def act(self, coarse, fine, incoming, position, state, generator):
        forecasts,_=self(coarse,fine,incoming)
        heads=sample_heads(generator,len(coarse),coarse.device)
        selected=forecasts[torch.arange(len(coarse),device=coarse.device),heads,:32]
        return selected,None,dict(head=heads,forecasts=forecasts,log_prob=None,entropy=None)


def head_geometry(forecasts):
    return forecasts[..., :64, :].reshape(*forecasts.shape[:-2], 16, 4, 2).mean(-2).float()


def pair_features(previous, current, actual_step):
    """[...old,16,2], [...new,16,2], [...2] -> [...old,new,20]."""
    shape = torch.broadcast_shapes(previous.shape[:-3], current.shape[:-3])
    old, new = previous.shape[-3], current.shape[-3]
    delta = previous[..., :, None, 8:, :] - current[..., None, :, :8, :]
    endpoint = ((previous.sum(-2) * 4 - actual_step[..., None, :])[..., :, None, :]
                - current[..., None, :, :8, :].sum(-2) * 4) / 32
    feedback = (previous[..., :8, :].sum(-2) * 4 - actual_step[..., None, :]) / 32
    feedback = feedback[..., :, None, :].expand(*shape, old, new, 2)
    return torch.asinh(torch.cat((delta.flatten(-2), endpoint, feedback), -1))


class Selector(nn.Module):
    def __init__(self, coherent=False):
        super().__init__()
        self.coherent = bool(coherent)
        self.unary = nn.Sequential(nn.Linear(112, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.unary[-1].weight); nn.init.zeros_(self.unary[-1].bias)
        if self.coherent:
            self.context = nn.Sequential(nn.Linear(96, 16), nn.SiLU())
            self.pair = nn.Sequential(nn.Linear(36, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.pair[-1].weight); nn.init.zeros_(self.pair[-1].bias)

    def logits(self, encoded, heads, previous=None, actual_step=None):
        latent = encoded.unsqueeze(-2).expand(*heads.shape[:-2], 96)
        unary = self.unary(torch.cat((latent, torch.asinh(heads[..., :8, :].flatten(-2))), -1))[..., 0]
        if previous is None or not self.coherent:
            return unary, None
        pairs = pair_features(previous, heads, actual_step)
        context = self.context(encoded)[..., None, None, :].expand(*pairs.shape[:-1], 16)
        transition = unary.unsqueeze(-2) + self.pair(torch.cat((pairs, context), -1))[..., 0]
        return unary, transition

    def sequence_loss(self, encoded, heads, displacement, emission, mask):
        previous = torch.cat((torch.zeros_like(heads[:, :1]), heads[:, :-1]), 1)
        unary, transition = self.logits(encoded, heads, previous, displacement)
        initial = unary.log_softmax(-1)
        if transition is not None: transition = transition.log_softmax(-1)
        posterior = torch.zeros_like(initial[:, 0]); total = torch.zeros_like(posterior[:, 0])
        for t in range(encoded.shape[1]):
            predictive = initial[:, t]
            if t and transition is not None:
                continued = torch.logsumexp(posterior[..., None] + transition[:, t], -2)
                predictive = torch.where(mask[:, t-1, None], continued, predictive)
            joint = predictive + emission[:, t]
            normalizer = torch.logsumexp(joint, -1)
            posterior = joint - normalizer[:, None]
            total = total - torch.where(mask[:, t], normalizer, 0.)
        return total / mask.sum(1).clamp_min(1)


class Components(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(32,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU())
        self.brake=nn.Linear(96,16*11)
        self.frequency=nn.Linear(96,16)
        self.hazard=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        with torch.no_grad():
            self.brake.weight.mul_(.05);self.brake.bias.mul_(.05)
            self.brake.bias.reshape(16,11)[:,0]=np.log(64.)+torch.linspace(-.5,.5,16)
            self.hazard[-1].bias.fill_(-3.)

    def forward(self,x):
        z=self.encoder(x);q=self.brake(z).reshape(-1,16,11)
        duration=q[:,:,0].clamp(np.log(4),np.log(192)).exp()
        coeff=q[:,:,1:].reshape(-1,16,5,2)*64.
        return duration,coeff,self.frequency(z),self.hazard(x)
