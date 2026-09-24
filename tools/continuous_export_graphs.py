"""Original neural graph boundaries used by the Continuous export."""
import numpy as np
import torch

class Motor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input = model.input
        self.gru = model.encoder
        self.trunk = model.trunk
        self.output = model.output

    def forward(self, coarse, fine, dynamics):
        _, hidden = self.gru(self.input(coarse))
        encoded = self.trunk(torch.cat((hidden[-1], fine, dynamics), 1))
        return encoded, self.output(encoded).reshape(1, 16, 21, 2)


class Choice(torch.nn.Module):
    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, encoded, unary_geometry, pairs, previous_valid):
        latent = encoded[:, None, :].expand(1, 16, 96)
        unary = self.selector.unary(torch.cat((latent, unary_geometry), -1))[..., 0]
        context = self.selector.context(encoded)[:, None, :].expand(1, 16, 16)
        transition = self.selector.pair(torch.cat((pairs, context), -1))[..., 0]
        return unary + transition * previous_valid


class ChoiceSplit(Choice):
    """Compute shared latent/context linear projections once per decision."""
    def forward(self, encoded, unary_geometry, pairs, previous_valid):
        linear = torch.nn.functional.linear
        s = self.selector
        shared = linear(encoded, s.unary[0].weight[:, :96], s.unary[0].bias)
        geometry = linear(unary_geometry, s.unary[0].weight[:, 96:])
        unary = s.unary[2](s.unary[1](shared[:, None] + geometry))[..., 0]
        context = s.context(encoded)
        shared_pair = linear(context, s.pair[0].weight[:, 20:], s.pair[0].bias)
        pair = linear(pairs, s.pair[0].weight[:, :20])
        transition = s.pair[2](s.pair[1](pair + shared_pair[:, None]))[..., 0]
        return unary + transition * previous_valid


class Brake(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.brake, self.frequency = model.encoder, model.brake, model.frequency

    def forward(self, context):
        encoded = self.encoder(context)
        q = self.brake(encoded).reshape(1, 16, 11)
        return q[:, :, 0].clamp(np.log(4), np.log(192)).exp(), q[:, :, 1:].reshape(1,16,5,2) * 64., self.frequency(encoded)
