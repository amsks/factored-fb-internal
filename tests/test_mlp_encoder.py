"""MLPEncoder — state-side analogue of DrQEncoder for the param-matched ablation.

Mirrors DrQEncoder's interface: build(obs_space) -> module exposing .output_space
= Box(feature_dim,). Trunk = hidden_layers x (Linear + ReLU); proj =
Linear -> LayerNorm -> Tanh to feature_dim (same output-norm as DrQ's proj).
"""
import numpy as np
import torch
from gymnasium.spaces import Box

from nn_models import MLPEncoderArchiConfig


def _obs_space(dim=28):
    return Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_output_shape_and_space():
    enc = MLPEncoderArchiConfig(feature_dim=256, hidden_dim=128, hidden_layers=3).build(_obs_space(28))
    out = enc(torch.zeros(4, 28))
    assert out.shape == (4, 256)
    assert tuple(enc.output_space.shape) == (256,)


def test_param_count_matches_closed_form():
    # params = (nl-1)*W^2 + (nl+284)*W + 768  for obs_dim=28, feature_dim=256.
    # (first Linear 28->W: 29W; nl-1 hidden WxW: (nl-1)(W^2+W); proj W->256: 256W+256;
    #  LayerNorm(256): 512.)
    W, nl = 1100, 5
    enc = MLPEncoderArchiConfig(feature_dim=256, hidden_dim=W, hidden_layers=nl).build(_obs_space(28))
    n = sum(p.numel() for p in enc.parameters())
    expected = (nl - 1) * W * W + (nl + 284) * W + 768
    assert n == expected
    # within 1% of one DrQ encoder (5,151,136) -> two of these match the pixel run.
    assert abs(n - 5_151_136) / 5_151_136 < 0.01


def test_zero_hidden_layers_is_linear_proj():
    # hidden_layers=0 -> trunk is a single Linear(obs->hidden); still projects to feature_dim.
    enc = MLPEncoderArchiConfig(feature_dim=64, hidden_dim=32, hidden_layers=1).build(_obs_space(10))
    out = enc(torch.zeros(2, 10))
    assert out.shape == (2, 64)
