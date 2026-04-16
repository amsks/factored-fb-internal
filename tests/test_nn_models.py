import torch
import numpy as np
import gymnasium


def _obs_space(dim):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def test_backward_map():
    from nn_models import BackwardArchiConfig
    cfg = BackwardArchiConfig(hidden_dim=32, hidden_layers=1)
    net = cfg.build(_obs_space(10), z_dim=8)
    x = torch.randn(4, 10)
    out = net(x)
    assert out.shape == (4, 8)


def test_forward_map():
    from nn_models import ForwardArchiConfig
    cfg = ForwardArchiConfig(hidden_dim=32, hidden_layers=1, num_parallel=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    act = torch.randn(4, 4)
    out = net(obs, z, act)
    assert out.shape == (2, 4, 8)  # (num_parallel, batch, z_dim)


def test_simple_actor():
    from nn_models import SimpleActorArchiConfig
    cfg = SimpleActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    dist = net(obs, z, std=0.2)
    action = dist.sample()
    assert action.shape == (4, 4)


def test_noise_conditioned_actor():
    from nn_models import NoiseConditionedActorArchiConfig
    cfg = NoiseConditionedActorArchiConfig(hidden_dim=32, hidden_layers=1, embedding_layers=2)
    net = cfg.build(_obs_space(10), z_dim=8, action_dim=4)
    obs = torch.randn(4, 10)
    z = torch.randn(4, 8)
    noise = torch.randn(4, 4)
    out = net(obs, z, noise)
    assert out.shape == (4, 4)


def test_vector_field():
    from nn_models import SimpleVectorFieldArchiConfig
    cfg = SimpleVectorFieldArchiConfig(hidden_dim=32, hidden_layers=1)
    net = cfg.build(_obs_space(10), action_dim=4)
    obs = torch.randn(4, 10)
    action = torch.randn(4, 4)
    t = torch.rand(4, 1)
    out = net(obs, action, t)
    assert out.shape == (4, 4)
