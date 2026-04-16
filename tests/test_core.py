def test_base_config_import():
    from base_config import BaseConfig
    assert BaseConfig is not None

def test_base_model_import():
    from base_model import BaseModel, BaseModelConfig, load_model, save_model
    assert BaseModel is not None

def test_normalizers_import():
    from normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig, BatchNormNormalizerConfig
    assert IdentityNormalizerConfig is not None

def test_utils_import():
    from utils import set_seed_everywhere, EveryNStepsChecker, get_local_workdir
    assert set_seed_everywhere is not None

def test_gym_spaces_import():
    from envs.gym_spaces import space_to_json, json_to_space
    import gymnasium
    import numpy as np
    space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    json_repr = space_to_json(space)
    recovered = json_to_space(json_repr)
    assert recovered.shape == space.shape
