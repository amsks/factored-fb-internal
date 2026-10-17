import jax.numpy as jnp
from impls.utils.env_utils import make_env_and_datasets


def test_cube_single_dataset_batch():
    env, train_ds, val_ds = make_env_and_datasets("cube-single-play-v0")
    batch = train_ds.sample(16)
    assert batch["observations"].shape[0] == 16
    assert batch["actions"].shape[0] == 16
    assert batch["next_observations"].shape == batch["observations"].shape
    # terminals are forced to 0
    assert float(jnp.asarray(batch["terminals"]).sum()) == 0.0
