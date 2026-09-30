"""OGBench environment + dataset construction, ported from OGBench upstream
impls/utils/env_utils.py. Uses the compact dataset and wraps it in FBDataset."""
import ogbench
from impls.utils.datasets import FBDataset


def make_env_and_datasets(dataset_name, add_info=False):
    """Make the OGBench environment and FB datasets (compact, unlabeled).

    `add_info` asks OGBench's loader for the raw simulator state columns ('qpos', 'qvel',
    and 'button_states' on scene) that ship in every npz but are dropped by default. The
    reward-relabel eval needs them; off otherwise, so the batch keys stay unchanged.
    """
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, add_info=add_info)
    train_dataset = FBDataset.create(**train_dataset)
    val_dataset = FBDataset.create(**val_dataset)
    env.reset()
    return env, train_dataset, val_dataset
