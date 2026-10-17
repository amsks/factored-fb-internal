"""OGBench environment and dataset construction."""
import ogbench
from impls.utils.datasets import FBDataset


def make_env_and_datasets(dataset_name, add_info=False):
    """Make the OGBench environment and train/val datasets.

    :param add_info: Also load the simulator state columns (qpos, qvel, button_states).
    """
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, add_info=add_info)
    train_dataset = FBDataset.create(**train_dataset)
    val_dataset = FBDataset.create(**val_dataset)
    env.reset()
    return env, train_dataset, val_dataset
