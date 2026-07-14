import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas import config


def test_constants_present_and_typed():
    assert config.N_MU == 256
    assert config.ACTION_DIM == 5 and config.OBS_DIM == 28
    assert config.PHASES == ('P0', 'P1', 'P2', 'P3')
    assert 0.0 < config.COV_SHRINK_LAMBDA < 1.0
    assert config.MPPI_REPLICATES >= 1
    assert isinstance(config.SEED, int)
