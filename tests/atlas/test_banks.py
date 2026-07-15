"""Banks build under the jax venv (mu.sample needs jax). Test checks the frozen npz."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.banks import load_banks

BANKS = 'analysis/atlas/banks.npz'


def test_banks_shapes_and_determinism():
    assert Path(BANKS).exists(), 'run build_banks first (Task 5 build step)'
    b = load_banks(BANKS)
    assert b['A_mu'].shape[-1] == 5
    assert b['A_global'].shape == (128, 5)
    assert b['goals_eval'].shape == (5, 28)
    assert set(np.unique(b['state_phase'])).issubset({0, 1, 2, 3})
    assert np.abs(b['A_global']).max() <= 1.0 + 1e-6
    # every phase populated
    for p in range(4):
        assert (b['state_phase'] == p).sum() == 32
