import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.primaryA import equivalence_verdict, spearman_gap


def test_spearman_perfect_and_reversed():
    A = np.array([3.0, 1.0, 2.0])
    assert spearman_gap(A, A) == 1.0
    assert spearman_gap(A, -A) == -1.0


def test_equivalence_verdict_uses_margin():
    assert equivalence_verdict(0.80, 0.79, delta=0.05) == 'tie'
    assert equivalence_verdict(0.90, 0.70, delta=0.05) == 'bilinear'
    assert equivalence_verdict(0.70, 0.90, delta=0.05) == 'mlp'
