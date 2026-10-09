"""Run pytest through the existing cached-dependency Python wrapper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
