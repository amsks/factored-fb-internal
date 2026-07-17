"""Minimal logging utilities for the impls/ tree. A full wandb/CsvLogger port lands with
the multi-critic PRs; PR1 needs only exp-name + a lightweight CSV logger."""
import csv
import os


def get_exp_name(seed):
    """Compose a run/experiment name. Time-free (deterministic) for PR1."""
    return f'seed_{seed}'


class CsvLogger:
    """Append rows to a CSV, writing a header on first use."""

    def __init__(self, path):
        self.path = path
        self._file = None
        self._writer = None
        self._keys = None
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def log(self, row, step=None):
        row = dict(row)
        if step is not None:
            row['step'] = step
        if self._file is None:
            self._keys = list(row.keys())
            self._file = open(self.path, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=self._keys)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k) for k in self._keys})
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()


def setup_wandb(**kwargs):
    """No-op wandb shim for PR1 (returns None; the real setup lands with logging PR)."""
    return None
