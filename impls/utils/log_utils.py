"""Experiment naming, CSV logging and wandb."""

import csv
import os
import warnings


def get_exp_name(seed, critic=None, actor=None, env_name=None):
    """Compose a deterministic experiment name."""
    parts = [p for p in (env_name, critic, actor) if p]
    parts.append(f"seed_{seed}")
    return "_".join(parts)


class CsvLogger:
    """Append rows to a CSV, writing a header on first use.

    An existing file is appended to with its own header. Keys missing from the header are
    dropped with a warning.
    """

    def __init__(self, path):
        self.path = path
        self._file = None
        self._writer = None
        self._keys = None
        self._warned = set()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row, step=None):
        row = dict(row)
        if step is not None:
            row["step"] = step
        if self._file is None:
            file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            if file_exists:
                with open(self.path, "r", newline="") as f:
                    header = next(csv.reader(f), None)
                self._keys = header if header else list(row.keys())
                self._file = open(self.path, "a", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=self._keys)
            else:
                self._keys = list(row.keys())
                self._file = open(self.path, "w", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=self._keys)
                self._writer.writeheader()
        dropped = sorted(set(row) - set(self._keys) - self._warned)
        if dropped:
            self._warned.update(dropped)
            warnings.warn(
                f"{self.path}: column(s) {dropped} not in header {self._keys}, DROPPED "
                f"from every row."
            )
        self._writer.writerow({k: row.get(k) for k in self._keys})
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()


def setup_wandb(config, project="factored-fb", group=None, name=None, mode=None):
    """Initialize wandb. Returns None when disabled or unavailable."""
    mode = mode or os.environ.get("WANDB_MODE")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=project, group=group, name=name, config=config, mode=mode, reinit=True
    )
