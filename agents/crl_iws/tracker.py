"""Lightweight experiment tracker (WandB + local jsonl/csv), mirroring UM's ExperimentTracker."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "run"


@dataclass
class TrackerPaths:
    run_dir: Path
    checkpoint_dir: Path
    metrics_jsonl_path: Path
    metrics_csv_path: Path
    config_path: Path
    video_dir: Path


class CRLIWSTracker:
    def __init__(
        self,
        *,
        output_root: str | Path,
        project: str,
        run_name: str,
        config: dict[str, Any],
        use_wandb: bool = True,
        wandb_mode: str = "online",
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        wandb_tags: list[str] | None = None,
        resume_run_dir: str | Path | None = None,
    ) -> None:
        if resume_run_dir is not None:
            run_dir = Path(resume_run_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_name = run_dir.name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolved_name = f"{timestamp}_{_slugify(run_name)}"
            run_dir = Path(output_root) / project / resolved_name
            run_dir.mkdir(parents=True, exist_ok=True)
        self.paths = TrackerPaths(
            run_dir=run_dir,
            checkpoint_dir=run_dir / "checkpoints",
            metrics_jsonl_path=run_dir / "metrics.jsonl",
            metrics_csv_path=run_dir / "metrics.csv",
            config_path=run_dir / "config.json",
            video_dir=run_dir / "eval_videos",
        )
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.paths.video_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._wandb_run = None
        self._wandb_run_path = run_dir / "wandb_run.json"

        if not self.paths.config_path.exists():
            self.paths.config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        if self.paths.metrics_jsonl_path.exists():
            for line in self.paths.metrics_jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._rows.append(json.loads(line))

        if use_wandb and wandb_mode != "disabled" and wandb is not None:
            wandb_kwargs: dict[str, Any] = {
                "project": project,
                "entity": wandb_entity,
                "group": wandb_group,
                "tags": list(wandb_tags) if wandb_tags else None,
                "config": config,
                "mode": wandb_mode,
            }
            saved_run_id = self._read_wandb_run_id()
            if resume_run_dir is not None and saved_run_id:
                wandb_kwargs["id"] = saved_run_id
                wandb_kwargs["resume"] = "must"
                wandb_kwargs["name"] = resolved_name
            else:
                wandb_kwargs["name"] = resolved_name
            self._wandb_run = wandb.init(**wandb_kwargs)
            if not self._wandb_run_path.exists() and self._wandb_run is not None:
                self._wandb_run_path.write_text(
                    json.dumps(
                        {
                            "id": str(self._wandb_run.id),
                            "name": str(self._wandb_run.name),
                            "project": str(self._wandb_run.project),
                            "entity": getattr(self._wandb_run, "entity", None),
                            "group": wandb_group,
                            "url": getattr(self._wandb_run, "url", None),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )

    def _read_wandb_run_id(self) -> str | None:
        if not self._wandb_run_path.exists():
            return None
        try:
            payload = json.loads(self._wandb_run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        run_id = payload.get("id")
        return str(run_id) if run_id else None

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        row = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self._rows.append(row)
        with self.paths.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if self._wandb_run is not None:
            wandb.log(row, step=int(step))

    def log_video(
        self,
        name: str,
        frames: list[Any],
        *,
        step: int,
        fps: int = 20,
    ) -> Path | None:
        if not frames:
            return None
        try:
            import imageio
        except ImportError:  # pragma: no cover
            return None

        slug = _slugify(name)
        gif_path = self.paths.video_dir / f"{int(step):08d}_{slug}.gif"
        rgb_frames = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                rgb_frames.append(arr.astype(np.uint8))
        if not rgb_frames:
            return None
        imageio.mimsave(
            gif_path,
            rgb_frames,
            format="GIF",
            loop=0,
            duration=max(1, int(1000 / max(fps, 1))),
        )
        if self._wandb_run is not None and wandb is not None:
            wandb.log({name: wandb.Video(str(gif_path), fps=fps, format="gif")}, step=int(step))
        return gif_path

    def finish(self) -> None:
        if self._rows:
            fieldnames = sorted({key for row in self._rows for key in row})
            with self.paths.metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)
        if self._wandb_run is not None:
            wandb.finish()
