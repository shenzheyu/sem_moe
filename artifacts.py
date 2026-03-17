from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_run_dir(output_dir: str | Path, run_name: str | None) -> Path:
    base_dir = Path(output_dir).expanduser().resolve()
    name = run_name or f"profile-{utc_timestamp()}"
    run_dir = base_dir / name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    (run_dir / "raw").mkdir(parents=True, exist_ok=False)
    (run_dir / "stats").mkdir(parents=True, exist_ok=False)
    (run_dir / "vocab").mkdir(parents=True, exist_ok=False)
    return run_dir


def collection_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "collection_manifest.json"


def stats_artifact_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "stats" / "token_expert_stats.pt"


def vocab_artifact_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "vocab" / "vocab_extension.pt"


def raw_shard_path(run_dir: str | Path, shard_index: int) -> Path:
    return Path(run_dir) / "raw" / f"raw_shard_{shard_index:05d}.pt"


def schedule_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "schedule"


def ensure_schedule_dir(run_dir: str | Path) -> Path:
    output_dir = schedule_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def schedule_manifest_path(run_dir: str | Path) -> Path:
    return schedule_dir(run_dir) / "manifest.json"


def schedule_layer_path(run_dir: str | Path, layer_id: int) -> Path:
    return schedule_dir(run_dir) / f"semmoe_layer{layer_id}.npz"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    artifact = dict(payload)
    artifact.setdefault("format_version", FORMAT_VERSION)
    torch.save(artifact, Path(path))


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
