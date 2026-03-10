from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from profile_artifacts import (
    collection_manifest_path,
    load_json,
    load_torch_artifact,
    schedule_layer_path,
    schedule_manifest_path,
)
from vllm.sem_moe import pick_dp_rank_for_request


@dataclass(frozen=True)
class EvalRequest:
    record_id: str
    token_ids: np.ndarray
    routed_experts: np.ndarray


def run_evaluate_dp_scheduling(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    result = evaluate_dp_scheduling_from_run(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_dp_scheduling_from_run(run_dir: Path) -> dict[str, Any]:
    collection_manifest = load_json(collection_manifest_path(run_dir))
    schedule_manifest = load_json(schedule_manifest_path(run_dir))
    requests = load_eval_requests(run_dir, collection_manifest)
    if not requests:
        raise ValueError("No raw requests found; cannot evaluate Attention-DP scheduling.")

    layer_labels, dp_score_full = load_schedule_state(run_dir, schedule_manifest)
    num_devices = int(schedule_manifest["num_devices"])

    return {
        "run_dir": str(run_dir),
        "model_name": schedule_manifest["model_name"],
        "num_devices": num_devices,
        "request_count": len(requests),
        "baseline": evaluate_strategy(
            strategy_name="round_robin",
            requests=requests,
            assignments=round_robin_assignments(len(requests), num_devices),
            layer_labels=layer_labels,
            num_devices=num_devices,
        ),
        "semmoe": evaluate_strategy(
            strategy_name="semmoe",
            requests=requests,
            assignments=sem_moe_assignments(requests, dp_score_full),
            layer_labels=layer_labels,
            num_devices=num_devices,
        ),
    }


def load_eval_requests(
    run_dir: Path,
    collection_manifest: dict[str, Any],
) -> list[EvalRequest]:
    requests: list[EvalRequest] = []
    for relative_path in collection_manifest["raw_shards"]:
        shard = load_torch_artifact(run_dir / relative_path)
        for record in shard["records"]:
            requests.append(
                EvalRequest(
                    record_id=str(record["record_id"]),
                    token_ids=record["prompt_token_ids"].cpu().numpy().astype(np.int64),
                    routed_experts=record["routed_experts"].cpu().numpy().astype(np.int64),
                )
            )
    return requests


def load_schedule_state(
    run_dir: Path,
    schedule_manifest: dict[str, Any],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    num_devices = int(schedule_manifest["num_devices"])
    layer_labels: dict[int, np.ndarray] = {}
    dp_score_full: np.ndarray | None = None

    for layer_id in schedule_manifest["moe_layer_ids"]:
        with np.load(schedule_layer_path(run_dir, int(layer_id))) as payload:
            layer_labels[int(layer_id)] = payload["E"].astype(np.int64, copy=True)
            if "T_score_full" in payload:
                layer_score = payload["T_score_full"].astype(np.float32, copy=True)
            elif "Tp_full" in payload and "T_full" in payload:
                labels = payload["T_full"].astype(np.int64, copy=False)
                conf = payload["Tp_full"].astype(np.float32, copy=False)
                layer_score = np.zeros((labels.shape[0], num_devices), dtype=np.float32)
                layer_score[np.arange(labels.shape[0]), labels] = conf
            else:
                labels = payload["T_full"].astype(np.int64, copy=False)
                layer_score = np.zeros((labels.shape[0], num_devices), dtype=np.float32)
                layer_score[np.arange(labels.shape[0]), labels] = 1.0

        if dp_score_full is None:
            dp_score_full = np.zeros_like(layer_score, dtype=np.float32)
        elif dp_score_full.shape != layer_score.shape:
            raise ValueError(
                "All schedule layers must share the same full-vocab score shape for DP evaluation."
            )
        dp_score_full += layer_score

    if dp_score_full is None:
        raise ValueError("No schedule layers found while evaluating Attention-DP scheduling.")
    return layer_labels, dp_score_full


def round_robin_assignments(num_requests: int, num_devices: int) -> np.ndarray:
    return np.arange(num_requests, dtype=np.int64) % num_devices


def sem_moe_assignments(
    requests: list[EvalRequest],
    dp_score_full: np.ndarray,
) -> np.ndarray:
    dev_mask = np.ones((dp_score_full.shape[1],), dtype=bool)
    assignments = np.empty((len(requests),), dtype=np.int64)
    for idx, request in enumerate(requests):
        assignments[idx] = pick_dp_rank_for_request(
            token_ids=request.token_ids.tolist(),
            score_full=dp_score_full,
            dev_mask=dev_mask,
        )
    return assignments


def evaluate_strategy(
    strategy_name: str,
    requests: list[EvalRequest],
    assignments: np.ndarray,
    layer_labels: dict[int, np.ndarray],
    num_devices: int,
) -> dict[str, Any]:
    token_load_per_device = np.zeros((num_devices,), dtype=np.int64)
    for request, assigned_device in zip(requests, assignments.tolist(), strict=True):
        token_load_per_device[assigned_device] += int(request.token_ids.shape[0])

    total_local = 0
    total_remote = 0
    per_layer: dict[str, dict[str, float]] = {}

    for layer_id, expert_labels in layer_labels.items():
        layer_local = 0
        layer_remote = 0
        for request, assigned_device in zip(requests, assignments.tolist(), strict=True):
            routed = request.routed_experts[:, int(layer_id)]
            for expert_ids in routed:
                labels = expert_labels[expert_ids]
                local_hits = int(np.count_nonzero(labels == assigned_device))
                layer_local += local_hits
                layer_remote += int(expert_ids.shape[0]) - local_hits

        total_local += layer_local
        total_remote += layer_remote
        layer_total = layer_local + layer_remote
        per_layer[str(layer_id)] = {
            "lar": float(layer_local / layer_total) if layer_total else 0.0,
            "local_activations": float(layer_local),
            "remote_activations": float(layer_remote),
            "all2all_volume_proxy": float(layer_remote),
        }

    med = median(token_load_per_device.tolist())
    if med == 0:
        imbalance = float("inf") if int(token_load_per_device.max()) > 0 else 1.0
    else:
        imbalance = float(token_load_per_device.max() / med)

    total = total_local + total_remote
    return {
        "strategy": strategy_name,
        "lar": float(total_local / total) if total else 0.0,
        "local_activations": float(total_local),
        "remote_activations": float(total_remote),
        "all2all_volume_proxy": float(total_remote),
        "token_load_per_device": token_load_per_device.tolist(),
        "load_imbalance_rate": imbalance,
        "per_layer": per_layer,
    }
