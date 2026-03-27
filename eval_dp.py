from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numba
import numpy as np

from artifacts import (
    collection_manifest_path,
    load_json,
    load_torch_artifact,
    schedule_layer_path,
    schedule_manifest_path,
)
from progress_utils import progress_iter


@dataclass(frozen=True)
class EvalRequest:
    record_id: str
    token_ids: np.ndarray
    routed_experts: np.ndarray


@dataclass(frozen=True)
class PreparedEvalData:
    all_token_ids: np.ndarray
    all_routed: np.ndarray
    req_lengths: np.ndarray
    req_offsets: np.ndarray
    request_indices: np.ndarray


def run_evaluate_dp_scheduling(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    result = evaluate_dp_scheduling_from_run(
        run_dir,
        include_layerwise_diagnostic=getattr(args, "layerwise_diagnostic", False),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_dp_scheduling_from_run(
    run_dir: Path,
    include_layerwise_diagnostic: bool = False,
) -> dict[str, Any]:
    collection_manifest = load_json(collection_manifest_path(run_dir))
    schedule_manifest = load_json(schedule_manifest_path(run_dir))
    requests = load_eval_requests(run_dir, collection_manifest)
    if not requests:
        raise ValueError("No raw requests found; cannot evaluate Attention-DP scheduling.")

    layer_order = [int(layer_id) for layer_id in schedule_manifest["moe_layer_ids"]]
    eval_data = prepare_eval_data(requests)
    layer_labels, layer_scores_by_layer, dp_score_full = load_schedule_state(
        run_dir, schedule_manifest
    )
    baseline_layer_labels = build_linear_layer_labels(layer_labels, schedule_manifest)
    num_devices = int(schedule_manifest["num_devices"])
    rr_assignments = round_robin_assignments(len(requests), num_devices)
    sem_assignments = sem_moe_assignments(eval_data, dp_score_full)

    result = {
        "run_dir": str(run_dir),
        "model_name": schedule_manifest["model_name"],
        "num_devices": num_devices,
        "request_count": len(requests),
        "baseline": evaluate_strategy(
            strategy_name="base_vllm_round_robin",
            eval_data=eval_data,
            assignments=rr_assignments,
            layer_order=layer_order,
            layer_labels=baseline_layer_labels,
            num_devices=num_devices,
        ),
        "semmoe_placement": evaluate_strategy(
            strategy_name="semmoe_placement_round_robin",
            eval_data=eval_data,
            assignments=rr_assignments,
            layer_order=layer_order,
            layer_labels=layer_labels,
            num_devices=num_devices,
        ),
        "semmoe": evaluate_strategy(
            strategy_name="semmoe",
            eval_data=eval_data,
            assignments=sem_assignments,
            layer_order=layer_order,
            layer_labels=layer_labels,
            num_devices=num_devices,
        ),
    }
    if include_layerwise_diagnostic:
        result["semmoe_layerwise"] = evaluate_layerwise_diagnostic(
            strategy_name="semmoe_layerwise",
            eval_data=eval_data,
            layer_order=layer_order,
            layer_labels=layer_labels,
            layer_scores_by_layer=layer_scores_by_layer,
            num_devices=num_devices,
        )
    return result


def load_eval_requests(
    run_dir: Path,
    collection_manifest: dict[str, Any],
) -> list[EvalRequest]:
    shard_paths = collection_manifest["raw_shards"]
    requests: list[EvalRequest] = []
    for relative_path in progress_iter(
        shard_paths, total=len(shard_paths), desc="Loading eval requests"
    ):
        shard = load_torch_artifact(run_dir / relative_path)
        for record in shard["records"]:
            requests.append(
                EvalRequest(
                    record_id=str(record["record_id"]),
                    token_ids=record["prompt_token_ids"].numpy(),
                    routed_experts=record["routed_experts"].numpy(),
                )
            )
    return requests


def prepare_eval_data(requests: list[EvalRequest]) -> PreparedEvalData:
    req_lengths = np.array([r.token_ids.shape[0] for r in requests], dtype=np.int64)
    req_offsets = np.zeros_like(req_lengths)
    if req_lengths.size > 1:
        req_offsets[1:] = np.cumsum(req_lengths[:-1])
    return PreparedEvalData(
        all_token_ids=np.concatenate([r.token_ids for r in requests]).astype(
            np.int64, copy=False
        ),
        all_routed=np.concatenate([r.routed_experts for r in requests]),
        req_lengths=req_lengths,
        req_offsets=req_offsets,
        request_indices=np.arange(len(requests), dtype=np.int64),
    )


def load_schedule_state(
    run_dir: Path,
    schedule_manifest: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    num_devices = int(schedule_manifest["num_devices"])
    layer_labels: dict[int, np.ndarray] = {}
    layer_scores_by_layer: dict[int, np.ndarray] = {}
    dp_score_full: np.ndarray | None = None

    for layer_id in schedule_manifest["moe_layer_ids"]:
        with np.load(schedule_layer_path(run_dir, int(layer_id))) as payload:
            layer_id_int = int(layer_id)
            layer_labels[layer_id_int] = payload["E"].astype(np.int64, copy=True)
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
            layer_scores_by_layer[layer_id_int] = layer_score

        if dp_score_full is None:
            dp_score_full = np.zeros_like(layer_score, dtype=np.float32)
        elif dp_score_full.shape != layer_score.shape:
            raise ValueError(
                "All schedule layers must share the same full-vocab score shape for DP evaluation."
            )
        dp_score_full += layer_score

    if dp_score_full is None:
        raise ValueError("No schedule layers found while evaluating Attention-DP scheduling.")
    return layer_labels, layer_scores_by_layer, dp_score_full


def build_linear_layer_labels(
    semmoe_layer_labels: dict[int, np.ndarray],
    schedule_manifest: dict[str, Any],
) -> dict[int, np.ndarray]:
    num_devices = int(schedule_manifest["num_devices"])
    if not semmoe_layer_labels:
        raise ValueError("No Sem-MoE layer labels found while building baseline placement.")

    num_experts = max(int(labels.shape[0]) for labels in semmoe_layer_labels.values())
    experts_per_device = num_experts // num_devices
    if experts_per_device * num_devices != num_experts:
        raise ValueError(
            f"num_experts={num_experts} is not divisible by num_devices={num_devices}."
        )

    linear_labels = np.repeat(np.arange(num_devices, dtype=np.int64), experts_per_device)
    return {
        layer_id: linear_labels.copy()
        for layer_id in semmoe_layer_labels
    }


def round_robin_assignments(num_requests: int, num_devices: int) -> np.ndarray:
    return np.arange(num_requests, dtype=np.int64) % num_devices


def sem_moe_assignments(
    eval_data: PreparedEvalData,
    dp_score_full: np.ndarray,
) -> np.ndarray:
    request_scores = score_requests_from_token_scores(
        all_token_ids=eval_data.all_token_ids,
        req_offsets=eval_data.req_offsets,
        score_full=dp_score_full,
    )
    return assign_requests_from_scores(request_scores)


def score_requests_from_token_scores(
    all_token_ids: np.ndarray,
    req_offsets: np.ndarray,
    score_full: np.ndarray,
) -> np.ndarray:
    num_requests = int(req_offsets.shape[0])
    num_devices = int(score_full.shape[1])
    request_scores = np.empty((num_requests, num_devices), dtype=np.float32)
    for device_id in range(num_devices):
        token_scores = score_full[all_token_ids, device_id]
        request_scores[:, device_id] = np.add.reduceat(token_scores, req_offsets)
    return request_scores


@numba.njit(cache=True)
def _assign_requests_from_scores_numba(request_scores: np.ndarray) -> np.ndarray:
    num_requests = request_scores.shape[0]
    num_devices = request_scores.shape[1]
    assignments = np.empty((num_requests,), dtype=np.int64)
    dev_mask = np.ones((num_devices,), dtype=np.bool_)

    for request_idx in range(num_requests):
        best_device = 0
        best_score = -1e30
        for device_id in range(num_devices):
            if not dev_mask[device_id]:
                continue
            score = request_scores[request_idx, device_id]
            if score > best_score:
                best_score = score
                best_device = device_id

        assignments[request_idx] = best_device
        dev_mask[best_device] = False

        any_valid = False
        for device_id in range(num_devices):
            if dev_mask[device_id]:
                any_valid = True
                break
        if not any_valid:
            for device_id in range(num_devices):
                dev_mask[device_id] = True

    return assignments


def assign_requests_from_scores(request_scores: np.ndarray) -> np.ndarray:
    return _assign_requests_from_scores_numba(
        request_scores.astype(np.float32, copy=False)
    )


def request_device_hits_for_layer(
    routed_layer: np.ndarray,
    expert_labels: np.ndarray,
    req_offsets: np.ndarray,
    num_requests: int,
    num_devices: int,
) -> np.ndarray:
    device_ids = expert_labels[routed_layer]
    request_hits = np.empty((num_requests, num_devices), dtype=np.int64)
    for device_id in range(num_devices):
        token_hits = (device_ids == device_id).sum(axis=1, dtype=np.int64)
        request_hits[:, device_id] = np.add.reduceat(token_hits, req_offsets)
    return request_hits


def token_load_per_device_from_assignments(
    assignments: np.ndarray,
    req_lengths: np.ndarray,
    num_devices: int,
) -> np.ndarray:
    return np.bincount(
        assignments,
        weights=req_lengths,
        minlength=num_devices,
    ).astype(np.int64, copy=False)


def load_imbalance_from_token_load(token_load_per_device: np.ndarray) -> float:
    med = median(token_load_per_device.tolist())
    if med == 0:
        return float("inf") if int(token_load_per_device.max()) > 0 else 1.0
    return float(token_load_per_device.max() / med)


def evaluate_strategy(
    strategy_name: str,
    eval_data: PreparedEvalData,
    assignments: np.ndarray,
    layer_order: list[int],
    layer_labels: dict[int, np.ndarray],
    num_devices: int,
) -> dict[str, Any]:
    token_load_per_device = token_load_per_device_from_assignments(
        assignments=assignments,
        req_lengths=eval_data.req_lengths,
        num_devices=num_devices,
    )

    total_local = 0
    total_remote = 0
    per_layer: dict[str, dict[str, float]] = {}

    for layer_offset, layer_id in progress_iter(
        list(enumerate(layer_order)),
        total=len(layer_order),
        desc=f"Evaluating {strategy_name}",
    ):
        expert_labels = layer_labels[layer_id]
        routed_layer = eval_data.all_routed[:, layer_offset, :]
        request_hits = request_device_hits_for_layer(
            routed_layer=routed_layer,
            expert_labels=expert_labels,
            req_offsets=eval_data.req_offsets,
            num_requests=assignments.shape[0],
            num_devices=num_devices,
        )
        layer_local = int(
            request_hits[eval_data.request_indices, assignments].sum(dtype=np.int64)
        )
        layer_total = int(request_hits.sum(dtype=np.int64))
        layer_remote = layer_total - layer_local

        total_local += layer_local
        total_remote += layer_remote
        per_layer[str(layer_id)] = {
            "lar": float(layer_local / layer_total) if layer_total else 0.0,
            "local_activations": float(layer_local),
            "remote_activations": float(layer_remote),
            "all2all_volume_proxy": float(layer_remote),
        }

    total = total_local + total_remote
    return {
        "strategy": strategy_name,
        "lar": float(total_local / total) if total else 0.0,
        "local_activations": float(total_local),
        "remote_activations": float(total_remote),
        "all2all_volume_proxy": float(total_remote),
        "token_load_per_device": token_load_per_device.tolist(),
        "load_imbalance_rate": load_imbalance_from_token_load(token_load_per_device),
        "per_layer": per_layer,
    }


def evaluate_layerwise_diagnostic(
    strategy_name: str,
    eval_data: PreparedEvalData,
    layer_order: list[int],
    layer_labels: dict[int, np.ndarray],
    layer_scores_by_layer: dict[int, np.ndarray],
    num_devices: int,
) -> dict[str, Any]:
    total_local = 0
    total_remote = 0
    per_layer: dict[str, dict[str, Any]] = {}
    imbalance_values: list[float] = []

    for layer_offset, layer_id in progress_iter(
        list(enumerate(layer_order)),
        total=len(layer_order),
        desc=f"Evaluating {strategy_name}",
    ):
        request_scores = score_requests_from_token_scores(
            all_token_ids=eval_data.all_token_ids,
            req_offsets=eval_data.req_offsets,
            score_full=layer_scores_by_layer[layer_id],
        )
        assignments = assign_requests_from_scores(request_scores)
        token_load_per_device = token_load_per_device_from_assignments(
            assignments=assignments,
            req_lengths=eval_data.req_lengths,
            num_devices=num_devices,
        )
        imbalance = load_imbalance_from_token_load(token_load_per_device)
        imbalance_values.append(imbalance)

        request_hits = request_device_hits_for_layer(
            routed_layer=eval_data.all_routed[:, layer_offset, :],
            expert_labels=layer_labels[layer_id],
            req_offsets=eval_data.req_offsets,
            num_requests=assignments.shape[0],
            num_devices=num_devices,
        )
        layer_local = int(
            request_hits[eval_data.request_indices, assignments].sum(dtype=np.int64)
        )
        layer_total = int(request_hits.sum(dtype=np.int64))
        layer_remote = layer_total - layer_local

        total_local += layer_local
        total_remote += layer_remote
        per_layer[str(layer_id)] = {
            "lar": float(layer_local / layer_total) if layer_total else 0.0,
            "local_activations": float(layer_local),
            "remote_activations": float(layer_remote),
            "all2all_volume_proxy": float(layer_remote),
            "token_load_per_device": token_load_per_device.tolist(),
            "load_imbalance_rate": imbalance,
        }

    total = total_local + total_remote
    mean_imbalance = (
        float(np.mean(np.asarray(imbalance_values, dtype=np.float64)))
        if imbalance_values
        else 1.0
    )
    max_imbalance = (
        float(np.max(np.asarray(imbalance_values, dtype=np.float64)))
        if imbalance_values
        else 1.0
    )
    return {
        "strategy": strategy_name,
        "diagnostic_scope": "per_layer_request_assignment",
        "lar": float(total_local / total) if total else 0.0,
        "local_activations": float(total_local),
        "remote_activations": float(total_remote),
        "all2all_volume_proxy": float(total_remote),
        "load_imbalance_rate_mean": mean_imbalance,
        "load_imbalance_rate_max": max_imbalance,
        "per_layer": per_layer,
    }
