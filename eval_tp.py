from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from artifacts import (
    collection_manifest_path,
    load_json,
    schedule_layer_path,
    schedule_manifest_path,
)
from eval_dp import (
    EvalRequest,
    PreparedEvalData,
    build_linear_layer_labels,
    load_eval_requests,
    load_imbalance_from_token_load,
    prepare_eval_data,
)
from progress_utils import progress_iter


@dataclass(frozen=True)
class TPLayerSchedule:
    E: np.ndarray  # [num_experts] -> device_id
    T_full: np.ndarray  # [vocab_size] -> device_id
    Tp_full: np.ndarray  # [vocab_size] -> confidence
    A: np.ndarray  # [num_devices^lookback] -> device_id
    Ap: np.ndarray  # [num_devices^lookback] -> confidence


def run_evaluate_tp_scheduling(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    result = evaluate_tp_scheduling_from_run(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_tp_scheduling_from_run(
    run_dir: Path,
) -> dict[str, Any]:
    collection_manifest = load_json(collection_manifest_path(run_dir))
    schedule_manifest = load_json(schedule_manifest_path(run_dir))
    requests = load_eval_requests(run_dir, collection_manifest)
    if not requests:
        raise ValueError("No raw requests found; cannot evaluate Attention-TP scheduling.")

    layer_order = [int(layer_id) for layer_id in schedule_manifest["moe_layer_ids"]]
    num_devices = int(schedule_manifest["num_devices"])
    lookback = int(schedule_manifest["lookback"])
    eval_data = prepare_eval_data(requests)
    layer_schedules, layer_labels = load_tp_schedule_state(run_dir, schedule_manifest)
    baseline_layer_labels = build_linear_layer_labels(layer_labels, schedule_manifest)

    # Build baseline schedules using linear E but same T/Tp/A/Ap.
    baseline_schedules: dict[int, TPLayerSchedule] = {}
    for layer_id, sched in layer_schedules.items():
        baseline_schedules[layer_id] = TPLayerSchedule(
            E=baseline_layer_labels[layer_id],
            T_full=sched.T_full,
            Tp_full=sched.Tp_full,
            A=sched.A,
            Ap=sched.Ap,
        )

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model_name": schedule_manifest["model_name"],
        "num_devices": num_devices,
        "lookback": lookback,
        "request_count": len(requests),
        "total_tokens": int(eval_data.all_token_ids.shape[0]),
        "baseline": evaluate_tp_strategy(
            strategy_name="tp_baseline_t_only",
            eval_data=eval_data,
            layer_order=layer_order,
            layer_schedules=baseline_schedules,
            num_devices=num_devices,
            lookback=lookback,
            use_a_tables=False,
        ),
        "semmoe_t_only": evaluate_tp_strategy(
            strategy_name="semmoe_tp_t_only",
            eval_data=eval_data,
            layer_order=layer_order,
            layer_schedules=layer_schedules,
            num_devices=num_devices,
            lookback=lookback,
            use_a_tables=False,
        ),
        "semmoe_tp": evaluate_tp_strategy(
            strategy_name="semmoe_tp",
            eval_data=eval_data,
            layer_order=layer_order,
            layer_schedules=layer_schedules,
            num_devices=num_devices,
            lookback=lookback,
            use_a_tables=True,
        ),
    }
    return result


def load_tp_schedule_state(
    run_dir: Path,
    schedule_manifest: dict[str, Any],
) -> tuple[dict[int, TPLayerSchedule], dict[int, np.ndarray]]:
    layer_schedules: dict[int, TPLayerSchedule] = {}
    layer_labels: dict[int, np.ndarray] = {}

    for layer_id in schedule_manifest["moe_layer_ids"]:
        layer_id_int = int(layer_id)
        with np.load(schedule_layer_path(run_dir, layer_id_int)) as payload:
            E = payload["E"].astype(np.int64, copy=True)
            T_full = payload["T_full"].astype(np.int64, copy=True)
            Tp_full = payload["Tp_full"].astype(np.float32, copy=True)
            A = payload["A"].astype(np.int64, copy=True)
            Ap = payload["Ap"].astype(np.float32, copy=True)

        layer_labels[layer_id_int] = E
        layer_schedules[layer_id_int] = TPLayerSchedule(
            E=E, T_full=T_full, Tp_full=Tp_full, A=A, Ap=Ap,
        )

    return layer_schedules, layer_labels


def encode_seq_ids_vectorized(
    device_trace_window: np.ndarray,
    num_devices: int,
) -> np.ndarray:
    """Encode lookback device sequences into seq_ids for all tokens at once.

    device_trace_window: [total_tokens, lookback]
    Returns: [total_tokens] int64 seq_ids.
    """
    lookback = device_trace_window.shape[1]
    seq_ids = device_trace_window[:, 0].astype(np.int64, copy=True)
    for lb in range(1, lookback):
        seq_ids = seq_ids * num_devices + device_trace_window[:, lb]
    return seq_ids


def evaluate_tp_strategy(
    strategy_name: str,
    eval_data: PreparedEvalData,
    layer_order: list[int],
    layer_schedules: dict[int, TPLayerSchedule],
    num_devices: int,
    lookback: int,
    use_a_tables: bool,
) -> dict[str, Any]:
    total_tokens = eval_data.all_token_ids.shape[0]
    num_layers = len(layer_order)
    device_trace = np.zeros((total_tokens, num_layers), dtype=np.int64)

    total_local = 0
    total_remote = 0
    per_layer: dict[str, dict[str, Any]] = {}
    imbalance_values: list[float] = []

    for layer_offset, layer_id in progress_iter(
        list(enumerate(layer_order)),
        total=num_layers,
        desc=f"Evaluating {strategy_name}",
    ):
        sched = layer_schedules[layer_id]

        # Step 1: Determine target device for every token.
        if layer_offset < lookback or not use_a_tables:
            target_devices = sched.T_full[eval_data.all_token_ids]
        else:
            window = device_trace[:, (layer_offset - lookback):layer_offset]
            seq_ids = encode_seq_ids_vectorized(window, num_devices)
            tp_conf = sched.Tp_full[eval_data.all_token_ids]
            ap_conf = sched.Ap[seq_ids]
            t_dev = sched.T_full[eval_data.all_token_ids]
            a_dev = sched.A[seq_ids]
            target_devices = np.where(tp_conf > ap_conf, t_dev, a_dev)

        # Step 2: Record in trace.
        device_trace[:, layer_offset] = target_devices

        # Step 3: Count local vs remote expert activations.
        routed_layer = eval_data.all_routed[:, layer_offset, :]  # [total_tokens, top_k]
        expert_device_ids = sched.E[routed_layer]  # [total_tokens, top_k]
        local_mask = expert_device_ids == target_devices[:, np.newaxis]

        layer_local = int(local_mask.sum())
        layer_total = int(routed_layer.size)
        layer_remote = layer_total - layer_local

        total_local += layer_local
        total_remote += layer_remote

        # Token-level load balance for this layer.
        token_load = np.bincount(target_devices, minlength=num_devices).astype(
            np.int64, copy=False
        )
        imbalance = load_imbalance_from_token_load(token_load)
        imbalance_values.append(imbalance)

        per_layer[str(layer_id)] = {
            "lar": float(layer_local / layer_total) if layer_total else 0.0,
            "local_activations": float(layer_local),
            "remote_activations": float(layer_remote),
            "all2all_volume_proxy": float(layer_remote),
            "token_load_per_device": token_load.tolist(),
            "token_load_imbalance": imbalance,
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
        "lar": float(total_local / total) if total else 0.0,
        "local_activations": float(total_local),
        "remote_activations": float(total_remote),
        "all2all_volume_proxy": float(total_remote),
        "token_load_imbalance_mean": mean_imbalance,
        "token_load_imbalance_max": max_imbalance,
        "per_layer": per_layer,
    }
