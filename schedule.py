from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from progress_utils import progress_iter
from artifacts import (
    collection_manifest_path,
    ensure_schedule_dir,
    load_json,
    load_torch_artifact,
    schedule_layer_path,
    schedule_manifest_path,
    stats_artifact_path,
    vocab_artifact_path,
    write_json,
)


@dataclass(frozen=True)
class ScheduleBuildConfig:
    run_dir: Path
    num_devices: int
    lookback: int
    seed: int
    n_steps: int
    ft_steps: int
    alpha_e: float
    beta_e: float
    gamma_e: float
    alpha_r: float
    beta_r: float
    theta: float
    show_progress: bool


@dataclass(frozen=True)
class ProfileRequest:
    record_id: str
    token_indices: torch.Tensor
    routed_experts: torch.Tensor


def run_build_model_schedule(args: Any) -> None:
    config = ScheduleBuildConfig(
        run_dir=Path(args.run_dir).resolve(),
        num_devices=args.num_devices,
        lookback=args.lookback,
        seed=args.seed,
        n_steps=args.n_steps,
        ft_steps=args.ft_steps,
        alpha_e=args.alpha_e,
        beta_e=args.beta_e,
        gamma_e=args.gamma_e,
        alpha_r=args.alpha_r,
        beta_r=args.beta_r,
        theta=args.theta,
        show_progress=not getattr(args, "no_progress", False),
    )
    _validate_schedule_config(config)

    run_dir = config.run_dir
    manifest = load_json(collection_manifest_path(run_dir))
    stats_artifact = load_torch_artifact(stats_artifact_path(run_dir))
    vocab_artifact = load_torch_artifact(vocab_artifact_path(run_dir))
    ensure_schedule_dir(run_dir)

    schedule_manifest, layer_payloads = build_model_schedule_from_run(
        run_dir=run_dir,
        collection_manifest=manifest,
        stats_artifact=stats_artifact,
        vocab_artifact=vocab_artifact,
        config=config,
    )

    for layer_id, payload in layer_payloads.items():
        _save_npz(schedule_layer_path(run_dir, layer_id), payload)
    write_json(schedule_manifest_path(run_dir), schedule_manifest)
    print(f"Built model schedule artifacts under {schedule_manifest_path(run_dir).parent}")


def _validate_schedule_config(config: ScheduleBuildConfig) -> None:
    if config.num_devices <= 0:
        raise ValueError("--num-devices must be positive.")
    if config.lookback <= 0:
        raise ValueError("--lookback must be positive.")
    if config.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if config.ft_steps < 0:
        raise ValueError("--ft-steps must be non-negative.")
    if not 0.0 <= config.theta <= 1.0:
        raise ValueError("--theta must be in the range [0, 1].")


def build_model_schedule_from_run(
    run_dir: Path,
    collection_manifest: dict[str, Any],
    stats_artifact: dict[str, Any],
    vocab_artifact: dict[str, Any],
    config: ScheduleBuildConfig,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    metadata = stats_artifact["metadata"]
    num_experts = int(metadata["num_experts"])
    if num_experts % config.num_devices != 0:
        raise ValueError(
            f"num_experts={num_experts} is not divisible by num_devices={config.num_devices}."
        )

    seen_token_ids = stats_artifact["seen_token_ids"].to(torch.long)
    requests = load_profile_requests(
        run_dir=run_dir,
        collection_manifest=collection_manifest,
        seen_token_ids=seen_token_ids,
        show_progress=config.show_progress,
    )
    if not requests:
        raise ValueError("No profile requests found in raw shards; cannot build a model schedule.")

    moe_layer_ids = [int(layer_id) for layer_id in metadata["moe_layer_ids"]]
    solved_layers: dict[int, dict[str, torch.Tensor]] = {}

    layer_iterator = progress_iter(
        list(enumerate(moe_layer_ids)),
        total=len(moe_layer_ids),
        desc="Solving layers",
        enabled=config.show_progress,
    )
    for layer_offset, layer_id in layer_iterator:
        cp, a = dense_layer_statistics(
            layer_artifact=stats_artifact["layers"][str(layer_id)],
            num_seen_tokens=int(seen_token_ids.numel()),
            num_experts=num_experts,
        )
        req_profiles, req_lengths = build_request_profiles(cp, requests)
        expert_labels, request_labels, t_score_seen, objective = solve_layer_schedule(
            cp=cp,
            a=a,
            req_profiles=req_profiles,
            req_lengths=req_lengths,
            requests=requests,
            config=config,
            layer_seed=config.seed + layer_offset * 9973,
            layer_id=layer_id,
        )
        t_seen, tp_seen = schedule_labels_and_confidence(t_score_seen)
        t_score_full = extend_token_scores_to_full_vocab(
            t_score_seen=t_score_seen,
            seen_token_ids=seen_token_ids,
            vocab_artifact=vocab_artifact,
        )
        t_full, tp_full = schedule_labels_and_confidence(t_score_full)
        expert_perm = build_expert_permutation(expert_labels, config.num_devices)
        expert_inv_perm = invert_permutation(expert_perm)

        solved_layers[layer_id] = {
            "E": expert_labels.to(torch.int32),
            "expert_permutation": expert_perm.to(torch.int32),
            "expert_inverse_permutation": expert_inv_perm.to(torch.int32),
            "gating_column_permutation": expert_perm.to(torch.int32),
            "gating_column_inverse_permutation": expert_inv_perm.to(torch.int32),
            "token_ids_seen": seen_token_ids.to(torch.int32),
            "T_score_seen": t_score_seen.to(torch.float32),
            "T_seen": t_seen.to(torch.int32),
            "Tp_seen": tp_seen.to(torch.float32),
            "T_score_full": t_score_full.to(torch.float32),
            "T_full": t_full.to(torch.int32),
            "Tp_full": tp_full.to(torch.float32),
            "objective": torch.tensor(objective, dtype=torch.float32),
        }

    a_prob_tables = build_activation_transition_tables(
        requests=requests,
        moe_layer_ids=moe_layer_ids,
        expert_labels_by_layer={layer_id: solved_layers[layer_id]["E"].to(torch.long) for layer_id in moe_layer_ids},
        num_devices=config.num_devices,
        lookback=config.lookback,
        show_progress=config.show_progress,
    )

    layer_payloads: dict[int, dict[str, np.ndarray]] = {}
    relative_layer_paths: list[str] = []
    for layer_id in moe_layer_ids:
        a_prob = a_prob_tables[layer_id]
        a_table, ap_table = schedule_labels_and_confidence(a_prob)
        payload = dict(solved_layers[layer_id])
        payload["A_prob"] = a_prob.to(torch.float32)
        payload["A"] = a_table.to(torch.int32)
        payload["Ap"] = ap_table.to(torch.float32)
        layer_payloads[layer_id] = {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in payload.items()
        }
        relative_layer_paths.append(str(schedule_layer_path(run_dir, layer_id).relative_to(run_dir)))

    schedule_manifest = {
        "format_version": int(collection_manifest.get("format_version", 1)),
        "paper_spec": {
            "conference_entry": "iclr2026_conference.tex",
            "method": "method_zhang.tex",
            "appendix": "motivation_zhang.tex",
        },
        "run_dir": str(run_dir),
        "run_name": collection_manifest["run_name"],
        "model_name": metadata["model_name"],
        "moe_layer_ids": moe_layer_ids,
        "num_devices": config.num_devices,
        "lookback": config.lookback,
        "effective_vocab_size": int(vocab_artifact["effective_vocab_size"]),
        "full_vocab_size": int(vocab_artifact["full_vocab_size"]),
        "source_artifacts": {
            "collection_manifest": str(collection_manifest_path(run_dir).relative_to(run_dir)),
            "stats_artifact": str(stats_artifact_path(run_dir).relative_to(run_dir)),
            "vocab_artifact": str(vocab_artifact_path(run_dir).relative_to(run_dir)),
            "raw_shards": list(collection_manifest["raw_shards"]),
        },
        "solver_hyperparameters": {
            "alpha_e": config.alpha_e,
            "beta_e": config.beta_e,
            "gamma_e": config.gamma_e,
            "alpha_r": config.alpha_r,
            "beta_r": config.beta_r,
            "theta": config.theta,
            "n_steps": config.n_steps,
            "ft_steps": config.ft_steps,
            "seed": config.seed,
        },
        "layers": relative_layer_paths,
    }
    return schedule_manifest, layer_payloads


def dense_layer_statistics(
    layer_artifact: dict[str, torch.Tensor],
    num_seen_tokens: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cp = torch.zeros((num_seen_tokens, num_experts), dtype=torch.float32)
    row_splits = layer_artifact["row_splits"].to(torch.long)
    expert_ids = layer_artifact["count_expert_ids"].to(torch.long)
    cp_values = layer_artifact["cp_values"].to(torch.float32)
    for token_index in range(num_seen_tokens):
        start = int(row_splits[token_index])
        end = int(row_splits[token_index + 1])
        if start == end:
            continue
        cp[token_index, expert_ids[start:end]] = cp_values[start:end]

    if "a_values" in layer_artifact:
        a = layer_artifact["a_values"].to(torch.float32)
    else:
        freq = layer_artifact["freq"].to(torch.float32)
        freq_total = float(freq.sum())
        if freq_total <= 0.0:
            raise ValueError("Encountered an empty freq vector while building a schedule.")
        a = freq / freq_total
    return cp, a


def load_profile_requests(
    run_dir: Path,
    collection_manifest: dict[str, Any],
    seen_token_ids: torch.Tensor,
    show_progress: bool = True,
) -> list[ProfileRequest]:
    token_to_index = {
        int(token_id): token_index for token_index, token_id in enumerate(seen_token_ids.tolist())
    }
    requests: list[ProfileRequest] = []
    shard_paths = list(collection_manifest["raw_shards"])
    for relative_path in progress_iter(
        shard_paths,
        total=len(shard_paths),
        desc="Loading raw shards",
        enabled=show_progress,
    ):
        shard = load_torch_artifact(run_dir / relative_path)
        for record in shard["records"]:
            prompt_token_ids = record["prompt_token_ids"].to(torch.long)
            token_indices = torch.tensor(
                [token_to_index[int(token_id)] for token_id in prompt_token_ids.tolist()],
                dtype=torch.long,
            )
            requests.append(
                ProfileRequest(
                    record_id=str(record["record_id"]),
                    token_indices=token_indices,
                    routed_experts=record["routed_experts"].to(torch.long),
                )
            )
    return requests


def build_request_profiles(
    cp: torch.Tensor,
    requests: list[ProfileRequest],
) -> tuple[torch.Tensor, torch.Tensor]:
    req_profiles = torch.stack(
        [cp.index_select(0, request.token_indices).sum(dim=0) for request in requests],
        dim=0,
    )
    req_lengths = torch.tensor(
        [int(request.token_indices.numel()) for request in requests],
        dtype=torch.long,
    )
    return req_profiles, req_lengths


def solve_layer_schedule(
    cp: torch.Tensor,
    a: torch.Tensor,
    req_profiles: torch.Tensor,
    req_lengths: torch.Tensor,
    requests: list[ProfileRequest],
    config: ScheduleBuildConfig,
    layer_seed: int,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(layer_seed)

    expert_load = torch.matmul(a, cp)
    expert_seed = init_expert_seed_assignment(expert_load, config.num_devices)
    best_expert_labels = expert_seed.clone()
    best_request_labels = request_schedule(
        req_profiles=req_profiles,
        req_lengths=req_lengths,
        expert_labels=expert_seed,
        num_devices=config.num_devices,
        alpha_r=config.alpha_r,
        beta_r=config.beta_r,
    )
    best_t_score = build_token_cluster_scores(
        num_seen_tokens=cp.shape[0],
        num_devices=config.num_devices,
        requests=requests,
        request_labels=best_request_labels,
    )
    best_objective = compute_schedule_objective(
        a=a,
        cp=cp,
        token_cluster_scores=best_t_score,
        expert_labels=best_expert_labels,
        num_devices=config.num_devices,
        theta=config.theta,
    )

    current_expert_labels = best_expert_labels
    current_request_labels = best_request_labels
    weighted_cp = cp * a.unsqueeze(1)
    expert_affinity = torch.matmul(weighted_cp.T, weighted_cp)

    step_iterator = progress_iter(
        range(config.n_steps),
        total=config.n_steps,
        desc=f"Layer {layer_id} steps",
        enabled=config.show_progress,
        leave=False,
    )
    for _ in step_iterator:
        current_expert_labels = expert_place(
            req_profiles=req_profiles,
            expert_load=expert_load,
            expert_affinity=expert_affinity,
            request_labels=current_request_labels,
            num_devices=config.num_devices,
            alpha_e=config.alpha_e,
            beta_e=config.beta_e,
            gamma_e=config.gamma_e,
            ft_steps=config.ft_steps,
            generator=generator,
            a=a,
            cp=cp,
            requests=requests,
            theta=config.theta,
        )
        current_request_labels = request_schedule(
            req_profiles=req_profiles,
            req_lengths=req_lengths,
            expert_labels=current_expert_labels,
            num_devices=config.num_devices,
            alpha_r=config.alpha_r,
            beta_r=config.beta_r,
        )
        t_score = build_token_cluster_scores(
            num_seen_tokens=cp.shape[0],
            num_devices=config.num_devices,
            requests=requests,
            request_labels=current_request_labels,
        )
        objective = compute_schedule_objective(
            a=a,
            cp=cp,
            token_cluster_scores=t_score,
            expert_labels=current_expert_labels,
            num_devices=config.num_devices,
            theta=config.theta,
        )
        if objective < best_objective:
            best_objective = objective
            best_expert_labels = current_expert_labels.clone()
            best_request_labels = current_request_labels.clone()
            best_t_score = t_score.clone()

    return best_expert_labels, best_request_labels, best_t_score, float(best_objective)


def init_expert_seed_assignment(expert_load: torch.Tensor, num_devices: int) -> torch.Tensor:
    num_experts = int(expert_load.numel())
    experts_per_device = num_experts // num_devices
    order = torch.argsort(expert_load, descending=True)
    labels = torch.full((num_experts,), -1, dtype=torch.long)
    device_load = torch.zeros(num_devices, dtype=torch.float32)
    device_count = torch.zeros(num_devices, dtype=torch.long)
    for expert_id in order.tolist():
        valid = device_count < experts_per_device
        scores = device_load.clone()
        scores[~valid] = torch.inf
        device_id = int(torch.argmin(scores))
        labels[expert_id] = device_id
        device_count[device_id] += 1
        device_load[device_id] += float(expert_load[expert_id])
    return labels


def request_schedule(
    req_profiles: torch.Tensor,
    req_lengths: torch.Tensor,
    expert_labels: torch.Tensor,
    num_devices: int,
    alpha_r: float,
    beta_r: float,
) -> torch.Tensor:
    num_requests = int(req_profiles.shape[0])
    quota = (num_requests + num_devices - 1) // num_devices
    request_order = torch.argsort(req_lengths, descending=True)
    req_norm = F.normalize(req_profiles, dim=1, eps=1e-12)
    cluster_expert_mask = F.one_hot(expert_labels, num_classes=num_devices).to(torch.float32).T

    labels = torch.full((num_requests,), -1, dtype=torch.long)
    cluster_req_norm_sum = torch.zeros((num_devices, req_profiles.shape[1]), dtype=torch.float32)
    cluster_count = torch.zeros(num_devices, dtype=torch.long)

    for request_id in request_order.tolist():
        valid = cluster_count < quota
        if not bool(valid.any()):
            valid = torch.ones(num_devices, dtype=torch.bool)

        rafr = torch.zeros(num_devices, dtype=torch.float32)
        active = cluster_count > 0
        if bool(active.any()):
            centroid = cluster_req_norm_sum[active] / cluster_count[active].unsqueeze(1).to(
                torch.float32
            )
            rafr[active] = torch.mv(centroid, req_norm[request_id])
        rafe = torch.mv(cluster_expert_mask, req_profiles[request_id])
        score = alpha_r * rafr + beta_r * rafe
        score[~valid] = -torch.inf
        device_id = int(torch.argmax(score))
        labels[request_id] = device_id
        cluster_req_norm_sum[device_id] += req_norm[request_id]
        cluster_count[device_id] += 1

    return labels


def expert_place(
    req_profiles: torch.Tensor,
    expert_load: torch.Tensor,
    expert_affinity: torch.Tensor,
    request_labels: torch.Tensor,
    num_devices: int,
    alpha_e: float,
    beta_e: float,
    gamma_e: float,
    ft_steps: int,
    generator: torch.Generator,
    a: torch.Tensor,
    cp: torch.Tensor,
    requests: list[ProfileRequest],
    theta: float,
) -> torch.Tensor:
    num_experts = int(expert_load.numel())
    experts_per_device = num_experts // num_devices
    labels = torch.full((num_experts,), -1, dtype=torch.long)
    cluster_count = torch.zeros(num_devices, dtype=torch.long)
    cluster_load = torch.zeros(num_devices, dtype=torch.float32)
    cluster_affinity = torch.zeros((num_devices, num_experts), dtype=torch.float32)
    cluster_req_sum = torch.zeros((num_devices, num_experts), dtype=torch.float32)

    for device_id in range(num_devices):
        req_mask = request_labels == device_id
        if bool(req_mask.any()):
            cluster_req_sum[device_id] = req_profiles[req_mask].sum(dim=0)

    expert_order = torch.argsort(expert_load, descending=True)
    for expert_id in expert_order.tolist():
        valid = cluster_count < experts_per_device
        eafe = cluster_affinity[:, expert_id]
        eafr = cluster_req_sum[:, expert_id]
        score = alpha_e * eafe + beta_e * eafr - gamma_e * cluster_load
        score[~valid] = -torch.inf
        device_id = int(torch.argmax(score))
        labels[expert_id] = device_id
        cluster_count[device_id] += 1
        cluster_load[device_id] += float(expert_load[expert_id])
        cluster_affinity[device_id] += expert_affinity[:, expert_id]

    # token_cluster_scores is invariant across ft_steps (request_labels doesn't change)
    token_cluster_scores = build_token_cluster_scores(
        num_seen_tokens=cp.shape[0],
        num_devices=num_devices,
        requests=requests,
        request_labels=request_labels,
    )

    best_labels = labels.clone()
    best_score = compute_schedule_objective(
        a=a,
        cp=cp,
        token_cluster_scores=token_cluster_scores,
        expert_labels=labels,
        num_devices=num_devices,
        theta=theta,
    )

    for _ in range(ft_steps):
        cluster_1, cluster_2 = torch.randperm(num_devices, generator=generator)[:2].tolist()
        experts_1 = torch.nonzero(labels == cluster_1, as_tuple=False).flatten()
        experts_2 = torch.nonzero(labels == cluster_2, as_tuple=False).flatten()
        if experts_1.numel() == 0 or experts_2.numel() == 0:
            continue
        expert_1 = int(experts_1[torch.randint(experts_1.numel(), (1,), generator=generator)])
        expert_2 = int(experts_2[torch.randint(experts_2.numel(), (1,), generator=generator)])
        swapped = labels.clone()
        label_1 = int(swapped[expert_1])
        label_2 = int(swapped[expert_2])
        swapped[expert_1] = label_2
        swapped[expert_2] = label_1
        score = compute_schedule_objective(
            a=a,
            cp=cp,
            token_cluster_scores=token_cluster_scores,
            expert_labels=swapped,
            num_devices=num_devices,
            theta=theta,
        )
        if score < best_score:
            best_score = score
            best_labels = swapped
            labels = swapped

    return best_labels


def build_token_cluster_scores(
    num_seen_tokens: int,
    num_devices: int,
    requests: list[ProfileRequest],
    request_labels: torch.Tensor,
) -> torch.Tensor:
    counts = torch.zeros((num_seen_tokens, num_devices), dtype=torch.float32)
    for request_id, request in enumerate(requests):
        device_id = int(request_labels[request_id])
        updates = torch.ones_like(request.token_indices, dtype=torch.float32)
        counts[:, device_id].index_add_(0, request.token_indices, updates)

    totals = counts.sum(dim=1, keepdim=True)
    scores = torch.zeros_like(counts)
    seen = totals.squeeze(1) > 0
    if bool(seen.any()):
        scores[seen] = counts[seen] / totals[seen]
    return scores


def compute_schedule_objective(
    a: torch.Tensor,
    cp: torch.Tensor,
    token_cluster_scores: torch.Tensor,
    expert_labels: torch.Tensor,
    num_devices: int,
    theta: float,
) -> float:
    token_load_by_device = (token_cluster_scores * a.unsqueeze(1)).sum(dim=0)
    target = float(a.sum()) / float(num_devices)
    token_load_imbalance = torch.abs(token_load_by_device - target).sum()

    expert_mask = F.one_hot(expert_labels, num_classes=num_devices).to(torch.float32)
    token_device_affinity = torch.matmul(cp, expert_mask)
    local_activation = (token_cluster_scores * token_device_affinity).sum(dim=1)
    remote_activation_cost = torch.sum(a * (1.0 - local_activation))
    objective = theta * token_load_imbalance + (1.0 - theta) * remote_activation_cost
    return float(objective)


def schedule_labels_and_confidence(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    confidence, labels = scores.max(dim=1)
    return labels.to(torch.long), confidence.to(torch.float32)


def extend_token_scores_to_full_vocab(
    t_score_seen: torch.Tensor,
    seen_token_ids: torch.Tensor,
    vocab_artifact: dict[str, torch.Tensor],
) -> torch.Tensor:
    nearest_seen_token_id = vocab_artifact["nearest_seen_token_id"].to(torch.long)
    seen_indices = torch.searchsorted(seen_token_ids, nearest_seen_token_id)
    return t_score_seen.index_select(0, seen_indices)


def build_activation_transition_tables(
    requests: list[ProfileRequest],
    moe_layer_ids: list[int],
    expert_labels_by_layer: dict[int, torch.Tensor],
    num_devices: int,
    lookback: int,
    show_progress: bool = True,
) -> dict[int, torch.Tensor]:
    seq_count = num_devices ** lookback
    counts_by_layer = {
        layer_id: torch.zeros((seq_count, num_devices), dtype=torch.float32)
        for layer_id in moe_layer_ids
    }

    for request in progress_iter(
        requests,
        total=len(requests),
        desc="Building A_prob",
        enabled=show_progress,
    ):
        token_count = int(request.token_indices.numel())
        device_trace = torch.zeros((token_count, len(moe_layer_ids)), dtype=torch.long)
        for layer_offset, layer_id in enumerate(moe_layer_ids):
            layer_expert_labels = expert_labels_by_layer[layer_id]
            routed = request.routed_experts[:, layer_id].to(torch.long)
            for token_index in range(token_count):
                device_trace[token_index, layer_offset] = majority_vote_device(
                    layer_expert_labels.index_select(0, routed[token_index]),
                    num_devices=num_devices,
                )

        for layer_offset, layer_id in enumerate(moe_layer_ids):
            if layer_offset < lookback:
                continue
            for token_index in range(token_count):
                seq_id = encode_device_sequence(
                    device_trace[token_index, layer_offset - lookback : layer_offset].tolist(),
                    base=num_devices,
                )
                next_device = int(device_trace[token_index, layer_offset])
                counts_by_layer[layer_id][seq_id, next_device] += 1.0

    tables: dict[int, torch.Tensor] = {}
    for layer_id, counts in counts_by_layer.items():
        totals = counts.sum(dim=1, keepdim=True)
        table = torch.zeros_like(counts)
        seen = totals.squeeze(1) > 0
        if bool(seen.any()):
            table[seen] = counts[seen] / totals[seen]
        tables[layer_id] = table
    return tables


def majority_vote_device(device_ids: torch.Tensor, num_devices: int) -> torch.Tensor:
    counts = torch.bincount(device_ids.to(torch.long), minlength=num_devices)
    return torch.argmax(counts)


def encode_device_sequence(sequence: list[int], base: int) -> int:
    seq_id = 0
    for device_id in sequence:
        seq_id = seq_id * base + int(device_id)
    return seq_id


def build_expert_permutation(expert_labels: torch.Tensor, num_devices: int) -> torch.Tensor:
    groups = [torch.nonzero(expert_labels == device_id, as_tuple=False).flatten() for device_id in range(num_devices)]
    return torch.cat(groups, dim=0)


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), dtype=permutation.dtype)
    return inverse


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)
