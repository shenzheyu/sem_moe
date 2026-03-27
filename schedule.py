from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numba
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


@dataclass
class RequestBatch:
    """Precomputed flat representation for vectorized operations over requests."""

    all_token_indices: torch.Tensor  # [total_tokens] — concatenated token indices
    request_id_per_token: torch.Tensor  # [total_tokens] — owning request of each token
    req_lengths: torch.Tensor  # [num_requests]
    num_requests: int
    requests: list[ProfileRequest]  # kept for routed_experts access

    @staticmethod
    def from_requests(requests: list[ProfileRequest]) -> "RequestBatch":
        # Use numpy concat (much faster than torch.cat for many small tensors).
        all_indices = torch.from_numpy(
            np.concatenate([r.token_indices.numpy() for r in requests])
        ).to(torch.long)
        lengths = torch.tensor(
            [r.token_indices.numel() for r in requests], dtype=torch.long
        )
        request_ids = torch.repeat_interleave(
            torch.arange(len(requests), dtype=torch.long), lengths
        )
        return RequestBatch(
            all_token_indices=all_indices,
            request_id_per_token=request_ids,
            req_lengths=lengths,
            num_requests=len(requests),
            requests=requests,
        )


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


# Module-level shared state for parallel layer solving (populated before fork).
_layer_worker_state: dict[str, Any] = {}


def _solve_single_layer_star(args: tuple[int, int]) -> tuple[int, dict[str, torch.Tensor]]:
    """Wrapper for imap_unordered (accepts single tuple argument)."""
    return _solve_single_layer(*args)


def _solve_single_layer(
    layer_offset: int, layer_id: int
) -> tuple[int, dict[str, torch.Tensor]]:
    """Module-level worker function for parallel layer solving."""
    s = _layer_worker_state
    stats_artifact = s["stats_artifact"]
    seen_token_ids = s["seen_token_ids"]
    vocab_artifact = s["vocab_artifact"]
    batch = s["batch"]
    config = s["config"]
    num_experts = s["num_experts"]

    cp, a = dense_layer_statistics(
        layer_artifact=stats_artifact["layers"][str(layer_id)],
        num_seen_tokens=int(seen_token_ids.numel()),
        num_experts=num_experts,
    )
    req_profiles = build_request_profiles(cp, batch)
    expert_labels, _, t_score_seen, objective = solve_layer_schedule(
        cp=cp,
        a=a,
        req_profiles=req_profiles,
        batch=batch,
        config=ScheduleBuildConfig(
            run_dir=config.run_dir,
            num_devices=config.num_devices,
            lookback=config.lookback,
            seed=config.seed,
            n_steps=config.n_steps,
            ft_steps=config.ft_steps,
            alpha_e=config.alpha_e,
            beta_e=config.beta_e,
            gamma_e=config.gamma_e,
            alpha_r=config.alpha_r,
            beta_r=config.beta_r,
            theta=config.theta,
            show_progress=False,
        ),
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

    return layer_id, {
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

    batch = RequestBatch.from_requests(requests)

    moe_layer_ids = [int(layer_id) for layer_id in metadata["moe_layer_ids"]]

    num_workers = 1

    # Set up shared state for parallel workers (fork will inherit these).
    _layer_worker_state.update(
        stats_artifact=stats_artifact,
        seen_token_ids=seen_token_ids,
        vocab_artifact=vocab_artifact,
        batch=batch,
        config=config,
        num_experts=num_experts,
    )

    solved_layers: dict[int, dict[str, torch.Tensor]] = {}

    if num_workers > 1:
        import torch.multiprocessing as mp

        ctx = mp.get_context("fork")
        print(f"Solving {len(moe_layer_ids)} layers with {num_workers} parallel workers")
        tasks = list(enumerate(moe_layer_ids))
        with ctx.Pool(num_workers) as pool:
            for layer_id, layer_result in progress_iter(
                pool.imap_unordered(_solve_single_layer_star, tasks),
                total=len(tasks),
                desc="Solving layers",
                enabled=config.show_progress,
            ):
                solved_layers[layer_id] = layer_result
    else:
        layer_iterator = progress_iter(
            list(enumerate(moe_layer_ids)),
            total=len(moe_layer_ids),
            desc="Solving layers",
            enabled=config.show_progress,
        )
        for layer_offset, layer_id in layer_iterator:
            _, layer_result = _solve_single_layer(layer_offset, layer_id)
            solved_layers[layer_id] = layer_result

    _layer_worker_state.clear()

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
    # Build a dense lookup table: index_lut[token_id] -> token_index.
    max_token_id = int(seen_token_ids.max().item())
    index_lut = torch.full((max_token_id + 1,), -1, dtype=torch.long)
    index_lut[seen_token_ids.to(torch.long)] = torch.arange(
        seen_token_ids.numel(), dtype=torch.long
    )

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
            token_indices = index_lut[prompt_token_ids]
            requests.append(
                ProfileRequest(
                    record_id=str(record["record_id"]),
                    token_indices=token_indices,
                    # Keep as int32 to avoid expensive copy; cast to long on demand.
                    routed_experts=record["routed_experts"],
                )
            )
    return requests


def build_request_profiles(
    cp: torch.Tensor,
    batch: RequestBatch,
) -> torch.Tensor:
    """Build per-request Cp profiles via chunked scatter_add.

    Processes tokens in chunks to avoid allocating a [total_tokens, num_experts]
    intermediate tensor (~76 GB for 78M tokens × 256 experts).
    """
    num_experts = cp.shape[1]
    req_profiles = torch.zeros(batch.num_requests, num_experts, dtype=torch.float32)
    total_tokens = batch.all_token_indices.shape[0]
    chunk = 1_000_000  # ~1M tokens × 256 × 4 bytes ≈ 1 GB per chunk
    for start in range(0, total_tokens, chunk):
        end = min(start + chunk, total_tokens)
        idx = batch.all_token_indices[start:end]
        rid = batch.request_id_per_token[start:end]
        chunk_cp = cp[idx]  # [chunk, num_experts]
        req_profiles.scatter_add_(
            0, rid.unsqueeze(1).expand_as(chunk_cp), chunk_cp
        )
    return req_profiles


def solve_layer_schedule(
    cp: torch.Tensor,
    a: torch.Tensor,
    req_profiles: torch.Tensor,
    batch: RequestBatch,
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
        req_lengths=batch.req_lengths,
        expert_labels=expert_seed,
        num_devices=config.num_devices,
        alpha_r=config.alpha_r,
        beta_r=config.beta_r,
    )
    best_t_score = build_token_cluster_scores(
        num_seen_tokens=cp.shape[0],
        num_devices=config.num_devices,
        batch=batch,
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
    expert_affinity = torch.matmul(cp.T, cp * a.unsqueeze(1))

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
            batch=batch,
            theta=config.theta,
        )
        current_request_labels = request_schedule(
            req_profiles=req_profiles,
            req_lengths=batch.req_lengths,
            expert_labels=current_expert_labels,
            num_devices=config.num_devices,
            alpha_r=config.alpha_r,
            beta_r=config.beta_r,
        )
        t_score = build_token_cluster_scores(
            num_seen_tokens=cp.shape[0],
            num_devices=config.num_devices,
            batch=batch,
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
    """Assign requests to devices — sequential greedy with Numba-JIT inner loop."""
    num_requests = int(req_profiles.shape[0])
    quota = (num_requests + num_devices - 1) // num_devices
    request_order = torch.argsort(req_lengths, descending=True).numpy()
    req_norm = F.normalize(req_profiles, dim=1, eps=1e-12).numpy()

    cluster_expert_mask = F.one_hot(expert_labels, num_classes=num_devices).to(torch.float32).T
    all_rafe = torch.matmul(req_profiles, cluster_expert_mask.T).numpy()

    labels = _request_schedule_numba(
        request_order, req_norm, all_rafe, num_devices, quota, alpha_r, beta_r
    )
    return torch.from_numpy(labels)


@numba.njit(cache=True)
def _request_schedule_numba(
    request_order: np.ndarray,
    req_norm: np.ndarray,
    all_rafe: np.ndarray,
    num_devices: int,
    quota: int,
    alpha_r: float,
    beta_r: float,
) -> np.ndarray:
    """Numba-compiled inner loop for request scheduling."""
    num_requests = req_norm.shape[0]
    num_experts = req_norm.shape[1]
    labels = np.full(num_requests, -1, dtype=np.int64)
    cluster_req_norm_sum = np.zeros((num_devices, num_experts), dtype=np.float32)
    cluster_count = np.zeros(num_devices, dtype=np.int64)
    _eps = 1e-12

    for idx in range(request_order.shape[0]):
        request_id = request_order[idx]

        # Check quota.
        any_valid = False
        for d in range(num_devices):
            if cluster_count[d] < quota:
                any_valid = True
                break
        if not any_valid:
            for d in range(num_devices):
                cluster_count[d] = 0

        # RAFR: centroid similarity.
        rafr = np.zeros(num_devices, dtype=np.float32)
        for d in range(num_devices):
            if cluster_count[d] > 0:
                dot = np.float32(0.0)
                for j in range(num_experts):
                    dot += cluster_req_norm_sum[d, j] * req_norm[request_id, j]
                rafr[d] = dot / np.float32(cluster_count[d])

        # RAFE: pre-computed.
        rafe = all_rafe[request_id]

        # Normalize each to [0, 1] among valid devices.
        rafr_max = np.float32(-1e30)
        rafe_max = np.float32(-1e30)
        for d in range(num_devices):
            if cluster_count[d] < quota:
                if rafr[d] > rafr_max:
                    rafr_max = rafr[d]
                if rafe[d] > rafe_max:
                    rafe_max = rafe[d]

        # Score and assign.
        best_score = np.float32(-1e30)
        best_device = 0
        for d in range(num_devices):
            if cluster_count[d] >= quota:
                continue
            rn = rafr[d] / (rafr_max + _eps) if rafr_max > 0 else np.float32(0.0)
            en = rafe[d] / (rafe_max + _eps) if rafe_max > 0 else np.float32(0.0)
            s = np.float32(alpha_r) * rn + np.float32(beta_r) * en
            if s > best_score:
                best_score = s
                best_device = d

        labels[request_id] = best_device
        for j in range(num_experts):
            cluster_req_norm_sum[best_device, j] += req_norm[request_id, j]
        cluster_count[best_device] += 1

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
    batch: RequestBatch,
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
        # Normalize each term to [0, 1] so alpha/beta/gamma weights are meaningful.
        _eps = 1e-12
        _eafe_max = float(eafe[valid].max()) if bool(valid.any()) else 0.0
        _eafr_max = float(eafr[valid].max()) if bool(valid.any()) else 0.0
        _load_max = float(cluster_load[valid].max()) if bool(valid.any()) else 0.0
        eafe_n = eafe / (_eafe_max + _eps) if _eafe_max > 0 else eafe
        eafr_n = eafr / (_eafr_max + _eps) if _eafr_max > 0 else eafr
        load_n = cluster_load / (_load_max + _eps) if _load_max > 0 else cluster_load
        score = alpha_e * eafe_n + beta_e * eafr_n - gamma_e * load_n
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
        batch=batch,
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

    # Pre-compute token_device_affinity for incremental updates during ft_steps.
    expert_mask = F.one_hot(labels, num_classes=num_devices).to(torch.float32)
    tda = torch.matmul(cp, expert_mask)  # [num_seen, num_devices]
    # token_load_imbalance is invariant (only depends on token_cluster_scores, not expert_labels).
    weighted_tcs = token_cluster_scores * a.unsqueeze(1)
    token_load_imb = float(torch.abs(weighted_tcs.sum(dim=0) - float(a.sum()) / num_devices).sum())
    # Maintain local_activation incrementally.
    local_act = (token_cluster_scores * tda).sum(dim=1)  # [num_seen]

    for _ in range(ft_steps):
        cluster_1, cluster_2 = torch.randperm(num_devices, generator=generator)[:2].tolist()
        experts_1 = torch.nonzero(labels == cluster_1, as_tuple=False).flatten()
        experts_2 = torch.nonzero(labels == cluster_2, as_tuple=False).flatten()
        if experts_1.numel() == 0 or experts_2.numel() == 0:
            continue
        expert_1 = int(experts_1[torch.randint(experts_1.numel(), (1,), generator=generator)])
        expert_2 = int(experts_2[torch.randint(experts_2.numel(), (1,), generator=generator)])

        # Swapping expert_1 (cluster_1->cluster_2) and expert_2 (cluster_2->cluster_1).
        # delta for cluster_1: loses e1, gains e2; cluster_2: gains e1, loses e2.
        delta = cp[:, expert_2] - cp[:, expert_1]  # [num_seen]
        # local_act change = tcs[:,c1]*delta + tcs[:,c2]*(-delta)
        delta_local = (token_cluster_scores[:, cluster_1] - token_cluster_scores[:, cluster_2]) * delta
        local_act_new = local_act + delta_local
        remote_cost = float(torch.sum(a * (1.0 - local_act_new)))
        score = theta * token_load_imb + (1.0 - theta) * remote_cost

        if score < best_score:
            best_score = score
            tda[:, cluster_1] += delta
            tda[:, cluster_2] -= delta
            local_act = local_act_new
            labels[expert_1] = cluster_2
            labels[expert_2] = cluster_1
            best_labels = labels.clone()

    return best_labels


def build_token_cluster_scores(
    num_seen_tokens: int,
    num_devices: int,
    batch: RequestBatch,
    request_labels: torch.Tensor,
) -> torch.Tensor:
    """Vectorized: scatter_add over all tokens at once instead of per-request loop."""
    device_ids = request_labels[batch.request_id_per_token]  # [total_tokens]
    flat_idx = batch.all_token_indices * num_devices + device_ids
    counts_flat = torch.zeros(
        num_seen_tokens * num_devices, dtype=torch.float32
    )
    counts_flat.scatter_add_(
        0, flat_idx, torch.ones(flat_idx.shape[0], dtype=torch.float32)
    )
    counts = counts_flat.reshape(num_seen_tokens, num_devices)

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
    """Vectorized: concat all routed_experts, compute device_trace in bulk."""
    num_layers = len(moe_layer_ids)
    seq_count = num_devices ** lookback

    # Concatenate all routed_experts across all requests: [total_tokens, num_layers, top_k]
    all_routed = torch.from_numpy(
        np.concatenate([r.routed_experts.numpy() for r in requests])
    ).to(torch.long)
    total_tokens = all_routed.shape[0]

    # Build device_trace [total_tokens, num_layers] via vectorized majority vote.
    device_trace = torch.zeros((total_tokens, num_layers), dtype=torch.long)
    for layer_offset, layer_id in enumerate(moe_layer_ids):
        expert_labels = expert_labels_by_layer[layer_id]
        # Map expert_ids -> device_ids for all tokens at once.
        routed_layer = all_routed[:, layer_offset, :]  # [total_tokens, top_k]
        device_ids = expert_labels[routed_layer]  # [total_tokens, top_k]
        # Majority vote: one_hot -> sum -> argmax.
        votes = torch.zeros(total_tokens, num_devices, dtype=torch.long)
        for k in range(device_ids.shape[1]):
            votes.scatter_add_(1, device_ids[:, k : k + 1], torch.ones(total_tokens, 1, dtype=torch.long))
        device_trace[:, layer_offset] = votes.argmax(dim=1)

    # Build transition counts via vectorized scatter_add.
    counts_by_layer: dict[int, torch.Tensor] = {
        layer_id: torch.zeros((seq_count, num_devices), dtype=torch.float32)
        for layer_id in moe_layer_ids
    }
    ones = torch.ones(total_tokens, dtype=torch.float32)
    for layer_offset, layer_id in enumerate(moe_layer_ids):
        if layer_offset < lookback:
            continue
        # Encode lookback device sequence as seq_id.
        seq_id = device_trace[:, layer_offset - lookback].clone()
        for lb in range(1, lookback):
            seq_id = seq_id * num_devices + device_trace[:, layer_offset - lookback + lb]
        next_device = device_trace[:, layer_offset]
        flat_idx = seq_id * num_devices + next_device
        counts_flat = torch.zeros(seq_count * num_devices, dtype=torch.float32)
        counts_flat.scatter_add_(0, flat_idx, ones)
        counts_by_layer[layer_id] = counts_flat.reshape(seq_count, num_devices)

    # Normalize to probabilities.
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
