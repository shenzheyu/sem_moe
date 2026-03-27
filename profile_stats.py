from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import torch
import torch.nn.functional as F
from safetensors import safe_open

from progress_utils import progress_iter
from artifacts import (
    collection_manifest_path,
    load_json,
    load_torch_artifact,
    save_torch_artifact,
    stats_artifact_path,
    vocab_artifact_path,
)


def run_build_token_expert_stats(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = load_json(collection_manifest_path(run_dir))
    stats_artifact = build_token_expert_stats_from_run(run_dir, manifest)
    save_torch_artifact(stats_artifact_path(run_dir), stats_artifact)
    print(f"Built token-expert stats at {stats_artifact_path(run_dir)}")


def build_token_expert_stats_from_run(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    model_metadata = manifest["model_metadata"]
    moe_layer_ids = [int(layer_id) for layer_id in model_metadata["moe_layer_ids"]]
    num_experts = int(model_metadata["num_experts"])
    top_k = int(model_metadata["top_k"])
    num_layers = len(moe_layer_ids)

    # Determine vocab_cap from a quick scan of the first + last shard.
    shard_paths = manifest["raw_shards"]
    vocab_cap = _estimate_vocab_cap(run_dir, shard_paths)

    # Pick device: use GPU with most free memory for fast scatter_add_.
    device = _pick_device(num_layers, vocab_cap=vocab_cap, num_experts=num_experts)

    # Dense accumulators — per-layer to avoid >2B element flat tensors.
    freq_acc = torch.zeros(vocab_cap, dtype=torch.int64, device=device)
    # count_acc[layer_idx]: flat [vocab_cap * num_experts], dtype int32
    count_acc = [
        torch.zeros(vocab_cap * num_experts, dtype=torch.int32, device=device)
        for _ in range(num_layers)
    ]

    # Pre-allocate reusable ones buffers on device.
    max_tokens_per_shard = 65_536  # generous upper bound
    ones_freq = torch.ones(max_tokens_per_shard, dtype=torch.int64, device=device)
    ones_count = torch.ones(
        max_tokens_per_shard * top_k, dtype=torch.int32, device=device
    )

    for relative_path in progress_iter(
        shard_paths, total=len(shard_paths), desc="Aggregating shards"
    ):
        shard = load_torch_artifact(run_dir / relative_path)
        records = shard["records"]
        if not records:
            continue

        # Use numpy concat (much faster than torch.cat for many small tensors),
        # then transfer to device in one shot.
        all_token_ids = torch.from_numpy(
            np.concatenate([r["prompt_token_ids"].numpy() for r in records])
        ).to(dtype=torch.int64, device=device)
        all_experts = torch.from_numpy(
            np.concatenate([r["routed_experts"].numpy() for r in records])
        ).to(dtype=torch.int64, device=device)
        n = all_token_ids.shape[0]

        # Grow buffers if needed (rare).
        max_tid = int(all_token_ids.max().item())
        if max_tid >= vocab_cap:
            new_cap = max_tid + 10_000
            freq_acc = _grow_tensor(freq_acc, vocab_cap, new_cap)
            for i in range(num_layers):
                count_acc[i] = _grow_count_1d(
                    count_acc[i], vocab_cap, new_cap, num_experts
                )
            vocab_cap = new_cap
        if n > max_tokens_per_shard:
            max_tokens_per_shard = n
            ones_freq = torch.ones(n, dtype=torch.int64, device=device)
            ones_count = torch.ones(n * top_k, dtype=torch.int32, device=device)

        # freq: single scatter_add.
        freq_acc.scatter_add_(0, all_token_ids, ones_freq[:n])

        # count: per-layer scatter_add (40 iters of fast GPU scatter).
        tok_expanded = (
            all_token_ids.unsqueeze(1).expand(n, top_k) * num_experts
        )  # [N, top_k]
        oc = ones_count[: n * top_k]
        for layer_idx in range(num_layers):
            flat_idx = (
                tok_expanded + all_experts[:, layer_idx, :]
            ).reshape(-1)
            count_acc[layer_idx].scatter_add_(0, flat_idx, oc)

    # --- Determine seen tokens and build output on CPU ---
    freq_cpu = freq_acc[:vocab_cap].cpu()
    seen_mask = freq_cpu > 0
    sorted_seen_token_ids = seen_mask.nonzero(as_tuple=False).squeeze(1)
    seen_freq = freq_cpu[sorted_seen_token_ids]

    layers: dict[str, dict[str, torch.Tensor]] = {}
    for layer_idx, layer_id in enumerate(moe_layer_ids):
        layer_2d = count_acc[layer_idx].reshape(vocab_cap, num_experts).cpu()
        seen_count = layer_2d[sorted_seen_token_ids]
        layers[str(layer_id)] = _dense_to_sparse_layer(seen_count, seen_freq)
        # Free GPU memory for this layer immediately.
        count_acc[layer_idx] = None  # type: ignore[assignment]

    return {
        "metadata": {
            "model_name": model_metadata["model_name"],
            "model_type": model_metadata["model_type"],
            "num_hidden_layers": model_metadata["num_hidden_layers"],
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_ids": moe_layer_ids,
            "raw_trace_shards": manifest["raw_shards"],
        },
        "seen_token_ids": sorted_seen_token_ids.to(torch.int32),
        "layers": layers,
    }


def _estimate_vocab_cap(run_dir: Path, shard_paths: list[str]) -> int:
    """Quick scan of first and last shard to set initial vocab_cap."""
    max_tid = 0
    for idx in (0, len(shard_paths) - 1):
        shard = load_torch_artifact(run_dir / shard_paths[idx])
        for r in shard["records"]:
            t = r["prompt_token_ids"].max().item()
            if t > max_tid:
                max_tid = t
    return int(max_tid) + 10_000


def _pick_device(
    num_layers: int, vocab_cap: int, num_experts: int
) -> torch.device:
    """Select GPU with enough free memory, or fall back to CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # Estimate: num_layers tensors of [vocab_cap * num_experts] int32 + freq int64.
    required = num_layers * vocab_cap * num_experts * 4 + vocab_cap * 8
    required_with_headroom = int(required * 1.3)
    best_gpu, best_free = -1, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except (RuntimeError, torch.cuda.CudaError):
            continue
        if free > best_free:
            best_free = free
            best_gpu = i
    if best_gpu >= 0 and best_free >= required_with_headroom:
        print(f"Using cuda:{best_gpu} ({best_free / 2**30:.1f} GB free)")
        return torch.device(f"cuda:{best_gpu}")
    print("No GPU with enough free memory; falling back to CPU")
    return torch.device("cpu")


def _grow_tensor(tensor: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
    grown = torch.zeros(new_size, dtype=tensor.dtype, device=tensor.device)
    grown[:old_size] = tensor[:old_size]
    return grown


def _grow_count_1d(
    count_1d: torch.Tensor,
    old_vocab: int,
    new_vocab: int,
    num_experts: int,
) -> torch.Tensor:
    old_2d = count_1d.reshape(old_vocab, num_experts)
    new_1d = torch.zeros(
        new_vocab * num_experts, dtype=count_1d.dtype, device=count_1d.device
    )
    new_1d.reshape(new_vocab, num_experts)[:old_vocab, :] = old_2d
    return new_1d


def _dense_to_sparse_layer(
    count_2d: torch.Tensor,
    freq: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert dense [num_seen, num_experts] counts + freq to sparse CSR format."""
    num_seen = count_2d.shape[0]

    # Build CSR from non-zero entries.
    nz_mask = count_2d > 0  # [num_seen, num_experts]
    row_nnz = nz_mask.sum(dim=1)  # [num_seen]
    row_splits = torch.zeros(num_seen + 1, dtype=torch.int32)
    torch.cumsum(row_nnz, dim=0, out=row_splits[1:])

    nz_indices = nz_mask.nonzero(as_tuple=False)  # [nnz, 2]
    count_expert_ids = nz_indices[:, 1].to(torch.int32)
    count_values = count_2d[nz_indices[:, 0], nz_indices[:, 1]].to(torch.int32)

    # Cp: normalize counts along expert dim per token.
    # row_totals[token] = sum of counts across all experts for that token.
    row_totals = count_2d.sum(dim=1).float()  # [num_seen]
    row_totals_per_entry = row_totals[nz_indices[:, 0]]
    cp_values = count_values.float() / row_totals_per_entry.clamp(min=1.0)

    freq_i32 = freq.to(torch.int32)
    freq_total = freq.sum().item()
    a_values = (
        freq.float() / float(freq_total)
        if freq_total > 0
        else torch.zeros(num_seen, dtype=torch.float32)
    )

    return {
        "freq": freq_i32,
        "a_values": a_values,
        "row_splits": row_splits,
        "count_expert_ids": count_expert_ids,
        "count_values": count_values,
        "cp_values": cp_values.to(torch.float32),
    }


def build_sparse_layer_artifact(
    count_by_token: dict[int, dict[int, int]],
    freq_by_token: dict[int, int],
    token_to_index: dict[int, int],
    num_seen_tokens: int,
) -> dict[str, torch.Tensor]:
    """Kept for test compatibility — wraps _dense_to_sparse_layer."""
    max_expert = 0
    for expert_counter in count_by_token.values():
        if expert_counter:
            max_expert = max(max_expert, max(expert_counter.keys()))
    num_experts = max_expert + 1

    count_2d = torch.zeros(num_seen_tokens, num_experts, dtype=torch.int32)
    for token_id, expert_counter in count_by_token.items():
        idx = token_to_index[token_id]
        for expert_id, cnt in expert_counter.items():
            count_2d[idx, expert_id] = cnt

    freq = torch.zeros(num_seen_tokens, dtype=torch.int64)
    for token_id, f in freq_by_token.items():
        freq[token_to_index[token_id]] = f

    return _dense_to_sparse_layer(count_2d, freq)


def run_extend_vocab(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    stats_artifact = load_torch_artifact(stats_artifact_path(run_dir))
    manifest = load_json(collection_manifest_path(run_dir))
    model_name = manifest["config"]["model"]
    trust_remote_code = bool(manifest["config"].get("trust_remote_code", False))

    embedding = load_model_embedding_tensor(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
    )
    extension = build_vocab_extension_from_embedding(
        embedding=embedding,
        seen_token_ids=stats_artifact["seen_token_ids"].to(torch.long),
        query_batch_size=args.query_batch_size,
        device=args.device,
        vocab_limit=args.vocab_limit,
        show_progress=not getattr(args, "no_progress", False),
    )
    extension["metadata"] = {
        "model_name": model_name,
        "algorithm": "embedding_cosine_exact_chunked",
        "query_batch_size": args.query_batch_size,
        "device": args.device,
        "vocab_limit": args.vocab_limit,
    }
    save_torch_artifact(vocab_artifact_path(run_dir), extension)
    print(f"Built vocab extension at {vocab_artifact_path(run_dir)}")


def build_vocab_extension_from_embedding(
    embedding: torch.Tensor,
    seen_token_ids: torch.Tensor,
    query_batch_size: int,
    device: str,
    vocab_limit: int | None = None,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if embedding.ndim != 2:
        raise ValueError("Embedding tensor must be 2D")

    unique_seen = torch.unique(seen_token_ids.to(torch.long), sorted=True)
    if unique_seen.numel() == 0:
        raise ValueError("Cannot extend the vocabulary with an empty seen-token set.")

    vocab_size = embedding.shape[0]
    effective_vocab = vocab_size if vocab_limit is None else min(vocab_limit, vocab_size)
    target_device = _resolve_device(device)
    unique_seen_on_device = unique_seen.to(target_device)

    seen_embedding = F.normalize(
        embedding[unique_seen].to(device=target_device, dtype=torch.float32),
        dim=1,
    )

    nearest_seen_token_id = torch.empty(effective_vocab, dtype=torch.int32)
    nearest_similarity = torch.empty(effective_vocab, dtype=torch.float32)
    exact_seen_mask = torch.zeros(effective_vocab, dtype=torch.bool)
    if unique_seen.numel() > 0:
        valid_seen = unique_seen[unique_seen < effective_vocab]
        exact_seen_mask[valid_seen] = True

    starts = range(0, effective_vocab, query_batch_size)
    for start in progress_iter(
        starts,
        total=math.ceil(effective_vocab / query_batch_size),
        desc="Extending vocab",
        enabled=show_progress,
    ):
        end = min(start + query_batch_size, effective_vocab)
        query_embedding = F.normalize(
            embedding[start:end].to(device=target_device, dtype=torch.float32),
            dim=1,
        )
        scores = query_embedding @ seen_embedding.T
        best_scores, best_indices = scores.max(dim=1)
        nearest_seen_token_id[start:end] = (
            unique_seen_on_device[best_indices].to(torch.int32).cpu()
        )
        nearest_similarity[start:end] = best_scores.cpu()

    if unique_seen.numel() > 0:
        valid_seen = unique_seen[unique_seen < effective_vocab]
        nearest_seen_token_id[valid_seen] = valid_seen.to(torch.int32)
        nearest_similarity[valid_seen] = 1.0

    return {
        "seen_token_ids": unique_seen.to(torch.int32),
        "nearest_seen_token_id": nearest_seen_token_id,
        "nearest_similarity": nearest_similarity,
        "exact_seen_mask": exact_seen_mask,
        "effective_vocab_size": torch.tensor(effective_vocab, dtype=torch.int32),
        "full_vocab_size": torch.tensor(vocab_size, dtype=torch.int32),
    }


def _resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def load_model_embedding_tensor(
    model_name: str,
    trust_remote_code: bool,
) -> torch.Tensor:
    from huggingface_hub import hf_hub_download

    candidate_names = embedding_tensor_candidates()

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        tensor_path, tensor_name = _resolve_local_embedding_path(model_path, candidate_names)
        return _load_tensor_from_path(tensor_path, tensor_name)

    index_filename = "model.safetensors.index.json"
    index_path = Path(
        hf_hub_download(
            repo_id=model_name,
            filename=index_filename,
        )
    )
    weight_map = load_json(index_path)["weight_map"]
    tensor_name = next(
        (name for name in candidate_names if name in weight_map),
        None,
    )
    if tensor_name is None:
        raise KeyError(
            f"Could not find an embedding tensor in {index_filename} for {model_name}. "
            f"Tried: {candidate_names}"
        )
    tensor_filename = weight_map[tensor_name]
    tensor_path = Path(
        hf_hub_download(
            repo_id=model_name,
            filename=tensor_filename,
        )
    )
    return _load_tensor_from_path(tensor_path, tensor_name)


def embedding_tensor_candidates() -> list[str]:
    return [
        "model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
    ]


def _resolve_local_embedding_path(
    model_path: Path,
    candidate_names: list[str],
) -> tuple[Path, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = load_json(index_path)["weight_map"]
        tensor_name = next(
            (name for name in candidate_names if name in weight_map),
            None,
        )
        if tensor_name is None:
            raise KeyError(
                f"Could not find an embedding tensor in {index_path}. Tried {candidate_names}"
            )
        return model_path / weight_map[tensor_name], tensor_name

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    for tensor_path in safetensors_files:
        with safe_open(str(tensor_path), framework="pt") as handle:
            for tensor_name in candidate_names:
                if tensor_name in handle.keys():
                    return tensor_path, tensor_name

    raise FileNotFoundError(
        f"Could not resolve an embedding tensor under {model_path}. "
        f"Tried {candidate_names}"
    )


def _load_tensor_from_path(tensor_path: Path, tensor_name: str) -> torch.Tensor:
    if tensor_path.suffix == ".safetensors":
        with safe_open(str(tensor_path), framework="pt") as handle:
            return handle.get_tensor(tensor_name)
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    return payload[tensor_name]


def run_inspect(args: Any) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = load_json(collection_manifest_path(run_dir))
    summary = {
        "run_name": manifest["run_name"],
        "model_name": manifest["config"]["model"],
        "datasets": [entry["name"] for entry in manifest["config"]["datasets"]],
        "raw_shards": len(manifest["raw_shards"]),
        "kept_prompts": manifest["summary"]["kept_prompts"],
        "collected_prompt_tokens": manifest["summary"]["collected_prompt_tokens"],
        "skipped_long_prompts": manifest["summary"]["skipped_long_prompts"],
        "trace_transport": manifest["config"].get("trace_transport", "unknown"),
    }

    stats_path = stats_artifact_path(run_dir)
    if stats_path.exists():
        stats_artifact = load_torch_artifact(stats_path)
        summary["stats"] = summarize_stats_artifact(stats_artifact)

    vocab_path = vocab_artifact_path(run_dir)
    if vocab_path.exists():
        vocab_artifact = load_torch_artifact(vocab_path)
        summary["vocab"] = {
            "effective_vocab_size": int(vocab_artifact["effective_vocab_size"]),
            "full_vocab_size": int(vocab_artifact["full_vocab_size"]),
            "seen_token_count": int(vocab_artifact["seen_token_ids"].numel()),
        }

    print(summary)


def summarize_stats_artifact(stats_artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = stats_artifact["metadata"]
    seen_token_ids = stats_artifact["seen_token_ids"]
    top_k = int(metadata["top_k"])
    layer_summaries = {}
    for layer_id in metadata["moe_layer_ids"]:
        layer = stats_artifact["layers"][str(layer_id)]
        row_splits = layer["row_splits"]
        cp_values = layer["cp_values"]
        cumulative_hotness: list[float] = []
        max_cold_hotness: list[float] = []
        for token_index in range(seen_token_ids.numel()):
            start = int(row_splits[token_index])
            end = int(row_splits[token_index + 1])
            if start == end:
                continue
            token_cp = cp_values[start:end].sort(descending=True).values
            cumulative_hotness.append(float(token_cp[:top_k].sum().item()))
            cold = token_cp[top_k:]
            max_cold_hotness.append(float(cold.max().item()) if cold.numel() else 0.0)
        layer_summaries[str(layer_id)] = {
            "token_occurrences": int(layer["freq"].sum().item()),
            "cum_hotness_topk_p50": _median(cumulative_hotness),
            "max_cold_hotness_p50": _median(max_cold_hotness),
        }
    return {
        "seen_token_count": int(seen_token_ids.numel()),
        "num_layers": len(metadata["moe_layer_ids"]),
        "layer_summaries": layer_summaries,
    }


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, 0.5).item())
