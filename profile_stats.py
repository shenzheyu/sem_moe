from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

    freq_by_layer: dict[int, Counter[int]] = {layer: Counter() for layer in moe_layer_ids}
    count_by_layer: dict[int, dict[int, Counter[int]]] = {
        layer: defaultdict(Counter) for layer in moe_layer_ids
    }
    seen_token_ids: set[int] = set()

    for relative_path in manifest["raw_shards"]:
        shard = load_torch_artifact(run_dir / relative_path)
        for record in shard["records"]:
            prompt_token_ids = record["prompt_token_ids"].tolist()
            routed_experts = record["routed_experts"]
            for token_index, token_id in enumerate(prompt_token_ids):
                seen_token_ids.add(int(token_id))
                for layer_id in moe_layer_ids:
                    experts = routed_experts[token_index, layer_id].tolist()
                    freq_by_layer[layer_id][int(token_id)] += 1
                    for expert_id in experts:
                        count_by_layer[layer_id][int(token_id)][int(expert_id)] += 1

    sorted_seen_token_ids = sorted(seen_token_ids)
    token_to_index = {
        token_id: token_index for token_index, token_id in enumerate(sorted_seen_token_ids)
    }

    layers: dict[str, dict[str, torch.Tensor]] = {}
    for layer_id in moe_layer_ids:
        layers[str(layer_id)] = build_sparse_layer_artifact(
            count_by_token=count_by_layer[layer_id],
            freq_by_token=freq_by_layer[layer_id],
            token_to_index=token_to_index,
            num_seen_tokens=len(sorted_seen_token_ids),
        )

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
        "seen_token_ids": torch.tensor(sorted_seen_token_ids, dtype=torch.int32),
        "layers": layers,
    }


def build_sparse_layer_artifact(
    count_by_token: dict[int, Counter[int]],
    freq_by_token: Counter[int],
    token_to_index: dict[int, int],
    num_seen_tokens: int,
) -> dict[str, torch.Tensor]:
    row_splits = [0]
    count_expert_ids: list[int] = []
    count_values: list[int] = []
    cp_values: list[float] = []
    freq = torch.zeros(num_seen_tokens, dtype=torch.int32)

    rows_by_index: dict[int, list[tuple[int, int]]] = {}
    for token_id, expert_counter in count_by_token.items():
        token_index = token_to_index[token_id]
        rows_by_index[token_index] = sorted(
            (expert_id, count) for expert_id, count in expert_counter.items()
        )
    for token_id, token_freq in freq_by_token.items():
        freq[token_to_index[token_id]] = int(token_freq)

    for token_index in range(num_seen_tokens):
        rows = rows_by_index.get(token_index, [])
        total = sum(count for _, count in rows)
        for expert_id, count in rows:
            count_expert_ids.append(int(expert_id))
            count_values.append(int(count))
            cp_values.append(float(count) / float(total) if total else 0.0)
        row_splits.append(len(count_expert_ids))

    freq_total = int(freq.sum())
    a_values = (
        freq.to(torch.float32) / float(freq_total)
        if freq_total > 0
        else torch.zeros(num_seen_tokens, dtype=torch.float32)
    )

    return {
        "freq": freq,
        "a_values": a_values,
        "row_splits": torch.tensor(row_splits, dtype=torch.int32),
        "count_expert_ids": torch.tensor(count_expert_ids, dtype=torch.int32),
        "count_values": torch.tensor(count_values, dtype=torch.int32),
        "cp_values": torch.tensor(cp_values, dtype=torch.float32),
    }


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
