from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from artifacts import (
    FORMAT_VERSION,
    collection_manifest_path,
    ensure_run_dir,
    raw_shard_path,
    save_torch_artifact,
    utc_timestamp,
    write_json,
)
from dataset_utils import DatasetSpec, PromptRecord, iter_prompt_records, parse_dataset_spec


ROUTED_EXPERTS_TRACE_DIR_ENV = "VLLM_ROUTED_EXPERTS_TRACE_DIR"
ROUTED_EXPERTS_TRACE_DIR_NAME = "worker_traces"


@dataclass(frozen=True)
class ProfileRunConfig:
    model: str
    dataset_specs: list[DatasetSpec]
    output_dir: str
    run_name: str | None
    profile_fraction: float
    batch_size: int
    shard_size: int
    max_prompts: int | None
    max_prompt_tokens: int | None
    seed: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    enable_expert_parallel: bool
    trust_remote_code: bool
    enforce_eager: bool


def run_collect_activations(args: Any) -> None:
    config = ProfileRunConfig(
        model=args.model,
        dataset_specs=[
            parse_dataset_spec(
                raw_value=value,
                split_override=args.dataset_split,
                trust_remote_code=args.trust_remote_code,
            )
            for value in args.dataset
        ],
        output_dir=args.output_dir,
        run_name=args.run_name,
        profile_fraction=args.profile_fraction,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        max_prompts=args.max_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    _validate_collect_config(config)

    run_dir = ensure_run_dir(config.output_dir, config.run_name)
    trace_dir = _ensure_worker_trace_dir(run_dir)
    model_metadata = load_model_profile_metadata(
        model_name=config.model,
        trust_remote_code=config.trust_remote_code,
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": utc_timestamp(),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "config": {
            "model": config.model,
            "datasets": [spec.__dict__ for spec in config.dataset_specs],
            "profile_fraction": config.profile_fraction,
            "batch_size": config.batch_size,
            "shard_size": config.shard_size,
            "max_prompts": config.max_prompts,
            "max_prompt_tokens": config.max_prompt_tokens,
            "seed": config.seed,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "enable_expert_parallel": config.enable_expert_parallel,
            "trust_remote_code": config.trust_remote_code,
            "enforce_eager": config.enforce_eager,
            "trace_transport": "vllm_worker_file_export",
        },
        "model_metadata": model_metadata,
        "raw_shards": [],
        "summary": {
            "requested_prompt_limit": config.max_prompts,
            "kept_prompts": 0,
            "skipped_long_prompts": 0,
            "collected_prompt_tokens": 0,
            "per_dataset_prompts": {},
        },
    }

    from vllm import LLM, SamplingParams

    with _scoped_env({ROUTED_EXPERTS_TRACE_DIR_ENV: str(trace_dir)}):
        llm = LLM(
            model=config.model,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            enable_expert_parallel=config.enable_expert_parallel,
            enforce_eager=config.enforce_eager,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        _clear_worker_trace_dir(trace_dir)

        shard_records: list[dict[str, Any]] = []
        prompt_batch: list[str] = []
        meta_batch: list[PromptRecord] = []
        shard_index = 0

        for prompt_record in iter_prompt_records(
            dataset_specs=config.dataset_specs,
            profile_fraction=config.profile_fraction,
            seed=config.seed,
            max_prompts=config.max_prompts,
        ):
            if _prompt_too_long(
                prompt_text=prompt_record.prompt_text,
                tokenizer=tokenizer,
                max_prompt_tokens=config.max_prompt_tokens,
            ):
                manifest["summary"]["skipped_long_prompts"] += 1
                continue

            prompt_batch.append(prompt_record.prompt_text)
            meta_batch.append(prompt_record)

            if len(prompt_batch) < config.batch_size:
                continue

            shard_index = _consume_batch(
                llm=llm,
                sampling_params=sampling_params,
                prompt_batch=prompt_batch,
                meta_batch=meta_batch,
                run_dir=run_dir,
                trace_dir=trace_dir,
                shard_records=shard_records,
                shard_index=shard_index,
                shard_size=config.shard_size,
                manifest=manifest,
            )
            prompt_batch.clear()
            meta_batch.clear()

        if prompt_batch:
            shard_index = _consume_batch(
                llm=llm,
                sampling_params=sampling_params,
                prompt_batch=prompt_batch,
                meta_batch=meta_batch,
                run_dir=run_dir,
                trace_dir=trace_dir,
                shard_records=shard_records,
                shard_index=shard_index,
                shard_size=config.shard_size,
                manifest=manifest,
            )

        if shard_records:
            shard_index = _flush_shard(
                run_dir=run_dir,
                shard_records=shard_records,
                shard_index=shard_index,
                manifest=manifest,
            )
            shard_records.clear()

    write_json(collection_manifest_path(run_dir), manifest)
    print(f"Collected raw activations into {run_dir}")


def _validate_collect_config(config: ProfileRunConfig) -> None:
    if not config.dataset_specs:
        raise ValueError("At least one dataset must be provided.")
    if not 0.0 < config.profile_fraction <= 1.0:
        raise ValueError("--profile-fraction must be in the range (0, 1].")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if config.max_prompts is not None and config.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive when provided.")
    if config.max_prompt_tokens is not None and config.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be positive when provided.")
    if config.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if config.enable_expert_parallel and config.tensor_parallel_size == 1:
        raise ValueError(
            "--enable-expert-parallel requires --tensor-parallel-size > 1 "
            "in the current offline collector."
        )


def load_model_profile_metadata(
    model_name: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    from vllm.transformers_utils.config import get_config

    config = get_config(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    text_config = resolve_text_config(config)
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    num_experts = _infer_num_experts(text_config)
    top_k = _infer_top_k(text_config)
    moe_layer_ids = infer_moe_layer_ids(text_config)
    return {
        "model_name": model_name,
        "model_type": getattr(text_config, "model_type", None),
        "num_hidden_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layer_ids": moe_layer_ids,
    }


def resolve_text_config(config: Any) -> Any:
    for attr_name in ("hf_text_config", "text_config", "llm_config"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return value
    return config


def infer_moe_layer_ids(config: Any) -> list[int]:
    num_layers = int(getattr(config, "num_hidden_layers"))
    mlp_only_layers = set(getattr(config, "mlp_only_layers", []))
    if hasattr(config, "decoder_sparse_step") and _infer_num_experts(config) > 0:
        step = int(getattr(config, "decoder_sparse_step"))
        return [
            layer_id
            for layer_id in range(num_layers)
            if layer_id not in mlp_only_layers and (layer_id + 1) % step == 0
        ]
    if hasattr(config, "moe_layer_freq") and _infer_num_experts(config) > 0:
        freq = int(getattr(config, "moe_layer_freq"))
        return [
            layer_id
            for layer_id in range(num_layers)
            if layer_id not in mlp_only_layers and (layer_id + 1) % freq == 0
        ]
    if _infer_num_experts(config) > 0:
        return [layer_id for layer_id in range(num_layers) if layer_id not in mlp_only_layers]
    return []


def _infer_num_experts(config: Any) -> int:
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    return 0


def _infer_top_k(config: Any) -> int:
    for key in ("num_experts_per_tok", "num_experts_per_token", "top_k"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    return 0


def _prompt_too_long(
    prompt_text: str,
    tokenizer: Any,
    max_prompt_tokens: int | None,
) -> bool:
    if max_prompt_tokens is None:
        return False
    return len(tokenizer.encode(prompt_text, add_special_tokens=False)) > max_prompt_tokens


def _ensure_worker_trace_dir(run_dir: Path) -> Path:
    trace_dir = run_dir / ROUTED_EXPERTS_TRACE_DIR_NAME
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _clear_worker_trace_dir(trace_dir: Path) -> None:
    for path in trace_dir.glob("*"):
        if path.is_file():
            path.unlink()


@contextmanager
def _scoped_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _iter_worker_trace_paths(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("routed_experts_dp*_tp*_batch*.pt"))


def _load_new_worker_traces(
    trace_dir: Path,
    seen_trace_names: set[str],
) -> dict[str, torch.Tensor]:
    trace_paths = [
        path for path in _iter_worker_trace_paths(trace_dir) if path.name not in seen_trace_names
    ]
    traces_by_request = _collect_worker_trace_records(trace_paths)
    for path in trace_paths:
        path.unlink()
    return traces_by_request


def _collect_worker_trace_records(trace_paths: list[Path]) -> dict[str, torch.Tensor]:
    traces_by_request: dict[str, list[torch.Tensor]] = defaultdict(list)
    for path in trace_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for record in payload["records"]:
            traces_by_request[str(record["request_id"])].append(record["routed_experts"])
    merged: dict[str, torch.Tensor] = {}
    for request_id, chunks in traces_by_request.items():
        merged[request_id] = (
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        )
    return merged


def _resolve_request_trace(
    request_output: Any,
    traces_by_request: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    request_id = str(request_output.request_id)
    if request_id not in traces_by_request:
        raise RuntimeError(f"Missing routed expert trace for request_id={request_id}.")

    prompt_token_ids = request_output.prompt_token_ids or []
    if not prompt_token_ids:
        raise RuntimeError(
            f"Missing prompt_token_ids for request_id={request_id} during profiling."
        )

    prompt_token_tensor = torch.tensor(prompt_token_ids, dtype=torch.int32)
    routed_tensor = traces_by_request[request_id]
    if prompt_token_tensor.shape[0] != routed_tensor.shape[0]:
        raise RuntimeError(
            "Prompt-token and routed-expert lengths differ for "
            f"request_id={request_id}: "
            f"{prompt_token_tensor.shape[0]} != {routed_tensor.shape[0]}."
        )

    return prompt_token_tensor, routed_tensor


def _consume_batch(
    llm: Any,
    sampling_params: Any,
    prompt_batch: list[str],
    meta_batch: list[PromptRecord],
    run_dir: Path,
    trace_dir: Path,
    shard_records: list[dict[str, Any]],
    shard_index: int,
    shard_size: int,
    manifest: dict[str, Any],
) -> int:
    seen_trace_names = {path.name for path in _iter_worker_trace_paths(trace_dir)}
    outputs = llm.generate(prompt_batch, sampling_params)
    traces_by_request = _load_new_worker_traces(trace_dir, seen_trace_names)
    output_request_ids = {str(request_output.request_id) for request_output in outputs}
    trace_request_ids = set(traces_by_request)
    missing_request_ids = output_request_ids - trace_request_ids
    if missing_request_ids:
        missing = ", ".join(sorted(missing_request_ids))
        raise RuntimeError(f"Missing worker trace files for requests: {missing}.")
    unexpected_request_ids = trace_request_ids - output_request_ids
    if unexpected_request_ids:
        unexpected = ", ".join(sorted(unexpected_request_ids))
        raise RuntimeError(f"Unexpected worker trace requests: {unexpected}.")

    for prompt_record, request_output in zip(meta_batch, outputs):
        prompt_token_tensor, routed_tensor = _resolve_request_trace(
            request_output=request_output,
            traces_by_request=traces_by_request,
        )

        shard_records.append(
            {
                "dataset_name": prompt_record.dataset_name,
                "record_id": prompt_record.record_id,
                "prompt_token_ids": prompt_token_tensor,
                "routed_experts": routed_tensor,
            }
        )
        manifest["summary"]["kept_prompts"] += 1
        manifest["summary"]["collected_prompt_tokens"] += int(prompt_token_tensor.numel())
        per_dataset = manifest["summary"]["per_dataset_prompts"]
        per_dataset[prompt_record.dataset_name] = per_dataset.get(prompt_record.dataset_name, 0) + 1

        if len(shard_records) >= shard_size:
            shard_index = _flush_shard(
                run_dir=run_dir,
                shard_records=shard_records,
                shard_index=shard_index,
                manifest=manifest,
            )
            shard_records.clear()

    return shard_index


def _flush_shard(
    run_dir: Path,
    shard_records: list[dict[str, Any]],
    shard_index: int,
    manifest: dict[str, Any],
) -> int:
    shard_path = raw_shard_path(run_dir, shard_index)
    save_torch_artifact(
        shard_path,
        {
            "metadata": {
                "run_name": manifest["run_name"],
                "model_name": manifest["model_metadata"]["model_name"],
            },
            "records": list(shard_records),
        },
    )
    manifest["raw_shards"].append(str(shard_path.relative_to(run_dir)))
    return shard_index + 1
