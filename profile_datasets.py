from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


DEFAULT_DATASET_SOURCES: dict[str, tuple[str, str]] = {
    "sharegpt-vicuna-unfiltered": ("Aeala/ShareGPT_Vicuna_unfiltered", "train"),
    "sharegpt": ("Aeala/ShareGPT_Vicuna_unfiltered", "train"),
    "lmsys-chat-1m": ("lmsys/lmsys-chat-1m", "train"),
    "mmlu": ("cais/mmlu", "test"),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    split: str
    trust_remote_code: bool = False


@dataclass(frozen=True)
class PromptRecord:
    dataset_name: str
    record_id: str
    prompt_text: str


def parse_dataset_spec(
    raw_value: str,
    split_override: str | None = None,
    trust_remote_code: bool = False,
) -> DatasetSpec:
    name, sep, source = raw_value.partition("=")
    dataset_name = name.strip().lower()
    if not dataset_name:
        raise ValueError(f"Invalid dataset selector: {raw_value!r}")

    default_source, default_split = DEFAULT_DATASET_SOURCES.get(
        dataset_name,
        (dataset_name, "train"),
    )
    resolved_source = source.strip() if sep else default_source
    resolved_split = split_override or default_split
    return DatasetSpec(
        name=dataset_name,
        source=resolved_source,
        split=resolved_split,
        trust_remote_code=trust_remote_code,
    )


def iter_prompt_records(
    dataset_specs: list[DatasetSpec],
    profile_fraction: float,
    seed: int,
    max_prompts: int | None = None,
) -> Iterator[PromptRecord]:
    emitted = 0
    for spec in dataset_specs:
        for row_index, row in enumerate(iter_dataset_rows(spec)):
            prompt = extract_prompt_text(spec, row)
            if not prompt:
                continue
            if not should_keep_prompt(
                dataset_name=spec.name,
                prompt_text=prompt,
                profile_fraction=profile_fraction,
                seed=seed,
            ):
                continue
            yield PromptRecord(
                dataset_name=spec.name,
                record_id=f"{spec.name}:{row_index}",
                prompt_text=prompt,
            )
            emitted += 1
            if max_prompts is not None and emitted >= max_prompts:
                return


def should_keep_prompt(
    dataset_name: str,
    prompt_text: str,
    profile_fraction: float,
    seed: int,
) -> bool:
    if profile_fraction >= 1.0:
        return True
    if profile_fraction <= 0.0:
        return False
    digest = hashlib.sha1(
        f"{seed}\n{dataset_name}\n{prompt_text}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64) < profile_fraction


def iter_dataset_rows(spec: DatasetSpec) -> Iterator[dict[str, Any]]:
    source_path = Path(spec.source).expanduser()
    if source_path.exists():
        yield from _iter_local_rows(source_path)
        return
    yield from _iter_hf_rows(spec)


def _iter_local_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_file() and path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload, dict):
            for key in ("data", "rows", "examples", "train", "test", "validation"):
                if isinstance(payload.get(key), list):
                    for row in payload[key]:
                        if isinstance(row, dict):
                            yield row
                    return
        raise ValueError(f"Unsupported JSON structure in dataset file: {path}")

    raise ValueError(
        f"Unsupported local dataset source: {path}. Use a .json or .jsonl file."
    )


def _iter_hf_rows(spec: DatasetSpec) -> Iterator[dict[str, Any]]:
    try:
        from datasets import (
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for HuggingFace dataset sources. "
            "Install it or provide a local JSON/JSONL dataset path."
        ) from exc

    if spec.name == "mmlu":
        subset_names = [
            subset
            for subset in get_dataset_config_names(spec.source)
            if subset.lower() != "all"
        ]
        for subset in subset_names:
            split_names = list(
                get_dataset_split_names(spec.source, config_name=subset)
            )
            if spec.split not in split_names:
                continue
            dataset = load_dataset(
                spec.source,
                name=subset,
                split=spec.split,
                streaming=True,
            )
            for row in dataset:
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("subject", subset.replace("_", " "))
                    yield row
        return

    split_name = _resolve_hf_split(
        spec.source,
        spec.split,
        get_dataset_split_names,
    )
    dataset = load_dataset(
        spec.source,
        split=split_name,
        streaming=True,
    )
    for row in dataset:
        if isinstance(row, dict):
            yield dict(row)


def _resolve_hf_split(
    source: str,
    requested_split: str,
    get_split_names: Callable[..., list[str]],
    config_name: str | None = None,
) -> str:
    split_names = list(get_split_names(source, config_name=config_name))
    if not split_names:
        if config_name is None:
            raise ValueError(f"HuggingFace dataset {source!r} has no available splits.")
        raise ValueError(
            f"HuggingFace dataset {source!r} config {config_name!r} has no available splits."
        )
    if requested_split in split_names:
        return requested_split
    for fallback_split in ("train", "validation", "test"):
        if fallback_split in split_names:
            return fallback_split
    return split_names[0]


def extract_prompt_text(spec: DatasetSpec, row: dict[str, Any]) -> str | None:
    if spec.name == "mmlu" or _looks_like_mmlu_row(row):
        prompt = format_mmlu_prompt(row)
        return prompt or None

    for key in ("conversations", "messages", "conversation", "conversation_a"):
        value = row.get(key)
        prompt = _extract_prompt_from_messages(value)
        if prompt:
            return prompt

    for key in ("prompt", "input", "question", "text", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _looks_like_mmlu_row(row: dict[str, Any]) -> bool:
    return _normalize_mmlu_row(row) is not None


def format_mmlu_prompt(row: dict[str, Any]) -> str:
    normalized_row = _normalize_mmlu_row(row)
    if normalized_row is None:
        return ""
    question = str(normalized_row.get("question", "")).strip()
    choices = list(normalized_row.get("choices", []))
    if not question or not choices:
        return ""
    subject = str(
        normalized_row.get("subject")
        or normalized_row.get("category")
        or normalized_row.get("topic")
        or "the subject"
    ).replace("_", " ")
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [
        f"The following are multiple choice questions (with answers) about {subject}.",
        "",
        question,
    ]
    for idx, choice in enumerate(choices):
        if idx >= len(labels):
            break
        lines.append(f"{labels[idx]}. {str(choice).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def _normalize_mmlu_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if _is_mmlu_question_row(row):
        return row
    if not isinstance(row, dict) or len(row) != 1:
        return None
    nested_row = next(iter(row.values()))
    if isinstance(nested_row, dict) and _is_mmlu_question_row(nested_row):
        return dict(nested_row)
    return None


def _is_mmlu_question_row(row: Any) -> bool:
    return isinstance(row, dict) and "question" in row and isinstance(
        row.get("choices"),
        (list, tuple),
    )


def _extract_prompt_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, Iterable) or isinstance(messages, (str, bytes, dict)):
        return None

    collected: list[str] = []
    for message in messages:
        role, content = _normalize_message(message)
        if not content:
            continue
        if role in {"assistant", "gpt", "bot"}:
            if collected:
                break
            continue
        if role in {"user", "human"}:
            collected.append(content)
            continue
        if role is None and not collected:
            collected.append(content)
            break

    prompt = "\n\n".join(part for part in collected if part)
    return prompt or None


def _normalize_message(message: Any) -> tuple[str | None, str | None]:
    if isinstance(message, dict):
        role = message.get("role") or message.get("from") or message.get("speaker")
        content = (
            message.get("content")
            or message.get("value")
            or message.get("text")
            or message.get("utterance")
        )
        return _normalize_role(role), _normalize_content(content)

    if isinstance(message, (list, tuple)) and len(message) >= 2:
        role, content = message[0], message[1]
        return _normalize_role(role), _normalize_content(content)

    if isinstance(message, str):
        return None, message.strip()

    return None, None


def _normalize_role(role: Any) -> str | None:
    if role is None:
        return None
    normalized = str(role).strip().lower()
    if normalized in {"user", "human", "assistant", "gpt", "bot"}:
        return normalized
    return normalized or None


def _normalize_content(content: Any) -> str | None:
    if content is None:
        return None
    text = str(content).strip()
    return text or None
