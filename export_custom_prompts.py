from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_datasets import (
    extract_prompt_text,
    iter_dataset_rows,
    parse_dataset_spec,
)


DEFAULT_DATASETS = [
    "mmlu",
    "sharegpt-vicuna-unfiltered",
    "lmsys-chat-1m",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export datasets into vLLM custom prompt JSONL format."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset selector. Repeat for multiple datasets. "
            "Defaults to mmlu, sharegpt-vicuna-unfiltered, lmsys-chat-1m."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/semantic_parallelism/data",
        help="Directory where exported .jsonl files will be written.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap per dataset.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code when loading HuggingFace datasets.",
    )
    parser.add_argument(
        "--include-output-tokens",
        action="store_true",
        help=(
            "Include an output_tokens field in each row if the source dataset "
            "already provides an answer/completion field. By default only prompt is exported."
        ),
    )
    return parser


def export_dataset(
    dataset_selector: str,
    output_dir: Path,
    max_prompts: int | None,
    trust_remote_code: bool,
    include_output_tokens: bool,
) -> dict[str, Any]:
    spec = parse_dataset_spec(
        dataset_selector,
        trust_remote_code=trust_remote_code,
    )
    output_path = output_dir / f"{spec.name}.custom.jsonl"

    written = 0
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for row_index, row in enumerate(iter_dataset_rows(spec)):
                prompt = extract_prompt_text(spec, row)
                if not prompt:
                    continue

                payload: dict[str, Any] = {
                    "prompt": prompt,
                    "dataset_name": spec.name,
                    "record_id": f"{spec.name}:{row_index}",
                }
                if include_output_tokens:
                    output_tokens = _extract_output_tokens(row)
                    if output_tokens is not None:
                        payload["output_tokens"] = output_tokens

                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                if max_prompts is not None and written >= max_prompts:
                    break
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return {
        "dataset": spec.name,
        "source": spec.source,
        "split": spec.split,
        "output_path": str(output_path),
        "prompts_written": written,
    }


def _extract_output_tokens(row: dict[str, Any]) -> int | None:
    for key in ("output_tokens",):
        value = row.get(key)
        if isinstance(value, int) and value > 0:
            return value

    for key in ("answer", "response", "chosen", "completion"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3.5-35B-A3B",
                    local_files_only=True,
                )
                return len(tokenizer(value, add_special_tokens=False).input_ids)
            except Exception:
                return None
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset in datasets:
        try:
            summaries.append(
                export_dataset(
                    dataset_selector=dataset,
                    output_dir=output_dir,
                    max_prompts=args.max_prompts,
                    trust_remote_code=args.trust_remote_code,
                    include_output_tokens=args.include_output_tokens,
                )
            )
        except Exception as exc:
            summaries.append(
                {
                    "dataset": dataset,
                    "error": str(exc),
                }
            )
    print(json.dumps({"exports": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
