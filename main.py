from __future__ import annotations

import argparse
from pathlib import Path

from profile_collect import run_collect_activations
from eval_dp import run_evaluate_dp_scheduling
from schedule import run_build_model_schedule
from profile_stats import (
    run_build_token_expert_stats,
    run_extend_vocab,
    run_inspect,
)


def _existing_run_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Run directory does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline profiling utilities for semantic parallelism."
    )
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser(
        "profile",
        help="Offline profiling pipeline commands.",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")

    collect_parser = profile_subparsers.add_parser(
        "collect-activations",
        help="Run vLLM and collect raw token-to-expert activation traces.",
    )
    collect_parser.add_argument("--model", required=True, help="Model name or path.")
    collect_parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset selector. Repeat to profile multiple datasets. "
            "Use NAME for a built-in dataset or NAME=SOURCE to override the "
            "default HuggingFace/local source."
        ),
    )
    collect_parser.add_argument(
        "--dataset-split",
        default=None,
        help="Override the default split for all selected datasets.",
    )
    collect_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory that will contain the profiling run.",
    )
    collect_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Defaults to a timestamped name.",
    )
    collect_parser.add_argument(
        "--profile-fraction",
        type=float,
        default=0.2,
        help="Deterministic fraction of prompts to keep for profiling.",
    )
    collect_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of prompts per vLLM generate call.",
    )
    collect_parser.add_argument(
        "--shard-size",
        type=int,
        default=128,
        help="Number of requests to store per raw shard.",
    )
    collect_parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on the total number of prompts to profile.",
    )
    collect_parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Skip prompts longer than this token length.",
    )
    collect_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for prompt filtering and vLLM generation.",
    )
    collect_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization target.",
    )
    collect_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used to shard model weights for vLLM collection.",
    )
    collect_parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable vLLM expert parallelism across the tensor-parallel workers.",
    )
    collect_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to model and dataset loading.",
    )
    collect_parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Run vLLM in eager mode while collecting traces.",
    )
    collect_parser.set_defaults(func=run_collect_activations)

    stats_parser = profile_subparsers.add_parser(
        "build-token-expert-stats",
        help="Aggregate raw shards into count/freq/Cp/a artifacts.",
    )
    stats_parser.add_argument("--run-dir", type=_existing_run_dir, required=True)
    stats_parser.set_defaults(func=run_build_token_expert_stats)

    vocab_parser = profile_subparsers.add_parser(
        "extend-vocab",
        help="Build embedding-space nearest-neighbor mappings for the full vocabulary.",
    )
    vocab_parser.add_argument("--run-dir", type=_existing_run_dir, required=True)
    vocab_parser.add_argument(
        "--device",
        default="cuda",
        help="Device for exact cosine search. Use cpu if CUDA is unavailable.",
    )
    vocab_parser.add_argument(
        "--query-batch-size",
        type=int,
        default=256,
        help="Number of query tokens per cosine-similarity chunk.",
    )
    vocab_parser.add_argument(
        "--vocab-limit",
        type=int,
        default=None,
        help="Optional debug fallback that limits vocab extension to the first N tokens.",
    )
    vocab_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars for long-running vocab extension work.",
    )
    vocab_parser.set_defaults(func=run_extend_vocab)

    schedule_parser = profile_subparsers.add_parser(
        "build-model-schedule",
        help="Run the offline co-scheduling solver and export paper-aligned schedule tables.",
    )
    schedule_parser.add_argument("--run-dir", type=_existing_run_dir, required=True)
    schedule_parser.add_argument(
        "--num-devices",
        type=int,
        default=2,
        help="Number of expert clusters / devices used by the offline solver.",
    )
    schedule_parser.add_argument(
        "--lookback",
        type=int,
        default=2,
        help="Number of previous MoE layers used to build A_prob / A / Ap.",
    )
    schedule_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for the offline solver.",
    )
    schedule_parser.add_argument(
        "--n-steps",
        type=int,
        default=8,
        help="Number of alternating optimization steps.",
    )
    schedule_parser.add_argument(
        "--ft-steps",
        type=int,
        default=64,
        help="Number of random expert-swap fine-tuning attempts per layer step.",
    )
    schedule_parser.add_argument(
        "--alpha-e",
        type=float,
        default=1.0,
        help="Weight for expert-expert affinity in expert placement.",
    )
    schedule_parser.add_argument(
        "--beta-e",
        type=float,
        default=1.0,
        help="Weight for request-expert affinity in expert placement.",
    )
    schedule_parser.add_argument(
        "--gamma-e",
        type=float,
        default=1.0,
        help="Penalty weight for imbalanced expert hotness across clusters.",
    )
    schedule_parser.add_argument(
        "--alpha-r",
        type=float,
        default=1.0,
        help="Weight for request-request affinity in request scheduling.",
    )
    schedule_parser.add_argument(
        "--beta-r",
        type=float,
        default=1.0,
        help="Weight for request-expert affinity in request scheduling.",
    )
    schedule_parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Tradeoff between token-load balance and remote activation cost.",
    )
    schedule_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars for long-running offline scheduling work.",
    )
    schedule_parser.set_defaults(func=run_build_model_schedule)

    evaluate_dp_parser = profile_subparsers.add_parser(
        "evaluate-dp-scheduling",
        help="Evaluate Attention-DP request scheduling metrics from raw traces and schedule tables.",
    )
    evaluate_dp_parser.add_argument("--run-dir", type=_existing_run_dir, required=True)
    evaluate_dp_parser.set_defaults(func=run_evaluate_dp_scheduling)

    inspect_parser = profile_subparsers.add_parser(
        "inspect",
        help="Print a summary of the collected raw shards and derived artifacts.",
    )
    inspect_parser.add_argument("--run-dir", type=_existing_run_dir, required=True)
    inspect_parser.set_defaults(func=run_inspect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
