"""
MoE communication vs compute sweep benchmark.

Reproduces Figure 1 from "Semantic Parallelism" paper:
  - Prefill-only (max_tokens=1) microbenchmark
  - Varies total token count: 2K / 4K / 8K / 16K
  - Generates separate *.pt.trace.json.gz per token count
  - Parses trace files to extract MoE dispatch/compute/combine breakdown

Usage (2-GPU, EP=2):
  CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python moe_comm_sweep.py \
      --model Qwen/Qwen3.5-35B-A3B \
      --dp-size 2 \
      --token-sizes 2048,4096,8192,16384

Usage (4-GPU, EP=4):
  CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python moe_comm_sweep.py \
      --model Qwen/Qwen3.5-35B-A3B \
      --dp-size 4 \
      --token-sizes 2048,4096,8192,16384

Usage (1-GPU baseline, no EP comm):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python moe_comm_sweep.py \
      --model Qwen/Qwen3.5-35B-A3B \
      --dp-size 1 \
      --token-sizes 512,1024,2048
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Kernel classification patterns
# ---------------------------------------------------------------------------

# MoE dispatch: AllGather via AgRsAll2AllManager.dispatch()
DISPATCH_PATTERNS = ["ncclDevKernel_AllGather"]

# MoE combine: ReduceScatter via AgRsAll2AllManager.combine()
COMBINE_PATTERNS = ["ncclDevKernel_ReduceScatter"]

# MoE compute: routing + expert FFN + activation
COMPUTE_PATTERNS = [
    "fused_moe_kernel",
    "topkGating",
    "moe_align_block_size_kernel",
    "count_and_sort_expert_tokens_kernel",
    "act_and_mul_kernel",
]


# ---------------------------------------------------------------------------
# Trace file parsing
# ---------------------------------------------------------------------------


def parse_trace_file(path: Path) -> dict[str, float]:
    """Parse .pt.trace.json.gz → {kernel_name: total_dur_us}."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    kernel_times: dict[str, float] = {}
    for event in data.get("traceEvents", []):
        if event.get("cat") == "kernel":
            name = event.get("name", "")
            dur = event.get("dur", 0)  # microseconds
            kernel_times[name] = kernel_times.get(name, 0) + dur
    return kernel_times


def compute_moe_breakdown(kernel_times: dict[str, float]) -> dict:
    """Compute MoE dispatch/compute/combine breakdown from kernel times."""
    total_us = sum(kernel_times.values())
    dispatch_us = sum(
        v for k, v in kernel_times.items() if any(p in k for p in DISPATCH_PATTERNS)
    )
    combine_us = sum(
        v for k, v in kernel_times.items() if any(p in k for p in COMBINE_PATTERNS)
    )
    compute_us = sum(
        v for k, v in kernel_times.items() if any(p in k for p in COMPUTE_PATTERNS)
    )
    return dict(
        total_ms=total_us / 1000,
        dispatch_ms=dispatch_us / 1000,
        compute_ms=compute_us / 1000,
        combine_ms=combine_us / 1000,
        dispatch_pct=dispatch_us / total_us * 100 if total_us > 0 else 0,
        compute_pct=compute_us / total_us * 100 if total_us > 0 else 0,
        combine_pct=combine_us / total_us * 100 if total_us > 0 else 0,
    )


def find_new_traces(output_dir: Path, existing: set[Path]) -> list[Path]:
    """Find trace files created since `existing` snapshot."""
    return sorted(set(output_dir.glob("*.pt.trace.json.gz")) - existing)


def find_rank0_trace(traces: list[Path]) -> Path | None:
    """Find the rank 0 trace file from a list of new traces."""
    for t in traces:
        if "_dp0_" in t.name or "_rank0." in t.name:
            return t
    return traces[0] if traces else None


def rename_traces(traces: list[Path], actual_total: int) -> None:
    """Rename trace files to include the correct token count."""
    for trace in traces:
        new_name = re.sub(r"tokens_\d+", f"tokens_{actual_total}", trace.name)
        if new_name != trace.name:
            trace.rename(trace.parent / new_name)


# ---------------------------------------------------------------------------
# Synthetic batch construction
# ---------------------------------------------------------------------------

def build_prompts_for_rank(tokenizer, num_tokens: int, prompt_len: int, rank: int, dp_size: int) -> list[str]:
    """Build this rank's share of a synthetic prefill batch totaling num_tokens."""
    total_prompts = max(dp_size, num_tokens // prompt_len)
    # Make divisible by dp_size
    total_prompts = (total_prompts // dp_size) * dp_size

    token_id = tokenizer.encode("hello", add_special_tokens=False)[0]
    base_prompt = tokenizer.decode([token_id] * prompt_len)

    per_rank = total_prompts // dp_size
    return [base_prompt] * per_rank


# ---------------------------------------------------------------------------
# Common profiling + parsing logic
# ---------------------------------------------------------------------------

def _profile_one_token_size(
    llm,
    prompts: list[str],
    sampling_params,
    actual_total: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Run profiling for one token size and return breakdown dict."""
    existing_traces = set(output_dir.glob("*.pt.trace.json.gz"))

    llm.start_profile(profile_prefix=f"tokens_{actual_total}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.num_iters):
        llm.generate(prompts, sampling_params)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / args.num_iters * 1000

    llm.stop_profile()

    print(f"  Waiting for profiler to write ({args.profiler_wait}s) ...")
    time.sleep(args.profiler_wait)

    # Find and parse the new rank 0 trace file
    new_traces = find_new_traces(output_dir, existing_traces)
    rank0_trace = find_rank0_trace(new_traces)

    breakdown = dict(
        total_ms=0, dispatch_ms=0, compute_ms=0, combine_ms=0,
        dispatch_pct=0, compute_pct=0, combine_pct=0,
    )
    if rank0_trace:
        kernel_times = parse_trace_file(rank0_trace)
        breakdown = compute_moe_breakdown(kernel_times)
    else:
        print("  WARNING: no trace file found for this run")

    # Rename traces to include correct token count
    rename_traces(new_traces, actual_total)

    print(
        f"  → latency={elapsed_ms:.1f}ms | "
        f"cuda_total={breakdown['total_ms']:.1f}ms  "
        f"dispatch={breakdown['dispatch_ms']:.1f}ms ({breakdown['dispatch_pct']:.1f}%)  "
        f"compute={breakdown['compute_ms']:.1f}ms ({breakdown['compute_pct']:.1f}%)  "
        f"combine={breakdown['combine_ms']:.1f}ms ({breakdown['combine_pct']:.1f}%)"
    )

    return dict(tokens=actual_total, latency_ms=elapsed_ms, **breakdown)


# ---------------------------------------------------------------------------
# Worker function (runs in each DP rank process)
# ---------------------------------------------------------------------------

def _worker(rank: int, dp_size: int, master_ip: str, master_port: int,
            args: argparse.Namespace, barrier: mp.Barrier,
            result_queue: mp.Queue) -> None:
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    if rank == 0:
        print(f"Loading model {args.model} (DP={dp_size}, EP={dp_size}) ...")

    llm_kwargs: dict = dict(
        model=args.model,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(output_dir),
        },
    )
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    token_sizes = [int(x) for x in args.token_sizes.split(",")]
    results = []

    for num_tokens in token_sizes:
        my_prompts = build_prompts_for_rank(tokenizer, num_tokens, args.prompt_len, rank, dp_size)
        actual_total = len(my_prompts) * dp_size * args.prompt_len

        if rank == 0:
            print(f"\n[tokens≈{actual_total}] {len(my_prompts) * dp_size} prompts "
                  f"({len(my_prompts)} per rank × {dp_size} ranks × {args.prompt_len} tok)")

        # Warmup — all ranks must participate simultaneously
        for _ in range(args.warmup):
            llm.generate(my_prompts, sampling_params)

        # Sync before profiling starts
        barrier.wait()

        # Snapshot existing trace files before profiling
        existing_traces = set(output_dir.glob("*.pt.trace.json.gz"))

        # All ranks start profiler
        llm.start_profile(profile_prefix=f"tokens_{actual_total}")

        # Rank 0 measures wall-clock time
        if rank == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        for _ in range(args.num_iters):
            llm.generate(my_prompts, sampling_params)

        if rank == 0:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / args.num_iters * 1000

        # Sync before stopping profiler
        barrier.wait()
        llm.stop_profile()

        if rank == 0:
            print(f"  Waiting for profiler to write ({args.profiler_wait}s) ...")
            time.sleep(args.profiler_wait)

            # Find and parse the new rank 0 trace file
            new_traces = find_new_traces(output_dir, existing_traces)
            rank0_trace = find_rank0_trace(new_traces)

            breakdown = dict(
                total_ms=0, dispatch_ms=0, compute_ms=0, combine_ms=0,
                dispatch_pct=0, compute_pct=0, combine_pct=0,
            )
            if rank0_trace:
                kernel_times = parse_trace_file(rank0_trace)
                breakdown = compute_moe_breakdown(kernel_times)
            else:
                print("  WARNING: no trace file found for this run")

            # Rename traces to include correct token count
            rename_traces(new_traces, actual_total)

            print(
                f"  → latency={elapsed_ms:.1f}ms | "
                f"cuda_total={breakdown['total_ms']:.1f}ms  "
                f"dispatch={breakdown['dispatch_ms']:.1f}ms ({breakdown['dispatch_pct']:.1f}%)  "
                f"compute={breakdown['compute_ms']:.1f}ms ({breakdown['compute_pct']:.1f}%)  "
                f"combine={breakdown['combine_ms']:.1f}ms ({breakdown['combine_pct']:.1f}%)"
            )
            results.append(dict(tokens=actual_total, latency_ms=elapsed_ms, **breakdown))

        # All ranks wait for rank 0 to finish before next iteration
        barrier.wait()

    if rank == 0:
        result_queue.put(results)


# ---------------------------------------------------------------------------
# Single-process path (dp_size == 1, no EP comm)
# ---------------------------------------------------------------------------

def _run_single(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    print(f"Loading model {args.model} (DP=1, single process, no EP comm) ...")
    llm_kwargs: dict = dict(
        model=args.model,
        data_parallel_size=1,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(output_dir),
        },
    )
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    token_sizes = [int(x) for x in args.token_sizes.split(",")]
    results = []

    for num_tokens in token_sizes:
        total_prompts = max(1, num_tokens // args.prompt_len)
        token_id = tokenizer.encode("hello", add_special_tokens=False)[0]
        prompts = [tokenizer.decode([token_id] * args.prompt_len)] * total_prompts
        actual_total = total_prompts * args.prompt_len

        print(f"\n[tokens≈{actual_total}] {total_prompts} prompts × {args.prompt_len} tok")

        for _ in range(args.warmup):
            llm.generate(prompts, sampling_params)

        result = _profile_one_token_size(
            llm, prompts, sampling_params, actual_total, output_dir, args,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MoE comm/compute sweep benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--dp-size", type=int, default=2,
                        help="DP size = EP size (number of GPUs). "
                             "1 = single process baseline (no EP comm). "
                             ">1 = multi-process with EP communication.")
    parser.add_argument("--token-sizes", default="2048,4096,8192,16384",
                        help="Comma-separated total token counts to sweep")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Tokens per prompt (fixed)")
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Timed iterations per data point")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (not profiled)")
    parser.add_argument("--output-dir", default="./vllm_profile/comm_sweep",
                        help="Output directory for profiler traces")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Disable chunked prefill by setting >= max batch tokens "
                             "(e.g. 32768). Default: vLLM auto (may chunk 16K batches).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDAGraph for cleaner profiling (default: off)")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager",
                        action="store_false", help="Enable CUDAGraph")
    parser.add_argument("--profiler-wait", type=int, default=15,
                        help="Seconds to wait after stop_profile() for trace write")
    args = parser.parse_args()

    dp_size = args.dp_size

    if dp_size == 1:
        results = _run_single(args)
    else:
        from vllm.utils.network_utils import get_open_port
        master_ip = "127.0.0.1"
        master_port = get_open_port()

        barrier = mp.Barrier(dp_size)
        result_queue: mp.Queue = mp.Queue()
        procs = []
        for rank in range(dp_size):
            p = mp.Process(
                target=_worker,
                args=(rank, dp_size, master_ip, master_port, args, barrier, result_queue),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        results = result_queue.get() if not result_queue.empty() else []

    # Summary
    if results:
        print("\n" + "=" * 120)
        print(f"  Model: {args.model}   DP=EP={dp_size}   prompt_len={args.prompt_len}")
        print(
            f"{'Tokens':>8}  {'Latency(ms)':>12}  {'CUDA(ms)':>10}  "
            f"{'Dispatch(ms)':>13}  {'Compute(ms)':>12}  {'Combine(ms)':>12}  "
            f"{'Dispatch%':>10}  {'Compute%':>9}  {'Combine%':>9}"
        )
        print("-" * 120)
        for r in results:
            print(
                f"{r['tokens']:>8}  {r['latency_ms']:>12.1f}  {r['total_ms']:>10.1f}  "
                f"{r['dispatch_ms']:>13.1f}  {r['compute_ms']:>12.1f}  {r['combine_ms']:>12.1f}  "
                f"{r['dispatch_pct']:>9.1f}%  {r['compute_pct']:>8.1f}%  {r['combine_pct']:>8.1f}%"
            )
        print("=" * 120)

    output_dir = Path(args.output_dir).resolve()
    print(f"\nTrace files: {output_dir}/tokens_*_*.pt.trace.json.gz")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
