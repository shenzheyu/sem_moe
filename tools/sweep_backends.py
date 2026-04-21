"""
Sweep MoE comm/compute across multiple all2all backends and token sizes.

Wraps tools/moe_comm_sweep.py: for each backend, invokes a fresh subprocess,
parses the summary table from stdout, and prints a side-by-side comparison.

Usage (2-GPU, EP=2, compare AgRs vs DeepEP-HT):
  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python tools/sweep_backends.py \
      --model Qwen/Qwen3.5-35B-A3B \
      --dp-size 2 \
      --backends allgather_reducescatter,deepep_high_throughput \
      --token-sizes 2048,4096,8192,16384 \
      --output-root ./vllm_profile/backend_sweep
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMM_SWEEP = SCRIPT_DIR / "moe_comm_sweep.py"

# Example row from moe_comm_sweep summary:
#     2048         45.3       38.2          12.1          18.5          7.6       31.7%     48.4%     19.9%
ROW_RE = re.compile(
    r"^\s*(\d+)\s+"         # tokens
    r"([\d.]+)\s+"          # latency_ms
    r"([\d.]+)\s+"          # total_ms
    r"([\d.]+)\s+"          # dispatch_ms
    r"([\d.]+)\s+"          # compute_ms
    r"([\d.]+)\s+"          # combine_ms
    r"([\d.]+)%\s+"         # dispatch_pct
    r"([\d.]+)%\s+"         # compute_pct
    r"([\d.]+)%\s*$"        # combine_pct
)

COL_NAMES = ["tokens", "latency_ms", "total_ms",
             "dispatch_ms", "compute_ms", "combine_ms",
             "dispatch_pct", "compute_pct", "combine_pct"]


def run_backend(backend: str, output_dir: Path, args: argparse.Namespace) -> list[dict] | None:
    """Invoke moe_comm_sweep.py for one backend, parse stdout, return rows."""
    cmd = [
        sys.executable, str(COMM_SWEEP),
        "--model", args.model,
        "--dp-size", str(args.dp_size),
        "--all2all-backend", backend,
        "--token-sizes", args.token_sizes,
        "--prompt-len", str(args.prompt_len),
        "--num-iters", str(args.num_iters),
        "--warmup", str(args.warmup),
        "--output-dir", str(output_dir),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--profiler-wait", str(args.profiler_wait),
    ]
    if args.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    print(f"\n{'#' * 80}")
    print(f"# Backend: {backend}")
    print(f"# Output:  {output_dir}")
    print(f"# Command: {' '.join(cmd)}")
    print(f"{'#' * 80}\n", flush=True)

    # Stream child output so user sees progress; capture it for parsing.
    captured: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)
    proc.wait()

    if proc.returncode != 0:
        print(f"\n!! backend {backend} failed (exit {proc.returncode})")
        return None

    rows = parse_summary(captured)
    if not rows:
        print(f"\n!! no summary rows parsed for backend {backend}")
        return None
    return rows


def parse_summary(lines: list[str]) -> list[dict]:
    """Extract numeric rows from moe_comm_sweep.py's summary table."""
    rows: list[dict] = []
    for line in lines:
        m = ROW_RE.match(line)
        if not m:
            continue
        vals = [float(x) for x in m.groups()]
        vals[0] = int(vals[0])  # tokens is int
        rows.append(dict(zip(COL_NAMES, vals)))
    return rows


def print_comparison(all_results: dict[str, list[dict]], metric: str, unit: str = "ms") -> None:
    """Side-by-side table for one metric across backends."""
    backends = list(all_results.keys())
    if not backends:
        return

    # Collect unique token sizes (preserve order from the first backend).
    tokens_seen: list[int] = []
    seen_set: set[int] = set()
    for rows in all_results.values():
        for r in rows:
            if r["tokens"] not in seen_set:
                seen_set.add(r["tokens"])
                tokens_seen.append(r["tokens"])

    # Index by (backend, tokens) -> value.
    idx = {(b, r["tokens"]): r[metric] for b, rows in all_results.items() for r in rows}

    header = f"========== {metric} ({unit}) =========="
    print(f"\n{header}")
    width = max(24, max(len(b) for b in backends) + 2)
    print(f"{'Tokens':>8}  " + "".join(f"{b:>{width}}" for b in backends) + f"{'speedup':>10}")
    print("-" * (10 + width * len(backends) + 10))
    for tok in tokens_seen:
        vals = [idx.get((b, tok)) for b in backends]
        row = f"{tok:>8}  "
        row += "".join(
            f"{v:>{width}.2f}" if v is not None else f"{'N/A':>{width}}"
            for v in vals
        )
        # Speedup: first backend / last backend (how much faster the last is)
        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None and vals[-1] > 0:
            row += f"{vals[0] / vals[-1]:>9.2f}x"
        else:
            row += f"{'N/A':>10}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep all2all backends via moe_comm_sweep")
    parser.add_argument("--backends", default="allgather_reducescatter,deepep_high_throughput",
                        help="Comma-separated all2all backends to sweep")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--token-sizes", default="2048,4096,8192,16384")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-root", default="./vllm_profile/backend_sweep",
                        help="Parent dir; each backend gets its own subdir under this")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--profiler-wait", type=int, default=15)
    args = parser.parse_args()

    if not COMM_SWEEP.exists():
        print(f"!! cannot find {COMM_SWEEP}", file=sys.stderr)
        sys.exit(1)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    all_results: dict[str, list[dict]] = {}

    for backend in backends:
        out = output_root / backend
        out.mkdir(parents=True, exist_ok=True)
        rows = run_backend(backend, out, args)
        if rows is not None:
            all_results[backend] = rows

    if not all_results:
        print("\n!! no backend succeeded", file=sys.stderr)
        sys.exit(1)

    # Save combined results for downstream analysis.
    combined = output_root / "combined_results.json"
    with open(combined, "w") as f:
        json.dump({
            "model": args.model,
            "dp_size": args.dp_size,
            "prompt_len": args.prompt_len,
            "token_sizes": args.token_sizes,
            "per_backend": all_results,
        }, f, indent=2)

    print(f"\n{'#' * 80}")
    print(f"# Comparison across backends (model={args.model}, DP=EP={args.dp_size})")
    print(f"{'#' * 80}")
    for metric, unit in [
        ("latency_ms", "ms"),
        ("total_ms", "ms CUDA"),
        ("dispatch_ms", "ms"),
        ("compute_ms", "ms"),
        ("combine_ms", "ms"),
    ]:
        print_comparison(all_results, metric, unit)

    print(f"\nCombined results: {combined}")


if __name__ == "__main__":
    main()
