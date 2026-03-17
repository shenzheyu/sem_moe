#!/usr/bin/env python3
"""Verify Sem-MoE TP correctness by comparing outputs across modes.

Usage:
    # Run single mode and save results:
    python verify_tp_correctness.py --run-mode baseline --save results_baseline.json
    python verify_tp_correctness.py --run-mode debug_fallback --save results_debug.json
    python verify_tp_correctness.py --run-mode srs_nccl --save results_srs.json

    # Compare saved results:
    python verify_tp_correctness.py --compare results_baseline.json results_debug.json results_srs.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

from comparison_utils import first_divergence_index, token_match_rate, verdict

TEST_PROMPTS = [
    "What is the capital of France?",
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain how photosynthesis works in three sentences.",
    "List the planets in our solar system in order from the sun.",
    "Write a haiku about machine learning.",
    "What is 2 + 2? Now multiply that result by 3.",
    "Translate to French: The weather is beautiful today.",
]

MAX_TOKENS = 64


@dataclass
class TokenResult:
    prompt: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


def run_single_mode(args):
    """Run inference for a single mode and save results."""
    mode = args.run_mode
    model = args.model
    tp = args.tp_size

    # Environment is set BEFORE this script is called (via wrapper)
    from vllm import LLM, SamplingParams

    print(f"[{mode}] Loading model {model} with TP={tp} ...")
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        seed=42,
        logprobs=1,
    )

    t0 = time.time()
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.80,
    )
    print(f"[{mode}] Model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    gen_time = time.time() - t0
    print(f"[{mode}] Generated {len(outputs)} outputs in {gen_time:.1f}s")

    results = []
    for output in outputs:
        text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        lps = []
        if output.outputs[0].logprobs:
            for lp_dict in output.outputs[0].logprobs:
                if lp_dict:
                    for tok_id, lp_info in lp_dict.items():
                        lps.append(lp_info.logprob if hasattr(lp_info, 'logprob') else float(lp_info))
                        break
        results.append(asdict(TokenResult(
            prompt=output.prompt,
            text=text,
            token_ids=token_ids,
            logprobs=lps,
        )))

    # Print outputs
    for r in results:
        print(f"\n  Prompt: {r['prompt'][:60]}...")
        print(f"  Output: {r['text'][:120]}")
        print(f"  Tokens: {len(r['token_ids'])}")

    save_path = args.save
    payload = {"mode": mode, "model": model, "tp_size": tp, "results": results}
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[{mode}] Results saved to {save_path}")


def compare_files(args):
    """Compare results from multiple saved files."""
    files = args.compare
    data = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
            data[d["mode"]] = d["results"]

    modes = list(data.keys())
    if "baseline" not in data:
        print("ERROR: baseline results not found. Available modes:", modes)
        sys.exit(1)

    base = data["baseline"]
    print(f"\n{'='*80}")
    print(f"{'Mode':<20} {'ExactMatch':>10} {'TokRate':>8} {'MeanLP':>10} {'MaxLP':>10} {'Verdict':<20}")
    print("-" * 80)

    for mode in modes:
        if mode == "baseline":
            continue
        test = data[mode]
        exact = 0
        tok_rates = []
        lp_diffs = []

        for b, t in zip(base, test):
            if b["text"] == t["text"]:
                exact += 1
            tok_rates.append(token_match_rate(b["token_ids"], t["token_ids"]))
            lp_min = min(len(b["logprobs"]), len(t["logprobs"]))
            for i in range(lp_min):
                lp_diffs.append(abs(b["logprobs"][i] - t["logprobs"][i]))

        avg_rate = sum(tok_rates) / len(tok_rates) if tok_rates else 0
        avg_lp = sum(lp_diffs) / len(lp_diffs) if lp_diffs else 0
        max_lp = max(lp_diffs) if lp_diffs else 0
        total = len(base)
        v = verdict(avg_rate, avg_lp)

        print(f"{mode:<20} {exact}/{total:>8} {avg_rate:>8.4f} {avg_lp:>10.6f} {max_lp:>10.6f} {v:<20}")

        # Show divergences
        for b, t in zip(base, test):
            if b["text"] != t["text"]:
                print(f"\n  DIFF [{b['prompt'][:50]}...]")
                first_div = first_divergence_index(b["token_ids"], t["token_ids"])
                if first_div is not None:
                    print(f"    First divergence at token {first_div}")
                print(f"    Base: {b['text'][:100]}")
                print(f"    Test: {t['text'][:100]}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Verify Sem-MoE TP correctness")
    sub = parser.add_subparsers(dest="cmd")

    # Run mode
    run_p = sub.add_parser("run", help="Run inference for one mode")
    run_p.add_argument("--run-mode", required=True,
                       choices=["baseline", "debug_fallback", "srs_nccl", "srs_triton"])
    run_p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    run_p.add_argument("--tp-size", type=int, default=2)
    run_p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    run_p.add_argument("--save", required=True, help="Output JSON file")

    # Compare mode
    cmp_p = sub.add_parser("compare", help="Compare saved results")
    cmp_p.add_argument("--compare", nargs="+", required=True, help="JSON files to compare")

    args = parser.parse_args()
    if args.cmd == "run":
        run_single_mode(args)
    elif args.cmd == "compare":
        compare_files(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
