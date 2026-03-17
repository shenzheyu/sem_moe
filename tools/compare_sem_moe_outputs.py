#!/usr/bin/env python3
"""Compare outputs between base vLLM and SEM_MOE vLLM serving instances."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from typing import Any

import requests

from comparison_utils import first_divergence_index, token_match_rate, verdict

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

TEST_PROMPTS: list[dict[str, Any]] = [
    {
        "name": "factual_short",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 32,
    },
    {
        "name": "math_reasoning",
        "messages": [
            {
                "role": "user",
                "content": "Solve step by step: If 3x + 7 = 22, what is x?",
            }
        ],
        "max_tokens": 128,
    },
    {
        "name": "code_generation",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to compute the nth Fibonacci number using iteration.",
            }
        ],
        "max_tokens": 256,
    },
    {
        "name": "explanation",
        "messages": [
            {
                "role": "user",
                "content": "Explain how photosynthesis works in three sentences.",
            }
        ],
        "max_tokens": 128,
    },
    {
        "name": "translation",
        "messages": [
            {
                "role": "user",
                "content": "Translate to French: The weather is beautiful today.",
            }
        ],
        "max_tokens": 64,
    },
    {
        "name": "list_planets",
        "messages": [
            {
                "role": "user",
                "content": "List the planets in our solar system in order from the sun.",
            }
        ],
        "max_tokens": 128,
    },
    {
        "name": "haiku",
        "messages": [
            {"role": "user", "content": "Write a haiku about machine learning."}
        ],
        "max_tokens": 64,
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
            {"role": "user", "content": "Now multiply that result by 3."},
        ],
        "max_tokens": 64,
    },
]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_health(base_url: str, label: str, retries: int = 3, wait: float = 2.0) -> bool:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{base_url}/health", timeout=10)
            if resp.status_code == 200:
                print(f"  [{label}] healthy")
                return True
            print(f"  [{label}] unhealthy (HTTP {resp.status_code}), attempt {attempt}/{retries}")
        except requests.exceptions.RequestException as exc:
            print(f"  [{label}] unreachable ({exc}), attempt {attempt}/{retries}")
        if attempt < retries:
            time.sleep(wait)
    return False


# ---------------------------------------------------------------------------
# Query server
# ---------------------------------------------------------------------------


def query_server(
    base_url: str,
    prompt: dict[str, Any],
    model: str,
    seed: int,
    timeout: int,
    top_logprobs: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "temperature": 0,
        "max_tokens": prompt.get("max_tokens", 128),
        "seed": seed,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "n": 1,
    }
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _extract_text(response: dict) -> str | None:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _extract_logprobs(response: dict) -> list[dict] | None:
    try:
        return response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def compare_tokens(base_resp: dict, sem_resp: dict) -> dict[str, Any]:
    base_text = _extract_text(base_resp)
    sem_text = _extract_text(sem_resp)
    if base_text is None or sem_text is None:
        return {"error": "could not extract text", "exact_match": False}

    base_lp = _extract_logprobs(base_resp) or []
    sem_lp = _extract_logprobs(sem_resp) or []

    base_tokens = [t["token"] for t in base_lp] if base_lp else list(base_text)
    sem_tokens = [t["token"] for t in sem_lp] if sem_lp else list(sem_text)

    match_rate = token_match_rate(base_tokens, sem_tokens)
    first_div = first_divergence_index(base_tokens, sem_tokens)

    result: dict[str, Any] = {
        "exact_match": base_text == sem_text,
        "token_match_rate": match_rate,
        "base_token_count": len(base_tokens),
        "sem_token_count": len(sem_tokens),
        "first_divergence": first_div,
    }
    if first_div is not None and first_div < min_len:
        result["div_base_token"] = base_tokens[first_div]
        result["div_sem_token"] = sem_tokens[first_div]

    return result


def compare_logprobs(base_resp: dict, sem_resp: dict) -> dict[str, Any]:
    base_lp = _extract_logprobs(base_resp)
    sem_lp = _extract_logprobs(sem_resp)
    if not base_lp or not sem_lp:
        return {"available": False}

    min_len = min(len(base_lp), len(sem_lp))
    diffs: list[float] = []
    jaccard_scores: list[float] = []

    for i in range(min_len):
        b_logprob = base_lp[i].get("logprob", 0.0)
        s_logprob = sem_lp[i].get("logprob", 0.0)
        if b_logprob is not None and s_logprob is not None:
            diffs.append(abs(b_logprob - s_logprob))

        b_top = {t["token"] for t in base_lp[i].get("top_logprobs", [])}
        s_top = {t["token"] for t in sem_lp[i].get("top_logprobs", [])}
        if b_top or s_top:
            union = b_top | s_top
            inter = b_top & s_top
            jaccard_scores.append(len(inter) / len(union) if union else 1.0)

    return {
        "available": True,
        "mean_logprob_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "max_logprob_diff": max(diffs) if diffs else 0.0,
        "mean_topk_jaccard": (
            sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 1.0
        ),
        "num_compared": min_len,
    }


def text_similarity(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    print("Checking server health ...")
    base_ok = check_health(args.base_url, "base")
    sem_ok = check_health(args.semmoe_url, "sem_moe")
    if not base_ok or not sem_ok:
        print("ERROR: one or both servers are not healthy. Aborting.")
        sys.exit(1)
    print()

    results: list[dict[str, Any]] = []
    for prompt in TEST_PROMPTS:
        name = prompt["name"]
        print(f"Testing [{name}] ...", end=" ", flush=True)

        base_resp = query_server(
            args.base_url, prompt, args.model, args.seed, args.timeout, args.top_logprobs
        )
        sem_resp = query_server(
            args.semmoe_url, prompt, args.model, args.seed, args.timeout, args.top_logprobs
        )

        if "error" in base_resp or "error" in sem_resp:
            err = base_resp.get("error") or sem_resp.get("error")
            print(f"ERROR: {err}")
            results.append({"name": name, "error": err})
            continue

        tok = compare_tokens(base_resp, sem_resp)
        lp = compare_logprobs(base_resp, sem_resp)
        sim = text_similarity(_extract_text(base_resp), _extract_text(sem_resp))

        status = "MATCH" if tok.get("exact_match") else "DIFF"
        print(status)

        results.append(
            {
                "name": name,
                "tokens": tok,
                "logprobs": lp,
                "text_similarity": sim,
                "base_text": _extract_text(base_resp),
                "sem_text": _extract_text(sem_resp),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print(f"{'Prompt':<20} {'Match':>6} {'TokRate':>8} {'TextSim':>8} "
          f"{'MeanLP':>10} {'MaxLP':>10} {'TopK-J':>8}")
    print("-" * 100)

    exact_matches = 0
    tok_rates: list[float] = []
    sims: list[float] = []
    lp_means: list[float] = []
    lp_maxes: list[float] = []
    topk_js: list[float] = []

    for r in results:
        name = r["name"]
        if "error" in r:
            print(f"{name:<20} {'ERR':>6}")
            continue

        tok = r["tokens"]
        lp = r["logprobs"]
        sim = r["text_similarity"]
        match_str = "YES" if tok["exact_match"] else "NO"
        tr = tok["token_match_rate"]
        lp_mean = lp.get("mean_logprob_diff", float("nan"))
        lp_max = lp.get("max_logprob_diff", float("nan"))
        topk_j = lp.get("mean_topk_jaccard", float("nan"))

        print(
            f"{name:<20} {match_str:>6} {tr:>8.4f} {sim:>8.4f} "
            f"{lp_mean:>10.6f} {lp_max:>10.6f} {topk_j:>8.4f}"
        )

        if tok["exact_match"]:
            exact_matches += 1
        tok_rates.append(tr)
        sims.append(sim)
        if lp.get("available"):
            lp_means.append(lp_mean)
            lp_maxes.append(lp_max)
            topk_js.append(topk_j)

    total = len([r for r in results if "error" not in r])
    print("-" * 100)
    print(f"Total prompts: {total}")
    print(f"Exact matches: {exact_matches}/{total} ({exact_matches / total * 100:.1f}%)" if total else "")
    if tok_rates:
        print(f"Mean token match rate: {sum(tok_rates) / len(tok_rates):.4f}")
    if sims:
        print(f"Mean text similarity:  {sum(sims) / len(sims):.4f}")
    if lp_means:
        print(f"Mean logprob diff:     {sum(lp_means) / len(lp_means):.6f}")
        print(f"Max logprob diff:      {max(lp_maxes):.6f}")
    if topk_js:
        print(f"Mean top-k Jaccard:    {sum(topk_js) / len(topk_js):.4f}")

    print()
    # Verdict
    avg_tr = sum(tok_rates) / len(tok_rates) if tok_rates else 0.0
    avg_lp = sum(lp_means) / len(lp_means) if lp_means else 0.0
    v = verdict(avg_tr, avg_lp)
    descriptions = {
        "EQUIVALENT": "outputs are numerically consistent (floating-point differences only)",
        "MOSTLY_EQUIVALENT": "minor divergences detected, likely from FP accumulation order",
        "DIVERGENT": "significant output differences; expert permutation may be incorrect",
    }
    print(f"VERDICT: {v} — {descriptions[v]}")
    print("=" * 100)

    # Show divergence details
    for r in results:
        if "error" in r:
            continue
        tok = r["tokens"]
        if not tok["exact_match"]:
            print(f"\n--- Divergence detail: [{r['name']}] ---")
            div_pos = tok.get("first_divergence")
            if div_pos is not None:
                print(f"  First divergence at token position {div_pos}")
                if "div_base_token" in tok:
                    print(f"    Base:    {tok['div_base_token']!r}")
                    print(f"    SemMoE:  {tok['div_sem_token']!r}")
            print(f"  Base output ({tok['base_token_count']} tokens):")
            print(f"    {r['base_text'][:200]}")
            print(f"  SemMoE output ({tok['sem_token_count']} tokens):")
            print(f"    {r['sem_text'][:200]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vLLM base vs SEM_MOE serving outputs"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--semmoe-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    results = run_comparison(args)
    print_summary(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to {args.output_json}")


if __name__ == "__main__":
    main()
