"""Shared comparison utilities for output verification scripts."""

from __future__ import annotations

# Verdict thresholds (used by both compare_sem_moe_outputs.py and verify_tp_correctness.py)
EQUIVALENT_TOKEN_RATE = 0.95
EQUIVALENT_MAX_LP_DIFF = 0.01
MOSTLY_EQUIVALENT_TOKEN_RATE = 0.7


def verdict(avg_token_rate: float, avg_lp_diff: float) -> str:
    if avg_token_rate > EQUIVALENT_TOKEN_RATE and avg_lp_diff < EQUIVALENT_MAX_LP_DIFF:
        return "EQUIVALENT"
    elif avg_token_rate > MOSTLY_EQUIVALENT_TOKEN_RATE:
        return "MOSTLY_EQUIVALENT"
    else:
        return "DIVERGENT"


def token_match_rate(a_tokens: list, b_tokens: list) -> float:
    min_len = min(len(a_tokens), len(b_tokens))
    if min_len == 0:
        return 1.0
    matches = sum(1 for i in range(min_len) if a_tokens[i] == b_tokens[i])
    return matches / min_len


def first_divergence_index(a_tokens: list, b_tokens: list) -> int | None:
    min_len = min(len(a_tokens), len(b_tokens))
    for i in range(min_len):
        if a_tokens[i] != b_tokens[i]:
            return i
    if len(a_tokens) != len(b_tokens):
        return min_len
    return None
