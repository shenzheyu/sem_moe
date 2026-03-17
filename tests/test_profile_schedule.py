from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest

import numpy as np
import torch

from artifacts import (
    collection_manifest_path,
    raw_shard_path,
    save_torch_artifact,
    schedule_layer_path,
    schedule_manifest_path,
    stats_artifact_path,
    vocab_artifact_path,
    write_json,
)
from schedule import (
    ProfileRequest,
    ScheduleBuildConfig,
    build_activation_transition_tables,
    build_request_profiles,
    build_token_cluster_scores,
    compute_schedule_objective,
    encode_device_sequence,
    extend_token_scores_to_full_vocab,
    init_expert_seed_assignment,
    majority_vote_device,
    request_schedule,
    run_build_model_schedule,
    solve_layer_schedule,
)
from profile_stats import build_sparse_layer_artifact


class ProfileScheduleTests(unittest.TestCase):
    def test_extend_token_scores_to_full_vocab_exact_copy(self) -> None:
        scores = torch.tensor(
            [
                [1.0, 0.0],
                [0.25, 0.75],
            ],
            dtype=torch.float32,
        )
        extended = extend_token_scores_to_full_vocab(
            t_score_seen=scores,
            seen_token_ids=torch.tensor([10, 11], dtype=torch.long),
            vocab_artifact={
                "nearest_seen_token_id": torch.tensor([10, 10, 11], dtype=torch.int32),
            },
        )
        self.assertTrue(torch.equal(extended[0], scores[0]))
        self.assertTrue(torch.equal(extended[1], scores[0]))
        self.assertTrue(torch.equal(extended[2], scores[1]))

    def test_majority_vote_device_breaks_ties_toward_lower_id(self) -> None:
        device_id = majority_vote_device(torch.tensor([1, 0], dtype=torch.long), num_devices=2)
        self.assertEqual(int(device_id), 0)

    def test_build_activation_transition_tables_counts_discrete_sequences(self) -> None:
        requests = [
            ProfileRequest(
                record_id="req-0",
                token_indices=torch.tensor([0], dtype=torch.long),
                routed_experts=torch.tensor([[[0, 1], [2, 3], [0, 1]]], dtype=torch.long),
            ),
            ProfileRequest(
                record_id="req-1",
                token_indices=torch.tensor([1], dtype=torch.long),
                routed_experts=torch.tensor([[[0, 1], [2, 3], [2, 3]]], dtype=torch.long),
            ),
        ]
        tables = build_activation_transition_tables(
            requests=requests,
            moe_layer_ids=[0, 1, 2],
            expert_labels_by_layer={
                0: torch.tensor([0, 0, 1, 1], dtype=torch.long),
                1: torch.tensor([0, 0, 1, 1], dtype=torch.long),
                2: torch.tensor([0, 0, 1, 1], dtype=torch.long),
            },
            num_devices=2,
            lookback=2,
        )
        seq_id = encode_device_sequence([0, 1], base=2)
        self.assertEqual(tuple(tables[0].shape), (4, 2))
        self.assertEqual(tuple(tables[1].shape), (4, 2))
        self.assertTrue(torch.allclose(tables[2][seq_id], torch.tensor([0.5, 0.5])))

    def test_solve_layer_schedule_balances_experts_and_keeps_rows_normalized(self) -> None:
        cp = torch.tensor(
            [
                [0.45, 0.45, 0.05, 0.05],
                [0.40, 0.40, 0.10, 0.10],
                [0.05, 0.05, 0.45, 0.45],
                [0.10, 0.10, 0.40, 0.40],
            ],
            dtype=torch.float32,
        )
        a = torch.tensor([0.3, 0.2, 0.3, 0.2], dtype=torch.float32)
        requests = [
            ProfileRequest(
                record_id="req-0",
                token_indices=torch.tensor([0, 1], dtype=torch.long),
                routed_experts=torch.zeros((2, 3, 2), dtype=torch.long),
            ),
            ProfileRequest(
                record_id="req-1",
                token_indices=torch.tensor([0], dtype=torch.long),
                routed_experts=torch.zeros((1, 3, 2), dtype=torch.long),
            ),
            ProfileRequest(
                record_id="req-2",
                token_indices=torch.tensor([2, 3], dtype=torch.long),
                routed_experts=torch.zeros((2, 3, 2), dtype=torch.long),
            ),
            ProfileRequest(
                record_id="req-3",
                token_indices=torch.tensor([3], dtype=torch.long),
                routed_experts=torch.zeros((1, 3, 2), dtype=torch.long),
            ),
        ]
        req_profiles, req_lengths = build_request_profiles(cp, requests)
        config = ScheduleBuildConfig(
            run_dir=Path("/tmp/unused"),
            num_devices=2,
            lookback=2,
            seed=0,
            n_steps=4,
            ft_steps=8,
            alpha_e=1.0,
            beta_e=1.0,
            gamma_e=1.0,
            alpha_r=1.0,
            beta_r=1.0,
            theta=0.5,
            show_progress=False,
        )

        init_expert = init_expert_seed_assignment(torch.matmul(a, cp), config.num_devices)
        init_request = request_schedule(
            req_profiles=req_profiles,
            req_lengths=req_lengths,
            expert_labels=init_expert,
            num_devices=config.num_devices,
            alpha_r=config.alpha_r,
            beta_r=config.beta_r,
        )
        init_score = build_token_cluster_scores(cp.shape[0], config.num_devices, requests, init_request)
        init_objective = compute_schedule_objective(
            a=a,
            cp=cp,
            token_cluster_scores=init_score,
            expert_labels=init_expert,
            num_devices=config.num_devices,
            theta=config.theta,
        )

        expert_labels, _, token_scores, objective = solve_layer_schedule(
            cp=cp,
            a=a,
            req_profiles=req_profiles,
            req_lengths=req_lengths,
            requests=requests,
            config=config,
            layer_seed=0,
            layer_id=0,
        )

        self.assertTrue(
            torch.equal(torch.bincount(expert_labels, minlength=2), torch.tensor([2, 2]))
        )
        self.assertTrue(torch.allclose(token_scores.sum(dim=1), torch.ones(4)))
        self.assertLessEqual(objective, init_objective + 1e-6)

    def test_run_build_model_schedule_writes_schedule_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            (run_dir / "raw").mkdir(parents=True)
            (run_dir / "stats").mkdir()
            (run_dir / "vocab").mkdir()

            write_json(
                collection_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "raw_shards": ["raw/raw_shard_00000.pt"],
                    "config": {"model": "dummy/model"},
                    "model_metadata": {
                        "model_name": "dummy/model",
                        "model_type": "dummy",
                        "num_hidden_layers": 3,
                        "num_experts": 4,
                        "top_k": 2,
                        "moe_layer_ids": [0, 1, 2],
                    },
                },
            )

            layer_artifact = build_sparse_layer_artifact(
                count_by_token={10: {0: 3, 1: 3}, 11: {2: 3, 3: 3}},
                freq_by_token={10: 3, 11: 3},
                token_to_index={10: 0, 11: 1},
                num_seen_tokens=2,
            )
            save_torch_artifact(
                stats_artifact_path(run_dir),
                {
                    "metadata": {
                        "model_name": "dummy/model",
                        "model_type": "dummy",
                        "num_hidden_layers": 3,
                        "num_experts": 4,
                        "top_k": 2,
                        "moe_layer_ids": [0, 1, 2],
                        "raw_trace_shards": ["raw/raw_shard_00000.pt"],
                    },
                    "seen_token_ids": torch.tensor([10, 11], dtype=torch.int32),
                    "layers": {
                        "0": layer_artifact,
                        "1": layer_artifact,
                        "2": layer_artifact,
                    },
                },
            )
            save_torch_artifact(
                vocab_artifact_path(run_dir),
                {
                    "seen_token_ids": torch.tensor([10, 11], dtype=torch.int32),
                    "nearest_seen_token_id": torch.tensor(
                        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11],
                        dtype=torch.int32,
                    ),
                    "nearest_similarity": torch.ones(12, dtype=torch.float32),
                    "exact_seen_mask": torch.tensor(
                        [False] * 10 + [True, True],
                        dtype=torch.bool,
                    ),
                    "effective_vocab_size": torch.tensor(12, dtype=torch.int32),
                    "full_vocab_size": torch.tensor(12, dtype=torch.int32),
                },
            )
            save_torch_artifact(
                raw_shard_path(run_dir, 0),
                {
                    "metadata": {"run_name": "run", "model_name": "dummy/model"},
                    "records": [
                        {
                            "dataset_name": "dummy",
                            "record_id": "req-0",
                            "prompt_token_ids": torch.tensor([10, 10], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[0, 1], [0, 1], [0, 1]],
                                    [[0, 1], [0, 1], [0, 1]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "dataset_name": "dummy",
                            "record_id": "req-1",
                            "prompt_token_ids": torch.tensor([11, 11], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[2, 3], [2, 3], [2, 3]],
                                    [[2, 3], [2, 3], [2, 3]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                    ],
                },
            )

            run_build_model_schedule(
                types.SimpleNamespace(
                    run_dir=run_dir,
                    num_devices=2,
                    lookback=2,
                    seed=0,
                    n_steps=2,
                    ft_steps=4,
                    alpha_e=1.0,
                    beta_e=1.0,
                    gamma_e=1.0,
                    alpha_r=1.0,
                    beta_r=1.0,
                    theta=0.5,
                )
            )

            self.assertTrue(schedule_manifest_path(run_dir).exists())
            self.assertTrue(schedule_layer_path(run_dir, 0).exists())
            with np.load(schedule_layer_path(run_dir, 2)) as payload:
                self.assertIn("E", payload)
                self.assertIn("T_score_full", payload)
                self.assertIn("A_prob", payload)
                self.assertEqual(payload["T_score_full"].shape, (12, 2))
                self.assertTrue(np.array_equal(np.bincount(payload["E"], minlength=2), [2, 2]))
                self.assertEqual(payload["A_prob"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
