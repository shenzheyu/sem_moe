from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from artifacts import (
    collection_manifest_path,
    raw_shard_path,
    schedule_layer_path,
    schedule_manifest_path,
    save_torch_artifact,
    write_json,
)
from eval_dp import evaluate_dp_scheduling_from_run


class ProfileOnlineDPTests(unittest.TestCase):
    def test_evaluate_dp_scheduling_prefers_semmoe_for_clustered_requests(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            (run_dir / "raw").mkdir(parents=True)
            (run_dir / "schedule").mkdir()

            write_json(
                collection_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "raw_shards": ["raw/raw_shard_00000.pt"],
                },
            )
            write_json(
                schedule_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "model_name": "dummy/model",
                    "moe_layer_ids": [0, 1],
                    "num_devices": 2,
                },
            )

            score_full = np.zeros((16, 2), dtype=np.float32)
            score_full[10] = [4.0, 0.0]
            score_full[11] = [0.0, 4.0]
            for layer_id in (0, 1):
                np.savez_compressed(
                    schedule_layer_path(run_dir, layer_id),
                    E=np.array([0, 0, 1, 1], dtype=np.int32),
                    T_score_full=score_full,
                )

            save_torch_artifact(
                raw_shard_path(run_dir, 0),
                {
                    "records": [
                        {
                            "record_id": "req-0",
                            "prompt_token_ids": torch.tensor([10, 10], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[0, 1], [0, 1]],
                                    [[0, 1], [0, 1]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "record_id": "req-1",
                            "prompt_token_ids": torch.tensor([11, 11], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[2, 3], [2, 3]],
                                    [[2, 3], [2, 3]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "record_id": "req-2",
                            "prompt_token_ids": torch.tensor([11, 11], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[2, 3], [2, 3]],
                                    [[2, 3], [2, 3]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "record_id": "req-3",
                            "prompt_token_ids": torch.tensor([10, 10], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [
                                    [[0, 1], [0, 1]],
                                    [[0, 1], [0, 1]],
                                ],
                                dtype=torch.int32,
                            ),
                        },
                    ]
                },
            )

            result = evaluate_dp_scheduling_from_run(
                run_dir,
                include_layerwise_diagnostic=True,
            )

        baseline = result["baseline"]
        semmoe = result["semmoe"]
        semmoe_layerwise = result["semmoe_layerwise"]
        self.assertLess(semmoe["remote_activations"], baseline["remote_activations"])
        self.assertGreater(semmoe["lar"], baseline["lar"])
        self.assertEqual(semmoe["token_load_per_device"], [4, 4])
        self.assertGreaterEqual(semmoe_layerwise["lar"], semmoe["lar"])

    def test_layerwise_diagnostic_reveals_cross_layer_assignment_conflicts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            (run_dir / "raw").mkdir(parents=True)
            (run_dir / "schedule").mkdir()

            write_json(
                collection_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "raw_shards": ["raw/raw_shard_00000.pt"],
                },
            )
            write_json(
                schedule_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "model_name": "dummy/model",
                    "moe_layer_ids": [0, 1],
                    "num_devices": 2,
                },
            )

            layer0_score = np.zeros((16, 2), dtype=np.float32)
            layer0_score[10] = [4.0, 0.0]
            layer0_score[11] = [0.0, 4.0]
            layer1_score = np.zeros((16, 2), dtype=np.float32)
            layer1_score[10] = [0.0, 1.0]
            layer1_score[11] = [4.0, 4.0]

            np.savez_compressed(
                schedule_layer_path(run_dir, 0),
                E=np.array([0, 0, 1, 1], dtype=np.int32),
                T_score_full=layer0_score,
            )
            np.savez_compressed(
                schedule_layer_path(run_dir, 1),
                E=np.array([0, 0, 1, 1], dtype=np.int32),
                T_score_full=layer1_score,
            )

            save_torch_artifact(
                raw_shard_path(run_dir, 0),
                {
                    "records": [
                        {
                            "record_id": "req-0",
                            "prompt_token_ids": torch.tensor([10], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [[[0, 1], [2, 3]]],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "record_id": "req-1",
                            "prompt_token_ids": torch.tensor([10], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [[[0, 1], [2, 3]]],
                                dtype=torch.int32,
                            ),
                        },
                        {
                            "record_id": "req-2",
                            "prompt_token_ids": torch.tensor([11], dtype=torch.int32),
                            "routed_experts": torch.tensor(
                                [[[2, 3], [0, 1]]],
                                dtype=torch.int32,
                            ),
                        },
                    ]
                },
            )

            result = evaluate_dp_scheduling_from_run(
                run_dir,
                include_layerwise_diagnostic=True,
            )

        self.assertGreater(result["semmoe_layerwise"]["lar"], result["semmoe"]["lar"])
        self.assertGreater(
            result["semmoe_layerwise"]["per_layer"]["1"]["lar"],
            result["semmoe"]["per_layer"]["1"]["lar"],
        )

    def test_result_is_json_serializable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            (run_dir / "raw").mkdir(parents=True)
            (run_dir / "schedule").mkdir()
            write_json(
                collection_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "raw_shards": ["raw/raw_shard_00000.pt"],
                },
            )
            write_json(
                schedule_manifest_path(run_dir),
                {
                    "format_version": 1,
                    "run_dir": str(run_dir),
                    "run_name": "run",
                    "model_name": "dummy/model",
                    "moe_layer_ids": [0],
                    "num_devices": 2,
                },
            )
            np.savez_compressed(
                schedule_layer_path(run_dir, 0),
                E=np.array([0, 0, 1, 1], dtype=np.int32),
                T_score_full=np.zeros((8, 2), dtype=np.float32),
            )
            save_torch_artifact(
                raw_shard_path(run_dir, 0),
                {
                    "records": [
                        {
                            "record_id": "req-0",
                            "prompt_token_ids": torch.tensor([1], dtype=torch.int32),
                            "routed_experts": torch.tensor([[[0, 1]]], dtype=torch.int32),
                        }
                    ]
                },
            )

            result = evaluate_dp_scheduling_from_run(
                run_dir,
                include_layerwise_diagnostic=True,
            )

        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
