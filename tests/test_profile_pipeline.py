from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

from profile_collect import (
    _collect_worker_trace_records,
    _resolve_request_trace,
    infer_moe_layer_ids,
)
from profile_datasets import (
    DatasetSpec,
    extract_prompt_text,
    format_mmlu_prompt,
    iter_dataset_rows,
)
from profile_stats import (
    build_sparse_layer_artifact,
    build_vocab_extension_from_embedding,
    embedding_tensor_candidates,
)


class DummyConfig:
    num_hidden_layers = 4
    num_experts = 8
    num_experts_per_tok = 2
    decoder_sparse_step = 2
    mlp_only_layers = [0]


class ProfilePipelineTests(unittest.TestCase):
    def test_infer_moe_layer_ids_for_qwen_style_config(self) -> None:
        self.assertEqual(infer_moe_layer_ids(DummyConfig()), [1, 3])

    def test_format_mmlu_prompt(self) -> None:
        prompt = format_mmlu_prompt(
            {
                "subject": "abstract_algebra",
                "question": "What is 2 + 2?",
                "choices": ["1", "2", "3", "4"],
            }
        )
        self.assertIn("about abstract algebra.", prompt)
        self.assertIn("A. 1", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_extract_sharegpt_prompt(self) -> None:
        spec = DatasetSpec(
            name="sharegpt-vicuna-unfiltered",
            source="unused",
            split="train",
        )
        prompt = extract_prompt_text(
            spec,
            {
                "conversations": [
                    {"from": "human", "value": "Explain Mixture of Experts."},
                    {"from": "gpt", "value": "Sure."},
                ]
            },
        )
        self.assertEqual(prompt, "Explain Mixture of Experts.")

    def test_iter_hf_rows_falls_back_to_available_split(self) -> None:
        fake_datasets = types.ModuleType("datasets")
        load_calls: list[tuple[str, str | None, str, bool]] = []

        def fake_get_dataset_config_names(source: str) -> list[str]:
            self.assertEqual(source, "foo/bar")
            return []

        def fake_get_dataset_split_names(
            source: str,
            config_name: str | None = None,
        ) -> list[str]:
            self.assertEqual(source, "foo/bar")
            self.assertIsNone(config_name)
            return ["train"]

        def fake_load_dataset(
            source: str,
            name: str | None = None,
            split: str = "train",
            streaming: bool = False,
        ):
            load_calls.append((source, name, split, streaming))
            return iter(
                [
                    {
                        "question": "What is 2 + 2?",
                        "choices": ["3", "4"],
                    }
                ]
            )

        fake_datasets.get_dataset_config_names = fake_get_dataset_config_names
        fake_datasets.get_dataset_split_names = fake_get_dataset_split_names
        fake_datasets.load_dataset = fake_load_dataset

        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = list(
                iter_dataset_rows(
                    DatasetSpec(
                        name="custom-hf",
                        source="foo/bar",
                        split="test",
                    )
                )
            )

        self.assertEqual(load_calls, [("foo/bar", None, "train", True)])
        self.assertEqual(rows[0]["question"], "What is 2 + 2?")

    def test_iter_mmlu_rows_skip_configs_without_requested_split(self) -> None:
        fake_datasets = types.ModuleType("datasets")
        load_calls: list[tuple[str, str | None, str, bool]] = []

        def fake_get_dataset_config_names(source: str) -> list[str]:
            self.assertEqual(source, "cais/mmlu")
            return ["all", "abstract_algebra", "auxiliary_train"]

        def fake_get_dataset_split_names(
            source: str,
            config_name: str | None = None,
        ) -> list[str]:
            self.assertEqual(source, "cais/mmlu")
            if config_name == "abstract_algebra":
                return ["test"]
            if config_name == "auxiliary_train":
                return ["train"]
            self.fail(f"Unexpected config_name: {config_name}")

        def fake_load_dataset(
            source: str,
            name: str | None = None,
            split: str = "train",
            streaming: bool = False,
        ):
            load_calls.append((source, name, split, streaming))
            return iter(
                [
                    {
                        "question": "What is 2 + 2?",
                        "choices": ["3", "4"],
                        "subject": "abstract algebra",
                    }
                ]
            )

        fake_datasets.get_dataset_config_names = fake_get_dataset_config_names
        fake_datasets.get_dataset_split_names = fake_get_dataset_split_names
        fake_datasets.load_dataset = fake_load_dataset

        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            rows = list(
                iter_dataset_rows(
                    DatasetSpec(
                        name="mmlu",
                        source="cais/mmlu",
                        split="test",
                    )
                )
            )

        self.assertEqual(load_calls, [("cais/mmlu", "abstract_algebra", "test", True)])
        self.assertEqual(rows[0]["subject"], "abstract algebra")

    def test_format_mmlu_prompt_unwraps_nested_row(self) -> None:
        prompt = format_mmlu_prompt(
            {
                "train": {
                    "subject": "abstract_algebra",
                    "question": "What is 2 + 2?",
                    "choices": ["1", "2", "3", "4"],
                }
            }
        )
        self.assertIn("about abstract algebra.", prompt)
        self.assertIn("D. 4", prompt)

    def test_collect_worker_trace_records_concatenates_chunks(self) -> None:
        first_trace_path = Path(self.id() + "_trace_first.pt")
        second_trace_path = Path(self.id() + "_trace_second.pt")
        try:
            torch.save(
                {
                    "records": [
                        {
                            "request_id": "req-1",
                            "routed_experts": torch.tensor([[[1, 2]]], dtype=torch.int32),
                        }
                    ]
                },
                first_trace_path,
            )
            torch.save(
                {
                    "records": [
                        {
                            "request_id": "req-1",
                            "routed_experts": torch.tensor([[[3, 4]]], dtype=torch.int32),
                        },
                        {
                            "request_id": "req-2",
                            "routed_experts": torch.tensor([[[5, 6]]], dtype=torch.int32),
                        },
                    ]
                },
                second_trace_path,
            )
            traces = _collect_worker_trace_records([first_trace_path, second_trace_path])
        finally:
            first_trace_path.unlink(missing_ok=True)
            second_trace_path.unlink(missing_ok=True)

        self.assertEqual(set(traces), {"req-1", "req-2"})
        self.assertTrue(
            torch.equal(
                traces["req-1"],
                torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                traces["req-2"],
                torch.tensor([[[5, 6]]], dtype=torch.int32),
            )
        )

    def test_resolve_request_trace_validates_lengths(self) -> None:
        request_output = types.SimpleNamespace(
            request_id="req-1",
            prompt_token_ids=[10, 11],
        )
        traces = {
            "req-1": torch.tensor([[[1, 2]]], dtype=torch.int32),
        }

        with self.assertRaisesRegex(RuntimeError, "Prompt-token and routed-expert lengths differ"):
            _resolve_request_trace(request_output, traces)

    def test_resolve_request_trace_requires_request_id(self) -> None:
        request_output = types.SimpleNamespace(
            request_id="req-1",
            prompt_token_ids=[10],
        )

        with self.assertRaisesRegex(RuntimeError, "Missing routed expert trace"):
            _resolve_request_trace(request_output, {})

    def test_build_sparse_layer_artifact(self) -> None:
        count_by_token = {
            10: {1: 3, 2: 1},
            11: {3: 1, 4: 1},
        }
        freq_by_token = {10: 2, 11: 1}
        artifact = build_sparse_layer_artifact(
            count_by_token=count_by_token,
            freq_by_token=freq_by_token,
            token_to_index={10: 0, 11: 1},
            num_seen_tokens=2,
        )
        self.assertTrue(torch.equal(artifact["freq"], torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(artifact["row_splits"], torch.tensor([0, 2, 4], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                artifact["count_expert_ids"],
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                artifact["count_values"],
                torch.tensor([3, 1, 1, 1], dtype=torch.int32),
            )
        )
        expected_cp = torch.tensor([0.75, 0.25, 0.5, 0.5], dtype=torch.float32)
        self.assertTrue(torch.allclose(artifact["cp_values"], expected_cp))
        expected_a = torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(artifact["a_values"], expected_a))

    def test_build_vocab_extension_from_embedding(self) -> None:
        embedding = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        extension = build_vocab_extension_from_embedding(
            embedding=embedding,
            seen_token_ids=torch.tensor([0, 2], dtype=torch.long),
            query_batch_size=2,
            device="cpu",
            vocab_limit=None,
        )
        self.assertTrue(
            torch.equal(
                extension["nearest_seen_token_id"],
                torch.tensor([0, 0, 2, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(extension["exact_seen_mask"][0].item())
        self.assertTrue(extension["exact_seen_mask"][2].item())
        self.assertFalse(extension["exact_seen_mask"][1].item())

    def test_embedding_tensor_candidates_support_qwen35_multimodal_layout(self) -> None:
        self.assertIn(
            "model.language_model.embed_tokens.weight",
            embedding_tensor_candidates(),
        )


if __name__ == "__main__":
    unittest.main()
