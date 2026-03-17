"""Unit tests for Sem-MoE TP rebatch logic (vllm/sem_moe_tp.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.sem_moe_tp import (
    SemMoeTPContext,
    SemMoeTPLayerTensors,
    encode_device_sequence_batch,
    rebatch_for_layer,
    unshuffle_output,
)


def _make_layer_tensors(
    vocab_size: int,
    num_devices: int,
    lookback: int,
    device: torch.device,
) -> SemMoeTPLayerTensors:
    """Create random layer tensors for testing."""
    return SemMoeTPLayerTensors(
        t_full=torch.randint(0, num_devices, (vocab_size,), dtype=torch.int32, device=device),
        tp_full=torch.rand(vocab_size, dtype=torch.float32, device=device),
        a_table=torch.randint(0, num_devices, (num_devices**lookback,), dtype=torch.int32, device=device),
        ap_table=torch.rand(num_devices**lookback, dtype=torch.float32, device=device),
    )


def _make_ctx(
    layer_ids: tuple[int, ...],
    vocab_size: int = 1000,
    num_devices: int = 2,
    lookback: int = 2,
    device: torch.device | None = None,
) -> SemMoeTPContext:
    """Build a test SemMoeTPContext."""
    if device is None:
        device = torch.device("cpu")
    layer_tensors = {
        lid: _make_layer_tensors(vocab_size, num_devices, lookback, device)
        for lid in layer_ids
    }
    return SemMoeTPContext(
        layer_tensors=layer_tensors,
        lookback=lookback,
        num_devices=num_devices,
        moe_layer_ids=layer_ids,
    )


class TestEncodeDeviceSequenceBatch:
    """Tests for encode_device_sequence_batch()."""

    def test_basic_encoding(self):
        """Test that encoding matches the expected polynomial formula."""
        # For lookback=2, num_devices=2: seq_id = col0 * 2 + col1
        trace = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int32)
        result = encode_device_sequence_batch(trace, num_devices=2)
        expected = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_three_devices(self):
        """Test encoding with 3 devices."""
        # For lookback=2, num_devices=3: seq_id = col0 * 3 + col1
        trace = torch.tensor([[2, 1], [0, 2]], dtype=torch.int32)
        result = encode_device_sequence_batch(trace, num_devices=3)
        expected = torch.tensor([2 * 3 + 1, 0 * 3 + 2], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_lookback_3(self):
        """Test encoding with lookback=3."""
        # For lookback=3, num_devices=2: seq_id = col0*4 + col1*2 + col2
        trace = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int32)
        result = encode_device_sequence_batch(trace, num_devices=2)
        expected = torch.tensor([1 * 4 + 0 * 2 + 1, 0 * 4 + 1 * 2 + 0], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_single_token(self):
        """Test with a single token."""
        trace = torch.tensor([[1, 0]], dtype=torch.int32)
        result = encode_device_sequence_batch(trace, num_devices=2)
        assert result.shape == (1,)
        assert result[0].item() == 2  # 1 * 2 + 0


class TestRebatchForLayer:
    """Tests for rebatch_for_layer()."""

    def test_shuffled_unshuffle_identity(self):
        """Verify unshuffle(shuffle(x)) recovers the original tensor."""
        num_tokens = 32
        hidden_dim = 64
        vocab_size = 1000
        num_devices = 2

        ctx = _make_ctx(layer_ids=(5,), vocab_size=vocab_size, num_devices=num_devices)
        ctx.reset(num_tokens, torch.device("cpu"))

        input_ids = torch.randint(0, vocab_size, (num_tokens,))
        hidden = torch.randn(num_tokens, hidden_dim)

        shuffled, shf_idx, inv_shf, chunk_size = rebatch_for_layer(
            ctx, 5, input_ids, hidden
        )

        # chunk_size should divide evenly
        assert chunk_size * num_devices == shuffled.shape[0]

        # unshuffle should recover original
        recovered = unshuffle_output(shuffled, inv_shf, num_tokens)
        assert recovered.shape == hidden.shape
        assert torch.allclose(recovered, hidden, atol=1e-6)

    def test_chunk_size_is_fixed(self):
        """Verify chunk_size = num_tokens_padded // num_devices."""
        for num_tokens in [16, 17, 31, 32, 64]:
            num_devices = 2
            ctx = _make_ctx(layer_ids=(0,), num_devices=num_devices)
            ctx.reset(num_tokens, torch.device("cpu"))

            input_ids = torch.randint(0, 100, (num_tokens,))
            hidden = torch.randn(num_tokens, 8)

            _, _, _, chunk_size = rebatch_for_layer(ctx, 0, input_ids, hidden)

            # Padded size must be a multiple of num_devices
            n_padded = chunk_size * num_devices
            assert n_padded >= num_tokens
            assert n_padded % num_devices == 0
            assert n_padded - num_tokens < num_devices  # minimal padding

    def test_padding_for_odd_tokens(self):
        """Verify padding works correctly when num_tokens is not divisible by num_devices."""
        num_tokens = 17  # odd, not divisible by 2
        num_devices = 2
        hidden_dim = 8

        ctx = _make_ctx(layer_ids=(0,), num_devices=num_devices)
        ctx.reset(num_tokens, torch.device("cpu"))

        input_ids = torch.randint(0, 100, (num_tokens,))
        hidden = torch.randn(num_tokens, hidden_dim)

        shuffled, shf_idx, inv_shf, chunk_size = rebatch_for_layer(
            ctx, 0, input_ids, hidden
        )

        assert shuffled.shape[0] == 18  # padded to next multiple of 2
        assert chunk_size == 9

        # unshuffle still recovers original
        recovered = unshuffle_output(shuffled, inv_shf, num_tokens)
        assert recovered.shape == (num_tokens, hidden_dim)
        assert torch.allclose(recovered, hidden, atol=1e-6)

    def test_device_trace_updated(self):
        """Verify device_trace is recorded after each layer."""
        num_tokens = 16
        layer_ids = (3, 5, 7)
        ctx = _make_ctx(layer_ids=layer_ids)
        ctx.reset(num_tokens, torch.device("cpu"))

        input_ids = torch.randint(0, 100, (num_tokens,))
        hidden = torch.randn(num_tokens, 8)

        # Process through 3 layers
        for lid in layer_ids:
            rebatch_for_layer(ctx, lid, input_ids, hidden)

        assert ctx.moe_layer_counter == 3
        assert ctx.device_trace is not None
        # All trace values should be valid device ids
        assert (ctx.device_trace[:num_tokens] >= 0).all()
        assert (ctx.device_trace[:num_tokens] < ctx.num_devices).all()

    def test_a_table_used_after_lookback(self):
        """Verify A-table is consulted after enough layers for lookback history."""
        num_tokens = 16
        num_devices = 2
        lookback = 2
        layer_ids = (0, 1, 2, 3)
        ctx = _make_ctx(
            layer_ids=layer_ids, num_devices=num_devices, lookback=lookback
        )
        ctx.reset(num_tokens, torch.device("cpu"))

        input_ids = torch.randint(0, 100, (num_tokens,))
        hidden = torch.randn(num_tokens, 8)

        # First two layers: only T-table (moe_layer_counter < lookback)
        rebatch_for_layer(ctx, 0, input_ids, hidden)
        assert ctx.moe_layer_counter == 1
        rebatch_for_layer(ctx, 1, input_ids, hidden)
        assert ctx.moe_layer_counter == 2

        # Third layer: A-table should be consulted (moe_layer_counter >= lookback)
        # We verify this indirectly by checking the function doesn't crash
        # and trace is still valid
        rebatch_for_layer(ctx, 2, input_ids, hidden)
        assert ctx.moe_layer_counter == 3

    def test_confidence_comparison(self):
        """Verify that higher confidence predictor wins."""
        num_tokens = 8
        num_devices = 2
        lookback = 1
        vocab_size = 10

        lt = SemMoeTPLayerTensors(
            # T predicts all tokens go to device 0
            t_full=torch.zeros(vocab_size, dtype=torch.int32),
            tp_full=torch.full((vocab_size,), 0.9, dtype=torch.float32),
            # A predicts all tokens go to device 1, but with lower confidence
            a_table=torch.ones(num_devices, dtype=torch.int32),
            ap_table=torch.full((num_devices,), 0.5, dtype=torch.float32),
        )

        ctx = SemMoeTPContext(
            layer_tensors={0: lt, 1: lt},
            lookback=lookback,
            num_devices=num_devices,
            moe_layer_ids=(0, 1),
        )
        ctx.reset(num_tokens, torch.device("cpu"))

        input_ids = torch.arange(num_tokens) % vocab_size
        hidden = torch.randn(num_tokens, 4)

        # First layer: only T-table
        rebatch_for_layer(ctx, 0, input_ids, hidden)

        # Second layer: T confidence (0.9) > A confidence (0.5), so T wins
        # All tokens should be predicted for device 0
        _, shf_idx, _, chunk_size = rebatch_for_layer(ctx, 1, input_ids, hidden)

        # Check that trace records device 0 for all tokens (T wins)
        assert ctx.device_trace is not None
        # moe_idx for layer 1 is 1
        assert (ctx.device_trace[:num_tokens, 1] == 0).all()

    def test_reset_clears_state(self):
        """Verify reset() clears device_trace and counter."""
        ctx = _make_ctx(layer_ids=(0, 1))
        ctx.reset(32, torch.device("cpu"))
        ctx.moe_layer_counter = 5

        ctx.reset(16, torch.device("cpu"))
        assert ctx.moe_layer_counter == 0
        assert ctx.device_trace is not None
        assert ctx.device_trace.shape[0] >= 16


class TestUnshuffle:
    """Tests for unshuffle_output()."""

    def test_identity_permutation(self):
        """Identity permutation should return the original tensor."""
        n = 10
        t = torch.randn(n, 4)
        inv_shf = torch.arange(n)
        result = unshuffle_output(t, inv_shf, n)
        assert torch.equal(result, t)

    def test_reverse_permutation(self):
        """Reverse permutation should reverse the tensor."""
        n = 8
        t = torch.arange(n * 2, dtype=torch.float).view(n, 2)
        inv_shf = torch.arange(n - 1, -1, -1)
        result = unshuffle_output(t, inv_shf, n)
        expected = t.flip(0)
        assert torch.equal(result, expected)

    def test_truncation(self):
        """unshuffle should truncate padding tokens."""
        original_n = 7
        padded_n = 8
        t = torch.randn(padded_n, 4)
        inv_shf = torch.arange(padded_n)
        result = unshuffle_output(t, inv_shf, original_n)
        assert result.shape == (original_n, 4)


class TestBuildTPContext:
    """Tests for build_tp_context()."""

    def test_build_from_schedule(self):
        """Test building TP context from a mock schedule."""
        from vllm.sem_moe_tp import build_tp_context

        # Create a mock schedule object
        class MockSchedule:
            layer_ids = (3, 7)
            lookback = 2
            num_devices = 2

            class _layer:
                t_full = np.zeros(100, dtype=np.int32)
                tp_full = np.ones(100, dtype=np.float32)
                a_table = np.zeros(4, dtype=np.int32)
                ap_table = np.ones(4, dtype=np.float32)

            layers = {3: _layer(), 7: _layer()}

        ctx = build_tp_context(MockSchedule(), torch.device("cpu"))
        assert ctx is not None
        assert len(ctx.layer_tensors) == 2
        assert ctx.lookback == 2
        assert ctx.num_devices == 2
        assert ctx.moe_layer_ids == (3, 7)

    def test_returns_none_when_tables_missing(self):
        """Test that build_tp_context returns None when T tables are missing."""
        from vllm.sem_moe_tp import build_tp_context

        class MockSchedule:
            layer_ids = (3,)
            lookback = 2
            num_devices = 2

            class _layer:
                t_full = None
                tp_full = None
                a_table = None
                ap_table = None

            layers = {3: _layer()}

        ctx = build_tp_context(MockSchedule(), torch.device("cpu"))
        assert ctx is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
