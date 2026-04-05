# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the cuteDSL d=256 backward path on Blackwell (SM100+).

These tests cover the case where cuDNN does not support d=256 bprop on Blackwell,
and the cuteDSL Python kernel (nvidia-cutlass-dsl) is used as a fallback.

Prerequisites:
- SM100+ (Blackwell) GPU
- nvidia-cutlass-dsl[cu13] package installed
- cudnn-frontend Python package in sys.path (3rdparty/cudnn-frontend/python)
"""

import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine_jax import get_device_compute_capability


def _is_blackwell():
    """Return True if any local device is SM100+."""
    try:
        return any(get_device_compute_capability(i) >= 100 for i in range(len(jax.local_devices())))
    except Exception:
        return False


def _cutedsl_available():
    return importlib.util.find_spec("cutlass") is not None


def _reference_attention(q, k, v, scale, is_causal=False):
    """Pure JAX reference attention (no fused kernel)."""
    # q, k, v: (B, S, H, D)
    attn_weights = jnp.einsum("bshd,bthd->bhst", q, k) * scale
    if is_causal:
        S = q.shape[1]
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_weights = jnp.where(mask[None, None], attn_weights, -1e9)
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.einsum("bhst,bthd->bshd", attn_weights, v)


skipif_not_blackwell_d256 = pytest.mark.skipif(
    not (_is_blackwell() and _cutedsl_available()),
    reason="Requires SM100+ GPU and nvidia-cutlass-dsl package",
)


@skipif_not_blackwell_d256
class TestBlackwellD256Bwd:
    """Tests for cuteDSL-backed d=256 backward pass on Blackwell."""

    @pytest.mark.parametrize("batch,seqlen,num_heads,is_causal", [
        (1, 128, 8, False),
        (2, 256, 4, False),
        (1, 64, 8, True),
    ])
    def test_backward_bshd(self, batch, seqlen, num_heads, is_causal):
        """Backward pass through fused_attn should match a JAX reference for d=256."""
        from transformer_engine.jax.attention import (  # pylint: disable=import-outside-toplevel
            AttnBiasType,
            AttnMaskType,
            QKVLayout,
            fused_attn,
            is_fused_attn_kernel_available,
        )
        from transformer_engine.jax.sharding import MeshResource  # pylint: disable=import-outside-toplevel
        from transformer_engine.jax.sharding import global_shard_guard  # pylint: disable=import-outside-toplevel

        head_dim = 256
        dtype = jnp.bfloat16
        scale = 1.0 / (head_dim ** 0.5)

        # Check that the cuteDSL path is reported as available
        attn_mask_type = AttnMaskType.CAUSAL_MASK if is_causal else AttnMaskType.NO_MASK
        assert is_fused_attn_kernel_available(
            True,  # is_training
            dtype,
            dtype,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnBiasType.NO_BIAS,
            attn_mask_type,
            None,
            0.0,
            num_heads,
            num_heads,
            seqlen,
            seqlen,
            head_dim,
            head_dim,
        ), "cuteDSL path should be available for d=256 on Blackwell with nvidia-cutlass-dsl"

        rng = jax.random.PRNGKey(42)
        q = jax.random.normal(rng, (batch, seqlen, num_heads, head_dim), dtype=dtype)
        k = jax.random.normal(jax.random.fold_in(rng, 1), (batch, seqlen, num_heads, head_dim), dtype=dtype)
        v = jax.random.normal(jax.random.fold_in(rng, 2), (batch, seqlen, num_heads, head_dim), dtype=dtype)

        def fused_fwd(q, k, v):
            mesh_resource = MeshResource()
            with global_shard_guard(mesh_resource):
                return fused_attn(
                    (q, k, v),
                    None,  # bias
                    None,  # sequence_descriptor
                    None,  # seed
                    attn_bias_type=AttnBiasType.NO_BIAS,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
                    scaling_factor=scale,
                    dropout_probability=0.0,
                    is_training=True,
                    max_segments_per_seq=1,
                    window_size=None,
                )

        def ref_fwd(q, k, v):
            return _reference_attention(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32), scale, is_causal).astype(dtype)

        # Compare gradients
        grad_output = jax.random.normal(jax.random.fold_in(rng, 3), (batch, seqlen, num_heads, head_dim), dtype=dtype)

        fused_grads = jax.grad(lambda q, k, v: jnp.sum(fused_fwd(q, k, v) * grad_output), argnums=(0, 1, 2))(q, k, v)
        ref_grads = jax.grad(lambda q, k, v: jnp.sum(ref_fwd(q, k, v) * grad_output), argnums=(0, 1, 2))(q, k, v)

        atol = 0.05  # bf16 has limited precision; allow some tolerance
        for name, fg, rg in zip(["dq", "dk", "dv"], fused_grads, ref_grads):
            np.testing.assert_allclose(
                np.array(fg.astype(jnp.float32)),
                np.array(rg.astype(jnp.float32)),
                atol=atol,
                rtol=0.1,
                err_msg=f"{name} gradient mismatch for batch={batch} seqlen={seqlen} heads={num_heads} is_causal={is_causal}",
            )
