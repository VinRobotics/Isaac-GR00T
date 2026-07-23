# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone sanity check for Qwen3Backbone's motion-query-token injection
(see gr00t/model/modules/qwen3_backbone.py) — the highest-risk piece of the
VLM-backbone motion head, since it splices a learnable parameter into the
HF model's own forward pass. Uses a tiny randomly-initialized Qwen3VL model
(no network access, no real Cosmos-Reason2-2B weights) so this runs as a
regular CPU test, not gated behind the `gpu` marker.
"""

from unittest.mock import patch

import pytest
import torch


qwen3_vl = pytest.importorskip(
    "transformers.models.qwen3_vl.configuration_qwen3_vl",
    reason="requires a transformers version with Qwen3-VL support",
)
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone  # noqa: E402
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402


def _tiny_qwen3_vl_model() -> Qwen3VLForConditionalGeneration:
    text_config = qwen3_vl.Qwen3VLTextConfig(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=1,
    )
    vision_config = qwen3_vl.Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    config = Qwen3VLConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict())
    return Qwen3VLForConditionalGeneration(config).eval()


def _make_backbone(num_motion_tokens: int, tiny_model) -> Qwen3Backbone:
    with patch.object(Qwen3VLForConditionalGeneration, "from_pretrained", return_value=tiny_model):
        return Qwen3Backbone(model_name="tiny", num_motion_tokens=num_motion_tokens, select_layer=2)


def _text_only_input(batch_size: int, seq_len: int) -> dict:
    input_ids = torch.randint(low=10, high=100, size=(batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": None,
        "image_grid_thw": None,
    }


def test_motion_tokens_output_shapes():
    tiny_model = _tiny_qwen3_vl_model()
    backbone = _make_backbone(num_motion_tokens=3, tiny_model=tiny_model)

    batch_size, seq_len = 2, 6
    out = backbone(_text_only_input(batch_size, seq_len))

    assert out["backbone_features"].shape == (batch_size, seq_len, 32)
    assert out["motion_token_features"].shape == (batch_size, 3, 32)
    assert out["image_mask"].shape == (batch_size, seq_len)
    assert out["backbone_attention_mask"].shape == (batch_size, seq_len)


def test_motion_tokens_receive_gradients():
    tiny_model = _tiny_qwen3_vl_model()
    backbone = _make_backbone(num_motion_tokens=3, tiny_model=tiny_model)

    out = backbone(_text_only_input(batch_size=2, seq_len=6))
    out["motion_token_features"].sum().backward()

    assert backbone.motion_query_tokens.grad is not None
    assert torch.isfinite(backbone.motion_query_tokens.grad).all()
    assert backbone.motion_query_tokens.grad.norm().item() > 0


def test_motion_tokens_are_a_pure_readout():
    """The whole point of appending motion tokens at the END of a causal
    sequence: real tokens must be BIT-IDENTICAL whether or not the motion
    tokens are appended — the same "pure readout" guarantee the old
    Action-Head keypoint head enforced with an explicit attention mask,
    here obtained for free from ordinary causal masking + position order."""
    tiny_model = _tiny_qwen3_vl_model()
    backbone_with_motion = _make_backbone(num_motion_tokens=3, tiny_model=tiny_model)
    backbone_no_motion = _make_backbone(num_motion_tokens=0, tiny_model=tiny_model)

    batch_size, seq_len = 2, 6
    torch.manual_seed(0)
    vl_input = _text_only_input(batch_size, seq_len)

    with torch.no_grad():
        out_with = backbone_with_motion(dict(vl_input))
        out_without = backbone_no_motion(dict(vl_input))

    assert out_without.get("motion_token_features") is None
    assert torch.allclose(
        out_with["backbone_features"], out_without["backbone_features"], atol=1e-5
    )


def test_disabled_motion_tokens_is_backward_compatible():
    """num_motion_tokens=0 (the default) must leave the backbone's output
    contract exactly as it was before this feature existed."""
    tiny_model = _tiny_qwen3_vl_model()
    backbone = _make_backbone(num_motion_tokens=0, tiny_model=tiny_model)

    assert backbone.motion_query_tokens is None
    out = backbone(_text_only_input(batch_size=2, seq_len=6))
    assert "motion_token_features" not in out
