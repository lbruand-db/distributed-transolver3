# Copyright 2024 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transolver3.common — timestep embeddings and MLP."""

import torch
import pytest

from transolver3.common import timestep_embedding, MLP


class TestTimestepEmbedding:
    def test_output_shape(self):
        """Embedding shape is (N, dim)."""
        t = torch.tensor([0.0, 1.0, 5.0])
        emb = timestep_embedding(t, dim=32)
        assert emb.shape == (3, 32)

    def test_odd_dim_padded(self):
        """Odd dim gets zero-padded to correct size."""
        t = torch.tensor([1.0])
        emb = timestep_embedding(t, dim=33)
        assert emb.shape == (1, 33)
        # Last element should be zero (padding)
        assert emb[0, -1].item() == 0.0

    def test_different_timesteps_differ(self):
        """Different timesteps produce different embeddings."""
        t = torch.tensor([0.0, 100.0])
        emb = timestep_embedding(t, dim=16)
        assert not torch.allclose(emb[0], emb[1])

    def test_deterministic(self):
        """Same input gives same output."""
        t = torch.tensor([3.14])
        emb1 = timestep_embedding(t, dim=16)
        emb2 = timestep_embedding(t, dim=16)
        assert torch.equal(emb1, emb2)


class TestMLP:
    def test_forward_residual(self):
        """MLP with residual connections runs and has correct output shape."""
        mlp = MLP(n_input=8, n_hidden=16, n_output=4, n_layers=2, res=True)
        x = torch.randn(2, 8)
        out = mlp(x)
        assert out.shape == (2, 4)

    def test_forward_no_residual(self):
        """MLP without residual connections runs and has correct output shape."""
        mlp = MLP(n_input=8, n_hidden=16, n_output=4, n_layers=2, res=False)
        x = torch.randn(2, 8)
        out = mlp(x)
        assert out.shape == (2, 4)

    def test_residual_vs_no_residual_differ(self):
        """Residual and non-residual MLPs produce different outputs."""
        torch.manual_seed(42)
        mlp_res = MLP(n_input=8, n_hidden=16, n_output=4, n_layers=2, res=True)
        torch.manual_seed(42)
        mlp_no = MLP(n_input=8, n_hidden=16, n_output=4, n_layers=2, res=False)

        x = torch.randn(2, 8)
        out_res = mlp_res(x)
        out_no = mlp_no(x)
        assert not torch.allclose(out_res, out_no)

    def test_invalid_activation_raises(self):
        """Unknown activation raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            MLP(n_input=8, n_hidden=16, n_output=4, act="nonexistent")

    def test_all_activations(self):
        """All named activations are accepted."""
        for act in ["gelu", "tanh", "sigmoid", "relu", "silu", "ELU", "softplus"]:
            mlp = MLP(n_input=4, n_hidden=8, n_output=2, act=act)
            out = mlp(torch.randn(1, 4))
            assert out.shape == (1, 2), f"Failed for activation {act}"
