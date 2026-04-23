# Copyright 2026 Databricks, Inc.
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

"""
Distributed samplers for mesh-sharded DDP training.
"""

import torch
from torch.utils.data import Sampler


class SyncedSampler(Sampler):
    """Shuffle with the same seed on every rank.

    In mesh-sharded DDP, all ranks must process the **same sample** at each
    optimizer step so that their spatial-shard gradients combine coherently
    via all-reduce into a full-mesh gradient.  DistributedSampler partitions
    the sample index across ranks (100 samples/GPU/epoch on 4 GPUs) which
    cuts optimizer steps from 400→100 and makes each step average gradients
    from 4 *different* geometries — diverging from the paper's batch_size=1.

    SyncedSampler gives every rank the same shuffled order, so:
      - 400 optimizer steps/epoch (matches paper's single-GPU 400 steps)
      - Each step: 4 ranks process 4 spatial shards of the *same* sample,
        all-reduced → full-mesh gradient at batch_size=1
    """

    def __init__(self, dataset, seed: int = 42):
        self.n = len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n
