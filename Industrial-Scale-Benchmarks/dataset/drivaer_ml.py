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

"""
DrivAerML dataset loader.

500 parametrically deformed variants of the DrivAer vehicle model with
hybrid RANS-LES simulations on ~140M cell volumetric meshes. Each surface
mesh has ~8.8M points.

Surface outputs: pressure (p_s), wall shear stress (tau)
Volume outputs: velocity (u), pressure (p_v)

Reference: Ashton et al. (2024b), Transolver-3 paper Appendix A.2.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

# Expected keys per field type
_SURFACE_KEYS = {"surface_coords", "surface_normals", "surface_pressure", "surface_wall_shear"}
_VOLUME_KEYS = {"volume_coords", "volume_velocity", "volume_pressure"}
_REQUIRED_KEYS = {"params"}


def validate_npz(path, field="surface"):
    """Validate a DrivAerML .npz file for schema and data quality.

    Args:
        path: path to .npz file
        field: 'surface', 'volume', or 'both'

    Returns:
        list of error strings (empty = valid)
    """
    errors = []
    try:
        data = np.load(path, allow_pickle=True, mmap_mode="r")
    except Exception as e:
        return [f"Cannot load {path}: {e}"]

    # Check required keys
    required = set(_REQUIRED_KEYS)
    if field in ("surface", "both"):
        required |= _SURFACE_KEYS
    if field in ("volume", "both"):
        required |= _VOLUME_KEYS

    missing = required - set(data.keys())
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")

    # Check for NaN/Inf in available arrays
    for key in data.keys():
        arr = data[key]
        if arr.dtype in (np.float32, np.float64):
            # Sample a slice to avoid reading entire mmap'd file
            sample = np.array(arr[: min(1000, arr.shape[0])])
            if np.any(np.isnan(sample)):
                errors.append(f"NaN detected in {key}")
            if np.any(np.isinf(sample)):
                errors.append(f"Inf detected in {key}")

    # Check coordinate dimensions
    for key in ("surface_coords", "volume_coords"):
        if key in data and data[key].ndim == 2 and data[key].shape[1] != 3:
            errors.append(f"{key} has {data[key].shape[1]} dims, expected 3")

    return errors


class DrivAerMLDataset(Dataset):
    """DrivAerML surface + volume dataset.

    This is the largest benchmark (~160M cells). Geometry amortized training
    is essential. Data is typically stored in chunked/sharded format.

    Expected data format per sample:
        - surface_coords: (N_s, 3) ~8.8M points
        - surface_normals: (N_s, 3)
        - surface_pressure: (N_s, 1)
        - surface_wall_shear: (N_s, 3)
        - volume_coords: (N_v, 3) ~140M points
        - volume_velocity: (N_v, 3)
        - volume_pressure: (N_v, 1)
        - params: (d_params,) geometric deformation parameters

    Args:
        data_dir: path to preprocessed data
        split: 'train', 'val', or 'test'
        field: 'surface', 'volume', or 'both'
        subset_size: points per training iteration (default 400K per paper)
        normalize_coords: apply min-max normalization
        coord_scale: scale factor after normalization
        lazy_load: if True, only load file paths and read data on __getitem__
        shard_id: which shard this worker owns (0-indexed). None = no sharding.
        num_shards: total number of shards. None = no sharding.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        field="surface",
        subset_size=400000,
        normalize_coords=True,
        coord_scale=1000.0,
        lazy_load=True,
        shard_id=None,
        num_shards=None,
        validate=False,
        seed=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.field = field
        self.subset_size = subset_size
        self.normalize_coords = normalize_coords
        self.coord_scale = coord_scale
        self.lazy_load = lazy_load
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.seed = seed
        self._call_count = 0

        split_file = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.samples = [line.strip() for line in f if line.strip()]
        else:
            self.samples = sorted([f for f in os.listdir(data_dir) if f.endswith((".npz", ".h5"))])

        if validate:
            for sample in self.samples:
                path = os.path.join(data_dir, sample)
                errors = validate_npz(path, field=field)
                if errors:
                    raise ValueError(f"Validation failed for {sample}: {errors}")

    def __len__(self):
        return len(self.samples)

    def _normalize_coords(self, coords):
        coord_min = coords.min(dim=0, keepdim=True).values
        coord_max = coords.max(dim=0, keepdim=True).values
        coords = (coords - coord_min) / (coord_max - coord_min + 1e-8)
        return coords * self.coord_scale

    def _shard_range(self, total_points):
        """Compute [start, end) for this worker's mesh shard."""
        if self.shard_id is None or self.num_shards is None:
            return 0, total_points
        from transolver3.distributed import mesh_shard_range

        return mesh_shard_range(total_points, self.shard_id, self.num_shards)

    def _load_array_shard(self, data, key):
        """Load an array from mmap, reading only this worker's shard.

        For sharded loading, reads only the [start:end] slice from the
        memory-mapped file — the OS only pages in the requested range.
        """
        arr = data[key]
        start, end = self._shard_range(arr.shape[0])
        return torch.tensor(np.array(arr[start:end]), dtype=torch.float32)

    def _subsample(self, *tensors):
        if self.subset_size is None:
            return tensors
        N = tensors[0].shape[0]
        if N <= self.subset_size:
            return tensors
        if self.seed is not None:
            # Deterministic: seed derived from base seed + call count
            g = torch.Generator()
            g.manual_seed(self.seed + self._call_count)
            self._call_count += 1
            indices = torch.randperm(N, generator=g)[: self.subset_size]
        else:
            indices = torch.randperm(N)[: self.subset_size]
        return tuple(t[indices] for t in tensors)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])

        # For very large files, use memory-mapped loading
        try:
            data = np.load(sample_path, allow_pickle=True, mmap_mode="r")
        except ValueError:
            data = np.load(sample_path, allow_pickle=True)

        params = torch.tensor(np.array(data["params"]), dtype=torch.float32)
        result = {"params": params, "sample_id": self.samples[idx]}

        if self.field in ("surface", "both"):
            coords = self._load_array_shard(data, "surface_coords")
            normals = self._load_array_shard(data, "surface_normals")
            pressure = self._load_array_shard(data, "surface_pressure")
            shear = self._load_array_shard(data, "surface_wall_shear")

            coords, normals, pressure, shear = self._subsample(coords, normals, pressure, shear)
            if self.normalize_coords:
                coords = self._normalize_coords(coords)

            N_s = coords.shape[0]
            p_broadcast = params.unsqueeze(0).expand(N_s, -1)
            x_surface = torch.cat([coords, normals, p_broadcast], dim=-1)
            target_surface = torch.cat([pressure, shear], dim=-1)

            result["surface_x"] = x_surface
            result["surface_pos"] = coords
            result["surface_target"] = target_surface

        if self.field in ("volume", "both"):
            coords = self._load_array_shard(data, "volume_coords")
            velocity = self._load_array_shard(data, "volume_velocity")
            pressure = self._load_array_shard(data, "volume_pressure")

            coords, velocity, pressure = self._subsample(coords, velocity, pressure)
            if self.normalize_coords:
                coords = self._normalize_coords(coords)

            N_v = coords.shape[0]
            p_broadcast = params.unsqueeze(0).expand(N_v, -1)
            x_volume = torch.cat([coords, p_broadcast], dim=-1)
            target_volume = torch.cat([velocity, pressure], dim=-1)

            result["volume_x"] = x_volume
            result["volume_pos"] = coords
            result["volume_target"] = target_volume

        return result
