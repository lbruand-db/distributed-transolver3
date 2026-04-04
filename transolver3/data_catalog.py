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
Delta Lake and Unity Catalog integration for Transolver-3 mesh data.

Provides utilities to:
  - Register mesh metadata (sample ID, cell counts, params) in a Delta table
  - Read mesh metadata back for querying and filtering
  - Log normalizer statistics to Delta for versioning and reuse
  - Load normalizer statistics from Delta

All functions require PySpark and are opt-in. The existing np.load path
is unaffected. Install with: pip install transolver3[databricks]
"""

import json
import os
from datetime import datetime, timezone


def _require_pyspark():
    try:
        from pyspark.sql import SparkSession  # noqa: F401
    except ImportError:
        raise ImportError("pyspark is required for data_catalog. Install with: pip install transolver3[databricks]")


def register_mesh_metadata(spark, catalog, schema, table, samples_dir):
    """Scan .npz files and register mesh metadata in a Delta table.

    Args:
        spark: SparkSession
        catalog: Unity Catalog name (e.g., "ml")
        schema: Schema name (e.g., "transolver3")
        table: Table name (e.g., "mesh_metadata")
        samples_dir: Path to directory containing .npz sample files
            (local path or UC Volume path like /Volumes/catalog/schema/volume/)

    Returns:
        PySpark DataFrame of the registered metadata
    """
    _require_pyspark()
    import numpy as np
    from pyspark.sql import Row

    rows = []
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(samples_dir, fname)
        sample_id = os.path.splitext(fname)[0]

        data = np.load(fpath, mmap_mode="r")
        keys = list(data.keys())

        surface_cells = int(data["surface_coords"].shape[0]) if "surface_coords" in keys else 0
        volume_cells = int(data["volume_coords"].shape[0]) if "volume_coords" in keys else 0
        params_dim = int(data["params"].shape[0]) if "params" in keys else 0
        file_size_bytes = os.path.getsize(fpath)

        rows.append(
            Row(
                sample_id=sample_id,
                file_name=fname,
                file_path=fpath,
                surface_cells=surface_cells,
                volume_cells=volume_cells,
                total_cells=surface_cells + volume_cells,
                params_dim=params_dim,
                available_keys=json.dumps(keys),
                file_size_bytes=file_size_bytes,
                registered_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    df = spark.createDataFrame(rows)
    full_table = f"{catalog}.{schema}.{table}"
    df.write.format("delta").mode("overwrite").saveAsTable(full_table)
    return spark.table(full_table)


def get_mesh_metadata(spark, catalog, schema, table):
    """Read mesh metadata from a Delta table.

    Args:
        spark: SparkSession
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name

    Returns:
        PySpark DataFrame with mesh metadata
    """
    _require_pyspark()
    return spark.table(f"{catalog}.{schema}.{table}")


def log_normalization_stats(spark, catalog, schema, table, normalizer, normalizer_name=None):
    """Write normalizer statistics to a Delta table.

    Appends a row with the normalizer's state_dict values, type, and timestamp.
    Supports both InputNormalizer and TargetNormalizer.

    Args:
        spark: SparkSession
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name (e.g., "normalization_stats")
        normalizer: an InputNormalizer or TargetNormalizer instance
        normalizer_name: optional label (e.g., "surface_input", "pressure_target")

    Returns:
        PySpark DataFrame of the logged row
    """
    _require_pyspark()
    from pyspark.sql import Row

    normalizer_type = type(normalizer).__name__
    state = {}
    for k, v in normalizer.state_dict().items():
        state[k] = v.tolist() if hasattr(v, "tolist") else v

    row = Row(
        normalizer_type=normalizer_type,
        normalizer_name=normalizer_name or normalizer_type,
        state_dict_json=json.dumps(state),
        logged_at=datetime.now(timezone.utc).isoformat(),
    )

    df = spark.createDataFrame([row])
    full_table = f"{catalog}.{schema}.{table}"
    df.write.format("delta").mode("append").saveAsTable(full_table)
    return df


def load_normalization_stats(spark, catalog, schema, table, normalizer_type):
    """Load the latest normalizer statistics from a Delta table.

    Args:
        spark: SparkSession
        catalog: Unity Catalog name
        schema: Schema name
        table: Table name
        normalizer_type: "InputNormalizer" or "TargetNormalizer"

    Returns:
        dict suitable for normalizer.load_state_dict() after tensor conversion
    """
    _require_pyspark()
    import torch

    full_table = f"{catalog}.{schema}.{table}"
    row = (
        spark.table(full_table)
        .filter(f"normalizer_type = '{normalizer_type}'")
        .orderBy("logged_at", ascending=False)
        .first()
    )
    if row is None:
        raise ValueError(f"No {normalizer_type} stats found in {full_table}")

    raw = json.loads(row.state_dict_json)
    state_dict = {}
    for k, v in raw.items():
        state_dict[k] = torch.tensor(v)
    return state_dict
