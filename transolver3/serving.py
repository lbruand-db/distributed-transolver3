"""
Model serving integration for Transolver-3 on Databricks.

Provides:
  - TransolverPyfunc: MLflow pyfunc wrapper for cached inference with normalization
  - register_serving_model: log model + normalizers to UC Model Registry
  - deploy_serving_endpoint: create/update a Databricks Model Serving endpoint

All Databricks-specific imports are guarded. Install with:
  pip install transolver3[databricks]
"""

import json
import os
import tempfile

import torch
import numpy as np

try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None


class TransolverPyfunc(mlflow.pyfunc.PythonModel if mlflow else object):
    """MLflow pyfunc wrapper for Transolver-3 cached inference.

    Handles normalization, cache building, and decoding in a single predict() call.
    Registered via mlflow.pyfunc.log_model() for serving on Databricks.

    Artifacts expected:
        - model_state: path to model state_dict (.pt)
        - model_config: path to model config JSON
        - input_normalizer (optional): path to InputNormalizer state_dict (.pt)
        - target_normalizer (optional): path to TargetNormalizer state_dict (.pt)
    """

    def load_context(self, context):
        """Load model and normalizers from MLflow artifacts."""
        from transolver3 import Transolver3, CachedInference
        from transolver3.normalizer import InputNormalizer, TargetNormalizer

        # Load model config
        with open(context.artifacts["model_config"]) as f:
            config = json.load(f)

        # Instantiate and load model
        self.model = Transolver3(**config)
        state_dict = torch.load(
            context.artifacts["model_state"],
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        # Create inference engine
        self.engine = CachedInference(
            self.model,
            cache_chunk_size=config.get("cache_chunk_size", 100_000),
            decode_chunk_size=config.get("decode_chunk_size", 50_000),
        )

        # Load normalizers (optional)
        self.input_norm = None
        self.target_norm = None

        if "input_normalizer" in context.artifacts:
            self.input_norm = InputNormalizer(per_sample=False)
            state = torch.load(context.artifacts["input_normalizer"], weights_only=True)
            # Resize buffers to match saved shapes before loading
            for k, v in state.items():
                if hasattr(self.input_norm, k):
                    setattr(self.input_norm, k, torch.zeros_like(v))
            self.input_norm.load_state_dict(state)
            self.input_norm = self.input_norm.to(self.device)

        if "target_normalizer" in context.artifacts:
            self.target_norm = TargetNormalizer()
            state = torch.load(context.artifacts["target_normalizer"], weights_only=True)
            for k, v in state.items():
                if hasattr(self.target_norm, k):
                    setattr(self.target_norm, k, torch.zeros_like(v))
            self.target_norm.load_state_dict(state)
            self.target_norm = self.target_norm.to(self.device)

    def predict(self, context, model_input):
        """Run cached inference on input coordinates.

        Args:
            context: MLflow PythonModelContext
            model_input: pandas DataFrame with column 'coordinates'
                containing JSON-serialized coordinate arrays,
                or a dict with key 'coordinates' containing a numpy array

        Returns:
            numpy array of predictions (N, out_dim)
        """
        # Parse input
        if hasattr(model_input, "to_dict"):
            # pandas DataFrame
            coords_raw = model_input["coordinates"].iloc[0]
            if isinstance(coords_raw, str):
                coords_raw = json.loads(coords_raw)
            coords = np.array(coords_raw, dtype=np.float32)
        elif isinstance(model_input, dict):
            coords = np.array(model_input["coordinates"], dtype=np.float32)
        else:
            coords = np.array(model_input, dtype=np.float32)

        # Ensure (B, N, space_dim) shape
        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]

        x = torch.tensor(coords, dtype=torch.float32, device=self.device)

        # Apply input normalization
        if self.input_norm is not None:
            x = self.input_norm.encode(x)

        # Run cached inference
        with torch.no_grad():
            output = self.engine.predict(x)

        # Apply target denormalization
        if self.target_norm is not None:
            output = self.target_norm.decode(output)

        return output.cpu().numpy()


def register_serving_model(
    model, config, normalizers=None, catalog="ml", schema="transolver3", model_name="transolver3"
):
    """Log a Transolver3 model as an MLflow pyfunc for serving.

    Args:
        model: trained Transolver3 model (unwrapped)
        config: dict of model constructor kwargs (space_dim, n_layers, etc.)
        normalizers: optional dict {"input": InputNormalizer, "target": TargetNormalizer}
        catalog: Unity Catalog name
        schema: UC schema name
        model_name: model name in registry

    Returns:
        MLflow ModelInfo
    """
    try:
        import mlflow
        import mlflow.pyfunc
    except ImportError:
        raise ImportError(
            "mlflow is required for register_serving_model. Install with: pip install transolver3[databricks]"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model state dict
        model_state_path = os.path.join(tmpdir, "model_state.pt")
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), model_state_path)

        # Save model config
        config_path = os.path.join(tmpdir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        artifacts = {
            "model_state": model_state_path,
            "model_config": config_path,
        }

        # Save normalizers
        if normalizers:
            if "input" in normalizers and normalizers["input"] is not None:
                path = os.path.join(tmpdir, "input_normalizer.pt")
                torch.save(normalizers["input"].state_dict(), path)
                artifacts["input_normalizer"] = path

            if "target" in normalizers and normalizers["target"] is not None:
                path = os.path.join(tmpdir, "target_normalizer.pt")
                torch.save(normalizers["target"].state_dict(), path)
                artifacts["target_normalizer"] = path

        registered_name = f"{catalog}.{schema}.{model_name}"

        # UC Model Registry requires a signature. Provide an input example
        # so MLflow can infer it automatically.
        import pandas as pd

        space_dim = config["space_dim"]
        sample_coords = np.random.randn(10, space_dim).tolist()
        input_example = pd.DataFrame({"coordinates": [json.dumps(sample_coords)]})

        return mlflow.pyfunc.log_model(
            artifact_path="transolver3_serving",
            python_model=TransolverPyfunc(),
            artifacts=artifacts,
            registered_model_name=registered_name,
            input_example=input_example,
            pip_requirements=["torch>=2.0", "einops>=0.7", "timm>=1.0", "numpy>=1.24"],
        )


def deploy_serving_endpoint(model_name, endpoint_name, catalog="ml", schema="transolver3"):
    """Create or update a Databricks Model Serving endpoint.

    Args:
        model_name: registered model name (without catalog.schema prefix)
        endpoint_name: serving endpoint name
        catalog: Unity Catalog name
        schema: UC schema name

    Returns:
        Endpoint details dict
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput,
            ServedEntityInput,
        )
    except ImportError:
        raise ImportError(
            "databricks-sdk is required for deploy_serving_endpoint. Install with: pip install transolver3[databricks]"
        )

    client = WorkspaceClient()
    full_model_name = f"{catalog}.{schema}.{model_name}"

    # Get latest model version
    versions = client.model_versions.list(full_model_name)
    latest_version = max((v.version for v in versions), default="1")

    served_entity = ServedEntityInput(
        entity_name=full_model_name,
        entity_version=str(latest_version),
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    config = EndpointCoreConfigInput(served_entities=[served_entity])

    # Try to create; if exists, update
    try:
        result = client.serving_endpoints.create(
            name=endpoint_name,
            config=config,
        )
        print(f"Created serving endpoint: {endpoint_name}")
    except Exception:
        result = client.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity],
        )
        print(f"Updated serving endpoint: {endpoint_name}")

    return result
