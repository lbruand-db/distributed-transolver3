# Distributed Training Architecture

Multi-GPU mesh-sharded training of Transolver-3 on Databricks via TorchDistributor.

## DAB Pipeline Overview

The full training pipeline is a 5-task Databricks Asset Bundle workflow. Each task runs on its own cluster, with MLflow as the single source of truth for model artifacts.

```mermaid
flowchart LR
    subgraph "DAB Pipeline (training_workflow.yml)"
        A["preprocess<br/><i>i3.xlarge (CPU)</i>"] --> B["train<br/><i>g5.12xlarge (4×A10G)</i>"]
        B --> C["evaluate<br/><i>g5.12xlarge (4×A10G)</i>"]
        C --> D["register<br/><i>i3.xlarge (CPU)</i>"]
        D --> E["deploy<br/><i>i3.xlarge (CPU)</i>"]
    end

    V[(UC Volume<br/>.npz meshes)] -.-> A
    V -.-> B
    V -.-> C

    B -- "mlflow_run_id.txt" --> V
    B -- "model + metrics" --> ML[(MLflow<br/>Experiment)]
    C -- "loads model from" --> ML
    D -- "promotes model" --> UC[(UC Model<br/>Registry)]
    E -- "creates endpoint<br/>pointing to" --> UC
```

## Train Task: Process Architecture

The train task uses TorchDistributor in `local_mode=True` on a single multi-GPU node. The Databricks Spark driver process launches TorchDistributor, which spawns child processes via `torchrun`.

```mermaid
flowchart TD
    subgraph "Databricks Job Cluster (g5.12xlarge)"
        subgraph "Spark Driver Process"
            EXEC["exec(compile(script))"]
            EXEC --> STRIP["Strip --use-distributor<br/>from sys.argv"]
            STRIP --> AUTH["_propagate_databricks_auth_env()<br/>Extract DATABRICKS_HOST/TOKEN<br/>from MLflow creds provider"]
            AUTH --> RESOLVE["_resolve_script_path(main)<br/>via __code__.co_filename"]
            RESOLVE --> TD["TorchDistributor<br/>num_processes=4<br/>local_mode=True<br/>use_gpu=True"]
            TD --> TORCHRUN["subprocess: torchrun<br/>--nproc_per_node=4<br/>script.py --data_dir ... --epochs 100"]
        end

        subgraph "torchrun (torch.distributed.run)"
            TORCHRUN --> W0["Worker 0<br/>cuda:0"]
            TORCHRUN --> W1["Worker 1<br/>cuda:1"]
            TORCHRUN --> W2["Worker 2<br/>cuda:2"]
            TORCHRUN --> W3["Worker 3<br/>cuda:3"]
        end
    end

    ENV["Inherited env vars:<br/>PYTHONPATH<br/>DATABRICKS_HOST<br/>DATABRICKS_TOKEN<br/>NCCL_P2P_DISABLE=1"] -.-> W0
    ENV -.-> W1
    ENV -.-> W2
    ENV -.-> W3
```

## Worker Lifecycle

Each worker process (spawned by torchrun) executes `main()` independently. Synchronization happens at DDP barriers and NCCL all-reduce operations.

```mermaid
sequenceDiagram
    participant W0 as Worker 0 (rank 0)
    participant W1 as Worker 1 (rank 1)
    participant W2 as Worker 2 (rank 2)
    participant W3 as Worker 3 (rank 3)
    participant NCCL as NCCL Backend
    participant Vol as UC Volume
    participant ML as MLflow

    Note over W0,W3: Phase 1: Initialization
    par All workers
        W0->>NCCL: init_process_group(nccl)
        W1->>NCCL: init_process_group(nccl)
        W2->>NCCL: init_process_group(nccl)
        W3->>NCCL: init_process_group(nccl)
    end

    Note over W0,W3: Phase 2: Data Loading (mesh sharding)
    par Each worker loads 1/4 of mesh via mmap
        W0->>Vol: mmap drivaer_001.npz [0:N/4]
        W1->>Vol: mmap drivaer_001.npz [N/4:N/2]
        W2->>Vol: mmap drivaer_001.npz [N/2:3N/4]
        W3->>Vol: mmap drivaer_001.npz [3N/4:N]
    end
    W0->>W0: dist.barrier()
    W1->>W1: dist.barrier()
    W2->>W2: dist.barrier()
    W3->>W3: dist.barrier()

    Note over W0,W3: Phase 3: Model Setup
    par All workers create identical models
        W0->>W0: Transolver3(...).to(cuda:0)
        W1->>W1: Transolver3(...).to(cuda:1)
        W2->>W2: Transolver3(...).to(cuda:2)
        W3->>W3: Transolver3(...).to(cuda:3)
    end
    Note over W0,W3: DDP wraps model, broadcasts params from rank 0
    W0->>NCCL: DDP broadcast params
    NCCL->>W1: sync params
    NCCL->>W2: sync params
    NCCL->>W3: sync params

    W0->>ML: mlflow.start_run()

    Note over W0,W3: Phase 4: Training Loop (per epoch)
    loop Each Epoch
        par Forward pass on local shards
            W0->>W0: loss = model(x_shard_0)
            W1->>W1: loss = model(x_shard_1)
            W2->>W2: loss = model(x_shard_2)
            W3->>W3: loss = model(x_shard_3)
        end
        par Backward pass (DDP all-reduces gradients)
            W0->>NCCL: all-reduce gradients
            W1->>NCCL: all-reduce gradients
            W2->>NCCL: all-reduce gradients
            W3->>NCCL: all-reduce gradients
        end
        par Optimizer step (identical on all ranks)
            W0->>W0: optimizer.step()
            W1->>W1: optimizer.step()
            W2->>W2: optimizer.step()
            W3->>W3: optimizer.step()
        end
        W0->>ML: mlflow.log_metric("train_loss", ...)
    end

    Note over W0,W3: Phase 5: Save & Log
    W0->>Vol: torch.save(best_model.pt)
    W0->>ML: log_model_with_signature()
    W0->>Vol: write mlflow_run_id.txt
    W0->>ML: mlflow.end_run()
```

## Mesh Sharding Strategy

Each GPU loads only 1/K of the mesh from disk using memory-mapped range reads. This avoids loading the full mesh (which can be 12 GB per sample) into each GPU's memory.

```mermaid
flowchart LR
    subgraph ".npz file on UC Volume (mmap)"
        M["140M mesh points<br/>(N × 22 features × float32)"]
    end

    M -- "points [0 : N/4]" --> G0["GPU 0<br/>35M points"]
    M -- "points [N/4 : N/2]" --> G1["GPU 1<br/>35M points"]
    M -- "points [N/2 : 3N/4]" --> G2["GPU 2<br/>35M points"]
    M -- "points [3N/4 : N]" --> G3["GPU 3<br/>35M points"]

    subgraph "Amortized Subsampling (per epoch)"
        G0 --> S0["Sample 100K<br/>from 35M"]
        G1 --> S1["Sample 100K<br/>from 35M"]
        G2 --> S2["Sample 100K<br/>from 35M"]
        G3 --> S3["Sample 100K<br/>from 35M"]
    end

    S0 --> FWD["Forward + Backward<br/>(local)"]
    S1 --> FWD
    S2 --> FWD
    S3 --> FWD
    FWD --> AR["NCCL All-Reduce<br/>(gradient sync)"]
```

- **No data duplication**: each GPU reads a disjoint byte range from the mmap'd file
- **Amortized training**: each GPU further subsamples 100K points from its shard per iteration
- **Communication**: only gradients are all-reduced (~27 MB for 6.7M params), not data

## Logging & Observability

```mermaid
flowchart TD
    subgraph "Child Process (rank 0)"
        PRINT["print(..., flush=True)"]
        MLFLOW["mlflow.log_metric()"]
    end

    subgraph "Child Process (rank 1-3)"
        PRINT2["print(..., flush=True)"]
    end

    PRINT --> PIPE["subprocess PIPE<br/>(stdout + stderr merged)"]
    PRINT2 --> PIPE
    PIPE --> DRIVER["Spark Driver stdout"]
    DRIVER --> UI["Databricks Job Output"]

    MLFLOW --> TRACKING["MLflow Tracking Server<br/>/Shared/transolver3-experiments"]
    TRACKING --> EXPUI["MLflow Experiment UI<br/>(live metrics)"]
```

Key design decisions:
- **`print(flush=True)` over `logging`**: TorchDistributor captures subprocess stdout via pipe. Python `logging` defaults to stderr and adds buffering. `print` with `flush=True` is the most reliable path.
- **Timestamps on all log lines**: `[HH:MM:SS]` prefix, since TorchDistributor buffers output and Databricks UI may not show real-time.
- **`logall()` vs `log()`**: `logall()` prints on all ranks with `[rank N]` prefix for debugging. `log()` prints on rank 0 only for production. Before `dist.init_process_group()`, `logall()` falls back to `[pid N]`.
- **MLflow only on rank 0**: avoids duplicate metrics. Auth propagated from driver via `DATABRICKS_HOST`/`DATABRICKS_TOKEN` env vars extracted from the driver's MLflow credential provider.

## Environment Variable Propagation

The driver process has implicit Databricks auth. Child processes (spawned by torchrun) don't. The auth bridge:

```mermaid
flowchart LR
    subgraph "Driver Process"
        CREDS["mlflow.utils.databricks_utils<br/>.get_databricks_host_creds()"]
        CREDS --> SET["os.environ[DATABRICKS_HOST] = host<br/>os.environ[DATABRICKS_TOKEN] = token"]
    end

    SET --> TD2["TorchDistributor.run()"]
    TD2 --> FORK["torchrun forks children"]

    subgraph "Child Processes (inherit env)"
        FORK --> C0["rank 0: mlflow.start_run() works"]
        FORK --> C1["rank 1-3: no MLflow calls"]
    end
```

Other propagated env vars:
| Variable | Purpose |
|---|---|
| `PYTHONPATH` | Points to workspace files so `transolver3` and `dataset` are importable |
| `NCCL_P2P_DISABLE=1` | Disable GPU P2P on PCIe-connected A10Gs (g5.12xlarge) |
| `NCCL_ASYNC_ERROR_HANDLING=1` | Non-blocking NCCL error detection |

## Instance Types

| Target | Instance | GPUs | VRAM | Mesh Sharding | Use Case |
|---|---|---|---|---|---|
| `a10g` | g5.12xlarge | 4× A10G | 96 GB | 4-way | Default training |
| `a100_40` | p4d.24xlarge | 8× A100 | 320 GB | 8-way | Large-scale training |
| `a100_80` | p4de.24xlarge | 8× A100-80 | 640 GB | 8-way | Full DrivAerML (140M pts) |

The `a10g` target uses ON_DEMAND instances to avoid spot capacity issues with g5.12xlarge.
