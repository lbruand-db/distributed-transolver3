"""
Download raw DrivAerML files from HuggingFace into a Databricks UC Volume.

Downloads boundary VTP files, per-run geo_parameters CSVs, and the
global geo_parameters_all.csv — without any conversion. The raw files
are stored under output_dir/raw/ preserving the HF directory structure:

    output_dir/raw/
        geo_parameters_all.csv
        run_056/
            boundary_056.vtp
            geo_parameters_056.csv
        run_058/
            boundary_058.vtp
            geo_parameters_058.csv
        ...

Usage (Databricks job or local):
    python scripts/download_drivaer_raw.py \
        --output_dir /Volumes/lucasbruand_catalog/cfd/data \
        [--runs 56 58 60]          # specific runs, or omit for all
        [--workers 8]              # concurrent downloads (default: 8)
        [--dry-run]                # list what would be downloaded
"""

import argparse
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


HF_REPO = "neashton/drivaerml"
HF_REPO_TYPE = "dataset"


def get_available_run_ids():
    """Query HuggingFace for all available run IDs with boundary VTP files."""
    import re

    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type=HF_REPO_TYPE)
    vtp_files = [f for f in files if "boundary" in f and f.endswith(".vtp")]
    return sorted(int(re.search(r"run_(\d+)", f).group(1)) for f in vtp_files)


def get_existing_run_ids(raw_dir):
    """List run IDs that already have VTP files in the raw directory."""
    import re

    if not os.path.exists(raw_dir):
        return []

    existing = []
    for entry in os.listdir(raw_dir):
        m = re.match(r"run_(\d+)", entry)
        if not m:
            continue
        run_dir = os.path.join(raw_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        run_id = int(m.group(1))
        vtp = os.path.join(run_dir, f"boundary_{run_id}.vtp")
        if os.path.exists(vtp):
            existing.append(run_id)
    return sorted(existing)


def download_run(run_id, raw_dir):
    """Download VTP + CSV for a single run into raw_dir/run_NNN/.

    Each thread gets its own temp dir to avoid HF cache contention.
    """
    from huggingface_hub import hf_hub_download

    tmp_dir = tempfile.mkdtemp(prefix=f"hf_run{run_id}_")
    try:
        run_dir = os.path.join(raw_dir, f"run_{run_id:03d}")
        os.makedirs(run_dir, exist_ok=True)

        # Download boundary VTP
        vtp_dest = os.path.join(run_dir, f"boundary_{run_id}.vtp")
        if not os.path.exists(vtp_dest):
            vtp_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"run_{run_id}/boundary_{run_id}.vtp",
                repo_type=HF_REPO_TYPE,
                cache_dir=tmp_dir,
            )
            shutil.copy2(vtp_path, vtp_dest)

        # Download per-run geo_parameters CSV
        csv_dest = os.path.join(run_dir, f"geo_parameters_{run_id}.csv")
        if not os.path.exists(csv_dest):
            try:
                csv_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=f"run_{run_id}/geo_parameters_{run_id}.csv",
                    repo_type=HF_REPO_TYPE,
                    cache_dir=tmp_dir,
                )
                shutil.copy2(csv_path, csv_dest)
            except Exception:
                pass  # per-run CSV is optional

        size_mb = os.path.getsize(vtp_dest) / 1024**2
        return run_id, True, f"{size_mb:.1f} MB"
    except Exception as e:
        return run_id, False, str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_global_csv(raw_dir):
    """Download geo_parameters_all.csv."""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(raw_dir, "geo_parameters_all.csv")
    if os.path.exists(dest):
        print("Global geo_parameters_all.csv already exists, skipping.", flush=True)
        return

    print("Downloading geo_parameters_all.csv ...", flush=True)
    tmp_dir = tempfile.mkdtemp(prefix="hf_global_")
    try:
        csv_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="geo_parameters_all.csv",
            repo_type=HF_REPO_TYPE,
            cache_dir=tmp_dir,
        )
        shutil.copy2(csv_path, dest)
        print(f"Saved {dest}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Thread-safe counters for progress reporting
_lock = threading.Lock()
_completed = 0


def main():
    global _completed

    parser = argparse.ArgumentParser(
        description="Download raw DrivAerML files from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        default="/Volumes/lucasbruand_catalog/cfd/data",
        help="UC Volume path. Raw files go under output_dir/raw/",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to download (default: all missing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    # UC Volumes must be pre-created; verify the base path exists
    if not os.path.exists(args.output_dir):
        print(f"ERROR: output_dir does not exist: {args.output_dir}", flush=True)
        print("UC Volumes must be created beforehand via SQL:", flush=True)
        print("  CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.<volume>", flush=True)
        raise SystemExit(1)

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Determine which runs to download
    print("Querying HuggingFace for available runs...", flush=True)
    available = set(get_available_run_ids())
    print(f"  {len(available)} runs available on HF", flush=True)

    existing = set(get_existing_run_ids(raw_dir))
    print(f"  {len(existing)} runs already in {raw_dir}", flush=True)

    if args.runs:
        to_download = sorted(r for r in args.runs if r in available)
        skipped = sorted(r for r in args.runs if r not in available)
        if skipped:
            print(f"  WARNING: runs not on HF (skipped): {skipped}", flush=True)
    else:
        to_download = sorted(available - existing)

    total = len(to_download)
    print(f"\n  {total} runs to download ({args.workers} workers)", flush=True)

    if args.dry_run:
        print(f"  Dry run — would download: {to_download}", flush=True)
        return

    if not to_download:
        print("  Nothing to do — all runs already downloaded.", flush=True)
        return

    # Download global CSV first (serial)
    try:
        download_global_csv(raw_dir)
    except Exception as e:
        print(f"WARNING: Could not download global CSV: {e}", flush=True)

    # Download runs concurrently
    _completed = 0
    success = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_run, run_id, raw_dir): run_id
            for run_id in to_download
        }
        for future in as_completed(futures):
            run_id, ok, msg = future.result()
            with _lock:
                _completed += 1
                progress = _completed
            if ok:
                success += 1
                print(f"  [{progress}/{total}] Run {run_id}: OK ({msg})", flush=True)
            else:
                failed.append(run_id)
                print(f"  [{progress}/{total}] Run {run_id}: FAILED ({msg})", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. {success} downloaded, {len(failed)} failed.", flush=True)
    if failed:
        print(f"Failed runs: {sorted(failed)}", flush=True)


if __name__ == "__main__":
    main()
