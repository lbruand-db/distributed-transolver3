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
        [--dry-run]                # list what would be downloaded
"""

import argparse
import os
import shutil


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
        # Check that the VTP file actually exists
        run_id = int(m.group(1))
        vtp = os.path.join(run_dir, f"boundary_{run_id}.vtp")
        if os.path.exists(vtp):
            existing.append(run_id)
    return sorted(existing)


def download_run(run_id, raw_dir, tmp_dir):
    """Download VTP + CSV for a single run into raw_dir/run_NNN/."""
    from huggingface_hub import hf_hub_download

    run_dir = os.path.join(raw_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # Download boundary VTP
    vtp_dest = os.path.join(run_dir, f"boundary_{run_id}.vtp")
    if os.path.exists(vtp_dest):
        print("  SKIP VTP: already exists", flush=True)
    else:
        print(f"  Downloading run_{run_id}/boundary_{run_id}.vtp ...", flush=True)
        vtp_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"run_{run_id}/boundary_{run_id}.vtp",
            repo_type=HF_REPO_TYPE,
            cache_dir=tmp_dir,
        )
        shutil.copy2(vtp_path, vtp_dest)
        size_mb = os.path.getsize(vtp_dest) / 1024**2
        print(f"  Saved VTP ({size_mb:.1f} MB)", flush=True)

    # Download per-run geo_parameters CSV
    csv_dest = os.path.join(run_dir, f"geo_parameters_{run_id}.csv")
    if os.path.exists(csv_dest):
        print("  SKIP CSV: already exists", flush=True)
    else:
        try:
            csv_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"run_{run_id}/geo_parameters_{run_id}.csv",
                repo_type=HF_REPO_TYPE,
                cache_dir=tmp_dir,
            )
            shutil.copy2(csv_path, csv_dest)
            print("  Saved geo_parameters CSV", flush=True)
        except Exception as e:
            print(f"  WARNING: no per-run CSV for run {run_id}: {e}", flush=True)


def download_global_csv(raw_dir, tmp_dir):
    """Download geo_parameters_all.csv."""
    from huggingface_hub import hf_hub_download

    dest = os.path.join(raw_dir, "geo_parameters_all.csv")
    if os.path.exists(dest):
        print("Global geo_parameters_all.csv already exists, skipping.", flush=True)
        return

    print("Downloading geo_parameters_all.csv ...", flush=True)
    csv_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="geo_parameters_all.csv",
        repo_type=HF_REPO_TYPE,
        cache_dir=tmp_dir,
    )
    shutil.copy2(csv_path, dest)
    print(f"Saved {dest}", flush=True)


def main():
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
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

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

    print(f"\n  {len(to_download)} runs to download", flush=True)

    if args.dry_run:
        print(f"  Dry run — would download: {to_download}", flush=True)
        return

    if not to_download:
        print("  Nothing to do — all runs already downloaded.", flush=True)
        return

    # Use a temp dir for HF cache to avoid filling the volume with cache metadata
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="hf_drivaer_")

    # Download global CSV first
    try:
        download_global_csv(raw_dir, tmp_dir)
    except Exception as e:
        print(f"WARNING: Could not download global CSV: {e}", flush=True)

    # Download each run
    success = 0
    failed = []
    for i, run_id in enumerate(to_download):
        print(f"\n[{i + 1}/{len(to_download)}] Run {run_id}", flush=True)
        try:
            download_run(run_id, raw_dir, tmp_dir)
            success += 1
        except Exception as e:
            print(f"  FAILED run {run_id}: {e}", flush=True)
            failed.append(run_id)

        # Periodically flush HF cache to save disk
        if (i + 1) % 10 == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir, exist_ok=True)

    # Final cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Done. {success} downloaded, {len(failed)} failed.", flush=True)
    if failed:
        print(f"Failed runs: {failed}", flush=True)


if __name__ == "__main__":
    main()
