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

"""Tests for convert_vtp_to_npz_batch.py: param loading and list_unconverted_runs.py."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from convert_vtp_to_npz_batch import load_run_params
from list_unconverted_runs import main as list_main


# ===== load_run_params =====


def test_load_params_per_run_csv():
    """Should load from per-run CSV when it exists."""
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = os.path.join(raw_dir, "run_056")
        os.makedirs(run_dir)

        # Write a per-run CSV
        with open(os.path.join(run_dir, "geo_parameters_56.csv"), "w") as f:
            f.write("run,param1,param2,param3\n")
            f.write("56,1.5,2.5,3.5\n")

        params = load_run_params(raw_dir, 56)

        assert params.dtype == np.float32
        assert len(params) == 3
        np.testing.assert_array_almost_equal(params, [1.5, 2.5, 3.5])


def test_load_params_global_csv_fallback():
    """Should fall back to global CSV when per-run CSV is missing."""
    with tempfile.TemporaryDirectory() as raw_dir:
        # No per-run CSV, but global CSV exists
        with open(os.path.join(raw_dir, "geo_parameters_all.csv"), "w") as f:
            f.write("run,a,b,c,d\n")
            f.write("55,1.0,2.0,3.0,4.0\n")
            f.write("56,5.0,6.0,7.0,8.0\n")
            f.write("57,9.0,10.0,11.0,12.0\n")

        params = load_run_params(raw_dir, 56)

        assert len(params) == 4
        np.testing.assert_array_almost_equal(params, [5.0, 6.0, 7.0, 8.0])


def test_load_params_global_csv_run_not_found():
    """Should return zeros when run_id is not in global CSV."""
    with tempfile.TemporaryDirectory() as raw_dir:
        with open(os.path.join(raw_dir, "geo_parameters_all.csv"), "w") as f:
            f.write("run,a,b\n")
            f.write("55,1.0,2.0\n")

        params = load_run_params(raw_dir, 999)

        assert len(params) == 16  # default zeros
        assert params.sum() == 0


def test_load_params_no_csv_at_all():
    """Should return zeros when no CSV files exist."""
    with tempfile.TemporaryDirectory() as raw_dir:
        params = load_run_params(raw_dir, 56)

        assert len(params) == 16
        assert params.sum() == 0


def test_load_params_per_run_excludes_run_column():
    """The 'run' column should be excluded from params."""
    with tempfile.TemporaryDirectory() as raw_dir:
        run_dir = os.path.join(raw_dir, "run_010")
        os.makedirs(run_dir)

        with open(os.path.join(run_dir, "geo_parameters_10.csv"), "w") as f:
            f.write("Run,x,y\n")
            f.write("10,3.14,2.71\n")

        params = load_run_params(raw_dir, 10)

        # Should only have x and y, not Run
        assert len(params) == 2
        np.testing.assert_array_almost_equal(params, [3.14, 2.71], decimal=2)


# ===== list_unconverted_runs =====


def _setup_raw_dir(tmpdir, raw_runs, npz_runs):
    """Create fake raw/ VTP files and NPZ files for testing."""
    raw_dir = os.path.join(tmpdir, "raw")
    os.makedirs(raw_dir)

    for run_id in raw_runs:
        run_dir = os.path.join(raw_dir, f"run_{run_id:03d}")
        os.makedirs(run_dir)
        # Create empty VTP file
        open(os.path.join(run_dir, f"boundary_{run_id}.vtp"), "w").close()

    for run_id in npz_runs:
        open(os.path.join(tmpdir, f"drivaer_{run_id:03d}.npz"), "w").close()


def test_list_discovers_unconverted(capsys):
    """Should find VTP files that don't have corresponding NPZ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_raw_dir(tmpdir, raw_runs=[56, 58, 60], npz_runs=[56])

        old_argv = sys.argv
        sys.argv = ["list_unconverted_runs.py", "--data_dir", tmpdir]
        try:
            list_main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "3 raw VTP files found" in output
        assert "1 NPZ files already exist" in output
        assert "2 runs to convert" in output


def test_list_all_converted(capsys):
    """Should report 0 runs when everything is converted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _setup_raw_dir(tmpdir, raw_runs=[56, 58], npz_runs=[56, 58])

        old_argv = sys.argv
        sys.argv = ["list_unconverted_runs.py", "--data_dir", tmpdir]
        try:
            list_main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "0 runs to convert" in output


def test_list_no_raw_dir():
    """Should exit with error when raw/ doesn't exist."""
    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        old_argv = sys.argv
        sys.argv = ["list_unconverted_runs.py", "--data_dir", tmpdir]
        try:
            with pytest.raises(SystemExit):
                list_main()
        finally:
            sys.argv = old_argv
