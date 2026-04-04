#!/usr/bin/env python3
"""
Generate weather forecasting dataset from raw HRRR Zarr archives.
(Reference script — documents how the .pt dataset was originally created.)

NOTE FOR FUTURE REFERENCE (after Tufts HPC access expires):
    This script was run on the Tufts Pax HPC cluster to generate the training
    dataset from raw HRRR (High-Resolution Rapid Refresh) weather data stored
    in zarr format. It is kept here for documentation purposes — to understand
    the data pipeline and reproduce the dataset if needed.

    Source data on HPC:
        /cluster/tufts/c26sp1cs0137/data/assignment2_data/hrrr_ne_anl.zarr
        /cluster/tufts/c26sp1cs0137/data/assignment2_data/hrrr_ne_apcp.zarr
    These are xarray-compatible zarr stores containing HRRR analysis fields
    and accumulated precipitation for the New England region.

    Generated output (416 GB total):
        dataset/inputs/YYYY/X_YYYYMMDDHH.pt  — one (450, 449, 42) bfloat16
                                               tensor per hour, containing
                                               42 weather variables over a
                                               3 km Lambert Conformal grid
        dataset/targets.pt                   — (T, 6) float32 target values
                                               at the Jumbo Statue grid point
                                               + binary precip label
        dataset/metadata.pt                  — variable names, grid coords,
                                               projection info, target location

    Dependencies (only available in HPC 'geo' conda env):
        xarray, zarr, cartopy, numpy, pandas, torch

    Data coverage: ~3 years of hourly data starting 2018-07-13, split as:
        Train: 2018-2019 | Validation: 2020 | Test: 2021

    Variables (42 total = 7 target-level + 35 atmospheric):
        See data_spec.py for full variable list (VAR_LEVELS).
        Target vars: TMP@2m, RH@2m, UGRD@10m, VGRD@10m, GUST@sfc, APCP@sfc
        Atmospheric: pressure-level temps, dewpoints, geopotential heights,
                     winds, cloud cover, precipitable water, etc.

Task 1: For each hour t (first 3 years), save all weather variables as a
        torch tensor of shape (450, 449, c) in bfloat16 to:
            dataset/inputs/YYYY/X_YYYYMMDDHH.pt

Task 2: Find the grid point nearest to the Jumbo Statue at Tufts, extract
        the 6 target variables and binary precipitation label for all time
        steps, and save to:
            dataset/targets.pt

Usage:
    conda activate geo
    python generate_dataset.py

Note: The x-dimension in the actual data is 449 (not 450 as in the project
specification), because the NE region slice is x[1350:1799] = 449 pixels.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from cartopy import crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_spec import ATMOS_VARS, TARGET_VARS, VAR_LEVELS, projection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATASET_DIR = DATA_DIR / "dataset"
N_STEPS = 3 * 365 * 24          # 26 280 hourly snapshots (≈ 3 years)
ZARR_CHUNK = 24                  # match the zarr time-chunk for efficient I/O

# Jumbo Statue at Tufts University
JUMBO_LAT = 42.40777867717294
JUMBO_LON = -71.12041637590173

# Target variables at the prediction point (order matters for the tensor)
TASK2_TARGET_VARS = [
    "TMP@2m_above_ground",
    "RH@2m_above_ground",
    "UGRD@10m_above_ground",
    "VGRD@10m_above_ground",
    "GUST@surface",
    "APCP_1hr_acc_fcst@surface",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_nearest_grid_point(ds):
    """Return (iy, ix) of the grid point closest to the Jumbo Statue."""
    proj_x, proj_y = projection.transform_point(
        JUMBO_LON, JUMBO_LAT, ccrs.PlateCarree()
    )
    ix = int(np.argmin(np.abs(ds.x.values - proj_x)))
    iy = int(np.argmin(np.abs(ds.y.values - proj_y)))
    return iy, ix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Weather Dataset Generator")
    print("=" * 60)

    # -- Open zarr stores ---------------------------------------------------
    print("\nOpening zarr stores …")
    ds_anl  = xr.open_zarr(DATA_DIR / "hrrr_ne_anl.zarr")
    ds_apcp = xr.open_zarr(DATA_DIR / "hrrr_ne_apcp.zarr")

    # Subset to first 3 years using the ANL time axis
    ds_anl_3yr = ds_anl.isel(time=slice(0, N_STEPS))
    times = ds_anl_3yr.time.values          # numpy datetime64[ns], shape (26280,)

    print(f"  ANL  time range : {times[0]} → {times[-1]}")
    print(f"  Steps           : {N_STEPS}")

    # Verify APCP time alignment (both stores are hourly from 2018-07-13)
    apcp_head = ds_apcp.time.values[:N_STEPS]
    if not np.array_equal(times, apcp_head):
        raise ValueError(
            "APCP and ANL time indices do not match for the first "
            f"{N_STEPS} steps – check the zarr stores."
        )
    ds_apcp_3yr = ds_apcp.isel(time=slice(0, N_STEPS))
    print("  APCP time alignment verified.")

    # -- Locate Jumbo Statue ------------------------------------------------
    iy, ix = find_nearest_grid_point(ds_anl)
    jumbo_x = float(ds_anl.x.values[ix])
    jumbo_y = float(ds_anl.y.values[iy])
    print(f"\nJumbo Statue nearest grid point: y_idx={iy}, x_idx={ix}")
    print(f"  Projection coords: x={jumbo_x:.0f} m, y={jumbo_y:.0f} m")

    # Create top-level dataset directory
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Task 2: Target values at Jumbo Statue location
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("Task 2 – Extracting target values at Jumbo Statue …")

    anl_target_vars = [v for v in TASK2_TARGET_VARS
                       if v != "APCP_1hr_acc_fcst@surface"]

    # Load into memory (small: T × 5 floats)
    targets_anl = (
        ds_anl_3yr[anl_target_vars]
        .isel(y=iy, x=ix)
        .compute()
    )
    apcp_vals = (
        ds_apcp_3yr["APCP_1hr_acc_fcst@surface"]
        .isel(y=iy, x=ix)
        .compute()
        .values                     # numpy (T,)
    )

    # Stack → (T, 6) float32
    cols = [targets_anl[v].values for v in anl_target_vars] + [apcp_vals]
    target_values  = np.stack(cols, axis=1).astype(np.float32)   # (T, 6)
    binary_label   = (apcp_vals > 2.0)                           # (T,) bool

    target_path = DATASET_DIR / "targets.pt"
    torch.save(
        {
            "time":           times,
            "variable_names": TASK2_TARGET_VARS,
            "values":         torch.tensor(target_values, dtype=torch.float32),
            "binary_label":   torch.tensor(binary_label,  dtype=torch.bool),
            "grid_y_idx":     iy,
            "grid_x_idx":     ix,
            "grid_proj_x":    jumbo_x,
            "grid_proj_y":    jumbo_y,
        },
        target_path,
    )
    print(f"  Saved  {target_path}")
    print(f"  values shape : {target_values.shape}  (T × 6 continuous targets)")
    print(f"  rainy samples: {binary_label.sum()} / {N_STEPS}  (APCP > 2 mm)")

    # -----------------------------------------------------------------------
    # Task 1: Input tensors – one .pt file per hour
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("Task 1 – Generating input tensors …")
    print(f"  Variables   : {len(VAR_LEVELS)}  (TARGET_VARS first, then ATMOS_VARS)")
    print(f"  Tensor shape: (450, 449, {len(VAR_LEVELS)})  bfloat16")
    print(f"  Output dir  : {DATASET_DIR}/inputs/YYYY/X_YYYYMMDDHH.pt")

    input_dir = DATASET_DIR / "inputs"

    for chunk_start in range(0, N_STEPS, ZARR_CHUNK):
        chunk_end   = min(chunk_start + ZARR_CHUNK, N_STEPS)
        chunk_times = times[chunk_start:chunk_end]

        # Load this time chunk into memory (all 41 ANL vars + APCP)
        anl_chunk  = ds_anl_3yr.isel(time=slice(chunk_start, chunk_end)).compute()
        apcp_chunk = (
            ds_apcp_3yr["APCP_1hr_acc_fcst@surface"]
            .isel(time=slice(chunk_start, chunk_end))
            .compute()
            .values                 # (T, y, x)
        )

        # Build (T, y, x, c) array in VAR_LEVELS order
        arrays = []
        for var in VAR_LEVELS:
            if var == "APCP_1hr_acc_fcst@surface":
                arrays.append(apcp_chunk)
            else:
                arrays.append(anl_chunk[var].values)
        chunk_data = np.stack(arrays, axis=-1)  # (T, 450, 449, c)

        # Save one .pt per hour
        for i, t in enumerate(chunk_times):
            dt = pd.Timestamp(t)
            year_dir = input_dir / str(dt.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            fname = year_dir / f"X_{dt.strftime('%Y%m%d%H')}.pt"

            # Skip if already generated (allows resume after interruption)
            if fname.exists():
                continue

            tensor = torch.tensor(chunk_data[i], dtype=torch.bfloat16)
            torch.save(tensor, fname)

        if chunk_start % (ZARR_CHUNK * 50) == 0:
            pct = 100.0 * chunk_end / N_STEPS
            print(f"  {chunk_end:>6}/{N_STEPS}  ({pct:5.1f}%)  "
                  f"last: {pd.Timestamp(chunk_times[-1])}")

    # Save metadata alongside the dataset
    meta_path = DATASET_DIR / "metadata.pt"
    torch.save(
        {
            "variable_names":  VAR_LEVELS,
            "n_vars":          len(VAR_LEVELS),
            "input_shape":     (450, 449, len(VAR_LEVELS)),
            "times":           times,
            "grid_x":          ds_anl.x.values,
            "grid_y":          ds_anl.y.values,
            "projection":      "LambertConformal(central_lon=262.5, central_lat=38.5)",
            "target_vars":     TASK2_TARGET_VARS,
            "jumbo_y_idx":     iy,
            "jumbo_x_idx":     ix,
        },
        meta_path,
    )
    print(f"\nSaved metadata: {meta_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  dataset/inputs/YYYY/X_YYYYMMDDHH.pt  ← input tensors")
    print(f"  dataset/targets.pt                    ← target scalars + label")
    print(f"  dataset/metadata.pt                   ← variable names, grid info")


if __name__ == "__main__":
    main()
