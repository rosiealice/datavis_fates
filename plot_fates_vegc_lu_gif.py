#!/usr/bin/env python
"""
Script to visualize FATES_VEGC_LU as a timeseries GIF.
Each subplot shows a map for a LU (land use class) member.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import glob

# CONFIGURATION
CASENAME = "i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta10.CRUJRA.06f_14hp_noLUC.2026-02-09"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta11.CRUJRA.14hlmpft_LUCy2k_2000FSx16pft_14fpft_d2bl1.2026-02-20"

jessie_run=1
if(jessie_run==1):
    DATA_PATH="/datalake/NS9188K/share/jessien/"
    CASENAME="noresm-fates_f45_2000s_ent_agbonly.2026-02-03"
    CASENAME='noresm-fates_ne16_transient_1900_borealseeds.2025-11-19'
    DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/run/")
    YEAR_RANGE = (1, 120)
else: 
    DATA_DIR = Path(f"//datalake/NS9560K/rosief/{CASENAME}/lnd/hist")
    YEAR_RANGE = (1, 120)

OUTPUT_PATH = f"/datalake/NS9560K/www/diagnostics/noresm/rosief/{CASENAME}/gifs"

VAR_NAME = "FATES_PATCHAREA_LU"
#VAR_NAME = "FATES_VEGC_LU"

FILE_INTERVAL = 24
OUTPUT_FORMAT = "gif"
tint = 200  # ms per frame

# Find files
file_pattern = str(DATA_DIR /"*h0*.nc")
all_files = sorted(glob.glob(file_pattern))
if not all_files:
    print(f"No files found: {file_pattern}")
    exit(1)

print(f"Found {len(all_files)} files matching pattern.")

# Build time map
file_time_map = []
start_timestep = YEAR_RANGE[0] * 12 if YEAR_RANGE else None
end_timestep = YEAR_RANGE[1] * 12 if YEAR_RANGE else None
timestep_counter = 0
for i, file_path in enumerate(all_files):
    if start_timestep is not None and timestep_counter < start_timestep:
        timestep_counter += 1
        continue
    if end_timestep is not None and timestep_counter >= end_timestep:
        break
    if (timestep_counter - (start_timestep or 0)) % FILE_INTERVAL == 0:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            file_time_map.append((file_path, 0, ds.time.values[0]))
    timestep_counter += 1
n_times = len(file_time_map)

print(f"Processing {n_times} timesteps.")

# Read metadata
with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
    var = ds[VAR_NAME]
    # Detect grid type
    if 'ncol' in var.dims:
        grid_type = 'ncol'
        lat = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
        lon = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
        n_points = len(lat)
    elif 'lndgrid' in var.dims:
        grid_type = 'lndgrid'
        lat = ds['lat'].values
        lon = ds['lon'].values
        n_points = len(lat)
    else:
        grid_type = 'regular'
        lat = ds.lat.values
        lon = ds.lon.values
    # LU dimension
    if 'lu' in var.dims:
        n_lu = ds.dims['lu']
    else:
        n_lu = var.shape[1] if len(var.shape) > 1 else 1
    print(f"Variable shape: {var.shape}, n_lu: {n_lu}, grid_type: {grid_type}")

import math
# Setup figure: LU subplots on 2x3 grid
n_rows, n_cols = 2, 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
axs = axs.flatten()
if n_lu < n_rows * n_cols:
    for idx in range(n_lu, n_rows * n_cols):
        axs[idx].axis('off')

# Find vmin/vmax for color scale using only the last file
print("Calculating color scale (vmin/vmax) from last file...")
last_file_path, last_time_idx, _ = file_time_map[-1]
with xr.open_dataset(last_file_path, decode_times=False) as ds:
    data = ds[VAR_NAME].isel(time=last_time_idx).values
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
print(f"Color scale: vmin={vmin}, vmax={vmax}")

# Animation update function
def update(frame):
    file_path, time_idx, time_val = file_time_map[frame]
    print(f"Rendering frame {frame+1}/{n_times}: {file_path}")
    with xr.open_dataset(file_path, decode_times=False) as ds:
        data = ds[VAR_NAME].isel(time=time_idx).values
        # Unpack LU for each grid cell: shape (ngrid, n_lu) or (n_lu, ngrid)
        if data.shape[0] == n_lu:
            # shape (n_lu, ngrid)
            data = data
        elif data.shape[1] == n_lu:
            # shape (ngrid, n_lu)
            data = data.T
        else:
            print(f"Warning: Unexpected data shape {data.shape}, cannot unpack LU.")
            data = np.full((n_lu, len(lon)), np.nan)
        lu_labels = ["Primary", "Secondary", "Rangeland", "Pasture", "Crop"]
        for i in range(n_lu):
            ax = axs[i]
            ax.clear()
            d = data[i, :]
            # Handle structured (2D) and unstructured (1D) grids
            if lat.ndim == 2 and lon.ndim == 2:
                # Structured grid: reshape d to match lat/lon
                if d.size == lat.size:
                    d_reshaped = d.reshape(lat.shape)
                    sc = ax.imshow(d_reshaped, vmin=vmin, vmax=vmax, cmap='viridis',
                                  origin='lower', aspect='auto')
                else:
                    print(f"Warning: d size {d.size} does not match lat/lon size {lat.size}.")
                    sc = None
            elif lat.ndim == 1 and lon.ndim == 1:
                # Try to handle structured grid with 1D lat/lon
                n_lat = len(np.unique(lat))
                n_lon = len(np.unique(lon))
                if d.size == n_lat * n_lon:
                    lat2d, lon2d = np.meshgrid(np.unique(lat), np.unique(lon), indexing='ij')
                    d_reshaped = d.reshape((n_lat, n_lon))
                    sc = ax.imshow(d_reshaped, vmin=vmin, vmax=vmax, cmap='viridis',
                                  origin='lower', aspect='auto')
                elif len(lat) == len(lon) == len(d):
                    # True unstructured grid
                    sc = ax.scatter(lon, lat, c=d, vmin=vmin, vmax=vmax, cmap='viridis', s=8)
                else:
                    print(f"Shape mismatch: lat {lat.shape}, lon {lon.shape}, d {d.shape}. Skipping plot.")
                    sc = None
            else:
                print(f"Unknown lat/lon shape: lat {lat.shape}, lon {lon.shape}. Skipping plot.")
                sc = None
            label = lu_labels[i] if i < len(lu_labels) else f'LU {i+1}'
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused axes if n_lu < n_rows*n_cols
        for idx in range(n_lu, n_rows * n_cols):
            axs[idx].clear()
            axs[idx].axis('off')
    fig.suptitle(f"{VAR_NAME} - Timestep {frame+1}/{n_times}")
    return fig.axes

# Create animation
print("Starting animation rendering...")
anim = animation.FuncAnimation(fig, update, frames=n_times, interval=tint, blit=False, repeat=True)

# Save animation
output_file = f"{OUTPUT_PATH}/{VAR_NAME}_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}.gif"
# Ensure output directory exists
output_dir = Path(OUTPUT_PATH)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving animation to {output_file} ...")
writer = animation.PillowWriter(fps=1000/tint)
anim.save(output_file, writer=writer, dpi=100)
print(f"Saved animation: {output_file}")
plt.close(fig)
