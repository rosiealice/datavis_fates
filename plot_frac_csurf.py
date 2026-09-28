#!/usr/bin/env python3
"""
Script to plot all variables in the NetCDF file.
For 2D variables with natpft dimension, creates subplots for each PFT.
For 1D variables, creates a single spatial plot.
Handles irregular grid data.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# File paths
nc_file = "/nird/home/rosief/fates_landuse_pft_map_to_surfdata_ne16np4_260209_cdf5.nc"
surfdata_file = "/nird/home/rosief/surfdata_ne16np4.pg3_hist_2000_16pfts_c260209.nc"

def plot_variable_2d(var_data, var_name, lat_flat, lon_flat, use_geographic):
    """Plot 2D variable with natpft dimension as subplots"""
    n_pft = var_data.shape[0]  # Assuming natpft is first dimension
    
    # Set up subplots
    n_cols = min(4, n_pft)
    n_rows = int(np.ceil(n_pft / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    if n_pft == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    print(f"Creating {n_rows}x{n_cols} subplot grid for {var_name}")
    
    # Plot each PFT
    for i in range(n_pft):
        ax = axes[i]
        
        # Extract data for this PFT
        data = var_data.isel(natpft=i).values
        
        # Flatten data if needed
        if data.ndim > 1:
            data_flat = data.flatten()
        else:
            data_flat = data
        
        # Remove NaN values for plotting
        mask = ~np.isnan(data_flat)
        
        if use_geographic and mask.sum() > 0:
            # Create scatter plot on irregular grid using lat/lon
            scatter = ax.scatter(lon_flat[mask], lat_flat[mask], c=data_flat[mask], 
                               s=10, cmap='viridis', vmin=0, vmax=np.nanmax(var_data.values))
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
        else:
            # Plot using grid indices
            grid_indices = np.arange(len(data_flat))
            if mask.sum() > 0:
                scatter = ax.scatter(grid_indices[mask], data_flat[mask], 
                                   c=data_flat[mask], s=10, cmap='viridis', 
                                   vmin=0, vmax=np.nanmax(var_data.values))
            else:
                scatter = None
            
            ax.set_xlabel('Grid Index')
            ax.set_ylabel(var_name)
        
        ax.set_title(f'PFT {i+1}')
        
        # Add colorbar
        if mask.sum() > 0:
            plt.colorbar(scatter, ax=ax, label=var_name)
    
    # Hide unused subplots
    for i in range(n_pft, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{var_name} by Natural PFT', fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = f'{var_name}_by_pft.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close(fig)

def plot_variable_1d(var_data, var_name, lat_flat, lon_flat, use_geographic):
    """Plot 1D variable as single spatial plot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    data_flat = var_data.values
    if data_flat.ndim > 1:
        data_flat = data_flat.flatten()
    
    # Remove NaN values for plotting
    mask = ~np.isnan(data_flat)
    
    if use_geographic and mask.sum() > 0:
        # Create scatter plot on irregular grid using lat/lon
        scatter = ax.scatter(lon_flat[mask], lat_flat[mask], c=data_flat[mask], 
                           s=10, cmap='viridis')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
    else:
        # Plot as line or scatter with grid indices
        grid_indices = np.arange(len(data_flat))
        if mask.sum() > 0:
            scatter = ax.scatter(grid_indices[mask], data_flat[mask], 
                               c=data_flat[mask], s=10, cmap='viridis')
        else:
            scatter = None
        
        ax.set_xlabel('Grid Index')
        ax.set_ylabel(var_name)
    
    ax.set_title(f'{var_name}')
    
    # Add colorbar
    if mask.sum() > 0:
        plt.colorbar(scatter, ax=ax, label=var_name)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{var_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close(fig)

def main():
    print(f"Reading NetCDF file: {nc_file}")
    
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)
    
    # Get list of all data variables
    data_vars = list(ds.data_vars.keys())
    print(f"\nFound {len(data_vars)} variables to plot: {data_vars}")
    
    # Get coordinates - try multiple approaches
    lat = None
    lon = None
    use_geographic = False
    
    if 'lat' in ds.coords and 'lon' in ds.coords:
        lat = ds['lat'].values
        lon = ds['lon'].values
        use_geographic = True
    elif 'lats' in ds.coords and 'lons' in ds.coords:
        lat = ds['lats'].values
        lon = ds['lons'].values
        use_geographic = True
    elif 'lat' in ds.variables and 'lon' in ds.variables:
        lat = ds['lat'].values
        lon = ds['lon'].values
        use_geographic = True
    elif 'LATIXY' in ds.variables and 'LONGXY' in ds.variables:
        lat = ds['LATIXY'].values
        lon = ds['LONGXY'].values
        use_geographic = True
    else:
        # Try loading from surfdata file
        try:
            print(f"Coordinates not found in main file. Trying surfdata file: {surfdata_file}")
            ds_surf = xr.open_dataset(surfdata_file)
            if 'LATIXY' in ds_surf.variables and 'LONGXY' in ds_surf.variables:
                lat = ds_surf['LATIXY'].values
                lon = ds_surf['LONGXY'].values
                use_geographic = True
                print("Successfully loaded coordinates from surfdata file")
            ds_surf.close()
        except Exception as e:
            print(f"Could not load coordinates from surfdata file: {e}")
        
        if not use_geographic:
            print("Available coordinates:", list(ds.coords.keys()))
            coord_vars = [v for v in ds.variables.keys() if 'lat' in v.lower() or 'lon' in v.lower()]
            print("Possible coordinate variables:", coord_vars)
            print("WARNING: Could not find lat/lon coordinates. Will plot using grid indices.")
            use_geographic = False
    
    if use_geographic:
        print(f"Latitude shape: {lat.shape}")
        print(f"Longitude shape: {lon.shape}")
        
        # Flatten coordinates if needed
        if lat.ndim > 1:
            lat_flat = lat.flatten()
            lon_flat = lon.flatten()
        else:
            lat_flat = lat
            lon_flat = lon
    else:
        lat_flat = None
        lon_flat = None
    
    # Plot each variable
    print("\n" + "="*60)
    for var_name in data_vars:
        print(f"\nProcessing variable: {var_name}")
        var_data = ds[var_name]
        print(f"  Shape: {var_data.shape}")
        print(f"  Dimensions: {var_data.dims}")
        
        try:
            # Check if variable has natpft dimension
            if 'natpft' in var_data.dims:
                plot_variable_2d(var_data, var_name, lat_flat, lon_flat, use_geographic)
            else:
                plot_variable_1d(var_data, var_name, lat_flat, lon_flat, use_geographic)
        except Exception as e:
            print(f"  ERROR plotting {var_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("All plots completed!")
    
    # Close dataset
    ds.close()

if __name__ == "__main__":
    main()
