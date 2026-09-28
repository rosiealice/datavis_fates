#!/usr/bin/env python
"""
Simplified script for creating tree-shaped visualizations of FATES_VEGC_SZPF data (Case 10 only)
Creates animations showing size x PFT multiplexed data as tree shapes
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Point
from shapely.prepared import prep
import glob
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation name
CASENAME = "NFM-L2-PFCLM51-SPITFIRE24-ECAregion2-RCcase35-c231222_r386_hist_GSWP3-W5E5"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.2026-01-08"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.31j.2026-02-01"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CRUJRA_BT3_LSP_MS3.25_BT1EBT.2026-01-24"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.BT3_LSP.2026-01-08"
CASENAME="i2000.f45_f45_mg37.fatesnocomp.noresm3_0_beta10.CRUJRA.06f_16hp.2026-02-06"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.02f_NWTcal.2026-02-02"
CASENAME="i2000.f45_f45_mg37.fatesnocomp.noresm3_0_beta10.CRUJRA.30j.2026-02-06"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta10.CRUJRA.06f_14hp_noLUC.2026-02-09"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta10.CRUJRA.06f_14hp_noLUC.2026-02-09"

# Data paths
DATA_DIR = Path(f"/datalake/NS9560K/www/diagnostics/noresm/{CASENAME}/lnd/hist")
DATA_DIR = Path(f"//datalake/NS9560K/rosief/{CASENAME}/lnd/hist")
OUTPUT_PATH = f"/datalake/NS9560K/www/diagnostics/noresm/rosief/{CASENAME}/gifs"
hend = "*h0a*.nc"

jessie_run=0
if(jessie_run==1):
    DATA_PATH= "/datalake/NS9188K/share/jessien/feb26/"
    CASENAME="noresm-fates_f45_2000s_ent_agbonly.2026-02-03"
    CASENAME="noresm-fates_ne16_transient_1900_borealseeds.2025-11-19"
    CASENAME="noresm-fates_f45_2000s_feb20c.2026-02-20"



    hend="*h0*.nc"

# Variable configuration for Case 10
VAR_NAMES = ["FATES_VEGC_SZPF"]
UNIT_LABELS = {var: "kgC/m2" for var in VAR_NAMES}
SCALING_FACTORS = {var: 1 for var in VAR_NAMES}

# Time range and processing
YEAR_RANGE = (1, 120)  # Year range to process as (start_year, end_year)
FILE_INTERVAL = 12  # Process every Nth timestep
tint = 200  # time interval in ms for animation
OUTPUT_FORMAT = "gif"  # "gif" or "mp4"

# PFT configurations
PFT_CONFIG = {
    'bet': {'name': 'BET'},
    'ent': {'name': 'ENT'},
    'bdt': {'name': 'BDT'}
}
PFT_LIST = ['bdt','bet','ent' ]  # Which PFT-dominated sites to plot

# Tree shape configuration for PFT dimension indices
PFT_INDEX_TO_SHAPE = {
    0: {'style': 'cloud', 'color': 'orange', 'width_scale': 0.6, 'height_scale': 1.3},
    1: {'style': 'christmas', 'color': 'green', 'width_scale': 1.0, 'height_scale': 1.0},
    2: {'style': 'christmas', 'color': '#6a1b9a', 'width_scale': 1.0, 'height_scale': 1.0},
    3: {'style': 'christmas', 'color': '#d84315', 'width_scale': 1.0, 'height_scale': 1.0},
    4: {'style': 'christmas', 'color': '#f57c00', 'width_scale': 1.0, 'height_scale': 1.0},
    5: {'style': 'cloud', 'color': 'blue', 'width_scale': 1.5, 'height_scale': 1.0},
    6: {'style': 'christmas', 'color': '#558b2f', 'width_scale': 1.0, 'height_scale': 1.0},
    7: {'style': 'christmas', 'color': '#ad1457', 'width_scale': 1.0, 'height_scale': 1.0},
    8: {'style': 'christmas', 'color': '#6d4c41', 'width_scale': 1.0, 'height_scale': 1.0},
    9: {'style': 'christmas', 'color': '#455a64', 'width_scale': 1.0, 'height_scale': 1.0},
    10: {'style': 'christmas', 'color': '#827717', 'width_scale': 1.0, 'height_scale': 1.0},
    11: {'style': 'christmas', 'color': '#4e342e', 'width_scale': 1.0, 'height_scale': 1.0},
    12: {'style': 'christmas', 'color': '#5d4037', 'width_scale': 1.0, 'height_scale': 1.0},
    'default': {'style': 'christmas', 'color': '#8b4513', 'width_scale': 1.0, 'height_scale': 1.0}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_nearest_point(target_lat, target_lon, lats, lons):
    """Find the nearest grid point to target lat/lon"""
    if lats.ndim == 1 and lons.ndim == 1 and lats.shape != lons.shape:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
        lats_flat = lats_2d.flatten()
        lons_flat = lons_2d.flatten()
    else:
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
    
    valid_mask = ~(np.isnan(lats_flat) | np.isnan(lons_flat))
    if not np.any(valid_mask):
        print("ERROR: All lat/lon coordinates are NaN!")
        return 0, np.nan, np.nan
    
    distances = np.full_like(lats_flat, np.inf)
    distances[valid_mask] = np.sqrt((lats_flat[valid_mask] - target_lat)**2 + 
                                    (lons_flat[valid_mask] - target_lon)**2)
    idx = np.argmin(distances)
    return idx, lats_flat[idx], lons_flat[idx]

def get_country(lat, lon):
    """Get country name from lat/lon coordinates"""
    try:
        import cartopy.io.shapereader as shpreader
        if lon > 180:
            lon = lon - 360
        
        shpfilename = shpreader.natural_earth(resolution='110m',
                                             category='cultural',
                                             name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        countries = reader.records()
        point = Point(lon, lat)
        
        for country in countries:
            if prep(country.geometry).contains(point):
                return country.attributes['NAME']
        return "Unknown"
    except Exception as e:
        return "Unknown"

def get_us_state(lat, lon):
    """Get US state name from lat/lon coordinates"""
    try:
        import cartopy.io.shapereader as shpreader
        if lon > 180:
            lon = lon - 360
        
        shpfilename = shpreader.natural_earth(resolution='110m',
                                             category='cultural',
                                             name='admin_1_states_provinces')
        reader = shpreader.Reader(shpfilename)
        states = reader.records()
        point = Point(lon, lat)
        
        for state in states:
            if state.attributes.get('admin') == 'United States of America':
                if prep(state.geometry).contains(point):
                    return state.attributes['name']
        return None
    except Exception as e:
        print(f"Error getting US state: {e}")
        return None

def read_location_file(pft_name, max_sites=8):
    """Read locations from PFT-specific file"""
    locations_file = Path(f'/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/pft_grid_cells/{pft_name}_grid_cells.txt')
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        return []
    
    locations = []
    with open(locations_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            # Skip header line or empty lines
            if not line or line.startswith('#') or line.lower().startswith('lat'):
                continue
            
            # Try space-separated format first
            parts = line.split()
            if len(parts) >= 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    # Generate a name if not provided
                    name = parts[2] if len(parts) >= 3 else f"Site {len(locations) + 1}"
                    locations.append((lat, lon, name))
                    if len(locations) >= max_sites:
                        break
                except ValueError:
                    continue
            # Try comma-separated format as fallback
            elif ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        name = parts[2].strip() if len(parts) >= 3 else f"Site {len(locations) + 1}"
                        locations.append((lat, lon, name))
                        if len(locations) >= max_sites:
                            break
                    except ValueError:
                        continue
    return locations

def demultiplex_data(data_1d, n_sizes, n_pfts):
    """Reshape multiplexed 1D data into 2D size x PFT array (PFT-major ordering)"""
    return data_1d.reshape(n_pfts, n_sizes).T

def create_tree_shape(x_center, height, base_width=0.3, style="christmas", 
                     width_scale_factor=1.0, height_scale_factor=1.0):
    """Create a tree-shaped polygon for visualization"""
    if np.isscalar(height) and height <= 0:
        return None
    elif not np.isscalar(height) and (height <= 0).any():
        return None
    
    height = height * height_scale_factor
    width_scale = 0.3 + (float(np.mean(height)) if not np.isscalar(height) else height) / 100.0 * 0.5
    width = base_width * width_scale * width_scale_factor
    trunk_proportion = 0.15
    
    if style == "cloud":
        if width_scale_factor < 0.8:
            crown_proportion = 0.65
            canopy_width_multiplier = 1.1
            trunk_proportion = 0.35
        else:
            crown_proportion = 0.85
            canopy_width_multiplier = 0.6
            trunk_proportion = 0.15
        
        tree_total_height = float(height) if np.isscalar(height) else float(np.mean(height))
        trunk_height = tree_total_height * trunk_proportion
        crown_height = tree_total_height * crown_proportion
        trunk_width = width * 0.15
        tree_base = 0
        crown_base = trunk_height
        n_points = 40
        
        vertices = [
            [x_center - trunk_width/2, tree_base],
            [x_center - trunk_width/2, crown_base],
        ]
        
        for i in range(n_points + 1):
            angle = np.pi * i / n_points
            x_offset = width * np.cos(angle) * canopy_width_multiplier
            
            if width_scale_factor < 0.8:
                base_height = crown_base + crown_height * np.sin(angle)
                lobes = crown_height * 0.18 * np.sin(angle * 6)
                small_bumps = crown_height * 0.06 * np.sin(angle * 13)
                y_pos = base_height + lobes + small_bumps
            else:
                base_height = crown_base + crown_height * (0.3 + 0.7 * np.sin(angle))
                bump = crown_height * 0.08 * np.sin(angle * 5)
                y_pos = base_height + bump
            
            vertices.append([x_center + x_offset, y_pos])
        
        vertices.append([x_center + trunk_width/2, crown_base])
        vertices.append([x_center + trunk_width/2, tree_base])
    else:
        trunk_height = float(height) * trunk_proportion if np.isscalar(height) else float(np.mean(height)) * trunk_proportion
        tree_base = 0
        crown_height = float(height) - trunk_height if np.isscalar(height) else float(np.mean(height)) - trunk_height
        trunk_width = width * 0.15
        n_tiers = 4
        
        vertices = [[x_center - trunk_width/2, tree_base], [x_center - trunk_width/2, trunk_height]]
        
        for i in range(n_tiers):
            tier_bottom = trunk_height + (i * crown_height / n_tiers)
            tier_mid = trunk_height + ((i + 0.5) * crown_height / n_tiers)
            bottom_width_ratio = 1.0 - (i * 0.7 / n_tiers)
            mid_width_ratio = 1.0 - ((i + 0.5) * 0.7 / n_tiers)
            vertices.append([x_center - width/2 * bottom_width_ratio, tier_bottom])
            vertices.append([x_center - width/2 * mid_width_ratio * 0.7, tier_mid])
        
        vertices.append([x_center, float(height) if np.isscalar(height) else float(np.mean(height))])
        
        for i in range(n_tiers - 1, -1, -1):
            tier_bottom = trunk_height + (i * crown_height / n_tiers)
            tier_mid = trunk_height + ((i + 0.5) * crown_height / n_tiers)
            bottom_width_ratio = 1.0 - (i * 0.7 / n_tiers)
            mid_width_ratio = 1.0 - ((i + 0.5) * 0.7 / n_tiers)
            vertices.append([x_center + width/2 * mid_width_ratio * 0.7, tier_mid])
            vertices.append([x_center + width/2 * bottom_width_ratio, tier_bottom])
        
        vertices.append([x_center + trunk_width/2, trunk_height])
        vertices.append([x_center + trunk_width/2, tree_base])
    
    return Polygon(vertices, closed=True)

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("="*80)
print("CASE 10: FATES_VEGC_SZPF Tree Visualization")
print("="*80)

# Build file list
print("\nFinding data files...")
file_pattern = str(DATA_DIR / "*h0a*.nc")
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    print(f"\nERROR: No files found matching pattern: {file_pattern}")
    exit(1)

print(f"Found {len(all_files)} files total")

# Build timestep list
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
print(f"Processing {n_times} timesteps")

# Get dimensions from first file
print("\nReading metadata from first file...")
with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
    VAR_NAME = VAR_NAMES[0]
    var_data = ds[VAR_NAME]
    
    print(f"\nVariable '{VAR_NAME}' dimensions: {var_data.dims}")
    
    # Get lat/lon
    lat = ds.lat.values
    lon = ds.lon.values
    
    # Get size dimension
    size_dim = None
    for dim in ds.dims:
        dim_lower = dim.lower()
        if any(pattern in dim_lower for pattern in ['size', 'sz', 'fates_levscls', 'levscls']):
            size_dim = dim
            n_sizes = len(ds[size_dim])
            print(f"Found {n_sizes} size classes from dimension '{size_dim}'")
            break
    
    if size_dim is None:
        print(f"ERROR: Could not determine size dimension")
        exit(1)
    
    # Get PFT dimension
    pft_dim = None
    for dim in ds.dims:
        dim_lower = dim.lower()
        if 'pft' in dim_lower or 'fates_levpft' in dim_lower or 'levpft' in dim_lower:
            pft_dim = dim
            n_pfts = len(ds[pft_dim])
            print(f"Found {n_pfts} PFT classes from dimension '{pft_dim}'")
            break
    
    if pft_dim is None:
        print(f"ERROR: Could not determine PFT dimension")
        exit(1)
    
    # Get size labels
    if f"{size_dim}_bounds" in ds.variables or f"fates_{size_dim}_bounds" in ds.variables:
        size_bounds_var = f"{size_dim}_bounds" if f"{size_dim}_bounds" in ds.variables else f"fates_{size_dim}_bounds"
        size_bounds = ds[size_bounds_var].values
        size_labels = [f"{size_bounds[i,0]:.1f}-{size_bounds[i,1]:.1f}" for i in range(n_sizes)]
    else:
        size_values = ds[size_dim].values
        size_labels = [f"{val:.0f}" for val in size_values]
    
    # Get PFT labels and styles
    pft_labels = []
    pft_styles = []
    pft_colors = []
    pft_width_scales = []
    pft_height_scales = []
    
    for i in range(n_pfts):
        pft_labels.append(f"PFT{i+1}")
        shape_config = PFT_INDEX_TO_SHAPE.get(i, PFT_INDEX_TO_SHAPE['default'])
        pft_styles.append(shape_config['style'])
        pft_colors.append(shape_config['color'])
        pft_width_scales.append(shape_config['width_scale'])
        pft_height_scales.append(shape_config['height_scale'])

# Process each PFT
for pft_name in PFT_LIST:
    pft_upper = pft_name.upper()
    print(f"\n{'='*60}")
    print(f"Processing PFT: {pft_upper}")
    print(f"{'='*60}\n")
    
    # Read locations
    LOCATIONS = read_location_file(pft_name, max_sites=8)
    if not LOCATIONS:
        print(f"Skipping PFT: {pft_upper}")
        continue
    
    print(f"Loaded {len(LOCATIONS)} locations for {pft_upper}")
    
    # Find nearest grid points
    location_indices = []
    for target_lat, target_lon, name in LOCATIONS:
        idx, actual_lat, actual_lon = find_nearest_point(target_lat, target_lon, lat, lon)
        location_indices.append((idx, actual_lat, actual_lon, name))
        print(f"Location '{name}': target=({target_lat}, {target_lon}), actual=({actual_lat:.2f}, {actual_lon:.2f})")
    
    # Setup figure
    n_locations = len(location_indices)
    n_cols = min(n_locations, 4)
    n_rows = (n_locations + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14*n_cols/4, 8*n_rows/2))
    
    # Set background
    funky_colors = ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20']
    funky_cmap = LinearSegmentedColormap.from_list('funky_green', funky_colors)
    fig.patch.set_facecolor('#c8e6c9')
    
    if n_locations == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    display_name = VAR_NAME.replace('FATES_', '')
    fig.suptitle(f'{pft_upper} Points - {display_name}', fontsize=20, fontweight='bold', y=0.98)
    
    # Initialize tree collections
    tree_collections_by_location = []
    x_pos = np.arange(n_sizes)
    
    for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
        ax = axs[loc_idx]
        
        # Add background gradient
        gradient_array = np.linspace(0, 1, 512).reshape(512, 1)
        gradient = ax.imshow(gradient_array, extent=[-0.5, n_sizes - 0.5, 0, 100], 
                             aspect='auto', cmap=funky_cmap, alpha=0.6, zorder=0,
                             origin='upper', interpolation='bilinear')
        
        # Get first frame data
        with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
            var_data = ds[VAR_NAME].isel(time=0)
            
            if 'ncol' in var_data.dims:
                data_1d = var_data.isel(ncol=grid_idx).values * SCALING_FACTORS[VAR_NAME]
            elif 'lndgrid' in var_data.dims:
                data_1d = var_data.isel(lndgrid=grid_idx).values * SCALING_FACTORS[VAR_NAME]
            else:
                lat_idx = grid_idx // len(ds.lon)
                lon_idx = grid_idx % len(ds.lon)
                data_1d = var_data.isel(lat=lat_idx, lon=lon_idx).values * SCALING_FACTORS[VAR_NAME]
            
            data_2d = demultiplex_data(data_1d, n_sizes, n_pfts)
        
        # Create tree collections
        pft_tree_collections = []
        pft_offset_width = 0.15
        for pft_idx in range(n_pfts):
            trees = []
            pft_x_offset = (pft_idx - (n_pfts - 1) / 2) * pft_offset_width
            for size_idx, height in enumerate(data_2d[:, pft_idx]):
                if height > 0:
                    tree = create_tree_shape(x_pos[size_idx] + pft_x_offset, height, 
                                          base_width=0.8, 
                                          style=pft_styles[pft_idx],
                                          width_scale_factor=pft_width_scales[pft_idx],
                                          height_scale_factor=pft_height_scales[pft_idx])
                    if tree is not None:
                        trees.append(tree)
            
            if trees:
                collection = PatchCollection(trees, facecolors=pft_colors[pft_idx],
                                            edgecolors='darkgreen', linewidths=1.0,
                                            alpha=0.7, zorder=2+pft_idx,
                                            label=pft_labels[pft_idx])
                ax.add_collection(collection)
                pft_tree_collections.append((collection, trees, pft_idx))
        
        tree_collections_by_location.append(pft_tree_collections)
        
        ax.set_xlabel('Size Class (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{UNIT_LABELS[VAR_NAME]}', fontsize=12, fontweight='bold')
        ax.set_title(f'Year 0, Month 01', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_labels, rotation=45, ha='right')
        ax.set_xlim(-0.5, 10.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add location info
        country_name = get_country(actual_lat, actual_lon)
        us_state = get_us_state(actual_lat, actual_lon)
        if us_state:
            location_text = f"{us_state}, USA - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
        else:
            location_text = f"{country_name} - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
        ax.text(0.98, 0.98, location_text, transform=ax.transAxes, 
               fontsize=9, fontweight='bold', ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=10)
    
    # Hide unused subplots
    for idx in range(n_locations, len(axs)):
        axs[idx].axis('off')
    
    plt.tight_layout()
    
    # Determine y-axis limits
    max_value = 0
    for file_path, time_idx, _ in file_time_map[-min(10, len(file_time_map)):]:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            var_data = ds[VAR_NAME].isel(time=time_idx)
            for grid_idx, _, _, _ in location_indices:
                if 'ncol' in var_data.dims:
                    data_1d = var_data.isel(ncol=grid_idx).values * SCALING_FACTORS[VAR_NAME]
                elif 'lndgrid' in var_data.dims:
                    data_1d = var_data.isel(lndgrid=grid_idx).values * SCALING_FACTORS[VAR_NAME]
                else:
                    lat_idx = grid_idx // len(ds.lon)
                    lon_idx = grid_idx % len(ds.lon)
                    data_1d = var_data.isel(lat=lat_idx, lon=lon_idx).values * SCALING_FACTORS[VAR_NAME]
                data_2d = demultiplex_data(data_1d, n_sizes, n_pfts)
                max_value = max(max_value, np.nanmax(data_2d))
    
    for ax in axs[:n_locations]:
        ax.set_ylim(0, max_value * 1.2)
        if len(ax.images) > 0:
            ax.images[0].set_extent([-0.5, n_sizes - 0.5, 0, max_value * 1.2])
    
    # Animation update function
    current_file = None
    current_ds = None
    
    def update_trees(frame):
        global current_file, current_ds
        print(f"\rProcessing frame {frame+1}/{n_times}", end='', flush=True)
        
        file_path, time_idx, time_val = file_time_map[frame]
        
        if current_file != file_path:
            if current_ds is not None:
                current_ds.close()
            current_ds = xr.open_dataset(file_path, decode_times=False)
            current_file = file_path
        
        for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
            ax = axs[loc_idx]
            pft_collections = tree_collections_by_location[loc_idx]
            
            var_data = current_ds[VAR_NAME].isel(time=time_idx)
            if 'ncol' in var_data.dims:
                data_1d = var_data.isel(ncol=grid_idx).values * SCALING_FACTORS[VAR_NAME]
            elif 'lndgrid' in var_data.dims:
                data_1d = var_data.isel(lndgrid=grid_idx).values * SCALING_FACTORS[VAR_NAME]
            else:
                lat_idx = grid_idx // len(current_ds.lon)
                lon_idx = grid_idx % len(current_ds.lon)
                data_1d = var_data.isel(lat=lat_idx, lon=lon_idx).values * SCALING_FACTORS[VAR_NAME]
            
            data_2d = demultiplex_data(data_1d, n_sizes, n_pfts)
            
            pft_offset_width = 0.15
            for collection, old_trees, pft_idx in pft_collections:
                new_trees = []
                pft_x_offset = (pft_idx - (n_pfts - 1) / 2) * pft_offset_width
                for size_idx, height in enumerate(data_2d[:, pft_idx]):
                    if height > 0:
                        tree = create_tree_shape(x_pos[size_idx] + pft_x_offset, height,
                                               base_width=0.8,
                                               style=pft_styles[pft_idx],
                                               width_scale_factor=pft_width_scales[pft_idx],
                                               height_scale_factor=pft_height_scales[pft_idx])
                        if tree is not None:
                            new_trees.append(tree)
                
                if new_trees:
                    collection.set_paths(new_trees)
            
            actual_timestep = (start_timestep or 0) + (frame * FILE_INTERVAL)
            year = actual_timestep // 12
            month = actual_timestep % 12 + 1
            ax.set_title(f'Year {year}, Month {month:02d}', fontsize=14, fontweight='bold')
        
        return [coll for loc_colls in tree_collections_by_location for coll, _, _ in loc_colls]
    
    # Create animation
    print(f"\nCreating tree animation with {n_times} frames...")
    anim = animation.FuncAnimation(
        fig, update_trees, frames=n_times,
        interval=tint,
        blit=True,
        repeat=True
    )
    
    # Save animation
    year_range_str = f"_y{YEAR_RANGE[0]}-{YEAR_RANGE[1]}"
    output_file = f"{OUTPUT_PATH}/fates_vegc_szpf_trees{year_range_str}_{pft_upper}.{OUTPUT_FORMAT}"
    print(f"\nSaving {pft_upper} animation to {output_file}...")
    
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_FORMAT == "mp4":
        import shutil
        if shutil.which('ffmpeg') is None:
            print("WARNING: ffmpeg not found. Falling back to GIF format.")
            output_file = output_file.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=1000/tint)
            anim.save(output_file, writer=writer, dpi=100)
        else:
            writer = animation.FFMpegWriter(fps=1000/tint, metadata=dict(artist='FATES Analysis'), bitrate=1800)
            anim.save(output_file, writer=writer, dpi=100)
    else:
        writer = animation.PillowWriter(fps=1000/tint)
        anim.save(output_file, writer=writer, dpi=100)
    
    print(f"Animation saved successfully!")
    plt.close(fig)
    
    if current_ds is not None:
        current_ds.close()

print("\n" + "="*80)
print("CASE 10 PROCESSING COMPLETED")
print("="*80)
