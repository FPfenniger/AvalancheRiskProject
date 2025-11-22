import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input Paths
CLEANED_DATA_PATH = 'data/cleaned_data.parquet'
GEOJSON_PATH = 'data/slf_boundaries_2020.json' # From the snippet provided earlier
GRID_META_PATH = 'data/grids/grid_metadata.npz'

# Output
OUTPUT_DIR = 'data/grids/targets'

# Grid Definitions (MUST match Script 1 exactly)
LAT_MIN, LAT_MAX = 45.8, 47.9
LON_MIN, LON_MAX = 5.9, 10.6
GRID_STEP = 0.01 

# Value to use for pixels OUTSIDE any warning region (e.g., France, Italy)
# We use -1 so we can mask these out in the Loss Function later.
NO_DATA_VALUE = -1 

# ==========================================
# 2. SETUP
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
# Load Forecast Data
df = pd.read_parquet(CLEANED_DATA_PATH)

# Load Polygons (Warning Regions)
gdf_regions = gpd.read_file(GEOJSON_PATH)

# Ensure IDs match type (dataset uses integers for 'warnreg')
# GeoJSON IDs might be strings or have a specific column name like 'id' or 'warnreg_id'
# Adjust 'id' below to whatever column holds the ID in the GeoJSON (likely 'id' or 'warnreg_id')
# We assume the GeoJSON has a column that matches df['warnreg']
if 'id' in gdf_regions.columns:
    gdf_regions['warnreg'] = gdf_regions['id'].astype(int)

# Load Grid Metadata to ensure alignment
grid_meta = np.load(GRID_META_PATH)
height = len(grid_meta['lats'])
width = len(grid_meta['lons'])

print(f"Target Grid Shape: ({height}, {width})")

# Reconstruct the Affine Transform (Critical for alignment)
# This maps Lat/Lon coordinates to Pixel Row/Col
transform = from_bounds(
    west=LON_MIN, south=LAT_MIN, east=LON_MAX, north=LAT_MAX, 
    width=width, height=height
)

# ==========================================
# 3. RASTERIZATION LOOP
# ==========================================
# We group by Date to create one map per day
daily_groups = df.groupby('datum')

print(f"Rasterizing targets for {len(daily_groups)} days...")

count = 0
for date, group in tqdm(daily_groups):
    # Format date for filename
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    save_path = os.path.join(OUTPUT_DIR, f"{date_str}.npy")
    
    if os.path.exists(save_path):
        continue

    # 1. Prepare the (Geometry, Value) list for this day
    # We need to map: Polygon -> Danger Level
    shapes = []
    
    # Get valid regions for this day
    # (Some days might not have forecasts for all regions)
    day_records = group[['warnreg', 'dangerLevel']].drop_duplicates()
    
    for _, row in day_records.iterrows():
        region_id = row['warnreg']
        danger_level = row['dangerLevel']
        
        # Find the polygon for this region
        region_poly = gdf_regions[gdf_regions['warnreg'] == region_id]
        
        if not region_poly.empty:
            # Append (geometry, value) tuple
            shapes.append((region_poly.geometry.values[0], danger_level))
            
    # 2. Rasterize
    if shapes:
        # rasterize() burns the shapes into an array
        target_grid = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=NO_DATA_VALUE,  # Background value (outside regions)
            transform=transform,
            all_touched=True,    # If a pixel touches a region, it counts
            dtype=np.int8
        )
        
        # Flip to match Script 2 (South-Up / Index 0 = South) if necessary
        # Recall: Script 1 required a flip. Rasterio usually renders North-Up (Top-Down).
        # Since Script 2 (np.meshgrid) produces South-Up, we likely need to flip this
        # to align with the weather grids. 
        target_grid = np.flipud(target_grid)
        
        # Save
        np.save(save_path, target_grid)
        count += 1
    else:
        # Should not happen if data is clean
        pass

print(f"\nProcessing complete. {count} target maps generated.")
print(f"Saved to: {OUTPUT_DIR}")