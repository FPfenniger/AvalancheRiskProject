import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import Point
import os
from tqdm import tqdm

# Inputs
CLEANED_DATA_PATH = '../data/cleaned_data.parquet'
GEOJSON_PATH = '../data/slf_boundaries_2020.json'
GRID_META_PATH = '../data/grids/grid_metadata.npz'
OUTPUT_DIR = '../data/grids/targets'

LAT_MIN, LAT_MAX = 45.8, 47.9
LON_MIN, LON_MAX = 5.9, 10.6
NO_DATA_VALUE = -1 

# Spatial Mapping
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(CLEANED_DATA_PATH)
gdf_regions = gpd.read_file(GEOJSON_PATH)

# standard ID for regions
region_id_col = 'id' if 'id' in gdf_regions.columns else 'warnreg_id'
gdf_regions['region_id'] = gdf_regions[region_id_col].astype(int)

# get stations
stations = df[['station_code', 'lon', 'lat']].drop_duplicates()

# GeoDataFrame
gdf_stations = gpd.GeoDataFrame(
    stations, 
    geometry=gpd.points_from_xy(stations.lon, stations.lat),
    crs="EPSG:4326"
)

# 3. Join! Find which polygon each station is inside
# 'op' is 'within' or 'intersects' (syntax varies by geopandas version, 'predicate' is newer)
# using sjoin checks if station is *inside* region
joined = gpd.sjoin(gdf_stations, gdf_regions[['geometry', 'region_id']], how="inner", predicate="within")

# 4. Create a mapping dictionary: station_code -> region_id
station_to_region = joined.set_index('station_code')['region_id'].to_dict()

print(f"Mapped {len(station_to_region)} stations to valid warning regions.")

# 5. Map this back to the main dataframe
df['mapped_region_id'] = df['station_code'].map(station_to_region)

# Drop rows where station isn't in a region (e.g. border stations outside polygons)
df_clean = df.dropna(subset=['mapped_region_id']).copy()
df_clean['mapped_region_id'] = df_clean['mapped_region_id'].astype(int)

# ==========================================
# 3. GRID SETUP
# ==========================================
grid_meta = np.load(GRID_META_PATH)
height = len(grid_meta['lats'])
width = len(grid_meta['lons'])
transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, width, height)

# ==========================================
# 4. RASTERIZATION LOOP
# ==========================================
daily_groups = df_clean.groupby('datum')
print(f"Rasterizing targets for {len(daily_groups)} days...")

count = 0
for date, group in tqdm(daily_groups):
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    save_path = os.path.join(OUTPUT_DIR, f"{date_str}.npy")
    
    if os.path.exists(save_path):
        continue

    shapes = []
    # Group by the CORRECT mapped ID we just found
    # We take the max danger level if multiple stations in one region disagree 
    # (Conservative approach for Ground Truth)
    region_danger = group.groupby('mapped_region_id')['dangerLevel'].max()
    
    for region_id, danger_level in region_danger.items():
        # Get polygon geometry
        poly = gdf_regions[gdf_regions['region_id'] == region_id].geometry.values[0]
        shapes.append((poly, danger_level))
            
    if shapes:
        target_grid = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=NO_DATA_VALUE,
            transform=transform,
            all_touched=True,
            dtype=np.int8
        )
        
        # FLIP Check: Match Script 1/2 (South-Up)
        target_grid = np.flipud(target_grid)
        
        np.save(save_path, target_grid)
        count += 1

print(f"\nProcessing complete. {count} target maps generated.")