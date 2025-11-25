import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import Point
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import Point
import os
from tqdm import tqdm


def rasterize_targets():
    """
    Rasterizes avalanche danger levels onto a 1km grid for each day.
    
    Maps station danger levels to SLF warning regions, then burns these
    polygon-based labels into a raster grid matching the CNN input dimensions.
    Saves daily target maps as .npy files.
    """
    # Inputs
    cleaned_data = '../data/cleaned_data.parquet'
    slf_json = '../data/slf_boundaries_2020.json'
    grid_metadata = '../data/grids/grid_metadata.npz'
    output = '../data/grids/targets'

    lat_min, lat_max = 45.8, 47.9
    lon_min, lon_max = 5.9, 10.6
    no_data_val = -1 

    # 1. Spatial Mapping
    os.makedirs(output, exist_ok=True)
    print("Loading data...")
    df = pd.read_parquet(cleaned_data)
    gdf_regions = gpd.read_file(slf_json)

    # Standard ID for regions
    region_id_col = 'id' if 'id' in gdf_regions.columns else 'warnreg_id'
    gdf_regions['region_id'] = gdf_regions[region_id_col].astype(int)
    stations = df[['station_code', 'lon', 'lat']].drop_duplicates()

    # Create GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        stations, 
        geometry=gpd.points_from_xy(stations.lon, stations.lat),
        crs="EPSG:4326"
    )

    # Spatial join: which stations fall inside which warning regions?
    print("Mapping stations to warning regions...")
    joined = gpd.sjoin(gdf_stations, gdf_regions[['geometry', 'region_id']], 
                       how="inner", predicate="within")
    station_to_region = joined.set_index('station_code')['region_id'].to_dict()
    print(f"Mapped {len(station_to_region)} stations to valid warning regions.")

    df['mapped_region_id'] = df['station_code'].map(station_to_region)
    df_clean = df.dropna(subset=['mapped_region_id']).copy()
    df_clean['mapped_region_id'] = df_clean['mapped_region_id'].astype(int)

    # 2. Grid Setup
    print("Loading grid metadata...")
    grid_meta = np.load(grid_metadata)
    height = len(grid_meta['lats'])
    width = len(grid_meta['lons'])
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    # 3. Rasterization Loop
    daily_groups = df_clean.groupby('datum')
    print(f"Rasterizing targets for {len(daily_groups)} days...")

    count = 0
    for date, group in tqdm(daily_groups):
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        save_path = os.path.join(output, f"{date_str}.npy")
        
        if os.path.exists(save_path):
            continue

        shapes = []
        region_danger = group.groupby('mapped_region_id')['danger_level'].max() # Take max danger level if multiple stations 
        for region_id, danger_level in region_danger.items():
            poly = gdf_regions[gdf_regions['region_id'] == region_id].geometry.values[0]
            shapes.append((poly, danger_level))
                
        if shapes:
            target_grid = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                fill=no_data_val,
                transform=transform,
                all_touched=True,
                dtype=np.int8
            )
            
            # Flip to match Script 1/2 orientation (South-Up)
            target_grid = np.flipud(target_grid)
            
            np.save(save_path, target_grid)
            count += 1

    print(f"Processing complete. {count} target maps generated.")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    rasterize_targets()