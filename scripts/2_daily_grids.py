import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
from tqdm import tqdm
import shutil


def interpolate_day(day_df, grid_lat, grid_lon, features):
    """
    Interpolates all features for a single day onto the grid.
    
    Arguments:
        day_df: DataFrame containing station data for one day
        grid_lat: 2D meshgrid of target latitudes
        grid_lon: 2D meshgrid of target longitudes
        features: List of feature column names to interpolate
    
    Returns:
        np.ndarray: Tensor of shape (Height, Width, Channels)
    """
    # Station coordinates for specific day
    points = day_df[['lat', 'lon']].values
    day_grids = []
    
    for feat in features:
        values = day_df[feat].values
        # 1. Linear Interpolation 
        grid_linear = griddata(points, values, (grid_lat, grid_lon), method='linear')
        # 2. Nearest Neighbor (Fill edges/extrapolate) for NANs
        mask_nan = np.isnan(grid_linear)
        if mask_nan.any():
            grid_nearest = griddata(points, values, (grid_lat[mask_nan], grid_lon[mask_nan]), method='nearest')
            grid_linear[mask_nan] = grid_nearest

        day_grids.append(grid_linear)
        
    return np.stack(day_grids, axis=-1)


def generate_daily_grids():
    """
    Generates daily interpolated grids for all dynamic weather features.
    
    Reads cleaned station data and interpolates each feature onto a 1km grid
    for each day in the dataset. Saves results as compressed .npz files.
    """
    # Inputs 
    clean_data = '../data/cleaned_data.parquet'
    grid_metadata = '../data/grids/grid_metadata.npz'
    output = '../data/grids/dynamic'

    # Input Channels of Dynamic Features (most important & predictive features from EDA)
    dynamic_features = [
        'delta_elevation',   
        'Pen_depth',          
        'HN24',              # Daily snowfall
        'MS_Snow',           
        'TA',                 
        'wind_trans24',       
        'RH',             
        'min_ccl_pen',        
        'relative_load_3d',   
    ]

    # 1. Setup
    os.makedirs(output, exist_ok=True)
    print("Loading data...")
    df = pd.read_parquet(clean_data)
    grid_meta = np.load(grid_metadata)

    # Prepare Target Grid (2D meshgrid of lat, lon)
    target_lats = grid_meta['lats']
    target_lons = grid_meta['lons']
    grid_lat_mesh, grid_lon_mesh = np.meshgrid(target_lats, target_lons, indexing='ij')
    print(f"Target Grid Shape: {grid_lat_mesh.shape}")

    # Creation of Wind vectors 
    if 'wind_u' not in df.columns:
        # Convert DW to radians (0 degrees = North, 90 = East)
        # Mathematical standard: 0 = East. Meteorological: 0 = North (blowing FROM).
        # Standard conversion: u = -ws * sin(wd), v = -ws * cos(wd)
        wd_rad = np.deg2rad(df['DW'])
        df['wind_u'] = -df['VW'] * np.sin(wd_rad)
        df['wind_v'] = -df['VW'] * np.cos(wd_rad)
    
    # Final feature list (use wind_u/wind_v instead of VW/DW)
    features_to_use = dynamic_features + ['wind_u', 'wind_v']

    # 3. Loop through days
    daily_groups = df.groupby('datum')
    print(f"Starting interpolation for {len(daily_groups)} days...")
    error_count = 0

    for date, group in tqdm(daily_groups):
        try:
            # Skip days with too few stations 
            if len(group) < 10:
                continue
               
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            save_path = os.path.join(output, f"{date_str}.npz")
            if os.path.exists(save_path):
                continue
                
            # Run Interpolation
            daily_tensor = interpolate_day(group, grid_lat_mesh, grid_lon_mesh, features_to_use)
            np.savez_compressed(save_path, data=daily_tensor.astype(np.float32))
            
        except Exception as e:
            print(f"Failed on {date}: {e}")
            error_count += 1

    print(f"\n✓ Processing complete: Complete. Saved {len(features_to_use)}-channel grids to {output}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    generate_daily_grids()