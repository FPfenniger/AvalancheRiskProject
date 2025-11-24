import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
from tqdm import tqdm


# Inputs 
CLEANED_DATA_PATH = '../data/cleaned_data.parquet'
GRID_META_PATH = '../data/grids/grid_metadata.npz'
OUTPUT_DIR = '../data/grids/dynamic'

# Define the list of dynamic features to interpolate
# These act as the input channels for your CNN
DYNAMIC_FEATURES = [
    # --- 1. The "Super Predictors" (Top Correlations) ---
    'delta_elevation',    # Rank #2 (r=-0.601). Critical context.
    'Pen_depth',          # Rank #3 (r=+0.581). Best proxy for "unstable new snow".
    'HN72_24',            # Rank #4 (r=+0.543). Best measure of cumulative loading.

    # --- 2. The "Driver" Group (Weather) ---
    'TA',                 # Rank #7 (r=-0.420). Temperature is the main "state" variable.
    'wind_trans24',       # Rank #8-ish. Captures the "loading" effect better than raw wind.
    'RH',                 # Rank #13 (r=+0.360). Proxy for storm presence.

    # --- 3. The "Stability" Group (Snowpack Structure) ---
    # We chose min_ccl_pen over Sn because your EDA showed it has higher correlation
    # (r=-0.374 vs r=-0.292).
    'min_ccl_pen',        

    # --- 4. The "Engineered" Context ---
    'relative_load_3d',   # You proved this had strong correlation (+0.410) in feature engineering.
    
    # --- 5. Raw Wind (Required for CNN Spatial Learning) ---
    # Even though raw 'VW' had lower correlation, the CNN NEEDS vectors to learn 
    # "Lee Slope Loading" patterns spatially.
    # Note: The script adds 'wind_u' and 'wind_v' automatically from VW/DW.
    'VW', 'DW' 
]

# ==========================================
# 2. SETUP
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
print("Loading data...")
df = pd.read_parquet(CLEANED_DATA_PATH)
grid_meta = np.load(GRID_META_PATH)

# Prepare Target Grid (Mesh)
# We need a 2D meshgrid of (lat, lon) for scipy.interpolate
target_lats = grid_meta['lats']
target_lons = grid_meta['lons']
grid_lat_mesh, grid_lon_mesh = np.meshgrid(target_lats, target_lons, indexing='ij')

print(f"Target Grid Shape: {grid_lat_mesh.shape}")

# Ensure wind vectors exist (if not in dataframe)
if 'wind_u' not in df.columns:
    print("Calculating wind vectors (u/v)...")
    # Convert DW to radians (0 degrees = North, 90 = East)
    # Mathematical standard: 0 = East. Meteorological: 0 = North (blowing FROM).
    # Standard conversion: u = -ws * sin(wd), v = -ws * cos(wd)
    wd_rad = np.deg2rad(df['DW'])
    df['wind_u'] = -df['VW'] * np.sin(wd_rad)
    df['wind_v'] = -df['VW'] * np.cos(wd_rad)
    DYNAMIC_FEATURES.extend(['wind_u', 'wind_v'])

# ==========================================
# 3. INTERPOLATION FUNCTION
# ==========================================
def interpolate_day(day_df, grid_lat, grid_lon, features):
    """
    Interpolates all features for a single day onto the grid.
    Returns a tensor of shape (Height, Width, Channels)
    """
    # Station coordinates for this day
    points = day_df[['lat', 'lon']].values
    
    day_grids = []
    
    for feat in features:
        values = day_df[feat].values
        
        # 1. Linear Interpolation (High Quality, but leaves NaNs at edges)
        grid_linear = griddata(points, values, (grid_lat, grid_lon), method='linear')
        
        # 2. Nearest Neighbor (Fill edges/extrapolate)
        # Find where Linear failed (NaNs)
        mask_nan = np.isnan(grid_linear)
        
        if mask_nan.any():
            # Calculate NN only for the missing pixels
            grid_nearest = griddata(points, values, (grid_lat[mask_nan], grid_lon[mask_nan]), method='nearest')
            grid_linear[mask_nan] = grid_nearest
            
        day_grids.append(grid_linear)
        
    # Stack into 3D volume
    return np.stack(day_grids, axis=-1)

# ==========================================
# 4. MAIN LOOP
# ==========================================
# Group by date
daily_groups = df.groupby('datum')

print(f"Starting interpolation for {len(daily_groups)} days...")
print(f"Features: {DYNAMIC_FEATURES}")

error_count = 0

for date, group in tqdm(daily_groups):
    try:
        # Skip days with too few stations (cannot interpolate safely)
        if len(group) < 10:
            continue
            
        # Format filename
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        save_path = os.path.join(OUTPUT_DIR, f"{date_str}.npz")
        
        # Check if already exists (to allow resuming)
        if os.path.exists(save_path):
            continue
            
        # Run Interpolation
        # Note: Passing raw values. Normalization happens in Data Loader!
        daily_tensor = interpolate_day(group, grid_lat_mesh, grid_lon_mesh, DYNAMIC_FEATURES)
        
        # Save compressed (float32 is enough for ML)
        np.savez_compressed(save_path, data=daily_tensor.astype(np.float32))
        
    except Exception as e:
        print(f"Failed on {date}: {e}")
        error_count += 1

print(f"\nProcessing complete.")
print(f"Errors: {error_count}")
print(f"Output directory: {OUTPUT_DIR}")