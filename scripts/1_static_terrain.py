import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr
import xrspatial
import matplotlib.pyplot as plt
import os

def normalize(arr):
    """Normalize array to zero mean and unit variance."""
    
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / (std + 1e-6), mean, std

def generate_static_terrain():
    """Generates static terrain features (Elevation, Slope, Aspect) on a 1km x 1km grid over Switzerland."""
    
    # 1. Configuration
    input_DEM = '../data/swiss_dem.tif'  
    output = '../data/grids/static_terrain.npy'
    metadata = '../data/grids/grid_metadata.npz' 

    # Bounds
    lat_min, lat_max = 45.8, 47.9
    lon_min, lon_max = 5.9, 10.6

    # Resolution: ~1km in degrees 
    resolution = 0.01 # 1 degree lat ~= 111km -> 0.01 deg ~= 1.1km

    # 2. Target Grid
    target_lats = np.arange(lat_min, lat_max, resolution)
    target_lons = np.arange(lon_min, lon_max, resolution)

    height = len(target_lats)
    width = len(target_lons)

    print(f"Target Grid Shape: ({height}, {width}) -> {height*width} pixels")

    # Define the affine transform for the target grid
    target_transform = rasterio.transform.from_bounds(
        west=lon_min, south=lat_min, east=lon_max, north=lat_max, 
        width=width, height=height
    )

    # 3. Resampling to 1kmx1km Grid
    print("Resampling DEM to 1km grid...")
    with rasterio.open(input_DEM) as src:
        dem_1km = np.zeros((height, width), dtype=np.float32)
        # Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_1km,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear 
        )

    dem_1km[dem_1km < -100] = 0 

    # 4. Slope & Aspect Calculation
    print("Calculating Slope and Aspect...")
    da_dem = xr.DataArray(dem_1km, coords=[target_lats, target_lons], dims=["lat", "lon"])
    slope = xrspatial.slope(da_dem).to_numpy()
    aspect = xrspatial.aspect(da_dem).to_numpy()

    # Handle edges/NaNs 
    slope = np.nan_to_num(slope, nan=0.0)
    aspect = np.nan_to_num(aspect, nan=0.0)

    # 5. Normalization
    print("Normalizing features...")
    dem_norm, dem_mean, dem_std = normalize(dem_1km)
    slope_norm, slope_mean, slope_std = normalize(slope)

    # sin/cos encoding for aspect
    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    # 6. Stack & Save
    # Final Tensor Shape: (Height, Width, Channels) with Channels: [Elevation, Slope, Aspect_Sin, Aspect_Cos]
    static_tensor = np.stack([dem_norm, slope_norm, aspect_sin, aspect_cos], axis=-1)

    os.makedirs('../data/grids', exist_ok=True)
    np.savez(metadata, 
             lats=target_lats, 
             lons=target_lons,
             dem_stats=(dem_mean, dem_std),
             slope_stats=(slope_mean, slope_std))

    static_flipped = np.flipud(static_tensor)
    np.save(output, static_flipped)
    
    print(f"Static terrain saved to {output}")
    print(f"Grid metadata saved to {metadata}")
    
    return static_flipped

if __name__ == "__main__":
    generate_static_terrain()