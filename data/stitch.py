import sys
import os
import zarr
import numpy as np
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Roi, Coordinate

chunks_dir = "MVB_testing"
out_zarr = "ob4_5.zarr"
#chunks_dir = 'MVB_training'
#out_zarr = 'ob4_4.zarr'
voxel_size = Coordinate([50, 2, 2])

# Number of tiles in each dimension
num_tiles_x = 10
num_tiles_y = 10

# Initialize total shape with zeros
total_shape = (0, 0)
total_shape = (0, 0)

# **Calculate tile shapes for accurate offsets**
tile_shapes = []
tiles = {}

# Iterate over the zarr volumes to get tile shapes
for i in range(num_tiles_x):
    for j in range(num_tiles_y):

        tile_id = f"{i}{j}"

        volume_name = f"../{chunks_dir}/{i}_{j}.zarr"

        if tile_id not in tiles:
            tiles[tile_id] = zarr.open(volume_name, mode='r')

        volume = tiles[tile_id]
        tile_shapes.append(volume['volumes/raw'].shape[1:])
        
        # Update the total shape
        num_sections = volume['volumes/raw'].shape[0]


assert len(tile_shapes) == 100
shapes = tile_shapes[::-1] # reverse order to go 99, 98, 97, ... 90, 89, ... 1, 0

# Create the new zarr volume
total_shape = (
        num_sections, 
        sum([x[0] for x in shapes[:10]]),
        sum([x[1] for x in shapes[::10]])+1 
)
print(total_shape)

total_roi = Roi((0, 0, 0), Coordinate(total_shape)*voxel_size)

f = zarr.open(out_zarr,"w")
raw = f.create_dataset(f"volumes/raw",shape=total_shape,chunks=(8,256,256),compressor=zarr.get_codec({'id':"blosc"}),dtype=np.uint8)
labels = f.create_dataset(f"volumes/labels",shape=total_shape,chunks=(8,256,256),compressor=zarr.get_codec({'id':"blosc"}),dtype=np.uint8)  

# Loop through the arrays and place them in the stitched array
row_start = 0
col_start = total_shape[2] - shapes[0][1]
for i in range(100):
    row = i % 10
    col = i // 10
    tile_id = f"{9 - col}{9 - row}"

    tile = tiles[tile_id]
    shape = tile["volumes/raw"].shape[1:]

    roi_start = Coordinate(0, row_start, col_start)
    roi_shape = Coordinate(num_sections, shape[0], shape[1])
    #roi = Roi(roi_start * voxel_size, roi_shape * voxel_size)
    roi = Roi(roi_start, roi_shape)
    print(i, row, col, tile_id, shape, row_start, col_start)

    raw[roi.to_slices()] = tile["volumes/raw"][:]
    labels[roi.to_slices()] = tile["volumes/labels"][:,:roi_shape[1],:roi_shape[2]]

    row_start += shape[0]
    if (i + 1) % 10 == 0:
        col_start -= shape[1] #cum_heights[row]
        row_start = 0

raw.attrs["offset"] = [0,0,0]
raw.attrs["resolution"] = [50,2,2]
labels.attrs["offset"] = [0,0,0]
labels.attrs["resolution"] = [50,2,2]
