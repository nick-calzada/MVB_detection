from tqdm import tqdm
import zarr
import numpy as np
from scipy.ndimage import binary_erosion
import sys

fn = sys.argv[1]
ds = sys.argv[2]

f = zarr.open(fn,"a")

arr = f[ds][:]


eroded = np.zeros_like(arr)


for z in tqdm(range(arr.shape[0])):
    eroded[z] = binary_erosion(arr[z], iterations=7)


#eroded = binary_erosion(arr)
#assert eroded.dtype == arr.dtype
#print("before", len(np.unique(arr)))
#print("after", len(np.unique(filtered)))

f[f"eroded_{ds}"] = eroded
f[f"eroded_{ds}"].attrs["offset"] = f[ds].attrs["offset"]
f[f"eroded_{ds}"].attrs["resolution"] = f[ds].attrs["resolution"]
