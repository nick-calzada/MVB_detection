import glob
from tifffile import imread
import numpy as np
import zarr
import sys
import os

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def return_image_stack(images):

    h, w = imread(images[0]).shape
    image = np.zeros((len(images),h,w), dtype=np.uint8)
    
    for i, img in enumerate(images):
        image[i] = imread(images[i])

    return image


def return_labels(labels_tif):
    return imread(labels_tif)


if __name__ == "__main__":

    for split in ["testing"]: #,"testing"]:
        for i in range(10): # X chunk
            for j in range(10): # Y chunk

                print(split,i,j)
                ims = sorted(glob.glob(f"../MVB_{split}/OB4-5_cropped_images/{i},{j}/*.tif"))
                im = return_image_stack(ims) 
                masks = return_labels(f"../MVB_{split}/OB4-5_MVB_labeled_wy_check/{i}-{j}.tif")

                # normalize
                masks = (masks > 0).astype(np.uint8)

                # write zarr
                f = zarr.open(f"../MVB_{split}/{i}_{j}.zarr","a")
                
                raw = f.create_dataset(f"volumes/raw",data=im,chunks=(8,256,256),compressor=zarr.get_codec({'id':"blosc"}),dtype=np.uint8)
                raw.attrs["offset"] = [0,0,0]
                raw.attrs["resolution"] = [50,2,2]
                
                labels = f.create_dataset(f"volumes/labels",data=masks,chunks=(8,256,256),compressor=zarr.get_codec({'id':"blosc"}),dtype=np.uint8)  
                labels.attrs["offset"] = [0,0,0]
                labels.attrs["resolution"] = [50,2,2]
