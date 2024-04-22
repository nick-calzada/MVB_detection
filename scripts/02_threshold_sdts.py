import multiprocessing
multiprocessing.set_start_method('fork')

import daisy
import subprocess
import json
import yaml
import logging
import numpy as np
import os
import pymongo
import sys
import time

from pathlib import Path
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds

logging.basicConfig(level=logging.INFO)

scripts_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def threshold_blockwise(
        config: dict) -> bool:
    '''
    Extract masks in parallel blocks. Requires that signed distance transforms
    (SDTs) have been predicted before.

    We assume there is a workers directory inside the current directory that
    contains worker scripts (e.g `workers/threshold_worker.py`).


    Args:
        pred_file (``string``):

            Path to file (zarr/n5) where predictions are stored.

        pred_dataset (``string``):

            Predictions dataset to use (e.g 'volumes/preds'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/preds/s0').

        out_file (``string``):

            Path to file (zarr/n5) to store masks (supervoxels) - generally
            a good idea to store in the same place as preds.

        out_dataset (``string``):

            Name of dataset to write masks (supervoxels) to (e.g
            'volumes/masks').

        num_workers (``int``):
        
            How many blocks to run in parallel. Default is 10.

        threshold_in_xy (``bool``):

            Whether to extract masks for each xy-section separately.
            Default is False (3D).

    '''

    print(config)

    # Extract parameters from the config dictionary
    pred_file = config['pred_file']
    pred_dataset = config['pred_dataset']
    out_file = config['out_file']
    out_dataset = config['out_dataset']
    num_workers = config['num_workers']
    threshold = config['threshold']

    logging.info(f"Reading preds from {pred_file}")
    preds = open_ds(pred_file, pred_dataset)
    voxel_size = preds.voxel_size

    # ROI
    if 'block_size' in config and config['block_size'] is not None:
        block_size = Coordinate(config["block_size"])
    else:
        block_size = Coordinate(preds.chunk_shape[1:]) * voxel_size

    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = preds.roi

    read_roi = Roi((0,)*preds.roi.dims, block_size)
    write_roi = Roi((0,)*preds.roi.dims, block_size)

    # Prepare masks dataset
    out_masks = prepare_ds(
        filename=out_file,
        ds_name=out_dataset,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        write_size=write_roi.shape,
        force_exact_write_size=False,
        compressor={"id": "blosc", "clevel": 5},
        delete=True,
    )

    # blockwise threshold
    task = daisy.Task(
        task_id="ThresholdPredictionsTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda b: threshold_in_block(
                b,
                preds,
                out_masks,
                threshold=threshold),
        num_workers=num_workers,
        timeout=10,
        max_retries=20,
        read_write_conflict=True,
        fit='shrink')
    
    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("At least one block failed!")
    
    return done


def threshold_predictions(
        preds,
        threshold=None):

    if threshold is None:
        threshold = 0.0

    return (preds > threshold).astype(np.uint8)[0]


def threshold_in_block(
        block,
        preds,
        out_dataset,
        threshold=None):

    total_roi = preds.roi

    logging.debug("reading preds from %s", block.read_roi)

    preds = preds.intersect(block.read_roi)
    preds.materialize()

    # extract masks
    out_masks_data = threshold_predictions(
            preds.data,
            threshold
    )
    # store masks
    logging.debug("writing masks to %s", block.write_roi)
    out_dataset[block.write_roi] = out_masks_data


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["threshold"]

    start = time.time()
    threshold_blockwise(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to extract masks: {seconds} ')
