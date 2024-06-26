import json
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr
import daisy
from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds

from model import Model

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def predict(config):
    iteration = config["iteration"]
    raw_file = config["raw_file"]
    raw_dataset = config["raw_datasets"] # [stacked_lsds_x, stacked_affs_x]
    out_file = config["out_file"]
    out_dataset_names = config["out_dataset_names"]
    num_cache_workers = config["num_cache_workers"]
    out_dataset = out_dataset_names[0]

    # load net config
    with open(os.path.join(setup_dir, "config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "config.json")
        )
        net_config = json.load(f)

    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]
   
    voxel_size = Coordinate(zarr.open(raw_file,"r")[raw_dataset[0]].attrs["resolution"])
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2
    
    model = Model()
    model.eval()

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset[0],
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
            })

    predict = gp.torch.Predict(
            model,
            checkpoint=os.path.join(setup_dir,f'model_checkpoint_{iteration}'),
            inputs = {
                'raw': raw,
            },
            outputs = {
                0: pred,
            })

    scan = gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                pred: 'write_roi'
            },
            num_workers=num_cache_workers)

    write = gp.ZarrWrite(
            dataset_names={
                pred: out_dataset,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None, mode="reflect") +
            gp.IntensityScaleShift(raw, 2, -1) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred]) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(run_config)

