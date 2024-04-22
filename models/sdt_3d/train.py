import os
import glob
import json
import logging
import math
import numpy as np
import random
import torch
import zarr
import gunpowder as gp

from model import Model, WeightedMSELoss
from utils import ComputeDT

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

#def init_weights(m):
#    if isinstance(m, (torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
#        torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')


#samples = glob.glob(os.path.join(setup_dir,"../../data/training/*.zarr"))
#samples = samples[:2]
#print(samples)
sample = "../../data/ob4_4.zarr"

def train(iterations):

    batch_size = 1
    
    model = Model()
    model.train()
#    model.apply(init_weights)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.9, 0.999))

    raw = gp.ArrayKey("RAW")
    gt_dt = gp.ArrayKey("GT_MASK")
    gt_mask = gp.ArrayKey("UNLABELLED_MASK")
    pred_dt = gp.ArrayKey("PRED_MASK")
    weights = gp.ArrayKey("MASK_WEIGHTS")
    
    with open(os.path.join(setup_dir, "config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "config.json")
        )
        net_config = json.load(f)

    shape_increase = [0, 0,0] #net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    print(input_shape) 
    if 'output_shape' not in net_config:
        output_shape = model.forward(
                raw=torch.empty(size=[1,1]+input_shape),
            )[0].shape[1:]
        print(output_shape)
        net_config['output_shape'] = list(output_shape)
        with open(os.path.join(setup_dir,"config.json"),"w") as f:
            json.dump(net_config,f,indent=4)
    else: 
        output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]

    voxel_size = gp.Coordinate([50,4,4])
    input_size = gp.Coordinate((input_shape)) * voxel_size
    output_size = gp.Coordinate((output_shape)) * voxel_size
    padding = (input_size - output_size) // 2
    sigma = 120
    
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_mask, output_size)
    request.add(gt_dt, output_size)
    request.add(pred_dt, output_size)
    request.add(weights, output_size)
    print(input_size, output_size)

    # construct pipeline
    source = gp.ZarrSource(
            sample,
            {raw: "volumes/raw/s1", gt_mask: "volumes/labels/s1"},
            {raw: gp.ArraySpec(interpolatable=True), gt_mask: gp.ArraySpec(interpolatable=False)}
        )
    source +=  gp.Pad(raw, None, mode="reflect")
    source +=  gp.Normalize(raw)
    source +=  gp.RandomLocation()
    source +=  gp.Reject(mask=gt_mask, min_masked=0.01, reject_probability=0.9999)
    
    pipeline = source #+ gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.ElasticAugment(
        control_point_spacing=[4, 50, 50],
        jitter_sigma=[0, 0, 0],
        rotation_interval=(0, math.pi / 2),
        subsample=4,
        spatial_dims=3,
    )

    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    
    pipeline += gp.DefectAugment(raw, prob_missing=0.0)
    
    pipeline += gp.GrowBoundary(gt_mask, steps=4, only_xy=True)

    pipeline += gp.BalanceLabels(gt_mask, weights)

    pipeline += ComputeDT(gt_mask, gt_dt, mode="2d")

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=20, cache_size=50)
    '''
    pipeline += gp.Squeeze([raw]) #NEW
############################ for aug fig ###################### 
    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_dt: "gt_dt",
            weights: "weights",
            #pred_dt: "pred_dt",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir,'snapshots'),
        every=1000,
    )
#################################################################   
    pipeline += gp.Unsqueeze([raw]) #NEW
    '''
    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            "raw": raw,
        },
        loss_inputs={0: pred_dt, 1: gt_dt, 2: weights},
        outputs={0: pred_dt},
        save_every=1000,
        log_dir=os.path.join(setup_dir,'log'),
        checkpoint_basename=os.path.join(setup_dir,'model'),
    )

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Squeeze([raw, pred_dt])     
    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_dt: "gt_dt",
            weights: "weights",
            pred_dt: "pred_dt",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir,'snapshots'),
        every=1000,
    )


    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    train(500001)
