import time
import logging
import sys
import yaml
import os
import numpy as np
from scipy.ndimage import label
from funlib.persistence import open_ds
from funlib.evaluate import detection_scores, rand_voi


def evaluate(config):

    raw_file = config["raw_file"]
    labels_dataset = config["labels_dataset"]
    mask_file = config["mask_file"]
    mask_dataset = config["mask_dataset"]

    truth = open_ds(raw_file, labels_dataset)
    test = open_ds(mask_file, mask_dataset)
    roi = truth.roi.intersect(test.roi)

    print("evaluating")
    truth_data, n_truth = label(truth.to_ndarray(roi))
    test_data, n_test = label(test.to_ndarray(roi))
    print("number of truth components", n_truth)
    print("number of test components", n_test)
    truth_data = truth_data.astype(np.uint64)
    test_data = test_data.astype(np.uint64)

    scores = detection_scores(truth_data, test_data)
    return config | scores 


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["evaluation"]

    start = time.time()
    config_with_metrics = evaluate(config)
    end = time.time()

    print(config_with_metrics)

    seconds = end - start
    logging.info(f'Total time to eval: {seconds} ')
