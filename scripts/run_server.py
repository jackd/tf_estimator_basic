#!/usr/bin/python
"""Example usage of model without the use of the estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def run_server_example(model_dir):
    import numpy as np
    from estimator.dummy_server import DummyServer
    with DummyServer(model_dir) as server:
        image = np.zeros(shape=(28, 28), dtype=np.uint8)
        image[3:22, 12:16] = 255  # bad `1`
        probs = server.serve(image)
        print(probs)
        # basic blurring
        image[3:22, 16:17] = 128
        image[3:22, 11:12] = 128
        probs = server.serve(image)
        print(probs)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--model_dir', default='/tmp/tf_estimator_basic')
parser.add_argument('-b', '--batch_size', type=int, default=64)
args = parser.parse_args()

run_server_example(args.model_dir)
