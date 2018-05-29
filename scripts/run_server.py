#!/usr/bin/python
"""Example usage of model without the use of the estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def run_server_example(model_id):
    import numpy as np
    from tf_estimator_basic.dummy_server import DummyServer
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    with DummyServer(model.get_logits, model.model_dir) as server:
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
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
args = parser.parse_args()

run_server_example(args.model_id)
