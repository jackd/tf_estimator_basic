#!/usr/bin/python
"""Training script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def train(model_id):
    import tensorflow as tf
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    max_steps = 10000
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = model.get_estimator()
    estimator.train(
        lambda: model.get_inputs('train'), max_steps=max_steps)


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
args = parser.parse_args()

train(args.model_id)
