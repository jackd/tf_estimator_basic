#!/usr/bin/python
"""Training script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def train(model_dir, max_steps, batch_size, corruption_stddev):
    import tensorflow as tf
    from tf_estimator_basic.model import get_estimator_spec
    from tf_estimator_basic.data import get_inputs
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.estimator.Estimator(get_estimator_spec, model_dir)
    estimator.train(
        lambda: get_inputs(
            'train', batch_size, corruption_stddev=corruption_stddev),
        max_steps=max_steps)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--model_dir', default='/tmp/tf_estimator_basic')
parser.add_argument('-s', '--max_steps', type=int, default=10000)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-c', '--corruption_stddev', type=float, default=0.05)
args = parser.parse_args()

train(args.model_dir, args.max_steps, args.batch_size, args.corruption_stddev)
