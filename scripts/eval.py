#!/usr/bin/python
"""Evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def evaluate(model_dir, batch_size):
    import tensorflow as tf
    from tf_estimator_basic.model import get_estimator_spec
    from tf_estimator_basic.data import get_inputs
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.estimator.Estimator(get_estimator_spec, model_dir)
    estimator.evaluate(lambda: get_inputs('eval', batch_size))


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--model_dir', default='/tmp/tf_estimator_basic')
parser.add_argument('-b', '--batch_size', type=int, default=64)
args = parser.parse_args()

evaluate(args.model_dir, args.batch_size)
