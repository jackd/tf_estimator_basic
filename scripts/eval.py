#!/usr/bin/python
"""Evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def evaluate(model_id):
    import tensorflow as tf
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = model.get_estimator()
    estimator.evaluate(lambda: model.get_inputs('eval'))


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
args = parser.parse_args()

evaluate(args.model_id)
