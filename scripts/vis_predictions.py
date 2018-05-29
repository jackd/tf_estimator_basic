#!/usr/bin/python
"""Prediction script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def predict(model_id):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = model.get_estimator()
    predictions = estimator.predict(
        lambda: model.get_inputs('infer'))
    for preds in predictions:
        print(preds['probs'])
        plt.imshow(preds['image'][:, :, 0], cmap='gray')
        plt.title(preds['preds'])
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
args = parser.parse_args()
predict(args.model_id)
