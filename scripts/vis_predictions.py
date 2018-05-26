#!/usr/bin/python
"""Prediction script."""
import argparse


def predict(model_dir, batch_size):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tf_estimator_basic.model import get_estimator_spec
    from tf_estimator_basic.data import get_inputs
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.estimator.Estimator(get_estimator_spec, model_dir)
    predictions = estimator.predict(
        lambda: get_inputs('infer', batch_size))
    for preds in predictions:
        print preds['probs']
        plt.imshow(preds['image'][:, :, 0])
        plt.title(preds['preds'], cmap='gray')
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--model_dir', default='/tmp/tf_estimator_basic')
parser.add_argument('-b', '--batch_size', type=int, default=64)
args = parser.parse_args()

predict(args.model_dir, args.batch_size)
