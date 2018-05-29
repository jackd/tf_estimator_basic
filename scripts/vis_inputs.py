#!/usr/bin/python
"""Script for visualizing inputs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def vis_inputs(model_id):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    image, labels = model.get_inputs('train')

    with tf.train.MonitoredSession() as sess:
        while not sess.should_stop():
            ims, labs = sess.run((image, labels))
            for im, lab in zip(ims, labs):
                plt.imshow(im[:, :, 0], cmap='gray')
                plt.title(str(lab))
                plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
args = parser.parse_args()
vis_inputs(args.model_id)
