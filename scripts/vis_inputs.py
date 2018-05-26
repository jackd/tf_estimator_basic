#!/usr/bin/python
"""Script for visualizing inputs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from tf_estimator_basic.data import get_inputs


image, labels = get_inputs('train', 64)

with tf.train.MonitoredSession() as sess:
    while not sess.should_stop():
        ims, labs = sess.run((image, labels))
        for im, lab in zip(ims, labs):
            plt.imshow(im[:, :, 0], cmap='gray')
            plt.title(str(lab))
            plt.show()
