"""
Provides example class for using a model without an estimator.

See `scripts/run_server.py`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DummyServer(object):
    def __init__(self, logits_fn, model_dir):
        self.logits_fn = logits_fn
        self.model_dir = model_dir
        self._sess = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        print('Opening DummyServer...')
        print('  building graph...')
        graph = tf.Graph()
        with graph.as_default():
            self._image = tf.placeholder(
                shape=(28, 28), dtype=tf.uint8)
            image = tf.expand_dims(self._image, axis=0)
            image = tf.expand_dims(image, axis=-1)
            image = tf.cast(image, tf.float32) / 255

            logits = self.logits_fn(image)
            self._probs = tf.nn.softmax(logits, axis=-1)
            saver = tf.train.Saver()

        print('  opening session...')
        self._sess = tf.Session(graph=graph)
        print('  restoring variables...')
        saver.restore(self._sess, tf.train.latest_checkpoint(self.model_dir))
        print('Done!')

    def close(self):
        print('Closing server...')
        self._sess.close()
        print('Goodbye!')

    def serve(self, image):
        if self._sess is None:
            raise RuntimeError('Must open server first!')
        probs = self._sess.run(self._probs, feed_dict={self._image: image})
        return probs
