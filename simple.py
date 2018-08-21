"""Functions required for constructing a super simple estimator and inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
try:
    import official.mnist.dataset as ds
except ImportError:
    print(
        'No `official` found on your path.\n'
        'Please clone tensorflow/models and add to your `PYTHONPATH`, e.g.\n'
        'cd ~\n'
        'git clone https://github.com/tensorflow/models.git\n'
        'export PYTHONPATH=$PYTHONPATH:~/models')
    raise


def get_inputs(mode, batch_size=64):
    """
    Get batched (features, labels) from mnist.

    Args:
        `mode`: string representing mode of inputs.
            Should be one of {"train", "eval", "predict", "infer"}

    Returns:
        `features`: float32 tensor of shape (batch_size, 28, 28, 1) with
            grayscale values between 0 and 1.
        `labels`: int32 tensor of shape (batch_size,) with labels indicating
            the digit shown in `features`.
    """
    # Get the base dataset
    if mode == 'train':
        dataset = ds.train('/tmp/mnist_data')
    elif mode in {'eval', 'predict', 'infer'}:
        dataset = ds.test('/tmp/mnist_data')
    else:
        raise ValueError(
            'mode must be one of {"train", "eval", "predict", "infer"}')

    # repeat and shuffle if training
    if mode == 'train':
        dataset = dataset.repeat()  # repeat indefinitely
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)

    image, labels = dataset.make_one_shot_iterator().get_next()
    image = tf.reshape(image, (-1, 28, 28, 1))
    return image, labels


def get_logits(image):
    """Get logits from image."""
    x = image
    for filters in (32, 64):
        x = tf.layers.conv2d(x, filters, 3)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2)
    x = tf.reduce_mean(x, axis=(1, 2))
    logits = tf.layers.dense(x, 10)
    return logits


def get_estimator_spec(features, labels, mode):
    """
    Get an estimator specification.

    Args:
      features: mnist image batch, flaot32 tensor of shape
          (batch_size, 28, 28, 1)
      labels: mnist label batch, int32 tensor of shape (batch_size,)
      mode: one of `tf.estimator.ModeKeys`, i.e. {"train", "infer", "predict"}

    Returns:
      tf.estimator.EstimatorSpec
    """
    if mode not in {"train", "infer", "eval"}:
        raise ValueError('mode should be in {"train", "infer", "eval"}')

    logits = get_logits(features)
    preds = tf.argmax(logits, axis=-1)
    probs = tf.nn.softmax(logits, axis=-1)
    predictions = dict(preds=preds, probs=probs, image=features)

    if mode == 'infer':
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=step)

    accuracy = tf.metrics.accuracy(labels, preds)

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions,
        loss=loss, train_op=train_op, eval_metric_ops=dict(accuracy=accuracy))


model_dir = '/tmp/mnist_simple'


def get_estimator(config=None):
    return tf.estimator.Estimator(
        get_estimator_spec, model_dir, config=config)
