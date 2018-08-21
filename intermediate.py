"""Like `simple.py`, but with slightly more complicated variants."""

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


def get_inputs(
        mode, batch_size=64, repeat=None, shuffle=None,
        data_dir='/tmp/mnist_data', corruption_stddev=5e-2):
    """
    Get optionally corrupted MNIST batches.

    Args:
        mode: `'train'` or in `{'eval', 'predict', 'infer'}`
        batch_size: size of returned batches
        repeat: bool indicating whether or not to repeat indefinitely. If None,
            repeats if `mode` is `'train'`
        shuffle: bool indicating whether or not to shuffle each epoch. If None,
            shuffles if `mode` is `'train'`
        data_dir: where to load/download data to
        corruption_stddev: if training, normally distributed noise is added
            to each pixel of the image.

    Returns:
        `image`, `labels` tensors, shape (?, 28, 28, 1) and (?) respecitvely.
        First dimension is batch_size except possibly on final batches.
    """
    # get the original dataset from `tensorflow/models/official`
    # https://github.com/tensorflow/models
    if mode == 'train':
        dataset = ds.train(data_dir)
    elif mode in {'eval', 'predict', 'infer'}:
        dataset = ds.test(data_dir)
    else:
        raise ValueError('mode "%s" not recognized' % mode)

    training = mode == 'train'

    # repeat before training is better for performance, though possibly worse
    # around epoch boundaries
    if repeat or repeat is None and training:
        dataset = dataset.repeat()

    if shuffle or shuffle is None and training:
        # A larger buffer size requires more memory but gives better shufffling
        dataset = dataset.shuffle(buffer_size=10000)

    def map_fn(image, labels):
        image += tf.random_normal(
            shape=image.shape, dtype=tf.float32, stddev=corruption_stddev)
        return image, labels

    # num_parallel_calls defaults to None, but included here to draw attention
    # for datasets with more preprocessing this may significantly speed things
    # up
    if training:
        dataset = dataset.map(map_fn, num_parallel_calls=None)
    dataset = dataset.batch(batch_size)

    # prefetching allows the CPU to preprocess/load data while the GPU is busy
    # prefetch_to_device should be faster, but likely won't make a difference
    # at this scale.
    dataset = dataset.prefetch(1)
    # dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    image, labels = dataset.make_one_shot_iterator().get_next()
    image = tf.reshape(image, (-1, 28, 28, 1))  # could also go in map_fn
    return image, labels


def get_logits(image, mode='infer'):
    """Get logits from image."""
    training = mode == tf.estimator.ModeKeys.TRAIN
    x = image
    for filters in (32, 64):
        x = tf.layers.conv2d(x, filters, 3)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2)
    x = tf.reduce_mean(x, axis=(1, 2))
    logits = tf.layers.dense(x, 10)
    return logits


def get_estimator_spec(features, labels, mode):
    """For use with `tf.estimator.Estimator`."""
    logits = get_logits(features, mode)
    probs = tf.nn.softmax(logits)
    preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    # we include image in predictions for demonstration purposes.
    # see scripts/vis_predictions.py
    predictions = dict(probs=probs, preds=preds, image=features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Create some summaries for visualizing on tensorboard.
    tf.summary.image('input', features)
    tf.summary.histogram('max_prob', tf.reduce_max(probs, axis=-1))
    n_correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32))
    batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
    batch_accuracy = n_correct / batch_size
    tf.summary.scalar('batch_accuracy', batch_accuracy)

    accuracy = tf.metrics.accuracy(labels, preds)
    eval_metric_ops = dict(accuracy=accuracy)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    step = tf.train.get_or_create_global_step()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=step)

    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss,
        eval_metric_ops=eval_metric_ops, train_op=train_op)


model_dir = '/tmp/mnist_intermediate'


def get_estimator(config=None):
    return tf.estimator.Estimator(
        get_estimator_spec, model_dir, config=config)
