from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_logits(image, mode):
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
    preds = tf.argmax(logits, axis=-1)

    # we include image in predictions for demonstration purposes.
    # see scripts/vis_predictions.py
    predictions = dict(probs=probs, preds=preds, image=features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Create some summaries for visualizing on tensorboard.
    tf.summary.image('input', features)
    tf.summary.histogram('max_prob', tf.reduce_max(probs, axis=-1))
    n_correct = tf.cast(tf.reduce_sum(tf.equal(preds, labels)), tf.float32)
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
