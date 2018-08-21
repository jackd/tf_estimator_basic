#!/usr/bin/python

"""Train and evaluate script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def train_and_eval(model_id, dt):
    import tensorflow as tf
    if model_id == 'simple':
        import tf_estimator_basic.simple as model
    elif model_id == 'intermediate':
        import tf_estimator_basic.intermediate as model

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.estimator.RunConfig(save_checkpoints_secs=dt)

    estimator = model.get_estimator(config=config)
    train_spec = tf.estimator.TrainSpec(
        lambda: model.get_inputs('train'),
        max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(
        lambda: model.get_inputs('eval'),
        steps=100,
        start_delay_secs=dt,
        throttle_secs=dt
    )
    return tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', default='simple', nargs='?',
    choices=('simple', 'intermediate'))
parser.add_argument(
    '--dt', '-t', default=100, type=int,
    help='time in secs between save/evaluation',
    nargs='?')
args = parser.parse_args()

train_and_eval(args.model_id, args.dt)
