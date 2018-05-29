#!/usr/bin/python
"""Convenience script for deleting model directories."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def clean(model_id):
    import os
    import shutil
    if model_id == 'simple':
        from tf_estimator_basic.simple import model_dir
    elif model_id == 'intermediate':
        from tf_estimator_basic.intermediate import model_dir
    else:
        raise ValueError('model_id must be in {"simple", "intermediate"}')

    if not os.path.isdir(model_dir):
        print('No directory at %s' % model_dir)
    else:
        shutil.rmtree(model_dir)
        print('Removed directory %s' % model_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_id', choices=('simple', 'intermediate'))
args = parser.parse_args()

clean(args.model_id)
