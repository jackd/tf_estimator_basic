#!/usr/bin/python
"""Convenience script for deleting model directories."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def clean(model_dir):
    import os
    import shutil
    try:
        inp = raw_input
    except NameError:
        inp = input
    if not os.path.isdir(model_dir):
        print('No directory at %s' % model_dir)
    else:
        i = inp('Confirm: delete directory %s ? (y/n) ' % model_dir)
        if i.lower() in {'y', 'yes'}:
            shutil.rmtree(model_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    'model_dir', default='/tmp/tf_estimator_basic', nargs='?')
args = parser.parse_args()

clean(args.model_dir)
