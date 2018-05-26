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
        mode, batch_size, repeat=None, shuffle=None, data_dir='/tmp/mnist',
        corruption_stddev=None):
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
        corruption_stddev: if not None, normally distributed noise is added
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
        if corruption_stddev is not None:
            image += tf.random_normal(
                shape=image.shape, dtype=tf.float32, stddev=corruption_stddev)
        return image, labels

    # num_parallel_calls defaults to None, but included here to draw attention
    # for datasets with more preprocessing this may significantly speed things
    # up
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
