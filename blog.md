# Anatomy of an Estimator: Tensorflow's High Level Session Management API
When tensorflow was released in late 2015, it had a fairly limited API. The hype surrounding its initial release resulted in a large number of tutorials and blog posts explaining how to use that minimal API, and multiple frameworks popped up to minimize boilerplate and make code more readable.

Since then, tensorflow's API has expanded dramatically. Unfortunately, titles like "Google releases higher level API" don't draw quite the same attention as "Google releases deep learning framework", so much of the information available to beginners is still based on the low-level APIs available upon release.

This article is my attempt to partially rectify this situation and bring some attention to tensorflow's native `Estimator` class. While the arguably confusing documentation and general lack of simple extensible examples scares a lot of people away, I've found them incredibly helpful in reducing boilerplate without sacrificing fine-grained control. We'll develop a simple (you guessed it) MNIST convolutional network (CNN) to classify digits. All code is available [here](https://github.com/jackd/tf_estimator_basic).

This article is intended for people with a basic understanding of machine learning, neural networks and some experience with python.

## Estimator Construction
If you're looking to understand a new class, the best place to start is normally the [documentation](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator#__init__).

```
Constructs an Estimator instance.
...

Args:
  model_fn: Model function. Follows the signature:

    * Args:

      * `features`: This is the first item returned from the `input_fn`
             passed to `train`, `evaluate`, and `predict`. This should be a
             single `Tensor` or `dict` of same.
      * `labels`: This is the second item returned from the `input_fn`
             passed to `train`, `evaluate`, and `predict`. This should be a
             single `Tensor` or `dict` of same (for multi-head models). If
             mode is `ModeKeys.PREDICT`, `labels=None` will be passed. If
             the `model_fn`'s signature does not accept `mode`, the
             `model_fn` must still be able to handle `labels=None`.
      * `mode`: Optional. Specifies if this training, evaluation or
             prediction. See `ModeKeys`.
      * `params`: Optional `dict` of hyperparameters.  Will receive what
             is passed to Estimator in `params` parameter. This allows
             to configure Estimators from hyper parameter tuning.
      * `config`: Optional configuration object. Will receive what is passed
             to Estimator in `config` parameter, or the default `config`.
             Allows updating things in your `model_fn` based on
             configuration such as `num_ps_replicas`, or `model_dir`.

    * Returns:
      `EstimatorSpec`

  model_dir: Directory to save model parameters, graph and etc.
  config: ...
```

If you find yourself reading through that a few times only to still not really know where to start, you're not alone. That said, before you type "tensorflow keras" into google, let's see the skeleton of a minimal example.

First, let's inspect a valid `model_fn`.

```python
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
        mode=mode, predictions=dict(preds=preds, probs=probs),
        loss=loss, train_op=train_op, eval_metric_ops=dict(accuracy=accuracy))
```

If you've written a neural network model before, hopefully most of that makes sense. We've hidden the details of the `get_logits` function for the moment, but this can be any neural network architecture that outputs 10-class logits (see the intermediate section below if using `batch_normalization`).

Apart from that we simply define a loss, train operation, evaluation metric and predictions (we include the image in the predictions as a hack for using `Estimator.predict` - see below).

Hopefully the only other unfamiliar concept is this `tf.estimator.EstimatorSpec`. This is simply a named tuple - an immutable container for multiple objects. They are more convenient than tuples because the implementer can provide sensible defaults, and users don't have to remember the order in which arguments are packed.

To create an estimator, we simply call the constructor with this function as the first argument, along with a directory in which the estimator will save variables and log progress.

```python
estimator = tf.estimator.Estimator(get_estimator_spec, '/tmp/mnist_simple')
```

A couple of things to note:
* we don't explicitly call `get_estimator_spec`. Rather, we pass the function itself to the Estimator's `__init__` method, and allow it to worry about when to call it; and
* we haven't yet defined how to construct the `features` and `labels` tensors. We'll do this later, but this is an intentional design choise from the tensorflow authors that forces model writers to separate the data pipeline from the trained model architecture.

### Data Pipeline
Before we start using our estimator, we'll quickly discuss our data pipeline.

We'll be using tensorflow's [tf.data.Dataset](https://www.tensorflow.org/versions/master/api_docs/python/tf/data/Dataset) class for our input pipeline. There are many ways of writing an initial dataset - none of which are particularly fun - so we'll skip over this part and just use [tensorflow/models](https://github.com/tensorflow/models)' official MNIST dataset. See the [quick start guide](https://www.tensorflow.org/get_started/datasets_quickstart) for more details on creating your own custom datasets.

We'll write a basic `get_inputs` function that gets us batched `features` and `labels` for use in our network.

```python
import official.mnist.dataset as ds
# requires tensorflow/models to be on your PYTHONPATH, e.g.
# cd ~/
# git clone https://github.com/tensorflow/models.git
# export PYTHONPATH=$PYTHONPATH:~/models

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
```

### Training, Evaluation and Prediction
Now that we've successfully constructed our estimator, we can train and evaluate our model.

While you can do this in one script, one nice thing about estimators is they'll handle saving and loading variables from disk. This means we can safely train in one script, then evaluate in a separate script. This way if our evaluation script throws an error, we don't have to retrain the model. It also means if we have an issue half-way through training - a power failure, or simply get bored waiting for it to finish and manually kill the program - the training process can pick up from the latest saved checkpoint.

```python
tf.logging.set_verbosity(tf.logging.INFO)
estimator.train(lambda: get_inputs('train'), max_steps=10000)

estimator.evaluate(lambda: get_inputs('eval'))
```

If we start `tensorboard` at the `model_dir` specified in the constructor we can see the loss evolution and the computation graph.
```
tensorboard --logdir=/tmp/mnist_simple
```

Prediction is done slightly differently, with the results of prediction being available as a generator
```python
for prediction in estimator.predictions(lambda: get_inputs('infer')):
    image = prediction['image']
    plt.imshow(image[..., 0])
    plt.title('inferred: %d' % prediction['preds'])
    plt.show()
```

A few notes:
* like the `model_fn` in the constructor, we pass each method a function to generate the features and labels, rather than the features and labels themselves. This allows the estimator to decide on the best context in which to construct the operations; and
* we never manually create a session like when using lower-level API.

This is not to say we cannot fall back to lower-level API calls if we want. For example, we can use the following for predictions based on placeholders.
```python
graph = tf.Graph()
with graph.as_default():
    image = tf.placeholder(
        shape=(28, 28), dtype=tf.uint8)
    image = tf.expand_dims(self._image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    image = tf.cast(image, tf.float32) / 255

    logits = get_logits(image)
    probs = tf.nn.softmax(logits, axis=-1)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('/tmp/mnist_simple'))

    inferred_probs = sess.run(
      probs, feed_dict={image: custom_image_batch_data})
```
See [this example](https://github.com/jackd/tf_estimator_basic/dummy_server.py) for a slightly more realistic example.

## Taking Things Further
Code for this section is available [here](https://github.com/jackd/tf_estimator_basic/intermediate.py).


#### Dataset Augmentation and Optimization
The above `get_inputs` function is fine for MNIST, but we'll investigate 2 possible improvements which could lead to significant improvements in larger problems. Firstly, we can augment our dataset by adding random perturbations to each image in the dataset.

```python
# after repeating/sampling

def map_fn(features, labels):
    features += tf.random_normal(
        stddev=5e-2, shape=tf.shape(features), dtype=tf.float32)
    return features, labels


dataset = dataset.map(map_fn, num_parallel_calls=8)
```
Note the `num_parallel_calls` used here is probably excessive for such a small modification and may even result in slow downs, but for larger manipulations it can have a significant result. For small manipulations you may find it more performant to `map` after `batch`ing, but the above will work in either order.  Note that a value greater than 1 will result in intra-batch shuffling, so if order within the batch is important, stick to 1.

Potentially the biggest improvement to performance you can make it to [prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) the next data batch using CPU cycles while your GPU is busy calculating inferences and updates for the current batch. This can be achieved by either of the following after batching.

```python
dataset = dataset.prefetch(1)
```
```python
dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
```

The second is slightly more restrictive (e.g. cannot fetch datasets with `tf.string` tensors) and being part of `contrib` the API is subject to change - but will likely result in a better improvement over `Dataset.prefetch` since the data is loaded directly onto the GPU.

See the [dataset performance guide](https://www.tensorflow.org/performance/datasets_performance) for more details.

#### Batch Normalization
[Batch normalization](https://www.tensorflow.org/versions/master/api_docs/python/tf/layers/BatchNormalization) is a popular method to improve training speed and generalization of networks. The exact details are beyond the scope of this article, but we will discuss some implementation details you should be aware of.

Firstly, calling `batch_normalization` (or a preconstructed `BatchNormalization` layer) requires a `training` flag. This can be deduced from the `model_fn`'s `mode` argument.

```python
def get_logits(image, mode='infer'):
    """Get logits from image."""
    training = mode == tf.estimator.ModeKeys.TRAIN  # == 'train'
    x = image
    for filters in (32, 64):
        x = tf.layers.conv2d(x, filters, 3)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 3, 2)
    x = tf.reduce_mean(x, axis=(1, 2))
    logits = tf.layers.dense(x, 10)
    return logits
```

Secondly, while the call to `tf.layers.batch_normalization` creates operationsthat update the moving averages, these operations won't be *run* unless explicitly requested. We can do this by adding dependencies to our `train_op`, since the layer automatically adds the relevant operations to the `UPDATE_OPS` collection.
```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=step)
```

#### More Summaries!
Estimators automatically log all summaries created in either the `get_inputs` function or `model_fn`, so it's super easy to fill up tensorboard with fancy plots, images and layered histograms.

```python
tf.summary.image('input_image', features)
tf.summary.histogram('max_prob', tf.reduce_max(probs, axis=-1))
# Nothing else required!
```

#### Fine-Grain Control
If you're a little worried about the lack of control you now seem to have, don't worry: there are a *lot* of details we've glossed over in this example. The default options in the constructor and `train`/`evaluate` functions are generally good starting points, but you can always specify your own values.

In particular, check out the constructor's `config` argument for session management configuration, and `train`'s `hooks` argument for adding custom functionality at key points during a session's lifetime.

#### Combining Different Train Ops
More complex architectures often have multiple losses and different optimizers (e.g. GANs). While I'll be the first to admit there are limitations to the estimator framework, don't be too quick to give up on these. A `train_op` can simply be a `no_op` with a dependency on two other `train_op`s
```python
with tf.control_dependencies([train_op0, train_op1]):
    train_op = tf.no_op()
```

On that note, if you are actually looking to implement a `GAN`, check out the [tf.contrib.GANEstimator](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/gan/estimator/GANEstimator) implementation!

## Summary
While potentially confusing at first, estimators are well-designed objects that provide a framework for training and evaluating tensorflow models. Default optional arguments provide good starting points for most setups without taking away your control, and the resulting trained models can easily be used from outside the estimator framework.

Finally, it should be stressed that this article is in no way meant to discuss every aspect of estimators. The documentation is a good place to start to learn more, and if that's not enough the source code is all open source.

Good luck estimating!
