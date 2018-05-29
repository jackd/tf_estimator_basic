Basic example usage of a custom [`tf.estimator.Estimator`](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator) for training, evaluation and prediction.

# Setup
Clone this repository and [tensorflow/models](https://github/tensorflow/models) and put the parent directory on you `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_estimator_basic.git
git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
cd tf_estimator_basic
```

# File Overview
* `simple.py`: very basic data/inference/evaluation model.
* `intermediate.py`: slightly more complicated data/inference/evaluation model.
* `dummy_server.py`: class used in `scripts/dummy_server.py` demonstrating manual session management with models trained with an estimator.

## Scripts
All scripts have a single argument - `model_id` which must be one of "simple" or "intermediate", e.g. `./train.py intermediate`. This defaults to "simple" for all except `clean.py` (for which one must be specified).

* `train.py`: trains the basic model.
* `eval.py`: evaluates a pre-trained model.
* `run_server.py`: runs a simple serving example.
* `vis_inputs.py`: visualizes the input images.
* `vis_predictions.py`: visualizes the predictions.
* `clean.py`: deletes the default/specified model directory.

# Basic Usage
```
cd scripts
./train.py
# this may take a few minutes
./eval.py
```

Progress can be observed using tensorboard
```
tensorboard --logdir=/tmp/mnist_simple
```
or
```
tensorboard --logdir=/tmp/mnist_intermediate
```

Models and data are saved to `/tmp/mnist_data/`, `/tmp/mnist_simple/` and `/tmp/mnist_intermediate/`. Note the default `/tmp` directory is normally deleted upon computer reboot.

For a more comprehensive write-up, see the [blog post](blog.md)
