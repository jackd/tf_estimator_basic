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
* `model.py`: defines the inference/training model used throughout.
* `data.py`: `tf.data.Dataset` augmentation/manipulation, used for training and evaluation.
* `dummy_server.py`: class used in `scripts/dummy_server.py` demonstrating manual session management with models trained with an estimator.

## Scripts
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
tensorboard --logdir=/tmp/tf_estimator_basic
```

By default, models files are saved to/loaded from `/tmp/tf_estimator_basic`. This can be changed with the `-d` flag in scripts.
```
./train.py -d /path/to/custom_dir
./eval.py -d /path/to/custom_dir
tensorboard --logdir=/path/to/custom_dir
```

Note the default `/tmp` directory is normally deleted upon computer reboot.
