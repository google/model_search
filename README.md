# Model Search

![header](https://raw.githubusercontent.com/google/model_search/master/model_search/images/model_search_logo.png)

Model search (MS) is a framework that implements AutoML algorithms for model architecture search at scale. It
aims to help researchers speed up their exploration process for finding the right
model architecture for their classification problems (i.e., DNNs with different types of layers).

The library enables you to:

* Run many AutoML algorithms out of the box on your data - including automatically searching
for the right model architecture, the right ensemble of models
and the best distilled models.

* Compare many different models that are found during the search.

* Create you own search space to customize the types of layers in your neural networks.

The technical description of the capabilities of this framework are found in
[InterSpeech paper](https://pdfs.semanticscholar.org/1bca/d4cdfbc01fbb60a815660d034e561843d67a.pdf).

While this framework can potentially be used for regression problems, the current
version supports classification problems only. Let's start by looking at some
classic classification problems and see how the framework can automatically find competitive
model architectures.

## Getting Started
Let us start with the simplest case. You have a csv file where the features are numbers
and you would like to run let AutoML find the best model architecture for you.

Below is a code snippet for doing so:

```python
import model_search
from model_search import constants
from model_search import single_trainer
from model_search.data import csv_data

trainer = single_trainer.SingleTrainer(
    data=csv_data.Provider(
        label_index=0,
        logits_dimension=2,
        record_defaults=[0, 0, 0, 0],
        filename="model_search/data/testdata/csv_random_data.csv"),
    spec=constants.DEFAULT_DNN)

trainer.try_models(
    number_models=200,
    train_steps=1000,
    eval_steps=100,
    root_dir="/tmp/run_example",
    batch_size=32,
    experiment_name="example",
    experiment_owner="model_search_user")
```

The above code will try 200 different models - all binary classification models,
as the `logits_dimension` is 2. The root directory will have a subdirectory of all
models, all of which will be already evaluated.
You can open the directory with tensorboard and see all the models with the
evaluation metrics.

The search will be performed according to the default specification. That can be found in:
`model_search/configs/dnn_config.pbtxt`.

For more details about the fields and if you want to create your own specification, you
can look at: `model_search/proto/phoenix_spec.proto`.

### Image data example
Below is an example of binary classification for images.

```python
import model_search
from model_search import constants
from model_search import single_trainer
from model_search.data import image_data

trainer = single_trainer.SingleTrainer(
    data=image_data.Provider(
        input_dir="model_search/data/testdata/images"
        image_height=100,
        image_width=100,
        eval_fraction=0.2),
    spec=constants.DEFAULT_CNN)

trainer.try_models(
    number_models=200,
    train_steps=1000,
    eval_steps=100,
    root_dir="/tmp/run_example",
    batch_size=32,
    experiment_name="example",
    experiment_owner="model_search_user")
```
The api above follows the same input fields as `tf.keras.preprocessing.image_dataset_from_directory`.

The search will be performed according to the default specification. That can be found in:
`model_search/configs/cnn_config.pbtxt`.

Now, what if you don't have a csv with the features or images? The next section shows
how to run without a csv.

## Non-csv, Non-image data
To run with non-csv data, you will have to implement a class inherited from the abstract
class `model_search.data.Provider`. This enables us to define our own
`input_fn` and hence customize the feature columns and the task (i.e., the number
of classes in the classification task).

```python
class Provider(object, metaclass=abc.ABCMeta):
  """A data provider interface.

  The Provider abstract class that defines three function for Estimator related
  training that return the following:
    * An input function for training and test input functions that return
      features and label batch tensors. It is responsible for parsing the
      dataset and buffering data.
    * The feature_columns for this dataset.
    * problem statement.
  """

  def get_input_fn(self, hparams, mode, batch_size: int):
    """Returns an `input_fn` for train and evaluation.

    Args:
      hparams: tf.HParams for the experiment.
      mode: Defines whether this is training or evaluation. See
        `estimator.ModeKeys`.
      batch_size: the batch size for training and eval.

    Returns:
      Returns an `input_fn` for train or evaluation.
    """

  def get_serving_input_fn(self, hparams):
    """Returns an `input_fn` for serving in an exported SavedModel.

    Args:
      hparams: tf.HParams for the experiment.

    Returns:
      Returns an `input_fn` that takes no arguments and returns a
        `ServingInputReceiver`.
    """

  @abc.abstractmethod
  def number_of_classes(self) -> int:
    """Returns the number of classes. Logits dim for regression."""

  def get_feature_columns(
      self
  ) -> List[Union[feature_column._FeatureColumn,
                  feature_column_v2.FeatureColumn]]:
    """Returns a `List` of feature columns."""
```

An example of an implementation can be found in `model_search/data/csv_data.py`.

Once you have this class, you can pass it to
`model_search.single_trainer.SingleTrainer` and your single trainer can now
read your data.

## Adding your models and architectures to a search space
You can use our platform to test your own existing models.

Our system searches over what we call `blocks`. We have created an abstract API
for an object that resembles a layer in a DNN. All that needs to be implemented for this class is
two functions:

```python
class Block(object, metaclass=abc.ABCMeta):
  """Block api for creating a new block."""

  @abc.abstractmethod
  def build(self, input_tensors, is_training, lengths=None):
    """Builds a block for phoenix.

    Args:
      input_tensors: A list of input tensors.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.

    Returns:
      output_tensors: A list of the output tensors.
    """

  @abc.abstractproperty
  def is_input_order_important(self):
    """Is the order of the entries in the input tensor important.

    Returns:
      A bool specifying if the order of the entries in the input is important.
      Examples where the order is important: Input for a cnn layer.
      (e.g., pixels an image). Examples when the order is not important:
      Input for a dense layer.
    """
```

Once you have implemented your own blocks (i.e., layers), you need to register them with a 
decorator. Example:

```python
@register_block(
    lookup_name='AVERAGE_POOL_2X2', init_args={'kernel_size': 2}, enum_id=8)
@register_block(
    lookup_name='AVERAGE_POOL_4X4', init_args={'kernel_size': 4}, enum_id=9)
class AveragePoolBlock(Block):
  """Average Pooling layer."""

  def __init__(self, kernel_size=2):
    self._kernel_size = kernel_size

  def build(self, input_tensors, is_training, lengths=None):
```

(All code above can be found in `model_search/blocks.py`).
Once registered, you can tell the system to search over these blocks by
supplying them in `blocks_to_use` in `PhoenixSpec` in
`model_search/proto/phoenix_spec.proto`. Namely, if you look at the default specification
for `dnn` found in `model_search/configs/dnn_config.pbtxt`, you can change the
repeated field `blocks_to_use` and add you own registered blocks.

Note: Our system stacks blocks one on top of each other to create tower
architectures that are then going to be ensembled. You can set the minimal and
maximal depth allowed in the config to 1 which will change the system to search
over which block perform best for the problem - I.e., your blocks can be now
an implementation of full classifiers and the system will choose the best one.

## Creating a training stand alone binary without writing a main
Now, let's assume you have the data class, but you don't want to write a `main`
function to run it.

We created a simple way to create a `main` that will just train a dataset and is
configurable via flags.

To create it, you need to follow two steps:

1. You need to register your data provider.

2. You need to call a help function to create a build rule.

Example:
Suppose you have a provider, then you need to register it via a decorator we
define it as follows:

```python
@data.register_provider(lookup_name='csv_data_provider', init_args={})
class Provider(data.Provider):
  """A csv data provider."""

  def __init__(self):
```

The above code can be found in `model_search/data/csv_data_for_binary.py`.

Next, once you have such library (data provider defined in a .py file and
registered), you can supply this library to a help build function an it will
create a binary rule as follows:

```build
model_search_oss_binary(
    name = "csv_data_binary",
    dataset_dep = ":csv_data_for_binary",
)
```

You can also add a test automatically to test integration of your provider with
the system as follows:

```build
model_search_oss_test(
    name = "csv_data_for_binary_test",
    dataset_dep = ":csv_data_for_binary",
    problem_type = "dnn",
    extra_args = [
        "--filename=$${TEST_SRCDIR}/model_search/data/testdata/csv_random_data.csv",
    ],
    test_data = [
        "//model_search/data/testdata:csv_random_data",
    ],
)
```

The above function will create a runable binary. The snippets are taken from the
following file: `model_search/data/BUILD`.
The binary is configurable by the flags in `model_search/oss_trainer_lib.py`.


## Distributed Runs
Our system can run a distributed search - I.e., run many search trainer in
parallel.

How does it work?

You need to run your binary on multiple machines. Additionally, you need to
make one change to configure the bookkeeping of the search.

On a single machine, the bookkeeping is done via a file. For a distributed
system however, we need a database.

In order to point our system to the database, you need to set the flags in the
file:

`model_search/metadata/ml_metadata_db.py`

to point to your database.

Once you have done so, the binaries created from the previous section will
connect to this database and an async search will begin.

## Cloud AutoML
Want to try higher performance AutoML without writing code? Try:
https://cloud.google.com/automl-tables
