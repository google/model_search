# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Interface for using model search with fast/light integration."""

import abc
import functools
from typing import List, Union
from model_search import registry
from model_search.data import utils

from tensorflow.python.feature_column import feature_column  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2  # pylint: disable=g-direct-tensorflow-import


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
  ) -> List[Union[feature_column._FeatureColumn,  # pylint: disable=protected-access
                  feature_column_v2.FeatureColumn]]:
    """Returns a `List` of feature columns."""

    raise NotImplementedError(
        "You must either implement get_feature_columns, or "
        "override get_input_layer_fn.")

  def get_input_layer_fn(self, problem_type):
    """Provides the function for converting feature Tensors to an input layer.

    Most users do not need to modify this function. In the typical use case,
    the user would only need to implement `get_feature_columns`, and the default
    implementation of this method would take care of converting the feature
    Tensors into the input layer accordingly. However, users who want to
    customize how they convert their feature Tensors to the input layer may
    override this method, and do not need to implement `get_feature_columns`.
    The function returned by this method will be called inside the model_fn,
    so it's okay for the returned input_layer_fn to create a Tensorflow subgraph
    with Variable ops (such as embedding weights) when it's called.

    For more details, see go/phx-custom-input-fn.

    Args:
      problem_type: A PhoenixSpec.ProblemType enum.

    Returns:
      A function like this:

      def input_layer_fn(features,
                         is_training,
                         scope_name="Phoenix/Input",
                         lengths_feature_name=None):
        with tf.variable_scope(scope_name):
          # create associated variables
          input_layer = ...
          length = ...  # if problem_type is an RNN type, Tensor, else None.

        return input_layer, lengths
    """
    feature_columns = self.get_feature_columns()

    return utils.default_get_input_layer_fn(problem_type, feature_columns)


register_provider = functools.partial(
    registry.register, base=Provider, enum_id=None)
