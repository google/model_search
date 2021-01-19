# Copyright 2021 Google LLC
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
"""A small wrapper to run our library when working on a single machine."""

import kerastuner
from model_search import oss_trainer_lib
from model_search import phoenix
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf
from google.protobuf import text_format


class SingleTrainer(object):
  """Small wrapper for one simple one machine."""

  def __init__(self, data, spec):
    self._data = data

    if isinstance(spec, str):
      self._spec = phoenix_spec_pb2.PhoenixSpec()
      with tf.io.gfile.GFile(spec, "r") as f:
        text_format.Parse(f.read(), self._spec)
    else:
      self._spec = spec

    self._tuner_id = "tuner-1"

  def try_models(self, number_models, train_steps, eval_steps, root_dir,
                 batch_size, experiment_name, experiment_owner):
    """Simple function to invoke automl on one machine.

    Args:
      number_models: The number of neural networks to try.
      train_steps: The number of steps to train every candidate architecture.
      eval_steps: The number of steps to evaluated every candidate architecture.
      root_dir: The root directory to write all information we need during the
        search.
      batch_size: The batch size (integer). Example, if batch size is 10, and
        train_steps is 100, we are training over 10x100 = 1,000 examples.
      experiment_name: A string identifier for the run.
      experiment_owner: A string identifier of the user making the run.
    """
    loss_fn, metric_fn, predictions_fn = (
        oss_trainer_lib.loss_and_metric_and_predictions_fn(self._data))
    phoenix_instance = phoenix.Phoenix(
        phoenix_spec=self._spec,
        input_layer_fn=self._data.get_input_layer_fn(self._spec.problem_type),
        logits_dimension=self._data.number_of_classes(),
        study_name=experiment_name,
        study_owner=experiment_owner,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        predictions_fn=predictions_fn,
        metadata=None)

    self._oracle = kerastuner.tuners.bayesian.BayesianOptimizationOracle(
        objective="loss",
        hyperparameters=phoenix.Phoenix.get_keras_hyperparameters_space(
            self._spec, train_steps),
        max_trials=number_models)

    # pylint: disable=protected-access
    self._oracle._set_project_dir(root_dir, experiment_name, overwrite=True)
    # pylint: enable=protected-access

    while oss_trainer_lib.run_parameterized_train_and_eval(
        root_dir=root_dir,
        max_trials=number_models,
        data_provider=self._data,
        phoenix_instance=phoenix_instance,
        oracle=self._oracle,
        tuner_id=self._tuner_id,
        train_steps=train_steps,
        eval_steps=eval_steps,
        batch_size=batch_size):
      pass
