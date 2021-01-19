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
"""Defines tf.train.SessionRunHooks for transfer learning in Phoenix."""

import abc

from absl import logging
from model_search.architecture import architecture_utils
import tensorflow.compat.v2 as tf


class BaseTransferLearningHook(
    tf.estimator.SessionRunHook, metaclass=abc.ABCMeta):
  """Abstract base class for transfer learning hooks.

  This hook initializes the model's variables with values from previously
  completed trials. Subclasses are responsible for determining how to combine
  matching variables from previous trials. If any variable of a particular
  previous trial does not match in name, shape, or dtype to corresponding
  variable of the current trial it will be ignored.
  """

  def __init__(self, vars_to_warm_start, current_trial_id, completed_trials,
               discount_factor, max_completed_trials, model_dir):
    """Initializes a new BaseTransferLearningHoook instance.

    Args:
      vars_to_warm_start: The variables to warm start from previous trials.
      current_trial_id: The id of the current trial.
      completed_trials: The list of successfully completed trials. Will be used
        to warm start the variables of the current trial.
      discount_factor: Determines the importance of variables from past trials.
        If the current trial is n, the variables W of trial k, where k < n, will
        be weighted as W * discount_factor^(n-k).
      max_completed_trials: The maximum number of completed trials to consider
        when warm starting the current trial variables.
      model_dir: The current model directory (string).
    """
    if not vars_to_warm_start:
      tf.loggin.warning("vars_to_warm_start list is empty; hook will have no "
                        "effect.")
    self._vars_to_warm_start = vars_to_warm_start
    self._current_trial_id = current_trial_id
    self._completed_trials = completed_trials
    self._discount_factor = discount_factor
    self._max_completed_trials = max_completed_trials
    self._assign_ops = []
    self._model_dir = model_dir

  @abc.abstractmethod
  def _sort_previous_trial_variables(self, trial_to_var):
    """Sorts trial_to_var in an order implemented by the subclass.

    Args:
      trial_to_var: A list of (Trial, tf.Variable) tuples corresponding to
        previous trial variables which match variables in the current trial.

    Returns:
      The trial_to_var list in sorted order.
    """

  @abc.abstractmethod
  def _combine_previous_trial_variables(self, trial_to_var):
    """Combines the variables from previous trials.

    Args:
      trial_to_var: A list of (Trial, tf.Variable) tuples corresponding to
        previous trial variables which match variables in the current trial.

    Returns:
      The tf.Variable with the combined value of all variables.
    """

  # TODO(b/172564129): handle partitioned variables.
  def begin(self):
    """Creates the ops needed to warm start the model's variables."""
    # Do not warm start the model if a checkopint already exists. If the model
    # has already been training, we do not want to overwrite its variables.
    if tf.train.latest_checkpoint(self._model_dir):
      return
    if not self._completed_trials:
      return

    # Create CheckpointReaders for all completed trials.
    readers = []
    shape_maps = []
    dtype_maps = []
    for trial in self._completed_trials:
      reader = tf.train.load_checkpoint(
          architecture_utils.DirectoryHandler.trial_dir(trial))
      readers.append(reader)
      shape_maps.append(reader.get_variable_to_shape_map())
      dtype_maps.append(reader._GetVariableToDataTypeMap())  # pylint: disable=protected-access

    # Create the warm start ops.
    for var in self._vars_to_warm_start:
      trial_to_var = []
      for trial, reader, shape_map, dtype_map in zip(self._completed_trials,
                                                     readers, shape_maps,
                                                     dtype_maps):
        var_name = var.op.name
        if (not reader.has_tensor(var_name) or
            var.shape != shape_map[var_name] or
            var.dtype.base_dtype.as_datatype_enum != dtype_map[var_name]):
          continue
        tensor = reader.get_tensor(var_name)
        discounted = self._discount(trial, tensor)
        trial_to_var.append((trial, discounted))
      if not trial_to_var:
        logging.info(
            "Skipped warm starting %s since it does not match any variables "
            "from previous trials.", var_name)
        continue
      logging.info("Warm starting %s with values from %d previous trials.",
                   var_name, len(trial_to_var))
      trial_to_var = self._sort_previous_trial_variables(trial_to_var)
      trial_to_var = trial_to_var[:self._max_completed_trials]
      combined = self._combine_previous_trial_variables(trial_to_var)
      self._assign_ops.append(var.assign(combined))

  def _discount(self, completed_trial, tensor):
    exp = max(0, int(self._current_trial_id) - int(completed_trial.id))
    return self._discount_factor**exp * tensor

  def after_create_session(self, session, coord):
    del coord  # Unused.
    if self._assign_ops:
      session.run(self._assign_ops)
      self._assign_ops = []


class UniformAverageTransferLearningHook(BaseTransferLearningHook):
  """Warm starts current trial with uniform average of previous trials."""

  def _sort_previous_trial_variables(self, trial_to_var):
    return sorted(trial_to_var, key=lambda x: int(x[0].id), reverse=True)

  def _combine_previous_trial_variables(self, trial_to_var):
    _, variables = list(zip(*trial_to_var))
    return tf.reduce_mean(input_tensor=tf.stack(variables, axis=0), axis=0)


class LossWeightedAverageTransferLearningHook(BaseTransferLearningHook):
  """Warm starts current trial with loss weighted average of previous trials."""

  def _sort_previous_trial_variables(self, trial_to_var):
    return sorted(
        trial_to_var, key=lambda x: x[0].final_measurement.objective_value)

  def _combine_previous_trial_variables(self, trial_to_var):
    trials, variables = list(zip(*trial_to_var))
    # Negate losses so we can softmax them.
    losses = [-trial.final_measurement.objective_value for trial in trials]
    losses = tf.convert_to_tensor(value=losses, dtype=tf.float32)
    weights = tf.nn.softmax(losses)
    weighted = [weights[i] * var for i, var in enumerate(variables)]
    return tf.reduce_mean(input_tensor=tf.stack(weighted, axis=0), axis=0)
