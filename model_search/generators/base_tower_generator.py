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
"""Tower generator base class.

This class is reponsible for generating architectures or towers.
"""

import abc
import os
import time

from absl import logging
from model_search.architecture import architecture_utils
import tensorflow.compat.v2 as tf


SEARCH_GENERATOR = "search_generator"
PRIOR_GENERATOR = "prior_generator"
REPLAY_GENERATOR = "replay_generator"

ALL_GENERATORS = [PRIOR_GENERATOR, REPLAY_GENERATOR, SEARCH_GENERATOR]


class BaseTowerGenerator(object, metaclass=abc.ABCMeta):
  """Tower Generator base class for Phoenix."""

  def __init__(self, phoenix_spec, metadata):
    """Initialize the object."""
    self._phoenix_spec = phoenix_spec
    # Whether to allow attaching an auxiliary head to the generated tower.
    # Should be overridden by the subclass.
    self._allow_auxiliary_head = False
    self._metadata = metadata

  def _wait_for_chief(self, model_dir):
    """Waits on a directory till a checkpoint appears in it.

    Args:
      model_dir: string - the directory to wait on.
    """
    my_id = architecture_utils.DirectoryHandler.get_trial_id(
        model_dir, self._phoenix_spec)
    while not tf.train.latest_checkpoint(model_dir):
      logging.info("Waiting for chief to create the graph.")
      time.sleep(60)
      if not self._phoenix_spec.HasField("replay"):
        self._metadata.unblock_stopped_infeasible_trial(my_id)

  @abc.abstractproperty
  def generator_name(self):
    """The name of the generator."""

  def _save_architecture(self, architecture, model_dir, trial_id):
    """Saves architecture in a text file."""
    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)
    filename = os.path.join(model_dir, "{}.arch.txt".format(trial_id))
    with tf.io.gfile.GFile(filename, "w") as f:
      for item in map(str, architecture.tolist()):
        f.write("{}\n".format(item))

  def _build_from_existing_checkpoint(self,
                                      model_dir,
                                      features,
                                      input_layer_fn,
                                      trial_mode,
                                      shared_input_tensor,
                                      logits_dimension,
                                      is_training,
                                      shared_lengths=None):
    """Builds the neural network from an existing checkpoint.

    This function builds or replicates the model network given a model_dir with
    a checkpoint.

    Args:
      model_dir: a string holding the directory of the model. Must have a
        checkpoint in it.
      features: feature dict in case the tower needs to create the input tensor
      input_layer_fn: A function that converts feature Tensors to input layer.
        See learning.autolx.model_search.data.Provider.get_input_layer_fn
        for details.
      trial_mode: the TrialMode for the current Phoenix trial.
      shared_input_tensor: a tf.Tensor to build the network on top of.
      logits_dimension: An int holding the dimension of the logits (number of
        classes for the multi class problem).
      is_training: a boolean indicating if we are in training.
      shared_lengths: A tf.Tensor with the lengths (dimenions: [batch_size])
        holding the length of each sequence for sequential problems. Keep as
        None, for non sequential problems.

    Returns:
      A list of LogitsSpec, and a list of architectures (I.e., one LogitsSpec
      and one architecture per tower that the generator creates).
    """

    del trial_mode  # Unused. Only used in subclasses.

    logging.info("Building from existing checkpoint.")
    towers = architecture_utils.get_number_of_towers(model_dir,
                                                     self.generator_name())
    logits_specs = []
    architectures = []
    for i in range(towers):
      tower_name = self.generator_name() + "_{}".format(str(i))
      tower_spec = architecture_utils.import_tower(
          phoenix_spec=self._phoenix_spec,
          features=features,
          input_layer_fn=input_layer_fn,
          shared_input_tensor=shared_input_tensor,
          original_tower_name=tower_name,
          new_tower_name=tower_name,
          model_directory=model_dir,
          new_model_directory=model_dir,
          is_training=is_training,
          logits_dimension=logits_dimension,
          shared_lengths=shared_lengths,
          force_snapshot=False,
          force_freeze=False,
          allow_auxiliary_head=self._allow_auxiliary_head)
      logits_specs.append(tower_spec.logits_spec)
      architectures.append(tower_spec.architecture)

    architecture_utils.set_number_of_towers(self.generator_name(), towers)
    return logits_specs, architectures

  @abc.abstractmethod
  def first_time_chief_generate(self, features, input_layer_fn, trial_mode,
                                shared_input_tensor, shared_lengths,
                                logits_dimension, hparams, run_config,
                                is_training, trials):
    """Creates the tower(s).

    This function runs on the chief and runs only once!

    You can create the Phoenix tower. You can use randomness in creating the
    tower, and this class will take care of everything for you (take care that
    the workers have the same graph as chief even in the presence of randomness
    and that if premepteed, the chief will recover the same graph.

    Limitations for implementing this function and using this class:
    The towers must be a Phoenix towers. Meaning, they need to be constructed
    with architectural_utils.construct_tower

    Additionally, tower names must follow the following naming convention:
    `generator_name`_1, `generator_name`_2, etc.

    The above is needed for _build_from_existing_checkpoint.

    Args:
      features: feature dict in case the tower needs to create the input tensor
      input_layer_fn: A function that converts feature Tensors to input layer.
        See learning.autolx.model_search.data.Provider.get_input_layer_fn
        for details.
      trial_mode: the TrialMode for the current Phoenix trial.
      shared_input_tensor: tf.Tensor of type tf.float. It is the input to the
        network.
      shared_lengths: tf.Tensor of ints. It is the lengths for rnn problem. Keep
        as None if the problem is not recurrent.
      logits_dimension: The last axis dimension of the logits.
      hparams: hp.HParams with the hyperparameters.
      run_config: tf.RunConfig instance.
      is_training: a boolean specifying if we are training.
      trials: a list of `Trial` objects (metadata.trial.Trial).

    Returns:
      A list of LogitsSpec, and a list of architectures (I.e., one LogitsSpec
      and one architecture per tower that the generator creates).
    """

  def generate(self, features, input_layer_fn, trial_mode, shared_input_tensor,
               shared_lengths, logits_dimension, hparams, run_config,
               is_training, trials):
    """Generates the next architecture to try.

    Args:
      features: feature dict in case the tower needs to create the input tensor
      input_layer_fn: A function that converts feature Tensors to input layer.
        See learning.autolx.model_search.data.Provider.get_input_layer_fn
        for details.
      trial_mode: the TrialMode for the current Phoenix trial.
      shared_input_tensor: tf.Tensor of type tf.float. It is the input to the
        network.
      shared_lengths: tf.Tensor of ints. It is the lengths for rnn problem. Keep
        as None if the problem is not recurrent.
      logits_dimension: An int holding the dimension of the logits (number of
        classes for the multi class problem).
      hparams: the hyperparameters as given from oracle.
      run_config: A tf.estimator.RunConfig instance for the training process.
      is_training: a boolean indicating if we are in training.
      trials: list of `Trial`, with all information we have so far (to be passed
        to the smart search algorithm).

    Returns:
      A list of LogitsSpec, and a list of architectures (I.e., one LogitsSpec
      and one architecture per tower that the generator creates).
    """
    if not run_config.is_chief:
      self._wait_for_chief(run_config.model_dir)

    if tf.train.latest_checkpoint(run_config.model_dir):
      return self._build_from_existing_checkpoint(
          model_dir=run_config.model_dir,
          features=features,
          input_layer_fn=input_layer_fn,
          trial_mode=trial_mode,
          shared_input_tensor=shared_input_tensor,
          logits_dimension=logits_dimension,
          is_training=is_training,
          shared_lengths=shared_lengths)

    return self.first_time_chief_generate(
        features=features,
        input_layer_fn=input_layer_fn,
        trial_mode=trial_mode,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        logits_dimension=logits_dimension,
        hparams=hparams,
        run_config=run_config,
        is_training=is_training,
        trials=trials)
