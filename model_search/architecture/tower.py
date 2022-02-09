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

# Lint as: python3
"""Utils to handle architectures in Phoenix."""

from model_search.architecture import architecture_utils
import tensorflow.compat.v2 as tf


class Tower(tf.keras.Model):
  """A tower (Stacked ML model). Consists: logits, architecture, and layers."""

  def __init__(self,
               phoenix_spec,
               tower_name,
               architecture,
               is_training,
               logits_dimension,
               is_frozen,
               hparams,
               model_directory,
               dropout_rate=None,
               allow_auxiliary_head=False):
    super(Tower, self).__init__(name="Phoenix/{}".format(tower_name))
    self._phoenix_spec = phoenix_spec
    self._tower_name = tower_name
    self._construct_architecture = architecture
    self._is_training = is_training
    self._logits_dimension = logits_dimension
    self._is_frozen = is_frozen
    self._hparams = hparams
    self._model_directory = model_directory
    self._dropout_rate = dropout_rate
    self._allow_auxiliary_head = allow_auxiliary_head
    self._input_tensor = None

  @property
  def logits_spec(self):
    if not hasattr(self, "_logits_spec"):
      raise AssertionError("Calling logits_spec in keras model before calling "
                           "__call__ for the model.")
    return self._logits_spec

  @property
  def architecture(self):
    if not hasattr(self, "_architecture"):
      raise AssertionError("Calling architecture in keras model before calling "
                           "__call__ for the model.")
    return self._architecture

  @property
  def layer_tensors(self):
    if not hasattr(self, "_layer_tensors"):
      raise AssertionError("Calling layer_tensors in keras model before calling"
                           " __call__ for the model.")
    return self._layer_tensors

  @property
  def model_dir(self):
    return self._model_directory

  @property
  def previous_model_dir(self):
    return getattr(self, "_prev_model_dir", None)

  def has_input_tensor(self):
    return self._input_tensor is not None

  def add_initialization(self, prev_model_dir, prev_tower_name):
    self._prev_model_dir = prev_model_dir
    self._prev_tower_name = prev_tower_name

  def add_feature_columns_input_layer(self, input_tensor, lengths=None):
    self._input_tensor = input_tensor
    self._lengths = lengths

  def call(self, inputs, training):
    if inputs is not None:
      input_tensor = inputs
      self._lengths = None
    else:
      input_tensor = self._input_tensor
    tower_spec = architecture_utils.construct_tower(
        phoenix_spec=self._phoenix_spec,
        input_tensor=input_tensor,
        tower_name=self._tower_name,
        architecture=self._construct_architecture,
        is_training=training,
        lengths=self._lengths,
        logits_dimension=self._logits_dimension,
        is_frozen=self._is_frozen,
        hparams=self._hparams,
        model_directory=self._model_directory,
        dropout_rate=self._dropout_rate,
        allow_auxiliary_head=self._allow_auxiliary_head)
    # Populate TowerSpec to Tower itself.
    self._logits_spec = tower_spec.logits_spec
    self._architecture = tower_spec.architecture
    self._layer_tensors = tower_spec.layer_tensors
    if hasattr(self, "_prev_model_dir") and hasattr(self, "_prev_tower_name"):
      architecture_utils.init_variables(
          tf.train.latest_checkpoint(self._prev_model_dir),
          "Phoenix/{}".format(self._prev_tower_name),
          "Phoenix/{}".format(self._tower_name))
    return self._logits_spec.logits

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None,
           save_traces=True):
    super(Tower, self).save(
        filepath=filepath,
        overwrite=overwrite,
        include_optimizer=include_optimizer,
        save_format=save_format,
        signatures=signatures,
        options=options,
        save_traces=save_traces)

  @staticmethod
  def load(phoenix_spec,
           original_tower_name,
           new_tower_name,
           model_directory,
           new_model_directory,
           is_training,
           logits_dimension,
           force_freeze=False,
           allow_auxiliary_head=False,
           skip_initialization=False):
    """Imports a tower from the given model directory and the tower's name.

    Args:
      phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
      original_tower_name: a unique name for the tower (string) to import.
      new_tower_name: the name to give the new tower.
      model_directory: string, holds the model directory of the trial to import
        the tower from.
      new_model_directory: the model directory of the current trial.
      is_training: a boolean indicating if we are in training.
      logits_dimension: The last axis dimension of the logits.
      force_freeze: Force the imported tower to be frozen. For correctness, only
        force freeze with snapshotting (otherwise, you will freeze an
        non-trained tower).
      allow_auxiliary_head: Whether to allow constructing an auxiliary head for
        the tower if possible. Only applicable for CNNs.
      skip_initialization: does not add init variable from Phoenix. Relies on
        estimator init.

    Returns:
      A LogitsSpec containing the main and auxiliary logits and the architecture
      of the underlying tower.
    """
    initial_architecture = architecture_utils.get_architecture(
        model_directory, original_tower_name)
    dropout_rate = architecture_utils.get_parameter(model_directory,
                                                    original_tower_name,
                                                    architecture_utils.DROPOUTS)
    is_frozen = architecture_utils.get_parameter(model_directory,
                                                 original_tower_name,
                                                 architecture_utils.IS_FROZEN)
    hparams = architecture_utils.get_hparams_from_dir(model_directory,
                                                      original_tower_name)

    tower_spec = Tower(
        phoenix_spec=phoenix_spec,
        tower_name=new_tower_name,
        architecture=initial_architecture,
        is_training=is_training,
        logits_dimension=logits_dimension,
        dropout_rate=dropout_rate,
        hparams=hparams,
        model_directory=new_model_directory,
        is_frozen=(is_frozen or force_freeze),
        allow_auxiliary_head=allow_auxiliary_head)

    if not skip_initialization:
      tower_spec.add_initialization(
          prev_model_dir=model_directory, prev_tower_name=original_tower_name)
    return tower_spec
