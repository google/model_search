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
"""Search candidate generator.

This class is reponsible for the architecture optimization (not the weights).
"""

import functools

from absl import logging

from model_search import blocks_builder as blocks
from model_search import search
from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils
from model_search.proto import phoenix_spec_pb2
from model_search.proto import transfer_learning_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf


class SearchCandidateGenerator(base_tower_generator.BaseTowerGenerator):
  """Generates candidates towers for Phoenix via search algorithms."""

  def __init__(self, phoenix_spec, metadata):
    """Initialize the object."""
    super(SearchCandidateGenerator, self).__init__(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._phoenix_spec = phoenix_spec
    self._max_depth = phoenix_spec.maximum_depth
    search_algorithms = {
        phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH:
            search.identity.Identity(phoenix_spec=phoenix_spec),
        phoenix_spec_pb2.PhoenixSpec.ADAPTIVE_COORDINATE_DESCENT:
            search.coordinate_descent.CoordinateDescent(
                phoenix_spec=phoenix_spec, metadata=self._metadata),
        phoenix_spec_pb2.PhoenixSpec.CONSTRAINED_ADAPTIVE_COORDINATE_DESCENT:
            search.constrained_descent.ConstrainedDescent(
                phoenix_spec=phoenix_spec, metadata=self._metadata),
        phoenix_spec_pb2.PhoenixSpec.HARMONICA_SEARCH:
            search.categorical_harmonica.Harmonica(phoenix_spec=phoenix_spec),
        phoenix_spec_pb2.PhoenixSpec.LINEAR_MODEL:
            search.linear_model.LinearModel(phoenix_spec=phoenix_spec),
    }
    self._search_algorithm = search_algorithms[phoenix_spec.search_type]
    self._ensemble_spec = phoenix_spec.ensemble_spec

    # Overridden from parent.
    self._allow_auxiliary_head = True

  def generator_name(self):
    return "search_generator"

  def _get_trial_from_id(self, trial_id, trials):
    for trial in trials:
      if trial.id == trial_id:
        return trial
    return None

  def _create_new_architecture(self, features, input_layer_fn,
                               shared_input_tensor, architecture, run_config,
                               my_id, is_training, shared_lengths, hparams,
                               logits_dimension, dropout_rate, prev_trial,
                               trials):
    logging.info("Creating new architecture: ")
    logging.info(architecture)

    input_tensor = shared_input_tensor
    lengths = shared_lengths
    if not self._phoenix_spec.is_input_shared:
      lengths_feature_name = self._phoenix_spec.lengths_feature_name
      if isinstance(features, dict) and lengths_feature_name not in features:
        lengths_feature_name = ""
      input_tensor, lengths = input_layer_fn(
          features=features,
          is_training=is_training,
          scope_name="Phoenix/" + self.generator_name() + "_0/Input",
          lengths_feature_name=lengths_feature_name)

    self._save_architecture(architecture, run_config.model_dir, my_id)

    tower_spec = architecture_utils.construct_tower(
        phoenix_spec=self._phoenix_spec,
        input_tensor=input_tensor,
        tower_name=self.generator_name() + "_0",
        architecture=architecture,
        is_training=is_training,
        lengths=lengths,
        logits_dimension=logits_dimension,
        is_frozen=False,
        hparams=hparams,
        model_directory=run_config.model_dir,
        dropout_rate=dropout_rate,
        allow_auxiliary_head=self._allow_auxiliary_head)
    logits_specs = [tower_spec.logits_spec]

    apply_snapshot = (
        self._phoenix_spec.transfer_learning_spec.transfer_learning_type ==
        transfer_learning_spec_pb2.TransferLearningSpec
        .SNAPSHOT_TRANSFER_LEARNING)
    if prev_trial and prev_trial > 0 and apply_snapshot:
      architecture_utils.init_variables(
          checkpoint=tf.train.latest_checkpoint(
              architecture_utils.DirectoryHandler.trial_dir(
                  self._get_trial_from_id(prev_trial, trials))),
          original_scope="Phoenix/{}_0".format(self.generator_name()),
          new_scope="Phoenix/{}_0".format(self.generator_name()))

    architecture_utils.set_number_of_towers(self.generator_name(), 1)
    return logits_specs, [tower_spec.architecture]

  def _get_user_suggestion(self, trial_id):
    suggestion = trial_id - 1
    architecture = self._phoenix_spec.user_suggestions[suggestion].architecture
    architecture = [blocks.BlockType[block_type] for block_type in architecture]
    return np.array(
        architecture_utils.fix_architecture_order(
            architecture, self._phoenix_spec.problem_type))

  def first_time_chief_generate(self, features, input_layer_fn, trial_mode,
                                shared_input_tensor, shared_lengths,
                                logits_dimension, hparams, run_config,
                                is_training, trials):
    dropout_rate = getattr(hparams, "dropout_rate", None)
    my_id = architecture_utils.DirectoryHandler.get_trial_id(
        run_config.model_dir, self._phoenix_spec)
    create_new_architecture_fn = functools.partial(
        self._create_new_architecture,
        features=features,
        input_layer_fn=input_layer_fn,
        shared_input_tensor=shared_input_tensor,
        run_config=run_config,
        my_id=my_id,
        hparams=hparams,
        is_training=is_training,
        shared_lengths=shared_lengths,
        logits_dimension=logits_dimension,
        dropout_rate=dropout_rate,
        trials=trials)

    # First, try out user suggestions.
    if my_id <= len(self._phoenix_spec.user_suggestions):
      return create_new_architecture_fn(
          architecture=self._get_user_suggestion(my_id), prev_trial=-1)

    if trial_mode == trial_utils.TrialMode.ENSEMBLE_SEARCH:

      # Non-adaptive ensemble search.
      if trial_utils.is_nonadaptive_ensemble_search(self._ensemble_spec):
        # Done searching if we've hit critical mass.
        architecture_utils.set_number_of_towers(self.generator_name(), 0)
        return [], []

      # Adaptive and residual ensemble search.
      elif (trial_utils.is_adaptive_ensemble_search(self._ensemble_spec) or
            trial_utils.is_residual_ensemble_search(self._ensemble_spec)):
        every = self._ensemble_spec.adaptive_search.increase_width_every
        relevant_trials = trials
        if every:
          relevant_trials = [
              trial for trial in trials if trial.id >= my_id // every * every
          ]
        architecture, prev_trial = self._search_algorithm.get_suggestion(
            relevant_trials, hparams, my_id, run_config.model_dir)
        return create_new_architecture_fn(
            architecture=architecture, prev_trial=prev_trial)

      # Intermixed ensemble search.
      elif trial_utils.is_intermixed_ensemble_search(self._ensemble_spec):
        every = self._ensemble_spec.intermixed_search.try_ensembling_every

        # Do not search if this is a non-exploration trial.
        if my_id % every == 0:
          architecture_utils.set_number_of_towers(self.generator_name(), 0)
          return [], []

        # Search if this is an exploration trial.
        relevant_trials = [trial for trial in trials if trial.id % every != 0]
        architecture, prev_trial = self._search_algorithm.get_suggestion(
            relevant_trials, hparams, my_id, run_config.model_dir)
        return create_new_architecture_fn(
            architecture=architecture, prev_trial=prev_trial)

      else:
        raise ValueError("Unknown ensemble search type '{}'".format(
            self._ensemble_spec.ensemble_search_type))

    if (trial_mode == trial_utils.TrialMode.DISTILLATION and
        trial_utils.is_intermixed_ensemble_search(self._ensemble_spec)):
      relevant_trials = trial_utils.get_intermixed_trials(
          trials, self._ensemble_spec.intermixed_search.try_ensembling_every,
          len(self._phoenix_spec.user_suggestions))
      best_trial = self._metadata.get_best_k(trials=relevant_trials, k=1)
      if best_trial is not None:
        model_dir = architecture_utils.DirectoryHandler.trial_dir(best_trial)
        assert architecture_utils.get_number_of_towers(
            model_dir, self.generator_name()) == 1
        tower_name = self.generator_name() + "_0"
        tower_spec = architecture_utils.import_tower(
            phoenix_spec=self._phoenix_spec,
            features=features,
            input_layer_fn=input_layer_fn,
            shared_input_tensor=shared_input_tensor,
            original_tower_name=tower_name,
            new_tower_name=tower_name,
            model_directory=model_dir,
            new_model_directory=run_config.model_dir,
            is_training=is_training,
            logits_dimension=logits_dimension,
            shared_lengths=shared_lengths,
            force_snapshot=False,
            force_freeze=False,
            allow_auxiliary_head=self._allow_auxiliary_head)
        architecture_utils.set_number_of_towers(self.generator_name(), 1)
        return [tower_spec.logits_spec], [tower_spec.architecture]

    # If no ensembling search method is specified, or this is a distillation
    # trial without intermixed ensemble_search, get a new tower based on the
    # architecture search algorithm.
    # This will serve as the student model if distillation occurs on this trial.
    architecture, prev_trial = self._search_algorithm.get_suggestion(
        trials, hparams, my_id, run_config.model_dir)
    return create_new_architecture_fn(
        architecture=architecture, prev_trial=prev_trial)
