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
"""Generate prior towers for the ensemble.

A prior here means a neural network tower(s) that is already trained, and we
know its performance in terms of loss.

This generator is exploite focused. It will attempt to import well performing
models from previous trials into the final Phoenix ensemble.
"""

import random

from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils
import tensorflow.compat.v2 as tf


class PriorGenerator(base_tower_generator.BaseTowerGenerator):
  """Generates prior towers for Phoenix."""

  def __init__(self, phoenix_spec, metadata):
    """Initializes the object."""
    super(PriorGenerator, self).__init__(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._ensemble_spec = self._phoenix_spec.ensemble_spec
    self._force_freeze = True

  def generator_name(self):
    return "prior_generator"

  def _get_best_trials(self, trials, minimal_pool_size, num_trials_to_consider):

    relevant_trials = [
        trial for trial in trials if trial.id <= minimal_pool_size
    ]

    best_trials = self._metadata.get_best_k(
        trials=relevant_trials, k=num_trials_to_consider)

    return best_trials

  def _nonadaptive_ensemble(self, features, input_layer_fn, shared_input_tensor,
                            shared_lengths, logits_dimension, relevant_trials,
                            is_training, num_trials_to_consider, width,
                            my_model_dir):
    best_trials = self._metadata.get_best_k(
        trials=relevant_trials, k=num_trials_to_consider)

    if not best_trials:
      raise tf.errors.FailedPreconditionError(
          None, None, "No completed trials to perform ensembling.")

    if len(best_trials) < width:
      # Not enough trials in the pool. Adding all available.
      previous_model_dirs = [
          architecture_utils.DirectoryHandler.trial_dir(trial)
          for trial in sorted(best_trials, key=lambda x: x.id)
      ]
    else:
      random_sample = random.sample(list(range(len(best_trials))), width)
      previous_model_dirs = [
          architecture_utils.DirectoryHandler.trial_dir(trial)
          for trial in sorted([best_trials[i] for i in random_sample],
                              key=lambda x: x.id)
      ]

    return trial_utils.import_towers_multiple_trials(
        features=features,
        input_layer_fn=input_layer_fn,
        phoenix_spec=self._phoenix_spec,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        is_training=is_training,
        logits_dimension=logits_dimension,
        previous_model_dirs=previous_model_dirs,
        force_freeze=self._force_freeze,
        allow_auxiliary_head=self._allow_auxiliary_head,
        caller_generator=self.generator_name(),
        my_model_dir=my_model_dir)

  def build_priors_intermixed(self, features, input_layer_fn,
                              shared_input_tensor, shared_lengths, is_training,
                              logits_dimension, trials, my_id, my_model_dir):
    intermixed_spec = self._ensemble_spec.intermixed_search
    return self._nonadaptive_ensemble(
        features=features,
        input_layer_fn=input_layer_fn,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        logits_dimension=logits_dimension,
        relevant_trials=trials,
        is_training=is_training,
        num_trials_to_consider=intermixed_spec.num_trials_to_consider,
        width=intermixed_spec.width,
        my_model_dir=my_model_dir)

  def build_priors_nonadaptively(self, features, input_layer_fn,
                                 shared_input_tensor, shared_lengths,
                                 is_training, logits_dimension, trials, my_id,
                                 my_model_dir):
    nonadaptive_spec = self._ensemble_spec.nonadaptive_search
    num_trials_to_consider = nonadaptive_spec.num_trials_to_consider
    assert num_trials_to_consider > 1
    best_trials = self._get_best_trials(trials,
                                        nonadaptive_spec.minimal_pool_size,
                                        num_trials_to_consider)

    if not best_trials:
      architecture_utils.set_number_of_towers(self.generator_name(), 0)
      return [], []

    return self._nonadaptive_ensemble(
        features=features,
        input_layer_fn=input_layer_fn,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        logits_dimension=logits_dimension,
        relevant_trials=best_trials,
        is_training=is_training,
        num_trials_to_consider=nonadaptive_spec.num_trials_to_consider,
        width=nonadaptive_spec.width,
        my_model_dir=my_model_dir)

  def build_priors_adaptively(self, features, input_layer_fn,
                              shared_input_tensor, shared_lengths, is_training,
                              trials, logits_dimension, my_id, my_model_dir):
    increase_every = self._ensemble_spec.adaptive_search.increase_width_every

    pool_size = my_id // increase_every * increase_every
    best_trial = self._get_best_trials(
        trials, pool_size, num_trials_to_consider=1)

    if not best_trial:
      architecture_utils.set_number_of_towers(self.generator_name(), 0)
      return [], []

    # TODO(b/172564129): In the adaptive case, if distillation happens before
    # ensembling, Phoenix will import both the teacher and the student and wire
    # them together. Ideally, we should only be importing the student. (This is
    # only an issue in the adaptive case since we call _import_towers_one_trial
    # which imports both the priors and the search tower, instead of just the
    # search tower).
    return trial_utils.import_towers_one_trial(
        features=features,
        input_layer_fn=input_layer_fn,
        phoenix_spec=self._phoenix_spec,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        is_training=is_training,
        logits_dimension=logits_dimension,
        prev_model_dir=architecture_utils.DirectoryHandler.trial_dir(
            best_trial),
        force_freeze=self._force_freeze,
        allow_auxiliary_head=self._allow_auxiliary_head,
        caller_generator=self.generator_name(),
        my_model_dir=my_model_dir)

  def build_priors_distillation(self, features, input_layer_fn,
                                shared_input_tensor, shared_lengths,
                                is_training, logits_dimension, trials, my_id,
                                my_model_dir):
    if not is_training:
      architecture_utils.set_number_of_towers(self.generator_name(), 0)
      return [], []

    # When distilling, we always want the best ensemble.
    num_trials_to_consider = 1
    best_trial = self._metadata.get_best_k(
        trials=trials, k=num_trials_to_consider)

    if not best_trial:
      architecture_utils.set_number_of_towers(self.generator_name(), 0)
      return [], []

    return trial_utils.import_towers_one_trial(
        features=features,
        input_layer_fn=input_layer_fn,
        phoenix_spec=self._phoenix_spec,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        is_training=is_training,
        logits_dimension=logits_dimension,
        prev_model_dir=architecture_utils.DirectoryHandler.trial_dir(
            best_trial),
        force_freeze=self._force_freeze,
        allow_auxiliary_head=self._allow_auxiliary_head,
        caller_generator=self.generator_name(),
        my_model_dir=my_model_dir)

  def _build_from_existing_checkpoint(self,
                                      model_dir,
                                      features,
                                      input_layer_fn,
                                      trial_mode,
                                      shared_input_tensor,
                                      logits_dimension,
                                      is_training,
                                      shared_lengths=None):
    """See parent class."""
    if trial_mode == trial_utils.TrialMode.DISTILLATION and not is_training:
      return [], []
    return super(PriorGenerator, self)._build_from_existing_checkpoint(
        model_dir,
        features,
        input_layer_fn,
        trial_mode,
        shared_input_tensor,
        logits_dimension,
        is_training,
        shared_lengths=None)

  def first_time_chief_generate(self, features, input_layer_fn, trial_mode,
                                shared_input_tensor, shared_lengths,
                                logits_dimension, hparams, run_config,
                                is_training, trials):
    """Creates the prior for the ensemble."""
    my_id = architecture_utils.DirectoryHandler.get_trial_id(
        run_config.model_dir, self._phoenix_spec)

    prior_build_args = dict(
        features=features,
        input_layer_fn=input_layer_fn,
        shared_input_tensor=shared_input_tensor,
        shared_lengths=shared_lengths,
        is_training=is_training,
        trials=trials,
        logits_dimension=logits_dimension,
        my_id=my_id,
        my_model_dir=run_config.model_dir)

    if trial_mode == trial_utils.TrialMode.DISTILLATION:
      return self.build_priors_distillation(**prior_build_args)

    if trial_utils.is_nonadaptive_ensemble_search(
        self._phoenix_spec.ensemble_spec):
      return self.build_priors_nonadaptively(**prior_build_args)

    if trial_utils.is_adaptive_ensemble_search(
        self._phoenix_spec.ensemble_spec):
      return self.build_priors_adaptively(**prior_build_args)

    if trial_utils.is_residual_ensemble_search(
        self._phoenix_spec.ensemble_spec):
      return self.build_priors_adaptively(**prior_build_args)

    if trial_utils.is_intermixed_ensemble_search(
        self._phoenix_spec.ensemble_spec):
      return self.build_priors_intermixed(**prior_build_args)

    # No ensemble spec or distillation spec was specified.
    architecture_utils.set_number_of_towers(self.generator_name(), 0)
    return [], []
