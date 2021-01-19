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
"""Generate replay towers for the ensemble.

A replay here means a neural network tower(s) that is already trained, and we
know its performance in terms of loss.

This generator is to retrain an existing ensemble.
"""

import os

from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils


class ReplayGenerator(base_tower_generator.BaseTowerGenerator):
  """Generates prior towers for Replay in Phoenix."""

  def __init__(self, phoenix_spec, metadata):
    """Initializes the object."""
    super(ReplayGenerator, self).__init__(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._ensemble_spec = self._phoenix_spec.ensemble_spec
    self._force_freeze = True

  def generator_name(self):
    return "replay_generator"

  def first_time_chief_generate(self, features, input_layer_fn, trial_mode,
                                shared_input_tensor, shared_lengths,
                                logits_dimension, hparams, run_config,
                                is_training, trials):
    """Creates the prior for the ensemble."""
    my_id = architecture_utils.DirectoryHandler.get_trial_id(
        run_config.model_dir, self._phoenix_spec)

    # Adaptive ensemble - build gradually, import last trial in replay.
    if trial_utils.adaptive_or_residual_ensemble(self._phoenix_spec):
      previous_model_dir = os.path.join(
          os.path.dirname(run_config.model_dir), str(int(my_id) - 1))
      return trial_utils.import_towers_one_trial(
          features=features,
          input_layer_fn=input_layer_fn,
          phoenix_spec=self._phoenix_spec,
          shared_input_tensor=shared_input_tensor,
          shared_lengths=shared_lengths,
          is_training=is_training,
          logits_dimension=logits_dimension,
          prev_model_dir=previous_model_dir,
          force_freeze=self._force_freeze,
          allow_auxiliary_head=self._allow_auxiliary_head,
          caller_generator=self.generator_name(),
          my_model_dir=run_config.model_dir)

    # Non adaptive - import all towers after all are trained.
    if trial_utils.non_adaptive_or_intermixed_ensemble(self._phoenix_spec):
      previous_model_dirs = [
          os.path.join(os.path.dirname(run_config.model_dir), str(i + 1))
          for i in range(my_id - 1)
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
          my_model_dir=run_config.model_dir)
