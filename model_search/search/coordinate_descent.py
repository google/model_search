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
"""Coordinate descent search algorithm for Phoenix."""

from absl import logging

from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
from model_search.search import common
from model_search.search import search_algorithm
import numpy as np


class CoordinateDescent(search_algorithm.SearchAlgorithm):
  """Mutates a block from best trial if no budget to increase depth."""

  def __init__(self, phoenix_spec, metadata):
    self._phoenix_spec = phoenix_spec
    self._max_depth = phoenix_spec.maximum_depth
    self._metadata = metadata

  # TODO(b/172564129): get_suggestion has a lot of boilerplate across search
  # algorithms (e.g. return initial architecture if not # enough trials, check
  # the allowed depth, etc). Possibly extract this boilerplate into private
  # functions of the base class.
  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """See the base class for details."""
    if self._phoenix_spec.beam_size < 1:
      raise ValueError("phoenix_spec.beam_size must be >= 1.")
    sorted_trials = self._metadata.get_best_k(
        trials, k=int(1e10), valid_only=True) or []
    num_completed_trials = len(sorted_trials)
    best_trials = sorted_trials[:self._phoenix_spec.beam_size]

    # No feasible trials yet.
    if not best_trials:
      return common.encode_architecture(hparams.initial_architecture,
                                        self._phoenix_spec.problem_type), None

    # Increase depth if possible.
    best_architecture, best_trial = (
        common.choose_random_trial_and_get_architecture(best_trials))
    allowed_depth = common.get_allowed_depth(
        num_completed_trials,
        depth_thresholds=self._phoenix_spec.increase_complexity_minimum_trials,
        max_depth=self._max_depth)
    logging.info("Maximal depth allowed: %d", allowed_depth)
    explore_mode = common.random(
        self._phoenix_spec.increase_complexity_probability)
    new_block = blocks.BlockType[hparams.new_block_type]

    if best_architecture.size < allowed_depth and explore_mode:
      common.write_fork_edge(model_dir, my_trial_id, best_trial)
      return architecture_utils.increase_structure_depth(
          best_architecture, new_block,
          self._phoenix_spec.problem_type), best_trial

    # Otherwise enter evolutionary mode.
    logging.info("using evolution")

    output_architecture = common.mutate_replace(best_architecture, new_block)
    output_architecture = [blocks.BlockType(x) for x in output_architecture]
    common.write_fork_edge(model_dir, my_trial_id, best_trial)
    return np.array(
        architecture_utils.fix_architecture_order(
            output_architecture, self._phoenix_spec.problem_type)), best_trial
