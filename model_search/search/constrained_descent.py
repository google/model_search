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
"""Constrained coordinate descent search algorithm for Phoenix."""

from absl import logging

from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
from model_search.search import common
from model_search.search import search_algorithm
import numpy as np


class ConstrainedDescent(search_algorithm.SearchAlgorithm):
  """Performs constrained coordinate descent on the architecture.

  This algorithm searches convolutions then inserts reduction blocks after
  every phoenix_spec.num_blocks_in_cell blocks. If phoenix_spec.replicate_cell
  is True, then only the first phoenix_spec.num_blocks_in_cell will be search
  via mutation and the whole architecture will be replicated with reduction
  blocks between each cell.
  """

  def __init__(self, phoenix_spec, metadata):
    self._phoenix_spec = phoenix_spec
    self._metadata = metadata

  def _is_reduction_block(self, block):
    name = blocks.BlockType(block).name
    return "REDUCTION" in name or "DOWNSAMPLE" in name or "POOL" in name

  def _remove_reduction_blocks(self, architecture):
    """Removes any reduction blocks from the architecture."""
    result = []
    for block in architecture:
      if self._is_reduction_block(block):
        continue
      result.append(block)
    return np.array(result)

  def _add_reduction_blocks(self, architecture, every, reduction_type):
    """Adds a reduction block of given type after `every` few blocks."""
    result = []
    for i, block in enumerate(architecture):
      result.append(block)
      if (i + 1) % every == 0:
        result.append(blocks.BlockType[reduction_type].value)
    return np.array(result)

  def _get_allowed_depth(self, num_completed_trials):
    """Returns the allowed depth not including reductions and flatten blocks."""
    if self._phoenix_spec.replicate_cell:
      allowed_depth = self._phoenix_spec.maximum_depth
    else:
      allowed_depth = common.get_allowed_depth(
          num_completed_trials,
          depth_thresholds=self._phoenix_spec
          .increase_complexity_minimum_trials,
          max_depth=self._phoenix_spec.maximum_depth)
    # We must take into account the number of reduction blocks to be added and
    # the final flatten block.
    allowed_depth -= allowed_depth // self._phoenix_spec.num_blocks_in_cell
    return allowed_depth

  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """See the base class for details."""
    del my_trial_id  # Unused.

    new_block = blocks.BlockType[hparams.new_block_type]
    if self._is_reduction_block(new_block):
      raise ValueError("ConstrainedDescent should not have reduction blocks in "
                       "its search space.")

    if self._phoenix_spec.beam_size < 1:
      raise ValueError("phoenix_spec.beam_size must be >= 1.")
    sorted_trials = self._metadata.get_best_k(
        trials, k=int(1e10), valid_only=True) or []
    num_completed_trials = len(sorted_trials)
    best_trials = sorted_trials[:self._phoenix_spec.beam_size]

    # No feasible trials yet, use initial architecture passed in from hparams.
    if not best_trials:
      best_architecture = common.encode_architecture(
          hparams.initial_architecture, self._phoenix_spec.problem_type)
      best_trial = None
    else:
      best_architecture, best_trial = (
          common.choose_random_trial_and_get_architecture(best_trials))

    # Get the architecture without reductions or replications which will be
    # grown or mutated.
    if self._phoenix_spec.replicate_cell:
      output_architecture = best_architecture[:self._phoenix_spec
                                              .num_blocks_in_cell]
      grow_mode = False
    else:
      output_architecture = self._remove_reduction_blocks(best_architecture)
      grow_mode = common.random(
          self._phoenix_spec.increase_complexity_probability)

    # Grow, mutate, and/or replicate architecture then add reductions & flatten.
    allowed_depth = self._get_allowed_depth(num_completed_trials)
    logging.info("Maximum depth allowed: %d", allowed_depth)
    if output_architecture.size < allowed_depth and grow_mode:
      logging.info("Growing the architecture.")
      output_architecture = architecture_utils.increase_structure_depth(
          output_architecture, new_block, self._phoenix_spec.problem_type)
    else:
      logging.info("Mutating the architecture.")
      output_architecture = common.mutate_replace(output_architecture,
                                                  new_block)

    if self._phoenix_spec.replicate_cell:
      replication_times = allowed_depth // self._phoenix_spec.num_blocks_in_cell
      output_architecture = np.concatenate(
          [output_architecture for _ in range(replication_times)])

    output_architecture = self._add_reduction_blocks(
        output_architecture, self._phoenix_spec.num_blocks_in_cell,
        self._phoenix_spec.reduction_block_type)
    output_architecture = [blocks.BlockType(x) for x in output_architecture]
    output_architecture = np.array(
        architecture_utils.fix_architecture_order(
            output_architecture, self._phoenix_spec.problem_type))
    return output_architecture, best_trial
