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
"""Set of common utilities for the search algorithms."""

import os

from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
import numpy as np
import tensorflow.compat.v2 as tf


def random(prob):
  """Returns True with probability `prob`.

  Exploration controller to indicate it's time to take a random action.

  Args:
    prob: triggering probability. If prob > random, then returns True.

  Raises:
    ValueError: if prob is not bounded in [0, 1].
  """
  # 1.00001 to guard against numerical precision errors.
  if prob < 0 or prob > 1.00001:
    raise ValueError(
        "phoenix_spec.increase_complexity_probility must be in [0, 1]")
  return np.random.random() < prob


def write_fork_edge(model_dir, to_id, from_id):
  """Write an edge for the search tree graph.

  Args:
    model_dir: a string with the model directory.
    to_id: the target trial id (int).
    from_id: the trial we forked from (int).
  """
  if model_dir is None or not model_dir:
    return

  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

  filename = os.path.join(model_dir, "{}.search_edge.txt".format(to_id))
  with tf.io.gfile.GFile(filename, "w") as f:
    f.write("{},{}".format(to_id, from_id))


def encode_architecture(architecture, problem_type):
  """Encodes the architecture of strings into the np.array.

  Args:
    architecture: A list of strings of the architecture.
    problem_type: The phoenix_spec.ProblemType.

  Returns:
    The np.array of the encoded architecture.
  """

  architecture = [blocks.BlockType[b] for b in architecture]
  return np.array(
      architecture_utils.fix_architecture_order(architecture, problem_type))


def _default_depth_thresholds(max_depth=20):
  """Returns the default thresholds above which to grow by 1."""
  return [5 * i for i in range(max_depth)]


def get_allowed_depth(num_completed_trials, depth_thresholds=None,
                      max_depth=20):
  """Returns the current allowed depth of the architecture."""
  if not depth_thresholds:
    depth_thresholds = _default_depth_thresholds(max_depth)
  if len(depth_thresholds) > max_depth:
    raise ValueError(
        "phoenix_spec.increase_complexity_min_trials cannot have more "
        "thresholds than phoenix_spec.maximum_depth.")
  if num_completed_trials >= depth_thresholds[-1]:
    allowed_depth = max_depth
  else:
    allowed_depth = next(
        depth for depth, trial_threshold in enumerate(depth_thresholds)
        if trial_threshold > num_completed_trials)
  return allowed_depth


def block_indices(phoenix_spec):
  """Returns a list of allowable BlockType enum values from a phoenix_spec."""
  return [
      blocks.BlockType[block_type] for block_type in phoenix_spec.blocks_to_use
  ]


def choose_random_trial_and_get_architecture(trials):
  """Returns (architecture, trial) of a randomly chosen `trial`."""
  idx = np.random.randint(0, len(trials))
  chosen_trial = trials[idx]
  architecture = architecture_utils.get_architecture(
      architecture_utils.DirectoryHandler.trial_dir(chosen_trial))
  return architecture, chosen_trial.id


def mutate_replace(architecture, new_block):
  """Replaces one random block with the chosen new block.

  Returns a copy; input is not modified. The element to replace is chosen
  uniformly at random. Special care is taken not to replace the FLATTEN block.

  Args:
    architecture: An np.ndarray of integers corresponding to BlockType enum.
    new_block: Integer value of the desired BlockType to insert.

  Returns:
    An np.array of the architecture containing the new block.
  """
  output_architecture = architecture.copy()
  while True:
    block_to_replace = np.random.randint(0, architecture.size)
    blocktype = blocks.BlockType(output_architecture[block_to_replace])
    if blocktype not in blocks.FLATTEN_TYPES:
      break
  output_architecture[block_to_replace] = new_block
  return output_architecture
