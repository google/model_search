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
"""Linear model search algorithm for Phoenix."""

from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
from model_search.proto import phoenix_spec_pb2
from model_search.search import common
from model_search.search import search_algorithm

import numpy as np
import sklearn
import sklearn.linear_model


def _one_nonzero_per_row(matrix):
  """For each row in matrix, randomly zero all but one of the nonzero values."""
  # TODO(b/172564129): can it be done without a loop?
  out = np.zeros_like(matrix)
  for i in range(matrix.shape[0]):
    nonzero_indices = np.flatnonzero(matrix[i])
    keep = np.random.choice(nonzero_indices)
    out[i, keep] = matrix[i, keep]
  return out


def _contains_row(matrix, row):
  for r in matrix:
    if np.all(r == row):
      return True
  return False


class LinearModel(search_algorithm.SearchAlgorithm):
  """Proposes new trials using a linear model.

  The model suggests an architecture no deeper than those in the trials.
  Sometimes (by a coin flip) adds one layer of depth to the suggestion.
  """

  def __init__(self, phoenix_spec):
    """Args: phoenix_spec: phoenix_spec_pb2 for the experiment."""
    self._phoenix_spec = phoenix_spec
    self._block_indices = np.unique(
        [blocks.BlockType.EMPTY_BLOCK.value] +
        [idx.value for idx in common.block_indices(phoenix_spec)])

  def _predict_best_architecture(self, architectures, losses):
    """Fits a linear model for loss = f(architecture) and finds its argmin.

    Main computational subroutine for trial data already in feature vector form.

    Args:
      architectures: (n_trials, depth) integer matrix of architectures.
      losses: (n_trials) positive validation error.

    Returns:
      predicted_loss: Scalar loss predicted for the chosen architecture.
      ints_best: (depth) integer vector representing
          the architecture that minimizes loss according to the model.
    """
    if self._phoenix_spec.linear_model.remove_outliers and len(losses) >= 10:
      median, decile = np.percentile(losses, [50, 90], interpolation="higher")
      keep = losses < min(decile, 10 * median)
      architectures = architectures[keep]
      losses = losses[keep]

    n_trials, depth = architectures.shape
    n_blocks = len(self._block_indices)

    # Reshaping because we have many values (layers) of one categorical feature
    # rather than many different categorical features.
    encoder = sklearn.preprocessing.OneHotEncoder(
        categories=[list(self._block_indices)])
    flat = architectures.reshape(-1, 1)
    x_onehot_flat = encoder.fit_transform(flat)
    x = x_onehot_flat.reshape((n_trials, depth * n_blocks))
    assert np.all(np.sum(x, axis=1) == depth)

    # Use ridge regession in case problem is underdetermined, which is likely.
    model = sklearn.linear_model.Ridge(
        alpha=self._phoenix_spec.linear_model.ridge_penalty)
    model = model.fit(x, losses)
    weights = model.coef_
    weights_bylayer = weights.reshape((depth, n_blocks))

    # Pick the block with minimum weight per layer. Break ties randomly.
    weights_min_bylayer = np.amin(weights_bylayer, axis=1)
    indicator_best = 1.0 * (weights_bylayer == weights_min_bylayer[:, None])
    onehot_best = _one_nonzero_per_row(indicator_best)

    predicted_loss = model.predict(onehot_best.reshape((1, -1)))[0]
    ints_best = encoder.inverse_transform(onehot_best).flatten()
    assert ints_best.shape == (depth,)
    return predicted_loss, ints_best

  def _suggest_by_padding(self, architectures, losses):
    """Pads architectures with EMPTY_BLOCK and call _predict_best_architecture.

    Variable-length architectures are padded into fixed dimensionality
    at either head or base, as determined by spec.network_alignment.

    Args:
      architectures: List of iterables of blocks.BlockType values (or integers).
      losses: Iterable of floats: objective value to be minimized.

    Returns:
      loss: Estimated loss value of best architecture according to the model.
      trimmed: Best architecture according to the model.
    """

    depths = np.array([len(arch) for arch in architectures])
    maxdepth = np.amax(depths)
    extended = np.array(
        [self._pad_architecture(arch, maxdepth) for arch in architectures])
    loss, suggestion = self._predict_best_architecture(extended, losses)
    trimmed = np.array([
        block for block in suggestion
        if block != blocks.BlockType.EMPTY_BLOCK.value
    ])
    return loss, trimmed

  def _pad_architecture(self, arch, maxdepth):
    """Pad with empty blocks according to spec network alignment."""
    empties = [blocks.BlockType.EMPTY_BLOCK.value] * (maxdepth - len(arch))
    align = self._phoenix_spec.linear_model.network_alignment
    if align == phoenix_spec_pb2.LinearModelSpec.NET_ALIGN_BASE:
      return empties + list(arch)
    elif (align == phoenix_spec_pb2.LinearModelSpec.NET_ALIGN_HEAD or
          align == phoenix_spec_pb2.LinearModelSpec.NET_ALIGN_UNSPECIFIED):
      return list(arch) + empties
    else:
      raise ValueError("Phoenix spec network_alignment unknown value")

  def _load_trials(self, trials):
    """Load trial architectures from filesystem."""

    completed_trials = trials

    architectures = []
    losses = []

    for trial in completed_trials:
      directory = architecture_utils.DirectoryHandler.trial_dir(trial)
      architecture = architecture_utils.get_architecture(directory)
      # The location of the flatten block is fixed
      # by the transition from convolutional to fully-connected layers.
      # It should not be a part of our search problem.
      # It will be placed by architecture_utils.fix_architecture_order().
      filtered = np.array([
          block for block in architecture if block not in blocks.FLATTEN_TYPES
      ])
      architectures.append(filtered)
      losses.append(trial.final_measurement.objective_value)

    return architectures, np.array(losses)

  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """See base class SearchAlgorithm."""

    architectures, losses = self._load_trials(trials)

    # No feasible trials yet.
    if len(architectures) < self._phoenix_spec.linear_model.trials_before_fit:
      return common.encode_architecture(hparams.initial_architecture,
                                        self._phoenix_spec.problem_type), None
    _, suggestion = self._suggest_by_padding(architectures, losses)

    # Decide whether to allow growth.
    # TODO(b/172564129): refactor common behavior with other search algorithms.
    allowed_depth = common.get_allowed_depth(
        len(architectures),
        depth_thresholds=self._phoenix_spec.increase_complexity_minimum_trials,
        max_depth=self._phoenix_spec.maximum_depth)
    explore_mode = common.random(
        self._phoenix_spec.increase_complexity_probability)

    new_block = blocks.BlockType[hparams.new_block_type]

    if suggestion.size <= allowed_depth and explore_mode:
      # increase_structure_depth expects that the architecture contains a
      # flatten block, which may not be true for the linear model's output.
      suggestion = np.array(
          architecture_utils.fix_architecture_order(
              suggestion, self._phoenix_spec.problem_type))
      suggestion = architecture_utils.increase_structure_depth(
          suggestion, new_block, self._phoenix_spec.problem_type)
    elif _contains_row(architectures, suggestion):
      # The linear model suggested an architecture we've already tried
      # in a previous trial, so we mutate it.
      # TODO(b/172564129): more intelligent _contains_row check: should handle
      # when mutate_replace output has been tried, but not just a while loop,
      # since that could run forver if # of untried architectures is small.
      suggestion = common.mutate_replace(suggestion, new_block)
    else:
      # The linear model suggested a novel architecture; use it.
      pass

    suggestion = [blocks.BlockType(b) for b in suggestion]
    return np.array(
        architecture_utils.fix_architecture_order(
            suggestion, self._phoenix_spec.problem_type)), None
