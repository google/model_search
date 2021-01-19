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
"""Harmonica Search Algorithm in Phoenix.

For more details about Harmonica, please see:
https://arxiv.org/pdf/1706.00764.pdf
"""

from absl import logging
import numpy as np

from sklearn import linear_model
from sklearn import preprocessing

from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
from model_search.search import common
from model_search.search import search_algorithm

PolynomialFeatures = preprocessing.PolynomialFeatures


class Harmonica(search_algorithm.SearchAlgorithm):
  """Suggests an architecture based on the Harmonica algorithm."""

  def __init__(self,
               phoenix_spec,
               alpha=0.05,
               degree=3,
               n_mono=5,
               min_for_regression=3,
               num_random_samples=10000,
               seed=None):
    """Initializes the Harmonica instance.

    Args:
      phoenix_spec: PhoenixSpec proto.
      alpha: The alpha of lasso solver (please read on lasso solver to
        understand this constant. In a nutshell, this control regularization for
        lasso. alpha equal zero means regular linear regression - however, try
        not to use alpha = 0 because of numerical stability of the lasso
        implementation with that value.
      degree: The degree (int) of the polynomial we try to fit the data with.
      n_mono: The number of monomials in the function that fits the data. I.e.,
        we take the n_mono highest coefficients in the function and remove all
        the rest.
      min_for_regression: Minimal amount (int) of data point before applying
        regression.
      num_random_samples: int, must be greater than 1. Specifies the amount of
        architectures to sample and predict the loss for before proposing a new
        architecture to test.
      seed: int, for the np.random.
    """
    # Spec of phoenix
    self._phoenix_spec = phoenix_spec
    # The alpha of lasso solver (please read on lasso solver to understand this
    # constant. In a nutshell, this control regularization for lasso. alpha
    # equal zero means regular linear regression - however, try not to use
    # alpha = 0 because of numerical stability of the lasso implementation with
    # that value.
    self._alpha = alpha
    # The degree of the polynomial we try to fit the data with.
    self._degree = degree
    # The number of monomials in the function that fits the data. I.e., we take
    # the n_mono highest coefficients in the function and remove all the rest.
    self._n_mono = n_mono
    # Seed for np.random.randint
    self._seed = seed
    self._num_params = self._phoenix_spec.minimum_depth * len(
        self._phoenix_spec.blocks_to_use)
    # Blocktype to index -- needed to translate block type to a one-hot vector.
    self._block_indices = common.block_indices(phoenix_spec)
    # Minimal amount of data point before applying regression.
    self._min_for_regression = min_for_regression
    # Must be greater than 1.
    self._num_random_samples = num_random_samples
    assert self._num_random_samples > 1

  def get_polynomial_expansion(self, x, degree):
    """Expand to polynomial features."""
    feature_extender = PolynomialFeatures(degree, interaction_only=True)
    return feature_extender.fit_transform(np.array(x))

  def translate_architecture_to_feature_assignment(self, architecture):
    """Translates the trial architecture to a {-1, 1} assignment."""
    x = np.empty(self._num_params)
    x.fill(-1)
    depth = 0
    for block in architecture:
      # These are connector blocks (non-trainable) that connect CNN and DNN.
      # They should not be a part of out search problem.
      if (block == blocks.BlockType.FLATTEN or
          block == blocks.BlockType.DOWNSAMPLE_FLATTEN or
          block == blocks.BlockType.PLATE_REDUCTION_FLATTEN):
        continue
      index = self._block_indices.index(block)
      # One-hot encoding but with -1 and 1.
      x[len(self._block_indices) * depth + index] = 1
      depth += 1
    return x

  def batch_sample(self, trials):
    """Returns all previous trials results as assignments and loss."""
    completed = trials
    x = []
    y = []
    for trial in completed:
      arc = architecture_utils.get_architecture(
          architecture_utils.DirectoryHandler.trial_dir(trial))
      x.append(self.translate_architecture_to_feature_assignment(arc))
      y.append(trial.final_measurement.objective_value)
    return x, y

  def get_good_architecture(self, num_samples, coefficients):
    """Randomly samples architectures, predict loss, and return minimal."""
    if self._seed:
      np.random.seed(seed=self._seed)

    assignments = []
    architectures = []
    for _ in range(num_samples):
      rand_arc = np.random.randint(
          len(self._block_indices), size=self._phoenix_spec.minimum_depth)
      arc_list = [self._block_indices[i] for i in rand_arc]
      # Tranlsate to a valid architecture for Phoenix.
      arc_list = architecture_utils.fix_architecture_order(
          arc_list, self._phoenix_spec.problem_type)
      arc = np.array(arc_list)
      architectures.append(arc)
      # Translate to assignments to predict loss.
      assignments.append(self.translate_architecture_to_feature_assignment(arc))

    # Expand to polynomial features.
    assignments = self.get_polynomial_expansion(assignments, self._degree)
    # Predict loss based on coefficient of the linear model
    predictions = np.matmul(assignments, coefficients)
    # Index of the minimal loos
    minimal_index = np.argmin(predictions)
    # Return the Phoenix architecutre for the minimal loss
    return architectures[minimal_index]

  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """Suggests a new architecture for Phoenix using the harmonica model.

    For details please see:
    https://arxiv.org/pdf/1706.00764.pdf

    Args:
      trials: a list of metadata.trial.Trial
      hparams: The suggested hparams.
      my_trial_id: integer - the trial id which is making the call.
      model_dir: string - the model directory.

    Returns:
      an architecture and None - a np.array of integers and None.
    """
    lasso_solver = linear_model.Lasso(fit_intercept=True, alpha=self._alpha)

    # Getting the features/labels for the linear regression.
    x, y = self.batch_sample(trials)

    # Not enough data points for regression. Falling on hparams architecture.
    if len(y) < self._min_for_regression:
      initial_architecture = [
          blocks.BlockType[block_type]
          for block_type in hparams.initial_architecture
      ]
      return np.array(
          architecture_utils.fix_architecture_order(
              initial_architecture, self._phoenix_spec.problem_type)), None

    # Expanding to polynomial features
    x = self.get_polynomial_expansion(x, self._degree)

    logging.info("Running linear regression..")
    lasso_solver.fit(x, y)
    coef = lasso_solver.coef_
    # Sort the coefficients
    index = np.argsort(-np.abs(coef))
    # Zero out all coefficients but the largets self._n_mono
    value = coef[index[self._n_mono - 1]]
    zeroed_coeff = np.asarray(coef)
    low_values_flags = zeroed_coeff < value  # Where values are low
    zeroed_coeff[low_values_flags] = 0  # All low values set to 0

    return self.get_good_architecture(self._num_random_samples,
                                      zeroed_coeff), None
