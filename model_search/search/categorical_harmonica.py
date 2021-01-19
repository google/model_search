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

import cmath

import os

from absl import logging
from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
from model_search.search import search_algorithm
import numpy as np

from sklearn import linear_model
from sklearn import preprocessing

import tensorflow.compat.v2 as tf


PolynomialFeatures = preprocessing.PolynomialFeatures


class Harmonica(search_algorithm.SearchAlgorithm):
  """Suggests an architecture based on the Harmonica algorithm."""

  def __init__(self,
               phoenix_spec,
               alpha=0.05,
               degree=2,
               n_mono=10,
               min_for_regression=3,
               num_random_samples=10000,
               num_of_restarts=3,
               seed=None,
               debug_mode=False):
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
      num_of_restarts: int, number of times to run the linear regression for
        determining the relevant variables.
      seed: int, for the np.random.
      debug_mode: If True, writes the coefficients of the model to the model
        directory for debugging purposes.
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
    self._num_params = self._phoenix_spec.minimum_depth
    # Blocktype to index -- needed to translate block type to a one-hot vector.
    self._block_indices = [
        blocks.BlockType[block_type]
        for block_type in self._phoenix_spec.blocks_to_use
    ]
    # Minimal amount of data point before applying regression.
    self._min_for_regression = min_for_regression
    # Must be greater than 1.
    self._num_random_samples = num_random_samples
    assert self._num_random_samples > 1
    self._num_of_restarts = num_of_restarts
    self._debug_mode = debug_mode

  def _get_polynomial_expansion(self, feature_extender, x):
    """Expand to polynomial features."""
    return feature_extender.fit_transform(np.array(x))

  def basis_function(self, t, k):
    r"""Calculates fourier basis.

    Args:
      t: integer.
      k: integer.

    Returns:
      Returns the following value:
      \Phi_t(k) = e^{(2\pi i k)/len(self._block_indices)}
    """
    value = cmath.exp(
        2 * cmath.pi * complex(0, 1) * t * k / len(self._block_indices))
    return value

  def translate_architecture_to_feature_assignment(self, architecture):
    """Translates the trial architecture to a categorical assignment."""
    category_size = len(self._block_indices)
    # TODO(b/172564129): Change to use architecture.size instead of _num_params
    x_real = np.empty(self._num_params * category_size)
    x_imag = np.empty(self._num_params * category_size)
    depth = 0
    for block in architecture:
      # These are connector blocks (non-trainable) that connect CNN and DNN.
      # They should not be a part of out search problem.
      if (block == blocks.BlockType.FLATTEN or
          block == blocks.BlockType.DOWNSAMPLE_FLATTEN or
          block == blocks.BlockType.PLATE_REDUCTION_FLATTEN):
        continue
      index = self._block_indices.index(block)
      for t in range(category_size):
        value = self.basis_function(t, index)
        x_real[category_size * depth + t] = value.real
        x_imag[category_size * depth + t] = value.imag
      depth += 1
    return [x_real, x_imag]

  def batch_sample(self, trials):
    """Returns all previous trials results as assignments and loss."""
    completed = trials
    x = []
    y = []
    for trial in completed:
      arc = architecture_utils.get_architecture(
          architecture_utils.DirectoryHandler.trial_dir(trial))
      # Returns two assignments to fit. One for the real values and one for
      # The imaginary
      x += self.translate_architecture_to_feature_assignment(arc)
      # Two objectives, one is for the real values assignment. The other is
      # zero for the imaginary assignment.
      y += [trial.final_measurement.objective_value, 0]
    return x, y

  def _parse_variable_name(self, name):
    """Returns the indices that form the variable given the name."""
    # Bias
    if name == "1":
      return []
    # Names are of the form 'x1 x6 x8'
    else:
      variables = name.split(" ")
      return [int(varname[1:]) for varname in variables]

  def _extract_relevant_variables_indices(self, feature_extender, coefficients):
    """Returns a list of the relevant variables indices based on coeff."""
    all_variable_names = feature_extender.get_feature_names()
    relevant_variables = []
    for i, coeff in enumerate(coefficients):
      if coeff > 0:
        relevant_variables += self._parse_variable_name(all_variable_names[i])
    return set(relevant_variables)

  def _get_good_architecture(self,
                             feature_extender,
                             num_samples,
                             coefficients,
                             relevant_variables=None):
    """Randomly samples architectures, predict loss, and return minimal.

    Args:
      feature_extender: sklearn PolynomialFeatures extnder.
      num_samples: the number of samples from the search space the function will
        try before returning the minimal point. If the search space over the
        relevant variables is smaller than num_samples, then, the method will
        also fix non-relevant variable to zero and search the whole space of the
        relevant variables.
      coefficients: coefficients of the polynomial model that predicts the loss.
      relevant_variables: set of indices of the most relevant variables in the
        linear model.

    Returns:
      A Phoenix architecture (np.array of ints) with minimal predicted loss.
    """
    if self._seed:
      np.random.seed(seed=self._seed)

    assignments = []
    architectures = []
    relevant_blocks = None
    if relevant_variables:
      logging.info("We have relevant variables")
      # From relevant variable in the fourier space to relevant categorical
      # Variables in the original space.
      relevant_blocks = set(
          [i // len(self._block_indices) for i in relevant_variables])

    # If true, we can search the whole space (the number of samples allowed
    # is less than the whole space).
    if (relevant_blocks and
        len(self._block_indices)**len(relevant_blocks) < num_samples):
      # All possible assignments - Creating the search space with cartesian
      # product
      search_space = [[0]] * self._num_params
      for i in relevant_blocks:
        search_space[i] = list(range(len(self._block_indices)))
      all_arch = np.array(np.meshgrid(*search_space)).T.reshape(
          -1, self._num_params)
      for data_point in all_arch:
        arch_list = [self._block_indices[i] for i in data_point]
        arch_list = architecture_utils.fix_architecture_order(
            arch_list, self._phoenix_spec.problem_type)
        arc = np.array(arch_list)
        architectures.append(arc)
        # Translates to two assignments to predict loss.
        assignments += self.translate_architecture_to_feature_assignment(arc)

    for i in range(num_samples):
      rand_arch = np.random.randint(
          len(self._block_indices), size=self._phoenix_spec.minimum_depth)
      arch_list = [self._block_indices[i] for i in rand_arch]
      # Translate to a valid architecture for Phoenix.
      arch_list = architecture_utils.fix_architecture_order(
          arch_list, self._phoenix_spec.problem_type)
      arch = np.array(arch_list)
      architectures.append(arch)
      # Translate to two assignments to predict loss.
      assignments += self.translate_architecture_to_feature_assignment(arch)

    # Expand to polynomial features.
    assignments = self._get_polynomial_expansion(feature_extender, assignments)
    # Predict loss based on coefficient of the linear model
    predictions_real_and_imag = np.square(np.matmul(assignments,
                                                    coefficients)).reshape(
                                                        -1, 2)

    # Summing the square of the real and imaginary values.
    predictions_sum_square_value = predictions_real_and_imag.sum(axis=1)
    # Index of the minimal loos
    minimal_index = np.argmin(predictions_sum_square_value)
    # Return the Phoenix architecutre for the minimal loss
    return architectures[minimal_index]

  def _write_model(self, model_dir, coefficients, my_trial_id):
    filename = os.path.join(model_dir, "harmonica.csv")
    with tf.io.gfile.GFile(filename, mode="w+") as f:
      np.savetxt(f, coefficients, delimiter=",")

  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """Suggests a new architecture for Phoenix using the harmonica model.

    For details please see:
    https://arxiv.org/pdf/1706.00764.pdf

    Args:
      trials: a list of Trial objects
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

    feature_extender = PolynomialFeatures(self._degree, interaction_only=True)

    # Expanding to polynomial features
    x = self._get_polynomial_expansion(feature_extender, x)

    zeroed_coeff = None
    relevant_variables = []
    for _ in range(self._num_of_restarts):
      logging.info("Running linear regression..")
      lasso_solver.fit(x, y)
      coef = lasso_solver.coef_
      # Sort the coefficients
      index = np.argsort(-np.abs(coef))
      # Zero out all coefficients but the largets self._n_mono
      value = coef[index[self._n_mono - 1]]
      zeroed_coeff = np.asarray(coef)
      before_zeroing = np.copy(zeroed_coeff)
      low_values_flags = zeroed_coeff < value  # Where values are low
      zeroed_coeff[low_values_flags] = 0  # All low values set to 0
      relevant_variables.extend(
          self._extract_relevant_variables_indices(feature_extender,
                                                   zeroed_coeff))

    if self._debug_mode and my_trial_id:
      self._write_model(model_dir, before_zeroing, my_trial_id)

    return self._get_good_architecture(feature_extender,
                                       self._num_random_samples, before_zeroing,
                                       relevant_variables), None
