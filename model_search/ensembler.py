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
"""Ensembler object.

This object is responsible for ensembling towers, for train and eval.
"""

import collections

from model_search import logit_bundler
from model_search.architecture import architecture_utils
from model_search.generators import trial_utils
from model_search.proto import ensembling_spec_pb2
import tensorflow.compat.v2 as tf


# Set of logits to be used for backward propagation (training).
# One logit tensor to be used for evaluation (accuracy metrics and loss).
EnsembleLogits = collections.namedtuple(
    "EnsembleLogits", ["train_logits_specs", "eval_logits_spec"])


class Ensembler(logit_bundler.LogitBundler):
  """Generates candidates for Phoenix."""

  def __init__(self, phoenix_spec):
    self._phoenix_spec = phoenix_spec
    self._ensemble_spec = phoenix_spec.ensemble_spec

  def _create_average_ensemble_logits(self, ensemble_logits,
                                      search_logits_specs):
    """Bundles together averaged logits."""

    logits = tf.add_n(ensemble_logits) / len(ensemble_logits)
    ensemble_logits_spec = architecture_utils.LogitsSpec(logits=logits)

    if (trial_utils.is_nonadaptive_ensemble_search(self._ensemble_spec) or
        trial_utils.is_intermixed_ensemble_search(self._ensemble_spec)):
      return EnsembleLogits(
          train_logits_specs=[], eval_logits_spec=ensemble_logits_spec)

    if trial_utils.is_adaptive_ensemble_search(self._ensemble_spec):
      return EnsembleLogits(
          train_logits_specs=search_logits_specs,
          eval_logits_spec=ensemble_logits_spec)

    if trial_utils.is_residual_ensemble_search(self._ensemble_spec):
      return EnsembleLogits(
          train_logits_specs=[ensemble_logits_spec],
          eval_logits_spec=ensemble_logits_spec)

  def _create_weighted_ensemble_logits(self, ensemble_logits,
                                       search_logits_specs, logits_dimension):
    """Bundles together weighted logits."""

    logits = tf.keras.layers.Dense(units=logits_dimension)(
        tf.concat(ensemble_logits, axis=-1))
    ensemble_logits_spec = architecture_utils.LogitsSpec(logits=logits)

    if (trial_utils.is_nonadaptive_ensemble_search(self._ensemble_spec) or
        trial_utils.is_intermixed_ensemble_search(self._ensemble_spec)):
      return EnsembleLogits(
          train_logits_specs=[ensemble_logits_spec],
          eval_logits_spec=ensemble_logits_spec)

    if trial_utils.is_adaptive_ensemble_search(self._ensemble_spec):
      return EnsembleLogits(
          train_logits_specs=[ensemble_logits_spec] + search_logits_specs,
          eval_logits_spec=ensemble_logits_spec)

    if trial_utils.is_residual_ensemble_search(self._ensemble_spec):
      return EnsembleLogits(
          train_logits_specs=[ensemble_logits_spec],
          eval_logits_spec=ensemble_logits_spec)

  def bundle_logits(self, priors_logits_specs, search_logits_specs,
                    logits_dimension):
    """Bundles the priors and the search candidate into an ensemble."""

    all_specs = priors_logits_specs + search_logits_specs
    assert all_specs, "Got no logits specs from both generators."

    with tf.compat.v1.variable_scope("Phoenix/Ensembler"):

      if search_logits_specs:
        assert len(search_logits_specs) == 1, "Search has more than one tower."

      # Simplest case - no ensemble yet, just a search candidate.
      if not priors_logits_specs and search_logits_specs:
        # Returning the only set of logits we have.
        return EnsembleLogits(
            train_logits_specs=search_logits_specs,
            eval_logits_spec=search_logits_specs[0])

      # Do not train already trained priors.
      stop_gradient_priors_specs = [
          spec._replace(logits=tf.stop_gradient(spec.logits))
          for spec in priors_logits_specs
      ]
      ensemble_logits = [
          spec.logits
          for spec in stop_gradient_priors_specs + search_logits_specs
      ]

      # Case 1: Average ensemble.
      # Train logits - the search generator logits if applicable.
      if (self._ensemble_spec.combining_type ==
          ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE):
        return self._create_average_ensemble_logits(ensemble_logits,
                                                    search_logits_specs)

      # Case 2: Weighted ensemble
      # Train logits should include the candidate training and the mixture
      # weights training.
      elif (self._ensemble_spec.combining_type ==
            ensembling_spec_pb2.EnsemblingSpec.WEIGHTED_ENSEMBLE):
        return self._create_weighted_ensemble_logits(ensemble_logits,
                                                     search_logits_specs,
                                                     logits_dimension)

      else:
        raise ValueError("Invalid Ensemble combining type.")
