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
"""A Phoenix controller - Responsible for choosing generators.

A controller is reponsible for choosing what to do in a trial.
It does so by choosing which generator to run:

* It could be a SearchGenerator - responsible for exploration
* It could be a PriorGenerator - responsible for exploitation - or adding good
trained models to the ensemble.
* It could be a ReplayGenerator - responsible for importing submodels in an
ensemble when performing replay.


It is also a modular way to introduce phases (e.g., compression in some of the
trials).
"""

import abc

import collections
from absl import logging

from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils
from model_search.generators.prior_generator import PriorGenerator
from model_search.generators.replay_generator import ReplayGenerator
from model_search.generators.search_candidate_generator import SearchCandidateGenerator


class ReplayState(object):
  """Helper class to understand Replay state."""

  def __init__(self, phoenix_spec):
    self._phoenix_spec = phoenix_spec

  def is_replay(self):
    """Returns True if this is a replay run."""
    return self._phoenix_spec.HasField("replay")

  def is_search(self):
    """Returns True if this is a search run."""
    return not self._phoenix_spec.HasField("replay")

  def replay_is_training_a_tower(self, my_id):
    """Returns True if we are training a new tower in a replay run.

    Example:
      1. In adaptive ensembling, every trial is training one new tower, so the
      return value is always True.
      2. In a non-adaptive ensembling, every trial except the last one is
      training a new tower, whereas the last trial just ensembles the trained
      towers.

    Args:
      my_id: trial id.  Returns True if we train a new tower, and False
        otherwise.
    """
    towers_number = len(self._phoenix_spec.replay.towers)
    output = (
        trial_utils.adaptive_or_residual_ensemble(self._phoenix_spec) or
        my_id < towers_number or towers_number == 1)
    return output

  def replay_is_importing_towers(self, my_id):
    """Returns true if we are importing a tower in this replay trial.

    Examples:
      1. For adaptive ensembling, we import towers for every trial with id
      greater than 1.
      2. For non-adaptive ensembling, we import towers only in the last trial.

    Args:
      my_id: trial id.

    Returns:
      True if we are importing a tower and False otherwise.
    """
    towers_number = len(self._phoenix_spec.replay.towers)
    output = ((my_id == towers_number and towers_number > 1) or
              (my_id > 1 and
               trial_utils.adaptive_or_residual_ensemble(self._phoenix_spec)))
    return output


class GeneratorWithTrials(
    collections.namedtuple("GeneratorWithTrials",
                           ["instance", "relevant_trials"])):
  """A generator instance with all relevant trials."""

  def __new__(cls, instance, relevant_trials):
    return super(GeneratorWithTrials, cls).__new__(cls, instance,
                                                   relevant_trials)


def _return_generators(generators):
  """Sets the number of towers to zero when generator isn't used."""
  for generator_name in base_tower_generator.ALL_GENERATORS:
    if generator_name not in generators.keys():
      architecture_utils.set_number_of_towers(generator_name, 0)

  return generators


class Controller(object, metaclass=abc.ABCMeta):
  """An api for a controller."""

  @abc.abstractmethod
  def get_generators(self, my_id, all_trials):
    """Returns the `Dict` of generators that need to be triggered.

    Args:
      my_id: an int with the current trial id.
      all_trials: a list of metadata.trial.Trial protos with all information in
        the current study.

    Returns:
      A dict of generator names as keys and GeneratorWithTrials as values.
    """


class InProcessController(Controller):
  """An In process Phoenix controller.

  This controller assumes search, ensembling, distillation and replay all run
  in the same binary.

  It will allocate trials for the various functionalities based on trial id.
  """

  def __init__(self, phoenix_spec, metadata):
    self._phoenix_spec = phoenix_spec
    self._search_candidate_generator = SearchCandidateGenerator(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._prior_candidate_generator = PriorGenerator(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._replay_generator = ReplayGenerator(
        phoenix_spec=phoenix_spec, metadata=metadata)
    self._replay_state = ReplayState(phoenix_spec)

  def get_generators(self, my_id, all_trials):
    """Determines which generators to run."""
    output = {}
    ensemble_spec = self._phoenix_spec.ensemble_spec
    distillation_spec = self._phoenix_spec.distillation_spec
    logging.info("trial id: %d", my_id)

    # Handling replay
    if self._replay_state.is_replay():
      if self._replay_state.replay_is_training_a_tower(my_id):
        output.update({
            base_tower_generator.SEARCH_GENERATOR:
                GeneratorWithTrials(self._search_candidate_generator, [])
        })
      if self._replay_state.replay_is_importing_towers(my_id):
        output.update({
            base_tower_generator.REPLAY_GENERATOR:
                GeneratorWithTrials(self._replay_generator, [])
        })

      return _return_generators(output)

    # Real Search from here on.
    # First: User suggestions first! No ensembling in suggestions.
    if my_id <= len(self._phoenix_spec.user_suggestions):
      logging.info("user suggestions mode")
      output.update({
          base_tower_generator.SEARCH_GENERATOR:
              GeneratorWithTrials(self._search_candidate_generator, [])
      })
      return _return_generators(output)

    # Second: Handle non-adaptive search
    if trial_utils.is_nonadaptive_ensemble_search(ensemble_spec):
      logging.info("non adaptive ensembling mode")
      pool_size = ensemble_spec.nonadaptive_search.minimal_pool_size
      search_trials = [t for t in all_trials if t.id <= pool_size]
      # Pool too small, continue searching
      if my_id <= pool_size:
        output.update({
            base_tower_generator.SEARCH_GENERATOR:
                GeneratorWithTrials(self._search_candidate_generator,
                                    search_trials)
        })
        return _return_generators(output)
      # Pool hit critical mass, start ensembling.
      else:
        output.update({
            base_tower_generator.PRIOR_GENERATOR:
                GeneratorWithTrials(self._prior_candidate_generator,
                                    search_trials)
        })
        return _return_generators(output)

    # Third: Adaptive / Residual ensemble search
    if (trial_utils.is_adaptive_ensemble_search(ensemble_spec) or
        trial_utils.is_residual_ensemble_search(ensemble_spec)):
      logging.info("adaptive/residual ensembling mode")
      increase_every = ensemble_spec.adaptive_search.increase_width_every
      pool_size = my_id // increase_every * increase_every
      ensembling_trials = [
          trial for trial in all_trials if trial.id <= pool_size
      ]
      search_trials = [trial for trial in all_trials if trial.id > pool_size]
      if ensembling_trials:
        output.update({
            base_tower_generator.SEARCH_GENERATOR:
                GeneratorWithTrials(self._search_candidate_generator,
                                    search_trials),
            base_tower_generator.PRIOR_GENERATOR:
                GeneratorWithTrials(self._prior_candidate_generator,
                                    ensembling_trials)
        })
        return _return_generators(output)
      else:
        output.update({
            base_tower_generator.SEARCH_GENERATOR:
                GeneratorWithTrials(self._search_candidate_generator,
                                    search_trials)
        })
        return _return_generators(output)

    # Fourth: Intermixed Search.
    if trial_utils.is_intermixed_ensemble_search(ensemble_spec):
      logging.info("intermix ensemble search mode")
      n = ensemble_spec.intermixed_search.try_ensembling_every
      search_trials = [t for t in all_trials if t.id % n != 0]
      if my_id % n != 0:
        output.update({
            base_tower_generator.SEARCH_GENERATOR:
                GeneratorWithTrials(self._search_candidate_generator,
                                    search_trials)
        })
        if (trial_utils.get_trial_mode(
            ensemble_spec, distillation_spec,
            my_id) == trial_utils.TrialMode.DISTILLATION):
          output.update({
              base_tower_generator.PRIOR_GENERATOR:
                  GeneratorWithTrials(self._prior_candidate_generator,
                                      all_trials)
          })
        return _return_generators(output)
      else:
        output.update({
            base_tower_generator.PRIOR_GENERATOR:
                GeneratorWithTrials(self._prior_candidate_generator,
                                    search_trials)
        })
        return _return_generators(output)

    # No ensembling
    output.update({
        base_tower_generator.SEARCH_GENERATOR:
            GeneratorWithTrials(self._search_candidate_generator, all_trials)
    })
    return _return_generators(output)
