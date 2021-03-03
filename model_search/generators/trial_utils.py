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
"""Utils to handle the specs for ensemble search and distillation."""

import copy
import enum
import os

from model_search import hparam as hp
from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.metadata import trial as trial_module
from model_search.proto import distillation_spec_pb2
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf
from google.protobuf import text_format


class TrialMode(enum.IntEnum):
  """Enum used to check whether to run a distillation or ensemble search trial.

  Ordering here is used to determine which mode to pick when the pool sizes
  are identical. Ex, if the distillation pool size and the ensembling pool size
  are the same, ENSEMBLE_SEARCH will be selected because it is higher.
  """
  NO_PRIOR = 0
  DISTILLATION = 1
  ENSEMBLE_SEARCH = 2


_PREVIOUS_DEPENDENCIES_FILENAME = 'previous_dirs_dependencies.txt'


def get_trial_mode(ensemble_spec, distillation_spec, trial_id):
  """Determines whether to bundle logits with Ensembler or Distiller.

  If distillation and ensembling are specified at the same time, checks pool
  sizes to see which phase this trial falls in. If the pool sizes are the same,
  then defaults to ENSEMBLE_SEARCH.

  Args:
    ensemble_spec: The spec defined in the Phoenix spec.
    distillation_spec: The spec defined in the Phoenix spec.
    trial_id: trial id.

  Returns:
    TrialMode
  """

  # Handle intermixed separately since there is no pool size
  if is_intermixed_ensemble_search(ensemble_spec):
    if has_distillation(distillation_spec):
      n = ensemble_spec.intermixed_search.try_ensembling_every
      # Immediately after an ensemble search trial, do a distillation trial
      if trial_id % n == 1:
        return TrialMode.DISTILLATION
    return TrialMode.ENSEMBLE_SEARCH

  # Get the pool size for each mode.
  modes = []
  if has_ensemble_search(ensemble_spec):
    minimal_pool_size = 0
    if is_nonadaptive_ensemble_search(ensemble_spec):
      minimal_pool_size = ensemble_spec.nonadaptive_search.minimal_pool_size
    elif (is_adaptive_ensemble_search(ensemble_spec) or
          is_residual_ensemble_search(ensemble_spec)):
      minimal_pool_size = ensemble_spec.adaptive_search.minimal_pool_size
    modes.append((TrialMode.ENSEMBLE_SEARCH, minimal_pool_size))
  if has_distillation(distillation_spec):
    modes.append((TrialMode.DISTILLATION, distillation_spec.minimal_pool_size))

  # Check pool sizes to see what phase this trial is in.
  modes.sort(key=lambda x: (x[1], x[0]), reverse=True)
  for mode, n in modes:
    if trial_id > n:
      return mode
  return TrialMode.NO_PRIOR


def get_intermixed_trials(trials, n, n_user_suggestions):
  """Filters the exploration trials for intermixed ensemble search."""
  return [
      trial for trial in trials
      if trial.id % n != 0 or trial.id <= n_user_suggestions
  ]


def has_ensemble_search(spec):
  return (spec.ensemble_search_type !=
          ensembling_spec_pb2.EnsemblingSpec.UNKNOWN_ENSEMBLE_SEARCH)


def has_distillation(spec):
  return (spec.distillation_type != distillation_spec_pb2.DistillationSpec
          .DistillationType.UNKNOWN_DISTILLATION_TYPE)


def is_intermixed_ensemble_search(spec):
  return (spec.ensemble_search_type == ensembling_spec_pb2.EnsemblingSpec
          .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH)


def is_nonadaptive_ensemble_search(spec):
  return (spec.ensemble_search_type ==
          ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)


def is_adaptive_ensemble_search(spec):
  return (spec.ensemble_search_type ==
          ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH)


def is_residual_ensemble_search(spec):
  return (spec.ensemble_search_type ==
          ensembling_spec_pb2.EnsemblingSpec.RESIDUAL_ENSEMBLE_SEARCH)


def non_adaptive_or_intermixed_ensemble(spec):
  return (is_nonadaptive_ensemble_search(spec.ensemble_spec) or
          is_intermixed_ensemble_search(spec.ensemble_spec))


def adaptive_or_residual_ensemble(spec):
  return (is_adaptive_ensemble_search(spec.ensemble_spec) or
          is_residual_ensemble_search(spec.ensemble_spec))


def create_test_trials_intermixed(root_dir):
  """Creates fake trials used for testing."""
  trials = [{
      'model_dir': os.path.join(root_dir, str(2)),
      'id': 2,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.94
      },
  }, {
      'model_dir': os.path.join(root_dir, str(3)),
      'id': 3,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.7
      },
  }, {
      'model_dir': os.path.join(root_dir, str(5)),
      'id': 5,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.3
      },
  }]
  return [trial_module.Trial(t) for t in trials]


def import_towers_one_trial(features, input_layer_fn, phoenix_spec,
                            shared_input_tensor, shared_lengths, is_training,
                            logits_dimension, prev_model_dir, force_freeze,
                            allow_auxiliary_head, caller_generator,
                            my_model_dir):
  """Imports a previous trial to current model."""
  logits_specs = []
  architectures = []
  imported_towers = 0
  for generator in base_tower_generator.ALL_GENERATORS:
    towers = architecture_utils.get_number_of_towers(prev_model_dir, generator)
    # Import all towers from a given generator.
    for i in range(towers):
      tower_name = generator + '_{}'.format(str(i))
      tower_spec = architecture_utils.import_tower(
          phoenix_spec=phoenix_spec,
          features=features,
          input_layer_fn=input_layer_fn,
          shared_input_tensor=shared_input_tensor,
          original_tower_name=tower_name,
          new_tower_name=caller_generator + '_{}'.format(str(imported_towers)),
          model_directory=prev_model_dir,
          is_training=is_training,
          logits_dimension=logits_dimension,
          shared_lengths=shared_lengths,
          force_snapshot=True,
          new_model_directory=my_model_dir,
          force_freeze=force_freeze,
          allow_auxiliary_head=allow_auxiliary_head)
      logits_specs.append(tower_spec.logits_spec)
      architectures.append(tower_spec.architecture)
      imported_towers += 1

  architecture_utils.set_number_of_towers(caller_generator, imported_towers)
  if my_model_dir:
    tf.io.gfile.makedirs(my_model_dir)
    with tf.io.gfile.GFile(
        os.path.join(my_model_dir, _PREVIOUS_DEPENDENCIES_FILENAME), 'w+') as f:
      f.write(prev_model_dir)
  return logits_specs, architectures


def import_towers_multiple_trials(features, input_layer_fn, phoenix_spec,
                                  shared_input_tensor, shared_lengths,
                                  is_training, logits_dimension,
                                  previous_model_dirs, force_freeze,
                                  allow_auxiliary_head, caller_generator,
                                  my_model_dir):
  """Imports search generators' model from many trials."""
  logits_specs = []
  architectures = []
  for i, prev_model_dir in enumerate(previous_model_dirs):
    tower_name = 'search_generator_0'
    tower_spec = architecture_utils.import_tower(
        phoenix_spec=phoenix_spec,
        features=features,
        input_layer_fn=input_layer_fn,
        shared_input_tensor=shared_input_tensor,
        original_tower_name=tower_name,
        new_tower_name=caller_generator + '_{}'.format(str(i)),
        model_directory=prev_model_dir,
        is_training=is_training,
        logits_dimension=logits_dimension,
        shared_lengths=shared_lengths,
        force_snapshot=True,
        new_model_directory=my_model_dir,
        force_freeze=force_freeze,
        allow_auxiliary_head=allow_auxiliary_head)
    logits_specs.append(tower_spec.logits_spec)
    architectures.append(tower_spec.architecture)

  architecture_utils.set_number_of_towers(
      generator_name=caller_generator,
      number_of_towers=len(previous_model_dirs))
  if my_model_dir:
    tf.io.gfile.makedirs(my_model_dir)
    with tf.io.gfile.GFile(
        os.path.join(my_model_dir, _PREVIOUS_DEPENDENCIES_FILENAME), 'w+') as f:
      f.write('\n'.join(previous_model_dirs))
  return logits_specs, architectures


def write_replay_spec(model_dir, filename, original_spec, search_architecture,
                      hparams):
  """Writes a replay spec to retrain the same model."""
  # Ensure the same search space as the original run
  replay_spec = copy.deepcopy(original_spec)
  # Remove user suggestions
  replay_spec.ClearField('user_suggestions')
  # If this is already a replay config
  replay_spec.ClearField('replay')

  dependency_file = os.path.join(model_dir, _PREVIOUS_DEPENDENCIES_FILENAME)
  if tf.io.gfile.exists(dependency_file):
    with tf.io.gfile.GFile(dependency_file, 'r') as f:
      data = f.read().split('\n')
    for dependency in data:
      spec = phoenix_spec_pb2.PhoenixSpec()
      with tf.io.gfile.GFile(os.path.join(dependency, filename), 'r') as f:
        text_format.Parse(f.read(), spec)
      for i, _ in enumerate(spec.replay.towers):
        replay_spec.replay.towers.add().CopyFrom(spec.replay.towers[i])

  # Currently we support non-ensembling models.
  search_tower = replay_spec.replay.towers.add()

  # Removing the "context" hparam. This hparam is added for tpu, by the tpu team
  # It is not compatible with the function "to_proto" used below.
  copy_hparams = hp.HParams(**hparams.values())
  if hasattr(copy_hparams, 'context'):
    copy_hparams.del_hparam('context')

  # We sometimes (for some search algorithms) override the hparam
  # initial_architecture. Updating with the ground truth.
  # TODO(b/172564129): Make initial_architecture consistent everywhere.
  copy_hparams.set_hparam('initial_architecture', search_architecture)

  search_tower.hparams.CopyFrom(copy_hparams.to_proto())
  search_tower.architecture[:] = search_architecture
  with tf.io.gfile.GFile(os.path.join(model_dir, filename), 'w') as f:
    f.write(str(replay_spec))
