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
"""Tests for model_search.logit_bundler."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
import mock

from model_search import hparam as hp
from model_search.architecture import architecture_utils
from model_search.generators import trial_utils
from model_search.proto import distillation_spec_pb2
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2

import tensorflow.compat.v2 as tf

from google.protobuf import text_format


def _fake_import_tower_one_tower(phoenix_spec, features, input_layer_fn,
                                 shared_input_tensor, original_tower_name,
                                 new_tower_name, model_directory, is_training,
                                 logits_dimension, shared_lengths,
                                 new_model_directory, force_snapshot,
                                 force_freeze, allow_auxiliary_head, output):
  del phoenix_spec, features, input_layer_fn, shared_input_tensor
  del model_directory, is_training, logits_dimension, shared_lengths
  del force_snapshot, force_freeze, allow_auxiliary_head, new_model_directory
  output.update({original_tower_name: new_tower_name})
  return architecture_utils.TowerSpec(None, None, None)


def _fake_set_num_towers(generator_name, number_of_towers, output):
  output.update({generator_name: number_of_towers})
  return None


def _write_dependency_reply(model_dir):
  replay_string = """
      minimum_depth: 2
      maximum_depth: 20
      problem_type: CNN
      search_type: ADAPTIVE_COORDINATE_DESCENT
      replay {
        towers {
          architecture: "RNN_128"
          architecture: "RNN_128"
          hparams {
            hparam {
              key: "learning_rate"
              value {
                float_value: 0.021
              }
            }
            hparam {
              key: "initial_architecture"
              value {
                bytes_list {
                  value: "RNN_256"
                  value: "RNN_128"
                }
              }
            }
          }
        }
      }
    """
  dependency_file = os.path.join(model_dir, "previous_dirs_dependencies.txt")
  with tf.io.gfile.GFile(dependency_file, "w") as f:
    f.write(os.path.join(model_dir, "1"))
  tf.io.gfile.makedirs(os.path.join(model_dir, "1"))
  with tf.io.gfile.GFile(
      os.path.join(model_dir, "1", "replay_config.pbtxt"), "w") as f:
    f.write(replay_string)


class TrialUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def _create_ensemble_spec(self, ensembling_type, n):

    ensembling_spec = ensembling_spec_pb2.EnsemblingSpec()
    ensembling_spec.ensemble_search_type = ensembling_type

    if (ensembling_type ==
        ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH):
      ensembling_spec.nonadaptive_search.minimal_pool_size = n
    elif (ensembling_type ==
          ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH):
      ensembling_spec.adaptive_search.minimal_pool_size = n
    elif (ensembling_type == ensembling_spec_pb2.EnsemblingSpec
          .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH):
      ensembling_spec.intermixed_search.try_ensembling_every = n

    return ensembling_spec

  @parameterized.named_parameters(
      {
          "testcase_name":
              "no_priors",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .UNKNOWN_DISTILLATION_TYPE,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.UNKNOWN_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              0,
          "expected":
              trial_utils.TrialMode.NO_PRIOR,
      }, {
          "testcase_name":
              "distillation_only",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.UNKNOWN_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              0,
          "expected":
              trial_utils.TrialMode.DISTILLATION,
      }, {
          "testcase_name":
              "ensembling_only",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .UNKNOWN_DISTILLATION_TYPE,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              10,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      }, {
          "testcase_name":
              "distillation_after_ensembling_chooses_ensembling",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              75,
          "n":
              25,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      }, {
          "testcase_name":
              "distillation_after_ensembling_chooses_distillation",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              40,
          "n":
              20,
          "expected":
              trial_utils.TrialMode.DISTILLATION,
      }, {
          "testcase_name":
              "ensembling_after_distillation_chooses_distillation",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              25,
          "n":
              75,
          "expected":
              trial_utils.TrialMode.DISTILLATION,
      }, {
          "testcase_name":
              "ensembling_after_distillation_chooses_ensembling",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              20,
          "n":
              40,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      }, {
          "testcase_name":
              "not_enough_trials_chooses_no_prior",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              100,
          "n":
              200,
          "expected":
              trial_utils.TrialMode.NO_PRIOR,
      }, {
          "testcase_name":
              "distillation_and_ensemble_defaults_to_ensemble",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              20,
          "n":
              20,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      }, {
          "testcase_name":
              "returns_ensembling_on_ensembling_trial",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              10,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      }, {
          "testcase_name":
              "returns_distillation_after_ensembling_trial",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              49,
          "expected":
              trial_utils.TrialMode.DISTILLATION,
      }, {
          "testcase_name":
              "returns_ensembling_on_nonensembling_trial",
          "distillation_type":
              distillation_spec_pb2.DistillationSpec.DistillationType
              .MSE_LOGITS,
          "ensembling_type":
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          "distillation_pool":
              0,
          "n":
              100,
          "expected":
              trial_utils.TrialMode.ENSEMBLE_SEARCH,
      })
  def test_get_trial_mode(self, distillation_type, ensembling_type,
                          distillation_pool, n, expected):

    trial_id = 50

    distillation_spec = distillation_spec_pb2.DistillationSpec()
    distillation_spec.distillation_type = distillation_type
    distillation_spec.minimal_pool_size = distillation_pool

    ensembling_spec = self._create_ensemble_spec(ensembling_type, n)

    actual = trial_utils.get_trial_mode(ensembling_spec, distillation_spec,
                                        trial_id)
    self.assertEqual(actual, expected)

  @mock.patch("model_search.architecture.architecture_utils"
              ".set_number_of_towers")
  @mock.patch("model_search.architecture.architecture_utils"
              ".get_number_of_towers")
  @mock.patch("model_search.architecture.architecture_utils" ".import_tower")
  def test_import_one_model(self, import_tower, get_num_towers, set_towers):
    get_num_towers.side_effect = [1, 2, 1]
    output = {}
    set_output = {}
    import_tower.side_effect = functools.partial(
        _fake_import_tower_one_tower, output=output)
    set_towers.side_effect = functools.partial(
        _fake_set_num_towers, output=set_output)
    trial_utils.import_towers_one_trial(
        features=None,
        input_layer_fn=None,
        phoenix_spec=None,
        shared_input_tensor=None,
        shared_lengths=None,
        is_training=None,
        logits_dimension=None,
        prev_model_dir="1",
        force_freeze=None,
        allow_auxiliary_head=None,
        caller_generator="caller",
        my_model_dir=self.get_temp_dir())
    self.assertEqual(
        output, {
            "prior_generator_0": "caller_0",
            "replay_generator_0": "caller_1",
            "replay_generator_1": "caller_2",
            "search_generator_0": "caller_3",
        })
    self.assertEqual(set_output, {"caller": 4})
    with tf.io.gfile.GFile(
        self.get_temp_dir() + "/previous_dirs_dependencies.txt", "r") as f:
      data = f.read()
    self.assertEqual(data, "1")

  @mock.patch("model_search.architecture.architecture_utils"
              ".set_number_of_towers")
  @mock.patch("model_search.architecture.architecture_utils" ".import_tower")
  def test_import_multiple(self, import_tower, set_towers):
    output = {}
    set_output = {}
    import_tower.side_effect = functools.partial(
        _fake_import_tower_one_tower, output=output)
    set_towers.side_effect = functools.partial(
        _fake_set_num_towers, output=set_output)
    trial_utils.import_towers_multiple_trials(
        features=None,
        input_layer_fn=None,
        phoenix_spec=None,
        shared_input_tensor=None,
        shared_lengths=None,
        is_training=None,
        logits_dimension=None,
        previous_model_dirs=["1", "2"],
        force_freeze=None,
        allow_auxiliary_head=None,
        caller_generator="caller",
        my_model_dir=self.get_temp_dir())
    self.assertEqual(output, {"search_generator_0": "caller_1"})
    self.assertEqual(set_output, {"caller": 2})
    with tf.io.gfile.GFile(
        self.get_temp_dir() + "/previous_dirs_dependencies.txt", "r") as f:
      data = f.read()
    self.assertEqual(data, "1\n2")

  def test_write_replay_spec(self):
    original_spec = phoenix_spec_pb2.PhoenixSpec()
    text_format.Parse(
        """
      minimum_depth: 2
      maximum_depth: 20
      problem_type: CNN
      search_type: ADAPTIVE_COORDINATE_DESCENT
      user_suggestions {
        architecture: "LSTM_128"
      }
    """, original_spec)

    _write_dependency_reply(self.get_temp_dir())
    trial_utils.write_replay_spec(
        model_dir=self.get_temp_dir(),
        filename="replay_config.pbtxt",
        original_spec=original_spec,
        search_architecture=["RNN_128", "RNN_128"],
        hparams=hp.HParams(
            learning_rate=0.001,
            initial_architecture=["not_cor_arc", "not_cor_arc2"],
            context="should_be_deleted"))

    output_spec = phoenix_spec_pb2.PhoenixSpec()
    with tf.io.gfile.GFile(self.get_temp_dir() + "/replay_config.pbtxt",
                           "r") as f:
      text_format.Parse(f.read(), output_spec)

    expected_spec = phoenix_spec_pb2.PhoenixSpec()
    text_format.Parse(
        """
      minimum_depth: 2
      maximum_depth: 20
      problem_type: CNN
      search_type: ADAPTIVE_COORDINATE_DESCENT
      replay {
        towers {
          architecture: "RNN_128"
          architecture: "RNN_128"
          hparams {
            hparam {
              key: "learning_rate"
              value {
                float_value: 0.021
              }
            }
            hparam {
              key: "initial_architecture"
              value {
                bytes_list {
                  value: "RNN_256"
                  value: "RNN_128"
                }
              }
            }
          }
        }
        towers {
          architecture: "RNN_128"
          architecture: "RNN_128"
          hparams {
            hparam {
              key: "learning_rate"
              value {
                float_value: 0.001
              }
            }
            hparam {
              key: "initial_architecture"
              value {
                bytes_list {
                  value: "RNN_128"
                  value: "RNN_128"
                }
              }
            }
          }
        }
      }
    """, expected_spec)
    self.assertEqual(output_spec, expected_spec)


if __name__ == "__main__":
  absltest.main()
