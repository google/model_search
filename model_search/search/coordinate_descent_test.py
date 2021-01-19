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
"""Tests for model_search.search.coordinate_descent."""

from absl import logging
from absl.testing import parameterized
import mock
from model_search import blocks_builder as blocks
from model_search import hparam as hp
from model_search.metadata import ml_metadata_db
from model_search.metadata import trial as trial_module
from model_search.proto import phoenix_spec_pb2
from model_search.search import coordinate_descent
from model_search.search import test_utils as search_test_utils

import numpy as np
import tensorflow.compat.v2 as tf


class CoordinateDescentTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(CoordinateDescentTest, self).setUp()
    self._metadata = ml_metadata_db.MLMetaData(None, None, None)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "increase_size",
          "spec":
              search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "init_architecture": [
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],  # np.array([2, 3, 4])
          "completed_trials":
              100000,
          "new_block":
              "FIXED_CHANNEL_CONVOLUTION_16",  # enum:1
          "should_increase_depth":
              True,
          "expected_fork_architecture":
              np.array([1, 2, 3, 4])
      },
      {
          "testcase_name":
              "no_completed_trials",
          "spec":
              search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "init_architecture": [
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],  # np.array([2, 3, 4])
          "completed_trials":
              0,
          "new_block":
              "FIXED_CHANNEL_CONVOLUTION_16",  # enum:1
          "should_increase_depth":
              False,
          "expected_fork_architecture":
              np.array([2, 3, 4])
      },
      {
          "testcase_name":
              "custom_depth_thresholds",
          "spec":
              search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN,
                                            [1, 2, 3, 4]),
          "init_architecture": [
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],  # np.array([2, 3, 4])
          "completed_trials":
              3,
          "new_block":
              "FIXED_CHANNEL_CONVOLUTION_16",  # enum:1
          "should_increase_depth":
              False,
          "expected_fork_architecture":
              np.array([1, 2, 3, 4])
      },
      {
          "testcase_name":
              "should_increase_depth",
          "spec":
              search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN,
                                            [1, 2, 3, 4]),
          "init_architecture": [
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],  # np.array([2, 3, 4])
          "completed_trials":
              4,
          "new_block":
              "FIXED_CHANNEL_CONVOLUTION_16",  # enum:1
          "should_increase_depth":
              True,
          "expected_fork_architecture":
              np.array([1, 2, 3, 4])
      },
      {
          "testcase_name":
              "increase_complexity_probability_eq_zero",
          "spec":
              search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN,
                                            [1, 2, 3, 4]),
          "init_architecture": [
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],  # np.array([2, 3, 4])
          "completed_trials":
              4,
          "new_block":
              "FIXED_CHANNEL_CONVOLUTION_16",  # enum:1
          "should_increase_depth":
              False,
          "expected_fork_architecture":
              np.array([1, 2, 3, 4]),
          "increase_complexity_probability":
              0.0,
      },
  )
  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_get_suggestion(self,
                          get_architecture,
                          spec,
                          init_architecture,
                          completed_trials,
                          new_block,
                          should_increase_depth,
                          expected_fork_architecture,
                          increase_complexity_probability=1.0):
    spec.increase_complexity_probability = increase_complexity_probability
    get_architecture.return_value = np.array([1, 2, 3, 4])
    algorithm = coordinate_descent.CoordinateDescent(spec, self._metadata)
    hparams = hp.HParams(
        initial_architecture=init_architecture, new_block_type=new_block)

    trials = []
    for i in range(completed_trials):
      trials.append(
          trial_module.Trial({
              "id": i,
              "model_dir": "/tmp/" + str(i),
              "status": "COMPLETED",
              "trial_infeasible": False,
              "final_measurement": {
                  "objective_value": 100 - i
              }
          }))

    # Adding one infeasible to make sure we don't fork from it.
    trials.append(
        trial_module.Trial({
            "id": 99,
            "model_dir": "/tmp/99",
            "status": "COMPLETED",
            "trial_infeasible": True,
            "final_measurement": {
                "objective_value": 0
            }
        }))
    logging.info(trials)
    output_architecture, fork_trial = algorithm.get_suggestion(trials, hparams)

    if completed_trials:
      self.assertEqual(fork_trial, completed_trials - 1)

    if should_increase_depth:
      self.assertAllEqual(
          output_architecture,
          np.append(expected_fork_architecture, blocks.BlockType[new_block]))
    else:
      self.assertEqual(output_architecture.shape,
                       expected_fork_architecture.shape)
      self.assertTrue(
          search_test_utils.is_mutation_or_equal(expected_fork_architecture,
                                                 output_architecture))

  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_get_suggestion_beam_size_gt_one(self, get_architecture):
    # Trials should be ranked by trial_id. That is, we should only ever fork
    # from the first 2 trials.
    beam_size = 2
    trial_to_arch = {
        0: np.array([1, 2, 3]),
        1: np.array([4, 5, 6]),
        2: np.array([7, 8, 9]),
        3: np.array([10, 11, 12])
    }
    get_architecture.side_effect = lambda idx: trial_to_arch[int(idx)]
    spec = search_test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN)
    spec.beam_size = beam_size
    spec.increase_complexity_minimum_trials.append(0)
    algorithm = coordinate_descent.CoordinateDescent(spec, self._metadata)
    hparams = hp.HParams(
        new_block_type="FIXED_CHANNEL_CONVOLUTION_16")  # enum: 1

    # Create one fake trial for each architecture.
    trials = []
    for i in range(3):
      trials.append(
          trial_module.Trial({
              "id": i,
              "model_dir": str(i),
              "status": "COMPLETED",
              "trial_infeasible": False,
              "final_measurement": {
                  "objective_value": i
              }
          }))

    # Adding one infeasible to make sure we don't fork from it.
    trials.append(
        trial_module.Trial({
            "id": 4,
            "model_dir": "4",
            "status": "COMPLETED",
            "trial_infeasible": True,
            "final_measurement": {
                "objective_value": 0
            }
        }))
    logging.info(trials)

    # Since forking is random, fork 1000 times then check that we forked from
    # only the trials we care about.
    forked = set()
    for i in range(1000):
      _, fork_trial = algorithm.get_suggestion(trials, hparams)
      forked.add(int(fork_trial))
    self.assertEqual(forked, {0, 1})


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
