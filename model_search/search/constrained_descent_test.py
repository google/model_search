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
"""Tests for model_search.search.constrained_descent."""

from absl.testing import parameterized

import mock
from model_search import hparam as hp
from model_search.metadata import ml_metadata_db
from model_search.metadata import trial as trial_module
from model_search.proto import phoenix_spec_pb2
from model_search.search import constrained_descent
from model_search.search import test_utils as search_test_utils

import numpy as np
import tensorflow.compat.v2 as tf


class ConstrainedDescentTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ConstrainedDescentTest, self).setUp()
    self._metadata = ml_metadata_db.MLMetaData(None, None, None)

  # Note: id=8 is for the reduction block.
  @parameterized.named_parameters(
      {
          "testcase_name": "increase_size_no_reduction",
          "completed_trials": 1000,
          "initial_architecture": np.array([1, 2]),
          "expected_architecture": np.array([1, 2, 1]),
          "replicate_cell": False,
      },
      {
          "testcase_name": "increase_size_reduction",
          "completed_trials": 1000,
          "initial_architecture": np.array([1, 2, 3]),
          "expected_architecture": np.array([1, 2, 3, 1, 8]),
          "replicate_cell": False,
      },
      {
          "testcase_name": "mutation_with_replication",
          "completed_trials": 3,
          "initial_architecture": np.array([1, 2, 3, 4, 8]),
          "expected_architecture": np.array([1, 2, 1, 4, 8]),
          "replicate_cell": False,
      },
      {
          "testcase_name": "mutation_no_replication",
          "completed_trials": 3,
          "initial_architecture": np.array([1, 2, 3, 4, 8]),
          "expected_architecture": np.array([1, 2, 1, 4, 8, 1, 2, 1, 4, 8]),
          "replicate_cell": True,
      },
  )
  @mock.patch("numpy.random.randint")
  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_get_suggestion(self, get_architecture, randint, completed_trials,
                          initial_architecture, expected_architecture,
                          replicate_cell):
    get_architecture.return_value = initial_architecture
    randint.return_value = 2

    hparams = hp.HParams(new_block_type="FIXED_CHANNEL_CONVOLUTION_16")  # id=1.
    # Use problem_type=DNN so we don't have to worry about the final
    # architecture being correct.
    spec = phoenix_spec_pb2.PhoenixSpec(
        problem_type=phoenix_spec_pb2.PhoenixSpec.DNN,
        maximum_depth=11,
        num_blocks_in_cell=4,
        reduction_block_type="AVERAGE_POOL_2X2",
        replicate_cell=replicate_cell,
        beam_size=3)
    algorithm = constrained_descent.ConstrainedDescent(spec, self._metadata)
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

    actual_architecture, fork_trial = algorithm.get_suggestion(trials, hparams)

    # We use a beam of size 3 and always choose the last trial to be the best.
    self.assertEqual(fork_trial, completed_trials - 3)
    self.assertTrue(
        search_test_utils.is_mutation_or_equal(actual_architecture,
                                               expected_architecture))


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
