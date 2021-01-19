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
"""Tests for model_search.search.harmonica."""

from absl.testing import parameterized

import mock
from model_search import hparam as hp
from model_search.metadata import trial as trial_module
from model_search.proto import phoenix_spec_pb2
from model_search.search import harmonica
from model_search.search import test_utils

import numpy as np

import tensorflow.compat.v2 as tf


def _create_trials():
  trials = [{
      "id": 99,
      "model_dir": "99",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.97
      },
  }, {
      "id": 97,
      "model_dir": "97",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.94
      },
  }, {
      "id": 96,
      "model_dir": "96",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.7
      },
  }, {
      "id": 95,
      "model_dir": "95",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.72
      },
  }, {
      "id": 94,
      "model_dir": "94",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.3
      },
  }, {
      "id": 93,
      "model_dir": "93",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.79
      },
  }, {
      "id": 92,
      "model_dir": "92",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.39
      },
  }, {
      "id": 91,
      "model_dir": "91",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.9
      },
  }, {
      "id": 90,
      "model_dir": "90",
      "status": "COMPLETED",
      "trial_infeasible": False,
      "final_measurement": {
          "objective_value": 0.19
      },
  }]
  return [trial_module.Trial(t) for t in trials]


class HarmonicaTest(parameterized.TestCase, tf.test.TestCase):

  def test_polynomial_expansion(self):
    algorithm = harmonica.Harmonica(
        test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN))
    self.assertAllEqual([[1, 0, 1, 0], [1, 0.5, 1, 0.5]],
                        algorithm.get_polynomial_expansion(
                            np.array([[0, 1], [0.5, 1]]), 3))

  def test_translate_architecture_to_assignment(self):
    algorithm = harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.DNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4))
    assignment = algorithm.translate_architecture_to_feature_assignment(
        np.array([2, 3, 4]))
    self.assertAllEqual(
        [-1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
        assignment)

  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_batch_sample(self, get_architecture):
    algorithm = harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.DNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4))
    get_architecture.side_effect = [
        np.array([1, 2, 3, 4]),
        np.array([2, 3, 4, 1]),
        np.array([3, 4, 1, 2]),
        # Adding 34 (flatten) to see that it is ignored
        np.array([4, 1, 2, 34, 3]),
        np.array([1, 1, 1, 1]),
        np.array([2, 2, 2, 2]),
        np.array([3, 3, 3, 3]),
        np.array([2, 3, 2, 3]),
        np.array([3, 4, 3, 4])
    ]
    x, y = algorithm.batch_sample(_create_trials())
    self.assertAllEqual(
        [[1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1],
         [-1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1],
         [-1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1],
         [-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
         [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1],
         [-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1],
         [-1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1],
         [-1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1],
         [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1]], x)
    self.assertAllEqual([0.97, 0.94, 0.7, 0.72, 0.3, 0.79, 0.39, 0.9, 0.19], y)

  @parameterized.named_parameters({
      "testcase_name": "cnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.CNN
  }, {
      "testcase_name": "dnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.DNN
  })
  def test_get_good_architecture(self, problem_type):
    algorithm = harmonica.Harmonica(
        test_utils.create_spec(
            problem_type,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=2),
        seed=73)
    expected_output = [3, 3]
    if problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
      expected_output += [34]
    self.assertAllEqual(
        expected_output,
        algorithm.get_good_architecture(
            10, np.array([0, 0.5, 0.6, -0.2] + [0] * 89)))

  @parameterized.named_parameters(
      {
          "testcase_name": "regular_cnn",
          "spec": test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "initial_architecture": [
              "FIXED_OUTPUT_FULLY_CONNECTED_128",
              "FIXED_CHANNEL_CONVOLUTION_32", "FIXED_CHANNEL_CONVOLUTION_64",
              "CONVOLUTION_3X3"
          ],
          "expected_architecture": np.array([2, 3, 4, 34, 22])
      }, {
          "testcase_name": "regular_dnn",
          "spec": test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "initial_architecture": [
              "FIXED_OUTPUT_FULLY_CONNECTED_128",
              "FIXED_OUTPUT_FULLY_CONNECTED_256"
          ],
          "expected_architecture": np.array([22, 23])
      })
  def test_not_enough_data_get_suggestion(self, spec, initial_architecture,
                                          expected_architecture):
    algorithm = harmonica.Harmonica(spec)
    output_architecture, _ = algorithm.get_suggestion(
        [], hp.HParams(initial_architecture=initial_architecture))
    self.assertAllEqual(expected_architecture, output_architecture)

  @parameterized.named_parameters({
      "testcase_name": "cnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.CNN
  }, {
      "testcase_name": "dnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.DNN
  })
  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_get_suggestion(self, get_architecture, problem_type):
    # Return value (architectures) for the various trials.
    get_architecture.side_effect = [
        np.array([1, 2, 3, 4]),
        np.array([2, 3, 4, 1]),
        np.array([3, 4, 1, 2]),
        np.array([4, 1, 2, 3]),
        np.array([1, 1, 1, 1]),
        np.array([2, 2, 2, 2]),
        np.array([3, 3, 3, 3]),
        np.array([2, 3, 2, 3]),
        np.array([3, 4, 3, 4])
    ]
    algorithm = harmonica.Harmonica(
        test_utils.create_spec(
            problem_type,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4),
        num_random_samples=10,
        seed=73)

    output_architecture, _ = algorithm.get_suggestion(_create_trials(),
                                                      hp.HParams())
    expected_output = [1, 1, 4, 2]
    if problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
      expected_output += [34]
    self.assertAllEqual(expected_output, output_architecture)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
