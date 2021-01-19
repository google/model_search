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
"""Tests for model_search.search.categorical_harmonica."""

from absl import logging
from absl.testing import parameterized
import mock
from model_search import hparam as hp
from model_search.metadata import trial as trial_module
from model_search.proto import phoenix_spec_pb2
from model_search.search import categorical_harmonica
from model_search.search import test_utils

import numpy as np
from sklearn import preprocessing
import tensorflow.compat.v2 as tf


PolynomialFeatures = preprocessing.PolynomialFeatures


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

  def assertAllAlmostEqual(self, left, right, places=3):
    for i in range(len(left)):
      self.assertAlmostEqual(left[i], right[i], places=places)

  def test_polynomial_expansion(self):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(phoenix_spec_pb2.PhoenixSpec.DNN))
    feature_extender = PolynomialFeatures(2, interaction_only=True)
    self.assertAllEqual([[1, 0, 1, 0], [1, 0.5, 1, 0.5]],
                        algorithm._get_polynomial_expansion(
                            feature_extender, np.array([[0, 1], [0.5, 1]])))

  def test_translate_architecture_to_assignment(self):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.DNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4))
    assignment = algorithm.translate_architecture_to_feature_assignment(
        np.array([2, 3, 4]))
    expected_output = [
        [
            1.00000000e+00, 6.12323400e-17, -1.00000000e+00, -1.83697020e-16,
            1.00000000e+00, -1.00000000e+00, 1.00000000e+00, -1.00000000e+00,
            1.00000000e+00, -1.83697020e-16, -1.00000000e+00, 5.51091060e-16,
            0.00000000e+00, -1.00000000e+00, 3.67394040e-16, 1.00000000e+00
        ],
        [
            0.00000000e+00, 1.00000000e+00, 1.22464680e-16, -1.00000000e+00,
            0.00000000e+00, 1.22464680e-16, -2.44929360e-16, 3.67394040e-16,
            0.00000000e+00, -1.00000000e+00, 3.67394040e-16, 1.00000000e+00,
            1.00000000e+00, -1.83697020e-16, -1.00000000e+00, 5.51091060e-16
        ]
    ]
    for i in range(len(assignment)):
      self.assertAllAlmostEqual(expected_output[i], assignment[i])

  @mock.patch("model_search.architecture.architecture_utils"
              ".get_architecture")
  def test_batch_sample(self, get_architecture):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.DNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4),
        degree=1)
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
    # Check first and last assignments
    self.assertAllAlmostEqual([
        1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
        1.00000000e+00, 6.12323400e-17, -1.00000000e+00, -1.83697020e-16,
        1.00000000e+00, -1.00000000e+00, 1.00000000e+00, -1.00000000e+00,
        1.00000000e+00, -1.83697020e-16, -1.00000000e+00, 5.51091060e-16
    ], x[0])
    self.assertAllAlmostEqual([
        0.00000000e+00, 1.22464680e-16, -2.44929360e-16, 3.67394040e-16,
        0.00000000e+00, -1.00000000e+00, 3.67394040e-16, 1.00000000e+00,
        0.00000000e+00, 1.22464680e-16, -2.44929360e-16, 3.67394040e-16,
        0.00000000e+00, -1.00000000e+00, 3.67394040e-16, 1.00000000e+00
    ], x[17])
    self.assertAllEqual([
        0.97, 0., 0.94, 0., 0.7, 0., 0.72, 0., 0.3, 0., 0.79, 0., 0.39, 0., 0.9,
        0., 0.19, 0.
    ], y)

  @parameterized.named_parameters({
      "testcase_name": "cnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.CNN
  }, {
      "testcase_name": "dnn",
      "problem_type": phoenix_spec_pb2.PhoenixSpec.DNN
  })
  def test_get_good_architecture(self, problem_type):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(
            problem_type,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=2),
        degree=2,
        seed=73)
    expected_output = [3, 3]
    if problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
      expected_output += [34]
    feature_extender = PolynomialFeatures(2, interaction_only=True)
    self.assertAllEqual(
        expected_output,
        algorithm._get_good_architecture(
            feature_extender, 10, np.array([0, 0.5, 0.6, -0.2] + [0] * 33),
            None))

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
    algorithm = categorical_harmonica.Harmonica(spec)
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
    algorithm = categorical_harmonica.Harmonica(
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
    expected_output = [3, 3, 3, 3]
    if problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
      expected_output += [34]
    self.assertAllEqual(expected_output, output_architecture)

  def test_extract_relevant_variables_indices(self):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.CNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=4),
        degree=2)
    feature_extender = PolynomialFeatures(2, interaction_only=True)
    algorithm._get_polynomial_expansion(feature_extender,
                                        np.array([[1, 2, 3, 4]]))
    output = algorithm._extract_relevant_variables_indices(
        feature_extender, np.array([1, 1, 1] + [0] * 7 + [1]))
    logging.info(output)
    self.assertAllEqual(output, set([0, 1, 2, 3]))
    output = algorithm._extract_relevant_variables_indices(
        feature_extender, np.array([1, 0, 0] + [0] * 7 + [1]))
    self.assertAllEqual(output, set([2, 3]))
    output = algorithm._extract_relevant_variables_indices(
        feature_extender, np.array([1, 0, 0] + [0] * 7 + [0]))
    self.assertEmpty(output)

  def test_get_good_architecture_with_relevant_variables(self):
    algorithm = categorical_harmonica.Harmonica(
        test_utils.create_spec(
            phoenix_spec_pb2.PhoenixSpec.CNN,
            blocks_to_use=[
                "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_32",
                "FIXED_CHANNEL_CONVOLUTION_64", "CONVOLUTION_3X3"
            ],
            min_depth=2),
        degree=2,
        seed=73)
    expected_output = [3, 1, 34]
    feature_extender = PolynomialFeatures(2, interaction_only=True)
    self.assertAllEqual(
        expected_output,
        algorithm._get_good_architecture(
            feature_extender, 20, np.array([0, 0.5, 0.6, -0.2] + [0] * 33),
            set([3, 3, 3, 3])))


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
