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
"""Tests for model_search.search.identity."""

from absl.testing import parameterized
from model_search import hparam as hp
from model_search.proto import phoenix_spec_pb2
from model_search.search import identity
from model_search.search import test_utils

import numpy as np

import tensorflow.compat.v2 as tf


class IdentityTest(parameterized.TestCase, tf.test.TestCase):

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
  def test_get_suggestion(self, spec, initial_architecture,
                          expected_architecture):
    algorithm = identity.Identity(spec)
    output_architecture, _ = algorithm.get_suggestion(
        [], hp.HParams(initial_architecture=initial_architecture))
    self.assertAllEqual(expected_architecture, output_architecture)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
