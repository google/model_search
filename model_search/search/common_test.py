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
"""Tests for model_search.search.common."""


from absl.testing import parameterized
from model_search.search import common
import tensorflow.compat.v2 as tf


class CommonTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "no_completed_trials",
          "num_completed_trials": 0,
          "expected": 1,
      }, {
          "testcase_name": "some_completed_trials",
          "num_completed_trials": 11,
          "expected": 3,
      }, {
          "testcase_name": "custom_depth_thresholds",
          "num_completed_trials": 2,
          "expected": 2,
          "depth_thresholds": [0, 1, 10, 20],
      }, {
          "testcase_name": "maximum_respected",
          "num_completed_trials": 1000,
          "expected": 5,
      })
  def test_get_allowed_depth(self,
                             num_completed_trials,
                             expected,
                             depth_thresholds=None):
    actual = common.get_allowed_depth(
        num_completed_trials, depth_thresholds, max_depth=5)
    self.assertEqual(expected, actual)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
