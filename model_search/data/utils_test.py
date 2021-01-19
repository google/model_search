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
"""Tests for model_search.data.utils."""
from absl.testing import parameterized
from model_search.data import utils
from model_search.proto import phoenix_spec_pb2

import tensorflow.compat.v2 as tf


class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "cnn",
          "input_tensor_shape": [20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.CNN,
          "feature_columns": [tf.feature_column.numeric_column(key="feature")],
          "output_shape": [20, 20],
          "scope": ""
      }, {
          "testcase_name": "cnn_v1_fc",
          "input_tensor_shape": [20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.CNN,
          "feature_columns":
              [tf.compat.v1.feature_column.numeric_column(key="feature")],
          "output_shape": [20, 20],
          "scope": ""
      }, {
          "testcase_name": "dnn",
          "input_tensor_shape": [20, 20],
          "input_tensor_shape2": [20, 3],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.DNN,
          "feature_columns": [
              tf.feature_column.numeric_column(key="feature", shape=[20]),
              tf.feature_column.numeric_column(key="feature2", shape=[3])
          ],
          "output_shape": [20, 23],
          "scope": "scope2"
      }, {
          "testcase_name": "dnn_v1_fc",
          "input_tensor_shape": [20, 20],
          "input_tensor_shape2": [20, 3],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.DNN,
          "feature_columns": [
              tf.compat.v1.feature_column.numeric_column(
                  key="feature", shape=[20]),
              tf.compat.v1.feature_column.numeric_column(
                  key="feature2", shape=[3])
          ],
          "output_shape": [20, 23],
          "scope": "scope3"
      }, {
          "testcase_name": "rnn_all_activations",
          "input_tensor_shape": [20, 20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS,
          "output_shape": [20, 20, 20],
          "feature_columns": [tf.feature_column.numeric_column(key="feature")],
          "lengths_shape": [1],
          "scope": ""
      }, {
          "testcase_name": "rnn_all_activations_v1_fc",
          "input_tensor_shape": [20, 20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS,
          "output_shape": [20, 20, 20],
          "feature_columns":
              [tf.compat.v1.feature_column.numeric_column(key="feature")],
          "lengths_shape": [1],
          "scope": ""
      }, {
          "testcase_name": "rnn_last_activations",
          "input_tensor_shape": [20, 20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS,
          "output_shape": [20, 20, 20],
          "feature_columns": [tf.feature_column.numeric_column(key="feature")],
          "lengths_shape": [1],
          "scope": ""
      }, {
          "testcase_name": "rnn_last_activations_v1_fc",
          "input_tensor_shape": [20, 20, 20],
          "problem_type": phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS,
          "output_shape": [20, 20, 20],
          "feature_columns":
              [tf.compat.v1.feature_column.numeric_column(key="feature")],
          "lengths_shape": [1],
          "scope": ""
      })
  def test_create_input(self,
                        input_tensor_shape,
                        feature_columns,
                        problem_type,
                        output_shape,
                        input_tensor_shape2=None,
                        lengths_shape=None,
                        scope="Phoenix/Input"):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      features = {
          "feature": tf.zeros(input_tensor_shape),
      }
      lengths_feature_name = ""
      if lengths_shape:
        features["lengths"] = tf.ones(lengths_shape)
        lengths_feature_name = "lengths"
      if input_tensor_shape2:
        features["feature2"] = tf.ones(input_tensor_shape2)

      input_layer_fn = utils.default_get_input_layer_fn(problem_type,
                                                        feature_columns)

      input_tensor, lengths = input_layer_fn(
          features,
          is_training=True,
          scope_name=scope,
          lengths_feature_name=lengths_feature_name)

      self.assertAllEqual(input_tensor.shape, output_shape)
      self.assertIn(scope, input_tensor.name)
      if lengths_shape:
        self.assertAllEqual(lengths.shape, lengths_shape)


if __name__ == "__main__":
  tf.test.main()
