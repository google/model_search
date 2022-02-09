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
"""Tests for model_search.architecture.graph_architecture."""

from absl.testing import parameterized

from model_search.architecture import architecture_utils
from model_search.architecture import graph_architecture
from model_search.proto import phoenix_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf

CombinerType = graph_architecture.CombinerType


def _create_spec(problem_type,
                 complexity_thresholds=None,
                 max_depth=None,
                 min_depth=None):
  output = phoenix_spec_pb2.PhoenixSpec()
  if complexity_thresholds is not None:
    output.increase_complexity_minimum_trials[:] = complexity_thresholds
  if max_depth is not None:
    output.maximum_depth = max_depth
  if min_depth is not None:
    output.minimum_depth = min_depth
  output.problem_type = problem_type
  return output


class GraphArchitectureTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name":
              "sequential_dnn",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(14),
                  graph_architecture.Node(14),
                  graph_architecture.Node(14),
              ]),
      },
      {
          "testcase_name":
              "one_skip_connection_dnn",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(14),
                  graph_architecture.Node(14, input_indices=[-2, -1]),
              ]),
      },
      {
          "testcase_name":
              "sequential_cnn",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(1),
                  graph_architecture.Node(1),
                  graph_architecture.Node(1),
              ]),
      },
      {
          "testcase_name":
              "nasnet",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(1),
                  graph_architecture.Node(
                      65,
                      input_indices=[-2, -1],
                      combiner_type=CombinerType.IDENTITY),
              ]),
      },
      {
          "testcase_name":
              "one_skip_connection_convnet",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(1),
                  graph_architecture.Node(1, input_indices=[-2, -1]),
              ]),
      },
      {
          "testcase_name":
              "densenet",
          "architecture":
              graph_architecture.Architecture(node_list=[
                  graph_architecture.Node(1),
                  graph_architecture.Node(1, input_indices=[-2, -1]),
                  graph_architecture.Node(1, input_indices=[-3, -2, -1]),
              ]),
      },
      {
          "testcase_name":
              "multiple_inputs",
          # Assume there are three inputs, which are layers 0, 1, and 2.
          "architecture":
              graph_architecture.Architecture(
                  node_list=[
                      graph_architecture.Node(1, input_indices=[0, 1]),
                      graph_architecture.Node(1, input_indices=[2]),
                      graph_architecture.Node(1, input_indices=[-2, -1]),
                      graph_architecture.Node(1, input_indices=[-1]),
                  ],
                  input_keys=[
                      "input_layer_a", "input_layer_b", "input_layer_c"
                  ]),
      },
  )
  def test_set_get_architecture(self, architecture):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      directory = self.get_temp_dir()
      with self.test_session(graph=tf.Graph()) as sess:
        architecture.save_to_graph()
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        saver.save(sess, directory + "/ckpt")

      output_architecture = graph_architecture.restore_from_checkpoint(
          directory)
      self.assertAllEqual(output_architecture, architecture)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "cnn",
          "input_shape": [1, 32, 32, 3],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(1),  # convolution
                  graph_architecture.Node(3),  # convolution
                  graph_architecture.Node(34),  # flatten plate
              ]),
          "dropout":
              -1,
          "logits_dimension":
              10,
          "expected_logits": [[
              -0.011483, 0.038699, -0.047852, 0.063045, 0.08866, -0.00095,
              -0.017504, -0.06242, 0.06376, -0.092758
          ]]
      },
      {
          "testcase_name":
              "cnn_dropout",
          "input_shape": [1, 32, 32, 3],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(1),  # convolution
                  graph_architecture.Node(3),  # convolution
                  graph_architecture.Node(34),  # flatten plate
              ]),
          "dropout":
              0.1,
          "logits_dimension":
              10,
          "expected_logits": [[
              0.146662, -0.014685, -0.09553, -0.00853, -0.081524, 0.103516,
              0.030273, 0.036939, 0.013888, 0.073447
          ]],
      },
      {
          "testcase_name":
              "cnn_skip_connection",
          "input_shape": [1, 32, 32, 3],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(1),  # convolution
                  graph_architecture.Node(3),  # convolution
                  graph_architecture.Node(
                      34,  # flatten plate
                      input_indices=[-2, -1]),
              ]),
          "dropout":
              -1,
          "logits_dimension":
              10,
          "expected_logits": [[
              0.021225, 0.046355, 0.114048, 0.083442, 0.199439, 0.161184,
              0.066445, -0.009043, 0.006886, -0.091054
          ]]
      },
      {
          "testcase_name":
              "dnn",
          "input_shape": [1, 10],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(14),  # fully-connected
                  graph_architecture.Node(14),  # fully-connected
              ]),
          "dropout":
              -1,
          "logits_dimension":
              10,
          "expected_logits": [[
              -0.18706, 0.39722, -0.4535, -0.11525, 0.16784, -0.16267, -0.37001,
              0.1183, -0.22078, -0.1590
          ]]
      },
      {
          "testcase_name":
              "dnn_dropout",
          "input_shape": [1, 10],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(14),  # fully-connected
                  graph_architecture.Node(14),  # fully-connected
              ]),
          "dropout":
              0.1,
          "logits_dimension":
              10,
          "expected_logits": [[
              -0.4340, -0.6529, -0.73601, -0.0029473, -0.09244, -0.40827, 0.319,
              -0.17377, 0.36107, 0.2386
          ]]
      },
      {
          "testcase_name":
              "dnn_skip_connection",
          "input_shape": [1, 10],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(14),  # fully-connected
                  graph_architecture.Node(
                      14,  # fully-connected
                      input_indices=[-2, -1]),
              ]),
          "dropout":
              -1,
          "logits_dimension":
              10,
          "expected_logits": [[
              -0.03144, 0.28124, -0.28010, 0.09960, 0.28147, -0.08596, 0.14614,
              0.18214, -0.13887, -0.73174
          ]]
      },
      {
          "testcase_name":
              "rnn_all_activations",
          "input_shape": [1, 4, 20],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(56),  # LSTM
                  graph_architecture.Node(56),  # LSTM
              ]),
          "dropout":
              -1,
          "logits_dimension":
              3,
          "expected_logits": [[[-0.002586, -0.0102, 0.001244],
                               [-0.014313, -0.000468, 0.006175],
                               [-0.029063, 0.007654, 0.008976],
                               [-0.046262, 0.015285, 0.015905]]]
      },
      {
          "testcase_name":
              "rnn_last_activations",
          "input_shape": [4, 4, 20],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(56),  # LSTM
                  graph_architecture.Node(56),  # LSTM
              ]),
          "dropout":
              -1,
          "logits_dimension":
              3,
          "expected_logits": [[-0.002586, -0.0102, 0.001244],
                              [-0.023986, 0.006879, 0.007942],
                              [-0.028532, -0.006273, -0.005542],
                              [-0.028763, 0.017457, 0.016798]],
          "lengths": [1, 2, 3, 4],
      },
      {
          "testcase_name":
              "rnn_last_activations_no_lengths",
          "input_shape": [1, 4, 5],
          "phoenix_spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS),
          "architecture":
              graph_architecture.Architecture([
                  graph_architecture.Node(56),  # LSTM
                  graph_architecture.Node(56),  # LSTM
              ]),
          "dropout":
              -1,
          "logits_dimension":
              3,
          "expected_logits": [[[0.002981, -0.005376, -0.005888],
                               [0.006844, -0.007068, -0.009205],
                               [0.012061, -0.009141, -0.014098],
                               [0.016091, -0.009119, -0.016839]]],
      },
  )
  def test_construct_network(self,
                             input_shape,
                             phoenix_spec,
                             architecture,
                             dropout,
                             logits_dimension,
                             expected_logits,
                             lengths=None):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      tf.random.set_seed(1234)
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, shape=input_shape, name="input")
      logits_spec = architecture.construct_tower(
          phoenix_spec=phoenix_spec,
          input_tensor=input_tensor,
          is_training=True,
          lengths=lengths,
          logits_dimension=logits_dimension,
          is_frozen=False,
          dropout_rate=dropout)

      np.random.seed(42)
      test_input = np.random.random(input_shape)

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        logits_val = sess.run(
            logits_spec.logits, feed_dict={input_tensor: test_input})

      self.assertAllClose(expected_logits, logits_val, rtol=1e-3)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "same_graph_snapshotting",
          "new_architecture":
              graph_architecture.Architecture(
                  node_list=[
                      graph_architecture.Node(1),  # convolution
                      graph_architecture.Node(3),  # convolution
                      graph_architecture.Node(34),  # flatten plate
                  ],
                  tower_name="test_tower"),
          "expected_output": [
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/kernel",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/bias",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/gamma",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/beta",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_variance",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/conv2d/kernel",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/conv2d/bias",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/gamma",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/beta",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/moving_mean",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/moving_variance",
              "Phoenix/test_tower/last_dense_1334/dense/kernel",
              "Phoenix/test_tower/last_dense_1334/dense/bias"
          ],
          "new_tower_name":
              "test_tower",
      },
      {
          "testcase_name":
              "same_graph_snapshotting_new_towername",
          "new_architecture":
              graph_architecture.Architecture(
                  node_list=[
                      graph_architecture.Node(1),  # convolution
                      graph_architecture.Node(3),  # convolution
                      graph_architecture.Node(34),  # flatten plate
                  ],
                  tower_name="test_tower_2"),
          "expected_output": [
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/kernel",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/bias",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/gamma",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/beta",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_variance",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/conv2d/kernel",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/conv2d/bias",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/gamma",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/beta",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/moving_mean",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/batch_normalization/moving_variance",
              "Phoenix/test_tower/last_dense_1334/dense/kernel",
              "Phoenix/test_tower/last_dense_1334/dense/bias"
          ],
          "new_tower_name":
              "test_tower_2",
      },
      {
          "testcase_name":
              "changing_second",
          "new_architecture":
              graph_architecture.Architecture(
                  node_list=[
                      graph_architecture.Node(1),  # convolution
                      graph_architecture.Node(2),  # different convolution
                      graph_architecture.Node(34),  # flatten plate
                  ],
                  tower_name="test_tower"),
          "expected_output": [
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/kernel",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/conv2d/bias",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/gamma",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/beta",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/batch_normalization/moving_variance"
          ],
          "new_tower_name":
              "test_tower",
      })
  def test_init_variables(self, new_architecture, expected_output,
                          new_tower_name):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      directory = self.get_temp_dir()
      architecture = graph_architecture.Architecture(
          node_list=[
              graph_architecture.Node(1),  # convolution
              graph_architecture.Node(3),  # convolution
              graph_architecture.Node(34),  # flatten plate
          ],
          tower_name="test_tower")
      phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor = tf.zeros([100, 32, 32, 3])
        _ = architecture.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            is_frozen=False,
            dropout_rate=None)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        saver.save(sess, directory + "/ckpt")

      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor = tf.zeros([100, 32, 32, 3])
        _ = new_architecture.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            is_frozen=False,
            dropout_rate=None)
        snapshotting_variables = architecture_utils.init_variables(
            tf.train.latest_checkpoint(directory), "Phoenix/test_tower",
            "Phoenix/{}".format(new_tower_name))
        self.assertCountEqual(snapshotting_variables, expected_output)


if __name__ == "__main__":
  tf.test.main()
