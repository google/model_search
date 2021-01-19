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
              -0.1481, 0.3328, 0.3028, 0.5652, 0.6860, 0.06171, 0.09998,
              -0.2622, 0.2186, -0.1322
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
              0.7001, -0.06655, -0.1711, 0.1274, -0.8175, 0.2932, 0.06242,
              0.2182, -0.06626, 0.7882
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
              -0.1235, 0.3672, 0.1316, 0.8520, 0.4459, -0.1585, 0.2064, -0.1511,
              0.3595, -0.1060
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
              -0.1267, 0.4418, -0.6564, -0.1298, 0.2320, -0.2099, -0.3553,
              0.2380, -0.3492, 0.03639
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
              -0.5352, -0.7152, -0.6324, -0.1227, -0.1643, -0.2359, 0.1793,
              -0.2127, 0.4157, 0.2386
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
              -0.01021, 0.1801, -0.2899, 0.1427, 0.2179, -0.1162, 0.3315,
              0.06475, -0.08932, -0.7199
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
          "expected_logits": [[[0.006536, 0.001960, -0.002512],
                               [0.006208, -0.003307, 0.005386],
                               [0.005239, -0.01759, 0.01820],
                               [0.007334, -0.02855, 0.02496]]]
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
          "expected_logits": [[0.006536, 0.001960, -0.002512],
                              [-0.007125, -0.008556, 0.001411],
                              [0.02144, -0.004181, 0.001058],
                              [0.007763, -0.02296, 0.006935]],
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
          "expected_logits": [[[-9.769e-05, -1.403e-03, -3.220e-03],
                               [2.679e-03, -4.195e-03, -8.697e-03],
                               [3.288e-03, -2.546e-03, -1.710e-02],
                               [5.260e-03, -7.602e-04, -2.394e-02]]],
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
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/"
              "Conv/biases",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/"
              "Conv/weights",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/"
              "Conv/biases",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/"
              "Conv/weights",
              "Phoenix/test_tower/last_dense_1334/dense/bias",
              "Phoenix/test_tower/last_dense_1334/dense/kernel",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "beta",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_variance",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "moving_variance",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "beta",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "moving_mean",
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
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/"
              "Conv/biases",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/"
              "Conv/weights",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/"
              "Conv/biases",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/"
              "Conv/weights",
              "Phoenix/test_tower/last_dense_1334/dense/bias",
              "Phoenix/test_tower/last_dense_1334/dense/kernel",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "beta",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_variance",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "moving_variance",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "beta",
              "Phoenix/test_tower/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/"
              "moving_mean",
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
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/"
              "Conv/weights",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/biases",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_mean",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "moving_variance",
              "Phoenix/test_tower/1_FIXED_CHANNEL_CONVOLUTION_16_1/BatchNorm/"
              "beta"
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
  tf.enable_v2_behavior()
  tf.test.main()
