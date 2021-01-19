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

from absl import logging
from model_search.ops import svdf_conv

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf


def get_test_batch_features_and_labels_numpy(input_dim, output_dim):
  input_shape = [2, 7, input_dim]
  output_shape = [2, 7, output_dim]
  input_values = np.arange(
      np.prod(input_shape), dtype=np.float32) / np.prod(input_shape)
  output_values = np.arange(
      np.prod(output_shape), dtype=np.float32) / np.prod(output_shape)
  return input_values.reshape(input_shape), output_values.reshape(output_shape)


def _get_test_svdf_layer_weights():
  """Returns weights for an SvdfCell with following params.

    (units=4, memory_size=3, rank=1, use_bias=True).
  """
  return [
      np.array([[-0.31614766, 0.37929568, 0.27584907, -0.36453721],
                [-0.35801932, 0.22514193, 0.27241215, -0.06950231],
                [0.01112892, 0.12732419, 0.38735834, -0.10957076],
                [-0.09451947, 0.15611194, 0.39319292, -0.03019224],
                [0.39612538, 0.16101542, 0.21615031, 0.30737072]],
               dtype=np.float32),
      np.array([[-0.31614769, 0.37929571, 0.27584907, -0.36453718],
                [-0.35801938, 0.22514194, 0.27241215, -0.06950228],
                [0.01112869, 0.12732419, 0.38735834, -0.10957073]],
               dtype=np.float32),
      np.array([-0.00316226, 0.00316225, -0.00316227, 0.00316227],
               dtype=np.float32)
  ]


def _get_test_svdf_expected_output():
  """Returns output of an svdf layer with the following params.

    Note: the values are obtained from _get_svdf_output_using_numpy computation.
  """
  return np.array([
      [[-0.00300881, 0.00605831, 0.01394408, 0.00183136],
       [-0.0082326, 0.0207185, 0.06872095, 0.00307236],
       [-0.00363621, 0.05575278, 0.15371153, 0.00205239],
       [0.01348117, 0.11057685, 0.25696135, 0.01239775],
       [0.03059855, 0.16540094, 0.36021116, 0.02274311],
       [0.04771594, 0.22022502, 0.46346098, 0.03308846],
       [0.06483334, 0.27504909, 0.56671083, 0.04343383]],
      [[-0.00501995, 0.07283279, 0.31317118, 0.01642792],
       [0.05445613, 0.20556745, 0.57838139, 0.02692774],
       [0.11618549, 0.43952131, 0.87646019, 0.07446991],
       [0.13330287, 0.49434543, 0.97971004, 0.08481527],
       [0.15042025, 0.54916948, 1.08295989, 0.09516063],
       [0.16753764, 0.60399354, 1.18620968, 0.10550598],
       [0.18465498, 0.65881759, 1.28945935, 0.11585134]],
  ],
                  dtype=np.float32)


def _get_svdf_output_using_numpy(input_values, weights):
  """Compute svdf output using numpy for expected values.

  NOTE: Use this helper function as a way to verify computations in tensorflow.
  This function assumes linear activation for svdf and projection.
  Args:
    input_values: ndarray (shape=[batch_size, sequence_length, input_dim]).
    weights: A list of 3 ndarrays from _get_test_svdf_layer_weights().

  Returns:
    output_values: ndarray with sequence of output values from svdf layer.
  """

  def _compute_single_sequence_output(sequence_input, weights):
    feature_activations = []
    for i in range(input_values.shape[1]):
      feature_activations.append(np.dot(sequence_input[i, :], weights[0]))
    memory_size = weights[1].shape[0]
    feature_activations = [np.zeros(weights[1].shape[1])
                          ] * (memory_size - 1) + feature_activations
    time_activations = []
    for i in range(memory_size - 1, len(feature_activations)):
      time_activations.append(weights[2] + np.sum(
          np.vstack(feature_activations[i - memory_size + 1:i + 1]) *
          weights[1],
          axis=0))
    return time_activations

  outputs = np.zeros(
      (input_values.shape[0], input_values.shape[1], weights[2].shape[0]))
  for b in range(input_values.shape[0]):
    outputs[b, :, :] = _compute_single_sequence_output(input_values[b, :, :],
                                                       weights)
  return outputs


class SvdfConvLayerTest(tf.test.TestCase):

  def setUp(self):
    super(SvdfConvLayerTest, self).setUp()
    self.sess = tf.Session()
    tf.keras.backend.set_session(self.sess)

  def tearDown(self):
    tf.keras.backend.clear_session()
    self.sess.close()
    super(SvdfConvLayerTest, self).tearDown()

  def _get_initializer(self):
    return tf.keras.initializers.RandomUniform(minval=-0.4, maxval=0.4, seed=0)

  def _get_optimizer(self):
    return tf.keras.optimizers.RMSprop(
        lr=0.001, rho=0.9, epsilon=None, decay=0.0)

  def test_svdf_conv_layer_accuracy(self):
    # Test non stacked convolutional layer implementation for accuracy.
    initializer = self._get_initializer()
    optimizer = self._get_optimizer()
    input_tensor = tf.keras.Input(shape=(None, 5))
    layer = svdf_conv.SvdfConvLayer(
        units=4,
        memory_size=3,
        rank=1,
        kernel_initializer=initializer,
        bias_initializer="ones",
        activation="linear")
    output = layer(input_tensor)
    model = tf.keras.models.Model(input_tensor, output)
    weights = _get_test_svdf_layer_weights()
    model.layers[1].set_weights(weights)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary(print_fn=logging.info)
    input_values, _ = (
        get_test_batch_features_and_labels_numpy(input_dim=5, output_dim=2))
    logging.info("tf.executing_eagerly: %s", tf.executing_eagerly())
    output_tensor = model(input_values)
    observed_output = output_tensor.numpy()
    expected_output = _get_test_svdf_expected_output()
    self.assertAllClose(observed_output, expected_output)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
