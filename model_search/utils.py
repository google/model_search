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
"""Useful utils for tensorflow."""

import tensorflow.compat.v2 as tf


# Adapted from the deprecated DynamicRNNEstimator in :
# tensorflow/contrib/learn/python/learn/estimators/dynamic_rnn_estimator.py
def last_activations_in_sequence(activations, sequence_lengths=None):
  """Selects the nth set of activations for each n in `sequence_length`.

  Returns a `Tensor` of shape `[batch_size, k]`. If `sequence_length` is not
  `None`, then `output[i, :] = activations[i, sequence_length[i] - 1, :]`. If
  `sequence_length` is `None`, then `output[i, :] = activations[i, -1, :]`.

  Args:
    activations: A `Tensor` with shape `[batch_size, padded_length, k]`.
    sequence_lengths: A `Tensor` with shape `[batch_size]` or `None`.

  Returns:
    A `Tensor` of shape `[batch_size, k]`.
  """
  activations_shape = tf.shape(input=activations)
  batch_size = activations_shape[0]
  padded_length = activations_shape[1]
  input_dim = activations_shape[2]
  if sequence_lengths is None:
    sequence_lengths = padded_length
  else:
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
  reshaped_activations = tf.reshape(activations, [-1, input_dim])
  indices = tf.range(batch_size) * padded_length + sequence_lengths - 1
  last_activations = tf.gather(reshaped_activations, indices)
  last_activations.set_shape(
      [activations.get_shape()[0],
       activations.get_shape()[2]])

  return last_activations
