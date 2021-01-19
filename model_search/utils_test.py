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
"""Tests for model_search.utils."""

from absl import logging
from model_search import utils
import numpy as np
import tensorflow.compat.v2 as tf


INPUT_TENSOR = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
                [8, 8], [9, 9]]


class UtilsTest(tf.test.TestCase):

  def test_last_activations_in_sequence(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.constant(INPUT_TENSOR)
      input_tensor = tf.expand_dims(input_tensor, axis=0)
      batch = tf.tile(input_tensor, tf.constant([5, 1, 1]))
      logging.info(batch)
      lengths = tf.constant([2, 4, 6, 8, 10])
      output = utils.last_activations_in_sequence(batch, lengths)
      with self.test_session() as sess:
        output = sess.run(output)
      self.assertAllEqual(output,
                          np.array([[1, 1], [3, 3], [5, 5], [7, 7], [9, 9]]))

  def test_last_activations_in_sequence_with_none(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.constant(INPUT_TENSOR)
      input_tensor = tf.expand_dims(input_tensor, axis=0)
      batch = tf.tile(input_tensor, tf.constant([5, 1, 1]))
      logging.info(batch)
      output = utils.last_activations_in_sequence(batch)
      with self.test_session() as sess:
        output = sess.run(output)
      self.assertAllEqual(output,
                          np.array([[9, 9], [9, 9], [9, 9], [9, 9], [9, 9]]))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
