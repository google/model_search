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
"""Tests for model_search.loss_fns."""

from absl.testing import parameterized

from model_search import loss_fns
import numpy as np
import tensorflow.compat.v2 as tf


# TODO(b/172564129): Add classification tests as well.
class LossFnsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'mse',
          'loss_fn_factory': loss_fns.make_regression_loss_fn,
          'labels': np.array([2., 4., 8]),
          'logits': np.array([0., 4., 1.]),
          'expected': np.float32((4. + 0 + 49.) / 3.0),
      }, {
          'testcase_name':
              'mae',
          'labels':
              np.array([11., 12., 13.]),
          'loss_fn_factory':
              loss_fns.make_regression_absolute_difference_loss_fn,
          'logits':
              np.array([1., 2., 3.]),
          'expected':
              np.float32((10. + 10. + 10.) / 3.0),
      }, {
          'testcase_name': 'msle',
          'loss_fn_factory': loss_fns.make_regression_logarithmic_loss_fn,
          'labels': np.array([0., np.e - 1, np.e**2 - 1]),
          'logits': np.array([1., 3., 5.]),
          'expected': np.float32((1. + 4 + 9.) / 3.0),
      })
  def test_regression_loss_fns(self, loss_fn_factory, labels, logits, expected):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      loss_fn = loss_fn_factory()
      with self.test_session() as sess:
        actual_unweighted = sess.run(loss_fn(labels, logits))
        weights = np.ones(len(labels)) * 2
        actual_weighted = sess.run(loss_fn(labels, logits, weights))
      self.assertAllClose(actual_unweighted, expected)
      self.assertAllClose(actual_weighted, expected * 2)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
