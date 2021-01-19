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
"""Tests for model_search.meta.distillation."""

from absl.testing import parameterized

from model_search.meta import distillation
from model_search.proto import distillation_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf


def _make_distillation_spec(type_):
  output = distillation_spec_pb2.DistillationSpec()
  output.distillation_type = type_
  output.balance_losses_spec.num_ramp_trials = 100
  return output


def _loss_fn(labels, logits, weights=1.0):
  return tf.reduce_mean(
      input_tensor=tf.compat.v1.losses.softmax_cross_entropy(
          onehot_labels=labels, logits=logits, weights=weights))


class DistillationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name':
              'mse_logits',
          'distillation_spec':
              _make_distillation_spec(
                  distillation_spec_pb2.DistillationSpec.MSE_LOGITS),
          'teacher_logits':
              np.array([2., 4., 8]),
          'my_id':
              -1,
          'labels':
              np.array([2., 4., 8]),
          'logits':
              np.array([0., 4., 1.]),
          'expected':
              np.float32((4. + 0 + 49.) / 3.0),
      }, {
          'testcase_name':
              'mse_softmax',
          'distillation_spec':
              _make_distillation_spec(
                  distillation_spec_pb2.DistillationSpec.MSE_SOFTMAX),
          'teacher_logits':
              np.array([2., 4., 8]),
          'my_id':
              -1,
          'labels':
              np.array([2., 4., 8]),
          'logits':
              np.array([0., 4., 1.]),
          'expected':
              np.float32((4. + 0 + 49.) / 3.0),
      }, {
          'testcase_name':
              'cross_entropy',
          'distillation_spec':
              _make_distillation_spec(
                  distillation_spec_pb2.DistillationSpec.CROSS_ENTROPY),
          'teacher_logits':
              np.array([0.2, 0.4, 0.4]),
          'my_id':
              -1,
          'labels':
              np.array([0., 0., 1.]),
          'logits':
              np.array([0., 4., 1.]),
          'expected':
              np.float32(2.065884),
          'weights':
              2.0,
      }, {
          'testcase_name':
              'adaptive',
          'distillation_spec':
              _make_distillation_spec(distillation_spec_pb2.DistillationSpec
                                      .ADAPTIVELY_BALANCE_LOSSES),
          'teacher_logits':
              np.array([0.2, 0.4, 0.4]),
          'my_id':
              50,
          'labels':
              np.array([0., 0., 1.]),
          'logits':
              np.array([0., 4., 1.]),
          'expected':
              np.float32(5.13176774979),
          'weights':
              2.0,
      })
  def test_regression_loss_fns(self,
                               distillation_spec,
                               teacher_logits,
                               my_id,
                               labels,
                               logits,
                               expected,
                               weights=np.float32([2., 2., 2.])):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      loss_fn = distillation.get_distillation_loss_fn(teacher_logits,
                                                      distillation_spec, my_id,
                                                      _loss_fn)

      with self.test_session() as sess:
        actual_unweighted = sess.run(loss_fn(labels=labels, logits=logits))
        actual_weighted = sess.run(
            loss_fn(labels=labels, logits=logits, weights=weights))
      self.assertAllClose(actual_unweighted, expected)
      self.assertAllClose(actual_weighted, expected * 2)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
