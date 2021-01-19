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
"""Tests for model_search.meta.transfer_learning."""

import os
from absl.testing import parameterized

from model_search import test_utils
from model_search.meta import transfer_learning
from model_search.metadata import trial as trial_module
import tensorflow.compat.v2 as tf


class TransferLearningTest(parameterized.TestCase, tf.test.TestCase):

  def _create_previous_trials(self, root_dir, shapes, dtypes, values):
    """Creates checkpoints for previous trials and returns num created."""
    assert len(shapes) == len(dtypes) == len(values)

    for i, (shape, dtype, value) in enumerate(zip(shapes, dtypes, values)):
      with self.test_session(graph=tf.Graph()) as sess:
        var = tf.compat.v1.get_variable(
            "var",
            shape=shape,
            dtype=dtype,
            initializer=tf.compat.v1.zeros_initializer)
        var = var.assign(value)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(var)
        saver.save(sess, "{}/{}/ckpt".format(root_dir, i))
    return len(shapes)

  def test_transfer_learning_hook_checkpoint_exists(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      expected = 4
      root_dir = self.get_temp_dir()
      n_prev = self._create_previous_trials(
          root_dir,
          shapes=[()] * 3,
          dtypes=[tf.float32] * 3,
          values=list(range(3)))
      # Create a checkpoint in the current trial directory.
      with self.test_session(graph=tf.Graph()) as sess:
        var = tf.compat.v1.get_variable(
            "var",
            shape=(),
            dtype=tf.float32,
            initializer=tf.compat.v1.zeros_initializer)
        var = var.assign_add(expected)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(var)
        saver.save(sess, "{}/{}/ckpt".format(root_dir, n_prev))

      # Assert that hook does not warm start variables if a checkpoint exists.
      with self.test_session(graph=tf.Graph()) as sess:
        var = tf.compat.v1.get_variable(
            "var",
            shape=(),
            dtype=tf.float32,
            initializer=tf.compat.v1.zeros_initializer)
        saver = tf.compat.v1.train.Saver()
        trials = []
        for i in range(3):
          trials.append(
              test_utils.trial_with_dir(
                  dir_value=os.path.join(root_dir, str(i)), id=i))
        hook = transfer_learning.UniformAverageTransferLearningHook(
            vars_to_warm_start=[var],
            current_trial_id=n_prev,
            completed_trials=trials,
            discount_factor=1.,
            max_completed_trials=1000000,
            model_dir=os.path.join(root_dir, str(n_prev)))
        hook.begin()
        saver.restore(sess, "{}/{}/ckpt".format(root_dir, n_prev))
        hook.after_create_session(sess, None)
        actual = sess.run(var)
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, sum(range(3)) / 3)

  @parameterized.named_parameters(
      {
          "testcase_name": "mismatched_dtype",
          "shapes": [(), (), ()],
          "dtypes": [tf.int32, tf.float32, tf.float32],
          "values": [1, 2, 3],
          "expected": 2.5
      }, {
          "testcase_name": "mismatched_shape",
          "shapes": [(), (), (1,)],
          "dtypes": [tf.float32, tf.float32, tf.float32],
          "values": [1, 2, [3]],
          "expected": 1.5
      }, {
          "testcase_name": "all_match",
          "shapes": [(), (), ()],
          "dtypes": [tf.float32, tf.float32, tf.float32],
          "values": [1, 2, 3],
          "expected": 2
      }, {
          "testcase_name": "one_value",
          "shapes": [()],
          "dtypes": [tf.float32],
          "values": [1],
          "expected": 1
      }, {
          "testcase_name": "rank_gt_zero",
          "shapes": [(2, 2), (2, 2)],
          "dtypes": [tf.float32, tf.float32],
          "values": [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
          "expected": [[1.5, 1.5], [1.5, 1.5]]
      }, {
          "testcase_name": "discount_factor",
          "shapes": [(2, 2), (2, 2)],
          "dtypes": [tf.float32, tf.float32],
          "values": [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
          "expected": [[1.305, 1.305], [1.305, 1.305]],
          "discount_factor": .9
      }, {
          "testcase_name": "max_completed_trials",
          "shapes": [(2, 2), (2, 2), (2, 2), (2, 2)],
          "dtypes": [tf.float32, tf.float32, tf.float32, tf.float32],
          "values": [[[4, 4], [4, 4]], [[3, 3], [3, 3]], [[1, 1], [1, 1]],
                     [[2, 2], [2, 2]]],
          "expected": [[1.5, 1.5], [1.5, 1.5]],
          "max_completed_trials": 2,
      })
  def test_uniform_average_transfer_learning_hook(self,
                                                  shapes,
                                                  dtypes,
                                                  values,
                                                  expected,
                                                  discount_factor=1.,
                                                  max_completed_trials=1000000):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      root_dir = self.get_temp_dir()
      n_prev = self._create_previous_trials(root_dir, shapes, dtypes, values)

      # Warm start the current trial for transfer learning and check output.
      with self.test_session(graph=tf.Graph()) as sess:
        var = tf.compat.v1.get_variable(
            "var",
            shape=shapes[0],
            dtype=tf.float32,
            initializer=tf.compat.v1.zeros_initializer)
        trials = []
        for i in range(n_prev):
          trials.append(
              test_utils.trial_with_dir(
                  dir_value=os.path.join(root_dir, str(i)), id=i))
        hook = transfer_learning.UniformAverageTransferLearningHook(
            vars_to_warm_start=[var],
            current_trial_id=n_prev,
            completed_trials=trials,
            discount_factor=discount_factor,
            max_completed_trials=max_completed_trials,
            model_dir=os.path.join(root_dir, str(n_prev)))
        hook.begin()
        sess.run(tf.compat.v1.global_variables_initializer())
        hook.after_create_session(sess, None)
        actual = sess.run(var)
        self.assertAllClose(actual, expected)

  @parameterized.named_parameters(
      {
          "testcase_name": "one_value",
          "shapes": [()],
          "dtypes": [tf.float32],
          "values": [1],
          "losses": [.1],
          "expected": 1
      }, {
          "testcase_name": "rank_gt_zero",
          "shapes": [(2, 2), (2, 2)],
          "dtypes": [tf.float32, tf.float32],
          "values": [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
          "losses": [.1, 2],
          "expected": [[.565054, .565054], [.565054, .565054]]
      }, {
          "testcase_name": "max_completed_trials",
          "shapes": [(2, 2), (2, 2), (2, 2), (2, 2)],
          "dtypes": [tf.float32, tf.float32, tf.float32, tf.float32],
          "values": [[[1, 1], [1, 1]], [[3, 3], [3, 3]], [[4, 4], [4, 4]],
                     [[2, 2], [2, 2]]],
          "losses": [.1, 100, 200, 2],
          "expected": [[.565054, .565054], [.565054, .565054]],
          "max_completed_trials": 2
      })
  def test_loss_weighted_transfer_learning_hook(self,
                                                shapes,
                                                dtypes,
                                                values,
                                                losses,
                                                expected,
                                                max_completed_trials=1000000):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      root_dir = self.get_temp_dir()
      n_prev = self._create_previous_trials(root_dir, shapes, dtypes, values)
      # Warm start the current trial for transfer learning and check output.
      with self.test_session(graph=tf.Graph()) as sess:
        var = tf.compat.v1.get_variable(
            "var",
            shape=shapes[0],
            dtype=tf.float32,
            initializer=tf.compat.v1.zeros_initializer)
        completed_trials = []
        for i, loss in enumerate(losses):
          trial = trial_module.Trial({
              "model_dir": os.path.join(root_dir, str(i)),
              "id": i,
              "final_measurement": {
                  "objective_value": loss
              }
          })
          completed_trials.append(trial)
        hook = transfer_learning.LossWeightedAverageTransferLearningHook(
            vars_to_warm_start=[var],
            current_trial_id=n_prev,
            completed_trials=completed_trials,
            discount_factor=1.,
            max_completed_trials=max_completed_trials,
            model_dir=os.path.join(root_dir, str(n_prev)))
        hook.begin()
        sess.run(tf.compat.v1.global_variables_initializer())
        hook.after_create_session(sess, None)
        actual = sess.run(var)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
