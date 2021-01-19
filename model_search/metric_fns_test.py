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

# List as: python3
"""Tests for model_search.metric_fns."""

from absl.testing import parameterized
from model_search import metric_fns

import numpy as np
import tensorflow.compat.v2 as tf


class MetricFnsTest(tf.test.TestCase, parameterized.TestCase):

  # pylint: disable=g-long-lambda
  # tf.constant must be called in a lambda, otherwise the Op would be created
  # in a different graph from where it would be used, which is not allowed.
  @parameterized.named_parameters(
      {
          'testcase_name':
              'int64_label_single_task',
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'predictions': tf.constant([1, 0, 0, 0, 0], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy': np.float32(0.2)
          }
      }, {
          'testcase_name':
              'string_label_single_task',
          'label_vocabulary': ['A', 'B', 'C', 'D', 'E'],
          'labels_fn':
              lambda: tf.constant(['A', 'B', 'C', 'D', 'E'], dtype=tf.string),
          'predictions_fn':
              lambda: {
                  'predictions': tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy': np.float32(0.2)
          }
      }, {
          'testcase_name':
              'string_label_no_vocab_single_task',
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant(['A', 'B', 'C', 'D', 'E'], dtype=tf.string),
          'predictions_fn':
              lambda: {
                  'predictions': tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
              },
          'expected_metric_dict': {}
      }, {
          'testcase_name':
              'int64_label_multi_task',
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: {
                  'task_a': tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
                  'task_b': tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
              },
          'predictions_fn':
              lambda: {
                  'predictions':
                      tf.constant([1, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_a':
                      tf.constant([1, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_b':
                      tf.constant([1, 1, 1, 0, 0], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy/task_a': np.float32(0.2),
              'accuracy/task_b': np.float32(0.6),
          },
      }, {
          'testcase_name':
              'string_label_multi_task',
          'label_vocabulary': {
              'task_a': ['A', 'B', 'C', 'D', 'E'],
              'task_b': ['F', 'G', 'H', 'I', 'J'],
          },
          'labels_fn':
              lambda: {
                  'task_a':
                      tf.constant(['A', 'B', 'C', 'D', 'E'], dtype=tf.string),
                  'task_b':
                      tf.constant(['F', 'G', 'H', 'I', 'J'], dtype=tf.string),
              },
          'predictions_fn':
              lambda: {
                  'predictions':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_a':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_b':
                      tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy/task_a': np.float32(0.2),
              'accuracy/task_b': np.float32(0.2),
          },
      }, {
          'testcase_name':
              'mixed_label_multi_task',
          'label_vocabulary': {
              'task_a': ['A', 'B', 'C', 'D', 'E'],
          },
          'labels_fn':
              lambda: {
                  'task_a':
                      tf.constant(['A', 'B', 'C', 'D', 'E'], dtype=tf.string),
                  'task_b':
                      tf.constant([1, 1, 0, 0, 0], dtype=tf.int64),
              },
          'predictions_fn':
              lambda: {
                  'predictions':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_a':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_b':
                      tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy/task_a': np.float32(0.2),
              'accuracy/task_b': np.float32(0.4),
          },
      }, {
          'testcase_name':
              'string_no_vocab_multi_task',
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: {
                  'task_a':
                      tf.constant(['A', 'B', 'C', 'D', 'E'], dtype=tf.string),
                  'task_b':
                      tf.constant([1, 1, 0, 0, 0], dtype=tf.int64),
              },
          'predictions_fn':
              lambda: {
                  'predictions':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_a':
                      tf.constant([0, 0, 0, 0, 0], dtype=tf.int64),
                  'predictions/task_b':
                      tf.constant([1, 1, 1, 1, 1], dtype=tf.int64),
              },
          'expected_metric_dict': {
              'accuracy/task_b': np.float32(0.4),
          },
      })
  # pylint: enable=g-long-lambda
  def test_make_accuracy_metric_fn(self, label_vocabulary, labels_fn,
                                   predictions_fn, expected_metric_dict):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      metric_fn = metric_fns.make_accuracy_metric_fn(label_vocabulary)
      actual_metric_dict = metric_fn(labels_fn(), predictions_fn())
      with self.test_session() as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        sess.run(tf.compat.v1.initializers.tables_initializer())
        actual_metric_dict_val = sess.run(actual_metric_dict)
      actual_metric_dict_val_clean = {
          metric_key: metric_val[1]
          for metric_key, metric_val in actual_metric_dict_val.items()
      }
      self.assertEqual(expected_metric_dict, actual_metric_dict_val_clean)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          'testcase_name':
              'roc_perfect',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_roc': np.float32(1.0)
          }
      }, {
          'testcase_name':
              'roc_perfect_vocab',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary': ['ZERO', 'ONE'],
          'labels_fn':
              lambda: tf.constant(['ONE', 'ZERO'], dtype=tf.string),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_roc': np.float32(1.0)
          }
      }, {
          'testcase_name':
              'roc_random',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_roc': np.float32(0.5)
          }
      }, {
          'testcase_name':
              'pr_perfect',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_pr': np.float32(1.0)
          }
      }, {
          'testcase_name':
              'pr_perfect_vocab',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary': ['ZERO', 'ONE'],
          'labels_fn':
              lambda: tf.constant(['ONE', 'ZERO'], dtype=tf.string),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_pr': np.float32(1.0)
          }
      }, {
          'testcase_name':
              'pr_random',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32),
              },
          'expected_metric_dict': {
              'auc_pr': np.float32(0.5)
          }
      })
  # pylint: enable=g-long-lambda
  def test_auc_metric_fn(self, metric_fn_factory, label_vocabulary, labels_fn,
                         predictions_fn, expected_metric_dict):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      metric_fn = metric_fn_factory(label_vocabulary)
      actual_metric_dict = metric_fn(labels_fn(), predictions_fn())
      with self.test_session() as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        sess.run(tf.compat.v1.initializers.tables_initializer())
        actual_metric_dict_val = sess.run(actual_metric_dict)
      actual_metric_dict_val_clean = {
          metric_key: metric_val[1]
          for metric_key, metric_val in actual_metric_dict_val.items()
      }
      self.assertAllClose(expected_metric_dict, actual_metric_dict_val_clean)

    # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          'testcase_name':
              'roc_multi_task',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: {
                  'task_a': tf.constant([1, 0], dtype=tf.int64),
                  'task_b': tf.constant([1, 0], dtype=tf.int64),
              },
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
                  'probabilities/task_a':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
                  'probabilities/task_b':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'exception_class':
              NotImplementedError,
      }, {
          'testcase_name':
              'roc_rank3_prob_tensor',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[[0.5, 0.5], [0.5, 0.5]],
                                   [[0.5, 0.5], [0.5, 0.5]]],
                                  dtype=tf.float32),
              },
          'exception_class':
              ValueError,
      }, {
          'testcase_name':
              'roc_prob_tensor_3_classes',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([2, 1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                                  dtype=tf.float32),
              },
          'exception_class':
              ValueError,
      }, {
          'testcase_name':
              'pr_multi_task',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: {
                  'task_a': tf.constant([1, 0], dtype=tf.int64),
                  'task_b': tf.constant([1, 0], dtype=tf.int64),
              },
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
                  'probabilities/task_a':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
                  'probabilities/task_b':
                      tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32),
              },
          'exception_class':
              NotImplementedError,
      }, {
          'testcase_name':
              'pr_rank3_prob_tensor',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[[0.5, 0.5], [0.5, 0.5]],
                                   [[0.5, 0.5], [0.5, 0.5]]],
                                  dtype=tf.float32),
              },
          'exception_class':
              ValueError,
      }, {
          'testcase_name':
              'pr_prob_tensor_3_classes',
          'metric_fn_factory':
              metric_fns.make_auc_pr_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant([2, 1, 0], dtype=tf.int64),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                                  dtype=tf.float32),
              },
          'exception_class':
              ValueError,
      }, {
          'testcase_name':
              'roc_string_label_no_vocab',
          'metric_fn_factory':
              metric_fns.make_auc_roc_metric_fn,
          'label_vocabulary':
              None,
          'labels_fn':
              lambda: tf.constant(['ONE', 'ZERO'], dtype=tf.string),
          'predictions_fn':
              lambda: {
                  'probabilities':
                      tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
              },
          'exception_class':
              ValueError,
      })
  # pylint: enable=g-long-lambda
  def test_auc_metric_fn_error(self, metric_fn_factory, label_vocabulary,
                               labels_fn, predictions_fn, exception_class):
    with self.assertRaises(exception_class):
      metric_fn = metric_fn_factory(label_vocabulary)
      metric_fn(labels_fn(), predictions_fn())

  def test_create_num_parameters_metric_fn_no_tower(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      _ = tf.compat.v1.get_variable(
          name='w', shape=[10, 2], dtype=tf.float32, trainable=True)
      _ = tf.compat.v1.get_variable(
          name='b', shape=[2], dtype=tf.float32, trainable=True)

      metric_fn = metric_fns.create_num_parameters_metric_fn(None)
      metrics_dict = metric_fn(None, None)
      with self.test_session() as sess:
        self.assertEqual(22, sess.run(metrics_dict['num_parameters'][1]))

  def test_create_num_parameters_metric_fn_with_tower(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      _ = tf.compat.v1.get_variable(
          name='Phoenix/name', shape=[10, 2], dtype=tf.float32, trainable=True)
      _ = tf.compat.v1.get_variable(
          name='b', shape=[2], dtype=tf.float32, trainable=True)

      metric_fn = metric_fns.create_num_parameters_metric_fn('name')
      metrics_dict = metric_fn(None, None)
      with self.test_session() as sess:
        self.assertEqual(20, sess.run(metrics_dict['num_parameters'][1]))

  def test_combine_metric_fns(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():

      def metric_fn_1(labels, predictions, weights=None):
        del labels
        del predictions
        del weights
        one = tf.constant(1, dtype=tf.int32)
        return {'foo1': (one, one)}

      def metric_fn_2(labels, predictions, weights=None):
        del labels
        del predictions
        del weights
        two = tf.constant(2, dtype=tf.int32)
        return {'foo2': (two, two)}

      metric_fn_combined = metric_fns.combine_metric_fns(
          [metric_fn_1, metric_fn_2])
      metrics_dict = metric_fn_combined(None, None)

      with self.test_session() as sess:
        self.assertEqual(1, sess.run(metrics_dict['foo1'][1]))
        self.assertEqual(2, sess.run(metrics_dict['foo2'][1]))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
