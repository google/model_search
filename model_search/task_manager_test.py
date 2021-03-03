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
"""Tests for model_search.task_manager."""

from absl.testing import parameterized

from model_search import hparam as hp
from model_search import loss_fns
from model_search import task_manager
from model_search.architecture import architecture_utils
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf

from google.protobuf import text_format


def _loss_fn(labels, logits, weights=1.0):
  """Cross entropy loss fn."""

  label_ids = tf.squeeze(labels)

  if label_ids.dtype == tf.float32:
    label_ids = tf.cast(label_ids, 'int32')
  one_hot_labels = tf.one_hot(indices=label_ids, depth=logits.shape[-1])

  return tf.reduce_mean(
      input_tensor=tf.compat.v1.losses.softmax_cross_entropy(
          onehot_labels=one_hot_labels, logits=logits, weights=weights))


def _default_predictions_fn(logits,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            temperature=1.0):
  """Converts logits to predictions dict. Assumes classification."""
  new_logits = logits
  if mode == tf.estimator.ModeKeys.PREDICT and temperature != 1.0:
    temp_const = tf.constant(1 / temperature, name='softmax_temperature_const')
    new_logits = tf.multiply(logits, temp_const, name='softmax_temperature_mul')

  predictions = tf.math.argmax(input=new_logits, axis=-1)
  probabilities = tf.nn.softmax(new_logits)
  log_probabilities = tf.nn.log_softmax(new_logits)

  predictions_dict = {
      'predictions': predictions,
      'probabilities': probabilities,
      'log_probabilities': log_probabilities,
  }
  return predictions_dict


class TaskManagerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'l2_reg',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'l2_regularization': 0.01
          },
          'contains_node': 'l2_weight_loss',
          'not_containing': ['clip_by_global_norm', 'ExponentialDecay']
      }, {
          'testcase_name': 'clipping',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'gradient_max_norm': 3
          },
          'contains_node': 'clip_by_global_norm',
          'not_containing': ['l2_weight_loss', 'ExponentialDecay']
      }, {
          'testcase_name': 'decay',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'exponential_decay_steps': 100,
              'exponential_decay_rate': 0.7
          },
          'contains_node': 'ExponentialDecay',
          'not_containing': ['l2_weight_loss', 'clip_by_global_norm']
      })
  def test_learning_spec(self, learning_rate_spec, contains_node,
                         not_containing):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      features = {'x': tf.zeros([10, 10])}
      loss_fn = loss_fns.make_multi_class_loss_fn()
      model = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels=tf.ones([20], dtype=tf.int32),
          loss_fn=loss_fn,
          mode=tf.estimator.ModeKeys.TRAIN,
          model_directory=self.get_temp_dir(),
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if contains_node in node.name
      ])
      for phrase in not_containing:
        self.assertEmpty([
            node.name
            for node in tf.compat.v1.get_default_graph().as_graph_def().node
            if phrase in node.name
        ])
      self.assertLen(model.predictions, 3)
      self.assertIn('probabilities', model.predictions)
      self.assertIn('log_probabilities', model.predictions)
      self.assertIn('predictions', model.predictions)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'l2_reg',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'l2_regularization': 0.01
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      }, {
          'testcase_name':
              'clipping',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'gradient_max_norm': 3
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      }, {
          'testcase_name':
              'decay',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'exponential_decay_steps': 100,
              'exponential_decay_rate': 0.7
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      })
  def test_learning_spec_on_eval(self, learning_rate_spec, not_containing):
    spec = phoenix_spec_pb2.PhoenixSpec(
        problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
    task_manager_instance = task_manager.TaskManager(spec)
    logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
    logits_spec = architecture_utils.LogitsSpec(logits=logits)
    features = {'x': tf.zeros([10, 10])}
    loss_fn = loss_fns.make_multi_class_loss_fn()
    model = task_manager_instance.create_model_spec(
        features=features,
        params=hp.HParams(optimizer='sgd'),
        learning_rate_spec=learning_rate_spec,
        train_logits_specs=[logits_spec],
        eval_logits_spec=logits_spec,
        labels=tf.ones([20], dtype=tf.int32),
        loss_fn=loss_fn,
        model_directory=self.get_temp_dir(),
        mode=tf.estimator.ModeKeys.EVAL,
        lengths=None,
        use_tpu=False,
        predictions_fn=_default_predictions_fn)
    for phrase in not_containing:
      self.assertEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if phrase in node.name
      ])
    self.assertLen(model.predictions, 3)
    self.assertIn('probabilities', model.predictions)
    self.assertIn('log_probabilities', model.predictions)
    self.assertIn('predictions', model.predictions)
    self.assertNotEqual(model.loss, None)

  @parameterized.named_parameters(
      {
          'testcase_name':
              'l2_reg',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'l2_regularization': 0.01
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      }, {
          'testcase_name':
              'clipping',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'gradient_max_norm': 3
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      }, {
          'testcase_name':
              'decay',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'exponential_decay_steps': 100,
              'exponential_decay_rate': 0.7
          },
          'not_containing':
              ['l2_weight_loss', 'clip_by_global_norm', 'ExponentialDecay']
      })
  def test_learning_spec_on_predict(self, learning_rate_spec, not_containing):
    spec = phoenix_spec_pb2.PhoenixSpec(
        problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
    task_manager_instance = task_manager.TaskManager(spec)
    logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
    logits_spec = architecture_utils.LogitsSpec(logits=logits)
    features = {'x': tf.zeros([10, 10])}
    loss_fn = loss_fns.make_multi_class_loss_fn()
    model = task_manager_instance.create_model_spec(
        features=features,
        params=hp.HParams(optimizer='sgd'),
        learning_rate_spec=learning_rate_spec,
        train_logits_specs=[logits_spec],
        eval_logits_spec=logits_spec,
        labels=tf.ones([20], dtype=tf.int32),
        loss_fn=loss_fn,
        mode=tf.estimator.ModeKeys.PREDICT,
        model_directory=self.get_temp_dir(),
        lengths=None,
        use_tpu=False,
        predictions_fn=_default_predictions_fn)
    for phrase in not_containing:
      self.assertEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if phrase in node.name
      ])
    self.assertLen(model.predictions, 3)
    self.assertIn('probabilities', model.predictions)
    self.assertIn('log_probabilities', model.predictions)
    self.assertIn('predictions', model.predictions)
    self.assertIsNone(model.loss)

  def test_tpu(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      learning_rate_spec = {'learning_rate': 0.001, 'gradient_max_norm': 3}
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      features = {'x': tf.zeros([10, 10])}
      loss_fn = loss_fns.make_multi_class_loss_fn()
      _ = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels=tf.ones([20], dtype=tf.int32),
          model_directory=self.get_temp_dir(),
          loss_fn=loss_fn,
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=True,
          predictions_fn=_default_predictions_fn)
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'CrossReplicaSum' in node.name
      ])

  @parameterized.named_parameters(
      {
          'testcase_name':
              'l2_reg',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'l2_regularization': 0.01
          },
          'contains_node':
              'l2_weight_loss',
          'not_containing': [
              'label1/clip_by_global_norm', 'label1/ExponentialDecay',
              'label2/clip_by_global_norm', 'label2/ExponentialDecay'
          ]
      }, {
          'testcase_name':
              'clipping',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'gradient_max_norm': 3
          },
          'contains_node':
              'clip_by_global_norm',
          'not_containing': [
              'label1/l2_weight_loss', 'label1/ExponentialDecay',
              'label2/l2_weight_loss', 'label2/ExponentialDecay'
          ]
      }, {
          'testcase_name':
              'decay',
          'learning_rate_spec': {
              'learning_rate': 0.001,
              'exponential_decay_steps': 100,
              'exponential_decay_rate': 0.7
          },
          'contains_node':
              'ExponentialDecay',
          'not_containing': [
              'label1/l2_weight_loss', 'label1/clip_by_global_norm',
              'label2/l2_weight_loss', 'label2/clip_by_global_norm'
          ]
      })
  def test_multitask(self, learning_rate_spec, contains_node, not_containing):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      text_format.Merge(
          """
          multi_task_spec {
            label_name: "label1"
            number_of_classes: 10
          }

          multi_task_spec {
            label_name: "label2"
            number_of_classes: 10
          }
      """, spec)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      features = {'x': tf.zeros([10, 10])}
      loss_fn = loss_fns.make_multi_class_loss_fn()
      model = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels={
              'label1': tf.ones([20], dtype=tf.int32),
              'label2': tf.ones([20], dtype=tf.int32)
          },
          loss_fn=loss_fn,
          model_directory=self.get_temp_dir(),
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if contains_node in node.name
      ])
      for phrase in not_containing:
        self.assertEmpty([
            node.name
            for node in tf.compat.v1.get_default_graph().as_graph_def().node
            if phrase in node.name
        ])
      self.assertLen(model.predictions, 3 * (1 + 2))
      self.assertContainsSubset([
          'probabilities',
          'probabilities/label1',
          'probabilities/label2',
          'log_probabilities',
          'log_probabilities/label1',
          'log_probabilities/label2',
          'predictions',
          'predictions/label1',
          'predictions/label2',
      ], model.predictions.keys())

  @parameterized.named_parameters(
      {
          'testcase_name': 'feature_weight_vanilla',
          'is_multitask': False,
          'weight_is_a_feature': False
      }, {
          'testcase_name': 'feature_weight_mutitask',
          'is_multitask': True,
          'weight_is_a_feature': False
      }, {
          'testcase_name': 'feature_weight_in_labels',
          'is_multitask': False,
          'weight_is_a_feature': True
      }, {
          'testcase_name': 'feature_weight_multitask_in_labels',
          'is_multitask': True,
          'weight_is_a_feature': True
      })
  def test_weight_feature(self, is_multitask, weight_is_a_feature):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      learning_rate_spec = {'learning_rate': 0.001, 'gradient_max_norm': 3}
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      labels = tf.ones([20], dtype=tf.int32)
      if is_multitask:
        text_format.Merge(
            """
            multi_task_spec {
              label_name: "label1"
              number_of_classes: 10
              weight_feature_name: "weight1"
              weight_is_a_feature: %s
            }
            multi_task_spec {
              label_name: "label2"
              number_of_classes: 10
              weight_feature_name: "weight2"
              weight_is_a_feature: %s
            }
        """ % (str(weight_is_a_feature), str(weight_is_a_feature)), spec)
        labels = {
            'label1': tf.ones([20], dtype=tf.int32),
            'label2': tf.ones([20], dtype=tf.int32)
        }

      weights = {
          'weight1': tf.constant([2] * 20),
          'weight2': tf.constant([3] * 20)
      }
      features = {'x': tf.zeros([10, 10])}
      if weight_is_a_feature:
        features.update(weights)
      elif isinstance(labels, dict):
        labels.update(weights)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      loss_fn = loss_fns.make_multi_class_loss_fn()

      _ = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels=labels,
          loss_fn=loss_fn,
          model_directory=self.get_temp_dir(),
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)

  @parameterized.named_parameters(
      {
          'testcase_name': 'feature_weight_mutitask',
          'weight_is_a_feature': False
      }, {
          'testcase_name': 'feature_weight_multitask_in_labels',
          'weight_is_a_feature': True
      })
  def test_wrong_dict_weight_feature(self, weight_is_a_feature):
    learning_rate_spec = {'learning_rate': 0.001, 'gradient_max_norm': 3}
    spec = phoenix_spec_pb2.PhoenixSpec(
        problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
    text_format.Merge(
        """
        multi_task_spec {
          label_name: "label1"
          number_of_classes: 10
          weight_feature_name: "weight1"
          weight_is_a_feature: %s
        }
        multi_task_spec {
          label_name: "label2"
          number_of_classes: 10
          weight_feature_name: "weight2"
          weight_is_a_feature: %s
        }
    """ % (str(weight_is_a_feature), str(weight_is_a_feature)), spec)
    labels = {
        'label1': tf.ones([20], dtype=tf.int32),
        'label2': tf.ones([20], dtype=tf.int32),
    }
    # Fix the size of the dict labels to bypass the assertion.
    if not weight_is_a_feature:
      labels.update({
          'not_used': tf.ones([20], dtype=tf.int32),
          'not_used2': tf.ones([20], dtype=tf.int32)
      })

    weights = {
        'weight1': tf.constant([2] * 20),
        'weight2': tf.constant([3] * 20)
    }
    features = {'x': tf.zeros([10, 10])}
    if not weight_is_a_feature:
      features.update(weights)
    task_manager_instance = task_manager.TaskManager(spec)
    logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
    logits_spec = architecture_utils.LogitsSpec(logits=logits)

    with self.assertRaises(KeyError):
      loss_fn = loss_fns.make_multi_class_loss_fn()

      _ = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels=labels,
          loss_fn=loss_fn,
          model_directory=self.get_temp_dir(),
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)

  def test_architecture(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      learning_rate_spec = {'learning_rate': 0.001, 'gradient_max_norm': 3}
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      text_format.Merge(
          """
          multi_task_spec {
            label_name: "label1"
            number_of_classes: 10
            architecture: "FIXED_OUTPUT_FULLY_CONNECTED_128"
          }
          multi_task_spec {
            label_name: "label2"
            number_of_classes: 10
            architecture: "FIXED_OUTPUT_FULLY_CONNECTED_256"
            architecture: "FIXED_OUTPUT_FULLY_CONNECTED_512"
          }
      """, spec)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      features = {'x': tf.zeros([10, 10])}
      loss_fn = loss_fns.make_multi_class_loss_fn()
      model = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels={
              'label1': tf.ones([20], dtype=tf.int32),
              'label2': tf.ones([20], dtype=tf.int32)
          },
          loss_fn=loss_fn,
          model_directory=self.get_temp_dir(),
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'task_label1_tower/1_FIXED_OUTPUT_FULLY_CONNECTED_128' in node.name
      ])
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'task_label2_tower/1_FIXED_OUTPUT_FULLY_CONNECTED_256' in node.name
      ])
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'task_label2_tower/2_FIXED_OUTPUT_FULLY_CONNECTED_512' in node.name
      ])
      self.assertLen(model.predictions, 3 * (1 + 2))
      self.assertIn('probabilities', model.predictions)
      self.assertIn('log_probabilities', model.predictions)
      self.assertIn('predictions', model.predictions)

  def test_projection(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      learning_rate_spec = {'learning_rate': 0.001, 'gradient_max_norm': 3}
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      text_format.Merge(
          """
          multi_task_spec {
            label_name: "label1"
            number_of_classes: 10
          }
          multi_task_spec {
            label_name: "label2"
            number_of_classes: 5
          }
      """, spec)
      task_manager_instance = task_manager.TaskManager(spec)
      logits = tf.keras.layers.Dense(10)(tf.zeros([20, 10]))
      logits_spec = architecture_utils.LogitsSpec(logits=logits)
      features = {'x': tf.zeros([10, 10])}
      model = task_manager_instance.create_model_spec(
          features=features,
          params=hp.HParams(optimizer='sgd'),
          learning_rate_spec=learning_rate_spec,
          train_logits_specs=[logits_spec],
          eval_logits_spec=logits_spec,
          labels={
              'label1': tf.ones([20], dtype=tf.int32),
              'label2': tf.ones([20], dtype=tf.int32)
          },
          loss_fn=_loss_fn,
          model_directory=self.get_temp_dir(),
          mode=tf.estimator.ModeKeys.TRAIN,
          lengths=None,
          use_tpu=False,
          predictions_fn=_default_predictions_fn)
      self.assertEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'task_label1_tower/dense' in node.name
      ])
      self.assertNotEmpty([
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
          if 'task_label2_tower/dense' in node.name
      ])
      self.assertLen(model.predictions, 3 * (1 + 2))
      self.assertIn('probabilities', model.predictions)
      self.assertIn('log_probabilities', model.predictions)
      self.assertIn('predictions', model.predictions)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
