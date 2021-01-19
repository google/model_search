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
"""Tests for model_search.phoenix."""

import os

from absl import flags
from absl.testing import parameterized

from model_search import hparam as hp
from model_search import loss_fns
from model_search import phoenix
from model_search.architecture import architecture_utils
from model_search.metadata import mock_metadata
from model_search.proto import phoenix_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf

from google.protobuf import text_format

FLAGS = flags.FLAGS

_SPEC_PATH_TEMPLATE = "model_search/model_search/configs/{}_config.pbtxt"


class PhoenixTest(parameterized.TestCase, tf.test.TestCase):

  def _create_phoenix_spec(self, problem_type):
    """Creates a new phoenix.PhoenixSpec for the given problem type."""
    spec_path = os.path.join(FLAGS.test_srcdir,
                             _SPEC_PATH_TEMPLATE.format(problem_type))
    spec = phoenix_spec_pb2.PhoenixSpec()
    with tf.io.gfile.GFile(spec_path, "r") as f:
      text_format.Merge(f.read(), spec)
    return spec

  def _create_phoenix_instance(self,
                               problem_type,
                               input_shape=None,
                               head=None,
                               logits_dimension=None,
                               label_vocabulary=None,
                               loss_fn=None,
                               metric_fn=None):
    """Creates a phoenix.Phoenix instance with mostly default parameters.

    Args:
      problem_type: The problem type (e.g. cnn, rnn, etc) for which to create a
        PhoenixSpec.
      input_shape: The shape of the input Tensor (including batch size).
      head: The head to pass to the Phoenix instance.
      logits_dimension: The shape of the logits to pass to the Phoenix instance.
      label_vocabulary: The label_vocabulary to pass to the Phoenix instance.
      loss_fn: The loss_fn to pass to the Phoenix instance.
      metric_fn: The metric_fn to pass to the Phoenix instance.

    Returns:
      A newly created phoenix.Phoenix instance.
    """
    spec = self._create_phoenix_spec(problem_type)
    input_shape = input_shape or [8, 32, 32, 3]
    logits_dimension = (10 if not logits_dimension and not head else
                        logits_dimension)

    def _input_layer_fn(features, is_training, scope_name,
                        lengths_feature_name):
      del is_training
      del scope_name
      input_layer = tf.cast(features["zeros"], dtype=tf.float32)
      if (spec.problem_type == phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS
          or
          spec.problem_type == phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS
         ):
        lengths = features[lengths_feature_name]
      else:
        lengths = None

      return input_layer, lengths

    return phoenix.Phoenix(
        phoenix_spec=spec,
        input_layer_fn=_input_layer_fn,
        study_owner="test_owner",
        study_name="test_name",
        head=head,
        logits_dimension=logits_dimension,
        label_vocabulary=label_vocabulary,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        metadata=mock_metadata.MockMetaData())

  @parameterized.named_parameters(
      {
          "testcase_name": "multi_task_with_head",
          "kwargs_to_pass": {
              "phoenix_spec":
                  phoenix_spec_pb2.PhoenixSpec(
                      multi_task_spec=[phoenix_spec_pb2.TaskSpec()]),
              "head":
                  tf.estimator.BinaryClassHead(),
          }
      }, {
          "testcase_name": "head_with_logits_dimension",
          "kwargs_to_pass": {
              "head": tf.estimator.BinaryClassHead(),
              "logits_dimension": 10,
          }
      }, {
          "testcase_name": "head_with_label_vocabulary",
          "kwargs_to_pass": {
              "head": tf.estimator.BinaryClassHead(),
              "label_vocabulary": {
                  "the_answer": 42
              },
          }
      }, {
          "testcase_name": "head_with_loss_fn",
          "kwargs_to_pass": {
              "head": tf.estimator.BinaryClassHead(),
              "loss_fn": lambda loss: 10,
          }
      }, {
          "testcase_name": "head_with_metrics_fn",
          "kwargs_to_pass": {
              "head": tf.estimator.BinaryClassHead(),
              "metric_fn": lambda labels, predictions, weights: 10,
          }
      })
  def test_preconditions(self, kwargs_to_pass=None):
    if "phoenix_spec" not in kwargs_to_pass:
      kwargs_to_pass["phoenix_spec"] = phoenix_spec_pb2.PhoenixSpec()
    kwargs_to_pass["input_layer_fn"] = lambda: None
    kwargs_to_pass["study_owner"] = "fake_owner"
    kwargs_to_pass["study_name"] = "fake_name"
    with self.assertRaises(AssertionError):
      phoenix.Phoenix(**kwargs_to_pass)

  @parameterized.named_parameters({
      "testcase_name": "cnn",
      "problem": "cnn"
  }, {
      "testcase_name": "dnn",
      "problem": "dnn"
  }, {
      "testcase_name": "rnn_all",
      "problem": "rnn_all"
  }, {
      "testcase_name": "rnn_last",
      "problem": "rnn_last"
  })
  def test_get_keras_hyperparameters_space(self, problem):
    spec = self._create_phoenix_spec(problem)
    _ = phoenix.Phoenix.get_keras_hyperparameters_space(
        phoenix_spec=spec, train_steps=10000)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "cnn",
          "problem":
              "cnn",
          "input_shape": [20, 32, 32, 3],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_CHANNEL_CONVOLUTION_16",
                      "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
                  ],
                  learning_rate=0.01,
                  optimizer="sgd"),
          "label_shape": [20]
      }, {
          "testcase_name":
              "dnn",
          "problem":
              "dnn",
          "input_shape": [20, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  optimizer="sgd"),
          "label_shape": [20]
      }, {
          "testcase_name":
              "rnn_all",
          "problem":
              "rnn_all",
          "input_shape": [20, 32, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "RNN_CELL_64",
                      "RNN_CELL_64",
                  ],
                  learning_rate=0.01,
                  optimizer="sgd"),
          "label_shape": [20, 32]
      }, {
          "testcase_name":
              "rnn_last",
          "problem":
              "rnn_last",
          "input_shape": [20, 32, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "RNN_CELL_64",
                      "RNN_CELL_64",
                  ],
                  learning_rate=0.01,
                  optimizer="sgd"),
          "label_shape": [20]
      })
  def test_model_fn(self, problem, input_shape, hparams, label_shape):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      instance = self._create_phoenix_instance(problem, input_shape)
      features = {"zeros": tf.zeros(input_shape), "lengths": tf.ones([20])}
      labels = tf.ones(label_shape, dtype=tf.int32)
      run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

      model_fn = instance._make_model_fn(run_config, train_steps=100)
      _ = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, hparams)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "cnn",
          "problem":
              "cnn",
          "input_shape": [20, 32, 32, 3],
          "hparams":
              hp.HParams(initial_architecture=[
                  "FIXED_CHANNEL_CONVOLUTION_16",
                  "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
              ]),
      }, {
          "testcase_name":
              "dnn",
          "problem":
              "dnn",
          "input_shape": [20, 32],
          "hparams":
              hp.HParams(initial_architecture=[
                  "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  "FIXED_OUTPUT_FULLY_CONNECTED_128",
              ]),
      }, {
          "testcase_name":
              "rnn_all",
          "problem":
              "rnn_all",
          "input_shape": [20, 32, 32],
          "hparams":
              hp.HParams(initial_architecture=[
                  "RNN_CELL_64",
                  "RNN_CELL_64",
              ]),
      }, {
          "testcase_name":
              "rnn_last",
          "problem":
              "rnn_last",
          "input_shape": [20, 32, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "RNN_CELL_64",
                      "RNN_CELL_64",
                  ],
                  learning_rate=0.01,
                  optimizer="sgd"),
      })
  def test_model_fn_predict(self, problem, input_shape, hparams):
    instance = self._create_phoenix_instance(problem, input_shape)
    features = {"zeros": tf.zeros(input_shape), "lengths": tf.ones([20])}
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    model_fn = instance._make_model_fn(run_config, train_steps=100)
    spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT, hparams)

    self.assertIsNone(spec.loss)
    self.assertNotEqual(spec.predictions, None)

  @parameterized.named_parameters(
      {
          "testcase_name": "no_scaled_training",
          "use_parameter_scaled_training": False,
          "train_steps": 10000,
          "maximum_depth": 100,
          "architecture": np.array([1, 2, 3]),
          "expected": 1
      },
      {
          "testcase_name": "no_architecture",
          "use_parameter_scaled_training": False,
          "train_steps": 10000,
          "maximum_depth": 100,
          "architecture": None,
          "expected": 1
      },
      {
          "testcase_name": "scaled_training",
          "use_parameter_scaled_training": True,
          "train_steps": 10000,
          "maximum_depth": 10,
          "architecture": np.array([1, 2, 3]),
          "expected": 3
      },
  )
  def test_increment_global_step(self, use_parameter_scaled_training,
                                 train_steps, maximum_depth, architecture,
                                 expected):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      with self.test_session(graph=tf.Graph()) as sess:
        tower_name = "tower"
        if architecture is not None:
          architecture_utils.set_architecture(
              architecture, tower_name=tower_name)
        spec = self._create_phoenix_spec(problem_type="cnn")
        spec.maximum_depth = maximum_depth
        spec.use_parameter_scaled_training = use_parameter_scaled_training
        instance = phoenix.Phoenix(
            phoenix_spec=spec,
            input_layer_fn=lambda: None,
            logits_dimension=0,
            study_name="test",
            study_owner="test")
        global_step = tf.compat.v1.train.get_or_create_global_step()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        before = sess.run(global_step)

        op = instance._increment_global_step(
            train_op=tf.no_op(), train_steps=train_steps, tower_name=tower_name)
        sess.run(op)
        after = sess.run(global_step)

        self.assertEqual(before, 0)
        self.assertEqual(after, expected)

  def test_merge_hparams(self):
    params = phoenix._merge_hparams(
        hp.HParams(learning_rate=1, untouched=4),
        hp.HParams(learning_rate=2, new_param=3))
    self.assertEqual(params.learning_rate, 2)
    self.assertEqual(params.untouched, 4)
    self.assertEqual(params.new_param, 3)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "cnn",
          "problem":
              "cnn",
          "input_shape": [20, 32, 32, 3],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_CHANNEL_CONVOLUTION_16",
                      "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_CHANNEL_CONVOLUTION_16",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "dnn",
          "problem":
              "dnn",
          "input_shape": [20, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "dnn_no_vocab",
          "problem":
              "dnn",
          "input_shape": [20, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
          "vocab": [],
          "loss_fn":
              loss_fns.make_multi_class_loss_fn(label_vocabulary=[
                  "CLASS1", "CLASS2", "CLASS3", "CLASS4", "CLASS5"
              ]),
      })
  def test_model_fn_with_vocab(self,
                               problem,
                               input_shape,
                               hparams,
                               vocab=None,
                               loss_fn=None):
    if vocab is None:
      vocab = ["CLASS1", "CLASS2", "CLASS3", "CLASS4", "CLASS5"]
    instance = self._create_phoenix_instance(
        problem_type=problem,
        input_shape=input_shape,
        logits_dimension=5,
        label_vocabulary=vocab,
        loss_fn=loss_fn)
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    def input_fn():
      features = {"zeros": tf.zeros(input_shape)}
      labels = tf.constant(["CLASS1", "CLASS1"] * 10, dtype=tf.string)
      return features, labels

    estimator = instance.get_estimator(run_config, hparams, 10)
    estimator.train(input_fn=input_fn, max_steps=10)
    estimator.evaluate(input_fn=input_fn, steps=10)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "dnn_regression",
          "problem":
              "dnn",
          "input_shape": [10, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
          "loss_fn":
              loss_fns.make_regression_loss_fn(),
      }, {
          "testcase_name":
              "cnn_regression",
          "problem":
              "dnn",
          "input_shape": [10, 32, 32, 3],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
          "loss_fn":
              loss_fns.make_regression_loss_fn(),
      })
  def test_model_fn_regression(self,
                               problem,
                               input_shape,
                               hparams,
                               loss_fn=None,
                               metric_fn=None):
    instance = self._create_phoenix_instance(
        problem_type=problem,
        input_shape=input_shape,
        logits_dimension=1,
        loss_fn=loss_fn,
        metric_fn=metric_fn)
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    def input_fn():
      features = {"zeros": tf.zeros(input_shape)}
      labels = tf.constant([[0.]] * 10, dtype=tf.float32)
      return features, labels

    estimator = instance.get_estimator(run_config, hparams, 10)
    estimator.train(input_fn=input_fn, max_steps=10)
    eval_result = estimator.evaluate(input_fn=input_fn, steps=10)

    self.assertAlmostEqual(0., eval_result["loss"])

  @parameterized.named_parameters(
      {
          "testcase_name":
              "cnn_custom_metric",
          "problem":
              "cnn",
          "input_shape": [20, 32, 32, 3],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_CHANNEL_CONVOLUTION_16",
                      "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_CHANNEL_CONVOLUTION_16",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "dnn_custom_metric",
          "problem":
              "dnn",
          "input_shape": [20, 32],
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
      })
  def test_model_fn_custom_metric_fn(self, problem, input_shape, hparams):
    num_classes = 5

    def metric_fn(labels, predictions, weights):
      del weights
      return {
          "mean_per_class_accuracy":
              (tf.compat.v1.metrics.mean_per_class_accuracy(
                  labels, predictions["predictions"], num_classes))
      }

    instance = self._create_phoenix_instance(
        problem_type=problem,
        input_shape=input_shape,
        logits_dimension=num_classes,
        metric_fn=metric_fn)
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    def input_fn():
      features = {"zeros": tf.zeros(input_shape)}
      labels = tf.constant([0] * 20, dtype=tf.int64)
      return features, labels

    estimator = instance.get_estimator(run_config, hparams, 10)
    estimator.train(input_fn=input_fn, max_steps=10)
    eval_result = estimator.evaluate(input_fn=input_fn, steps=10)

    self.assertIn("mean_per_class_accuracy", eval_result)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "dnn",
          "problem":
              "dnn",
          "input_shape": [10, 32],
          "label_vocabulary":
              None,
          "labels_fn":
              lambda: tf.constant([0] * 10, dtype=tf.int64),
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "dnn_vocab",
          "problem":
              "dnn",
          "input_shape": [10, 32],
          "label_vocabulary": ["ZERO", "ONE"],
          "labels_fn":
              lambda: tf.constant(["ZERO"] * 10, dtype=tf.string),
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                      "FIXED_OUTPUT_FULLY_CONNECTED_128",
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_OUTPUT_FULLY_CONNECTED_128",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "cnn",
          "problem":
              "cnn",
          "input_shape": [10, 32, 32, 3],
          "label_vocabulary":
              None,
          "labels_fn":
              lambda: tf.constant([0] * 10, dtype=tf.int64),
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_CHANNEL_CONVOLUTION_16",
                      "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_CHANNEL_CONVOLUTION_16",
                  optimizer="sgd"),
      }, {
          "testcase_name":
              "cnn_vocab",
          "problem":
              "cnn",
          "input_shape": [10, 32, 32, 3],
          "label_vocabulary": ["ZERO", "ONE"],
          "labels_fn":
              lambda: tf.constant(["ZERO"] * 10, dtype=tf.string),
          "hparams":
              hp.HParams(
                  initial_architecture=[
                      "FIXED_CHANNEL_CONVOLUTION_16",
                      "FIXED_CHANNEL_CONVOLUTION_64", "PLATE_REDUCTION_FLATTEN"
                  ],
                  learning_rate=0.01,
                  new_block_type="FIXED_CHANNEL_CONVOLUTION_16",
                  optimizer="sgd"),
      })
  def test_model_fn_binary_class(self, problem, input_shape, label_vocabulary,
                                 labels_fn, hparams):
    num_classes = 2
    loss_fn = loss_fns.make_multi_class_loss_fn(
        label_vocabulary=label_vocabulary)
    instance = self._create_phoenix_instance(
        problem_type=problem,
        input_shape=input_shape,
        logits_dimension=num_classes,
        label_vocabulary=label_vocabulary,
        loss_fn=loss_fn)
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    def input_fn():
      features = {"zeros": tf.zeros(input_shape)}
      labels = labels_fn()
      return features, labels

    estimator = instance.get_estimator(run_config, hparams, 10)
    estimator.train(input_fn=input_fn, max_steps=10)
    eval_result = estimator.evaluate(input_fn=input_fn, steps=10)

    self.assertIn("auc_roc", eval_result)

  # pylint: disable=unnecessary-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "regression",
          "head_fn": lambda: tf.estimator.RegressionHead()
      },
      {
          "testcase_name": "binary_classification",
          "head_fn": lambda: tf.estimator.BinaryClassHead()
      },
      {
          "testcase_name": "multi_class_classification",
          "head_fn": lambda: tf.estimator.MultiClassHead(n_classes=10)
      },
      {
          "testcase_name": "multi_label_classification",
          "head_fn": lambda: tf.estimator.MultiLabelHead(n_classes=10),
          # (batch_size, logits_dimension)
          "label_fn": lambda: tf.zeros((8, 10), dtype=tf.int32)
      },
      {
          "testcase_name":
              "multi_class_classification_with_vocab_and_weights",
          # pylint: disable=g-long-lambda
          "head_fn":
              lambda: tf.estimator.MultiClassHead(
                  n_classes=3,
                  label_vocabulary=["ONE", "TWO", "THREE"],
                  weight_column="weights"),
          # pylint: enable=g-long-lambda
          "label_fn":
              lambda: tf.constant(["ONE"] * 8, dtype=tf.string),  # batch_size
      })
  # pylint: enable=unnecessary-lambda
  def test_head(self, head_fn, label_fn=None, loss_fn=None, metric_fn=None):
    batch_size = 8
    input_shape = [batch_size, 32, 32, 3]
    hparams = hp.HParams(
        initial_architecture=[
            "FIXED_CHANNEL_CONVOLUTION_16", "FIXED_CHANNEL_CONVOLUTION_64",
            "PLATE_REDUCTION_FLATTEN"
        ],
        new_block_type="FIXED_CHANNEL_CONVOLUTION_32",
        learning_rate=1000.0,  # Approximating constant so we'll never diverge.
        optimizer="sgd")
    instance = self._create_phoenix_instance(
        problem_type="cnn",
        input_shape=input_shape,
        head=head_fn(),
        loss_fn=loss_fn,
        metric_fn=metric_fn)
    run_config = tf.estimator.RunConfig(model_dir=self.get_temp_dir() + "/1")

    def input_fn():
      features = {"zeros": tf.zeros(input_shape)}
      if getattr(head_fn(), "_weight_column", None):
        features["weights"] = tf.ones(batch_size) * .5
      labels = label_fn() if label_fn else tf.zeros(batch_size, dtype=tf.int32)
      return features, labels

    estimator = instance.get_estimator(run_config, hparams, 10)
    estimator.train(input_fn=input_fn, max_steps=10)
    eval_result = estimator.evaluate(input_fn=input_fn, steps=10)

    self.assertAlmostEqual(0., eval_result["loss"], places=3)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
