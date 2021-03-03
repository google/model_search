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
"""Tests for learning.adanets.phoenix.architecture.architecture_utils."""

from absl.testing import parameterized

from model_search import blocks_builder as blocks
from model_search import hparam as hp
from model_search.architecture import architecture_utils
from model_search.metadata import trial
from model_search.proto import phoenix_spec_pb2
from model_search.proto import transfer_learning_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf


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


def _replay_spec():
  output = phoenix_spec_pb2.PhoenixSpec()
  output.replay.CopyFrom(phoenix_spec_pb2.ArchitectureReplay())
  return output


class ArchitectureUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_blocks_search_space(self):
    hps = architecture_utils.get_blocks_search_space()
    self.assertIn("TUNABLE_SVDF_output_size", hps)
    self.assertIn("TUNABLE_SVDF_rank", hps)
    self.assertIn("TUNABLE_SVDF_projection_size", hps)
    self.assertIn("TUNABLE_SVDF_memory_size", hps)
    hps = architecture_utils.get_blocks_search_space(["TUNABLE_SVDF"])
    self.assertIn("TUNABLE_SVDF_output_size", hps)
    self.assertIn("TUNABLE_SVDF_rank", hps)
    self.assertIn("TUNABLE_SVDF_projection_size", hps)
    self.assertIn("TUNABLE_SVDF_memory_size", hps)

  def test_get_block_hparams(self):
    hp_ = architecture_utils.get_block_hparams(
        hp.HParams(TUNABLE_SVDF_target=10, non_target=15), "TUNABLE_SVDF")
    self.assertLen(hp_.values(), 1)
    self.assertEqual(hp_.target, 10)

  def test_store_and_get_hparams(self):
    hp_ = hp.HParams(hello="world", context="toberemoved")
    dirname = self.get_temp_dir()
    architecture_utils.store_hparams_to_dir(hp_, dirname, "tower")
    hp_replica = architecture_utils.get_hparams_from_dir(dirname, "tower")
    self.assertLen(hp_replica.values(), 1)
    self.assertEqual(hp_replica.hello, "world")

  @parameterized.named_parameters(
      {
          "testcase_name": "empty",
          "input_dir": "",
          "expected_output": None,
          "spec": phoenix_spec_pb2.PhoenixSpec(),
      }, {
          "testcase_name": "normal",
          "input_dir": "some/random/path/512",
          "expected_output": 512,
          "spec": phoenix_spec_pb2.PhoenixSpec(),
      }, {
          "testcase_name": "normal_tfx",
          "input_dir": "/some/random/Trial-000034/active/Run-0000000",
          "expected_output": 34,
          "spec": phoenix_spec_pb2.PhoenixSpec(),
      }, {
          "testcase_name": "hybrid",
          "input_dir": "/some/random/Trial-000034/active/Run-0000000/7",
          "expected_output": 7,
          "spec": phoenix_spec_pb2.PhoenixSpec(),
      }, {
          "testcase_name": "replay",
          "input_dir": "/some/random/Trial-000034/active/Run-0000000/7",
          "expected_output": 7,
          "spec": _replay_spec(),
      })
  def test_get_trial_id(self, input_dir, expected_output, spec):
    output = architecture_utils.DirectoryHandler.get_trial_id(input_dir, spec)
    self.assertEqual(output, expected_output)

  @parameterized.named_parameters(
      {
          "testcase_name": "empty",
          "input_trial": trial.Trial({"model_dir": ""}),
          "expected_output": ""
      }, {
          "testcase_name": "normal",
          "input_trial": trial.Trial({"model_dir": "random/path"}),
          "expected_output": "random/path"
      })
  def test_get_trial_dir(self, input_trial, expected_output):
    output = architecture_utils.DirectoryHandler.trial_dir(input_trial)
    self.assertEqual(output, expected_output)

  @parameterized.named_parameters(
      {
          "testcase_name": "empty",
          "initial_architecture": [],
          "fixed_architecture": [blocks.BlockType.PLATE_REDUCTION_FLATTEN]
      }, {
          "testcase_name": "flatten",
          "initial_architecture": [blocks.BlockType.PLATE_REDUCTION_FLATTEN],
          "fixed_architecture": [blocks.BlockType.PLATE_REDUCTION_FLATTEN]
      }, {
          "testcase_name": "downsample_flatten",
          "initial_architecture": [blocks.BlockType.DOWNSAMPLE_FLATTEN],
          "fixed_architecture": [blocks.BlockType.DOWNSAMPLE_FLATTEN]
      }, {
          "testcase_name":
              "basic_case",
          "initial_architecture": [
              blocks.BlockType.PLATE_REDUCTION_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED, blocks.BlockType.CONVOLUTION_3X3
          ],
          "fixed_architecture": [
              blocks.BlockType.CONVOLUTION_3X3,
              blocks.BlockType.PLATE_REDUCTION_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED
          ]
      }, {
          "testcase_name":
              "basic_downsample_case",
          "initial_architecture": [
              blocks.BlockType.DOWNSAMPLE_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED, blocks.BlockType.CONVOLUTION_3X3
          ],
          "fixed_architecture": [
              blocks.BlockType.CONVOLUTION_3X3,
              blocks.BlockType.DOWNSAMPLE_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED
          ]
      }, {
          "testcase_name":
              "advanced_case",
          "initial_architecture": [
              blocks.BlockType.PLATE_REDUCTION_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED,
              blocks.BlockType.FULLY_CONNECTED_PYRAMID,
              blocks.BlockType.CONVOLUTION_3X3,
              blocks.BlockType.DOWNSAMPLE_CONVOLUTION_3X3
          ],
          "fixed_architecture": [
              blocks.BlockType.CONVOLUTION_3X3,
              blocks.BlockType.DOWNSAMPLE_CONVOLUTION_3X3,
              blocks.BlockType.PLATE_REDUCTION_FLATTEN,
              blocks.BlockType.FULLY_CONNECTED,
              blocks.BlockType.FULLY_CONNECTED_PYRAMID
          ]
      })
  def test_fix_architecture_order(self, initial_architecture,
                                  fixed_architecture):
    out_architecture = architecture_utils.fix_architecture_order(
        initial_architecture, phoenix_spec_pb2.PhoenixSpec.CNN)
    self.assertAllEqual(fixed_architecture, out_architecture)

  @parameterized.named_parameters(
      {
          "testcase_name": "empty_rnn",
          "problem_type": phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS,
          "new_block": blocks.BlockType.LSTM_128,
          "initial_architecture": np.array([]),
          "expected_architecture": [blocks.BlockType.LSTM_128]
      }, {
          "testcase_name":
              "empty_dnn",
          "problem_type":
              phoenix_spec_pb2.PhoenixSpec.DNN,
          "new_block":
              blocks.BlockType.FIXED_OUTPUT_FULLY_CONNECTED_128,
          "initial_architecture":
              np.array([]),
          "expected_architecture":
              [blocks.BlockType.FIXED_OUTPUT_FULLY_CONNECTED_128]
      })
  def test_increase_structure_depth(self, problem_type, initial_architecture,
                                    new_block, expected_architecture):
    out_architecture = architecture_utils.increase_structure_depth(
        initial_architecture, new_block, problem_type)
    self.assertAllEqual(expected_architecture, out_architecture)

  @parameterized.named_parameters(
      {
          "testcase_name": "test1",
          "architecture": np.array([1, 2, 3, 4]),
      }, {
          "testcase_name": "test2",
          "architecture": np.array([2, 3, 4, 5]),
      })
  def test_set_get_architecture(self, architecture):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      directory = self.get_temp_dir()
      with self.test_session(graph=tf.Graph()) as sess:
        architecture_utils.set_architecture(architecture)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        saver.save(sess, directory + "/ckpt")

      output_architecture = architecture_utils.get_architecture(directory)
      self.assertAllEqual(output_architecture, architecture)

  @parameterized.named_parameters(
      {
          "testcase_name": "no_architecture",
          "architecture": None,
      }, {
          "testcase_name": "has_architecture",
          "architecture": np.array([2, 3, 4, 5]),
      })
  def test_get_architecture_size(self, architecture):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      with self.test_session(graph=tf.Graph()):
        tower_name = "tower"
        if architecture is not None:
          architecture_utils.set_architecture(
              architecture, tower_name=tower_name)
        size = architecture_utils.get_architecture_size(tower_name=tower_name)
        if architecture is not None:
          self.assertEqual(size, architecture.size)
        else:
          self.assertIsNone(size)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "regular_network",
          "dropout":
              -1,
          "expected_logits": [[
              -0.1481, 0.3328, 0.3028, 0.5652, 0.6860, 0.06171, 0.09998,
              -0.2622, 0.2186, -0.1322
          ]]
      }, {
          "testcase_name":
              "dropout_network",
          "dropout":
              0.1,
          "expected_logits": [[
              0.7001, -0.06655, -0.1711, 0.1274, -0.8175, 0.2932, 0.06242,
              0.2182, -0.06626, 0.7882
          ]],
      })
  def test_construct_network(self, dropout, expected_logits):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      tf.random.set_seed(1234)
      # convolutions and then flatten plate.
      architecture = np.array([1, 3, 34])
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, shape=[None, 32, 32, 3], name="input")
      phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      tower_spec = architecture_utils.construct_tower(
          phoenix_spec=phoenix_spec,
          input_tensor=input_tensor,
          tower_name="test_tower",
          architecture=architecture,
          is_training=True,
          lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(),
          model_directory=self.get_temp_dir(),
          is_frozen=False,
          dropout_rate=dropout)
      np.random.seed(42)
      test_input = np.random.random([1, 32, 32, 3])

      with tf.compat.v1.Session() as sess:
        sess.run([
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer()
        ])
        logits_val = sess.run(
            tower_spec.logits_spec.logits, feed_dict={input_tensor: test_input})

      self.assertAllClose(expected_logits, logits_val, rtol=1e-3)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "disabled",
          "transfer_learning_type":
              transfer_learning_spec_pb2.TransferLearningSpec
              .NO_TRANSFER_LEARNING
      }, {
          "testcase_name":
              "enabled",
          "transfer_learning_type":
              transfer_learning_spec_pb2.TransferLearningSpec
              .UNIFORM_AVERAGE_TRANSFER_LEARNING
      })
  def test_construct_tower_with_transfer_learning(
      self,
      transfer_learning_type=transfer_learning_spec_pb2.TransferLearningSpec
      .NO_TRANSFER_LEARNING):
    # convolutions and then flatten plate.
    architecture = np.array([1, 3, 34])
    str_signature = "_1334"
    input_tensor = tf.zeros([100, 32, 32, 3])
    tower_name = "test_tower"
    transfer_learning_spec = transfer_learning_spec_pb2.TransferLearningSpec(
        transfer_learning_type=transfer_learning_type)
    phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
        problem_type=phoenix_spec_pb2.PhoenixSpec.CNN,
        transfer_learning_spec=transfer_learning_spec)
    _ = architecture_utils.construct_tower(
        phoenix_spec=phoenix_spec,
        input_tensor=input_tensor,
        tower_name=tower_name,
        architecture=architecture,
        is_training=True,
        lengths=None,
        logits_dimension=10,
        hparams=hp.HParams(),
        model_directory=self.get_temp_dir(),
        is_frozen=False,
        dropout_rate=None)
    tensors = architecture_utils.get_tower_variables(tower_name)
    for tensor in tensors:
      if (transfer_learning_type ==
          transfer_learning_spec_pb2.TransferLearningSpec.NO_TRANSFER_LEARNING):
        self.assertEndsWith(tensor.op.name, str_signature)
      else:
        self.assertNotEndsWith(tensor.op.name, str_signature)

  @parameterized.named_parameters(
      {
          "testcase_name": "shared_input",
          "shared_input": True
      }, {
          "testcase_name": "not_shared_input",
          "shared_input": False
      })
  def test_import_tower(self, shared_input):
    np.random.seed(42)
    test_input = np.random.random([1, 32, 32, 3])
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      directory = self.get_temp_dir()
      architecture = np.array([1, 3, 34])
      phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      phoenix_spec.is_input_shared = shared_input
      features = {}
      shared_input_tensor = None
      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor_1 = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 32, 32, 3], name="input_1")
        tf.random.set_seed(1234)
        tower_spec_1 = architecture_utils.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor_1,
            tower_name="test_tower",
            architecture=architecture,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            hparams=hp.HParams(),
            model_directory=self.get_temp_dir(),
            is_frozen=False,
            dropout_rate=None)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        logits_val_1 = sess.run(
            tower_spec_1.logits_spec.logits,
            feed_dict={input_tensor_1: test_input})
        saver.save(sess, directory + "/ckpt")

      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor_2 = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 32, 32, 3], name="input_2")
        if shared_input:
          shared_input_tensor = input_tensor_2

          def _input_layer_fn(features,
                              is_training,
                              scope_name="Phoenix/Input",
                              lengths_feature_name=None):
            del features, is_training, scope_name, lengths_feature_name
            return None, None
        else:
          features = {"x": input_tensor_2}

          def _input_layer_fn(features,
                              is_training,
                              scope_name="Phoenix/Input",
                              lengths_feature_name=None):
            del is_training, lengths_feature_name
            with tf.compat.v1.variable_scope(scope_name):
              return tf.cast(features["x"], dtype=tf.float32), None

        tf.random.set_seed(1234)
        tower_spec_2 = architecture_utils.import_tower(
            features=features,
            input_layer_fn=_input_layer_fn,
            phoenix_spec=phoenix_spec,
            shared_input_tensor=shared_input_tensor,
            original_tower_name="test_tower",
            new_tower_name="imported_tower",
            model_directory=directory,
            new_model_directory=self.get_temp_dir(),
            is_training=True,
            logits_dimension=10,
            shared_lengths=None,
            force_snapshot=False,
            force_freeze=False)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        logits_val_2 = sess.run(
            tower_spec_2.logits_spec.logits,
            feed_dict={input_tensor_2: test_input})

      self.assertAllClose(logits_val_1, logits_val_2, rtol=1e-3)

  @parameterized.named_parameters(
      {
          "testcase_name": "cnn",
          "input_tensor_shape": [20, 20],
          "spec": _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "output_shape": [20, 7],
      },
      {
          "testcase_name": "dnn",
          "input_tensor_shape": [20, 20],
          "spec": _create_spec(phoenix_spec_pb2.PhoenixSpec.DNN),
          "output_shape": [20, 7],
      },
      {
          "testcase_name":
              "rnn_all_activations",
          "input_tensor_shape": [20, 20, 20],
          "spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS),
          "output_shape": [20, 20, 7],
      },
      {
          "testcase_name":
              "rnn_last_activations",
          "input_tensor_shape": [20, 20, 20],
          "spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS),
          "output_shape": [20, 7],
          "lengths":
              list(range(20))
      },
      {
          "testcase_name":
              "rnn_last_activations_no_length",
          "input_tensor_shape": [20, 20, 20],
          "spec":
              _create_spec(phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS),
          "output_shape": [20, 20, 7],
      },
      {
          "testcase_name": "nasnet_aux_head",
          "input_tensor_shape": [20, 20],
          "spec": _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "output_shape": [20, 7],
          "extra_block": 6,  # Downsample convolution.
          "extra_block_shape": [20, 20, 20, 20],
          "use_auxiliary_head": True
      },
      {
          "testcase_name": "deep_skip_head",
          "input_tensor_shape": [20, 20],
          "spec": _create_spec(phoenix_spec_pb2.PhoenixSpec.CNN),
          "output_shape": [20, 7],
          "extra_block": 16,  # Flatten.
          "use_auxiliary_head": True
      })
  def test_create_tower_spec(
      self,
      input_tensor_shape,
      spec,
      output_shape,
      lengths=None,
      extra_block=14,  # Fully connected.
      extra_block_shape=None,
      use_auxiliary_head=False):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec.use_auxiliary_head = use_auxiliary_head
      spec.auxiliary_head_loss_weight = .4
      input_tensors = [
          tf.zeros(extra_block_shape or (1,)),  # Raw features -- discarded.
          tf.zeros(extra_block_shape or input_tensor_shape),  # Extra block.
          tf.zeros(extra_block_shape or input_tensor_shape),  # Extra block.
          tf.zeros(input_tensor_shape)  # Fully connected block.
      ]
      fake_architecture = [extra_block, extra_block, 14]
      logits_spec, _, _ = architecture_utils.create_tower_spec(
          phoenix_spec=spec,
          inputs=input_tensors,
          architecture=fake_architecture,
          dimension=7,
          is_frozen=False,
          lengths=lengths,
          allow_auxiliary_head=use_auxiliary_head)
      self.assertAllEqual(output_shape, logits_spec.logits.shape)
      if use_auxiliary_head:
        self.assertIsNotNone(logits_spec.aux_logits)
        self.assertNear(logits_spec.aux_logits_weight,
                        spec.auxiliary_head_loss_weight, 1e-6)
        self.assertNear(logits_spec.logits_weight, 1.0, 1e-6)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "same_graph_snapshotting",
          "new_architecture":
              np.array([1, 3, 34]),
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
          ]
      }, {
          "testcase_name": "same_graph_snapshotting_new_towername",
          "new_architecture": np.array([1, 3, 34]),
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
          "new_tower_name": "test_tower_2"
      }, {
          "testcase_name":
              "changing_second",
          "new_architecture":
              np.array([1, 2, 34]),
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
          ]
      })
  def test_init_variables(self,
                          new_architecture,
                          expected_output,
                          new_tower_name="test_tower"):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      directory = self.get_temp_dir()
      architecture = np.array([1, 3, 34])
      phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor = tf.zeros([100, 32, 32, 3])
        _ = architecture_utils.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor,
            tower_name="test_tower",
            architecture=architecture,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            model_directory=self.get_temp_dir(),
            hparams=hp.HParams(),
            is_frozen=False,
            dropout_rate=None)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        saver.save(sess, directory + "/ckpt")

      with self.test_session(graph=tf.Graph()) as sess:
        input_tensor = tf.zeros([100, 32, 32, 3])
        _ = architecture_utils.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor,
            tower_name=new_tower_name,
            architecture=new_architecture,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            hparams=hp.HParams(),
            model_directory=self.get_temp_dir(),
            is_frozen=False,
            dropout_rate=None)
        snapshotting_variables = architecture_utils.init_variables(
            tf.train.latest_checkpoint(directory), "Phoenix/test_tower",
            "Phoenix/{}".format(new_tower_name))
        self.assertCountEqual(snapshotting_variables, expected_output)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
