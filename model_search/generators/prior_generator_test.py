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
"""Tests for model_search.generators.prior_generator."""

import collections
import os
from absl import flags
from absl.testing import parameterized

from model_search import hparam as hp
from model_search.architecture import architecture_utils
from model_search.generators import prior_generator
from model_search.generators import trial_utils
from model_search.metadata import ml_metadata_db
from model_search.metadata import trial as trial_module
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2

import numpy as np
import tensorflow.compat.v2 as tf


_NONADAPTIVE_GRAPH_NODES = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/prior_generator_0/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/weights',
    u'Phoenix/prior_generator_0/1_FIXED_CHANNEL_CONVOLUTION_16_1/LeakyRelu',
    u'Phoenix/prior_generator_0/2_FIXED_CHANNEL_CONVOLUTION_64_13/Conv/biases',
    u'Phoenix/prior_generator_0/2_FIXED_CHANNEL_CONVOLUTION_64_13/LeakyRelu',
    u'Phoenix/prior_generator_0/3_PLATE_REDUCTION_FLATTEN_1334/Mean',
    u'Phoenix/prior_generator_0/last_dense_1334/dense/kernel',
    u'Phoenix/prior_generator_0/last_dense_1334/logits',
    u'Phoenix/prior_generator_0/last_dense_1334/StopGradient',
    u'architectures/prior_generator_0',
    u'params/prior_generator_0/dropout_rate',
    u'params/prior_generator_0/is_frozen',
    u'checkpoint_initializer/prefix',
    u'checkpoint_initializer/tensor_names',
    u'checkpoint_initializer/shape_and_slices',
    u'checkpoint_initializer',
    u'checkpoint_initializer_1/prefix',
    u'Phoenix/prior_generator_1/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/weights',
    u'Phoenix/prior_generator_1/1_FIXED_CHANNEL_CONVOLUTION_16_1/LeakyRelu',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/Conv/biases',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/LeakyRelu',
    u'Phoenix/prior_generator_1/3_PLATE_REDUCTION_FLATTEN_1334/Mean',
    u'Phoenix/prior_generator_1/last_dense_1334/logits',
    u'Phoenix/prior_generator_1/last_dense_1334/StopGradient',
    u'architectures/prior_generator_1',
    u'params/prior_generator_1/dropout_rate',
    u'params/prior_generator_1/is_frozen',
    u'checkpoint_initializer_6/prefix',
    u'number_of_towers/prior_generator',
]

_ADAPTIVE_GRAPH_NODE = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/prior_generator_0/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/weights',
    u'Phoenix/prior_generator_0/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/biases',
    u'Phoenix/prior_generator_0/1_FIXED_CHANNEL_CONVOLUTION_16_1/LeakyRelu',
    u'Phoenix/prior_generator_0/2_FIXED_CHANNEL_CONVOLUTION_64_13/Conv/biases',
    u'Phoenix/prior_generator_0/2_FIXED_CHANNEL_CONVOLUTION_64_13/LeakyRelu',
    u'Phoenix/prior_generator_0/3_PLATE_REDUCTION_FLATTEN_1334/Mean',
    u'Phoenix/prior_generator_0/last_dense_1334/dense/kernel',
    u'Phoenix/prior_generator_0/last_dense_1334/dense/bias',
    u'Phoenix/prior_generator_0/last_dense_1334/logits',
    u'Phoenix/prior_generator_0/last_dense_1334/StopGradient',
    u'architectures/prior_generator_0',
    u'params/prior_generator_0/dropout_rate',
    u'params/prior_generator_0/is_frozen',
    u'checkpoint_initializer/prefix',
    u'checkpoint_initializer/tensor_names',
    u'Phoenix/prior_generator_1/1_FIXED_CHANNEL_CONVOLUTION_16_1/Conv/weights',
    u'Phoenix/prior_generator_1/1_FIXED_CHANNEL_CONVOLUTION_16_1/LeakyRelu',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/Conv/biases',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/LeakyRelu',
    u'Phoenix/prior_generator_1/3_PLATE_REDUCTION_FLATTEN_1334/Mean',
    u'Phoenix/prior_generator_1/last_dense_1334/logits',
    u'Phoenix/prior_generator_1/last_dense_1334/StopGradient',
    u'architectures/prior_generator_1',
    u'params/prior_generator_1/dropout_rate',
    u'params/prior_generator_1/is_frozen',
    u'checkpoint_initializer_6/prefix',
    u'number_of_towers/prior_generator',
]

_DISTILLATION_GRAPH_NODE = [
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/Const_1',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/Const_2',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/FusedBatchNormV3',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/Const_3',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/AssignMovingAvg/sub/x',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/AssignMovingAvg/sub',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/AssignMovingAvg/ReadVariableOp',
    u'Phoenix/prior_generator_1/2_FIXED_CHANNEL_CONVOLUTION_64_13/BatchNorm/AssignMovingAvg/sub_1',
    u'architectures/prior_generator_1/Assign',
    u'architectures/prior_generator_1/Read/ReadVariableOp',
    u'params/prior_generator_1/dropout_rate/Initializer/Const',
    u'params/prior_generator_1/dropout_rate',
    u'params/prior_generator_1/dropout_rate/IsInitialized/VarIsInitializedOp',
]


def _create_trials(root_dir):
  trials = [{
      'model_dir': os.path.join(root_dir, str(1)),
      'id': 1,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.97
      },
  }, {
      'model_dir': os.path.join(root_dir, str(2)),
      'id': 2,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.94
      },
  }, {
      'model_dir': os.path.join(root_dir, str(3)),
      'id': 3,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.7
      },
  }, {
      'model_dir': os.path.join(root_dir, str(4)),
      'id': 4,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.72
      },
  }, {
      'model_dir': os.path.join(root_dir, str(5)),
      'id': 5,
      'status': 'COMPLETED',
      'trial_infeasible': False,
      'final_measurement': {
          'objective_value': 0.3
      },
  }]
  return [trial_module.Trial(t) for t in trials]


class PriorGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  def _create_checkpoint(self, towers, trial_id):
    with self.test_session(graph=tf.Graph()) as sess:
      architecture = np.array([1, 3, 34])
      input_tensor = tf.zeros([100, 32, 32, 3])
      phoenix_spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      dirname = os.path.join(flags.FLAGS.test_tmpdir, str(trial_id))
      if dirname and not tf.io.gfile.exists(dirname):
        tf.io.gfile.makedirs(dirname)
      for tower in towers:
        _ = architecture_utils.construct_tower(
            phoenix_spec=phoenix_spec,
            input_tensor=input_tensor,
            tower_name=str(tower) + '_0',
            architecture=architecture,
            is_training=True,
            lengths=None,
            logits_dimension=10,
            hparams=hp.HParams(),
            model_directory=dirname,
            is_frozen=False,
            dropout_rate=None)
        architecture_utils.set_number_of_towers(tower, 1)
      architecture_utils.set_number_of_towers('replay_generator', 0)
      directory = flags.FLAGS.test_tmpdir
      saver = tf.compat.v1.train.Saver()
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      saver.save(sess, os.path.join(directory, str(trial_id)) + '/ckpt')

  @parameterized.named_parameters(
      {
          'testcase_name': '1',
          'width': 1,
          'consider': 3
      }, {
          'testcase_name': '2',
          'width': 2,
          'consider': 3
      }, {
          'testcase_name': '3',
          'width': 3,
          'consider': 3
      }, {
          'testcase_name': '4',
          'width': 1,
          'consider': 2
      }, {
          'testcase_name': '5',
          'width': 2,
          'consider': 2
      }, {
          'testcase_name': '6',
          'width': 3,
          'consider': 2
      })
  def test_nonadaptive_prior(self, width, consider):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)
      spec.ensemble_spec.nonadaptive_search.width = width
      spec.ensemble_spec.nonadaptive_search.num_trials_to_consider = consider
      spec.is_input_shared = True
      generator = prior_generator.PriorGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      fake_config = collections.namedtuple('RunConfig', ['model_dir'])
      run_config = fake_config(model_dir=flags.FLAGS.test_tmpdir + '/10000')
      tf.io.gfile.makedirs(run_config.model_dir)
      # Best three trials checkpoint are generated. If the generator chooses
      # the suboptimal (wrong) trials, the test will fail.
      self._create_checkpoint(['search_generator'], 3)
      self._create_checkpoint(['search_generator'], 4)
      self._create_checkpoint(['search_generator'], 5)
      logits, _ = generator.first_time_chief_generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=tf.zeros([100, 32, 32, 3]),
          shared_lengths=None,
          logits_dimension=10,
          hparams={},
          run_config=run_config,
          is_training=True,
          trials=_create_trials(flags.FLAGS.test_tmpdir))
      self.assertLen(logits, min(width, consider))

  def test_nonadaptive_prior_graph(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)
      spec.ensemble_spec.nonadaptive_search.width = 2
      spec.ensemble_spec.nonadaptive_search.num_trials_to_consider = 3
      spec.is_input_shared = True
      generator = prior_generator.PriorGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      fake_config = collections.namedtuple('RunConfig', ['model_dir'])
      run_config = fake_config(model_dir=flags.FLAGS.test_tmpdir + '/10000')
      tf.io.gfile.makedirs(run_config.model_dir)
      # Best three trials checkpoint are generated. If the generator chooses
      # the suboptimal (wrong) trials, the test will fail.
      self._create_checkpoint(['search_generator'], 3)
      self._create_checkpoint(['search_generator'], 4)
      self._create_checkpoint(['search_generator'], 5)
      logits, _ = generator.first_time_chief_generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=tf.zeros([100, 32, 32, 3]),
          shared_lengths=None,
          logits_dimension=10,
          hparams={},
          run_config=run_config,
          is_training=True,
          trials=_create_trials(flags.FLAGS.test_tmpdir))
      self.assertLen(logits, 2)
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_NONADAPTIVE_GRAPH_NODES, all_nodes)

  def test_intermixed_prior_graph(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec
          .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH)
      spec.ensemble_spec.intermixed_search.width = 2
      spec.ensemble_spec.intermixed_search.try_ensembling_every = 4
      spec.ensemble_spec.intermixed_search.num_trials_to_consider = 3
      spec.is_input_shared = True
      generator = prior_generator.PriorGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      fake_config = collections.namedtuple('RunConfig', ['model_dir'])
      # Should be multplication of 4.
      run_config = fake_config(model_dir=flags.FLAGS.test_tmpdir + '/10000')
      tf.io.gfile.makedirs(run_config.model_dir)
      # Best three trials checkpoint are generated. If the generator chooses
      # the suboptimal (wrong) trials, the test will fail.
      self._create_checkpoint(['search_generator'], 2)
      self._create_checkpoint(['search_generator'], 3)
      self._create_checkpoint(['search_generator'], 5)
      logits, _ = generator.first_time_chief_generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=tf.zeros([100, 32, 32, 3]),
          shared_lengths=None,
          logits_dimension=10,
          hparams={},
          run_config=run_config,
          is_training=True,
          trials=trial_utils.create_test_trials_intermixed(
              flags.FLAGS.test_tmpdir))
      self.assertLen(logits, 2)
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_NONADAPTIVE_GRAPH_NODES, all_nodes)

  def test_adaptive_prior(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH)
      spec.ensemble_spec.adaptive_search.increase_width_every = 10
      spec.is_input_shared = True
      generator = prior_generator.PriorGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      fake_config = collections.namedtuple('RunConfig', ['model_dir'])
      run_config = fake_config(model_dir=flags.FLAGS.test_tmpdir + '/10000')
      tf.io.gfile.makedirs(run_config.model_dir)
      # Best three trials checkpoint are generated. If the generator chooses
      # the suboptimal (wrong) trials, the test will fail.
      self._create_checkpoint(['prior_generator', 'search_generator'], 5)
      logits, _ = generator.first_time_chief_generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=tf.zeros([100, 32, 32, 3]),
          shared_lengths=None,
          logits_dimension=10,
          hparams={},
          run_config=run_config,
          is_training=True,
          trials=_create_trials(flags.FLAGS.test_tmpdir))
      self.assertLen(logits, 2)
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_ADAPTIVE_GRAPH_NODE, all_nodes)

  # TODO(b/172564129): Re-enable when fixed (b/150649593).
  # def test_distillation_prior(self):
  #   # Force graph mode
  #   with tf.compat.v1.Graph().as_default():
  #     spec = phoenix_spec_pb2.PhoenixSpec(
  #         problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
  #     spec.distillation_spec.distillation_type = (
  #         distillation_spec_pb2.DistillationSpec.MSE_LOGITS)
  #     spec.is_input_shared = True
  #     generator = prior_generator.PriorGenerator(spec, 'test_study',
  #                                                'test_owner')
  #     fake_config = collections.namedtuple(
  #         'RunConfig', ['model_dir'])
  #     run_config = fake_config(model_dir=flags.FLAGS.test_tmpdir + '/10000')
  #     # Generate a checkpoint for each trial.
  #     for i in range(1, 6):
  #       self._create_checkpoint(['prior_generator', 'search_generator'], i)
  #     logits, _ = generator.first_time_chief_generate(
  #         features={},
  #         input_layer_fn=lambda: None,
  #         trial_mode=trial_utils.TrialMode.DISTILLATION,
  #         shared_input_tensor=tf.zeros([100, 32, 32, 3]),
  #         shared_lengths=None,
  #         logits_dimension=10,
  #         hparams={},
  #         run_config=run_config,
  #         is_training=True,
  #         trials=_create_trials(flags.FLAGS.test_tmpdir))
  #     self.assertLen(logits, 2)
  #     all_nodes = [
  #         node.name
  #         for node in tf.compat.v1.get_default_graph().as_graph_def().node
  #     ]
  #     self.assertAllInSet(_DISTILLATION_GRAPH_NODE, all_nodes)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
