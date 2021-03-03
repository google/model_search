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
"""Tests for model_search.generators.

search_candidate_generator.
"""

import collections
import os

from absl import flags

from model_search import hparam as hp
from model_search.architecture import architecture_utils
from model_search.generators import search_candidate_generator
from model_search.generators import trial_utils
from model_search.metadata import ml_metadata_db
from model_search.metadata import trial as trial_module
from model_search.proto import distillation_spec_pb2
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2
from model_search.proto import transfer_learning_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf


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
  }]
  return [trial_module.Trial(t) for t in trials]


_FIRST_GRAPH_NODE_SUBSET = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/weights',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/biases',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/LeakyRelu',
    u'Phoenix/search_generator_0/last_dense_4/dense/kernel',
    u'Phoenix/search_generator_0/last_dense_4/dense/bias',
    u'Phoenix/search_generator_0/last_dense_4/logits',
    u'architectures/search_generator_0/Initializer/Const',
    u'params/search_generator_0/dropout_rate',
    u'params/search_generator_0/is_frozen',
    u'number_of_towers/search_generator',
]

_SUGGESTIONS_GRAPH_NODE_SUBSET = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/search_generator_0/1_DILATED_CONVOLUTION_4_20/Conv/weights',
    u'Phoenix/search_generator_0/1_DILATED_CONVOLUTION_4_20/Conv/biases',
    u'Phoenix/search_generator_0/1_DILATED_CONVOLUTION_4_20/LeakyRelu',
    u'Phoenix/search_generator_0/last_dense_20/dense/kernel',
    u'Phoenix/search_generator_0/last_dense_20/dense/bias',
    u'Phoenix/search_generator_0/last_dense_20/logits',
    u'architectures/search_generator_0/Initializer/Const',
    u'params/search_generator_0/dropout_rate',
    u'params/search_generator_0/is_frozen',
    u'number_of_towers/search_generator',
]

_DROPOUT_GRAPH_NODE = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/weights',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/biases',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/LeakyRelu',
    u'Phoenix/search_generator_0/last_dense_4/dense/kernel',
    u'Phoenix/search_generator_0/last_dense_4/dense/bias',
    u'Phoenix/search_generator_0/last_dense_4/logits',
    u'architectures/search_generator_0/Initializer/Const',
    u'params/search_generator_0/dropout_rate',
    u'params/search_generator_0/is_frozen',
    u'number_of_towers/search_generator',
]

_DISTILLATION_GRAPH_NODE_SUBSET = [
    u'zeros/shape_as_tensor',
    u'zeros/Const',
    u'zeros',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/weights/Initializer/random_uniform',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/Conv/weights',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/BatchNorm/Const',
    u'Phoenix/search_generator_0/1_CONVOLUTION_3X3_4/BatchNorm/beta',
    u'Phoenix/search_generator_0/last_dense_4/dense/kernel/Initializer/random_uniform',
    u'Phoenix/search_generator_0/last_dense_4/dense/kernel',
    u'Phoenix/search_generator_0/last_dense_4/dense/bias',
    u'number_of_towers/search_generator',
]


class SearchCandidateGeneratorTest(tf.test.TestCase):

  def _create_checkpoint(self, towers, trial_id):
    with self.test_session(graph=tf.Graph()) as sess:
      architecture = np.array([4])
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
      directory = flags.FLAGS.test_tmpdir
      saver = tf.compat.v1.train.Saver()
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      saver.save(sess, os.path.join(directory, str(trial_id)) + '/ckpt')

  def test_generator(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
      spec.is_input_shared = True
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      input_tensor = tf.zeros([20, 32, 32, 3])
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/1', is_chief=True)
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.NO_PRIOR,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(initial_architecture=['CONVOLUTION_3X3']),
          run_config=run_config,
          is_training=True,
          trials=[])
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_FIRST_GRAPH_NODE_SUBSET, all_nodes)

  def test_generator_with_suggestions(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)
      spec.is_input_shared = True
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      input_tensor = tf.zeros([20, 32, 32, 3])
      suggestion = spec.user_suggestions.add()
      suggestion.architecture[:] = ['DILATED_CONVOLUTION_4']
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/1', is_chief=True)
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(initial_architecture=['CONVOLUTION_3X3']),
          run_config=run_config,
          is_training=True,
          trials=[])
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_SUGGESTIONS_GRAPH_NODE_SUBSET, all_nodes)

  def test_generator_with_dropouts(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
      spec.is_input_shared = True
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      input_tensor = tf.zeros([20, 32, 32, 3])
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/1', is_chief=True)
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.NO_PRIOR,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(
              initial_architecture=['CONVOLUTION_3X3'], dropout_rate=0.3),
          run_config=run_config,
          is_training=True,
          trials=[])
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_DROPOUT_GRAPH_NODE, all_nodes)

  def test_generator_with_snapshot(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.ADAPTIVE_COORDINATE_DESCENT
      spec.transfer_learning_spec.transfer_learning_type = (
          transfer_learning_spec_pb2.TransferLearningSpec
          .SNAPSHOT_TRANSFER_LEARNING)
      spec.is_input_shared = True
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      input_tensor = tf.zeros([20, 32, 32, 3])
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      tf.io.gfile.makedirs(flags.FLAGS.test_tmpdir + '/3')
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/3', is_chief=True)
      self._create_checkpoint(['search_generator'], 2)
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.ENSEMBLE_SEARCH,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(
              initial_architecture=['CONVOLUTION_3X3'],
              dropout_rate=0.3,
              new_block_type='CONVOLUTION_3X3'),
          run_config=run_config,
          is_training=True,
          trials=_create_trials(flags.FLAGS.test_tmpdir))
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_DROPOUT_GRAPH_NODE, all_nodes)

  def test_generator_with_distillation(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.DNN)
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
      spec.distillation_spec.distillation_type = (
          distillation_spec_pb2.DistillationSpec.DistillationType.MSE_LOGITS)
      spec.is_input_shared = True
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      input_tensor = tf.zeros([20, 32, 32, 3])
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/1', is_chief=True)
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.DISTILLATION,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(initial_architecture=['CONVOLUTION_3X3']),
          run_config=run_config,
          is_training=True,
          trials=[])
      all_nodes = [
          node.name
          for node in tf.compat.v1.get_default_graph().as_graph_def().node
      ]
      self.assertAllInSet(_DISTILLATION_GRAPH_NODE_SUBSET, all_nodes)

  def test_generator_with_distillation_and_intermixed(self):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec(
          problem_type=phoenix_spec_pb2.PhoenixSpec.CNN)
      spec.is_input_shared = True
      spec.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
      spec.ensemble_spec.ensemble_search_type = (
          ensembling_spec_pb2.EnsemblingSpec
          .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH)
      spec.ensemble_spec.intermixed_search.width = 2
      spec.ensemble_spec.intermixed_search.try_ensembling_every = 4
      spec.ensemble_spec.intermixed_search.num_trials_to_consider = 3
      spec.distillation_spec.distillation_type = (
          distillation_spec_pb2.DistillationSpec.DistillationType.MSE_LOGITS)
      generator = search_candidate_generator.SearchCandidateGenerator(
          phoenix_spec=spec,
          metadata=ml_metadata_db.MLMetaData(
              phoenix_spec=spec, study_name='', study_owner=''))
      fake_config = collections.namedtuple('RunConfig',
                                           ['model_dir', 'is_chief'])
      run_config = fake_config(
          model_dir=flags.FLAGS.test_tmpdir + '/10000', is_chief=True)
      tf.io.gfile.makedirs(run_config.model_dir)

      self._create_checkpoint(['search_generator'], 2)
      self._create_checkpoint(['search_generator'], 3)
      self._create_checkpoint(['search_generator'], 5)
      input_tensor = tf.zeros([20, 32, 32, 3])
      _ = generator.generate(
          features={},
          input_layer_fn=lambda: None,
          trial_mode=trial_utils.TrialMode.DISTILLATION,
          shared_input_tensor=input_tensor,
          shared_lengths=None,
          logits_dimension=10,
          hparams=hp.HParams(initial_architecture=['CONVOLUTION_3X3']),
          run_config=run_config,
          is_training=True,
          trials=trial_utils.create_test_trials_intermixed(
              flags.FLAGS.test_tmpdir))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
