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
"""Tests for model_search.generators.replay_generator."""

from absl.testing import absltest
import mock

from model_search.generators import replay_generator
from model_search.metadata import ml_metadata_db
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf


def _fake_import_fn(**kwargs):
  return kwargs


def _create_spec(ensemble_type='none'):
  output = phoenix_spec_pb2.PhoenixSpec()
  if ensemble_type == 'adaptive':
    output.ensemble_spec.ensemble_search_type = (
        ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH)
  if ensemble_type == 'nonadaptive':
    output.ensemble_spec.ensemble_search_type = (
        ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)
  return output


class ReplayGeneratorTest(absltest.TestCase):

  @mock.patch('model_search.generators.trial_utils' '.import_towers_one_trial')
  def test_first_time_chief_generate_adaptive(self, import_fn):
    import_fn.side_effect = _fake_import_fn
    generator = replay_generator.ReplayGenerator(
        phoenix_spec=_create_spec('adaptive'),
        metadata=ml_metadata_db.MLMetaData(
            _create_spec('adaptive'), study_owner='test', study_name='test'))
    call_args = generator.first_time_chief_generate(
        features='features',
        input_layer_fn='input_layer',
        trial_mode='mode',
        shared_input_tensor='input',
        shared_lengths='input2',
        logits_dimension=5,
        hparams=None,
        run_config=tf.estimator.RunConfig(model_dir='mydir/5'),
        is_training=True,
        trials=[])
    self.assertEqual(
        call_args, {
            'features': 'features',
            'input_layer_fn': 'input_layer',
            'phoenix_spec': _create_spec('adaptive'),
            'shared_input_tensor': 'input',
            'shared_lengths': 'input2',
            'is_training': True,
            'logits_dimension': 5,
            'prev_model_dir': 'mydir/4',
            'force_freeze': True,
            'allow_auxiliary_head': False,
            'caller_generator': 'replay_generator',
            'my_model_dir': 'mydir/5'
        })

  @mock.patch('model_search.generators.trial_utils'
              '.import_towers_multiple_trials')
  def test_first_time_chief_generate_nonadaptive(self, import_fn):
    import_fn.side_effect = _fake_import_fn
    generator = replay_generator.ReplayGenerator(
        phoenix_spec=_create_spec('nonadaptive'),
        metadata=ml_metadata_db.MLMetaData(
            _create_spec('nonadaptive'), study_owner='test', study_name='test'))
    call_args = generator.first_time_chief_generate(
        features='features',
        input_layer_fn='input_layer',
        trial_mode='mode',
        shared_input_tensor='input',
        shared_lengths='input2',
        logits_dimension=5,
        hparams=None,
        run_config=tf.estimator.RunConfig(model_dir='mydir/5'),
        is_training=True,
        trials=[])
    self.assertEqual(
        call_args, {
            'features': 'features',
            'input_layer_fn': 'input_layer',
            'phoenix_spec': _create_spec('nonadaptive'),
            'shared_input_tensor': 'input',
            'shared_lengths': 'input2',
            'is_training': True,
            'logits_dimension': 5,
            'previous_model_dirs': ['mydir/1', 'mydir/2', 'mydir/3', 'mydir/4'],
            'force_freeze': True,
            'allow_auxiliary_head': False,
            'caller_generator': 'replay_generator',
            'my_model_dir': 'mydir/5'
        })


if __name__ == '__main__':
  absltest.main()
