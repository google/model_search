# Copyright 2021 Google LLC
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
"""Tests for model_search.controller."""

from absl.testing import absltest
from model_search.hparams import hyperparameters


class HyperparametersTest(absltest.TestCase):

  def test_merge(self):
    hps = hyperparameters.Hyperparameters()
    hps.Int('output_size', 50, 100)
    hps.Int('memory_size', 4, 32)
    hps_two = hyperparameters.Hyperparameters()
    hps_two.merge(hps, 'TUNABLE1_')
    self.assertIn('TUNABLE1_output_size', hps_two)
    self.assertIn('TUNABLE1_memory_size', hps_two)
    self.assertNotIn('TUNABLE2_output_size', hps_two)
    self.assertNotIn('TUNABLE2_memory_size', hps_two)
    hps_three = hyperparameters.Hyperparameters()
    hps_three.Int('output_size', 50, 100)
    hps_three.Int('memory_size', 4, 32)
    hps_two.merge(hps_three, 'TUNABLE2_')
    self.assertIn('TUNABLE1_output_size', hps_two)
    self.assertIn('TUNABLE1_memory_size', hps_two)
    self.assertIn('TUNABLE2_output_size', hps_two)
    self.assertIn('TUNABLE2_memory_size', hps_two)



if __name__ == '__main__':
  absltest.main()
