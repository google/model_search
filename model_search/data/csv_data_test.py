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
"""Tests for model_search.single_trainer."""

import os
from absl import flags
from absl.testing import absltest
from model_search import constants
from model_search import single_trainer
from model_search.data import csv_data

FLAGS = flags.FLAGS


class SingleTrainerTest(absltest.TestCase):

  def test_try_models(self):
    # Test is source code is deployed in FLAGS.test_srcdir
    spec_path = os.path.join(FLAGS.test_srcdir, constants.DEFAULT_DNN)
    trainer = single_trainer.SingleTrainer(
        data=csv_data.Provider(
            label_index=0,
            logits_dimension=2,
            record_defaults=[0, 0, 0, 0],
            filename=os.path.join(
                FLAGS.test_srcdir,
                "model_search/model_search/data/testdata/"
                "csv_random_data.csv")),
        spec=spec_path)

    trainer.try_models(
        number_models=7,
        train_steps=10,
        eval_steps=10,
        root_dir=FLAGS.test_tmpdir,
        batch_size=2,
        experiment_name="test",
        experiment_owner="test")


if __name__ == "__main__":
  absltest.main()
