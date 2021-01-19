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
"""Tests for model_search.metadata.ml_metadata."""

import os
from absl import flags
from absl.testing import absltest

from model_search.metadata import ml_metadata_db
from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS


class MlMetadataTest(absltest.TestCase):

  def test_before_generating_trial_model(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = os.path.join(FLAGS.test_tmpdir, "1")
    connection_config.sqlite.connection_mode = 3
    handler = ml_metadata_db.MLMetaData(
        None, None, None, connection_config=connection_config)
    handler.before_generating_trial_model(trial_id=1, model_dir="/tmp/1")
    output = handler._store.get_executions_by_type("Trial")
    self.assertLen(output, 1)
    output = output[0]
    self.assertEqual(output.properties["id"].int_value, 1)
    self.assertEqual(output.properties["state"].string_value, "RUNNING")
    self.assertEqual(output.properties["serialized_data"].string_value, "")
    self.assertEqual(output.properties["model_dir"].string_value, "/tmp/1")
    self.assertEqual(output.properties["evaluation"].string_value, "")

  def test_report(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = os.path.join(FLAGS.test_tmpdir, "2")
    connection_config.sqlite.connection_mode = 3
    handler = ml_metadata_db.MLMetaData(
        None, None, None, connection_config=connection_config)
    handler.before_generating_trial_model(trial_id=1, model_dir="/tmp/1")
    handler.report(eval_dictionary={"loss": 0.5}, model_dir="/tmp/1")
    output = handler.get_completed_trials()
    self.assertLen(output, 1)
    output = output[0]
    self.assertEqual(output.id, 1)
    self.assertEqual(output.status, "COMPLETED")
    self.assertEqual(output.model_dir, "/tmp/1")
    self.assertEqual(output.final_measurement.objective_value, 0.5)

  def test_get_completed_trials(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = os.path.join(FLAGS.test_tmpdir, "3")
    connection_config.sqlite.connection_mode = 3
    handler = ml_metadata_db.MLMetaData(
        None, None, None, connection_config=connection_config)
    handler.before_generating_trial_model(trial_id=1, model_dir="/tmp/1")
    handler.before_generating_trial_model(trial_id=2, model_dir="/tmp/2")
    handler.report(eval_dictionary={"loss": 0.1}, model_dir="/tmp/1")
    handler.before_generating_trial_model(trial_id=3, model_dir="/tmp/3")
    handler.report(eval_dictionary={"loss": 0.3}, model_dir="/tmp/3")
    handler.report(eval_dictionary={"loss": 0.2}, model_dir="/tmp/2")
    output = handler.get_completed_trials()
    self.assertLen(output, 3)
    for i in range(3):
      self.assertEqual(output[i].status, "COMPLETED")
      self.assertEqual(output[i].model_dir, "/tmp/" + str(output[i].id))
      self.assertEqual(output[i].final_measurement.objective_value,
                       float(output[i].id) / 10)


if __name__ == "__main__":
  absltest.main()
