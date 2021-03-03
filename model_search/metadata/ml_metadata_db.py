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
"""Library to save metadata with MLMD."""

import json
import os
import random
from absl import flags
from absl import logging

from model_search.metadata import metadata
from model_search.metadata import trial
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


flags.DEFINE_string(
    "mlmd_default_sqllite_filename", None,
    "If specified and ConnectionConfig is not passed to object below, a sqllite"
    " config with this filename is established.")

### If one flag is set below, then all four flags below should be set.
### If one flag is set below, then the flag above should be None
flags.DEFINE_string(
    "mlmd_socket", None,
    "If specified and ConnectionConfig is not passed to object below, a mysql "
    "config with this socket is established.")
flags.DEFINE_string(
    "mlmd_database", None,
    "If specified and ConnectionConfig is not passed to object below, a mysql"
    "config with this database is established.")
flags.DEFINE_string(
    "mlmd_user", None,
    "If specified and ConnectionConfig is not passed to object below, a mysql "
    "config with this user is established.")
flags.DEFINE_string(
    "mlmd_password", None,
    "If specified and ConnectionConfig is not passed to object below, a mysql "
    "config with this password is established.")

FLAGS = flags.FLAGS


class MLMetaData(metadata.MetaData):
  """An object which handles communicating with metadata db through MLMD."""

  def __init__(self,
               phoenix_spec,
               study_name,
               study_owner,
               optimization_goal="minimize",
               optimization_metric="loss",
               connection_config=None):
    """Initializes a new MLMD connection instance.

    Args:
      phoenix_spec: PhoenixSpec proto.
      study_name: The name of the study.
      study_owner: The owner (username) of the study.
      optimization_goal: minimize or maximize (string).
      optimization_metric: what metric are we optimizing (string).
      connection_config: a metadata_store_pb2.ConnectionConfig() proto. If None,
        we fall back on the flags above.
    """
    self._study_name = study_name
    self._study_owner = study_owner
    self._phoenix_spec = phoenix_spec
    self._optimization_goal = optimization_goal
    self._optimization_metric = optimization_metric

    self._connection_config = connection_config
    if not FLAGS.is_parsed():
      logging.error("Flags are not parsed. Using default in file mlmd database."
                    " Please run main with absl.app.run(main) to fix this. "
                    "If running in distributed mode, this means that the "
                    "trainers are not sharing information between one another.")
    if self._connection_config is None:
      if FLAGS.is_parsed() and FLAGS.mlmd_default_sqllite_filename:
        self._connection_config = metadata_store_pb2.ConnectionConfig()
        self._connection_config.sqlite.filename_uri = (
            FLAGS.mlmd_default_sqllite_filename)
        self._connection_config.sqlite.connection_mode = 3
      elif FLAGS.is_parsed() and FLAGS.mlmd_socket:
        self._connection_config = metadata_store_pb2.ConnectionConfig()
        self._connection_config.mysql.socket = FLAGS.mlmd_socket
        self._connection_config.mysql.database = FLAGS.mlmd_database
        self._connection_config.mysql.user = FLAGS.mlmd_user
        self._connection_config.mysql.password = FLAGS.mlmd_password
      else:
        self._connection_config = metadata_store_pb2.ConnectionConfig()
        self._connection_config.sqlite.filename_uri = (
            "/tmp/filedb-%d" % random.randint(0, 1000000))
        self._connection_config.sqlite.connection_mode = 3
    self._store = metadata_store.MetadataStore(self._connection_config)

    trial_type = metadata_store_pb2.ExecutionType()
    trial_type.name = "Trial"
    trial_type.properties["id"] = metadata_store_pb2.INT
    trial_type.properties["state"] = metadata_store_pb2.STRING
    trial_type.properties["serialized_data"] = metadata_store_pb2.STRING
    trial_type.properties["model_dir"] = metadata_store_pb2.STRING
    trial_type.properties["evaluation"] = metadata_store_pb2.STRING
    self._trial_type_id = self._store.put_execution_type(trial_type)
    self._trial_id_to_run_id = {}

  @property
  def name(self):
    return "MLMetaData"

  def before_generating_trial_model(self, trial_id, model_dir):
    trial_run = metadata_store_pb2.Execution()
    trial_run.type_id = self._trial_type_id
    trial_run.properties["id"].int_value = trial_id
    trial_run.properties["state"].string_value = "RUNNING"
    trial_run.properties["serialized_data"].string_value = ""
    trial_run.properties["model_dir"].string_value = model_dir
    trial_run.properties["evaluation"].string_value = ""
    run_id = self._store.put_executions([trial_run])[0]
    self._trial_id_to_run_id[trial_id] = (run_id, trial_run)
    return

  def _convert_to_trial_object(self, trial_str):
    """Returns a Trial object like object."""
    evaluation_dictionary = json.loads(
        trial_str.properties["evaluation"].string_value)
    data_as_json = {
        "id": trial_str.properties["id"].int_value,
        "status": trial_str.properties["state"].string_value,
        "final_measurement": {
            "objective_value": evaluation_dictionary[self._optimization_metric]
        },
        "model_dir": trial_str.properties["model_dir"].string_value,
        "trial_infeasible": False
    }
    return trial.Trial(data_as_json)

  def get_completed_trials(self):
    trials = self._store.get_executions_by_type("Trial")
    trials = [
        self._convert_to_trial_object(t)
        for t in trials
        if t.properties["state"].string_value == "COMPLETED"
    ]
    return trials

  def after_generating_trial_model(self, trial_id):
    return

  def report(self, eval_dictionary, model_dir):
    logging.info("Storing the following evaluation dictionary,")
    logging.info(eval_dictionary)
    logging.info("For the model in the following model dictionary,")
    logging.info(model_dir)
    trial_id = int(os.path.basename(model_dir))
    id_, trial_run = self._trial_id_to_run_id[trial_id]
    trial_run.id = id_
    trial_run.properties["state"].string_value = "COMPLETED"
    trial_run.properties["evaluation"].string_value = json.dumps(
        eval_dictionary)
    self._store.put_executions([trial_run])
    del self._trial_id_to_run_id[trial_id]
    return

  def get_best_k(self, trials=None, k=1, valid_only=False):
    trials_ = trials if trials is not None else self.get_completed_trials()
    return trial.get_best_k(
        trials=trials_,
        k=k,
        status_whitelist=["COMPLETED"],
        optimization_goal=self._optimization_goal)
