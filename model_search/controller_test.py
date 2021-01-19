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
"""Tests for model_search.controller."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from model_search import controller
from model_search.metadata import ml_metadata_db
from model_search.metadata import trial as trial_module
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2


def _create_spec(replay_towers=0, ensemble_type="none"):
  output = phoenix_spec_pb2.PhoenixSpec()
  output.search_type = phoenix_spec_pb2.PhoenixSpec.NONADAPTIVE_RANDOM_SEARCH
  if replay_towers:
    output.replay.CopyFrom(phoenix_spec_pb2.ArchitectureReplay())
    for _ in range(replay_towers):
      output.replay.towers.add()

  if ensemble_type == "adaptive":
    output.ensemble_spec.ensemble_search_type = (
        ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH)
  if ensemble_type == "nonadaptive":
    output.ensemble_spec.ensemble_search_type = (
        ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH)
  if ensemble_type == "intermix":
    output.ensemble_spec.ensemble_search_type = (
        ensembling_spec_pb2.EnsemblingSpec
        .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH)
  return output


_TRIALS = [trial_module.Trial({"id": id + 1}) for id in range(105)]


def _create_trials(num_trials):
  return _TRIALS[:num_trials]


class ControllerTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "search_replay",
          "spec": _create_spec(1),
          "my_id": 1,
          "trials": [],
          "expected_output": {
              "search_generator": []
          },
      }, {
          "testcase_name": "adaptive_replay",
          "spec": _create_spec(2, "adaptive"),
          "my_id": 2,
          "trials": [],
          "expected_output": {
              "search_generator": [],
              "replay_generator": [],
          },
      }, {
          "testcase_name": "nonadaptive_replay",
          "spec": _create_spec(3, "nonadaptive"),
          "my_id": 3,
          "trials": [],
          "expected_output": {
              "replay_generator": [],
          },
      }, {
          "testcase_name": "search",
          "spec": _create_spec(0, "none"),
          "my_id": 3,
          "trials": _create_trials(2),
          "expected_output": {
              "search_generator": _create_trials(2),
          },
      }, {
          "testcase_name": "adaptive_ensemble_search",
          "spec": _create_spec(0, "adaptive"),
          "my_id": 3,
          "trials": _create_trials(2),
          "expected_output": {
              "search_generator": _create_trials(2),
          },
      }, {
          "testcase_name": "adaptive_ensemble_search_itr2",
          "spec": _create_spec(0, "adaptive"),
          "my_id": 7,
          "trials": _create_trials(6),
          "expected_output": {
              "search_generator": [t for t in _create_trials(6) if t.id > 5],
              "prior_generator": [t for t in _create_trials(6) if t.id <= 5]
          },
      }, {
          "testcase_name": "nonadaptive_ensemble_search",
          "spec": _create_spec(0, "nonadaptive"),
          "my_id": 7,
          "trials": _create_trials(6),
          "expected_output": {
              "search_generator": _create_trials(6),
          },
      }, {
          "testcase_name": "nonadaptive_ensemble_search_full_pool",
          "spec": _create_spec(0, "nonadaptive"),
          "my_id": 105,
          "trials": _create_trials(104),
          "expected_output": {
              "prior_generator": _create_trials(100),
          },
      }, {
          "testcase_name": "intermix_ensemble_search",
          "spec": _create_spec(0, "intermix"),
          "my_id": 7,
          "trials": _create_trials(6),
          "expected_output": {
              "search_generator":
                  [t for t in _create_trials(6) if t.id % 5 != 0],
          },
      }, {
          "testcase_name": "intermix_ensemble_search_ensembling",
          "spec": _create_spec(0, "intermix"),
          "my_id": 105,
          "trials": _create_trials(104),
          "expected_output": {
              "prior_generator":
                  [t for t in _create_trials(104) if t.id % 5 != 0],
          },
      })
  def test_controller(self, spec, my_id, trials, expected_output):
    controller_ = controller.InProcessController(
        phoenix_spec=spec,
        metadata=ml_metadata_db.MLMetaData(
            phoenix_spec=spec, study_name="", study_owner=""))
    generators = controller_.get_generators(my_id, trials)
    logging.info(generators)
    self.assertEqual(len(expected_output.keys()), len(generators.keys()))
    for k, v in generators.items():
      self.assertEqual(k, v.instance.generator_name())
      self.assertIn(k, expected_output.keys())
      self.assertEqual(v.relevant_trials, expected_output[k])


if __name__ == "__main__":
  absltest.main()
