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
"""Tests for model_search.metadata.trial."""

from absl.testing import absltest
from model_search.metadata import trial


def _create_trials():
  trials = [
      trial.Trial({
          "id": 99,
          "model_dir": "99",
          "status": "COMPLETED",
          "trial_infeasible": False,
          "final_measurement": {
              "objective_value": 0.97
          },
      }),
      trial.Trial({
          "id": 97,
          "model_dir": "97",
          "status": "COMPLETED",
          "trial_infeasible": False,
          "final_measurement": {
              "objective_value": 0.94
          },
      }),
      trial.Trial({
          "id": 96,
          "model_dir": "96",
          "status": "COMPLETED",
          "trial_infeasible": True,
          "final_measurement": {
              "objective_value": 0.7
          },
      }),
      trial.Trial({
          "id": 95,
          "model_dir": "95",
          "status": "COMPLETED",
          "trial_infeasible": False,
          "final_measurement": {
              "objective_value": 0.72
          },
      }),
  ]
  return trials


class TrialTest(absltest.TestCase):

  def test_get_best_k(self):
    self.assertEqual(trial.get_best_k(_create_trials()).id, 96)
    self.assertEqual(
        trial.get_best_k(_create_trials(), status_whitelist=["COMPLETED"]).id,
        95)
    self.assertEqual(
        trial.get_best_k(_create_trials(), optimization_goal="maximize").id, 99)
    self.assertLen(trial.get_best_k(_create_trials(), k=2), 2)

  def test_fields(self):
    output = _create_trials()[0]
    self.assertEqual(output.id, 99)
    self.assertEqual(output.final_measurement.objective_value, 0.97)

  def test_util_fns(self):
    output = _create_trials()[0]
    self.assertEqual(output.is_completed(), True)
    self.assertEqual(output.is_completed_or_deleted(), True)
    self.assertEqual(output.final_objective_measurement(), 0.97)


if __name__ == "__main__":
  absltest.main()
