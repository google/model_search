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
"""Library for mock metadata handler for testing."""

from model_search.metadata import metadata


class MockMetaData(metadata.MetaData):
  """An object which handles communicating with metadata db through MLMD."""

  @property
  def name(self):
    return "MockMetaData"

  def before_generating_trial_model(self, trial_id, model_dir):
    return

  def get_completed_trials(self):
    return []

  def after_generating_trial_model(self, trial_id):
    return

  def report(self, eval_dictionary, model_dir):
    return

  def get_best_k(self, trials=None, k=1, valid_only=False):
    return None
