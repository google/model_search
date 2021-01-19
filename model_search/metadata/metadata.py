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
"""Small helper wrapper api to save metadata."""

import abc


class MetaData(object, metaclass=abc.ABCMeta):
  """Api for metadata storage."""

  @abc.abstractproperty
  def name(self):
    """The name of the metadata storage handler."""

  @abc.abstractmethod
  def before_generating_trial_model(self, trial_id, model_dir):
    """To be called at the beginning of model_fn for training a new trial."""

  @abc.abstractmethod
  def get_completed_trials(self):
    """Fetch all trials metadata in the database so far."""

  @abc.abstractmethod
  def after_generating_trial_model(self, trial_id):
    """To be run at the end of model_fn for a trial."""

  @abc.abstractmethod
  def get_best_k(self, trials=None, k=1, valid_only=False):
    """Return best k trials so far."""

  @abc.abstractmethod
  def report(self, eval_dictionary, model_dir):
    """Report final evaluation dictionary."""
