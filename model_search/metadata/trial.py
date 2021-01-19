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
"""A phoenix trial library."""

import collections
import json



def get_best_k(trials,
               k=1,
               status_whitelist=None,
               optimization_goal='minimize'):
  """Returns the top k trials sorted by objective_value.

  Args:
    trials: The trials (Trail objects) to sort and return the top_k of.
    k: The top k trials to return. If k=1 we don't return a list.
    status_whitelist: list of statuses to whitelist. If None, we use all trials.
    optimization_goal: string, minimize or maximize.

  Returns:
    The top k trials (list unless k=1) sorted by objective_value.
  """
  trials_ = trials
  valid_status = status_whitelist
  if valid_status is not None:
    trials_ = [
        trial for trial in trials_
        if trial.status in valid_status and not trial.trial_infeasible
    ]
  if not trials_:
    return None

  maximizing = optimization_goal != 'minimize'
  top = sorted(
      trials_,
      key=lambda t: t.final_measurement.objective_value,
      reverse=maximizing)
  return top[:k] if k > 1 else top[0]


def _json_to_python_object(data):
  """Takes nested dictionaries and return a python object.

  Args:
    data: a nested dictionary.

  Returns:
    A python object with the dictionary values as attributes

  Example:
  Given the dictionary {"name": "john", "license": {"class": "d"}}
  returns an object output, where output.license.class is equal to "d" and
  output.name is "john"
  """
  data_string = json.dumps(data)
  return json.loads(
      data_string,
      object_hook=lambda d: collections.namedtuple('X', d.keys())(*d.values()))


class Trial(object):
  """A Phoenix Trial wrapper. Stores trial metadata."""

  def __init__(self, trial_data):
    if isinstance(trial_data, dict):
      # mlmd (json dictionary)
      self._internal_trial_representation = _json_to_python_object(trial_data)

  def __getattr__(self, name):
    attribute_requested = getattr(self._internal_trial_representation, name,
                                  None)
    if attribute_requested is not None:
      return attribute_requested

  def is_completed(self):
    valid_statuses = ('COMPLETED',)
    return self._internal_trial_representation.status in valid_statuses

  def is_completed_or_deleted(self):
    valid_statuses = ('COMPLETED', 'DELETED')
    return self._internal_trial_representation.status in valid_statuses

  def final_objective_measurement(self):
    return self._internal_trial_representation.final_measurement.objective_value
