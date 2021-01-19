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
"""Utilities shared across tests in search submodule of Phoenix."""

import collections

import numpy as np

from model_search.proto import phoenix_spec_pb2

FakeMeasurement = collections.namedtuple("FakeMeasurement", ["objective_value"])
FakeTrial = collections.namedtuple(
    "FakeTrial", ["id", "status", "trial_infeasible", "final_measurement"])


def create_spec(problem_type,
                complexity_thresholds=None,
                max_depth=None,
                min_depth=None,
                blocks_to_use=None):
  """Creates a phoenix_spec_pb2.PhoenixSpec with the given options."""
  output = phoenix_spec_pb2.PhoenixSpec()
  if complexity_thresholds is not None:
    output.increase_complexity_minimum_trials[:] = complexity_thresholds
  if max_depth is not None:
    output.maximum_depth = max_depth
  if min_depth is not None:
    output.minimum_depth = min_depth
  output.problem_type = problem_type
  if blocks_to_use:
    output.blocks_to_use[:] = blocks_to_use
  return output


def is_mutation_or_equal(previous_architecture, new_architecture):
  """Returns whether if new arch is mutation of or equal to previous arch."""
  if previous_architecture.shape != new_architecture.shape:
    return False
  mismatch = (
      previous_architecture.size -
      np.sum(previous_architecture == new_architecture))
  return mismatch <= 1
