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
"""Identity search algorithm for Phoenix."""

from model_search.search import common
from model_search.search import search_algorithm


class Identity(search_algorithm.SearchAlgorithm):
  """Passes hparams suggested architecture as is."""

  def __init__(self, phoenix_spec):
    self._phoenix_spec = phoenix_spec

  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """Suggests a new architecture for a Phoenix model.

    Passes the suggested architecture from hparams as is.

    Args:
      trials: a list of metadata.trial.Trail
      hparams: The suggested hparams.
      my_trial_id: integer - the trial id which is making the call.
      model_dir: string - the model directory.

    Returns:
      an architecture: a np.array of integers and None
    """
    return common.encode_architecture(hparams.initial_architecture,
                                      self._phoenix_spec.problem_type), None
