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
"""Api for a search algorithm in Phoenix."""

import abc


class SearchAlgorithm(object, metaclass=abc.ABCMeta):
  """Search Algorithm api for creating a new architecture to try in Phoenix."""

  @abc.abstractmethod
  def get_suggestion(self, trials, hparams, my_trial_id=None, model_dir=None):
    """Suggests a new architecture for a Phoenix model.

    Note that this algorithm performs on top of hparams oracle. Meaning, it will
    receive suggested trial hparams, and determine the Phoenix
    architecture. This algorithm has the final say. We implemented a simple
    "Identity" algorithm, that passes oracle's suggestion as is, to benefit
    from the various hyper-parameter optimization algorithms already implemented
    as oracles.

    Args:
      trials: a list of metadata.trial.Trail
      hparams: The oracle's suggested hparams.
      my_trial_id: integer - the trial id which is making the call.
      model_dir: string - the model directory.

    Returns:
      (np.array(int), int) - an architecture a np.array of integers and an
      optional int representing the trial we forked from for snapshotting. Use
      None, if you are not forking from an existing trial.
    """
