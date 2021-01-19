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
"""Module containing the interface for logit bundlers."""

import abc


class LogitBundler(object, metaclass=abc.ABCMeta):
  """Interface for bundling together logits from different towers."""

  @abc.abstractmethod
  def bundle_logits(self,
                    priors_logits_specs,
                    search_logits_specs,
                    logits_dimension=None):
    """Bundles the logits from the priors and the search candidate.

    Args:
      priors_logits_specs: List of LogitSpecs associated with the prior towers.
      search_logits_specs: List containing the LogitSpecs associated with the
        search (new) tower. (Empty if there is no search tower.)
      logits_dimension: The dimension of the logits.

    Returns:
      Named tuple holding relevant logits.
    """
