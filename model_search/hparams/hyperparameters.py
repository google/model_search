# Copyright 2021 Google LLC
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
"""A simple wrapper for kerastuner hyperparameters."""

from absl import logging
import kerastuner


class Hyperparameters(kerastuner.engine.hyperparameters.HyperParameters):
  """Small simple wrapper to override the merge function of spaces."""

  def merge(self, hps, name_prefix="", overwrite=True):
    """Merges hyperparameters into this object.

    Arguments:
      hps: A `HyperParameters` object or list of `HyperParameter` objects.
      name_prefix: A string to add to all hparams names in hps.
      overwrite: bool. Whether existing `HyperParameter`s should be overridden
        by those in `hps` with the same name and conditions.
    """
    if isinstance(hps, kerastuner.engine.hyperparameters.HyperParameters):
      hps = hps.space

    if not overwrite:
      hps = [
          hp for hp in hps
          if not self._exists(name_prefix + hp.name, hp.conditions)
      ]

    for hp in hps:
      hp.name = name_prefix + hp.name
      self._register(hp, overwrite)

