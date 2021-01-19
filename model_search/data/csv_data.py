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
"""Simple csv reader for small classification problems."""

from model_search.data import data
import tensorflow.compat.v2 as tf


class Provider(data.Provider):
  """A csv data provider."""

  def __init__(self,
               label_index,
               logits_dimension,
               record_defaults,
               filename,
               field_delim=","):
    self._filename = filename
    self._logits_dimension = logits_dimension
    self._record_defaults = record_defaults
    self._field_delim = field_delim
    self._label_index = label_index
    # Indices of the features
    self._features = [str(i) for i in range(len(record_defaults))]

  def get_input_fn(self, hparams, mode, batch_size):
    """See `data.Provider` get_input_fn."""
    del hparams

    def input_fn(params=None):
      """Provides batches of data."""
      del params

      features_dataset = tf.data.experimental.CsvDataset(
          self._filename,
          record_defaults=self._record_defaults,
          header=True,
          field_delim=self._field_delim,
          use_quote_delim=True)

      def _map_fn(*args):
        features = {str(i): tensor for i, tensor in enumerate(args)}
        label = features.pop(str(self._label_index))
        return features, label

      features_dataset = features_dataset.map(_map_fn).prefetch(
          tf.data.experimental.AUTOTUNE).batch(batch_size)
      if mode == tf.estimator.ModeKeys.TRAIN:
        features_dataset = features_dataset.repeat().shuffle(100 * batch_size)

      return features_dataset

    return input_fn

  def _get_feature_spec(self):
    return tf.estimator.classifier_parse_example_spec(
        self.get_feature_columns(),
        label_key=str(self._label_index),
        label_dtype=tf.int32)

  def get_serving_input_fn(self, hparams):
    """Returns an `input_fn` for serving in an exported SavedModel.

    Args:
      hparams: tf.HParams object.

    Returns:
      Returns an `input_fn` that takes no arguments and returns a
        `ServingInputReceiver`.
    """

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        self._get_feature_spec())

  def number_of_classes(self):
    return self._logits_dimension

  def get_feature_columns(self):
    """Returns feature columns."""
    features = [f for f in self._features if f != str(self._label_index)]
    feature_columns = [
        tf.feature_column.numeric_column(key=key) for key in features
    ]
    return feature_columns
