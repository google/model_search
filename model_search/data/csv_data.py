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
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf


class Provider(data.Provider):
  """A csv data provider."""

  def __init__(self,
               label_index,
               logits_dimension,
               record_defaults,
               filename,
               validation_filename=None,
               field_delim=","):
    self._filename = filename
    self._validation_filename = validation_filename
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
      filename = self._filename
      if self._validation_filename and mode != tf.estimator.ModeKeys.TRAIN:
        filename = self._validation_filename

      features_dataset = tf.data.experimental.CsvDataset(
          filename,
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

  def get_serving_input_fn(self, hparams):
    """Returns an `input_fn` for serving in an exported SavedModel.

    Args:
      hparams: tf.HParams object.

    Returns:
      Returns an `input_fn` that takes no arguments and returns a
        `ServingInputReceiver`.
    """
    features_ind = [
        idx for idx in self._features if idx != str(self._label_index)
    ]
    tf.compat.v1.disable_eager_execution()
    features = {
        idx: tf.compat.v1.placeholder(tf.float32, [None], idx)
        for idx in features_ind
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=features)

  def number_of_classes(self):
    return self._logits_dimension

  def get_feature_columns(self):
    """Returns feature columns."""
    features = [f for f in self._features if f != str(self._label_index)]
    feature_columns = [
        tf.feature_column.numeric_column(key=key) for key in features
    ]
    return feature_columns

  def get_keras_input(self, batch_size):
    """Returns keras input as explained in data.py module."""
    del batch_size
    dataset = pd.read_csv(self._filename)
    labels = dataset.pop(dataset.columns.values[self._label_index])
    labels = np.array(labels)
    features = np.array(dataset)

    validation_features = None
    validation_labels = None
    if self._validation_filename:
      validation_data = pd.read_csv(self._validation_filename)
      validation_labels = validation_data.pop(
          validation_data.columns.values[self._label_index])
      validation_labels = np.array(validation_labels)
      validation_features = np.array(validation_data)

    return features, labels, (validation_features, validation_labels)
