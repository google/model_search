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

from absl import flags

from model_search.data import data
import tensorflow.compat.v2 as tf

flags.DEFINE_integer('label_index', 0,
                     'The index of the label column in the csv.')

flags.DEFINE_integer('logits_dimension', 2,
                     'The logits dimension. Use 2 for binary classification.')

flags.DEFINE_string(
    'record_defaults', '0,0,0,0',
    'The comma separated list of floats for defaults (every column).')

flags.DEFINE_string('filename', '', 'The filename of the csv file.')

flags.DEFINE_string('field_delim', ',', 'The delimiter used in the csv file.')

FLAGS = flags.FLAGS


@data.register_provider(lookup_name='csv_data_provider', init_args={})
class Provider(data.Provider):
  """A csv data provider."""

  def __init__(self):
    self._filename = FLAGS.filename
    self._logits_dimension = FLAGS.logits_dimension
    self._record_defaults = [float(i) for i in FLAGS.record_defaults.split(',')]
    self._field_delim = FLAGS.field_delim
    self._label_index = FLAGS.label_index
    # Indices of the features
    self._features = [str(i) for i in range(len(self._record_defaults))]

    # For testing only.
    if '${TEST_SRCDIR}' in self._filename:
      self._filename = self._filename.replace('${TEST_SRCDIR}',
                                              FLAGS.test_srcdir)

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
