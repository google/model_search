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

flags.DEFINE_string(
    'input_dir', '', 'The path containing the input data. Should provide a dir '
    'that has 0 and 1 as subdirs.')

flags.DEFINE_float('eval_fraction', 0.2,
                   'The amount of data (fraction) to hold for evaluation.')

flags.DEFINE_integer('image_height', 320,
                     'The height (dimension) of the image.')

flags.DEFINE_integer('image_width', 240, 'The width (dimension) of the image.')

FLAGS = flags.FLAGS


@data.register_provider(lookup_name='image_data_provider', init_args={})
class Provider(data.Provider):
  """A csv data provider."""

  def __init__(self):
    self._input_dir = FLAGS.input_dir
    self._image_width = FLAGS.image_width
    self._image_height = FLAGS.image_height
    self._eval_fraction = FLAGS.eval_fraction

    if '${TEST_SRCDIR}' in self._input_dir:
      self._input_dir = self._input_dir.replace('${TEST_SRCDIR}',
                                                FLAGS.test_srcdir)

  def get_input_fn(self, hparams, mode, batch_size):
    """See `data.Provider` get_input_fn."""
    del hparams

    def input_fn(params=None):
      del params
      split = ('training'
               if mode == tf.estimator.ModeKeys.TRAIN else 'validation')

      dataset = tf.keras.preprocessing.image_dataset_from_directory(
          directory=self._input_dir,
          labels='inferred',
          label_mode='binary',
          class_names=None,
          color_mode='rgb',
          batch_size=batch_size,
          image_size=(self._image_height, self._image_width),
          shuffle=True,
          seed=73,
          validation_split=self._eval_fraction,
          subset=split,
          interpolation='bilinear',
          follow_links=False)

      if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

      return dataset

    return input_fn

  def get_serving_input_fn(self, hparams):
    """Returns an `input_fn` for serving in an exported SavedModel.

    Args:
      hparams: tf.HParams object.

    Returns:
      Returns an `input_fn` that takes no arguments and returns a
        `ServingInputReceiver`.
    """
    tf.compat.v1.disable_eager_execution()
    features = {
        'image':
            tf.compat.v1.placeholder(
                tf.float32, [None, self._image_height, self._image_width, 3],
                'image')
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=features)

  def number_of_classes(self):
    return 2

  def get_feature_columns(self):
    """Returns feature columns."""
    feature_columns = [
        tf.feature_column.numeric_column(
            key='image', shape=(self._image_height, self._image_width, 3))
    ]
    return feature_columns

  def get_keras_input(self, batch_size):
    """Returns keras input as explained in data.py module."""
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=self._input_dir,
        labels='inferred',
        label_mode='binary',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(self._image_height, self._image_width),
        shuffle=True,
        seed=73,
        validation_split=self._eval_fraction,
        subset='training',
        interpolation='bilinear',
        follow_links=False)

    train_dataset = train_dataset.cache().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    validation = tf.keras.preprocessing.image_dataset_from_directory(
        directory=self._input_dir,
        labels='inferred',
        label_mode='binary',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(self._image_height, self._image_width),
        shuffle=True,
        seed=73,
        validation_split=self._eval_fraction,
        subset='validation',
        interpolation='bilinear',
        follow_links=False)

    return train_dataset, None, validation
