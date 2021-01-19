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
"""Losses library for Phoenix."""

import tensorflow.compat.v2 as tf


def make_regression_loss_fn():
  """Returns the Mean Squared Error loss_fn for regression."""

  def _loss_fn(labels, logits, weights=1.0):
    return tf.compat.v1.losses.mean_squared_error(
        labels=labels,
        predictions=logits,
        weights=weights,
        reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)

  return _loss_fn


def make_regression_absolute_difference_loss_fn():
  """Returns the Mean Average Error loss_fn for regression."""

  def _loss_fn(labels, logits, weights=1.0):
    return tf.compat.v1.losses.absolute_difference(
        labels=labels,
        predictions=logits,
        weights=weights,
        reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)

  return _loss_fn


def make_regression_logarithmic_loss_fn():
  """Returns Mean Squared Logarithmic Error loss_fn for regression."""

  def _loss_fn(labels, logits, weights=1.0):
    return tf.compat.v1.losses.mean_squared_error(
        labels=tf.math.log1p(tf.nn.relu(labels)),
        predictions=logits,
        weights=weights,
        reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)

  return _loss_fn


def make_multi_class_loss_fn(label_vocabulary=None):
  """Returns multi class loss function."""

  def _labels_are_one_hot(labels, logits, label_vocabulary):
    if label_vocabulary:
      return False
    elif labels.dtype.is_integer:
      return False
    elif tf.squeeze(labels).shape == logits.shape:
      return True
    else:
      return False

  def _loss_fn(labels, logits, weights=1.0):
    """Cross entropy loss fn."""

    if _labels_are_one_hot(labels, logits, label_vocabulary):
      one_hot_labels = labels
    else:

      if label_vocabulary:
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=label_vocabulary,
            values=range(len(label_vocabulary)),
            name="class_id_lookup",
            value_dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(initializer, 1, name="lookup")
        label_ids = table.lookup(labels)
      else:
        label_ids = tf.squeeze(labels)

      if label_ids.dtype == tf.float32:
        label_ids = tf.cast(label_ids, "int32")
      one_hot_labels = tf.one_hot(indices=label_ids, depth=logits.shape[-1])

    return tf.reduce_mean(
        input_tensor=tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels, logits=logits, weights=weights))

  return _loss_fn


def make_multi_label_loss_fn():
  """Returns multi label loss function."""

  def _loss_fn(labels, logits, weights=1.0):

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights=weights)

  return _loss_fn
