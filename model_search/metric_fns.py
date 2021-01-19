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

# List as: python3
"""Metrics library for Phoenix."""

from absl import logging
import tensorflow.compat.v2 as tf


def make_accuracy_metric_fn(label_vocabulary=None):
  """Makes a metric_fn for accuracy from an optional label_vocabulary.

  Args:
    label_vocabulary: A 1-D string Tensor or string list (in the single task
      setup); or a dictionary mapping string keys to those (in the multi task
      setup). The string keys correspond to the task name, allowing different
      tasks to have different label vocabularies. If `label_vocabulary` does not
      contain the task name as a key, or `label_vocabulary` is None, then the
      task is assumed to have int64 labels.

  Returns:
    A function that takes `labels` and `predictions`, and returns a dictionary
    mapping metric names to (tensor, update_op) tuples.
  """

  def _metric_fn(labels, predictions, weights=None):
    """Metrics for tensorboard.

    Args:
      labels: A int64 Tensor or a string Tensor; or a dictionary mapping task
        names (strings) to those. If a task name maps to a string Tensor, then
        label_vocabulary needs to contain that task name as a key as well,
        otherwise the task name would not have metrics computed for it.
      predictions: A dictionary mapping strings to Tensors. This dictionary
        always contains a `predictions` string key. For regression, the
        `predictions` key maps to a tf.float32 Tensor. For classification, the
        `predictions` key maps to a tf.int64 Tensor, and the dictionary also
        contains `probabilities` and `log_probabilities` string keys which map
        to tf.float32 Tensors. In the multi task setup, the dictionary also
        contains task-specific keys, such as `predictions/task_a`, and the
        `predictions` key maps to the primary task predictions.
      weights: weights for the metric. Works only for one label tensor.

    Returns:
      A dictionary mapping string to (tensor, update_op) tuples. In the single
      task setup, the dictionary contains only the `accuracy` key. In the
      multi task setup, the dictionary contains the task name suffixed keys,
      e.g., `accuracy/task_a` mapping to their respective accuracy metric
      tuples.
    """

    metrics_dict = {}

    if not isinstance(labels, dict):
      if labels.dtype == tf.string and label_vocabulary:
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=label_vocabulary,
            values=range(len(label_vocabulary)),
            name="class_id_lookup",
            value_dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(initializer, 1, name="lookup")
        label_ids = table.lookup(labels)
        metrics_dict["accuracy"] = tf.compat.v1.metrics.accuracy(
            label_ids, predictions["predictions"], weights=weights)
      elif labels.dtype.is_integer and not label_vocabulary:
        metrics_dict["accuracy"] = tf.compat.v1.metrics.accuracy(
            labels, predictions["predictions"], weights=weights)
      elif (labels.dtype.is_floating and not label_vocabulary and
            tf.squeeze(labels).shape == tf.squeeze(
                predictions["predictions"]).shape):
        metrics_dict["accuracy"] = tf.compat.v1.metrics.accuracy(
            tf.cast(labels, "int32"),
            predictions["predictions"],
            weights=weights)
      else:
        logging.warning("Unable to compute accuracy, since labels has dtype "
                        "tf.string but label_vocabulary is not supplied.")

    else:
      for task_name, label in labels.items():
        if (label.dtype == tf.string and label_vocabulary and
            task_name in label_vocabulary):
          initializer = tf.lookup.KeyValueTensorInitializer(
              keys=label_vocabulary[task_name],
              values=range(len(label_vocabulary[task_name])),
              name="class_id_lookup",
              value_dtype=tf.int64)
          table = tf.lookup.StaticVocabularyTable(initializer, 1, name="lookup")
          label_ids = table.lookup(label)
          metrics_dict["accuracy/" + task_name] = tf.compat.v1.metrics.accuracy(
              label_ids, predictions["predictions/" + task_name])
        elif label.dtype.is_integer and (not label_vocabulary or
                                         task_name not in label_vocabulary):
          metrics_dict["accuracy/" + task_name] = tf.compat.v1.metrics.accuracy(
              label, predictions["predictions/" + task_name])
        elif (label.dtype.is_floating and
              (not label_vocabulary or task_name not in label_vocabulary) and
              tf.squeeze(label).shape == tf.squeeze(
                  predictions["predictions/" + task_name]).shape):
          metrics_dict["accuracy/" + task_name] = tf.compat.v1.metrics.accuracy(
              tf.cast(label, "int32"), predictions["predictions/" + task_name])
        else:
          logging.warning(
              "Unable to compute accuracy for task %s, since labels has dtype "
              "tf.string but label_vocabulary is not supplied.", task_name)

    return metrics_dict

  return _metric_fn


def _make_auc_metric_fn(curve, label_vocabulary):
  """Makes a metric_fn to compute AUC-ROC or AUC-PR.

  Wraps around tf.metrics.auc(), so that it is easier to keep track of the
  metric name with the string key. This only works in the single-task
  binary-classification setup.

  Args:
    curve: "ROC" or "PR".
    label_vocabulary: A 1-D string Tensor or string list. If `label_vocabulary`
      is None, then the labels are assumed to be castable to bool.

  Returns:
    A function that takes `labels` and `predictions`, and returns a dictionary
    mapping "auc_roc" or "auc_pr" to (tensor, update_op) tuples.
  """

  def _metric_fn(labels, predictions, weights=None):
    """Metrics for tensorboard.

    Args:
      labels: A 1-D Tensor castable to bool, where True means that the label for
        that instance is class 1, and False means that the label for that
        instance is class 0.
      predictions: A dictionary mapping strings to Tensors. This dictionary
        contains a `predictions` string key which maps to a tf.int64 Tensor, as
        well as `probabilities` and `log_probabilities` string keys which map to
        tf.float32 Tensors. The `probabilities` key must map to a 2-D tf.float32
        Tensor with values in range [0, 1]. The first dimension of this Tensor
        is over the examples, and the second dimension is over the classes.
      weights: weights for the metric. Works only for one label tensor.

    Returns:
      A dictionary mapping "auc_roc" or "auc_pr" to (tensor, update_op) tuples.

    Raises:
      NotImplementedError: if user tries to use this in a multi-task setup.
      ValueError: if predictions["probabilities"] is not a 2-D Tensor,
        or if the second dimension is not 2, or if labels has dtype string but
        label_vocabulary is not provided.
    """

    if isinstance(labels, dict):
      raise NotImplementedError(
          "Multi-task setup is not yet implemented for AUC.")

    if predictions["probabilities"].shape.rank != 2:
      raise ValueError(
          "predictions['probabilities'] must have a 2-D Tensor, where the "
          "first dimension is over the examples, and the second dimension is "
          "over the classes.")

    if predictions["probabilities"].shape.dims[-1] != 2:
      raise ValueError(
          "The second dimension of predictions['probabilities'] must be 2, "
          "because this AUC function is only defined for binary "
          "classification.")

    if labels.dtype == tf.string:
      if not label_vocabulary:
        raise ValueError(
            "labels is of dtype string, but label_vocabulary is not present.")
      else:
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=label_vocabulary,
            values=range(len(label_vocabulary)),
            name="class_id_lookup",
            value_dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(initializer, 1, name="lookup")
        label_ids = table.lookup(labels)
    else:
      label_ids = labels

    return {
        "auc_{}".format(curve.lower()):
            tf.compat.v1.metrics.auc(
                label_ids,
                tf.unstack(predictions["probabilities"], axis=-1)[1],
                curve=curve,
                weights=weights,
                summation_method="careful_interpolation")
    }

  return _metric_fn


def make_auc_roc_metric_fn(label_vocabulary=None):
  """Makes the metric function to compute AUC-ROC.

  Args:
    label_vocabulary: A 1-D string Tensor or string list. If `label_vocabulary`
      is None, then the labels are assumed to be castable to bool.

  Returns:
    A function that takes `labels` and `predictions`, and returns a dictionary
    mapping "auc_roc" to (tensor, update_op) tuples. This function computes the
    area under the Receiver Operating Characteristic (ROC) curve. Wraps around
    `tf.metrics.auc`, so that it is easier to keep track of the metric name with
    the string key. This only works in the single-task binary-classification
    setup.

    Args:
      labels: A 1-D Tensor castable to bool, where True means that the label for
        that instance is class 1, and False means that the label for that
        instance is class 0.
      predictions: A dictionary mapping strings to Tensors. This dictionary
        contains a `predictions` string key which maps to a tf.int64 Tensor, as
        well as `probabilities` and `log_probabilities` string keys which map to
        tf.float32 Tensors. The `probabilities` key must map to a 2-D tf.float32
        Tensor with values in range [0, 1]. The first dimension of this Tensor
        is over the examples, and the second dimension is over the classes.

    Returns:
      A dictionary mapping "auc_roc" to (tensor, update_op) tuples.

    Raises:
      NotImplementedError: if user tries to use this in a multi-task setup.
      ValueError: if predictions["probabilities"] is not a 2-D Tensor,
        or if the second dimension is not 2.
  """

  return _make_auc_metric_fn(curve="ROC", label_vocabulary=label_vocabulary)


def make_auc_pr_metric_fn(label_vocabulary=None):
  """Makes the metric function to compute AUC-ROC.

  Args:
    label_vocabulary: A 1-D string Tensor or string list. If `label_vocabulary`
      is None, then the labels are assumed to be castable to bool.

  Returns:
    A function that takes `labels` and `predictions`, and returns a dictionary
    mapping "auc_pr" to (tensor, update_op) tuples. This function computes the
    area under the Precision-Recall (PR) curve. Wraps around `tf.metrics.auc`,
    so that it is easier to keep track of the metric name with the string key.
    This only works in the single-task binary-classification setup.

    Args:
      labels: A 1-D Tensor castable to bool, where True means that the label for
        that instance is class 1, and False means that the label for that
        instance is class 0.
      predictions: A dictionary mapping strings to Tensors. This dictionary
        contains a `predictions` string key which maps to a tf.int64 Tensor, as
        well as `probabilities` and `log_probabilities` string keys which map to
        tf.float32 Tensors. The `probabilities` key must map to a 2-D tf.float32
        Tensor with values in range [0, 1]. The first dimension of this Tensor
        is over the examples, and the second dimension is over the classes.

    Returns:
      A dictionary mapping "auc_pr" to (tensor, update_op) tuples.

    Raises:
      NotImplementedError: if user tries to use this in a multi-task setup.
      ValueError: if predictions["probabilities"] is not a 2-D Tensor,
        or if the second dimension is not 2.
  """

  return _make_auc_metric_fn(curve="PR", label_vocabulary=label_vocabulary)


def create_num_parameters_metric_fn(tower_name=None):
  """Makes the function to count the number of trainable parameters.

  Args:
    tower_name: The name of the tower that contains variables we want to count.
      If it is None, then use all variables.

  Returns:
    A function that returns a dict with a single string key `num_parameters`
    that maps to a tuple containing two int32 0-D Tensors, both containing the
    number of trainable parameters.
  """

  def _metric_fn(labels, predictions, weights=None):
    """Counts the number of trainable parameters.

    Args:
      labels: Unused.
      predictions: Unused.
      weights: Unused.

    Returns:
      dict with a single string key `num_parameters` that maps to a tuple
      containing two int32 0-D Tensors, both containing the number of trainable
      parameters.
    """

    del labels  # unused
    del predictions  # unused
    del weights  # unused

    trainable = tf.compat.v1.trainable_variables()
    if tower_name:
      counted_variables = [
          var for var in trainable
          if var.name.startswith("Phoenix/{}".format(tower_name))
      ]
    else:
      counted_variables = trainable

    if counted_variables:
      parameters = tf.add_n([tf.size(input=var) for var in counted_variables])
    else:
      parameters = tf.constant(0, dtype=tf.int32)

    return {"num_parameters": (parameters, parameters)}

  return _metric_fn


def combine_metric_fns(metric_fn_list):
  """Returns a single metric_fn that combines the outputs of metric_fn_list.

  Args:
    metric_fn_list: A list of functions that each takes arguments `labels` and
      `predictions` and returns a dictionary mapping string keys to (tensor,
      update_op) tuples.

  Returns:
    A dictionary mapping string keys to (tensor, update_op) tuples.
  """

  def _metric_fn(labels, predictions, weights=None):
    """Returns a dictionary mapping string to (tensor, update_op) tuples."""

    metrics_dict = {}
    for child_metric_fn in metric_fn_list:
      metrics_dict.update(child_metric_fn(labels, predictions, weights))
    return metrics_dict

  return _metric_fn
