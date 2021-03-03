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
"""TaskSpec object.

This object is responsible for ensembling towers, and creating the loss, logits,
and training_ops.
"""

import collections
import functools

from absl import logging
from model_search import blocks_builder as blocks
from model_search.architecture import architecture_utils
import numpy as np
import tensorflow.compat.v2 as tf

_OPTIMIZERS = {
    "sgd":
        tf.compat.v1.train.GradientDescentOptimizer,
    "momentum":
        functools.partial(tf.compat.v1.train.MomentumOptimizer, momentum=.9),
    "adagrad":
        tf.compat.v1.train.AdagradOptimizer,
    "adam":
        tf.compat.v1.train.AdamOptimizer,
    "rmsprop":
        tf.compat.v1.train.RMSPropOptimizer,
}


def _get_optimizer_fn(optimizer,
                      learning_rate,
                      use_tpu,
                      exponential_decay_steps=-1,
                      exponential_decay_rate=-1,
                      lr_warmup_steps=0):
  """Returns a function that gives the optimizer."""

  def optimizer_fn():
    global_step = tf.compat.v1.train.get_or_create_global_step()
    new_learning_rate = learning_rate
    if exponential_decay_steps > 0 and exponential_decay_rate > 0:
      new_learning_rate = tf.compat.v1.train.exponential_decay(
          learning_rate=learning_rate,
          global_step=global_step,
          decay_steps=exponential_decay_steps,
          decay_rate=exponential_decay_rate,
          staircase=True)

    if lr_warmup_steps > 0:
      warmup_lr = (
          learning_rate * tf.cast(global_step, tf.float32) /
          tf.cast(lr_warmup_steps, tf.float32))

      warmup_fn = lambda: tf.minimum(warmup_lr, learning_rate)

      new_learning_rate = tf.cond(
          pred=global_step < lr_warmup_steps,
          true_fn=warmup_fn,
          false_fn=lambda: new_learning_rate)

    if use_tpu:
      opt = tf.compat.v1.tpu.CrossShardOptimizer(
          _OPTIMIZERS[optimizer](learning_rate=new_learning_rate))
    else:
      opt = _OPTIMIZERS[optimizer](learning_rate=new_learning_rate)
    return opt

  return optimizer_fn


TaskSpec = collections.namedtuple(
    "TaskSpec", ["name", "logits", "loss", "train_op_list", "train_hooks_list"])


class ModelSpec(
    collections.namedtuple("ModelSpec",
                           ["loss", "train_op", "predictions", "train_hooks"])):
  """Definition of model training."""

  def __new__(cls, loss, train_op, predictions, train_hooks=None):
    if train_hooks is None:
      train_hooks = []
    return super(ModelSpec, cls).__new__(
        cls,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        train_hooks=train_hooks)


def _compute_tolerance(workers):
  """Computes how many workers can be ignored for one gradient update.

  These numbers were chosen based on just a few expriments.  They are rather
  arbitrary, feel free to change them.

  Args:
    workers: Number of workers used during computations.

  Returns:
    int: how many workers can be preemptied during one step.
  """
  if workers <= 4:
    return 0
  elif workers <= 10:
    return 1
  elif workers <= 30:
    return int(0.1 * workers)
  else:
    return int(0.5 * workers)


# Helper function.
def _train_op_fn(loss,
                 optimizer_fn,
                 l2_regularization=-1,
                 gradient_max_norm=-1,
                 use_synchronous_optimizer=False):
  """Returns the op to optimize the loss.

  Supports l2 regularization, learning rate decay and gradient clipping.

  Args:
    loss: The training loss before regularization.
    optimizer_fn: the optimization function.
    l2_regularization: a float that will multiply the l2 weight norms in the
      loss function.
    gradient_max_norm: a float - maximal gradient update allowed.
    use_synchronous_optimizer: a bool whether to use synchronous optimization.

  Returns:
    `ModelSpec` with logits, loss, train_ops and train_hooks.
  """
  total_loss = loss
  if l2_regularization > 0:
    weight_losses = [
        tf.multiply(
            tf.nn.l2_loss(weight), l2_regularization, name="l2_weight_loss")
        for weight in tf.compat.v1.trainable_variables()
    ]
    total_loss = tf.add_n(weight_losses + [loss], name="total_loss")

  global_step = tf.compat.v1.train.get_or_create_global_step()

  opt = optimizer_fn()

  train_hooks = []
  if use_synchronous_optimizer:
    config = tf.estimator.RunConfig()
    workers = config.num_worker_replicas + 1
    tolerance = _compute_tolerance(workers)
    to_aggregate = workers - tolerance
    opt = tf.compat.v1.train.SyncReplicasOptimizer(
        opt, replicas_to_aggregate=to_aggregate, total_num_replicas=workers)
    sync_replicas_hook = opt.make_session_run_hook(config.is_chief)
    train_hooks.append(sync_replicas_hook)

  tvars = tf.compat.v1.trainable_variables()
  grads_and_vars = opt.compute_gradients(loss=total_loss, var_list=tvars)
  # TODO(b/172564129): switch to tf.contrib.estimator.clip_gradients_by_norm
  if gradient_max_norm > 0.0:
    grads = [gv[0] for gv in grads_and_vars]
    tvars = [gv[1] for gv in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, gradient_max_norm)
    grads_and_vars = list(zip(grads, tvars))

  if use_synchronous_optimizer:
    apply_gradients_op = opt.apply_gradients(grads_and_vars, global_step)
  else:
    apply_gradients_op = opt.apply_gradients(grads_and_vars)

  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    return tf.group(apply_gradients_op), train_hooks


def _merge_train_op_list(train_op_list, no_train_speedup):
  """Merges a train_op list into one train_op."""
  if not train_op_list:
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # Train is 100 times faster. No parameters to train here.
    train_op = tf.compat.v1.assign(global_step, global_step + no_train_speedup)
  elif len(train_op_list) == 1:
    train_op = train_op_list[0]
  else:
    train_op = tf.group(*train_op_list)

  return train_op


def eval_op_fn(loss):
  del loss
  return tf.no_op(), []


class TaskManager(object):
  """Generates TaskSpecs for various tasks in Phoenix."""

  def __init__(self, phoenix_spec):
    self._phoenix_spec = phoenix_spec
    self._ensemble_spec = phoenix_spec.ensemble_spec

  def _add_projection_if_needed(self, logits, number_of_classes):
    if logits.get_shape().as_list()[-1] == number_of_classes:
      return logits
    else:
      return tf.keras.layers.Dense(number_of_classes)(tf.nn.relu(logits))

  def _create_task_spec(self, labels, weights, train_logits_specs,
                        eval_logits_spec, train_op_fn, name, mode, loss_fn):
    """Creates the task spec for a task."""
    train_op_list = []
    hooks_list = []
    eval_loss = None
    eval_logits = eval_logits_spec.logits
    for spec in train_logits_specs:
      if mode != tf.estimator.ModeKeys.PREDICT:
        loss = loss_fn(labels=labels, logits=spec.logits, weights=weights)
        # Optimization only. I.e., can be removed.
        # Tensorflow creates two nodes if invoking cross_entropy on the same
        # tensor twice. If this statement is removed the code will continue to
        # work only slower.
        if spec.logits is eval_logits:
          eval_loss = loss

        loss *= spec.logits_weight

        if spec.aux_logits is not None:
          aux_loss = loss_fn(
              labels=labels, logits=spec.aux_logits, weights=weights)
          loss += spec.aux_logits_weight * aux_loss
        train_op, train_hooks = train_op_fn(loss)
        train_op_list.append(train_op)
        hooks_list.extend(train_hooks)

    # The loss to display in tensorboard. Display loss even then training
    # logits is an empty list. I.e., no parameters to train.
    if mode != tf.estimator.ModeKeys.PREDICT and eval_loss is None:
      eval_loss = loss_fn(labels=labels, logits=eval_logits, weights=weights)

    return TaskSpec(
        name=name,
        logits=eval_logits,
        loss=eval_loss,
        train_op_list=train_op_list,
        train_hooks_list=hooks_list)

  def create_model_spec(self,
                        features,
                        params,
                        learning_rate_spec,
                        use_tpu,
                        train_logits_specs,
                        eval_logits_spec,
                        labels,
                        mode,
                        lengths,
                        loss_fn,
                        model_directory,
                        predictions_fn,
                        optimizer_fn=None):
    """Creates model spec for all tasks."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if not optimizer_fn and is_training:
      optimizer_fn = _get_optimizer_fn(
          optimizer=params.optimizer,
          learning_rate=learning_rate_spec["learning_rate"],
          use_tpu=use_tpu,
          exponential_decay_steps=learning_rate_spec.get(
              "exponential_decay_steps", -1),
          exponential_decay_rate=learning_rate_spec.get(
              "exponential_decay_rate", -1),
          lr_warmup_steps=self._phoenix_spec.learning_spec.lr_warmup_steps)

    train_op_fn = eval_op_fn
    if is_training:
      train_op_fn = functools.partial(
          _train_op_fn,
          optimizer_fn=optimizer_fn,
          use_synchronous_optimizer=self._phoenix_spec
          .use_synchronous_optimizer,
          l2_regularization=learning_rate_spec.get("l2_regularization", -1),
          gradient_max_norm=learning_rate_spec.get("gradient_max_norm", -1))

    # One label / One task - Old Phoenix behavior
    if (self._phoenix_spec.multi_task_spec is None or
        len(self._phoenix_spec.multi_task_spec) <= 1):
      with tf.compat.v1.variable_scope("Phoenix/Trainer"):
        weights = 1.0
        if (self._phoenix_spec.weight_feature_name and
            mode != tf.estimator.ModeKeys.PREDICT):
          weights = features[self._phoenix_spec.weight_feature_name]

        task_spec = self._create_task_spec(
            labels=labels,
            weights=weights,
            train_logits_specs=train_logits_specs,
            eval_logits_spec=eval_logits_spec,
            train_op_fn=train_op_fn,
            name="Trainer",
            mode=mode,
            loss_fn=loss_fn)

        train_op = None
        if is_training:
          train_op = _merge_train_op_list(task_spec.train_op_list,
                                          self._ensemble_spec.no_train_speedup)

        # create predictions here.
        predictions = predictions_fn(
            eval_logits_spec.logits,
            mode=mode,
            temperature=self._phoenix_spec.temperature)

        return ModelSpec(
            loss=task_spec.loss,
            train_op=train_op,
            predictions=predictions,
            train_hooks=task_spec.train_hooks_list)

    # MultiTask. New Phoenix Behavior
    # Details in:
    # go/phoenix-multitask
    num_weights_in_labels = 0
    for task_spec in self._phoenix_spec.multi_task_spec:
      if not task_spec.weight_is_a_feature:
        num_weights_in_labels += 1

    # In predict mode we don't have labels
    if labels:
      assert (len(labels) == len(self._phoenix_spec.multi_task_spec) +
              num_weights_in_labels)
    if len(train_logits_specs) > 1:
      logging.warning("Using ensembling in a multi-task training. If there is "
                      "no task that restrict the searchable logits from "
                      "rotating, then training is going to produce bad "
                      "non-aligned ensembles.")

    primary_task = None
    task_spec_list = []
    for task_spec in self._phoenix_spec.multi_task_spec:
      logits_spec = train_logits_specs[0]
      # Build tower on top of searched model for the specific task
      if task_spec.architecture:
        tower_architecture = [
            blocks.BlockType[block_type]
            for block_type in task_spec.architecture
        ]
        task_tower_spec = architecture_utils.construct_tower(
            phoenix_spec=self._phoenix_spec,
            input_tensor=tf.nn.relu(logits_spec.logits),
            tower_name="task_{}_tower".format(task_spec.label_name),
            architecture=np.array(tower_architecture),
            is_training=is_training,
            lengths=lengths,
            hparams=params,
            model_directory=model_directory,
            logits_dimension=task_spec.number_of_classes,
            is_frozen=False,
            # TODO(b/172564129): add dropouts.
            dropout_rate=None)
        # Ignore auxiliary heads for task towers, if any.
        task_logits = task_tower_spec.logits_spec.logits
      else:
        with tf.compat.v1.variable_scope("Phoenix/task_{}_tower".format(
            task_spec.label_name)):
          task_logits = self._add_projection_if_needed(
              logits_spec.logits, task_spec.number_of_classes)

      # Replace the base tower logits with those of the task tower.
      logits_spec = logits_spec._replace(logits=task_logits)
      if logits_spec.aux_logits:
        aux_logits = self._add_projection_if_needed(logits_spec.aux_logits,
                                                    task_spec.number_of_classes)
        logits_spec = logits_spec._replace(aux_logits=aux_logits)
      with tf.compat.v1.variable_scope("Phoenix/trainer_{}".format(
          task_spec.label_name)):
        if mode != tf.estimator.ModeKeys.PREDICT:
          task_labels = labels[task_spec.label_name]
        else:
          task_labels = None

        weights = 1.0
        if (task_spec.weight_feature_name and
            mode != tf.estimator.ModeKeys.PREDICT):
          if task_spec.weight_is_a_feature:
            weights = features[task_spec.weight_feature_name]
          else:
            weights = labels[task_spec.weight_feature_name]

        task = self._create_task_spec(
            labels=task_labels,
            weights=weights,
            train_logits_specs=[logits_spec],
            eval_logits_spec=logits_spec,
            train_op_fn=train_op_fn,
            name=task_spec.label_name,
            mode=mode,
            loss_fn=loss_fn)
        task_spec_list.append(task)
        if task_spec.label_name == self._phoenix_spec.primary_task_name:
          primary_task = task

    if not self._phoenix_spec.primary_task_name:
      primary_task = task_spec_list[0]

    model_spec_predictions = {}
    for task in task_spec_list:
      task_predictions = predictions_fn(
          task.logits, mode=mode, temperature=self._phoenix_spec.temperature)
      for prediction_key, prediction_value in task_predictions.items():
        prediction_key_name = prediction_key + "/" + task.name
        model_spec_predictions[prediction_key_name] = prediction_value
        if task.name == primary_task.name:
          model_spec_predictions[prediction_key] = prediction_value
    logging.info(model_spec_predictions)

    train_op_list = []
    train_hooks_list = []
    for task in task_spec_list:
      train_op_list.extend(task.train_op_list)
      train_hooks_list.extend(task.train_hooks_list)

    train_op = None
    if is_training:
      train_op = _merge_train_op_list(train_op_list,
                                      self._ensemble_spec.no_train_speedup)

    return ModelSpec(
        loss=primary_task.loss,
        train_op=train_op,
        predictions=model_spec_predictions,
        train_hooks=train_hooks_list)
