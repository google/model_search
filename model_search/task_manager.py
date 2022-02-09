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
import inspect

from absl import logging
from model_search import block_builder
from model_search import ensembler
from model_search.architecture import architecture_utils
from model_search.architecture import tower
from model_search.ensembler import EnsembleLogits
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils
from model_search.meta import distillation

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


def extract_task_specific(instance, task_name):
  output = instance
  if isinstance(instance, dict):
    return instance[task_name]
  return output


def supply_params_if_needed(instance, params):
  if "params" in inspect.signature(instance).parameters:
    return functools.partial(instance, params=params)
  return instance


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


TaskSpec = collections.namedtuple("TaskSpec", [
    "name", "logits", "loss", "train_op_list", "train_hooks_list",
    "train_losses"
])


class ModelSpec(
    collections.namedtuple(
        "ModelSpec",
        ["loss", "train_op", "predictions", "train_hooks", "eval_logits"])):
  """Definition of model training."""

  def __new__(cls,
              loss,
              train_op,
              predictions,
              train_hooks=None,
              eval_logits=None):
    if train_hooks is None:
      train_hooks = []
    return super(ModelSpec, cls).__new__(
        cls,
        loss=loss,
        train_op=train_op,
        predictions=predictions,
        train_hooks=train_hooks,
        eval_logits=eval_logits)


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


class Task(tower.Tower):
  """A simple object to capture a task head (tower)."""

  def __init__(self,
               phoenix_spec,
               tower_name,
               architecture,
               is_training,
               logits_dimension,
               is_frozen,
               hparams,
               model_directory,
               generator_name=None):
    super(Task, self).__init__(
        phoenix_spec=phoenix_spec,
        tower_name=tower_name,
        architecture=architecture,
        is_training=is_training,
        logits_dimension=logits_dimension,
        is_frozen=is_frozen,
        hparams=hparams,
        model_directory=model_directory,
        dropout_rate=None,
        allow_auxiliary_head=False)
    self._generator_name = generator_name

  # Please use this factory method to create a task tower.
  @staticmethod
  def get_task(phoenix_spec,
               tower_name,
               architecture,
               is_training,
               logits_dimesnion,
               is_frozen,
               hparams,
               model_directory,
               generator_name=None,
               previous_tower_name=None,
               previous_model_dir=None):
    # Building from checkpoint - no need to boostrap models
    if (tf.train.latest_checkpoint(model_directory) or
        generator_name == base_tower_generator.SEARCH_GENERATOR or
        architecture.size == 0):
      task_ = Task(
          phoenix_spec=phoenix_spec,
          tower_name=tower_name,
          architecture=architecture,
          is_training=is_training,
          logits_dimension=logits_dimesnion,
          is_frozen=is_frozen,
          hparams=hparams,
          model_directory=model_directory,
          generator_name=generator_name)
      if (architecture.size == 0 and
          not tf.train.latest_checkpoint(model_directory) and
          generator_name != base_tower_generator.SEARCH_GENERATOR):
        task_.add_initialization(
            prev_model_dir=previous_model_dir,
            prev_tower_name=previous_tower_name)
      return task_
    else:
      return Task.load(
          phoenix_spec=phoenix_spec,
          original_tower_name=previous_tower_name,
          new_tower_name=tower_name,
          model_directory=previous_model_dir,
          new_model_directory=model_directory,
          is_training=is_training,
          logits_dimension=logits_dimesnion,
          force_freeze=True,
          allow_auxiliary_head=False,
          skip_initialization=False)

  def call(self, inputs, training):
    if self._construct_architecture.size:
      return super(Task, self).call(inputs, training)

    # No architecture
    logits = self._add_projection_if_needed(inputs, self._logits_dimension)

    # Load the kernel of the additional layer if it is already trained model
    if (not tf.train.latest_checkpoint(self._model_directory) and
        hasattr(self, "_prev_model_dir") and hasattr(self, "_prev_tower_name")):
      architecture_utils.init_variables(
          tf.train.latest_checkpoint(self._prev_model_dir),
          "Phoenix/{}".format(self._prev_tower_name),
          "Phoenix/{}".format(self._tower_name))
    self._logits_spec = architecture_utils.LogitsSpec(logits=logits)
    self._architecture = [
        block_builder.BlockType(block).name
        for block in self._construct_architecture
    ]
    self._layer_tensors = None  # We don't need the tensors here
    return logits

  def _add_projection_if_needed(self, logits, number_of_classes):
    if logits.get_shape().as_list()[-1] == number_of_classes:
      return logits
    else:
      return tf.keras.layers.Dense(
          number_of_classes, name="maybe_project")(
              logits)


class TaskManager(object):
  """Generates TaskSpecs for various tasks in Phoenix."""

  def __init__(self, phoenix_spec, logits_dimension, loss_fn, head):
    self._phoenix_spec = phoenix_spec
    self._ensemble_spec = phoenix_spec.ensemble_spec
    self._ensembler = ensembler.Ensembler(phoenix_spec)
    self._distiller = distillation.Distiller(phoenix_spec.distillation_spec)
    self._logits_dimension = logits_dimension
    self._loss_fn = loss_fn
    self._head = head

  def _create_task_spec(self,
                        labels,
                        weights,
                        train_logits_specs,
                        eval_logits_spec,
                        train_op_fn,
                        name,
                        mode,
                        loss_fn,
                        add_train_ops=True):
    """Creates the task spec for a task."""
    train_op_list = []
    hooks_list = []
    eval_loss = None
    eval_logits = eval_logits_spec.logits
    train_losses = []
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

        if add_train_ops:
          train_op, train_hooks = train_op_fn(loss)
          train_op_list.append(train_op)
          hooks_list.extend(train_hooks)

        train_losses.append(loss)

    # The loss to display in tensorboard. Display loss even then training
    # logits is an empty list. I.e., no parameters to train.
    if mode != tf.estimator.ModeKeys.PREDICT and eval_loss is None:
      eval_loss = loss_fn(labels=labels, logits=eval_logits, weights=weights)

    return TaskSpec(
        name=name,
        logits=eval_logits,
        loss=eval_loss,
        train_op_list=train_op_list,
        train_hooks_list=hooks_list,
        train_losses=train_losses)

  def _get_loss_fn(self,
                   original_loss_fn,
                   features,
                   mode,
                   my_id,
                   teacher_logits_spec=None):
    """Gets the applicable loss_fn to use.

    If head is not None, wraps the head's loss function to match the interface
    Phoenix expects (see loss_fns.py), unless distillation is being used.

    Args:
      original_loss_fn: The original loss_fn from the user.
      features: The features pass to model_fn.
      mode: The mode passed to model_fn.
      my_id: My trial id (integer).
      teacher_logits_spec: Logits of the teacher network to use when distilling.

    Returns:
      The loss_fn to use.

    Raises:
      RuntimeError if unable to find the loss function in the head object.
    """

    if (mode == tf.estimator.ModeKeys.TRAIN and
        teacher_logits_spec is not None):
      return distillation.get_distillation_loss_fn(
          teacher_logits=teacher_logits_spec.logits,
          distillation_spec=self._phoenix_spec.distillation_spec,
          my_id=my_id,
          original_loss_fn=original_loss_fn)

    if not self._head:
      return original_loss_fn

    def head_loss_fn(labels, logits, weights=1.0):
      """Create a loss fn from the Head object."""
      del weights  # Head already has weights built in.

      training_loss = None
      # There is two types of head, and their api is different.
      if getattr(self._head, "loss", None) is not None:
        training_loss = self._head.loss(
            labels=labels, logits=logits, features=features, mode=mode)
      elif getattr(self._head, "create_loss", None) is not None:
        training_loss = self._head.create_loss(
            labels=labels, logits=logits, features=features,
            mode=mode).training_loss
      else:
        raise RuntimeError("unable to find loss function in Head object.")

      return training_loss

    return head_loss_fn

  def get_train_and_eval_logits(self, towers, trial_mode):
    """Helper function to get the various logits for a task."""
    priors_logits_specs = []
    search_logits_specs = []
    if base_tower_generator.SEARCH_GENERATOR in towers.keys():
      search_logits_specs = [
          t.logits_spec for t in towers[base_tower_generator.SEARCH_GENERATOR]
      ]
    if base_tower_generator.PRIOR_GENERATOR in towers.keys():
      priors_logits_specs = [
          t.logits_spec for t in towers[base_tower_generator.PRIOR_GENERATOR]
      ]
    if base_tower_generator.REPLAY_GENERATOR in towers.keys():
      priors_logits_specs = [
          t.logits_spec for t in towers[base_tower_generator.REPLAY_GENERATOR]
      ]

    teacher_logits = None
    if trial_mode == trial_utils.TrialMode.ENSEMBLE_SEARCH:
      ensemble_logits = self._ensembler.bundle_logits(
          priors_logits_specs=priors_logits_specs,
          search_logits_specs=search_logits_specs,
          logits_dimension=self._logits_dimension)
    elif trial_mode == trial_utils.TrialMode.DISTILLATION:
      # TODO(b/146067345): Initialize some random architecture if search
      # logits specs is empty.
      ensemble_logits = self._distiller.bundle_logits(
          priors_logits_specs=priors_logits_specs,
          search_logits_specs=search_logits_specs)
      teacher_logits = ensemble_logits.teacher_logits_spec
    else:
      ensemble_logits = EnsembleLogits(
          train_logits_specs=search_logits_specs,
          eval_logits_spec=search_logits_specs[0])
    return (ensemble_logits.train_logits_specs,
            ensemble_logits.eval_logits_spec, teacher_logits)

  def create_model_spec(self,
                        features,
                        params,
                        learning_rate_spec,
                        use_tpu,
                        trial_mode,
                        towers,
                        labels,
                        mode,
                        my_id,
                        model_directory,
                        predictions_fn,
                        optimizer_fn=None):
    """Creates model spec for all tasks."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if not optimizer_fn and is_training:
      final_optimizer_fn = _get_optimizer_fn(
          optimizer=params.optimizer,
          learning_rate=learning_rate_spec["learning_rate"],
          use_tpu=use_tpu,
          exponential_decay_steps=learning_rate_spec.get(
              "exponential_decay_steps", -1),
          exponential_decay_rate=learning_rate_spec.get(
              "exponential_decay_rate", -1),
          lr_warmup_steps=self._phoenix_spec.learning_spec.lr_warmup_steps)
    elif is_training:
      final_optimizer_fn = optimizer_fn
      if "params" in inspect.signature(optimizer_fn).parameters:
        final_optimizer_fn = functools.partial(optimizer_fn, params=params)

    train_op_fn = eval_op_fn
    if is_training:
      train_op_fn = functools.partial(
          _train_op_fn,
          optimizer_fn=final_optimizer_fn,
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

        train_logits_specs, eval_logits_spec, teacher_logits = (
            self.get_train_and_eval_logits(
                towers=towers, trial_mode=trial_mode))

        final_loss_fn = self._get_loss_fn(
            original_loss_fn=supply_params_if_needed(self._loss_fn, params),
            features=features,
            mode=mode,
            my_id=my_id,
            teacher_logits_spec=teacher_logits)

        task_spec = self._create_task_spec(
            labels=labels,
            weights=weights,
            train_logits_specs=train_logits_specs,
            eval_logits_spec=eval_logits_spec,
            train_op_fn=train_op_fn,
            name="Trainer",
            mode=mode,
            loss_fn=final_loss_fn)

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
            train_hooks=task_spec.train_hooks_list,
            eval_logits=eval_logits_spec.logits)

    # MultiTask. New Phoenix Behavior
    # Details in:
    # go/phoenix-multitask
    num_weights_in_labels = 0
    for task_spec in self._phoenix_spec.multi_task_spec:
      if not task_spec.weight_is_a_feature:
        num_weights_in_labels += 1

    # In predict mode we don't have labels
    if labels and not self._phoenix_spec.pass_label_dict_as_is:
      assert (len(labels) == len(self._phoenix_spec.multi_task_spec) +
              num_weights_in_labels)

    primary_task = None
    task_spec_list = []
    for task_spec in self._phoenix_spec.multi_task_spec:
      # Build tower on top of searched model for the specific task
      task_towers = collections.defaultdict(list)
      for generator_name, tower_list in towers.items():
        for i, tower_ in enumerate(tower_list):
          new_tower_name = "{}_{}_{}".format(task_spec.label_name, str(i),
                                             generator_name)
          previous_tower_name = "{}_0_search_generator".format(
              task_spec.label_name)
          previous_model_dir = tower_.previous_model_dir
          is_prior = (generator_name != base_tower_generator.SEARCH_GENERATOR)
          task_tower = Task.get_task(
              phoenix_spec=self._phoenix_spec,
              tower_name=new_tower_name,
              architecture=np.array([
                  block_builder.BlockType[block_type]
                  for block_type in task_spec.architecture
              ]),
              is_training=is_training,
              logits_dimesnion=task_spec.number_of_classes,
              is_frozen=is_prior,
              hparams=params,
              model_directory=model_directory,
              generator_name=generator_name,
              previous_tower_name=previous_tower_name,
              previous_model_dir=previous_model_dir)
          base_tensor = tower_.logits_spec.logits
          if task_spec.apply_activation_on_shared_logits:
            base_tensor = tf.nn.relu(base_tensor)
          task_tower(base_tensor, training=is_training)
          task_towers[generator_name].append(task_tower)
      train_logits_spec, eval_logits_spec, teacher_logits = (
          self.get_train_and_eval_logits(task_towers, trial_mode=trial_mode))

      with tf.compat.v1.variable_scope("Phoenix/trainer_{}".format(
          task_spec.label_name)):
        if mode != tf.estimator.ModeKeys.PREDICT:
          task_labels = labels[task_spec.label_name]
          if self._phoenix_spec.pass_label_dict_as_is:
            task_labels = labels
        else:
          task_labels = None

        weights = 1.0
        if (task_spec.weight_feature_name and
            mode != tf.estimator.ModeKeys.PREDICT):
          if task_spec.weight_is_a_feature:
            weights = features[task_spec.weight_feature_name]
          else:
            weights = labels[task_spec.weight_feature_name]

        task_loss_fn = self._get_loss_fn(
            supply_params_if_needed(
                extract_task_specific(self._loss_fn, task_spec.label_name),
                params),
            features=features,
            mode=mode,
            my_id=my_id,
            teacher_logits_spec=teacher_logits)
        task = self._create_task_spec(
            labels=task_labels,
            weights=weights,
            train_logits_specs=train_logits_spec,
            eval_logits_spec=eval_logits_spec,
            train_op_fn=train_op_fn,
            name=task_spec.label_name,
            mode=mode,
            loss_fn=task_loss_fn,
            add_train_ops=(not self._phoenix_spec.merge_losses_of_multitask))
        task_spec_list.append(task)
        if task_spec.label_name == self._phoenix_spec.primary_task_name:
          primary_task = task

    if not self._phoenix_spec.primary_task_name:
      primary_task = task_spec_list[0]

    model_spec_predictions = {}
    for task in task_spec_list:
      task_predictions = extract_task_specific(predictions_fn, task.name)(
          task.logits, mode=mode, temperature=self._phoenix_spec.temperature)
      for prediction_key, prediction_value in task_predictions.items():
        prediction_key_name = prediction_key + "/" + task.name
        model_spec_predictions[prediction_key_name] = prediction_value
        if task.name == primary_task.name:
          model_spec_predictions[prediction_key] = prediction_value
    logging.info(model_spec_predictions)

    train_op_list = []
    train_hooks_list = []
    train_losses = []
    for task in task_spec_list:
      train_op_list.extend(task.train_op_list)
      train_hooks_list.extend(task.train_hooks_list)
      train_losses.extend(task.train_losses)

    train_op = None
    if is_training:
      if self._phoenix_spec.merge_losses_of_multitask:
        # In this case, tasks in task_spec_list should not have train_ops nor
        # hooks as in this case we sum up all losses to one and create
        # training ops. Follow the usage of
        # self._phoenix_spec.merge_losses_of_multitask above for when we call
        # self._create_task_spec for more details.
        if train_losses:
          merged_loss = tf.add_n(train_losses, name="merged_loss")
          train_op, train_hooks_list = train_op_fn(merged_loss)
        else:
          train_op = _merge_train_op_list([],
                                          self._ensemble_spec.no_train_speedup)
      else:
        train_op = _merge_train_op_list(train_op_list,
                                        self._ensemble_spec.no_train_speedup)

    return ModelSpec(
        loss=primary_task.loss,
        train_op=train_op,
        predictions=model_spec_predictions,
        train_hooks=train_hooks_list,
        eval_logits=primary_task.logits)
