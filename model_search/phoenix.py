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
"""A Phoenix estimator builder."""

import functools
import inspect

from absl import logging
import keras_tuner

from model_search import controller
from model_search import hparam as hp
from model_search import loss_fns
from model_search import metric_fns
from model_search import task_manager
from model_search.architecture import architecture_utils
from model_search.generators import base_tower_generator
from model_search.generators import trial_utils
from model_search.meta import transfer_learning
from model_search.metadata import ml_metadata_db
from model_search.proto import phoenix_spec_pb2
from model_search.proto import transfer_learning_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf


REPLAY_CONFIG_FILENAME = "replay_config.pbtxt"


_TL_HOOKS = {
    transfer_learning_spec_pb2.TransferLearningSpec
    .UNIFORM_AVERAGE_TRANSFER_LEARNING:
        transfer_learning.UniformAverageTransferLearningHook,
    transfer_learning_spec_pb2.TransferLearningSpec
    .LOSS_WEIGHTED_AVERAGE_TRANSFER_LEARNING:
        transfer_learning.LossWeightedAverageTransferLearningHook,
}


def aggregate_initial_architecture(hparams):
  """Helper function to aggregate initial architecture into an array hparam."""
  output = hparams.copy()
  initial_architecture_size = len(
      [hp for hp in hparams.keys() if hp.startswith("initial_architecture_")])
  if initial_architecture_size:
    output["initial_architecture"] = [
        hparams["initial_architecture_{}".format(i)]
        for i in range(initial_architecture_size)
    ]
  return output


def _merge_hparams(original_hparams, overrides):
  """Merges to hp.HParams objects."""
  # make a copy
  hparams = hp.HParams(**original_hparams.values())
  existing_ones = {k: v for k, v in overrides.values().items() if k in hparams}
  new_ones = {k: v for k, v in overrides.values().items() if k not in hparams}
  hparams.override_from_dict(existing_ones)
  for k, v in new_ones.items():
    hparams.add_hparam(k, v)
  return hparams


def _default_predictions_fn(logits,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            temperature=1.0):
  """Converts logits to predictions dict. Assumes classification."""
  new_logits = logits
  if mode == tf.estimator.ModeKeys.PREDICT and temperature != 1.0:
    assert temperature > 0
    temp_const = tf.constant(1 / temperature, name="softmax_temperature_const")
    logging.info("Applying temperature to logits")
    new_logits = tf.multiply(logits, temp_const, name="softmax_temperature_mul")

  predictions = tf.math.argmax(input=new_logits, axis=-1)
  probabilities = tf.nn.softmax(new_logits)
  log_probabilities = tf.nn.log_softmax(new_logits)

  predictions_dict = {
      "predictions": predictions,
      "probabilities": probabilities,
      "log_probabilities": log_probabilities,
      "logits": logits
  }
  return predictions_dict


class Estimator(tf.estimator.Estimator):
  """Estimator wrapper to add reporting to metadata storage after evaluation."""

  def __init__(self,
               model_fn,
               model_dir=None,
               config=None,
               params=None,
               warm_start_from=None,
               metadata=None):
    tf.estimator.Estimator._assert_members_are_not_overridden = staticmethod(  # pylint: disable=protected-access
        lambda _: None)
    super(Estimator, self).__init__(
        model_fn=model_fn,
        config=config,
        params=params,
        warm_start_from=warm_start_from)
    self._metadata = metadata
    self._model_dir = config.model_dir

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    eval_results = super(Estimator, self).evaluate(
        input_fn=input_fn,
        steps=steps,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        name=name)
    if self._metadata is not None:
      native_results = {k: v.item() for k, v in eval_results.items()}
      logging.info("Saving the following evaluation dictionary.")
      logging.info(native_results)
      self._metadata.report(native_results, self._model_dir)
    return eval_results


class Phoenix(object):
  """Phoenix: A smart search AutoML algorithm."""

  def __init__(self,
               phoenix_spec,
               input_layer_fn,
               study_owner,
               study_name,
               head=None,
               logits_dimension=None,
               label_vocabulary=None,
               loss_fn=None,
               metric_fn=None,
               predictions_fn=None,
               metadata=None,
               optimizer_fn=None):
    """Constructs a Phoenix instance.

    Args:
      phoenix_spec: A `PhoenixSpec` proto with the spec for the run.
      input_layer_fn: A function that converts feature Tensors to input layer.
        See learning.autolx.model_search.data.Provider.get_input_layer_fn
        for details.
      study_owner: A string holding the ldap of the study owner. We use tuner
        platforms to conduct the various architectures training. This field
        specifies the study owner.
      study_name: A string holding the study name.
      head: A head to use with Phoenix for creating the loss and eval metrics.
        If no head is given, Phoenix falls back to using the loss_fn and
        metric_fn. N.B.: Phoenix creates its own EstimatorSpec so everything
          besides the loss and eval metrics returned by head will be ignored.
      logits_dimension: An int holding the dimension of the output. Must be
        provided if head is None. Will be ignored if head is not None.
      label_vocabulary: List or tuple with labels vocabulary. Needed only if the
        labels are of type string. This list is used by the loss function if
        loss_fn is not provided. It is also used in the metric function to
        create the accuracy metric ops. Use only with multiclass classification
        problems.
      loss_fn: A function to compute the loss. Ignored if `head` is not None.
        Must accept as inputs a `labels` Tensor, a `logits` Tensor, and
        a `weights` Tensor. `weights` must either be rank 0 or have
        the same rank as labels. If None, Phoenix defaults to using softmax
        cross-entropy.
        Additional `params` holding the hparameters of the trial can be added
        as input to the signature.
        For multitaks, you have the option to pass a dict of functions keyed
        by the task name to apply different loss functions for different tasks.
      metric_fn: Metrics for Tensorboard. Ignored if `head` is not None.
        metric_fn takes `label` and `predictions` as input, and outputs a
        dictionary of (tensor, update_op) tuples. `label` is a Tensor (in the
        single task case) or a dict of Tensors (in the case of multi-task, where
        the key of the dicts correspond to the task names). `predictions` is a
        dict of Tensors. In the single task case, it consists of `predictions`,
        `probabilities`, and `log_probabilities`. In the multi-task case, it
        consists of the same keys as that of the single task case, but also
        those corresponding to each task (e.g., predictions/task_name_1). See
        `metric_fns` for more detail. If `metric_fn` is None, it will include a
        metric for the number of parameters, accuracy (if logit_dimensions >=
        2), and AUC metrics (if logit_dimensions == 2).
      predictions_fn: A function to convert eval logits to the
        `predictions` dictionary passed to metric_fn. If `None`, defaults to
        computing 'predictions', 'probabilities', and 'log_probabilities'.
        For multitaks, you have the option to pass a dict of functions keyed
        by the task name to apply different prediction functions for different
        tasks.
      metadata: An object that implements metadata api in
        learning.adanets.phoenix.metadata.Metadata
      optimizer_fn: A function that follows two possible signatures: 1. takes
        `params` as args and returns a tensorflow v1 optimizer instance.
        2. A function with no args that returns a tensorflow v1 optimizer.
        Please keep as None to use our default optimizers (i.e. let the
        search choose an optimizer).
    """

    # Check Phoenix preconditions and fail early if any of them are broken.
    if phoenix_spec.multi_task_spec:
      # TODO(b/172564129): Add support for head and custom loss_fns in
      # multi-task.
      assert not head, "head is not supported for multi-task."
    if head:
      msg = "Do not specify {} when using head as head already contains it."
      assert not logits_dimension, msg.format("logits_dimension")
      assert not label_vocabulary, msg.format("label_vocabulary")
      assert not loss_fn, msg.format("loss_fn")
      assert not metric_fn, msg.format("metric_fn")

    # Check ensemble search / distillation preconditions.
    ensemble_spec = phoenix_spec.ensemble_spec
    distillation_spec = phoenix_spec.distillation_spec
    if trial_utils.has_distillation(
        distillation_spec) and trial_utils.has_ensemble_search(
            ensemble_spec
        ) and not trial_utils.is_intermixed_ensemble_search(ensemble_spec):
      ensemble_search_spec = (
          ensemble_spec.nonadaptive_search
          if trial_utils.is_nonadaptive_ensemble_search(ensemble_spec) else
          ensemble_spec.adaptive_search)
      if (distillation_spec.minimal_pool_size ==
          ensemble_search_spec.minimal_pool_size):
        logging.warning("minimal_pool_size is the same for ensemble spec and "
                        "distillation spec, so distillation will be ignored.")

    self._phoenix_spec = phoenix_spec
    self._input_layer_fn = input_layer_fn
    self._study_owner = study_owner
    self._study_name = study_name
    self._head = head
    self._logits_dimension = (
        self._head.logits_dimension if head else logits_dimension)
    self._label_vocabulary = label_vocabulary
    if self._label_vocabulary:
      assert self._logits_dimension == len(self._label_vocabulary)

    self._loss_fn = loss_fn or loss_fns.make_multi_class_loss_fn(
        label_vocabulary=label_vocabulary)

    self._user_specified_metric_fn = metric_fn

    self._predictions_fn = (predictions_fn or _default_predictions_fn)

    if metadata is None:
      self._metadata = ml_metadata_db.MLMetaData(phoenix_spec, study_name,
                                                 study_owner)
    else:
      self._metadata = metadata
    self._task_manager = task_manager.TaskManager(
        phoenix_spec=phoenix_spec,
        logits_dimension=logits_dimension,
        loss_fn=self._loss_fn,
        head=self._head)
    self._controller = controller.InProcessController(
        phoenix_spec=phoenix_spec, metadata=self._metadata)
    self._user_optimizer_fn = optimizer_fn

  @property
  def metadata(self):
    return self._metadata

  def keras_compile(self, towers, hparams):
    """Compiles the keras model based on hparams."""
    optimizer_args = dict()

    # Learning rate
    lr = hparams.learning_rate
    if getattr(hparams, "exponential_decay_rate", None) is not None:
      max_times = self._phoenix_spec.learning_spec.max_decay_times
      steps = hparams.exponential_decay_steps
      decay = hparams.exponential_decay_rate
      lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries=[steps * j for j in range(1, max_times)],
          values=[lr * (decay**i) for i in range(1, max_times + 1)])
    optimizer_args["learning_rate"] = lr

    # Gradient Clip
    if getattr(hparams, "gradient_max_norm", None) is not None:
      optimizer_args["clipnorm"] = hparams.gradient_max_norm

    # L2 Error
    if getattr(hparams, "l2_regularization", None) is not None:
      logging.error("Keras mode doesn't support L2 regularization.")

    # Optimizer
    optimizer = None
    if hparams.optimizer == "sgd":
      optimizer = tf.keras.optimizers.SGD(**optimizer_args)
    elif hparams.optimizer == "momentum":
      optimizer = tf.keras.optimizers.SGD(**optimizer_args, momentum=0.9)
    elif hparams.optimizer == "adam":
      optimizer = tf.keras.optimizers.Adam(**optimizer_args)
    elif hparams.optimizer == "adagrad":
      optimizer = tf.keras.optimizers.Adagrad(**optimizer_args)
    elif hparams.optimizer == "rmsprop":
      optimizer = tf.keras.optimizers.RMSprop(**optimizer_args)
    elif hparams.optimizer == "lazy_adam":
      logging.error("Lazy adam is not implemented in Keras. Falling back to "
                    "Adam")
      optimizer = tf.keras.optimizers.Adam(**optimizer_args)

    towers["search_generator"][0].compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return towers["search_generator"][0]

  def keras_model_builder(self,
                          hparams,
                          run_config=None,
                          is_training=None,
                          input_layer_fn=None,
                          compile_model=True):
    """Builds a keras model based on hparams."""
    if compile_model:
      # Validate keras mode config is correct.
      if self._phoenix_spec.HasField("ensemble_spec"):
        logging.error("Cannot run ensembling in keras mode.")
        self._phoenix_spec.ClearField("ensemble_spec")
      if self._phoenix_spec.HasField("distillation_spec"):
        logging.error("Cannot run distillation in keras mode.")
        self._phoenix_spec.ClearField("distillation_spec")

    run_config_ = run_config
    if run_config is None:
      run_config_ = tf.estimator.RunConfig()

    if isinstance(hparams, hp.HParams):
      params = hparams
    elif isinstance(hparams,
                    keras_tuner.engine.hyperparameters.HyperParameters):
      parameter_values = aggregate_initial_architecture(hparams.values)
      params = hp.HParams(**parameter_values)
    else:
      parameter_values = aggregate_initial_architecture(hparams.values())
      params = hp.HParams(**parameter_values)

    logging.info(run_config_)
    logging.info(run_config_.model_dir)
    my_id = architecture_utils.DirectoryHandler.get_trial_id(
        run_config_.model_dir, self._phoenix_spec)

    # Get all information we have so far.
    trials = []
    # TODO(b/172564129): Only the chief needs the trials. Test to see if
    # workers need them
    if not self._phoenix_spec.HasField("replay"):
      trials = self._metadata.get_completed_trials()
    else:
      params = _merge_hparams(
          params,
          hp.HParams.from_proto(self._phoenix_spec.replay.towers[my_id -
                                                                 1].hparams))
      params.set_hparam(
          "initial_architecture",
          self._phoenix_spec.replay.towers[my_id - 1].architecture[:])

    # Update our database - clean up and sync ops.
    if run_config_.is_chief:
      self._metadata.before_generating_trial_model(my_id, run_config_.model_dir)

    # Determine whether to do ensemble search or distillation on this trial.
    trial_mode = trial_utils.get_trial_mode(
        self._phoenix_spec.ensemble_spec, self._phoenix_spec.distillation_spec,
        my_id)

    generators = self._controller.get_generators(my_id, trials)
    towers = {}
    for name, generator in generators.items():
      logging.info(generators)
      tower = generator.instance.generate(
          input_layer_fn=input_layer_fn,
          trial_mode=trial_mode,
          logits_dimension=self._logits_dimension,
          hparams=params,
          run_config=run_config_,
          is_training=is_training,
          trials=generator.relevant_trials)
      towers[name] = tower

    if input_layer_fn is not None and self._phoenix_spec.is_input_shared:
      shared_input_tensor, shared_lengths = input_layer_fn(
          is_training=is_training, scope_name="Phoenix/SharedInput")
      for generator_towers in towers.values():
        for tower in generator_towers:
          tower.add_feature_columns_input_layer(shared_input_tensor,
                                                shared_lengths)
    elif input_layer_fn is not None:
      logging.info(towers)
      for generator_towers in towers.values():
        for tower in generator_towers:
          if not tower.has_input_tensor():
            input_tensor, lengths = input_layer_fn(
                is_training=is_training,
                scope_name="{}/input".format(tower.name))
            tower.add_feature_columns_input_layer(input_tensor, lengths)

    if compile_model:
      # Keras mode: Search only
      return self.keras_compile(towers, params)

    return towers, trials

  def _make_model_fn(self, run_config, train_steps, use_tpu=False):
    """Returns a model_fn for the estimator."""

    def model_fn(features, labels, mode, params):
      """Model function that wraps the model specified."""
      self._metric_fn = self._user_specified_metric_fn
      self._default_metric_fn_list = []
      if self._logits_dimension >= 2:
        self._default_metric_fn_list.append(
            metric_fns.make_accuracy_metric_fn(self._label_vocabulary))
      if self._logits_dimension == 2:
        self._default_metric_fn_list += [
            metric_fns.make_auc_roc_metric_fn(self._label_vocabulary),
            metric_fns.make_auc_pr_metric_fn(self._label_vocabulary)
        ]

      my_id = architecture_utils.DirectoryHandler.get_trial_id(
          run_config.model_dir, self._phoenix_spec)

      # Create a copy of hparams
      hparams = params
      if my_id <= len(self._phoenix_spec.user_suggestions):
        hparams = _merge_hparams(
            params,
            hp.HParams.from_proto(
                self._phoenix_spec.user_suggestions[my_id - 1].hparams))

      # When predicting for RNN, we might not need the length.
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      lengths_feature_name = self._phoenix_spec.lengths_feature_name
      if mode == tf.estimator.ModeKeys.PREDICT:
        if isinstance(features, dict) and lengths_feature_name not in features:
          lengths_feature_name = ""

      if "params" in inspect.signature(self._input_layer_fn).parameters:
        input_layer_fn = functools.partial(
            self._input_layer_fn,
            features=features,
            params=hparams,
            lengths_feature_name=lengths_feature_name)
      else:
        input_layer_fn = functools.partial(
            self._input_layer_fn,
            features=features,
            lengths_feature_name=lengths_feature_name)

      towers, trials = self.keras_model_builder(
          hparams=hparams,
          is_training=is_training,
          run_config=run_config,
          input_layer_fn=input_layer_fn,
          compile_model=False)

      for generator_towers in towers.values():
        for tower in generator_towers:
          tower(None, training=is_training)

      trial_mode = trial_utils.get_trial_mode(
          self._phoenix_spec.ensemble_spec,
          self._phoenix_spec.distillation_spec, my_id)

      training_hooks = []
      # TODO(b/172564129): Figure out how to handle transfer learning for multi
      # task. Install transfer learning hook.
      if (is_training and
          self._phoenix_spec.transfer_learning_spec.transfer_learning_type in
          _TL_HOOKS):
        tower_name = base_tower_generator.SEARCH_GENERATOR
        vars_to_warm_start = architecture_utils.get_tower_variables(tower_name)
        if vars_to_warm_start:
          hook_fn = _TL_HOOKS[
              self._phoenix_spec.transfer_learning_spec.transfer_learning_type]
          tl_spec = self._phoenix_spec.transfer_learning_spec
          tl_hook = hook_fn(
              vars_to_warm_start=vars_to_warm_start,
              current_trial_id=my_id,
              completed_trials=trials,
              discount_factor=tl_spec.previous_trials_discount_factor,
              max_completed_trials=tl_spec.max_completed_trials,
              model_dir=run_config.model_dir)
          training_hooks.append(tl_hook)

      learning_rate_spec_keys = [
          "learning_rate", "l2_regularization", "gradient_max_norm",
          "exponential_decay_steps", "exponential_decay_rate"
      ]
      learning_rate_spec = {
          key: value
          for key, value in hparams.values().items()
          if key in learning_rate_spec_keys
      }
      tower_name = None

      # Create the metric_fn if it wasn't specified.
      if not self._metric_fn:
        metric_fn = metric_fns.create_num_parameters_metric_fn(tower_name)
        self._default_metric_fn_list.append(metric_fn)
        self._metric_fn = metric_fns.combine_metric_fns(
            self._default_metric_fn_list)

      model_spec = self._task_manager.create_model_spec(
          features=features,
          params=hparams,
          learning_rate_spec=learning_rate_spec,
          use_tpu=use_tpu,
          trial_mode=trial_mode,
          towers=towers,
          labels=labels,
          mode=mode,
          my_id=my_id,
          model_directory=run_config.model_dir,
          predictions_fn=self._predictions_fn,
          optimizer_fn=self._user_optimizer_fn)

      if run_config.is_chief:
        self._metadata.after_generating_trial_model(my_id)
        search_architecture = [["no_search"]]
        if base_tower_generator.SEARCH_GENERATOR in towers.keys():
          search_architecture = [
              t.architecture
              for t in towers[base_tower_generator.SEARCH_GENERATOR]
          ]
        trial_utils.write_replay_spec(
            model_dir=run_config.model_dir,
            filename=REPLAY_CONFIG_FILENAME,
            original_spec=self._phoenix_spec,
            search_architecture=search_architecture[0],
            hparams=hparams)

      # No need to add train op for the eval graph.
      train_op = None
      if is_training:
        train_op = self._increment_global_step(
            model_spec.train_op, train_steps,
            base_tower_generator.SEARCH_GENERATOR)

      if (isinstance(labels, dict) and
          not self._phoenix_spec.pass_label_dict_as_is):
        label_names = [
            label_spec.label_name
            for label_spec in self._phoenix_spec.multi_task_spec
        ]
        actual_labels = {
            name: label for name, label in labels.items() if name in label_names
        }
      else:
        actual_labels = labels

      if use_tpu:
        eval_metrics = None
        weights = None
        if self._phoenix_spec.weight_feature_name:
          weights = features[self._phoenix_spec.weight_feature_name]
        if mode != tf.estimator.ModeKeys.PREDICT and not self._head:
          eval_metrics = (self._metric_fn,
                          [actual_labels, model_spec.predictions, weights])
        return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=model_spec.loss,
            predictions=model_spec.predictions,
            train_op=train_op,
            eval_metrics=eval_metrics,
            training_hooks=training_hooks)

      eval_metric_ops = None
      if mode != tf.estimator.ModeKeys.PREDICT and not self._head:
        weights = None
        if self._phoenix_spec.weight_feature_name:
          weights = features[self._phoenix_spec.weight_feature_name]
        eval_metric_ops = self._metric_fn(actual_labels, model_spec.predictions,
                                          weights)
      if self._head:
        return self._head.create_estimator_spec(
            features,
            mode,
            model_spec.eval_logits,
            labels,
            train_op_fn=lambda _: train_op)

      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model_spec.loss,
          predictions=model_spec.predictions,
          train_op=train_op,
          training_hooks=training_hooks + model_spec.train_hooks,
          eval_metric_ops=eval_metric_ops)

    return model_fn

  # TODO(b/172564129): Move increment_global_step to TaskManager.
  # TODO(b/172564129): Figure out how to set train steps for multi-task.
  def _increment_global_step(self, train_op, train_steps, tower_name):
    """Increments the global step based on the tower size.

    N.B. if the tower size does not divide evenly into the train_steps, it will
    train for longer than required.

    Args:
      train_op: The train_op to execute before incrementing the global_step.
      train_steps: The total number of steps to train for.
      tower_name: The name of the tower which is currently training.

    Returns:
      An tf.Op which increments the global step by the required amount.
    """
    if self._phoenix_spec.use_synchronous_optimizer:
      return train_op

    increment_amount = 1
    tower_size = architecture_utils.get_architecture_size(tower_name)
    if (self._phoenix_spec.use_parameter_scaled_training and tower_size):
      train_step_per_block = max(
          int(train_steps // self._phoenix_spec.maximum_depth), 1)
      tower_train_steps = tower_size * train_step_per_block
      increment_amount = max(int(train_steps // tower_train_steps), 1)

    with tf.control_dependencies([train_op]):
      global_step = tf.compat.v1.train.get_or_create_global_step()
      return tf.compat.v1.assign_add(global_step, increment_amount)

  def get_estimator(self, run_config, hparams, train_steps):
    """Returns a Phoenix `Estimator` for train and evaluation.

    Args:
      run_config: `RunConfig` object to configure the runtime settings.
      hparams: `HParams` instance defining custom hyperparameters.
      train_steps: The total number of training steps.

    Returns:
      Returns an `Estimator`.

    Raises:
      ValueError: in case flatten is used as a search block or is missing from
      the initial architecture.
    """

    if not all("FLATTEN" not in block for block in hparams.new_block_type):
      raise ValueError("Flatten cannot be a search block type")

    return Estimator(
        model_fn=self._make_model_fn(
            run_config=run_config, train_steps=train_steps, use_tpu=False),
        config=run_config,
        params=hparams,
        metadata=self._metadata)

  def get_tpu_estimator(self,
                        run_config,
                        hparams,
                        train_steps,
                        train_batch_size,
                        eval_on_tpu,
                        embedding_config_spec=None,
                        eval_batch_size=None):
    """Returns a Phoenix `Estimator` for train and evaluation.

    Args:
      run_config: `RunConfig` object to configure the runtime settings.
      hparams: `HParams` instance defining custom hyperparameters.
      train_steps: The total number of training steps.
      train_batch_size: batch size for train.
      eval_on_tpu: whether to use tpu for evaluation.
      embedding_config_spec: (Optional) Embedding config spec instance.
      eval_batch_size: (Optional) if not set, we use train batch size.

    Returns:
      Returns an `TPUEstimator`.

    Raises:
      ValueError: in case flatten is used as a search block or is missing from
      the initial architecture.
    """

    if not all("FLATTEN" not in block for block in hparams.new_block_type):
      raise ValueError("Flatten cannot be a search block type")

    return tf.compat.v1.estimator.tpu.TPUEstimator(
        model_fn=self._make_model_fn(
            run_config=run_config, train_steps=train_steps, use_tpu=True),
        config=run_config,
        use_tpu=True,
        params=hparams,
        train_batch_size=train_batch_size,
        eval_batch_size=(eval_batch_size or train_batch_size),
        embedding_config_spec=embedding_config_spec,
        eval_on_tpu=eval_on_tpu)


  @staticmethod
  def get_keras_hyperparameters_space(phoenix_spec, train_steps):
    """Gets the Phoenix search space as keras Hyperparameters object."""
    hp_space = keras_tuner.HyperParameters()
    hp_space.merge(
        architecture_utils.get_blocks_search_space(phoenix_spec.blocks_to_use))
    hp_space.Float("learning_rate", 1e-6, 0.01, sampling="log")
    hp_space.Choice("new_block_type", phoenix_spec.blocks_to_use)

    # Try different optimizers.
    hp_space.Choice("optimizer",
                    ["momentum", "sgd", "adagrad", "adam", "rmsprop"])

    # Search for the best tower of depth phoenix_spec.minimum_depth
    # Used for initial structure (before evolution + going deeper).
    for i in range(phoenix_spec.minimum_depth):
      hp_space.Choice("initial_architecture_{}".format(i),
                      phoenix_spec.blocks_to_use)

    learning_spec = phoenix_spec.learning_spec

    # Exponential decay.
    if learning_spec.apply_exponential_decay:
      hp_space.Float("exponential_decay_rate",
                     learning_spec.min_learning_rate_decay_rate,
                     learning_spec.max_learning_rate_decay_rate)
      decay_steps = [
          train_steps // i for i in range(learning_spec.min_decay_times,
                                          learning_spec.max_decay_times)
      ]
      seen = set()
      unique_decay_steps = [
          x for x in decay_steps if not (x in seen or seen.add(x))
      ]
      hp_space.Choice("exponential_decay_steps", unique_decay_steps)

    # Gradient clipping
    if learning_spec.apply_gradient_clipping:
      hp_space.Int("gradient_max_norm",
                   learning_spec.min_gradient_norm_when_clipping,
                   learning_spec.max_gradient_norm_when_clipping)

    # L2 regularization
    if learning_spec.apply_l2_regularization:
      hp_space.Float("l2_regularization", learning_spec.min_l2_regularization,
                     learning_spec.max_l2_regularization)

    # Apply dropout between blocks. Here -1 wouldn't apply any dropouts.
    if phoenix_spec.apply_dropouts_between_blocks:
      assert learning_spec.min_dropout < learning_spec.max_dropout
      step = (learning_spec.max_dropout - learning_spec.min_dropout) / 10
      dropout = np.arange(learning_spec.min_dropout, learning_spec.max_dropout,
                          step)
      hp_space.Choice("dropout_rate", [-1.0] + dropout.tolist())

    return hp_space
