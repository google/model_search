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
r"""A binary for running and benchmarking Model Search."""

import contextlib
import copy
import json
import os
import random

from absl import flags
from absl import logging
import kerastuner

from model_search import hparam as hp
from model_search import phoenix
from model_search import registry
from model_search.data import data as ms_data
from model_search.metadata import ml_metadata_db
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf
from google.protobuf import text_format

flags.DEFINE_integer("phoenix_batch_size", 100,
                     "Batch size used for training, eval and inference.")
flags.DEFINE_string(
    "phoenix_dataset", "", "Which dataset to train on. "
    "If only provider is registered then you do not have to set this flag. "
    "Needs to be set in all other cases with the dataset name.")
flags.DEFINE_integer("phoenix_train_steps", 1000, "Number of training steps.")
flags.DEFINE_integer("phoenix_save_summary_steps", 2000,
                     "Save summaries every this many steps.")
flags.DEFINE_integer(
    "phoenix_save_checkpoints_secs", 120,
    "Number of seconds between checkpoint saves. "
    "This flag is ignored when autotune is used. "
    "Cannot be used with save_checkpoints_steps -- exactly one of "
    "save_checkpoints_secs and save_checkpoints_steps must be zero, and the "
    "other must be a strictly positive integer. Defaults to 120s.")
flags.DEFINE_integer(
    "phoenix_save_checkpoints_steps", 0,
    "Number of global steps between checkpoint saves."
    "This flag is ignored when autotune is used. "
    "Cannot be used with save_checkpoints_secs -- exactly one of "
    "save_checkpoints_secs and save_checkpoints_steps must be zero, and the "
    "other must be a strictly positive integer. Defaults to 0, which means "
    "save_checkpoints_steps is ignored. To use save_checkpoints_steps "
    "instead, set save_checkpoints_secs to 0 and set save_checkpoints_steps "
    "to a positive integer.")
flags.DEFINE_integer(
    "phoenix_keep_checkpoint_max", 5,
    "The maximum number of recent checkpoint files to keep. As new files are "
    "created, older files are deleted. If None or 0, all checkpoint files are "
    "kept. Defaults to 5 (i.e. the 5 most recent checkpoint files are kept.)")
flags.DEFINE_string(
    "phoenix_hparams", "",
    """A comma-separated list of `name=value` hyperparameter values. This flag
    is used to override hyperparameter settings either when manually selecting
    hyperparameters or when using tunning. If a hyperparameter setting is
    specified by this flag then it must be a valid hyperparameter name for the
    model; See create_hparams for the set of valid hyperparameter names.""")
flags.DEFINE_string(
    "phoenix_spec_filename", None, "Filename of a pbtxt with "
    "the phoenix spec. See configs/cnn.pbtxt as an example.")
flags.DEFINE_string(
    "optimization_metric", "loss",
    "Metric to optimize during hyperparameter tuning. Note "
    "this must be among the evaluation metrics computed (loss or accuracy).")
flags.DEFINE_enum(
    "optimization_goal", "minimize", ["minimize", "maximize"],
    "Whether to minimize or maximize the optimization objective.")
flags.DEFINE_bool(
    "phoenix_use_tpu", False, "Whether or not to use TPU. Limitations: "
    "TPU currently only support multiclass classification.")
flags.DEFINE_string("phoenix_master", "",
                    "BNS name of the TensorFlow master to use.")
flags.DEFINE_bool(
    "phoenix_eval_on_tpu", False,
    "Whether to use tpu for evaluation. Must be using 2x2 topology. "
    "Meaning, it has to be 1 tpu. Other topologies, like 8x8 are a tpu pod, "
    "which mean distributed evaluation; that is not supported yet in TF."
    "Additionally, use_tpu must be True for this to work.")
flags.DEFINE_bool("phoenix_export_saved_model", False,
                  "Whether to export saved models.")
flags.DEFINE_integer("phoenix_tf_random_seed", None,
                     "Graph level random seed for TensorFlow.")
flags.DEFINE_integer("phoenix_eval_steps", None,
                     "Number of batches used for evaluation.")
flags.DEFINE_string("tuner_id", "", "A tuner identifier.")

flags.DEFINE_enum(
    "hypertuning_method", "random", ["random", "bayesian"],
    "Whether to minimize or maximize the optimization objective.")
flags.DEFINE_string("model_dir", None, "A directory for the models (output).")
flags.DEFINE_string("experiment_name", None,
                    "A string holding the experiment name.")
flags.DEFINE_string("experiment_owner", None, "A string holding user id.")
flags.DEFINE_integer("experiment_max_num_trials", 200,
                     "Number of models to try (integer).")

FLAGS = flags.FLAGS

_MODEL_DIR_KEY = "model_dir"
_TF_CONFIG_ENV = "TF_CONFIG"
_SESSION_MASTER_KEY = "session_master"


# TODO(b/172564129): Split this file into library w/t flags and one with.
@contextlib.contextmanager
def _set_model_dir_for_run_config(model_dir=None):
  """ContextManager for overwriting environment configuration for RunConfig."""
  old_tf_config_str = os.environ.get(_TF_CONFIG_ENV)

  new_tf_config = (
      copy.deepcopy(json.loads(old_tf_config_str)) if old_tf_config_str else {})

  if model_dir is not None:
    new_tf_config[_MODEL_DIR_KEY] = model_dir

  if FLAGS.phoenix_master is not None:
    new_tf_config[_SESSION_MASTER_KEY] = FLAGS.phoenix_master

  os.environ[_TF_CONFIG_ENV] = json.dumps(new_tf_config)
  try:
    yield

  finally:
    if old_tf_config_str is not None:
      os.environ[_TF_CONFIG_ENV] = old_tf_config_str
    else:
      del os.environ[_TF_CONFIG_ENV]


def get_dataset_provider():
  """Helper function to get the data provider."""
  logging.info("Getting the registered data provider")
  # Reigstration API
  data_providers = registry.lookup_all(ms_data.Provider)
  if len(data_providers) == 1:
    return data_providers[0]

  # Registering more than one data provider
  else:
    logging.info("Registered data provider: %s", FLAGS.phoenix_dataset)
    return registry.lookup(FLAGS.phoenix_dataset, ms_data.Provider)


def loss_and_metric_and_predictions_fn(provider):
  """Helper function to create loss and metric fns."""
  metric_fn = None
  loss_fn = None
  predictions_fn = None
  if getattr(provider, "get_metric_fn", None) is not None:
    metric_fn = provider.get_metric_fn()
  if getattr(provider, "get_loss_fn", None) is not None:
    loss_fn = provider.get_loss_fn()
  if getattr(provider, "get_predictions_fn", None) is not None:
    predictions_fn = provider.get_predictions_fn()

  return (loss_fn, metric_fn, predictions_fn)


def make_run_config(model_dir=None, use_tpu=False):
  """Makes a RunConfig object with FLAGS.

  Args:
    model_dir: string - the model directory - to be used in the tpu run config
      only.
    use_tpu: boolean indicating if to use tpu run config or not.

  Returns:
    tf.estimator.RunConfig: Run config.

  Raises:
    ValueError: If not exactly one of `save_checkpoints_secs` and
      `save_checkpoints_steps` is specified.
  """
  save_checkpoints_secs = FLAGS.phoenix_save_checkpoints_secs or None
  save_checkpoints_steps = FLAGS.phoenix_save_checkpoints_steps or None
  if save_checkpoints_secs and save_checkpoints_steps:
    raise ValueError("save_checkpoints_secs and save_checkpoints_steps "
                     "cannot both be non-zero.")
  if not (save_checkpoints_secs or save_checkpoints_steps):
    raise ValueError("save_checkpoints_secs and save_checkpoints_steps "
                     "cannot both be zero.")

  if use_tpu:
    return tf.compat.v1.estimator.tpu.RunConfig(
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=200),
        model_dir=model_dir,
        master=FLAGS.phoenix_master,
        save_summary_steps=FLAGS.phoenix_save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.phoenix_keep_checkpoint_max,
        tf_random_seed=None)

  return tf.estimator.RunConfig(
      model_dir=model_dir,
      save_summary_steps=FLAGS.phoenix_save_summary_steps,
      save_checkpoints_secs=save_checkpoints_secs,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.phoenix_keep_checkpoint_max,
      tf_random_seed=FLAGS.phoenix_tf_random_seed)


def run_train_and_eval(hparams, model_dir, phoenix_instance, data_provider,
                       train_steps, eval_steps, batch_size):
  """Trains one trial."""
  run_config = make_run_config(
      model_dir=model_dir, use_tpu=FLAGS.phoenix_use_tpu)

  if FLAGS.phoenix_use_tpu:
    estimator = phoenix_instance.get_tpu_estimator(
        run_config=run_config,
        hparams=hparams,
        train_steps=train_steps,
        train_batch_size=batch_size,
        eval_on_tpu=FLAGS.phoenix_eval_on_tpu)
  else:
    estimator = phoenix_instance.get_estimator(
        run_config=run_config, hparams=hparams, train_steps=train_steps)
  estimator.train(
      input_fn=data_provider.get_input_fn(
          hparams=hparams,
          mode=tf.estimator.ModeKeys.TRAIN,
          batch_size=batch_size),
      max_steps=train_steps)
  tf.compat.v1.reset_default_graph()
  tf.keras.backend.clear_session()
  tf.compat.v1.Session.reset(target=FLAGS.phoenix_master)
  assert eval_steps > 0
  eval_results = estimator.evaluate(
      input_fn=data_provider.get_input_fn(
          hparams=hparams,
          mode=tf.estimator.ModeKeys.EVAL,
          batch_size=batch_size),
      steps=eval_steps)
  tf.compat.v1.reset_default_graph()
  tf.keras.backend.clear_session()
  tf.compat.v1.Session.reset(target=FLAGS.phoenix_master)
  logging.info("Evaluation results: %s", eval_results)
  # Oracle save metrics as json which cannot handle numpy
  eval_results = {k: v.item() for k, v in eval_results.items()}
  return eval_results


def get_trial_dir(model_dir, tuner_id):
  """Helper function to get trial directory."""
  tuner_dir = os.path.join(model_dir, tuner_id)
  if not tf.io.gfile.exists(tuner_dir):
    tf.io.gfile.makedirs(tuner_dir)
  existing_trials = tf.io.gfile.listdir(tuner_dir)
  if not existing_trials:
    trial_dir = os.path.join(tuner_dir, "1")
  else:
    last_id = sorted([
        int(os.path.basename(t_dir))
        for t_dir in existing_trials
        if os.path.basename(t_dir).isdigit()
    ])[-1]
    trial_dir = os.path.join(tuner_dir, str(int(last_id) + 1))

  if tf.io.gfile.exists(trial_dir):
    tf.io.gfile.rmtree(trial_dir)

  logging.info("creating directory: %s", trial_dir)
  tf.io.gfile.makedirs(trial_dir)
  return trial_dir


def aggregate_initial_architecture(hparams):
  """Helper function to aggregate initial architecture into an array hparam."""
  output = hparams.copy()
  initial_architecture_size = len(
      [hp for hp in hparams.keys() if hp.startswith("initial_architecture")])
  output["initial_architecture"] = [
      hparams["initial_architecture_{}".format(i)]
      for i in range(initial_architecture_size)
  ]
  return output


def run_parameterized_train_and_eval(phoenix_instance, oracle, tuner_id,
                                     root_dir, max_trials, data_provider,
                                     train_steps, eval_steps, batch_size):
  """Train, getting parameters from a tuner.

  Args:
    phoenix_instance: a phoenix.Phoenix object.
    oracle: a kerastuner oracle.
    tuner_id: identifier of the tuner (integer).
    root_dir: the root directory to save the models.
    max_trials: the maximal number of trials allowed.
    data_provider: The data provider object.
    train_steps: The number of training steps.
    eval_steps: The number of evaluation steps.
    batch_size: The batch size (integer).

  Returns:
    True if the tuner provided a trial to run, False if the tuner
    has run out of trials.
  """
  trial = oracle.create_trial(tuner_id)

  trial_dir = get_trial_dir(root_dir, tuner_id)
  my_id = int(os.path.basename(trial_dir))
  if my_id > max_trials:
    return False

  hparams = trial.hyperparameters
  phoenix_hparams = aggregate_initial_architecture(hparams.values)

  logging.info("Tuner id: %s", tuner_id)
  logging.info("Training with the following hyperparameters: ")
  logging.info(phoenix_hparams)

  with _set_model_dir_for_run_config(model_dir=trial_dir):
    evaluation_metrics = run_train_and_eval(
        hparams=hp.HParams(**phoenix_hparams),
        model_dir=trial_dir,
        phoenix_instance=phoenix_instance,
        data_provider=data_provider,
        train_steps=train_steps,
        eval_steps=eval_steps,
        batch_size=batch_size)

  oracle.update_trial(
      trial_id=trial.trial_id,
      metrics=evaluation_metrics,
      step=evaluation_metrics["global_step"])
  oracle.end_trial(trial.trial_id,
                   kerastuner.engine.trial.TrialStatus.COMPLETED)
  oracle.update_space(trial.hyperparameters)
  # Display needs the updated trial scored by the Oracle.
  # self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
  oracle.save()
  return True


def main(unused_argv):
  filename = FLAGS.phoenix_spec_filename
  spec = phoenix_spec_pb2.PhoenixSpec()
  with tf.io.gfile.GFile(filename, "r") as f:
    text_format.Merge(f.read(), spec)

  dataset_provider = get_dataset_provider()
  loss_fn, metric_fn, predictions_fn = (
      loss_and_metric_and_predictions_fn(dataset_provider))
  metadata = None
  if (FLAGS.optimization_goal != "minimize" or
      FLAGS.optimization_metric != "loss"):
    metadata = ml_metadata_db.MLMetaData(
        phoenix_spec=spec,
        study_name=FLAGS.experiment_name,
        study_owner=FLAGS.experiment_owner,
        optimization_goal=FLAGS.optimization_goal,
        optimization_metric=FLAGS.optimization_metric)
  phoenix_instance = phoenix.Phoenix(
      phoenix_spec=spec,
      input_layer_fn=dataset_provider.get_input_layer_fn(spec.problem_type),
      logits_dimension=dataset_provider.number_of_classes(),
      study_name=FLAGS.experiment_name,
      study_owner=FLAGS.experiment_owner,
      loss_fn=loss_fn,
      metric_fn=metric_fn,
      predictions_fn=predictions_fn,
      metadata=metadata)

  # Replay only!!
  if spec.HasField("replay"):
    hparams = hp.HParams.from_proto(spec.replay.towers[0].hparams)
    run_train_and_eval(
        hparams,
        FLAGS.model_dir,
        phoenix_instance,
        dataset_provider,
        train_steps=FLAGS.phoenix_train_steps,
        eval_steps=FLAGS.phoenix_eval_steps,
        batch_size=FLAGS.phoenix_batch_size)
    return 0

  tuner_id = FLAGS.tuner_id or "phoenix-tuner-%d" % random.randint(0, 10000000)
  hyperparameters = phoenix.Phoenix.get_keras_hyperparameters_space(
      spec, FLAGS.phoenix_train_steps)

  if FLAGS.hypertuning_method == "random":
    oracle = kerastuner.tuners.randomsearch.RandomSearchOracle(
        objective="loss",
        max_trials=FLAGS.experiment_max_num_trials,
        seed=73,
        hyperparameters=hyperparameters,
        allow_new_entries=True,
        tune_new_entries=True)
  else:
    oracle = kerastuner.tuners.bayesian.BayesianOptimizationOracle(
        objective="loss",
        hyperparameters=hyperparameters,
        max_trials=FLAGS.experiment_max_num_trials)

  # pylint: disable=protected-access
  oracle._set_project_dir(
      FLAGS.model_dir, FLAGS.experiment_name, overwrite=True)
  # pylint: enable=protected-access

  data_provider = get_dataset_provider()
  while run_parameterized_train_and_eval(
      phoenix_instance=phoenix_instance,
      oracle=oracle,
      tuner_id=tuner_id,
      root_dir=FLAGS.model_dir,
      max_trials=FLAGS.experiment_max_num_trials,
      data_provider=data_provider,
      train_steps=FLAGS.phoenix_train_steps,
      eval_steps=FLAGS.phoenix_eval_steps,
      batch_size=FLAGS.phoenix_batch_size):
    pass
