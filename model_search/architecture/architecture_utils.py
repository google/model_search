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
"""Utils to handle architectures in Phoenix."""

import collections
import os
import time

from absl import logging
from model_search import blocks as blocks_lib
from model_search import blocks_builder as blocks
from model_search import hparam as hp
from model_search import utils
from model_search.proto import hparam_pb2
from model_search.proto import phoenix_spec_pb2
from model_search.proto import transfer_learning_spec_pb2
import numpy as np
import tensorflow.compat.v2 as tf
import tf_slim
from google.protobuf import text_format


NUMBER_OF_TOWERS = "number_of_towers"
DROPOUTS = "dropout_rate"
IS_FROZEN = "is_frozen"

arg_scope = tf_slim.arg_scope

# Some of the blocks use the ops below which require data_format.
# (specifying the axis channel in a rank 4 tensors). We use the list below
# and arg_scope to provide the data_format directly to blocks.
DATA_FORMAT_OPS = [
    tf_slim.avg_pool2d,
    tf_slim.max_pool2d,
    tf_slim.conv2d,
    tf_slim.batch_norm,
    tf_slim.separable_conv2d,
    blocks_lib.get_channel_dim,
]


# TODO(b/172564129): Consider not using a class
class DirectoryHandler(object):
  """Utility static functions to handle model directories."""

  @staticmethod
  def get_trial_id(directory, phoenix_spec):
    """Returns the trial id - best effort, None if unable.

    Args:
      directory: model directory (string).
      phoenix_spec: PhoenixSpec proto.

    Returns:
      an integer holding the trial id. Works for regular trial runs where the
      trial id is the directory name, and for TFX pipelines, where the trial
      id is encoded in the path as "Trial-xxxxxx".
      Returns None in other cases.
    """
    if phoenix_spec.HasField("replay") and len(phoenix_spec.replay.towers) == 1:
      return 1

    dir_name = os.path.basename(directory)
    if dir_name.isdigit():
      return int(dir_name)

    tfx_trial_prefix = "Trial-"
    dirs = directory.split("/")
    for level in dirs:
      if level.startswith(tfx_trial_prefix):
        return int(level[len(tfx_trial_prefix):])

    return None

  @staticmethod
  def trial_dir(trial):
    """Returns the model directory.

    Args:
      trial: model_search.metadata.trial.Trial

    Returns:
      model directory (string).
    """
    if getattr(trial, "model_dir", None) is not None:
      return trial.model_dir



def get_blocks_search_space(blocks_to_use=None):
  return blocks.Blocks.search_space(blocks_to_use=blocks_to_use)


def get_block_hparams(hparams, block_name):
  if not hparams:
    return None
  return hp.HParams(
      **{
          k[len(block_name + "_"):]: v
          for k, v in hparams.values().items()
          if k.startswith(block_name + "_")
      })


def get_hparams_from_dir(model_directory, tower_name):
  hparams = hparam_pb2.HParamDef()
  with tf.io.gfile.GFile(os.path.join(model_directory, tower_name), "r") as f:
    text_format.Parse(f.read(), hparams)
  output = hp.HParams(hparam_def=hparams)
  return output


def store_hparams_to_dir(hparams, model_directory, tower_name):
  # Removing the "context" hparam. This hparam is added for tpu, by the tpu team
  # It is not compatible with the function "to_proto" used below.
  copy_hparams = hp.HParams(**hparams.values())
  if hasattr(copy_hparams, "context"):
    copy_hparams.del_hparam("context")
  with tf.io.gfile.GFile(os.path.join(model_directory, tower_name), "w") as f:
    f.write(str(copy_hparams.to_proto()))


def get_architecture(directory, tower_name="search_generator_0"):
  """Given a trial directory and a tower name, returns its architecture.

  Args:
    directory: string - a model directory (that includes checkpoints).
    tower_name: string - the tower name we are trying to retrieve the
      architecutre for.

  Returns:
    np.array of ints holding the architecture of the Phoenix tower.
  """

  checkpoint = tf.train.latest_checkpoint(directory)
  trials = 3
  while checkpoint is None and trials >= 1:
    checkpoint = tf.train.latest_checkpoint(directory)
    time.sleep(1)
    trials -= 1

  reader = tf.compat.v1.train.NewCheckpointReader(checkpoint)
  return reader.get_tensor("architectures/{}".format(tower_name))


def set_architecture(architecture, tower_name="search_generator_0"):
  """Saves the tower architecture to the checkpoint as a tensor.

  Args:
    architecture: np.array of ints - holding the architecture.
    tower_name: string - the tower name we are trying to set the architecutre
      for.
  """
  tf.compat.v1.get_variable(
      name="architectures/{}".format(tower_name),
      shape=[architecture.size],
      trainable=False,
      initializer=tf.compat.v1.constant_initializer(architecture),
      dtype=tf.int32)


# TODO(b/172564129): Figure out a long term solution for getting the tower size.
def get_architecture_size(tower_name="search_generator_0"):
  """Returns the number of blocks of the architecture or None if ensembling."""
  prefix = "architectures/{}".format(tower_name)
  architecture = [
      var for var in tf.compat.v1.global_variables() if prefix in var.name
  ]
  assert len(architecture) <= 1
  size = None
  if architecture:
    size = architecture[0].shape[0]
  return size


# Getter and setter for the number of towers. These functions are not really
# necessary as they are equivalent to set and get architecture, however, this
# way the code will be more readable.
def get_number_of_towers(directory, generator_name):
  """Given a trial directory and a generator name, returns the number of towers.

  Args:
    directory: string - a model directory (that includes checkpoints).
    generator_name: string - the generator name that built the towers.

  Returns:
    np.array of ints - holding the number of towers (one integer) the generator
      created.
  """
  checkpoint = tf.train.latest_checkpoint(directory)
  reader = tf.compat.v1.train.NewCheckpointReader(checkpoint)
  return reader.get_tensor(NUMBER_OF_TOWERS + "/{}".format(generator_name))


def set_number_of_towers(generator_name, number_of_towers):
  """Saves the number of towers a generator has in the checkpoint as a tensor.

  Args:
    generator_name: string - the name of the generator.
    number_of_towers: int - the number of towers the generator created.
  """
  tf.compat.v1.get_variable(
      name=NUMBER_OF_TOWERS + "/{}".format(generator_name),
      shape=[],
      trainable=False,
      initializer=tf.compat.v1.constant_initializer(number_of_towers),
      dtype=tf.int32)


# These functions help Phoenix store and retrieve extra parameters for given
# towers, like for example "dropout_rate". This is important to be able to
# precisely reconstruct a tower solely from the checkpoint and eliminate
# external dependencies.
def get_parameter(directory, tower_name, parameter_name):
  """Get a param given a trial directory, a tower, and the parameter name.

  Args:
    directory: string - a model directory (that includes checkpoints).
    tower_name: string - the tower name.
    parameter_name: string - the parameter name.

  Returns:
    np.array - holding the parameter.
  """
  checkpoint = tf.train.latest_checkpoint(directory)
  reader = tf.compat.v1.train.NewCheckpointReader(checkpoint)
  return reader.get_tensor("params/{}/{}".format(tower_name, parameter_name))


def set_parameter(tower_name, parameter_name, value, dtype=tf.int32):
  """Saves a tower's parameter in the checkpoint as a tensor.

  Args:
    tower_name: string - the name of the tower.
    parameter_name: string - the name of the parameter.
    value: the value you wish to store.
    dtype: the type of the value. E.g., tf.int32.
  """
  tf.compat.v1.get_variable(
      name="params/{}/{}".format(tower_name, parameter_name),
      shape=[],
      trainable=False,
      initializer=tf.compat.v1.constant_initializer(value),
      dtype=dtype)


def fix_architecture_order(architecture, problem_type):
  """Fixes the architecture order of cnns.

  This function fixes the architecture for convolutional neural networks.
  Namely, if a dense block is before a convolutional block, then it switches
  the order. For the dnn and rnn case, the function doesn't do anything for
  the architecture as all architectures are valid.

  Args:
    architecture: an iterable of integers or `blocks.BlockType`.
    problem_type: a `PhoenixSpec.ProblemType` enum.

  Returns:
    a list of `blocks.BlockType`.
  """
  # All achitectures are valid in the dnn and rnn case.
  if problem_type != phoenix_spec_pb2.PhoenixSpec.CNN:
    return architecture

  output_architecture = []
  flattens = tuple(block for block in architecture
                   if "FLATTEN" in blocks.BlockType(block).name)
  if not flattens:
    output_architecture = [blocks.BlockType.PLATE_REDUCTION_FLATTEN]
    logging.warning("initial_architecture does not have a flattening " "block.")
    logging.info("Adding a Flatten block to the architecture.")
  else:
    output_architecture = [flattens[0]]

  for block in architecture:
    if (block == blocks.BlockType.FLATTEN or
        block == blocks.BlockType.DOWNSAMPLE_FLATTEN or
        block == blocks.BlockType.PLATE_REDUCTION_FLATTEN):
      continue
    output_architecture = increase_structure_depth(
        np.array(output_architecture), block, problem_type)
    output_architecture = [i.item() for i in output_architecture]
  return [blocks.BlockType(i) for i in output_architecture]


def increase_structure_depth(previous_architecture, added_block, problem_type):
  """Returns new structure given the old one and the added block.

  Increases the depth of the neural network by adding `added_block`.
  For the case of cnns, if the block is convolutional, it will add it before
  the flattening operation. Otherwise, if it is a dense block, then it will
  be added at the end.
  For the dnn and rnn case, the added_block is always added at the end.

  Args:
    previous_architecture: the input architecture. An np.array holding
      `blocks.BlockType` (i.e., holding integers).
    added_block: a `blocks.BlockType` to add to previous_architecture.
    problem_type: a `PhoenixSpec.ProblemType` enum.

  Returns:
    np.array of `blocks.BlockType` (integers).
  """
  if added_block == blocks.BlockType.EMPTY_BLOCK:
    return previous_architecture.copy()
  output = previous_architecture.copy()

  # No problems for DNN of RNN
  if problem_type != phoenix_spec_pb2.PhoenixSpec.CNN:
    return np.append(output, added_block)

  # TODO(b/172564129): Change this class (blocks) to a singleton
  builder = blocks.Blocks()
  # CNN case - convolution before fully connected.
  if not builder[added_block].is_input_order_important:
    return np.append(output, added_block)
  # First block index in which order is not important
  index_for_new_block = next(
      index for index, block in enumerate(previous_architecture)
      if not builder[block].is_input_order_important)
  return np.insert(output, index_for_new_block, added_block)


def init_variables(checkpoint, original_scope, new_scope):
  """Best effort: Initializes variables in a scope from a checkpoint.

  This function aims to warm start variables in a given scope from a
  checkpoint. The function will not fail if a variable was not found in the
  checkpoint. The function will fail if we are trying to warmstart a sharded
  variable from the checkpoint.

  Args:
    checkpoint: a filename of a valid TensorFlow checkpoint.
    original_scope: a string holding the scope of the tower in the checkpoint.
    new_scope: a string holding the scope of the tower in the current graph.

  Returns:
    The names of the tensors this function was able to warm start.
  """
  var_map = {
      v.op.name: v for v in tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=new_scope)
  }
  checkpoint_variables_list = [
      name for name, _ in tf.train.list_variables(checkpoint)
  ]
  no_partion_map = {}
  for var_name, variable in var_map.items():
    # Variable outsize the scope
    if not var_name.startswith(new_scope):
      continue

    # Creating the name as it should appear in the checkpoint
    normalized = var_name.replace(new_scope, original_scope)
    if var_name.endswith("/part_0"):
      normalized = normalized[:-len("/part_0")]

    # The variable is not in the checkpoint
    if normalized not in checkpoint_variables_list:
      logging.info("Cannot find %s in the checkpoint", normalized)
      continue
    # The variable is in the checkpoint but sharded.
    if var_name.endswith("/part_1"):
      logging.fatal("Partitioned variables are not allowed in "
                    "init_variables_from_checkpoint_if_exists")
    no_partion_map[normalized] = variable

  logging.info("warm starting the following tensors")
  logging.info(no_partion_map)
  tf.compat.v1.train.init_from_checkpoint(checkpoint, no_partion_map)
  return list(no_partion_map.keys())


# TODO(b/172564129): figure out a better long term solution.
def strip_scope(scope, transfer_learning_type, str_signature):
  """Strips signature from the scope if uniform average transfer learning.

  If we are warm starting the tower with the average values of the previous
  trials, do not append the signature to the scope as this will cause a mismatch
  when searching via mutation.

  Args:
    scope: The scope to potentially strip.
    transfer_learning_type: The current transfer learning type.
    str_signature: The scope signature to strip from the scope.

  Returns:
    Either the original scope or the stripped scope.
  """
  if (transfer_learning_type == transfer_learning_spec_pb2.TransferLearningSpec
      .UNIFORM_AVERAGE_TRANSFER_LEARNING):
    scope = scope[:-(len(str_signature) + 1)]
  return scope


def construct_tower(phoenix_spec,
                    input_tensor,
                    tower_name,
                    architecture,
                    is_training,
                    lengths,
                    logits_dimension,
                    is_frozen,
                    hparams,
                    model_directory,
                    dropout_rate=None,
                    allow_auxiliary_head=False):
  """Creates a tower giving an architecture.

  Args:
    phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
    input_tensor: An input `tf.Tensor` to build the network on top of.
    tower_name: a unique name for the tower (string).
    architecture: np.array of ints (`blocks.BlockType`) with the architecture of
      the neural network to build.
    is_training: a boolean indicating if we are in training.
    lengths: A `tf.Tensor` with the lengths (dimenions: [batch_size]) holding
      the length of each sequence for sequential problems. Keep as None, for non
      sequential problems.
    logits_dimension: The last axis dimension of the logits.
    is_frozen: Is the tower frozen - integer and not boolean.
    hparams: The hparams for the tower.
    model_directory: current trial model directory (a string).
    dropout_rate: a float indicating the rate of dropouts to apply between
      blocks. Applied only if the value is above zero.
    allow_auxiliary_head: Whether to allow importing the tower's auxiliary head,
      if the tower has one. Only applicable for CNNs.

  Returns:
    The output `tf.Tensor` of the last layer in the built neural network.
  """
  blocks_builders = blocks.Blocks()
  output = [input_tensor]
  block_index = 1
  str_signature = ""
  with tf.compat.v1.variable_scope("Phoenix/{}".format(tower_name)):
    for block_type in architecture:
      str_signature += str(block_type)
      # TODO(b/172564129): Should block_index also be ignored when uniform
      # average transfer learning? How would we handle repeated blocks, e.g. two
      # FC layers stacked on top of each other.
      scope = "{0}_{1}_{2}".format(
          str(block_index),
          blocks.BlockType(block_type).name, str_signature)
      scope = strip_scope(
          scope, phoenix_spec.transfer_learning_spec.transfer_learning_type,
          str_signature)
      with tf.compat.v1.variable_scope(scope):
        with (arg_scope(
            DATA_FORMAT_OPS, data_format=phoenix_spec.cnn_data_format)):
          output = blocks_builders[block_type].build(
              input_tensors=output,
              is_training=is_training,
              lengths=lengths,
              hparams=get_block_hparams(hparams,
                                        blocks.BlockType(block_type).name))
          if dropout_rate and dropout_rate > 0:
            output[-1] = tf.compat.v1.layers.dropout(
                output[-1], rate=dropout_rate, training=is_training)
          block_index += 1

    # Create the logits.
    scope = "last_dense_{}".format(str_signature)
    scope = strip_scope(
        scope, phoenix_spec.transfer_learning_spec.transfer_learning_type,
        str_signature)
    with tf.compat.v1.variable_scope(scope):
      tower_spec = create_tower_spec(phoenix_spec, output, architecture,
                                     logits_dimension, is_frozen, lengths,
                                     allow_auxiliary_head)

  set_architecture(architecture, tower_name)
  set_parameter(tower_name, DROPOUTS,
                (-1.0 if dropout_rate is None else dropout_rate), tf.float32)
  set_parameter(tower_name, IS_FROZEN, int(is_frozen))
  if not tf.io.gfile.exists(model_directory):
    tf.io.gfile.makedirs(model_directory)
  store_hparams_to_dir(
      hparams=hparams, model_directory=model_directory, tower_name=tower_name)
  return tower_spec


def import_tower(phoenix_spec,
                 features,
                 input_layer_fn,
                 shared_input_tensor,
                 original_tower_name,
                 new_tower_name,
                 model_directory,
                 new_model_directory,
                 is_training,
                 logits_dimension,
                 shared_lengths,
                 force_snapshot=False,
                 force_freeze=False,
                 allow_auxiliary_head=False):
  """Imports a tower from the given model directory and the tower's name.

  Args:
    phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
    features: feature dict in case the tower needs to create the input tensor
    input_layer_fn: A function that converts feature Tensors to input layer.
      See learning.autolx.model_search.data.Provider.get_input_layer_fn
      for details.
    shared_input_tensor: An input `tf.Tensor` to build the network on top of.
    original_tower_name: a unique name for the tower (string) to import.
    new_tower_name: the name to give the new tower.
    model_directory: string, holds the model directory of the trial to import
      the tower from.
    new_model_directory: the model directory of the current trial.
    is_training: a boolean indicating if we are in training.
    logits_dimension: The last axis dimension of the logits.
    shared_lengths: A `tf.Tensor` with the lengths (dimenions: [batch_size])
      holding the length of each sequence for sequential problems. Keep as None,
      for non sequential problems.
    force_snapshot: Force the tower to be imported from the previous trial even
      if transfer_learning_type is not SNAPSHOT_TRANSFER_LEARNING. This is used
      during ensembling to ensure that the imported towers are always warm
      started.
    force_freeze: Force the imported tower to be frozen. For correctness, only
      force freeze with snapshotting (otherwise, you will freeze an non-trained
      tower).
    allow_auxiliary_head: Whether to allow constructing an auxiliary head for
      the tower if possible. Only applicable for CNNs.

  Returns:
    A LogitsSpec containing the main and auxiliary logits and the architecture
    of the underlying tower.
  """
  initial_architecture = get_architecture(model_directory, original_tower_name)
  dropout_rate = get_parameter(model_directory, original_tower_name, DROPOUTS)
  is_frozen = get_parameter(model_directory, original_tower_name, IS_FROZEN)

  input_tensor = shared_input_tensor
  lengths = shared_lengths
  hparams = get_hparams_from_dir(model_directory, original_tower_name)
  if not phoenix_spec.is_input_shared:
    lengths_feature_name = phoenix_spec.lengths_feature_name
    if isinstance(features, dict) and lengths_feature_name not in features:
      lengths_feature_name = ""
    input_tensor, lengths = input_layer_fn(
        features=features,
        is_training=is_training,
        scope_name="Phoenix/" + new_tower_name + "/Input",
        lengths_feature_name=lengths_feature_name)

  tower_spec = construct_tower(
      phoenix_spec=phoenix_spec,
      input_tensor=input_tensor,
      tower_name=new_tower_name,
      architecture=initial_architecture,
      lengths=lengths,
      is_training=is_training,
      logits_dimension=logits_dimension,
      dropout_rate=dropout_rate,
      hparams=hparams,
      model_directory=new_model_directory,
      is_frozen=(is_frozen or force_freeze),
      allow_auxiliary_head=allow_auxiliary_head)

  if force_snapshot:
    init_variables(
        tf.train.latest_checkpoint(model_directory),
        "Phoenix/{}".format(original_tower_name),
        "Phoenix/{}".format(new_tower_name))
  return tower_spec


def get_tower_variables(tower_name):
  """Returns all the variables belonging to the tower with the given name."""
  prefix = "Phoenix/{}".format(tower_name)
  return [
      var for var in tf.compat.v1.trainable_variables() if prefix in var.name
  ]


class LogitsSpec(
    collections.namedtuple(
        "LogitsSpec",
        ["logits", "logits_weight", "aux_logits", "aux_logits_weight"])):
  """Structure which holds the logits and any aux head logits."""

  def __new__(cls,
              logits,
              logits_weight=1.,
              aux_logits=None,
              aux_logits_weight=None):
    if aux_logits is not None:
      msg = "aux_logits_weight cannot be None if aux_logits is not None"
      assert aux_logits_weight, msg
    return super(LogitsSpec, cls).__new__(cls, logits, logits_weight,
                                          aux_logits, aux_logits_weight)


class TowerSpec(
    collections.namedtuple("TowerSpec",
                           ["logits_spec", "architecture", "layer_tensors"])):
  """A tower's logits, architecture, and layer tensors."""

  def __new__(cls, logits_spec, architecture, layer_tensors):
    return super(TowerSpec, cls).__new__(cls, logits_spec, architecture,
                                         layer_tensors)


def create_tower_spec(phoenix_spec,
                      inputs,
                      architecture,
                      dimension,
                      is_frozen,
                      lengths=None,
                      allow_auxiliary_head=False):
  """Creates the logits for the tower.

  Args:
    phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
    inputs: The list of `tf.Tensors` of the tower.
    architecture: The list of `blocks.BlockType` of the tower architecture.
    dimension: int - the output tensor last axis dimension.
    is_frozen: Whether the tower should be frozen.
    lengths: A tensor of shape [batch] holding the sequence length for a
      sequential problem (rnn).
    allow_auxiliary_head: Whether to allow creating an auxiliary head if
      possible. Only applicable for CNNs.

  Returns:
    A LogitsSpec containing the main and auxiliary logits and the architecture
    of the underlying tower.
  """

  # Discard inputs[0] since this is the raw features.
  all_layer_tensors = inputs
  pre_logits = inputs[-1]
  logits_weight = 1.0
  aux_logits = None
  aux_logits_weight = None
  if (phoenix_spec.problem_type ==
      phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS):
    logits = tf.compat.v1.layers.conv1d(
        inputs=pre_logits, filters=dimension, kernel_size=1)
  elif (phoenix_spec.problem_type ==
        phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS):
    if lengths is not None:
      logits = utils.last_activations_in_sequence(
          tf.compat.v1.layers.conv1d(
              inputs=pre_logits, filters=dimension, kernel_size=1), lengths)
    else:
      logging.warning("Length is missing for rnn_last problem type.")
      logits = tf.compat.v1.layers.conv1d(
          inputs=pre_logits, filters=dimension, kernel_size=1)
  elif phoenix_spec.problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
    logits = tf.keras.layers.Dense(dimension, name="dense")(pre_logits)
    if allow_auxiliary_head and phoenix_spec.use_auxiliary_head:
      reductions = []
      flattens = []
      for i, block in enumerate(architecture):
        name = blocks.BlockType(block).name
        if "DOWNSAMPLE" in name or "REDUCTION" in name:
          reductions.append(i)
        # Some blocks reduce and flatten.
        if "FLATTEN" in name:
          flattens.append(i)
      if reductions:
        # Add the auxiliary head right before the reduction cell.
        idx = reductions[-1]
        aux_logits = _build_nas_aux_head(inputs[idx], dimension,
                                         phoenix_spec.cnn_data_format)
        if aux_logits is not None:
          aux_logits_weight = phoenix_spec.auxiliary_head_loss_weight
      if flattens and aux_logits is None:
        idx = flattens[-1]
        aux_logits = tf.keras.layers.Dense(
            dimension, name="aux_dense")(
                inputs[idx])
        aux_logits_weight = phoenix_spec.auxiliary_head_loss_weight
  elif phoenix_spec.problem_type == phoenix_spec_pb2.PhoenixSpec.DNN:
    logits = tf.keras.layers.Dense(dimension, name="dense")(pre_logits)
  else:
    raise ValueError("phoenix_spec.problem_type must be either DNN, CNN, "
                     "RNN_LAST_ACTIVATIONS, or RNN_ALL_ACTIVATIONS.")

  logits = tf.identity(logits, name="logits")
  if aux_logits is not None:
    aux_logits = tf.identity(aux_logits, name="aux_logits")

  # TODO(b/172564129): Remove from eval graph.
  if is_frozen:
    logits = tf.stop_gradient(logits)
    if aux_logits is not None:
      aux_logits = tf.stop_gradient(aux_logits)

  return TowerSpec(
      logits_spec=LogitsSpec(logits, logits_weight, aux_logits,
                             aux_logits_weight),
      architecture=[blocks.BlockType(block).name for block in architecture],
      layer_tensors=all_layer_tensors)


def _build_nas_aux_head(inputs, dimension, data_format):
  """Builds the auxiliary head described in the NAS paper."""
  shape = inputs.shape
  if shape.rank < 4:
    return None
  shape = shape.as_list()
  shape = shape[1:3] if data_format == "NHWC" else shape[2:4]
  if np.any(np.array(shape) < np.array([5, 5])):
    return None

  with tf.compat.v1.variable_scope("aux_logits"):
    with arg_scope(DATA_FORMAT_OPS, data_format=data_format):
      aux_logits = tf_slim.avg_pool2d(inputs, [5, 5], stride=3, padding="SAME")
      aux_logits = tf_slim.conv2d(aux_logits, 128, [1, 1], scope="proj")
      aux_logits = tf_slim.batch_norm(aux_logits, scope="aux_bn0")
      aux_logits = tf.nn.relu6(aux_logits)
      # Shape of feature map before the final layer.
      shape = aux_logits.shape
      shape = shape[1:3] if data_format == "NHWC" else shape[2:4]
      aux_logits = tf_slim.conv2d(aux_logits, 768, shape, padding="VALID")
      aux_logits = tf_slim.batch_norm(aux_logits, scope="aux_bn1")
      aux_logits = tf.nn.relu6(aux_logits)
      aux_logits = tf.keras.layers.Flatten()(aux_logits)
      aux_logits = tf_slim.fully_connected(aux_logits, dimension)

  return aux_logits
