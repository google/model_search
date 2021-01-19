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
"""Arbitrary DAG architectures for Phoenix."""

import abc
import collections
import enum
import time

from absl import logging

from model_search import blocks_builder
from model_search import utils
from model_search.architecture import architecture_utils
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf

INPUT_INDICES_PADDED_TENSOR_NAME = "input_indices_padded"
INPUT_INDICES_LENGTHS_TENSOR_NAME = "input_indices_lengths"
COMBINER_TYPES_TENSOR_NAME = "combiner_type"
BLOCK_TYPES_TENSOR_NAME = "block_types"
INPUT_KEYS_TENSOR_NAME = "input_keys"


def _compute_paddings(height_pad_amt, width_pad_amt, patch_axes):
  """Convert the total pad amounts to the format needed by tf.pad()."""

  top_pad = height_pad_amt // 2
  bottom_pad = height_pad_amt - top_pad
  left_pad = width_pad_amt // 2
  right_pad = width_pad_amt - left_pad

  paddings = [[0, 0] for _ in range(4)]
  paddings[patch_axes[0]] = [top_pad, bottom_pad]
  paddings[patch_axes[1]] = [left_pad, right_pad]

  return paddings


def _pad_to_match(tensor_list, patch_axes):
  """Pads outputs of CNN examples until the patch sizes are equal."""

  tensor_heights = [tensor.shape[patch_axes[0]] for tensor in tensor_list]
  new_height = max(tensor_heights)
  height_pad_amts = [
      new_height - tensor_height for tensor_height in tensor_heights
  ]

  tensor_widths = [tensor.shape[patch_axes[1]] for tensor in tensor_list]
  new_width = max(tensor_widths)
  width_pad_amts = [new_width - tensor_width for tensor_width in tensor_widths]

  padded_tensor_list = []
  for tensor, height_pad_amt, width_pad_amt in zip(tensor_list, height_pad_amts,
                                                   width_pad_amts):
    paddings = _compute_paddings(height_pad_amt, width_pad_amt, patch_axes)
    padded_tensor = tf.pad(tensor, paddings)
    padded_tensor_list.append(padded_tensor)

  return padded_tensor_list


class InputSelector:
  """Selects which previous layers to use."""

  def __init__(self, input_indices):
    """Constructs an InputSelector.

    Args:
      input_indices: List of ints that determines which previous layers to pass
        to the next Combiner. For example, [-1] means to pass the only the
        output of the previous block, [-1, -2] means to pass the outputs of the
        previous two blocks.
    """
    self._input_indices = input_indices

  def __call__(self, inputs):
    """Returns the inputs selected by the input_indices."""

    return [inputs[index] for index in self._input_indices]


class Combiner(metaclass=abc.ABCMeta):
  """Used to create the input layer(s) for a Block.

  The most typical use case combines multiple inputs from previous layers into
  a single Tensor, by operations such as sum, depthwise concat, projection, and
  padding operations. Some blocks (such as NASNet) takes 2 Tensors as
  inputs, so we cannot make all Reducers output only one Tensor.
  """

  @abc.abstractproperty
  def name(self):
    """Name of the Combiner."""

  @abc.abstractmethod
  def __call__(self, inputs, **kwargs):
    """Combines inputs to produce the Tensor(s) used by the next Block.

    Args:
      inputs: A list of Tensors selected by the InputSelector.  Returns a list
        of Tensors to be used as input layers.
      **kwargs: Allows data_format to be propagated using arg_scope.

    Returns:
      A list of Tensors after being combined.
    """


class IdentityCombiner(Combiner):
  """Returns the same Tensors passed into it."""

  @property
  def name(self):
    return "identity"

  def __call__(self, inputs, **kwargs):
    del kwargs
    return inputs


class ConcatCombiner(Combiner):
  """Concatenates the Tensors passed into it along the channels dimension."""

  @property
  def name(self):
    return "concat"

  def __call__(self, inputs, **kwargs):
    if len(inputs) == 1:
      return inputs

    input_shapes = [input_t.shape for input_t in inputs]
    with tf.compat.v1.name_scope("combiner_" + self.name):
      # Fully connected case
      if all(len(shape) <= 2 for shape in input_shapes):
        return [tf.concat(inputs, axis=-1)]

      # CNN case
      if all(len(shape) == 4 for shape in input_shapes):
        if ("data_format" in kwargs and
            kwargs["data_format"].lower() in ["nchw", "channels_first"]):
          concat_axis = 1
          patch_axes = [2, 3]
        else:
          concat_axis = -1
          patch_axes = [1, 2]

        padded_inputs = _pad_to_match(inputs, patch_axes)
        return [tf.concat(padded_inputs, axis=concat_axis)]

    raise NotImplementedError(
        "ConcatCombiner does not know how to deal with inputs of these shapes. "
        "Input shapes: {}".format(input_shapes))


class CombinerType(enum.IntEnum):
  CONCAT = 0
  IDENTITY = 1


COMBINER_MAP = {
    CombinerType.CONCAT: ConcatCombiner(),
    CombinerType.IDENTITY: IdentityCombiner(),
}

BLOCK_BUILDER_MAP = blocks_builder.Blocks()


class Node(
    collections.namedtuple("Node",
                           ["block_type", "input_indices", "combiner_type"])):
  """Container for the repeated units in an Architecture."""

  def __new__(cls, block_type, input_indices=None, combiner_type=None):
    """Constructs an Node.

    Args:
      block_type: int for the Block type.
      input_indices: List of ints that determines which previous layers to pass
        to the next Combiner. For example, [-1] means to pass the only the
        output of the previous block, [-2, -1] means to pass the outputs of the
        previous two blocks. If None, defaults to [-1] (the previous layer).
      combiner_type: int for the Combiner type. If None, defaults to 0
        (identity).

    Returns:
      Node.
    """

    if input_indices is None:
      input_indices = [-1]

    if combiner_type is None:
      combiner_type = CombinerType.CONCAT

    return super().__new__(cls, block_type, input_indices, combiner_type)

  @property
  def input_selector(self):
    return InputSelector(self.input_indices)

  @property
  def combiner(self):
    return COMBINER_MAP[self.combiner_type]

  @property
  def block_builder(self):
    return BLOCK_BUILDER_MAP[self.block_type]

  @property
  def block_name(self):
    return str(blocks_builder.BlockType(self.block_type).name)


class Architecture(
    collections.namedtuple("Architecture",
                           ["tower_name", "node_list", "input_keys"])):
  """Defines how the inputs, hidden layers, and outputs are connected.

  This represents the architecture of a NN by a sequence with this pattern:

  (Layer 0)_______
  |InputSelector  |
  |Combiner       | (Node 0)
  |Block__________|
  (Layer 1)_______
  |InputSelector  |
  |Combiner       | (Node 1)
  |Block__________|
  (Layer 2)_______
  |InputSelector  |
  |Combiner       | (Node 2)
  |Block__________|
  (Layer 3)
  .
  .
  .
  (Layer d)

  Since every InputSelector has access to all preceding layers, we can represent
  any arbitrary DAG. To obtain the adjacency matrix, we can traverse the
  sequence and concatentate the InputSelectors' outputs.
  """

  def __new__(cls, node_list, input_keys=None, tower_name="search_generator_0"):
    """Constructs an Architecture.

    Args:
      node_list: List of Nodes. The first Node is closest to the inputs, the
        last Node is closest to the outputs.
      input_keys: List of string keys, which fixes the order of the input
        Tensors from the input dict passed into construct_tower call. If None,
        assumes that the construct_tower will be called with a single input
        Tensor instead of a dict, and the NN takes only one input Tensor.
      tower_name: str name of the instance. Defaults to "search_generator_0".

    Returns:
      Architecture.
    """

    return super().__new__(
        cls, node_list=node_list, input_keys=input_keys, tower_name=tower_name)

  def create_logits_spec(self,
                         phoenix_spec,
                         pre_logits,
                         dimension,
                         is_frozen,
                         lengths=None):
    """Creates the logits for the tower.

    Args:
      phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
      pre_logits: `tf.Tensor` of the layer before the logits layer.
      dimension: int - the output tensor last axis dimension.
      is_frozen: Whether the tower should be frozen.
      lengths: A tensor of shape [batch] holding the sequence length for a
        sequential problem (rnn).

    Returns:
      A LogitsSpec containing the main and auxiliary logits and the architecture
      of the underlying tower.
    """

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
    elif phoenix_spec.problem_type in (phoenix_spec_pb2.PhoenixSpec.CNN,
                                       phoenix_spec_pb2.PhoenixSpec.DNN):
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

    return architecture_utils.LogitsSpec(logits, logits_weight, aux_logits,
                                         aux_logits_weight)

  def construct_tower(self,
                      phoenix_spec,
                      input_tensor,
                      is_training,
                      lengths,
                      logits_dimension,
                      is_frozen,
                      dropout_rate=None):
    """Creates a tower.

    Forked from architecture_utils.construct_tower.

    Args:
      phoenix_spec: The trial's `phoenix_spec_pb2.PhoenixSpec` proto.
      input_tensor: An input `tf.Tensor` or a Dict[str, tf.Tensor] to build the
        network on top of.
      is_training: a boolean indicating if we are in training.
      lengths: A `tf.Tensor` with the lengths (dimenions: [batch_size]) holding
        the length of each sequence for sequential problems. Keep as None, for
        non sequential problems.
      logits_dimension: The last axis dimension of the logits.
      is_frozen: Is the tower frozen - integer and not boolean.
      dropout_rate: a float indicating the rate of dropouts to apply between
        blocks. Applied only if the value is above zero.

    Returns:
      The output `tf.Tensor` of the last layer in the built neural network.
    """

    if self.input_keys is None and not isinstance(input_tensor, dict):
      all_layers = [input_tensor]
    elif self.input_keys and isinstance(input_tensor, dict):
      all_layers = [input_tensor[k] for k in self.input_keys]
    else:
      raise ValueError(
          "If Architecture's input_keys is None, input_tensor must be a "
          "Tensor. Otherwise, input_tensor must be a dict.")
    block_index = 1
    str_signature = ""
    with tf.compat.v1.variable_scope("Phoenix/{}".format(self.tower_name)):
      for node in self.node_list:
        str_signature += str(node.block_type)
        # TODO(b/172564129): Should block_index also be ignored when uniform
        # average transfer learning? How would we handle repeated blocks, e.g.
        # two FC layers stacked on top of each other.
        scope = "{0}_{1}_{2}".format(
            str(block_index), node.block_name, str_signature)
        scope = architecture_utils.strip_scope(
            scope, phoenix_spec.transfer_learning_spec.transfer_learning_type,
            str_signature)
        with tf.compat.v1.variable_scope(scope):
          with (architecture_utils.arg_scope(
              architecture_utils.DATA_FORMAT_OPS,
              data_format=phoenix_spec.cnn_data_format)):
            selected_layers = node.input_selector(all_layers)
            combined_layers = node.combiner(selected_layers)
            output = node.block_builder.build(
                input_tensors=combined_layers,
                is_training=is_training,
                lengths=lengths)
            if dropout_rate and dropout_rate > 0:
              output[-1] = tf.compat.v1.layers.dropout(
                  output[-1], rate=dropout_rate, training=is_training)
            all_layers.append(output[-1])
            block_index += 1

      # Create the logits.
      scope = "last_dense_{}".format(str_signature)
      scope = architecture_utils.strip_scope(
          scope, phoenix_spec.transfer_learning_spec.transfer_learning_type,
          str_signature)
      with tf.compat.v1.variable_scope(scope):
        logits_spec = self.create_logits_spec(phoenix_spec, all_layers[-1],
                                              logits_dimension, is_frozen,
                                              lengths)

    self.save_to_graph()
    architecture_utils.set_parameter(
        self.tower_name, architecture_utils.DROPOUTS,
        (-1.0 if dropout_rate is None else dropout_rate), tf.float32)
    architecture_utils.set_parameter(self.tower_name,
                                     architecture_utils.IS_FROZEN,
                                     int(is_frozen))
    return logits_spec

  def save_to_graph(self):
    """Creates the variables in the tf.Graph.

    Helps in saving the architecture to from Checkpoint. This does not save the
    computation graph or trainable variables themselves, only the high-level
    structure.
    """
    input_indices_padded_name = "architectures/{}/{}".format(
        self.tower_name, INPUT_INDICES_PADDED_TENSOR_NAME)
    input_indices_lengths_name = "architectures/{}/{}".format(
        self.tower_name, INPUT_INDICES_LENGTHS_TENSOR_NAME)
    combiner_types_name = "architectures/{}/{}".format(
        self.tower_name, COMBINER_TYPES_TENSOR_NAME)
    block_types_name = "architectures/{}/{}".format(self.tower_name,
                                                    BLOCK_TYPES_TENSOR_NAME)
    input_keys_name = "architectures/{}/{}".format(self.tower_name,
                                                   INPUT_KEYS_TENSOR_NAME)

    input_indices_max_length = max(
        len(node.input_indices) for node in self.node_list)
    input_indices_padded_list = []
    input_indices_lengths = []
    combiner_types = []
    block_types = []
    for node in self.node_list:
      input_indices_padded = [0] * input_indices_max_length
      for i, input_index in enumerate(node.input_indices):
        input_indices_padded[i] = input_index
      input_indices_padded_list.append(input_indices_padded)
      input_indices_lengths.append(len(node.input_indices))
      combiner_types.append(node.combiner_type)
      block_types.append(node.block_type)

    tf.compat.v1.get_variable(
        name=input_indices_padded_name,
        shape=[len(self.node_list), input_indices_max_length],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(
            input_indices_padded_list),
        dtype=tf.int32)

    tf.compat.v1.get_variable(
        name=input_indices_lengths_name,
        shape=[len(self.node_list)],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(input_indices_lengths),
        dtype=tf.int32)

    tf.compat.v1.get_variable(
        name=combiner_types_name,
        shape=[len(self.node_list)],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(combiner_types),
        dtype=tf.int32)

    tf.compat.v1.get_variable(
        name=block_types_name,
        shape=[len(self.node_list)],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(block_types),
        dtype=tf.int32)

    tf.compat.v1.get_variable(
        name=input_keys_name,
        shape=[len(self.input_keys) if self.input_keys else 0],
        trainable=False,
        initializer=tf.compat.v1.constant_initializer(self.input_keys or []),
        dtype=tf.string)


def restore_from_checkpoint(directory, tower_name="search_generator_0"):
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

  input_indices_padded_np = reader.get_tensor("architectures/{}/{}".format(
      tower_name, INPUT_INDICES_PADDED_TENSOR_NAME))
  input_indices_lengths_np = reader.get_tensor("architectures/{}/{}".format(
      tower_name, INPUT_INDICES_LENGTHS_TENSOR_NAME))
  input_indices_lengths = input_indices_lengths_np.tolist()
  input_indices_padded_list = input_indices_padded_np.tolist()
  input_indices_list = []
  for input_indices_length, input_indices_padded in zip(
      input_indices_lengths, input_indices_padded_list):
    input_indices_list.append(input_indices_padded[:input_indices_length])

  combiner_types_np = reader.get_tensor("architectures/{}/{}".format(
      tower_name, COMBINER_TYPES_TENSOR_NAME))
  combiner_types = combiner_types_np.tolist()

  block_types_np = reader.get_tensor("architectures/{}/{}".format(
      tower_name, BLOCK_TYPES_TENSOR_NAME))
  block_types = block_types_np.tolist()

  input_keys_np = reader.get_tensor("architectures/{}/{}".format(
      tower_name, INPUT_KEYS_TENSOR_NAME))
  input_keys_b = input_keys_np.tolist()
  if input_keys_b:
    input_keys = [input_key_b.decode() for input_key_b in input_keys_b]
  else:
    input_keys = None

  node_list = []
  for input_indices, combiner_type, block_type in zip(input_indices_list,
                                                      combiner_types,
                                                      block_types):
    node_list.append(Node(block_type, input_indices, combiner_type))
  return Architecture(node_list, input_keys)


def old_to_new_architecture(old_architecture):
  """Convert architectures defined by block_types only.

  These architectures are more restricted -- they always have one input layer
  and one logits layer, and all blocks are only connected to the previous one or
  two blocks.

  Args:
    old_architecture: List of block_type ints.

  Returns:
    Architecture.
  """

  node_list = []
  for block_type in old_architecture:
    block_type_name = blocks_builder.BlockType(block_type).name
    if ("NASNET" in block_type_name) or ("AMOEBA" in block_type_name):
      input_indices = [-2, -1]
    else:
      input_indices = [-1]
    node_list.append(
        Node(
            block_type,
            input_indices=input_indices,
            combiner_type=CombinerType.CONCAT))
  return Architecture(node_list)
