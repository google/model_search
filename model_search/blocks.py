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
# pylint: disable=super-init-not-called
# No need to call init of superclass "Block", since Block is an abstract class.
"""Well known building blocks for Convolutional networks."""

import abc

import enum
import functools

import kerastuner

from model_search import registry
from model_search.ops import svdf_cell
from model_search.ops import svdf_conv

import tensorflow.compat.v2 as tf
import tf_slim
# TODO(b/172564129): better documentation for this file. http://b/130796421


@tf_slim.add_arg_scope
def get_channel_dim(input_tensor, data_format='INVALID'):
  """Returns the number of channels in the input tensor."""
  shape = input_tensor.get_shape().as_list()
  assert data_format != 'INVALID'
  assert len(shape) == 4
  if data_format == 'NHWC':
    return int(shape[3])
  elif data_format == 'NCHW':
    return int(shape[1])
  else:
    raise ValueError('Not a valid data_format', data_format)


class Block(object, metaclass=abc.ABCMeta):
  """Block api for creating a new block."""

  @abc.abstractmethod
  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds a block for phoenix.

    Args:
      input_tensors: A list of input tensors.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: The training HParams.

    Returns:
      output_tensors: A list of the output tensors.
    """

  @abc.abstractproperty
  def is_input_order_important(self):
    """Is the order of the entries in the input tensor important.

    Returns:
      A bool specifying if the order of the entries in the input is important.
      Examples where the order is important: Input for a cnn layer.
      (e.g., pixels an image). Examples when the order is not important:
      Input for a dense layer.
    """

  def requires_hparams(self):
    """Returns a search space of hparams in case the block is tunable.

    Returns:
      kerastuner.engine.hyperparameters.HyperParameters object
    """
    return None


# NEXT ID: 146
# NEXT EXPERIMENTAL ID: 10017 (experiment id starts at 10,001)
register_block = functools.partial(registry.register, base=Block)

# IMPORTANT NOTE:
# When you use Keras layers with variables, always give them a name!
# If not, keras will add "_#" (e.g., dense_1 instead of dense). It will add
# the suffix even if the outer-scope is different. This is a surprising behavior
# TODO(b/172564129): Contact the Keras team about this.


@register_block(
    lookup_name='FIXED_CHANNEL_CONVOLUTION_16',
    init_args={'num_filters': 16},
    enum_id=1)
@register_block(
    lookup_name='FIXED_CHANNEL_CONVOLUTION_32',
    init_args={'num_filters': 32},
    enum_id=2)
@register_block(
    lookup_name='FIXED_CHANNEL_CONVOLUTION_64',
    init_args={'num_filters': 64},
    enum_id=3)
@register_block(
    lookup_name='FIXED_CHANNEL_CONVOLUTION_120',
    init_args={'num_filters': 120},
    enum_id=33)
@register_block(
    lookup_name='FIXED_CHANNEL_STRIDE_96_2',
    init_args={
        'num_filters': 96,
        'apply_batch_norm': True,
        'stride': (2, 2)
    },
    enum_id=68)
@register_block(
    lookup_name='FIXED_CHANNEL_CONVOLUTION_96',
    init_args={
        'num_filters': 96,
        'apply_batch_norm': True
    },
    enum_id=72)
class FixedChannelConvolutionBlock(Block):
  """First block to increase the number of channels in an image."""

  def __init__(self,
               num_filters=64,
               kernel_size=3,
               apply_batch_norm=True,
               stride=(1, 1)):
    self._num_filters = num_filters
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm
    self._stride = stride

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    # We always add a layer on the last tensor provided.
    input_tensor = input_tensors[-1]
    # TODO(b/172564129): Revert to keras layers http://b/130791880
    net = tf_slim.conv2d(
        input_tensor,
        self._num_filters,
        kernel_size=self._kernel_size,
        stride=self._stride,
        padding='same')

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    # TODO(b/172564129): give an options to choose relu.
    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='CONVOLUTION_3X3', init_args={'kernel_size': 3}, enum_id=4)
@register_block(
    lookup_name='CONVOLUTION_5X5', init_args={'kernel_size': 5}, enum_id=5)
@register_block(
    lookup_name='CONVOLUTION_1X3',
    init_args={'kernel_size': (1, 3)},
    enum_id=26)
@register_block(
    lookup_name='CONVOLUTION_1X5',
    init_args={'kernel_size': (1, 5)},
    enum_id=27)
@register_block(
    lookup_name='CONVOLUTION_1X7',
    init_args={'kernel_size': (1, 7)},
    enum_id=28)
@register_block(
    lookup_name='CONVOLUTION_3X1',
    init_args={'kernel_size': (3, 1)},
    enum_id=29)
@register_block(
    lookup_name='CONVOLUTION_5X1',
    init_args={'kernel_size': (5, 1)},
    enum_id=30)
@register_block(
    lookup_name='CONVOLUTION_7X1',
    init_args={'kernel_size': (7, 1)},
    enum_id=31)
class ConvolutionBlock(Block):
  """Regular convolution layer."""

  def __init__(self, kernel_size=3, apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='INCREASE_CHANNELS_3X3',
    init_args={'kernel_size': 3},
    enum_id=12)
@register_block(
    lookup_name='INCREASE_CHANNELS_5X5',
    init_args={'kernel_size': 5},
    enum_id=13)
class IncreaseChannelsBlock(Block):
  """Increase the number of channels times two."""

  def __init__(self, kernel_size=3, apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor) * 2,
        kernel_size=self._kernel_size,
        padding='same')

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='DOWNSAMPLE_CONVOLUTION_3X3',
    init_args={'kernel_size': 3},
    enum_id=6)
@register_block(
    lookup_name='DOWNSAMPLE_CONVOLUTION_5X5',
    init_args={'kernel_size': 5},
    enum_id=7)
class DownsampleConvolutionBlock(Block):
  """Downsample with stride and increase channels times two."""

  def __init__(self, kernel_size=3, strides=(2, 2), apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._strides = strides
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor) * 2,
        kernel_size=self._kernel_size,
        stride=self._strides,
        padding='same')

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='AVERAGE_POOL_2X2', init_args={'kernel_size': 2}, enum_id=8)
@register_block(
    lookup_name='AVERAGE_POOL_4X4', init_args={'kernel_size': 4}, enum_id=9)
class AveragePoolBlock(Block):
  """Average Pooling layer."""

  def __init__(self, kernel_size=2):
    self._kernel_size = kernel_size

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    if input_tensor.get_shape().as_list()[2] < self._kernel_size:
      return input_tensors

    # Average pool - reduce the size of the image.
    return input_tensors + [
        tf.keras.layers.AveragePooling2D(
            pool_size=self._kernel_size,
            strides=2,
            padding='SAME',
            name='avgpool2d')(input_tensor)
    ]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='RESNET_3X3', init_args={'kernel_size': 3}, enum_id=10)
@register_block(
    lookup_name='RESNET_5X5', init_args={'kernel_size': 5}, enum_id=11)
class ResnetBlock(Block):
  """A Resnet Block - 2 Convolutions with a skip connection.

  Note that we do not apply batch normalization or dropouts in the block.
  """

  def __init__(self, kernel_size=3, apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Residual unit with 2 sub layers."""
    input_tensor = input_tensors[-1]
    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)
    net = tf.nn.leaky_relu(net)

    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    net += input_tensor
    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


class ResidualConnectionType(enum.IntEnum):
  """Specifies the residual connection type for the fully connected layer."""
  # No residual connections.
  NONE = 1
  # If the output shape of the FullyConnectedBlock does not match the input
  # shape, it will be either padded or sliced so the shapes match. Computes
  # y = x' + f(x), where x' is the padded/sliced shape.
  FORCE_MATCH_SHAPES = 2
  # The input will be concatenated with the output from the
  # FullyConnectedBlock. Computes y = tf.concat(x, f(x)).
  CONCAT = 3
  # The input will first be projected to have the same shape as the output
  # before being added together. Computes y = x + (w*f(x) + b). Note that this
  # operation may increase the number of parameters. If the shapes of the
  # input and output tensors match, the tensors are simply added together.
  PROJECT = 4


@register_block(lookup_name='FULLY_CONNECTED', init_args={}, enum_id=14)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_FORCE_MATCH_SHAPES',
    init_args={
        'residual_connection_type': ResidualConnectionType.FORCE_MATCH_SHAPES
    },
    enum_id=79)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_FORCE_MATCH_SHAPES_BATCHNORM',
    init_args={
        'residual_connection_type': ResidualConnectionType.FORCE_MATCH_SHAPES,
        'apply_batch_norm': True
    },
    enum_id=80)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_CONCAT',
    init_args={'residual_connection_type': ResidualConnectionType.CONCAT},
    enum_id=81)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_CONCAT_BATCHNORM',
    init_args={
        'residual_connection_type': ResidualConnectionType.CONCAT,
        'apply_batch_norm': True
    },
    enum_id=82)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_PROJECT',
    init_args={'residual_connection_type': ResidualConnectionType.PROJECT},
    enum_id=83)
@register_block(
    lookup_name='FULLY_CONNECTED_RESIDUAL_PROJECT_BATCHNORM',
    init_args={
        'residual_connection_type': ResidualConnectionType.PROJECT,
        'apply_batch_norm': True
    },
    enum_id=84)
class FullyConnectedBlock(Block):
  """A fully connected layer with leaky relu activation.

  Output number of hidden nodes is equal to the input number of hidden nodes
  with some restrictions:
    - output number of hidden nodes is at least 2
    - output number of hidden nodes is at most max_output_size
    - the number of elements in the kernel matrix is at most
      max_number_of_parameters
  """

  def __init__(self,
               max_output_size=100,
               max_number_of_parameters=None,
               apply_batch_norm=False,
               residual_connection_type=None):
    """Initializes a new FullyConnectedBlock instance.

    Args:
      max_output_size: The maximum number of output neurons.
      max_number_of_parameters: The maximum number of parameters allowed.
      apply_batch_norm: Whether to apply batch normalization to the layer.
      residual_connection_type: The ResidualConnectionType to use in the layer.
        Suppose the input is x and the fully connected layer is f. Then this
        block will return y = x + f(x).
    """
    self._max_number_of_parameters = max_number_of_parameters
    self._max_output_size = max_output_size
    self._apply_batch_norm = apply_batch_norm
    self._residual_connection_type = (
        residual_connection_type or ResidualConnectionType.NONE)

  def _add_residual_connection(self, input_tensor, output_tensor):
    """Creates the residual connection between the input and the output."""
    if self._residual_connection_type == ResidualConnectionType.NONE:
      return output_tensor

    in_shape = input_tensor.shape[-1]
    out_shape = output_tensor.shape[-1]
    if (self._residual_connection_type ==
        ResidualConnectionType.FORCE_MATCH_SHAPES):
      if in_shape == out_shape:
        return input_tensor + output_tensor
      if in_shape > out_shape:
        return input_tensor[:, :out_shape] + output_tensor
      if in_shape < out_shape:
        input_tensor = tf.pad(
            tensor=input_tensor, paddings=[[0, 0], [0, out_shape - in_shape]])
        return input_tensor + output_tensor
    if self._residual_connection_type == ResidualConnectionType.CONCAT:
      return tf.concat((input_tensor, output_tensor), axis=1)
    # TODO(b/172564129): We do a similar projection for CNN auxiliary heads.
    # Extract the projection code into a util.
    if self._residual_connection_type == ResidualConnectionType.PROJECT:
      if in_shape != out_shape:
        input_tensor = tf.keras.layers.Dense(
            output_tensor.shape[-1], name='dense')(
                input_tensor)
      return input_tensor + output_tensor
    raise ValueError('Invalid ResidualConnectionType: {}'.format(
        self._residual_connection_type))

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    input_tensor = tf.keras.layers.Flatten(name='flatten')(input_tensor)
    net = input_tensor
    max_output_size = self._max_output_size
    if self._max_number_of_parameters:
      max_output_size = min(
          self._max_output_size,
          self._max_number_of_parameters // net.get_shape()[1])
    net = tf.keras.layers.Dense(
        max(min(max_output_size,
                net.get_shape()[1]), 2), name='dense')(
                    net)
    net = tf.nn.leaky_relu(net)
    if self._apply_batch_norm:
      net = tf.compat.v1.layers.batch_normalization(net, training=is_training)
    net = self._add_residual_connection(input_tensor, net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(
    lookup_name='LOWRANK_1', init_args={'kernel_rank': 1}, enum_id=130)
@register_block(
    lookup_name='LOWRANK_2', init_args={'kernel_rank': 2}, enum_id=131)
@register_block(
    lookup_name='LOWRANK_3', init_args={'kernel_rank': 3}, enum_id=132)
@register_block(
    lookup_name='LOWRANK_1_SKIP',
    init_args={
        'kernel_rank': 1,
        'skip_connect': True
    },
    enum_id=133)
@register_block(
    lookup_name='LOWRANK_2_SKIP',
    init_args={
        'kernel_rank': 2,
        'skip_connect': True
    },
    enum_id=134)
@register_block(
    lookup_name='LOWRANK_3_SKIP',
    init_args={
        'kernel_rank': 3,
        'skip_connect': True
    },
    enum_id=135)
class LowRankLayerBlock(Block):
  """Dense layer with a kernel that is low rank matrix.

  Dense layer with a kernel that is low rank matrix. A skip connection is
  optional.
  """

  def __init__(self, kernel_rank=1, skip_connect=False):
    self._kernel_rank = kernel_rank
    self._skip_connect = skip_connect

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf.keras.layers.Flatten(name='flatten')(input_tensor)
    net = tf.keras.layers.Dense(
        self._kernel_rank, name='dense_lowrank_left', use_bias=False)(
            net)
    net = tf.keras.layers.Dense(
        input_tensor.get_shape()[1], name='dense_lowrank_right', use_bias=True)(
            net)
    net = tf.nn.leaky_relu(net)
    if self._skip_connect:
      net += input_tensor
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(
    lookup_name='FIXED_OUTPUT_FULLY_CONNECTED_128',
    init_args={'output_size': 128},
    enum_id=22)
@register_block(
    lookup_name='FIXED_OUTPUT_FULLY_CONNECTED_256',
    init_args={'output_size': 256},
    enum_id=23)
@register_block(
    lookup_name='FIXED_OUTPUT_FULLY_CONNECTED_512',
    init_args={'output_size': 512},
    enum_id=24)
@register_block(
    lookup_name='FIXED_OUTPUT_FULLY_CONNECTED_1024',
    init_args={'output_size': 1024},
    enum_id=25)
class FixedOutputFullyConnectedBlock(Block):
  """A fully connected layer with leaky relu activation.

    Output number of hidden nodes is fixed and equal to output_size.
  """

  def __init__(self, output_size=100, relu_alpha=0.2):
    self._output_size = output_size
    self._relu_alpha = relu_alpha

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf.keras.layers.Flatten(name='flatten')(input_tensor)
    net = tf.keras.layers.Dense(self._output_size, name='dense')(net)
    net = tf.nn.leaky_relu(net, alpha=self._relu_alpha)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(
    lookup_name='BOTTLENECK_32', init_args={'projection_size': 32}, enum_id=126)
@register_block(
    lookup_name='BOTTLENECK_64', init_args={'projection_size': 64}, enum_id=127)
@register_block(
    lookup_name='BOTTLENECK_32_SKIP',
    init_args={
        'projection_size': 32,
        'skip_connect': True
    },
    enum_id=128)
@register_block(
    lookup_name='BOTTLENECK_64_SKIP',
    init_args={
        'projection_size': 64,
        'skip_connect': True
    },
    enum_id=129)
class BottleNeckBlock(Block):
  """A bottle-neck layer.

  This layer consists a projection to lower dimension followed by a projection
  back to the original dimension.
  A skip connection is optional.
  """

  def __init__(self, projection_size=32, skip_connect=False):
    self._projection_size = projection_size
    self._skip_connect = skip_connect

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf.keras.layers.Flatten(name='flatten')(input_tensor)
    net = tf.keras.layers.Dense(self._projection_size, name='lower_dim')(net)
    net = tf.nn.leaky_relu(net)
    net = tf.keras.layers.Dense(
        input_tensor.get_shape()[1], name='expand_dim')(
            net)
    net = tf.nn.leaky_relu(net)
    if self._skip_connect:
      net += input_tensor
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(lookup_name='IDENTITY', init_args={}, enum_id=125)
class IdentityBlock(Block):
  """An empty block for when using baysian opt. search algorithm."""

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    return input_tensors

  @property
  def is_input_order_important(self):
    return False


@register_block(lookup_name='FULLY_CONNECTED_PYRAMID', init_args={}, enum_id=15)
class FullyConnectedPyramidBlock(Block):
  """A fully connected layer with leaky relu.

    Output number of hidden nodes is equal to the input number of hidden nodes
    divided by 2, with some restrictions:
    - output number of hidden nodes is at least 2.
    - output number of hidden nodes is at most max_output_size.
    - The number of elements in the kernel matrix is at most:
      max_number_of_parameters.
  """

  def __init__(self, max_output_size=100, max_number_of_parameters=None):
    # force the kernel matrix to have less than max_number_of_parameters
    # entries.
    self._max_number_of_parameters = max_number_of_parameters
    self._max_output_size = max_output_size

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf.keras.layers.Flatten(name='flatten')(input_tensor)
    max_output_size = self._max_output_size
    if self._max_number_of_parameters:
      max_output_size = min(
          self._max_output_size,
          self._max_number_of_parameters // net.get_shape()[1])
    net = tf.keras.layers.Dense(
        min(max_output_size, max(net.get_shape()[1] // 2, 2)), name='dense')(
            net)
    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(
    lookup_name='MAX_POOLING_3X3', init_args={'pool_size': 3}, enum_id=17)
@register_block(
    lookup_name='MAX_POOLING_5X5', init_args={'pool_size': 5}, enum_id=18)
class MaxPoolingBlock(Block):
  """Max Pooling block."""

  def __init__(self, pool_size=2, strides=(1, 1)):
    self._pool_size = pool_size
    self._strides = strides

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Applies 2d max pooling on the input tensor."""
    input_tensor = input_tensors[-1]
    if input_tensor.get_shape().as_list()[2] < self._pool_size:
      return input_tensors

    max_pool = tf_slim.max_pool2d(
        input_tensor,
        self._pool_size,
        stride=self._strides,
        padding='same',
    )
    return input_tensors + [max_pool]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='DILATED_CONVOLUTION_2',
    init_args={'dilation_rate': 2},
    enum_id=19)
@register_block(
    lookup_name='DILATED_CONVOLUTION_4',
    init_args={'dilation_rate': 4},
    enum_id=20)
class DilatedConvolutionBlock(Block):
  """Dilated convolution block.

  Dilated convolution to get higher coverage when performing cnn in terms of
  pixels (if you want to catch global context in high resolution picture for
  example).
  """

  def __init__(self,
               kernel_size=3,
               dilation_rate=(2, 2),
               apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._dilation_rate = dilation_rate
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same',
        rate=self._dilation_rate)

    # Batch norm
    if self._apply_batch_norm:
      net = tf_slim.batch_norm(net, is_training=is_training)

    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(lookup_name='FLATTEN', init_args={}, enum_id=16)
class FlattenBlock(Block):
  """Flattens the input."""

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    return input_tensors + [
        tf.keras.layers.Flatten(name='flatten')(input_tensor)
    ]

  @property
  def is_input_order_important(self):
    return False


@register_block(lookup_name='DOWNSAMPLE_FLATTEN', init_args={}, enum_id=21)
class DownsampleFlattenBlock(Block):
  """Flattens the input by downsampling till the plate is 1x1."""

  def __init__(self, kernel_size=2, strides=(2, 2), max_channels=1000):
    self._kernel_size = kernel_size
    self._strides = strides
    self._max_channels = max_channels

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds a cnn tower with flatten at the top.

    Args:
      input_tensors: A list of rank four tf.Tensors with the input.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: hparams for the build.

    Returns:
      A cnn tower, where each cnn reduce the plate (with strides) and
      increase the number of channels by two. leaky_relu is applied between
      cnn blocks and the number of channels is limitied to max_channels
    """
    input_tensor = input_tensors[-1]
    net = input_tensor
    while True:
      plate_dimension = net.get_shape()[2]
      if plate_dimension < self._kernel_size:
        break

      net = tf_slim.conv2d(
          net,
          min(get_channel_dim(net) * 2, self._max_channels),
          kernel_size=self._kernel_size,
          stride=self._strides,
          padding='same')
      net = tf.nn.leaky_relu(net)

    net = tf.keras.layers.Flatten(name='flatten')(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return False


@register_block(lookup_name='DUAL_RESNET', init_args={}, enum_id=35)
class DualResnetBlock(Block):
  """Residual unit with 2 sub layers."""

  def __init__(self, kernel_size=3):
    self._kernel_size = kernel_size

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Returns a ReLU activated output of a residual unit with 2 sub layers."""
    input_tensor = input_tensors[-1]
    net1 = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    net = tf.nn.leaky_relu(net1)

    net1 = tf_slim.conv2d(
        net,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    net2 = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    net = tf.nn.leaky_relu(net2)

    net2 = tf_slim.conv2d(
        net,
        get_channel_dim(input_tensor),
        kernel_size=self._kernel_size,
        padding='same')

    net1 /= 2
    net2 /= 2
    input_tensor += net1
    input_tensor += net2

    net = tf.nn.leaky_relu(input_tensor)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(lookup_name='GENERAL_BLOCK', init_args={}, enum_id=32)
class GeneralBlock(Block):
  """A general block; This block is custom made."""

  def __init__(self, kernel_size=3, apply_batch_norm=True):
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Custom (wide) convolution block with some pooling."""
    # Guard so that we won't have zero channels
    input_tensor = input_tensors[-1]
    if get_channel_dim(input_tensor) < 6:
      return input_tensors

    reduced = tf_slim.conv2d(
        input_tensor,
        get_channel_dim(input_tensor) // 5,
        kernel_size=1,
        padding='same')

    reduced_with_activation = tf.nn.leaky_relu(reduced)

    # Batch norm
    if self._apply_batch_norm:
      reduced_with_activation = tf_slim.batch_norm(
          reduced_with_activation, is_training=is_training)

    convo1 = tf_slim.conv2d(
        reduced_with_activation,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    convo1 = tf.nn.leaky_relu(convo1)

    convo1 = tf_slim.conv2d(
        convo1,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    convo2 = tf_slim.conv2d(
        reduced_with_activation,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    convo2 = tf.nn.leaky_relu(convo2)

    convo2 = tf_slim.conv2d(
        convo2,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    avg_pool = tf.keras.layers.AveragePooling2D(
        strides=1,
        pool_size=self._kernel_size,
        padding='same',
        name='avgpool2d')(
            reduced_with_activation)

    avg_pool = tf_slim.conv2d(
        avg_pool,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    max_pool = tf_slim.max_pool2d(
        reduced_with_activation,
        self._kernel_size,
        stride=(1, 1),
        padding='same')

    max_pool = tf_slim.conv2d(
        max_pool,
        get_channel_dim(input_tensor) // 5,
        kernel_size=self._kernel_size,
        padding='same')

    net = tf.concat([convo1, convo2, avg_pool, max_pool, reduced], axis=3)
    net = tf.nn.leaky_relu(net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(lookup_name='PLATE_REDUCTION_FLATTEN', init_args={}, enum_id=34)
class PlateReductionFlatten(Block):
  """Mean of plate."""

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    if len(input_tensors[-1].get_shape().as_list()) == 2:
      return input_tensors
    return input_tensors + [
        tf.reduce_mean(input_tensor=input_tensors[-1], axis=[1, 2])
    ]

  @property
  def is_input_order_important(self):
    return False


def _add_projection_if_needed(inputs, outputs):
  if inputs.get_shape().as_list()[-1] == outputs.get_shape().as_list()[-1]:
    return inputs
  else:
    return tf.keras.layers.Dense(
        outputs.get_shape().as_list()[-1], name='dense')(
            inputs)


@register_block(
    lookup_name='RNN_CELL_64', init_args={'output_size': 64}, enum_id=60)
@register_block(
    lookup_name='RNN_CELL_128', init_args={'output_size': 128}, enum_id=36)
@register_block(
    lookup_name='RNN_CELL_256', init_args={'output_size': 256}, enum_id=37)
@register_block(
    lookup_name='RNN_CELL_512', init_args={'output_size': 512}, enum_id=38)
@register_block(
    lookup_name='RNN_CELL_1024', init_args={'output_size': 1024}, enum_id=39)
@register_block(
    lookup_name='SKIP_RNN_CELL_64',
    init_args={
        'output_size': 64,
        'skip': True
    },
    enum_id=110)
@register_block(
    lookup_name='SKIP_RNN_CELL_128',
    init_args={
        'output_size': 128,
        'skip': True
    },
    enum_id=111)
@register_block(
    lookup_name='SKIP_RNN_CELL_256',
    init_args={
        'output_size': 256,
        'skip': True
    },
    enum_id=124)
@register_block(
    lookup_name='SKIP_RNN_CELL_512',
    init_args={
        'output_size': 512,
        'skip': True
    },
    enum_id=112)
@register_block(
    lookup_name='SKIP_RNN_CELL_1024',
    init_args={
        'output_size': 1024,
        'skip': True
    },
    enum_id=113)
class RnnBlock(Block):
  """A basic rnn cell."""

  def __init__(self, output_size=100, skip=False):
    self._output_size = output_size
    self._skip = skip

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds a basic rnn block.

    Args:
      input_tensors: A tf.Tensor with the input.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: hparams for the build.

    Returns:
      output tensor
    """
    input_tensor = input_tensors[-1]
    rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(
        num_units=self._output_size, activation=tf.nn.tanh)
    net, _ = tf.compat.v1.nn.dynamic_rnn(
        rnn_cell, input_tensor, sequence_length=lengths, dtype=tf.float32)

    if self._skip:
      net += _add_projection_if_needed(input_tensor, net)

    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='RECURRENT_DENSE_64', init_args={'output_size': 64}, enum_id=61)
@register_block(
    lookup_name='RECURRENT_DENSE_128',
    init_args={'output_size': 128},
    enum_id=40)
@register_block(
    lookup_name='RECURRENT_DENSE_256',
    init_args={'output_size': 256},
    enum_id=41)
@register_block(
    lookup_name='RECURRENT_DENSE_512',
    init_args={'output_size': 512},
    enum_id=42)
@register_block(
    lookup_name='RECURRENT_DENSE_1024',
    init_args={'output_size': 1024},
    enum_id=43)
@register_block(
    lookup_name='CONV1D_128_3',
    init_args={
        'output_size': 128,
        'kernel_size': 3
    },
    enum_id=89)
@register_block(
    lookup_name='CONV1D_128_5',
    init_args={
        'output_size': 128,
        'kernel_size': 5
    },
    enum_id=90)
@register_block(
    lookup_name='SKIP_CONV1D_128_3',
    init_args={
        'output_size': 128,
        'kernel_size': 3,
        'skip': True
    },
    enum_id=114)
@register_block(
    lookup_name='SKIP_CONV1D_128_5',
    init_args={
        'output_size': 128,
        'kernel_size': 5,
        'skip': True
    },
    enum_id=115)
class Conv1DBlock(Block):
  """A dense block for recurrent input."""

  def __init__(self,
               output_size=50,
               kernel_size=1,
               activation=None,
               skip=False,
               dilation_rate=1):
    assert kernel_size > 0
    self._output_size = output_size
    self._kernel_size = kernel_size
    self._activation = activation
    self._skip = skip
    self._dilation_rate = dilation_rate

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds a one dimensional convolutional block.

    Args:
      input_tensors: A tf.Tensor with the input.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: hparams for the build.

    Returns:
      output tensor
    """
    if not self._activation and self._kernel_size > 1:
      self._activation = tf.compat.v1.nn.relu

    input_tensor = input_tensors[-1]
    net = tf.compat.v1.layers.conv1d(
        inputs=input_tensor,
        filters=self._output_size,
        kernel_size=self._kernel_size,
        activation=self._activation,
        dilation_rate=self._dilation_rate,
        padding='same')

    if self._skip:
      net += _add_projection_if_needed(input_tensor, net)

    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


class SvdfImplementationFashion(enum.IntEnum):
  """Specifies the svdf implementation fashion."""
  # RNN Cell implementation for SVDF, works best for streaming features.
  SVDF_RNN_CELL = 1
  # CNN implementation for SVDF, works best when all features are available.
  SVDF_CONV = 2


@register_block(
    lookup_name='SVDF_CELL_64_4_1',
    init_args={
        'output_size': 64,
        'memory_size': 4,
        'rank': 1
    },
    enum_id=62)
@register_block(
    lookup_name='SVDF_CELL_128_4_1',
    init_args={
        'output_size': 128,
        'memory_size': 4,
        'rank': 1
    },
    enum_id=44)
@register_block(
    lookup_name='SVDF_CELL_256_4_1',
    init_args={
        'output_size': 256,
        'memory_size': 4,
        'rank': 1
    },
    enum_id=45)
@register_block(
    lookup_name='SVDF_CELL_512_4_1',
    init_args={
        'output_size': 512,
        'memory_size': 4,
        'rank': 1
    },
    enum_id=46)
@register_block(
    lookup_name='SVDF_CELL_1024_4_1',
    init_args={
        'output_size': 1024,
        'memory_size': 4,
        'rank': 1
    },
    enum_id=47)
@register_block(
    lookup_name='SVDF_CELL_64_8_1',
    init_args={
        'output_size': 64,
        'memory_size': 8,
        'rank': 1
    },
    enum_id=63)
@register_block(
    lookup_name='SVDF_CELL_128_8_1',
    init_args={
        'output_size': 128,
        'memory_size': 8,
        'rank': 1
    },
    enum_id=48)
@register_block(
    lookup_name='SVDF_CELL_256_8_1',
    init_args={
        'output_size': 256,
        'memory_size': 8,
        'rank': 1
    },
    enum_id=49)
@register_block(
    lookup_name='SVDF_CELL_512_8_1',
    init_args={
        'output_size': 512,
        'memory_size': 8,
        'rank': 1
    },
    enum_id=50)
@register_block(
    lookup_name='SVDF_CELL_1024_8_1',
    init_args={
        'output_size': 1024,
        'memory_size': 8,
        'rank': 1
    },
    enum_id=51)
@register_block(
    lookup_name='SVDF_CELL_64_16_1',
    init_args={
        'output_size': 64,
        'memory_size': 16,
        'rank': 1
    },
    enum_id=64)
@register_block(
    lookup_name='SVDF_CELL_128_16_1',
    init_args={
        'output_size': 128,
        'memory_size': 16,
        'rank': 1
    },
    enum_id=52)
@register_block(
    lookup_name='SVDF_CELL_256_16_1',
    init_args={
        'output_size': 256,
        'memory_size': 16,
        'rank': 1
    },
    enum_id=53)
@register_block(
    lookup_name='SVDF_CELL_512_16_1',
    init_args={
        'output_size': 512,
        'memory_size': 16,
        'rank': 1
    },
    enum_id=54)
@register_block(
    lookup_name='SVDF_CELL_1024_16_1',
    init_args={
        'output_size': 1024,
        'memory_size': 16,
        'rank': 1
    },
    enum_id=55)
@register_block(
    lookup_name='SVDF_CELL_32_32_1',
    init_args={
        'output_size': 32,
        'memory_size': 32,
        'rank': 1
    },
    enum_id=75)
@register_block(
    lookup_name='SVDF_CELL_576_8_1_64',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64
    },
    enum_id=137)
@register_block(
    lookup_name='SVDF_CELL_576_8_1_16',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 16
    },
    enum_id=138)
@register_block(
    lookup_name='SVDF_CELL_576_8_1_7',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 7
    },
    enum_id=139)
@register_block(
    lookup_name='SVDF_CELL_32_32_1_2',
    init_args={
        'output_size': 32,
        'memory_size': 32,
        'rank': 1,
        'projection_size': 2
    },
    enum_id=140)
@register_block(
    lookup_name='SVDF_CONV_576_8_1_64',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=141)
@register_block(
    lookup_name='SVDF_CONV_576_8_1_16',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 16,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=142)
@register_block(
    lookup_name='SVDF_CONV_576_8_1_7',
    init_args={
        'output_size': 576,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 7,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=143)
@register_block(
    lookup_name='SVDF_CONV_32_32_1',
    init_args={
        'output_size': 32,
        'memory_size': 32,
        'rank': 1,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=144)
@register_block(
    lookup_name='SVDF_CONV_32_32_1_2',
    init_args={
        'output_size': 32,
        'memory_size': 32,
        'rank': 1,
        'projection_size': 2,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=145)
@register_block(
    lookup_name='SVDF_CONV_64_4_1_64',
    init_args={
        'output_size': 64,
        'memory_size': 4,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10001)
@register_block(
    lookup_name='SVDF_CONV_128_4_1_64',
    init_args={
        'output_size': 128,
        'memory_size': 4,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10002)
@register_block(
    lookup_name='SVDF_CONV_256_4_1_64',
    init_args={
        'output_size': 256,
        'memory_size': 4,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10003)
@register_block(
    lookup_name='SVDF_CONV_512_4_1_64',
    init_args={
        'output_size': 512,
        'memory_size': 4,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10004)
@register_block(
    lookup_name='SVDF_CONV_1024_4_1_64',
    init_args={
        'output_size': 1024,
        'memory_size': 4,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10005)
@register_block(
    lookup_name='SVDF_CONV_64_8_1_64',
    init_args={
        'output_size': 64,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10006)
@register_block(
    lookup_name='SVDF_CONV_128_8_1_64',
    init_args={
        'output_size': 128,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10007)
@register_block(
    lookup_name='SVDF_CONV_256_8_1_64',
    init_args={
        'output_size': 256,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10008)
@register_block(
    lookup_name='SVDF_CONV_512_8_1_64',
    init_args={
        'output_size': 512,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10009)
@register_block(
    lookup_name='SVDF_CONV_1024_8_1_64',
    init_args={
        'output_size': 1024,
        'memory_size': 8,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10010)
@register_block(
    lookup_name='SVDF_CONV_64_16_1_64',
    init_args={
        'output_size': 64,
        'memory_size': 16,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10011)
@register_block(
    lookup_name='SVDF_CONV_128_16_1_64',
    init_args={
        'output_size': 128,
        'memory_size': 16,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10012)
@register_block(
    lookup_name='SVDF_CONV_256_16_1_64',
    init_args={
        'output_size': 256,
        'memory_size': 16,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10013)
@register_block(
    lookup_name='SVDF_CONV_512_16_1_64',
    init_args={
        'output_size': 512,
        'memory_size': 16,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10014)
@register_block(
    lookup_name='SVDF_CONV_1024_16_1_64',
    init_args={
        'output_size': 1024,
        'memory_size': 16,
        'rank': 1,
        'projection_size': 64,
        'fashion': SvdfImplementationFashion.SVDF_CONV,
    },
    enum_id=10015)
class SvdfBlock(Block):
  """Creates and SVDF layer.

  An SVDF block - Recurrent cell or convolutional layer with low rank matrix
  update. For more details: https://research.google.com/pubs/pub43813.html.
  """

  def __init__(self,
               output_size=100,
               memory_size=10,
               rank=1,
               projection_size=0,
               fashion=SvdfImplementationFashion.SVDF_RNN_CELL):
    self._output_size = output_size
    self._memory_size = memory_size
    self._projection_size = projection_size
    self._rank = rank
    self._fashion = fashion

  def _get_svdf_rnn_cell_output(self, input_tensor, lengths):
    svdf = svdf_cell.SvdfCell(
        num_units=self._output_size,
        memory_size=self._memory_size,
        rank=self._rank,
        activation=tf.nn.relu)
    net, _ = tf.compat.v1.nn.dynamic_rnn(
        svdf, input_tensor, sequence_length=lengths, dtype=tf.float32)
    return net

  def _get_svdf_conv_output(self, input_tensor):
    svdf_conv_layer = svdf_conv.SvdfConvLayer(
        units=self._output_size,
        memory_size=self._memory_size,
        rank=self._rank,
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.16, 0.16),
        name='svdf_conv_layer',
    )
    layer_output = svdf_conv_layer(input_tensor)
    return layer_output

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    if self._fashion == SvdfImplementationFashion.SVDF_CONV:
      net = self._get_svdf_conv_output(input_tensor)
    elif self._fashion == SvdfImplementationFashion.SVDF_RNN_CELL:
      net = self._get_svdf_rnn_cell_output(input_tensor, lengths)
    else:
      raise ValueError('Invalid SVDF Implementation fashion %s' %
                       self._fashion.name)

    if self._projection_size:
      net = tf.keras.layers.Dense(
          self._projection_size, activation=None, name='svdf_projection')(
              net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(lookup_name='TUNABLE_SVDF', init_args={}, enum_id=10016)
class TunableSvdfBlock(Block):
  """Creates an SVDF layer.

  An SVDF block - Recurrent cell or convolutional layer with low rank matrix
  update. For more details: https://research.google.com/pubs/pub43813.html.
  """

  def _get_svdf_conv_output(self, input_tensor, hparams):
    svdf_conv_layer = svdf_conv.SvdfConvLayer(
        units=hparams.output_size,
        memory_size=hparams.memory_size,
        rank=hparams.rank,
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.16, 0.16),
        name='svdf_conv_layer',
    )
    layer_output = svdf_conv_layer(input_tensor)
    return layer_output

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    input_tensor = input_tensors[-1]
    net = self._get_svdf_conv_output(input_tensor, hparams)

    if hparams.projection_size:
      net = tf.keras.layers.Dense(
          hparams.projection_size, activation=None, name='svdf_projection')(
              net)
    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True

  def requires_hparams(self):
    hps = kerastuner.engine.hyperparameters.HyperParameters()
    hps.Int('output_size', 50, 100)
    hps.Int('memory_size', 4, 32)
    hps.Int('rank', 1, 3)
    hps.Int('projection_size', 0, 128)
    return hps


@register_block(
    lookup_name='LSTM_128', init_args={
        'output_size': 128,
    }, enum_id=56)
@register_block(
    lookup_name='LSTM_256', init_args={
        'output_size': 256,
    }, enum_id=57)
@register_block(
    lookup_name='LSTM_512', init_args={
        'output_size': 512,
    }, enum_id=58)
@register_block(
    lookup_name='LSTM_1024', init_args={
        'output_size': 1024,
    }, enum_id=59)
@register_block(
    lookup_name='SKIP_LSTM_128',
    init_args={
        'output_size': 128,
        'skip': True
    },
    enum_id=116)
@register_block(
    lookup_name='SKIP_LSTM_256',
    init_args={
        'output_size': 256,
        'skip': True
    },
    enum_id=117)
@register_block(
    lookup_name='SKIP_LSTM_512',
    init_args={
        'output_size': 512,
        'skip': True
    },
    enum_id=118)
@register_block(
    lookup_name='SKIP_LSTM_1024',
    init_args={
        'output_size': 1024,
        'skip': True
    },
    enum_id=119)
class LSTMBlock(Block):
  """An LSTM block for sequential input."""

  def __init__(self, output_size=100, skip=False):
    self._output_size = output_size
    self._skip = skip

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds as LSTM block.

    Args:
      input_tensors: A list of tf.Tensors with the input.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: hparams for the build.

    Returns:
      output tensor
    """
    input_tensor = input_tensors[-1]
    rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
        num_units=self._output_size, activation=tf.nn.tanh)
    net, _ = tf.compat.v1.nn.dynamic_rnn(
        rnn_cell, input_tensor, sequence_length=lengths, dtype=tf.float32)

    if self._skip:
      net += _add_projection_if_needed(input_tensor, net)

    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


@register_block(
    lookup_name='BI_LSTM_128', init_args={
        'output_size': 128,
    }, enum_id=85)
@register_block(
    lookup_name='BI_LSTM_256', init_args={
        'output_size': 256,
    }, enum_id=86)
@register_block(
    lookup_name='BI_LSTM_512', init_args={
        'output_size': 512,
    }, enum_id=87)
@register_block(
    lookup_name='BI_LSTM_1024', init_args={
        'output_size': 1024,
    }, enum_id=88)
@register_block(
    lookup_name='SKIP_BI_LSTM_128',
    init_args={
        'output_size': 128,
        'skip': True
    },
    enum_id=120)
@register_block(
    lookup_name='SKIP_BI_LSTM_256',
    init_args={
        'output_size': 256,
        'skip': True
    },
    enum_id=121)
@register_block(
    lookup_name='SKIP_BI_LSTM_512',
    init_args={
        'output_size': 512,
        'skip': True
    },
    enum_id=122)
@register_block(
    lookup_name='SKIP_BI_LSTM_1024',
    init_args={
        'output_size': 1024,
        'skip': True
    },
    enum_id=123)
class BidirectionalLSTMBlock(Block):
  """An Bidirectional LSTM block for sequential input."""

  def __init__(self, output_size=100, skip=False):
    self._output_size = output_size
    self._skip = skip

  def build(self, input_tensors, is_training, lengths=None, hparams=None):
    """Builds as LSTM block.

    Args:
      input_tensors: A list of tf.Tensors with the input.
      is_training: Whether we are training. Used for regularization.
      lengths: The lengths of the input sequences in the batch.
      hparams: hparams for the build.

    Returns:
      output tensor
    """
    input_tensor = input_tensors[-1]
    fw_rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
        num_units=self._output_size, activation=tf.nn.tanh)
    bw_rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
        num_units=self._output_size, activation=tf.nn.tanh)
    (out1, out2), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        fw_rnn_cell,
        bw_rnn_cell,
        input_tensor,
        sequence_length=lengths,
        dtype=tf.float32)

    net = tf.concat([out1, out2], axis=2)
    if self._skip:
      net += _add_projection_if_needed(input_tensor, net)

    return input_tensors + [net]

  @property
  def is_input_order_important(self):
    return True


tf_arg_scope = tf_slim.arg_scope
