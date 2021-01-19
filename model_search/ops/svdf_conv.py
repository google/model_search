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
"""SVDF Conv Layer Op definition.

"""

from typing import Any, Callable, Dict, List, Optional, Text, Union  # pylint: disable=g-bad-import-order

from absl import logging
from six.moves import range
import tensorflow.compat.v1 as tf

CallOrText = Union[Text, Callable]


def _generate_dropout_mask(ones: tf.Tensor,
                           rate: float,
                           training: Optional[bool] = None,
                           count: int = 1) -> tf.Tensor:
  """Returns a mask for implementing dropout."""

  def dropped_inputs():
    return tf.keras.backend.dropout(ones, rate)

  if count > 1:
    return [
        tf.keras.backend.in_train_phase(
            dropped_inputs, ones, training=training) for _ in range(count)
    ]
  return tf.keras.backend.in_train_phase(
      dropped_inputs, ones, training=training)


class SvdfConvLayer(tf.keras.layers.Layer):
  """Svdf implementations with conv layers.

  This is an equalivent implementation to class SvdfCell in svdf_cell.py.
  The difference is, training with SvdfConvLayer implementation is much faster,
  but SvdfCell is better during inference time as it is implemented in streaming
  way.

  # Attributes:
    units: Positive integer, dimensionality of the output space.
    memory_size: Dimension of feature vector.
    rank: int or long, the rank of the SVD approximation.
    activation: Activation function to use
      (see [activations](../activations.md)).
      Default: Rectified linear unit (`relu`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs
      (see [initializers](../initializers.md)).
    bias_initializer: Initializer for the bias vector
      (see [initializers](../initializers.md)).
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    use_batch_norm: whether to apply batch normalization
    bn_scale: whether to apply batch norm scaling. When the next layer is linear
      (e.g. nn.relu), this can be disabled since the scaling will be done by the
      next layer.
  """

  def __init__(self,
               units: int,
               memory_size: int = 0,
               rank: int = 1,
               activation: Text = "relu",
               use_bias: bool = True,
               kernel_initializer: CallOrText = "glorot_uniform",
               bias_initializer: CallOrText = "zeros",
               dropout: float = 0.,
               use_batch_norm: bool = False,
               bn_scale: bool = False,
               name: Optional[str] = None,
               **kwargs):
    if memory_size <= 1:
      raise ValueError("memory_size must be > 1")
    if rank <= 0:
      raise ValueError("rank must be > 0")

    self.units = units
    self.memory_size = memory_size
    self.rank = rank
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.use_batch_norm = use_batch_norm
    self.bn_scale = bn_scale
    self.dropout = min(1., max(0., dropout))
    self.num_filters = self.rank * self.units
    self._dropout_mask = None
    self.feature_kernel = []
    self.time_kernel = []
    self.bias = []

    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]
    self.output_size = self.units

    self._name = name or type(self).__name__
    super(SvdfConvLayer, self).__init__(name=self._name, **kwargs)

  def build(self, input_shape: List[int]):
    """Implements build interface for tf.keras.layers.Layer."""
    # Sub-classes should check for input_shape and then call super.build.
    self.num_features = input_shape[-1]
    self.feature_kernel = self.add_weight(
        shape=(self.num_features, self.num_filters),
        name="feature_kernel",
        initializer=self.kernel_initializer)
    self.time_kernel = self.add_weight(
        shape=(self.memory_size, self.num_filters),
        name="time_kernel",
        initializer=self.kernel_initializer)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,), name="bias", initializer=self.bias_initializer)
    else:
      self.bias = None
    # Sets self.built = True
    super(SvdfConvLayer, self).build(input_shape)

  def get_config(self) -> Dict[Text, Any]:
    """Configs of the model hparams for logging and debugging purposes."""
    config = {
        "units":
            self.units,
        "memory_size":
            self.memory_size,
        "rank":
            self.rank,
        "activation":
            tf.keras.activations.serialize(self.activation),
        "use_bias":
            self.use_bias,
        "kernel_initializer":
            tf.keras.initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self.bias_initializer),
        "dropout":
            self.dropout
    }
    base_config = super(SvdfConvLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
    """Implements call interface for tf.keras.layers.Layer."""
    # Handle drop out.
    if 0 < self.dropout < 1 and self._dropout_mask is not None:
      self._dropout_mask = _generate_dropout_mask(
          tf.keras.backend.ones_like(inputs), self.dropout, training=training)
    if self._dropout_mask is not None:
      inputs *= self._dropout_mask

    logging.info("Running regular SVDF conv layer with unit: %s and memory: %s",
                 self.units, self.memory_size)
    # svdf_output: [batch_size, time_steps, units]
    svdf_output, svdf_activations = self._run_svdf_conv_calculation(
        inputs, self.feature_kernel, self.time_kernel, self.memory_size,
        self.units, self.bias)

    if tf.executing_eagerly():
      logging.info("Starting inputs: %s", inputs)
      logging.info("svdf_activations 0: %s", svdf_activations)
      logging.info("svdf_output : %s", svdf_output)

    return svdf_output

  def _run_svdf_conv_calculation(self, inputs, feature_kernel, time_kernel,
                                 memory_size, units, cnn_bias):
    # padded_inputs: [batch_size, time_steps+mem_size-1, feature_size]
    padded_inputs = tf.keras.backend.temporal_padding(
        inputs, padding=(memory_size - 1, 0))
    # feature_conv_kernel: [1, feature_size, num_filters]
    feature_conv_kernel = tf.expand_dims(feature_kernel, 0)
    # svdf_feature_activations: [batch_size, time_steps+padsize, num_filters]
    svdf_feature_activations = tf.keras.backend.conv1d(
        padded_inputs, feature_conv_kernel, padding="valid")

    # time_conv_kernel: [memory_size, 1, num_filters, 1]
    time_conv_kernel = tf.expand_dims(tf.expand_dims(time_kernel, 1), -1)
    # svdf_time_activations: [batch_size, time_steps, 1, num_filters]
    svdf_time_activations = tf.nn.depthwise_conv2d(
        tf.expand_dims(svdf_feature_activations, -2),
        time_conv_kernel,
        strides=[1, 1, 1, 1],
        padding="VALID")
    # svdf_time_activations: [batch_size, time_steps, num_filters]
    svdf_time_activations = tf.keras.backend.squeeze(svdf_time_activations, -2)
    new_shape = [-1, tf.shape(svdf_time_activations)[1], units, self.rank]
    svdf_time_activations = tf.reshape(svdf_time_activations, new_shape)
    # time_activations_t: [batch_size, time_steps, units]
    svdf_activations = tf.keras.backend.sum(svdf_time_activations, axis=-1)

    if cnn_bias is not None:
      svdf_activations = tf.add(svdf_activations, cnn_bias)

    if self.use_batch_norm:
      svdf_activations = tf.keras.layers.BatchNormalization(
          scale=self.bn_scale)(
              svdf_activations)

    # svdf_output: [batch_size, time_steps, units]
    svdf_output = self.activation(svdf_activations)
    return svdf_output, svdf_activations
