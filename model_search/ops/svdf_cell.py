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
"""SVDF Cell Op definition.

This module defines a class that implements an SVDF Cell Op for RNNs.
The SVDF Op is a decomposition of a densely connected Op into
low rank filters.
For details: https://research.google.com/pubs/pub43813.html
Technically speaking the SVDF is not recurrent but we implement it under the
tensorflow tf.nn.rnn_cell.RNNCell framework in order to have automatic state
handling.
"""

import tensorflow.compat.v2 as tf



class SvdfCell(tf.compat.v1.nn.rnn_cell.RNNCell):
  """SVDF cell implementation.

  All variables are created and reused.
  """

  def __init__(self,
               num_units,
               memory_size=0,
               rank=0,
               use_bias=True,
               activation=None,
               feature_weights_initializer=None,
               time_weights_initializer=None,
               bias_initializer=tf.compat.v1.zeros_initializer(),
               trainable=True,
               name=None,
               image_summary=False):
    """Initializes SvdfCell.

    Arguments:
      num_units: int or long, the number of units in the layer.
      memory_size: int or long, the size of the memory (i.e. every new inference
        iteration we push a new memory entry, and remove the oldest one).
      rank: int or long, the rank of the SVD approximation.
      use_bias: bool, whether the layer uses a bias.
      activation: Callable, the activation function to use (use None for
        linear).
      feature_weights_initializer: (optional) Initializer function for the
        feature weight matrix.
      time_weights_initializer: (optional) Initializer function for the time
        weight matrix.
      bias_initializer: (optional) Initializer function for the bias.
      trainable: (optional) bool, whether variables created by this function
        should be added to the global list of trainable variables.
      name: (optional) String, the layer's name. It's used as default scope
        name, if none is provided when calling the cell.
      image_summary: (optional) bool, whether to log the filters (feature and
        time) as grayscale images.
    """
    self._name = name or type(self).__name__
    self._num_units = num_units
    self._memory_size = memory_size
    if self._memory_size <= 0:
      raise ValueError(
          "`SvdfCell:{}` should have a memory size greater than zero.".format(
              self._name))
    if rank <= 0:
      raise ValueError("rank must be > 0")
    self._rank = rank
    self._use_bias = use_bias
    self._activation = activation
    self._feature_weights_initializer = feature_weights_initializer
    self._time_weights_initializer = time_weights_initializer
    self._bias_initializer = bias_initializer
    self._trainable = trainable
    self._reuse_variables = False
    self._image_summary = image_summary

    self._state_size = num_units * (self._memory_size - 1) * rank

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """SVDF Cell computation.

    Arguments:
      inputs: 2D Tensor, where the first dimension could be used for batching
        purposes, and last dimension corresponds to the features. The size of
        this last dimension determines the size of the feature filters.
      state: The state as of the last inference in the input sequence. This
        should be a `2-D Tensor` with shape `[batch_size x state_size]`, where
        state_size = num_units * (self._memory_size - 1) * rank
      scope: A string or `tf.VariableScope` for this cell's ops and variables;
        otherwise it fallsback to the Op's name.

    Returns:
      Tuple (activations, new_state).
        activations: Tensor, with output activations, and shape corresponding
          to the input tensor (e.g. batch dimension), but with the last
          dimension corresponding to the units.
        new_state: The state after processing the current input.

    Raises:
      ValueError: If the inputs tensor is incorrectly shaped.

    """
    with tf.compat.v1.variable_scope(scope or self._name) as varscope:
      # Scoping.
      if self._reuse_variables:
        varscope.reuse_variables()
      else:
        self._reuse_variables = True
      input_shape = inputs.get_shape()
      # Validation.
      if input_shape.ndims is None:
        raise ValueError(
            "Inputs to `SvdfCell:{}` should have known rank.".format(
                self._name))
      if len(input_shape) != 2:
        raise ValueError(
            "Inputs to `SvdfCell:{}` should have rank == 2.".format(self._name))
      if input_shape[-1] is None:
        raise ValueError(
            "The last dimension of the inputs to `SvdfCell:{}` should be defined. Found `None`."
            .format(self._name))
      # Computation.
      feature_dim = inputs.shape[-1]
      # Expand to add input channels dimension.
      # rank --> [ batch_size, feature_dim, 1]
      inputs = tf.expand_dims(inputs, 2)
      # Number of filters: pairs of feature and time filters.
      num_filters = self._rank * self._num_units
      # Create the feature filters.
      if (self._feature_weights_initializer is not None and
          not callable(self._feature_weights_initializer)):
        # Because initializer is a constant, do not specify shape.
        weights_feature_shape = None
      else:
        # rank --> [ feature_dim, num_filters]
        weights_feature_shape = [feature_dim, num_filters]
      weights_feature = tf.compat.v1.get_variable(
          "SVDF_weights_feature",
          shape=weights_feature_shape,
          initializer=self._feature_weights_initializer,
          trainable=self._trainable)
      if self._image_summary:
        self._add_filter_image_summary(
            tf.identity(weights_feature), "weights_feature")
      # Expand to add input channels dimensions.
      #   weights_feature: [feature_dim, 1, num_filters]
      weights_feature = tf.expand_dims(weights_feature, 1)
      # Convolve the 1D feature filters sliding over the time dimension.
      # This is pretty much having a dense layer over the feature_dim
      # Not sure why they choose to put is as 1d conv.
      #   activations_time_t: [batch, 1, num_filters]
      activations_time_t = tf.nn.conv1d(
          input=inputs,
          filters=weights_feature,
          stride=feature_dim,
          padding="VALID")
      # Rearrange such that we can perform the batched matmul:
      #   activations_time_t: [num_filters, batch, 1]
      activations_time_t = tf.transpose(a=activations_time_t, perm=[2, 0, 1])

      # Prepare memory for the time filtering, receiving the previous:
      #   state: [batch, self._state_size]
      # We need to reshape the state into:
      #   memory: [batch, num_filters, self._memory_size - 1]
      memory = tf.reshape(state, [-1, num_filters, self._memory_size - 1])
      # We need to transpose the memory into:
      #   memory: [num_filters, batch, self._memory_size - 1]
      memory = tf.transpose(a=memory, perm=[1, 0, 2])
      # We need to insert (concat) activations_time_t to memory, such that:
      #   activations_time_t: [num_filters, batch, 1]
      #   memory: [num_filters, batch, self._memory_size]
      memory = tf.concat([memory, activations_time_t], 2)

      # Create the time filters.
      if (self._time_weights_initializer is not None and
          not callable(self._time_weights_initializer)):
        # Because initializer is a constant, do not specify shape.
        weights_time_shape = None
      else:
        weights_time_shape = [num_filters, self._memory_size]
      weights_time = tf.compat.v1.get_variable(
          "SVDF_weights_time",
          shape=weights_time_shape,
          initializer=self._time_weights_initializer,
          trainable=self._trainable)
      if self._image_summary:
        self._add_filter_image_summary(
            tf.transpose(a=weights_time), "weights_time")
      # Apply the time filter on the outputs of the feature filters.
      # weights_time: [num_filters, self._memory_size, 1]
      # outputs: [num_filters, batch, 1]
      weights_time = tf.expand_dims(weights_time, 2)
      outputs = tf.matmul(memory, weights_time)
      # Split num_units and rank into separate dimensions (the remaining
      # dimension is the input_shape[0] -i.e. batch size). This also squeezes
      # the last dimension, since it's not used.
      # [num_filters, batch, 1] => [num_units, rank, batch]
      outputs = tf.reshape(outputs, [self._num_units, self._rank, -1])
      # Sum the rank outputs per unit => [num_units, batch].
      units_output = tf.reduce_sum(
          input_tensor=outputs, axis=1, name="SVDF_rank_sum")
      # Transpose to shape [batch, num_units]
      units_output = tf.transpose(a=units_output)
      # Appy bias (if any).
      if self._use_bias:
        bias = tf.compat.v1.get_variable(
            "SVDF_bias",
            shape=[self._num_units],
            initializer=self._bias_initializer,
            trainable=self._trainable)
        units_output = tf.nn.bias_add(units_output, bias, name="SVDF_bias_add")
      # No activation is equivalent to "linear".
      if self._activation is not None:
        units_output = self._activation(units_output)

      # After all memory operations are done, we need to drop the memory slot
      # contained the oldest activations_time_t, such that:
      #   memory: [num_filters, batch, self._memory_size - 1]
      memory = memory[:, :, 1:]
      # We need to transpose back:
      #   memory: [batch, num_filters, self._memory_size - 1]
      memory = tf.transpose(a=memory, perm=[1, 0, 2])
      # We need to reshape back into the state:
      #   new_state: [batch, self._state_size]
      new_state = tf.reshape(memory, [-1, self._state_size])

    return units_output, new_state

  def _add_filter_image_summary(self, filters, name):
    """Adds image summaries for the given filter (an image per rank).

    Arguments:
      filters: A Tensor containing the filters. Expected shape: [rank *
        num_units, filter_dim]. Thus, the tensor groups the rank filters for
        each unit, and the function will split them to generate an image per
        rank. Each image will have the shape: [filter_dim, num_units]
      name: The string name to use as parth of the image log path.
    """
    for r in range(self._rank):
      rank_filters = tf.strided_slice(filters, [r, 0], filters.get_shape(),
                                      [self._rank, 1])
      rank_filters = tf.expand_dims(rank_filters, 0)
      rank_filters = tf.expand_dims(rank_filters, 3)
      tf.compat.v1.summary.image(
          "%s/rank_%d" % (name, (r + 1)), rank_filters, max_outputs=1)



