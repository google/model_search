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
"""Module containing util functions."""

from model_search.proto import phoenix_spec_pb2

import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_lib


def default_get_input_layer_fn(problem_type, feature_columns):
  """Default implementation of get_input_layer_fn."""

  def _input_layer_fn(features,
                      is_training,
                      scope_name="Phoenix/Input",
                      lengths_feature_name=None):

    with tf.compat.v1.variable_scope(scope_name):
      if problem_type == phoenix_spec_pb2.PhoenixSpec.CNN:
        # Sometimes we only get the image feature as a tensor.
        if not isinstance(features, dict):
          return features, None
        return tf.cast(
            features[feature_columns[0].name], dtype=tf.float32), None
      # DNN
      elif problem_type == phoenix_spec_pb2.PhoenixSpec.DNN:
        # To allow running a custom evaluation where multiple batches are
        # aggregated in a single metric_fn call, we need to define the
        # batch_size based on the input_fn, but DenseFeatures does not allow
        # this.
        if (len(feature_columns) == 1 and isinstance(
            feature_columns[0], type(tf.feature_column.numeric_column("x")))):
          return tf.cast(
              features[feature_columns[0].name], dtype=tf.float32), None
        # All are TF1 feature columns
        elif all([
            not feature_column_lib.is_feature_column_v2([fc])
            for fc in feature_columns
        ]):
          return tf.compat.v1.feature_column.input_layer(
              features, feature_columns, trainable=is_training), None
        # Some are TF1 feature columns
        elif any([
            not feature_column_lib.is_feature_column_v2([fc])
            for fc in feature_columns
        ]):
          fc_v1 = [
              fc for fc in feature_columns
              if not feature_column_lib.is_feature_column_v2([fc])
          ]
          fc_v2 = [
              fc for fc in feature_columns
              if feature_column_lib.is_feature_column_v2([fc])
          ]
          input_1 = tf.compat.v1.feature_column.input_layer(
              features, fc_v1, trainable=is_training)
          input_2 = tf.keras.layers.DenseFeatures(
              fc_v2, name="input_layer_fc_v2", trainable=is_training)(
                  features)
          return tf.concat([input_1, input_2], axis=1), None

        # None is TF1 feature columns
        else:
          return tf.keras.layers.DenseFeatures(
              feature_columns, name="input_layer",
              trainable=is_training)(features), None

      # RNN
      elif (problem_type == phoenix_spec_pb2.PhoenixSpec.RNN_ALL_ACTIVATIONS or
            problem_type == phoenix_spec_pb2.PhoenixSpec.RNN_LAST_ACTIVATIONS):
        if lengths_feature_name:
          return (tf.cast(features[feature_columns[0].name],
                          dtype=tf.float32), features[lengths_feature_name])
        elif (
            feature_columns[0].name in features and
            not isinstance(features[feature_columns[0].name], tf.SparseTensor)):
          return tf.cast(
              features[feature_columns[0].name], dtype=tf.float32), None
        else:
          # IMPORTANT NOTE:
          # When you use Keras layers with variables, always give them a name!
          # If not, keras will add "_#" (e.g., dense_1 instead of dense).
          # It will add the suffix even if the outer-scope is different.
          # This is a surprising behavior.
          # TODO(mazzawi): Contact the Keras team about this.
          return tf.keras.experimental.SequenceFeatures(
              feature_columns=feature_columns,
              trainable=is_training,
              name=scope_name)(
                  features)
      else:
        raise ValueError("Unknown problem type")

  return _input_layer_fn
