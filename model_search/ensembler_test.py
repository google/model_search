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

# Lint as: python2, python3
"""Tests for model_search.ensembler."""

from absl import logging
from absl.testing import parameterized

from model_search import ensembler
from model_search.architecture import architecture_utils
from model_search.proto import ensembling_spec_pb2
from model_search.proto import phoenix_spec_pb2
import tensorflow.compat.v2 as tf

_NONADAPTIVE_GRAPH_NODES_PRIORS = [
    'Phoenix/Ensembler/AddN',
    'Phoenix/Ensembler/truediv',
    'Phoenix/Ensembler/truediv/y',
    'Phoenix/Ensembler/StopGradient',
    'Phoenix/Ensembler/StopGradient_1',
    'zeros',
    'zeros_1',
]

_NONADAPTIVE_GRAPH_NODES_PRIORS_WEIGHTED = [
    'Phoenix/Ensembler/StopGradient',
    'Phoenix/Ensembler/StopGradient_1',
    'Phoenix/Ensembler/concat',
    'Phoenix/Ensembler/concat/axis',
    'Phoenix/Ensembler/dense/BiasAdd',
    'Phoenix/Ensembler/dense/BiasAdd/ReadVariableOp',
    'Phoenix/Ensembler/dense/MatMul',
    'Phoenix/Ensembler/dense/MatMul/ReadVariableOp',
    'Phoenix/Ensembler/dense/bias',
    'Phoenix/Ensembler/dense/bias/Assign',
    'Phoenix/Ensembler/dense/bias/Initializer/zeros',
    'Phoenix/Ensembler/dense/bias/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense/bias/Read/ReadVariableOp',
    'Phoenix/Ensembler/dense/kernel',
    'Phoenix/Ensembler/dense/kernel/Assign',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/RandomUniform',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/max',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/min',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/mul',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/shape',
    'Phoenix/Ensembler/dense/kernel/Initializer/random_uniform/sub',
    'Phoenix/Ensembler/dense/kernel/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense/kernel/Read/ReadVariableOp',
    'zeros',
    'zeros_1',
]

_SEARCH_GRAPH_NODES = [
    'dense/BiasAdd',
    'dense/BiasAdd/ReadVariableOp',
    'dense/MatMul',
    'dense/MatMul/ReadVariableOp',
    'dense/bias',
    'dense/bias/Assign',
    'dense/bias/Initializer/zeros',
    'dense/bias/IsInitialized/VarIsInitializedOp',
    'dense/bias/Read/ReadVariableOp',
    'dense/kernel',
    'dense/kernel/Assign',
    'dense/kernel/Initializer/random_uniform',
    'dense/kernel/Initializer/random_uniform/RandomUniform',
    'dense/kernel/Initializer/random_uniform/max',
    'dense/kernel/Initializer/random_uniform/min',
    'dense/kernel/Initializer/random_uniform/mul',
    'dense/kernel/Initializer/random_uniform/shape',
    'dense/kernel/Initializer/random_uniform/sub',
    'dense/kernel/IsInitialized/VarIsInitializedOp',
    'dense/kernel/Read/ReadVariableOp',
    'zeros',
]

_ADAPTIVE_AVERAGE_NODE_PRIORS = [
    'Phoenix/Ensembler/AddN',
    'Phoenix/Ensembler/truediv',
    'Phoenix/Ensembler/truediv/y',
    'Phoenix/Ensembler/StopGradient',
    'dense/BiasAdd',
    'dense/BiasAdd/ReadVariableOp',
    'dense/MatMul',
    'dense/MatMul/ReadVariableOp',
    'dense/bias',
    'dense/bias/Assign',
    'dense/bias/Initializer/zeros',
    'dense/bias/IsInitialized/VarIsInitializedOp',
    'dense/bias/Read/ReadVariableOp',
    'dense/kernel',
    'dense/kernel/Assign',
    'dense/kernel/Initializer/random_uniform',
    'dense/kernel/Initializer/random_uniform/RandomUniform',
    'dense/kernel/Initializer/random_uniform/max',
    'dense/kernel/Initializer/random_uniform/min',
    'dense/kernel/Initializer/random_uniform/mul',
    'dense/kernel/Initializer/random_uniform/shape',
    'dense/kernel/Initializer/random_uniform/sub',
    'dense/kernel/IsInitialized/VarIsInitializedOp',
    'dense/kernel/Read/ReadVariableOp',
    'zeros',
    'zeros_1',
]

_RESIDUAL_AVERAGE_PRIOR = [
    'Phoenix/Ensembler/AddN',
    'Phoenix/Ensembler/truediv',
    'Phoenix/Ensembler/truediv/y',
    'Phoenix/Ensembler/StopGradient',
    'dense/BiasAdd',
    'dense/BiasAdd/ReadVariableOp',
    'dense/MatMul',
    'dense/MatMul/ReadVariableOp',
    'dense/bias',
    'dense/bias/Assign',
    'dense/bias/Initializer/zeros',
    'dense/bias/IsInitialized/VarIsInitializedOp',
    'dense/bias/Read/ReadVariableOp',
    'dense/kernel',
    'dense/kernel/Assign',
    'dense/kernel/Initializer/random_uniform',
    'dense/kernel/Initializer/random_uniform/RandomUniform',
    'dense/kernel/Initializer/random_uniform/max',
    'dense/kernel/Initializer/random_uniform/min',
    'dense/kernel/Initializer/random_uniform/mul',
    'dense/kernel/Initializer/random_uniform/shape',
    'dense/kernel/Initializer/random_uniform/sub',
    'dense/kernel/IsInitialized/VarIsInitializedOp',
    'dense/kernel/Read/ReadVariableOp',
    'zeros',
    'zeros_1',
]

_ADAPTIVE_WEIGHTED_PRIORS = [
    'Phoenix/Ensembler/StopGradient',
    'Phoenix/Ensembler/concat',
    'Phoenix/Ensembler/concat/axis',
    'Phoenix/Ensembler/dense_1/BiasAdd',
    'Phoenix/Ensembler/dense_1/BiasAdd/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/MatMul',
    'Phoenix/Ensembler/dense_1/MatMul/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/bias',
    'Phoenix/Ensembler/dense_1/bias/Assign',
    'Phoenix/Ensembler/dense_1/bias/Initializer/zeros',
    'Phoenix/Ensembler/dense_1/bias/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense_1/bias/Read/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/kernel',
    'Phoenix/Ensembler/dense_1/kernel/Assign',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/RandomUniform',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/max',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/min',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/mul',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/shape',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/sub',
    'Phoenix/Ensembler/dense_1/kernel/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense_1/kernel/Read/ReadVariableOp',
    'dense/BiasAdd',
    'dense/BiasAdd/ReadVariableOp',
    'dense/MatMul',
    'dense/MatMul/ReadVariableOp',
    'dense/bias',
    'dense/bias/Assign',
    'dense/bias/Initializer/zeros',
    'dense/bias/IsInitialized/VarIsInitializedOp',
    'dense/bias/Read/ReadVariableOp',
    'dense/kernel',
    'dense/kernel/Assign',
    'dense/kernel/Initializer/random_uniform',
    'dense/kernel/Initializer/random_uniform/RandomUniform',
    'dense/kernel/Initializer/random_uniform/max',
    'dense/kernel/Initializer/random_uniform/min',
    'dense/kernel/Initializer/random_uniform/mul',
    'dense/kernel/Initializer/random_uniform/shape',
    'dense/kernel/Initializer/random_uniform/sub',
    'dense/kernel/IsInitialized/VarIsInitializedOp',
    'dense/kernel/Read/ReadVariableOp',
    'zeros',
    'zeros_1',
]

_RESIDUAL_WEIGHTED_PRIOR = [
    'Phoenix/Ensembler/StopGradient',
    'Phoenix/Ensembler/concat',
    'Phoenix/Ensembler/concat/axis',
    'Phoenix/Ensembler/dense_1/BiasAdd',
    'Phoenix/Ensembler/dense_1/BiasAdd/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/MatMul',
    'Phoenix/Ensembler/dense_1/MatMul/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/bias',
    'Phoenix/Ensembler/dense_1/bias/Assign',
    'Phoenix/Ensembler/dense_1/bias/Initializer/zeros',
    'Phoenix/Ensembler/dense_1/bias/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense_1/bias/Read/ReadVariableOp',
    'Phoenix/Ensembler/dense_1/kernel',
    'Phoenix/Ensembler/dense_1/kernel/Assign',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/RandomUniform',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/max',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/min',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/mul',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/shape',
    'Phoenix/Ensembler/dense_1/kernel/Initializer/random_uniform/sub',
    'Phoenix/Ensembler/dense_1/kernel/IsInitialized/VarIsInitializedOp',
    'Phoenix/Ensembler/dense_1/kernel/Read/ReadVariableOp',
    'dense/BiasAdd',
    'dense/BiasAdd/ReadVariableOp',
    'dense/MatMul',
    'dense/MatMul/ReadVariableOp',
    'dense/bias',
    'dense/bias/Assign',
    'dense/bias/Initializer/zeros',
    'dense/bias/IsInitialized/VarIsInitializedOp',
    'dense/bias/Read/ReadVariableOp',
    'dense/kernel',
    'dense/kernel/Assign',
    'dense/kernel/Initializer/random_uniform',
    'dense/kernel/Initializer/random_uniform/RandomUniform',
    'dense/kernel/Initializer/random_uniform/max',
    'dense/kernel/Initializer/random_uniform/min',
    'dense/kernel/Initializer/random_uniform/mul',
    'dense/kernel/Initializer/random_uniform/shape',
    'dense/kernel/Initializer/random_uniform/sub',
    'dense/kernel/IsInitialized/VarIsInitializedOp',
    'dense/kernel/Read/ReadVariableOp',
    'zeros',
    'zeros_1',
]


class EnsemblerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name':
              'nonadaptive_average_priors',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              2,
          'logits':
              0,
          'output_graph':
              _NONADAPTIVE_GRAPH_NODES_PRIORS,
      }, {
          'testcase_name':
              'intermix_average_priors',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              2,
          'logits':
              0,
          'output_graph':
              _NONADAPTIVE_GRAPH_NODES_PRIORS,
      }, {
          'testcase_name':
              'nonadaptive_average_search',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              0,
          'logits':
              1,
          'output_graph':
              _SEARCH_GRAPH_NODES,
      }, {
          'testcase_name':
              'intermix_average_search',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              0,
          'logits':
              1,
          'output_graph':
              _SEARCH_GRAPH_NODES,
      }, {
          'testcase_name':
              'adaptive_average_search',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              0,
          'logits':
              1,
          'output_graph':
              _SEARCH_GRAPH_NODES,
      }, {
          'testcase_name':
              'residual_average_search',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.RESIDUAL_ENSEMBLE_SEARCH,
          'priors':
              0,
          'logits':
              1,
          'output_graph':
              _SEARCH_GRAPH_NODES,
      }, {
          'testcase_name':
              'nonadaptive_weighted',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.WEIGHTED_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              2,
          'logits':
              0,
          'output_graph':
              _NONADAPTIVE_GRAPH_NODES_PRIORS_WEIGHTED,
      }, {
          'testcase_name':
              'intermix_weighted',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.WEIGHTED_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec
              .INTERMIXED_NONADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              2,
          'logits':
              0,
          'output_graph':
              _NONADAPTIVE_GRAPH_NODES_PRIORS_WEIGHTED,
      }, {
          'testcase_name':
              'adaptive_average_prior',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              1,
          'logits':
              1,
          'output_graph':
              _ADAPTIVE_AVERAGE_NODE_PRIORS,
      }, {
          'testcase_name':
              'residual_average_prior',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.AVERAGE_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.RESIDUAL_ENSEMBLE_SEARCH,
          'priors':
              1,
          'logits':
              1,
          'output_graph':
              _RESIDUAL_AVERAGE_PRIOR,
      }, {
          'testcase_name':
              'adaptive_weighted_prior',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.WEIGHTED_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.ADAPTIVE_ENSEMBLE_SEARCH,
          'priors':
              1,
          'logits':
              1,
          'output_graph':
              _ADAPTIVE_WEIGHTED_PRIORS,
      }, {
          'testcase_name':
              'residual_weighted_prior',
          'combining_type':
              ensembling_spec_pb2.EnsemblingSpec.WEIGHTED_ENSEMBLE,
          'search_type':
              ensembling_spec_pb2.EnsemblingSpec.RESIDUAL_ENSEMBLE_SEARCH,
          'priors':
              1,
          'logits':
              1,
          'output_graph':
              _RESIDUAL_WEIGHTED_PRIOR,
      })
  def test_ensembler(self, combining_type, search_type, priors, logits,
                     output_graph):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      spec = phoenix_spec_pb2.PhoenixSpec()
      spec.ensemble_spec.combining_type = combining_type
      spec.ensemble_spec.ensemble_search_type = search_type
      ensembler_instance = ensembler.Ensembler(spec)
      priors_logits_specs = []
      search_logits_specs = []
      if priors:
        for _ in range(priors):
          spec = architecture_utils.LogitsSpec(logits=tf.zeros([20, 10]))
          priors_logits_specs.append(spec)
      if logits:
        spec = architecture_utils.LogitsSpec(
            logits=tf.keras.layers.Dense(10)(tf.zeros([20, 10])))
        search_logits_specs.append(spec)
      _ = ensembler_instance.bundle_logits(
          priors_logits_specs=priors_logits_specs,
          search_logits_specs=search_logits_specs,
          logits_dimension=10)
      nodes = tf.compat.v1.get_default_graph().as_graph_def().node
      logging.info([node.name for node in nodes])

      self.assertCountEqual([n.name for n in nodes], output_graph)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
