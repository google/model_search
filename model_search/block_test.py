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
"""Tests for model_search.block."""

from absl.testing import parameterized

from model_search import block
from model_search import hparam as hp
from model_search.architecture import architecture_utils
import tensorflow.compat.v2 as tf
import tf_slim

arg_scope = tf_slim.arg_scope


class BlockTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'fixed_channel_block',
          'tested_block': block.FixedChannelConvolutionBlock(),
          'output_shape': [2, 32, 32, 64]
      }, {
          'testcase_name': 'conv_block',
          'tested_block': block.ConvolutionBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'conv_block_batch_norm',
          'tested_block': block.ConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'non_symmetric_conv_block',
          'tested_block': block.ConvolutionBlock(kernel_size=(3, 1)),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dilated_conv_block',
          'tested_block': block.DilatedConvolutionBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dilated_conv_block_batch_norm',
          'tested_block': block.DilatedConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'max_pool_block',
          'tested_block': block.MaxPoolingBlock(),
          'output_shape': [2, 16, 16, 3]
      }, {
          'testcase_name': 'increase_block',
          'tested_block': block.IncreaseChannelsBlock(),
          'output_shape': [2, 32, 32, 6]
      }, {
          'testcase_name': 'increase_block_batch_norm',
          'tested_block': block.IncreaseChannelsBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 6]
      }, {
          'testcase_name': 'down_block',
          'tested_block': block.DownsampleConvolutionBlock(),
          'output_shape': [2, 16, 16, 6]
      }, {
          'testcase_name':
              'down_block_batch_norm',
          'tested_block':
              block.DownsampleConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 16, 16, 6]
      }, {
          'testcase_name': 'avg_block',
          'tested_block': block.AveragePoolBlock(),
          'output_shape': [2, 16, 16, 3]
      }, {
          'testcase_name': 'resnet_block',
          'tested_block': block.ResnetBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'resnet_block_batch_norm',
          'tested_block': block.ResnetBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dense_block',
          'tested_block': block.FullyConnectedBlock(4000, 1000000000),
          'output_shape': [2, 3072]
      }, {
          'testcase_name': 'fixed_dense_block',
          'tested_block': block.FixedOutputFullyConnectedBlock(100),
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'low_rank',
          'tested_block': block.LowRankLayerBlock(kernel_rank=2),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'low_rank_2',
          'tested_block': block.LowRankLayerBlock(kernel_rank=1),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name':
              'low_rank_skip',
          'tested_block':
              block.LowRankLayerBlock(kernel_rank=2, skip_connect=True),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'dense_block_parameter_restricted',
          'tested_block': block.FullyConnectedBlock(4000, 6553600),
          'output_shape': [2, 2133]
      }, {
          'testcase_name': 'dense_block_restricted_output',
          'tested_block': block.FullyConnectedBlock(1024),
          'output_shape': [2, 1024]
      }, {
          'testcase_name':
              'dense_block_with_batch_norm',
          'tested_block':
              block.FullyConnectedBlock(
                  4000, 1000000000, apply_batch_norm=True),
          'output_shape': [2, 3072]
      }, {
          'testcase_name':
              'dense_block_residual_force_match_shapes_input_larger',
          'input_shape': [2, 1000],
          'tested_block':
              block.FullyConnectedBlock(
                  100,
                  1000000000,
                  residual_connection_type=block.ResidualConnectionType
                  .FORCE_MATCH_SHAPES),
          'output_shape': [2, 100]
      }, {
          'testcase_name':
              'dense_block_residual_force_match_shapes_input_smaller',
          'input_shape': [2, 1],
          'tested_block':
              block.FullyConnectedBlock(
                  100,
                  1000000000,
                  residual_connection_type=block.ResidualConnectionType
                  .FORCE_MATCH_SHAPES),
          'output_shape': [2, 2]
      }, {
          'testcase_name':
              'dense_block_residual_concat',
          'input_shape': [2, 1000],
          'tested_block':
              block.FullyConnectedBlock(
                  4000,
                  1000000000,
                  residual_connection_type=block.ResidualConnectionType.CONCAT),
          'output_shape': [2, 2000]
      }, {
          'testcase_name':
              'dense_block_residual_project',
          'input_shape': [2, 1000],
          'tested_block':
              block.FullyConnectedBlock(
                  40,
                  1000000000,
                  residual_connection_type=block.ResidualConnectionType.PROJECT
              ),
          'output_shape': [2, 40]
      }, {
          'testcase_name':
              'dense_block_residual_project_with_batchnorm',
          'tested_block':
              block.FullyConnectedBlock(
                  4000,
                  1000000000,
                  residual_connection_type=block.ResidualConnectionType.PROJECT,
                  apply_batch_norm=True),
          'output_shape': [2, 3072]
      }, {
          'testcase_name': 'downsample_flatten',
          'tested_block': block.DownsampleFlattenBlock(),
          'output_shape': [2, 96]
      }, {
          'testcase_name': 'downsample_flatten_limit_hit',
          'tested_block': block.DownsampleFlattenBlock(max_channels=50),
          'output_shape': [2, 50]
      }, {
          'testcase_name': 'dual_resnet_block',
          'tested_block': block.DualResnetBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block1',
          'tested_block': block.GeneralBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block2',
          'tested_block': block.GeneralBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block3',
          'tested_block': block.GeneralBlock(apply_batch_norm=True),
          'input_shape': [2, 32, 32, 7],
          'output_shape': [2, 32, 32, 5],
      }, {
          'testcase_name': 'general_block4',
          'tested_block': block.GeneralBlock(apply_batch_norm=True),
          'input_shape': [2, 32, 32, 20],
          'output_shape': [2, 32, 32, 20],
      }, {
          'testcase_name': 'plate_reduction',
          'tested_block': block.PlateReductionFlatten(),
          'output_shape': [2, 3],
      }, {
          'testcase_name': 'rnn_block1',
          'tested_block': block.RnnBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'rnn_block1_skip',
          'tested_block': block.RnnBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'rnn_block2',
          'tested_block': block.RnnBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'rnn_block2_skip',
          'tested_block': block.RnnBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'recurrent_dense_block1',
          'tested_block': block.Conv1DBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 50],
      }, {
          'testcase_name': 'recurrent_dense_block2',
          'tested_block': block.Conv1DBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'recurrent_dense_block_kernel_2',
          'tested_block': block.Conv1DBlock(output_size=200, kernel_size=2),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name':
              'recurrent_dense_block_kernel_2_skip',
          'tested_block':
              block.Conv1DBlock(output_size=200, kernel_size=2, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'svdf_block1',
          'tested_block': block.SvdfBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'svdf_block2',
          'tested_block': block.SvdfBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name':
              'svdf_block_conv_layer',
          'tested_block':
              block.SvdfBlock(
                  output_size=200,
                  fashion=block.SvdfImplementationFashion.SVDF_CONV),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'svdf_block_with_projection',
          'tested_block': block.SvdfBlock(output_size=200, projection_size=64),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 64],
      }, {
          'testcase_name':
              'svdf_conv_block_with_projection',
          'tested_block':
              block.SvdfBlock(
                  output_size=199,
                  projection_size=32,
                  fashion=block.SvdfImplementationFashion.SVDF_CONV),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 32],
      }, {
          'testcase_name': 'lstm_block1',
          'tested_block': block.LSTMBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'lstm_block1_skip',
          'tested_block': block.LSTMBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'lstm_block2',
          'tested_block': block.LSTMBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'lstm_block2_skip',
          'tested_block': block.LSTMBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block1',
          'tested_block': block.BidirectionalLSTMBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block1_skip',
          'tested_block': block.BidirectionalLSTMBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block2',
          'tested_block': block.BidirectionalLSTMBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 400],
      }, {
          'testcase_name':
              'bi_lstm_block2_skip',
          'tested_block':
              block.BidirectionalLSTMBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 400],
      }, {
          'testcase_name': 'Identity',
          'tested_block': block.IdentityBlock(),
          'input_shape': [2, 3, 3, 4],
          'output_shape': [2, 3, 3, 4],
      }, {
          'testcase_name': 'Bottleneck',
          'tested_block': block.BottleNeckBlock(),
          'input_shape': [2, 3],
          'output_shape': [2, 3],
      }, {
          'testcase_name': 'Bottleneck_skip',
          'tested_block': block.BottleNeckBlock(skip_connect=True),
          'input_shape': [2, 3],
          'output_shape': [2, 3],
      }, {
          'testcase_name':
              'tunable_svdf',
          'tested_block':
              block.TunableSvdfBlock(),
          'hparams':
              hp.HParams(
                  output_size=10, rank=1, memory_size=4, projection_size=34),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 34],
      })
  def test_build_and_apply_block(self,
                                 tested_block,
                                 output_shape,
                                 hparams=None,
                                 input_shape=None):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      if input_shape:
        input_tensor = tf.zeros(input_shape)
      else:
        input_tensor = tf.zeros([2, 32, 32, 3])

      with arg_scope(architecture_utils.DATA_FORMAT_OPS, data_format='NHWC'):
        output = tested_block.block_build([input_tensor],
                                          is_training=True,
                                          hparams=hparams)
      self.assertLessEqual(len(output), 2)
      self.assertAllEqual(output[-1].shape, output_shape)


class BlockImportTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'fixed_conv',
          'tested_block': block.FixedChannelConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'conv',
          'tested_block': block.ConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'inc_conv',
          'tested_block': block.IncreaseChannelsBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'down_conv',
          'tested_block': block.DownsampleConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'resnet',
          'tested_block': block.ResnetBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'fc',
          'tested_block': block.FullyConnectedBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'fc_low_rank',
          'tested_block': block.LowRankLayerBlock(kernel_rank=2),
          'input_rank': 2
      }, {
          'testcase_name':
              'fc_low_rank_skip',
          'tested_block':
              block.LowRankLayerBlock(kernel_rank=2, skip_connect=True),
          'input_rank':
              2
      }, {
          'testcase_name': 'fixed_fc',
          'tested_block': block.FixedOutputFullyConnectedBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'fc_pyramid',
          'tested_block': block.FullyConnectedPyramidBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'dilated_conv',
          'tested_block': block.DilatedConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'down_flatten_conv',
          'tested_block': block.DownsampleFlattenBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'dual_resnet',
          'tested_block': block.DualResnetBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'general',
          'tested_block': block.GeneralBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'rnn',
          'tested_block': block.RnnBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'rnn_skip',
          'tested_block': block.RnnBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'conv1d',
          'tested_block': block.Conv1DBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'conv1d_skip',
          'tested_block': block.Conv1DBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'svdf',
          'tested_block': block.SvdfBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'svdf_with_projection',
          'tested_block': block.SvdfBlock(projection_size=64),
          'input_rank': 3
      }, {
          'testcase_name':
              'svdf_conv',
          'tested_block':
              block.SvdfBlock(fashion=block.SvdfImplementationFashion.SVDF_CONV
                             ),
          'input_rank':
              3
      }, {
          'testcase_name': 'lstm',
          'tested_block': block.LSTMBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'lstm_skip',
          'tested_block': block.LSTMBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'bilstm',
          'tested_block': block.BidirectionalLSTMBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'bilstm_skip',
          'tested_block': block.BidirectionalLSTMBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'bottleneck',
          'tested_block': block.BottleNeckBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'bottleneck_skip',
          'tested_block': block.BottleNeckBlock(skip_connect=True),
          'input_rank': 2
      }, {
          'testcase_name':
              'tunable_svdf',
          'tested_block':
              block.TunableSvdfBlock(),
          'input_rank':
              3,
          'hparams':
              hp.HParams(
                  output_size=10, rank=1, memory_size=4, projection_size=34)
      })
  def test_ability_to_import(self, tested_block, input_rank, hparams=None):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.zeros([32] * (input_rank))

      with arg_scope(architecture_utils.DATA_FORMAT_OPS, data_format='NHWC'):
        with tf.compat.v1.variable_scope('scope_a'):
          _ = tested_block.block_build([input_tensor],
                                       is_training=True,
                                       hparams=hparams)
        with tf.compat.v1.variable_scope('scope_b'):
          _ = tested_block.block_build([input_tensor],
                                       is_training=True,
                                       hparams=hparams)

      nodes = tf.compat.v1.get_default_graph().as_graph_def().node
      scope_a = [
          node.name
          for node in nodes
          if 'scope_a' in node.name and 'global_step' not in node.name
      ]
      scope_b = [
          node.name
          for node in nodes
          if 'scope_b' in node.name and 'global_step' not in node.name
      ]
      self.assertAllEqual(
          scope_a, [name.replace('scope_b', 'scope_a') for name in scope_b])


if __name__ == '__main__':
  tf.test.main()
