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
"""Tests for model_search.blocks."""

from absl.testing import parameterized

from model_search import blocks
from model_search import hparam as hp
from model_search.architecture import architecture_utils
import tensorflow.compat.v2 as tf
import tf_slim

arg_scope = tf_slim.arg_scope


class BlocksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'fixed_channel_block',
          'block': blocks.FixedChannelConvolutionBlock(),
          'output_shape': [2, 32, 32, 64]
      }, {
          'testcase_name': 'conv_block',
          'block': blocks.ConvolutionBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'conv_block_batch_norm',
          'block': blocks.ConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'non_symmetric_conv_block',
          'block': blocks.ConvolutionBlock(kernel_size=(3, 1)),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dilated_conv_block',
          'block': blocks.DilatedConvolutionBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dilated_conv_block_batch_norm',
          'block': blocks.DilatedConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'max_pool_block',
          'block': blocks.MaxPoolingBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'increase_block',
          'block': blocks.IncreaseChannelsBlock(),
          'output_shape': [2, 32, 32, 6]
      }, {
          'testcase_name': 'increase_block_batch_norm',
          'block': blocks.IncreaseChannelsBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 6]
      }, {
          'testcase_name': 'down_block',
          'block': blocks.DownsampleConvolutionBlock(),
          'output_shape': [2, 16, 16, 6]
      }, {
          'testcase_name': 'down_block_batch_norm',
          'block': blocks.DownsampleConvolutionBlock(apply_batch_norm=True),
          'output_shape': [2, 16, 16, 6]
      }, {
          'testcase_name': 'avg_block',
          'block': blocks.AveragePoolBlock(),
          'output_shape': [2, 16, 16, 3]
      }, {
          'testcase_name': 'resnet_block',
          'block': blocks.ResnetBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'resnet_block_batch_norm',
          'block': blocks.ResnetBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'dense_block',
          'block': blocks.FullyConnectedBlock(4000, 1000000000),
          'output_shape': [2, 3072]
      }, {
          'testcase_name': 'fixed_dense_block',
          'block': blocks.FixedOutputFullyConnectedBlock(100),
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'low_rank',
          'block': blocks.LowRankLayerBlock(kernel_rank=2),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'low_rank_2',
          'block': blocks.LowRankLayerBlock(kernel_rank=1),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'low_rank_skip',
          'block': blocks.LowRankLayerBlock(kernel_rank=2, skip_connect=True),
          'input_shape': [2, 100],
          'output_shape': [2, 100]
      }, {
          'testcase_name': 'dense_block_parameter_restricted',
          'block': blocks.FullyConnectedBlock(4000, 6553600),
          'output_shape': [2, 2133]
      }, {
          'testcase_name': 'dense_block_restricted_output',
          'block': blocks.FullyConnectedBlock(1024),
          'output_shape': [2, 1024]
      }, {
          'testcase_name':
              'dense_block_with_batch_norm',
          'block':
              blocks.FullyConnectedBlock(
                  4000, 1000000000, apply_batch_norm=True),
          'output_shape': [2, 3072]
      }, {
          'testcase_name':
              'dense_block_residual_force_match_shapes_input_larger',
          'input_shape': [2, 1000],
          'block':
              blocks.FullyConnectedBlock(
                  100,
                  1000000000,
                  residual_connection_type=blocks.ResidualConnectionType
                  .FORCE_MATCH_SHAPES),
          'output_shape': [2, 100]
      }, {
          'testcase_name':
              'dense_block_residual_force_match_shapes_input_smaller',
          'input_shape': [2, 1],
          'block':
              blocks.FullyConnectedBlock(
                  100,
                  1000000000,
                  residual_connection_type=blocks.ResidualConnectionType
                  .FORCE_MATCH_SHAPES),
          'output_shape': [2, 2]
      }, {
          'testcase_name':
              'dense_block_residual_concat',
          'input_shape': [2, 1000],
          'block':
              blocks.FullyConnectedBlock(
                  4000,
                  1000000000,
                  residual_connection_type=blocks.ResidualConnectionType.CONCAT
              ),
          'output_shape': [2, 2000]
      }, {
          'testcase_name':
              'dense_block_residual_project',
          'input_shape': [2, 1000],
          'block':
              blocks.FullyConnectedBlock(
                  40,
                  1000000000,
                  residual_connection_type=blocks.ResidualConnectionType.PROJECT
              ),
          'output_shape': [2, 40]
      }, {
          'testcase_name':
              'dense_block_residual_project_with_batchnorm',
          'block':
              blocks.FullyConnectedBlock(
                  4000,
                  1000000000,
                  residual_connection_type=blocks.ResidualConnectionType
                  .PROJECT,
                  apply_batch_norm=True),
          'output_shape': [2, 3072]
      }, {
          'testcase_name': 'downsample_flatten',
          'block': blocks.DownsampleFlattenBlock(),
          'output_shape': [2, 96]
      }, {
          'testcase_name': 'downsample_flatten_limit_hit',
          'block': blocks.DownsampleFlattenBlock(max_channels=50),
          'output_shape': [2, 50]
      }, {
          'testcase_name': 'dual_resnet_block',
          'block': blocks.DualResnetBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block1',
          'block': blocks.GeneralBlock(),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block2',
          'block': blocks.GeneralBlock(apply_batch_norm=True),
          'output_shape': [2, 32, 32, 3]
      }, {
          'testcase_name': 'general_block3',
          'block': blocks.GeneralBlock(apply_batch_norm=True),
          'input_shape': [2, 32, 32, 7],
          'output_shape': [2, 32, 32, 5],
      }, {
          'testcase_name': 'general_block4',
          'block': blocks.GeneralBlock(apply_batch_norm=True),
          'input_shape': [2, 32, 32, 20],
          'output_shape': [2, 32, 32, 20],
      }, {
          'testcase_name': 'plate_reduction',
          'block': blocks.PlateReductionFlatten(),
          'output_shape': [2, 3],
      }, {
          'testcase_name': 'rnn_block1',
          'block': blocks.RnnBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'rnn_block1_skip',
          'block': blocks.RnnBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'rnn_block2',
          'block': blocks.RnnBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'rnn_block2_skip',
          'block': blocks.RnnBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'recurrent_dense_block1',
          'block': blocks.Conv1DBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 50],
      }, {
          'testcase_name': 'recurrent_dense_block2',
          'block': blocks.Conv1DBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'recurrent_dense_block_kernel_2',
          'block': blocks.Conv1DBlock(output_size=200, kernel_size=2),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name':
              'recurrent_dense_block_kernel_2_skip',
          'block':
              blocks.Conv1DBlock(output_size=200, kernel_size=2, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'svdf_block1',
          'block': blocks.SvdfBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'svdf_block2',
          'block': blocks.SvdfBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name':
              'svdf_block_conv_layer',
          'block':
              blocks.SvdfBlock(
                  output_size=200,
                  fashion=blocks.SvdfImplementationFashion.SVDF_CONV),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'svdf_block_with_projection',
          'block': blocks.SvdfBlock(output_size=200, projection_size=64),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 64],
      }, {
          'testcase_name':
              'svdf_conv_block_with_projection',
          'block':
              blocks.SvdfBlock(
                  output_size=199,
                  projection_size=32,
                  fashion=blocks.SvdfImplementationFashion.SVDF_CONV),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 32],
      }, {
          'testcase_name': 'lstm_block1',
          'block': blocks.LSTMBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'lstm_block1_skip',
          'block': blocks.LSTMBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 100],
      }, {
          'testcase_name': 'lstm_block2',
          'block': blocks.LSTMBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'lstm_block2_skip',
          'block': blocks.LSTMBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block1',
          'block': blocks.BidirectionalLSTMBlock(),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block1_skip',
          'block': blocks.BidirectionalLSTMBlock(skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 200],
      }, {
          'testcase_name': 'bi_lstm_block2',
          'block': blocks.BidirectionalLSTMBlock(output_size=200),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 400],
      }, {
          'testcase_name': 'bi_lstm_block2_skip',
          'block': blocks.BidirectionalLSTMBlock(output_size=200, skip=True),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 400],
      }, {
          'testcase_name': 'Identity',
          'block': blocks.IdentityBlock(),
          'input_shape': [2, 3, 3, 4],
          'output_shape': [2, 3, 3, 4],
      }, {
          'testcase_name': 'Bottleneck',
          'block': blocks.BottleNeckBlock(),
          'input_shape': [2, 3],
          'output_shape': [2, 3],
      }, {
          'testcase_name': 'Bottleneck_skip',
          'block': blocks.BottleNeckBlock(skip_connect=True),
          'input_shape': [2, 3],
          'output_shape': [2, 3],
      }, {
          'testcase_name':
              'tunable_svdf',
          'block':
              blocks.TunableSvdfBlock(),
          'hparams':
              hp.HParams(
                  output_size=10, rank=1, memory_size=4, projection_size=34),
          'input_shape': [2, 10, 10],
          'output_shape': [2, 10, 34],
      })
  def test_build_and_apply_block(self,
                                 block,
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
        output = block.build([input_tensor], is_training=True, hparams=hparams)
      self.assertLessEqual(len(output), 2)
      self.assertAllEqual(output[-1].shape, output_shape)


class BlocksImportTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'fixed_conv',
          'block': blocks.FixedChannelConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'conv',
          'block': blocks.ConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'inc_conv',
          'block': blocks.IncreaseChannelsBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'down_conv',
          'block': blocks.DownsampleConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'resnet',
          'block': blocks.ResnetBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'fc',
          'block': blocks.FullyConnectedBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'fc_low_rank',
          'block': blocks.LowRankLayerBlock(kernel_rank=2),
          'input_rank': 2
      }, {
          'testcase_name': 'fc_low_rank_skip',
          'block': blocks.LowRankLayerBlock(kernel_rank=2, skip_connect=True),
          'input_rank': 2
      }, {
          'testcase_name': 'fixed_fc',
          'block': blocks.FixedOutputFullyConnectedBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'fc_pyramid',
          'block': blocks.FullyConnectedPyramidBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'dilated_conv',
          'block': blocks.DilatedConvolutionBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'down_flatten_conv',
          'block': blocks.DownsampleFlattenBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'dual_resnet',
          'block': blocks.DualResnetBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'general',
          'block': blocks.GeneralBlock(),
          'input_rank': 4
      }, {
          'testcase_name': 'rnn',
          'block': blocks.RnnBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'rnn_skip',
          'block': blocks.RnnBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'conv1d',
          'block': blocks.Conv1DBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'conv1d_skip',
          'block': blocks.Conv1DBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'svdf',
          'block': blocks.SvdfBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'svdf_with_projection',
          'block': blocks.SvdfBlock(projection_size=64),
          'input_rank': 3
      }, {
          'testcase_name':
              'svdf_conv',
          'block':
              blocks.SvdfBlock(
                  fashion=blocks.SvdfImplementationFashion.SVDF_CONV),
          'input_rank':
              3
      }, {
          'testcase_name': 'lstm',
          'block': blocks.LSTMBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'lstm_skip',
          'block': blocks.LSTMBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'bilstm',
          'block': blocks.BidirectionalLSTMBlock(),
          'input_rank': 3
      }, {
          'testcase_name': 'bilstm_skip',
          'block': blocks.BidirectionalLSTMBlock(skip=True),
          'input_rank': 3
      }, {
          'testcase_name': 'bottleneck',
          'block': blocks.BottleNeckBlock(),
          'input_rank': 2
      }, {
          'testcase_name': 'bottleneck_skip',
          'block': blocks.BottleNeckBlock(skip_connect=True),
          'input_rank': 2
      }, {
          'testcase_name':
              'tunable_svdf',
          'block':
              blocks.TunableSvdfBlock(),
          'input_rank':
              3,
          'hparams':
              hp.HParams(
                  output_size=10, rank=1, memory_size=4, projection_size=34)
      })
  def test_ability_to_import(self, block, input_rank, hparams=None):
    # Force graph mode
    with tf.compat.v1.Graph().as_default():
      input_tensor = tf.zeros([32] * (input_rank))

      with arg_scope(architecture_utils.DATA_FORMAT_OPS, data_format='NHWC'):
        with tf.compat.v1.variable_scope('scope_a'):
          _ = block.build([input_tensor], is_training=True, hparams=hparams)
        with tf.compat.v1.variable_scope('scope_b'):
          _ = block.build([input_tensor], is_training=True, hparams=hparams)

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
  tf.enable_v2_behavior()
  tf.test.main()
