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
"""Tests for model_search.blocks_builder."""

from model_search import blocks_builder
import tensorflow.compat.v2 as tf


class BlocksBuilderTest(tf.test.TestCase):

  def test_constructor(self):
    blocks = blocks_builder.Blocks()
    input_tensor = tf.zeros([3, 32, 32, 3])
    block_type = blocks_builder.BlockType.FIXED_CHANNEL_CONVOLUTION_16
    _ = blocks[block_type].build([input_tensor], is_training=True)

  def test_all_blocks_are_there(self):
    blocks = blocks_builder.Blocks()
    for block_type in blocks_builder.BlockType:
      if block_type == blocks_builder.BlockType.EMPTY_BLOCK:
        continue
      blocks[block_type]  # pylint: disable=pointless-statement

  def test_blocks_search_space(self):
    hps = blocks_builder.Blocks.search_space()
    self.assertIn("TUNABLE_SVDF_output_size", hps)
    self.assertIn("TUNABLE_SVDF_rank", hps)
    self.assertIn("TUNABLE_SVDF_projection_size", hps)
    self.assertIn("TUNABLE_SVDF_memory_size", hps)
    hps = blocks_builder.Blocks.search_space(["TUNABLE_SVDF"])
    self.assertIn("TUNABLE_SVDF_output_size", hps)
    self.assertIn("TUNABLE_SVDF_rank", hps)
    self.assertIn("TUNABLE_SVDF_projection_size", hps)
    self.assertIn("TUNABLE_SVDF_memory_size", hps)

  def test_naming_of_tunable(self):
    # If this test is failing, it is because the user have registered two
    # tunable blocks with names that substrings of one another.
    blocks = blocks_builder.Blocks()
    names = []
    for k, v in blocks._block_builders.items():
      if v is not None:
        hps = v.requires_hparams()
        if hps:
          names.append(k)

    for idx, name in enumerate(names):
      for idx2, name2 in enumerate(names):
        if idx != idx2:
          self.assertNotStartsWith(name, name2)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
