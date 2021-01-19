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
"""Tests for model_search.search.linear_model."""

from absl.testing import parameterized

import mock
from model_search import blocks_builder as blocks
from model_search import hparam as hp
from model_search.metadata import trial as trial_module
from model_search.proto import phoenix_spec_pb2
from model_search.search import linear_model
from model_search.search import test_utils as search_test_utils

import numpy as np

import tensorflow.compat.v2 as tf


# In real applications, oracle will specify a block to try.
# In these tests, we'll specify it manually.
NEW_BLOCK = "FIXED_CHANNEL_CONVOLUTION_16"


def _get_suggestion(architectures,
                    blocks_to_use,
                    losses,
                    grow=False,
                    remove_outliers=False,
                    pass_flatten=False):
  """Testing subroutine to handle boilerplate Trial construction, dirs, etc."""

  # TODO(b/172564129): Figure out how to use mock decorator for free functions.
  with mock.patch("model_search.architecture"
                  ".architecture_utils.get_architecture") as mock_get_arch:

    blocks_strs = [blocks.BlockType(b).name for b in blocks_to_use]
    spec = search_test_utils.create_spec(
        phoenix_spec_pb2.PhoenixSpec.CNN,
        blocks_to_use=blocks_strs,
    )
    spec.search_type = phoenix_spec_pb2.PhoenixSpec.LINEAR_MODEL
    spec.increase_complexity_probability = 1.0 if grow else 0.0
    spec.linear_model.remove_outliers = remove_outliers
    spec.linear_model.trials_before_fit = 1
    algorithm = linear_model.LinearModel(spec)

    mock_get_arch.side_effect = lambda idx: architectures[int(idx)]

    trials = []
    for i, loss in enumerate(losses):
      if isinstance(loss, (np.floating, np.integer)):
        loss = loss.item()
      trials.append(
          trial_module.Trial({
              "id": i,
              "model_dir": str(i),
              "status": "COMPLETED",
              "trial_infeasible": False,
              "final_measurement": {
                  "objective_value": loss
              }
          }))

    hparams = hp.HParams(new_block_type=NEW_BLOCK)
    # Second return val fork_trial is a nonsense concept for LinearModel.
    output_architecture, _ = algorithm.get_suggestion(trials, hparams)
    if not pass_flatten:
      output_architecture = np.array(
          [b for b in output_architecture if b not in blocks.FLATTEN_TYPES])
    return output_architecture


class LinearModelTest(parameterized.TestCase, tf.test.TestCase):

  def test_one_trial(self):
    """Degenerate case: one data point. Just make sure it doesn't explode."""
    blocks_to_use = np.arange(1, 4)
    architectures = np.array([[1, 2, 1]])
    losses = ([1.0])
    best = _get_suggestion(architectures, blocks_to_use, losses)
    # The degenerate model might end up suggesting some empty blocks.
    self.assertLessEqual(best.shape, (3,))

  def test_two_trials(self):
    """Underdetermined case - should find pattern in subset of dimensions."""
    blocks_to_use = np.arange(1, 4)
    architectures = np.array([[1, 2, 1], [1, 1, 2]])
    losses = ([1.0, 2.0])
    best = _get_suggestion(architectures, blocks_to_use, losses)
    # The model won't be able to predict any effect from layer 0.
    self.assertEqual(list(best[1:]), [2, 1])

  def test_three_trials(self):
    """Suggestion should combine trial 1 and 2's improvements over trial 0."""
    blocks_to_use = np.arange(1, 4)
    architectures = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 2]])
    losses = ([2.0, 1.0, 1.0])
    best = _get_suggestion(architectures, blocks_to_use, losses)
    self.assertEqual(list(best)[1:], [2, 2])

  def test_loss_equals_id(self):
    """Larger-scale overdetermined case with easily predicted model output.

    Each block contributes its own id worth of loss,
    so tower of all block 1 should be best.
    """
    nblocks = 4
    blocks_to_use = np.arange(1, nblocks + 1)
    depth = 9
    ntrials = 10 * nblocks * depth
    np.random.seed(0)
    architectures = np.random.randint(1, nblocks + 1, size=(ntrials, depth))
    losses = np.sum(architectures, axis=1)
    best = _get_suggestion(architectures, blocks_to_use, losses)
    self.assertEqual(list(best), [1] * depth)

  def test_randomized(self):
    """Overdetermined case with randomly chosen linear model.

    Hard to predict the argmin, but can check its expected performance.
    """
    np.random.seed(0)
    nblocks = 10
    blocks_to_use = np.arange(1, nblocks + 1)
    depth = 10
    ntrials = 2 * nblocks * depth
    architectures = np.random.randint(1, nblocks + 1, size=(ntrials, depth))

    # For chosen w, compute b such that all losses are positive.
    w = np.random.normal(size=(depth, nblocks))
    # Index arch-1 is because we had to avoid using the empty zero block,
    # because LinearModel will trim it, but we want to use block as index.
    losses = ([np.sum(w[np.arange(depth), arch - 1]) for arch in architectures])
    b = -np.amin(losses) + 0.001
    losses_positive = losses + b

    best = _get_suggestion(architectures, blocks_to_use, losses_positive)
    self.assertLen(best, depth)
    loss_by_model = np.sum(w[np.arange(depth), best - 1]) + b
    self.assertTrue(np.all(losses_positive >= loss_by_model))

  def test_ids_nonrange(self):
    """Make sure we correctly handle non-contiguous range of blocks."""
    np.random.seed(0)
    # TODO(b/172564129): This test is not correct.
    block_ints = [
        b.value
        for b in blocks.BlockType
        if b.value not in [126, 127, 128, 129]
    ]
    blocks_to_use = np.random.choice(block_ints, 3)
    architectures = np.array([blocks_to_use])
    losses = ([1.0])
    best = _get_suggestion(architectures, blocks_to_use, losses)
    for blockid in best:
      self.assertIn(blockid, blocks_to_use)

  def test_outlier_decile(self):
    """Validate outlier removal."""
    blocks_to_use = np.array([1, 2])
    n = 15
    architectures = np.tile([2, 1], (n, 1))
    architectures[0, :] = 1
    losses = np.concatenate([[2.0], np.ones(n - 3), np.repeat(10000.0, 2)])
    # With outliers included, block 2 in layer 0 looks really bad.
    # With outliers rejected, block 2 in layer 0 looks good.
    best = _get_suggestion(
        architectures, blocks_to_use, losses, remove_outliers=True)
    self.assertEqual(best[0], 2)

  def test_grow(self):
    """Verify that depth growing happens."""
    nblocks = 4
    blocks_to_use = np.arange(1, nblocks + 1)
    depth = 9
    ntrials = 10 * nblocks * depth
    np.random.seed(0)
    architectures = np.random.randint(1, nblocks + 1, size=(ntrials, depth))
    losses = np.sum(architectures, axis=1)
    expected_best = np.repeat(min(blocks_to_use), depth)

    best_nogrow = _get_suggestion(
        architectures, blocks_to_use, losses, grow=False)
    self.assertAllEqual(best_nogrow, expected_best)

    best_grow = _get_suggestion(architectures, blocks_to_use, losses, grow=True)
    self.assertAllEqual(best_grow,
                        np.append(expected_best, blocks.BlockType[NEW_BLOCK]))

  def test_flatten_modelfitting(self):
    """Ensure that we correctly deal with the flatten block in fitting.

    1) Flatten blocks won't be in spec.blocks_to_use, even though
       the trial architectures loaded from filesystem will contain them.

    2) The model shouldn't try to place a flatten block, since there is only
       one valid position at the convolutional -> dense transition.
    """

    blocks_to_use = np.arange(4)
    depth = 3
    ntrials = 10 * len(blocks_to_use) * depth
    flatten_blocks = (
        blocks.BlockType.FLATTEN,
        blocks.BlockType.DOWNSAMPLE_FLATTEN,
        blocks.BlockType.PLATE_REDUCTION_FLATTEN,
    )
    assert not set(b.value for b in flatten_blocks) & set(blocks_to_use)

    # Generate architectures that contain a flatten block.
    def random_arch():
      depth = np.random.randint(2, 5)
      arch = list(np.random.choice(blocks_to_use, size=depth))
      if np.random.choice([True, False]):
        flatten_loc = np.random.randint(depth)
        flatten_type = np.random.choice(flatten_blocks)
        arch.insert(flatten_loc, flatten_type)
      return np.array(arch)

    np.random.seed(0)
    architectures = [random_arch() for _ in range(ntrials)]
    losses = np.random.uniform(0.1, 1, size=ntrials)

    # Since flatten isn't in the spec blocks_to_use, the one-hot converter
    # will raise a ValueError unless we filter it out correctly.
    # We don't care what the linear model suggests; we only care that
    # it doesn't choke on the flatten block that's not in blocks_to_use
    try:
      _ = _get_suggestion(architectures, blocks_to_use, losses)
    except ValueError:
      self.fail("LinearModel failed to ignore flatten block")

  @parameterized.named_parameters(
      {
          "testcase_name": "grow",
          "grow": True,
      },
      {
          "testcase_name": "mutate",
          "grow": False,
      },
  )
  def test_flatten_output(self, grow):
    """Ensure we output suggestions with a flatten block correctly placed."""

    # Make trials s.t. the linear model will output all convolutions.
    architectures = [
        np.repeat(blocks.BlockType.EMPTY_BLOCK, 4),
        np.repeat(blocks.BlockType.CONVOLUTION_3X3, 4)
    ]
    losses = [0.1, 0.01]
    blocks_to_use = [blocks.BlockType.CONVOLUTION_3X3]

    # Make sure the model suggestion includes a flatten block,
    # despite raw model output being all convolutional.
    best = _get_suggestion(
        architectures, blocks_to_use, losses, grow=grow, pass_flatten=True)
    flattens = [b for b in best if "FLATTEN" in blocks.BlockType(b).name]
    nflat = len(flattens)
    self.assertGreater(nflat, 0)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
