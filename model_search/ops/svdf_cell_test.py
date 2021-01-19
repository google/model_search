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

import collections

from absl import logging
from model_search.ops import svdf_cell

import numpy as np
import tensorflow.compat.v2 as tf


class SvdfCellTest(tf.test.TestCase):

  def _collectOutputsAndGradients(self, sess, inputs, activations):
    trainable_variables = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    gradients = tf.gradients(
        ys=[activations], xs=[inputs] + trainable_variables)
    variable_names = [v.name for v in trainable_variables]
    names = ["activations", "input_grad"] + variable_names
    values = sess.run([activations] + gradients)
    return collections.OrderedDict(list(zip(names, values)))

  def _buildInitializer(self):
    return tf.compat.v1.random_uniform_initializer(
        minval=-0.4, maxval=0.4, seed=0)

  def _runSvdf(self, inputs, num_units, memory_size, rank, use_bias,
               activation):
    # Force graph mode
    g = tf.Graph()
    with g.as_default():
      with self.session(graph=g) as sess:
        initializer = self._buildInitializer()
        act_func = None
        if activation != "linear":
          act_func = getattr(tf.nn, activation)
        svdf_layer = svdf_cell.SvdfCell(
            num_units=num_units,
            memory_size=memory_size,
            rank=rank,
            use_bias=use_bias,
            activation=act_func,
            feature_weights_initializer=initializer,
            time_weights_initializer=initializer,
            image_summary=True)

        state = svdf_layer.zero_state(
            batch_size=inputs.shape[1], dtype=tf.float32)
        i = 0
        for input_t in inputs:
          logging.debug("state[%d]:\n %s", i, state.eval())
          input_t_constant = tf.constant(input_t)
          activations, state = svdf_layer(inputs=input_t_constant, state=state)
          if i == 0:
            # Initialize only the first iteration.
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
          outputs = self._collectOutputsAndGradients(sess, input_t_constant,
                                                     activations)
          i += 1
        return outputs

  def _runTest(self, num_units, memory_size, rank, inputs, use_bias, activation,
               expected):
    """Executes SVDF test run.

    Checks svdf produces the same outputs (activations/gradients) and
    parameters.

    Arguments:
      num_units: int or long, the number of units in the layer.
      memory_size: int or long, the size of the SVDF cell memory.
      rank: int or long, the rank of the SVD approximation.
      inputs: the input Tensor to use in the run.
      use_bias: bool, whether the layer uses a bias.
      activation: callable, the activation function to use.
      expected: dictionary with the expected variable name as key, and value.

    """
    # _runSvdf will provide the outputs after iterating over as many inputs
    # there are in time; thus we validate that final state and activations.
    svdf_outputs = self._runSvdf(
        inputs, num_units, memory_size, rank, use_bias, activation)

    for name, value in expected.items():
      svdf_output = np.squeeze(svdf_outputs[name])  # Remove dummy batch dim.
      logging.debug("expected [%s]  value = \n%s", name, value)
      logging.debug("SVDF [%s]  value = \n%s", name, svdf_output)
      self.assertAllClose(value, svdf_output)

  # Tests
  def testRank1(self):
    # Create list of time_dim inputs of shape [batch_size, frame_size].
    # time_dim = 10
    # batch_size = 2
    # frame_size = 3
    # NOTE: this is equivalent to the svdf_test.py setup, and shows that the
    # SvdfCell is equivalent to the non-recurrent Svdf when doing both forward
    # and backward passes on the same data, with the same configuration.
    inputs_list = [
        [
            np.array([[0.12609188, -0.46347019, -0.89598465],
                      [0.35867718, 0.36897406, 0.73463392]],
                     dtype=np.float32)],
        [
            np.array([[0.14278367, -1.64410412, -0.75222826],
                      [-0.57290924, 0.12729003, 0.7567004]],
                     dtype=np.float32)],
        [
            np.array([[0.49837467, 0.19278903, 0.26584083],
                      [0.17660543, 0.52949083, -0.77931279]],
                     dtype=np.float32)],
        [
            np.array([[-0.11186574, 0.13164264, -0.05349274],
                      [-0.72674477, -0.5683046, 0.55900657]],
                     dtype=np.float32)],
        [
            np.array([[-0.68892461, 0.37783599, 0.18263303],
                      [-0.63690937, 0.44483393, -0.71817774]],
                     dtype=np.float32)],
        [
            np.array([[-0.81299269, -0.86831826, 1.43940818],
                      [-0.95760226, 1.82078898, 0.71135032]],
                     dtype=np.float32)],
        [
            np.array([[-1.45006323, -0.82251364, -1.69082689],
                      [-1.65087092, -1.89238167, 1.54172635]],
                     dtype=np.float32)],
        [
            np.array([[0.03966608, -0.24936394, -0.77526885],
                      [2.06740379, -1.51439476, 1.43768692]],
                     dtype=np.float32)],
        [
            np.array([[0.11771342, -0.23761693, -0.65898693],
                      [0.31088525, -1.55601168, -0.87661445]],
                     dtype=np.float32)],
        [
            np.array([[-0.89477462, 1.67204106, -0.53235275],
                      [-0.6230064, 0.29819036, 1.06939757]],
                     dtype=np.float32)]]
    expected = {
        "activations": [
            [
                0.36726, -0.522303, -0.456502, -0.175475],
            [
                0.170129, -0.344477, 0.385056, -0.281581]],
        "input_grad": [
            [
                0.179993, 0.115846, 0.169998],
            [
                0.179993, 0.115846, 0.169998]],
        "SvdfCell/SVDF_weights_feature:0": [
            [
                -0.918723, -1.700028, -2.022024, 1.395944],
            [
                -0.667541, -0.83933, -0.461926, -0.542757],
            [
                0.367526, -0.330445, 1.587667, -0.107099]],
        "SvdfCell/SVDF_weights_time:0": [
            [
                -0.121947, 0.685224, -0.480493, 0.429518, 0.121952,
                0.238488, 1.969541, -0.030494, 0.498736, -0.22269],
            [
                0.141328, -0.497931, 0.35046, -0.349594, -0.382569,
                -0.187508, -1.78753, 0.483268, -0.427602, -0.066856],
            [
                0.04620457, -0.53625858, 0.18684755, -0.15690115, -0.35235715,
                0.60837597, -1.67157829, 0.36053753, -0.97437823, 0.32919434],
            [
                -0.15174544, 0.25558627, -0.23719919, 0.27823007, 0.48153371,
                0.34780267, 1.31657171, -0.7149213, 0.12750566, 0.36063662]],
        "SvdfCell/SVDF_bias:0": [2.0, 2.0, 2.0, 2.0]}
    inputs = np.vstack(inputs_list)
    self._runTest(num_units=4,
                  rank=1,
                  memory_size=inputs.shape[0],
                  inputs=inputs,
                  use_bias=True,
                  activation="linear",
                  expected=expected)

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
