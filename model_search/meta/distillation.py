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
"""Distill an ensemble into a smaller model.

Functions in this file implement network distillation in Phoenix. Distillation
aims to transfer the predictive power of a larger ensemble into a smaller model
by predicting the logits of the larger model, rather than the original labels.
Currently, ensemble search and distillation will not take place during the same
run. In the future, we will support doing both on the same run but not on the
same trial.
Design Doc: go/phx-distillation
"""

import collections
import functools

from model_search import logit_bundler
from model_search.architecture import architecture_utils
from model_search.proto import distillation_spec_pb2
import tensorflow.compat.v2 as tf

DistillationLogits = collections.namedtuple(
    "DistillationLogits",
    ["train_logits_specs", "eval_logits_spec", "teacher_logits_spec"])


def get_distillation_loss_fn(teacher_logits, distillation_spec, my_id,
                             original_loss_fn):
  """Force the loss_fn to compare the student logits to the teacher logits."""

  # The logits input below is the student logits
  def _mse_teacher_loss_fn(labels, logits, weights=1.0):
    """A mean square error with the teacher's logits/predictions."""
    del labels  # Unused.

    return tf.compat.v1.losses.mean_squared_error(
        labels=teacher_logits, predictions=logits, weights=weights)

  def _cross_entropy_loss_fn(labels, logits, weights=1.0):
    """A cross entropy with the teacher's predictions."""
    del labels  # Unused.

    return tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=teacher_logits, logits=logits, weights=weights)

  # TODO(b/172564129): Once b/148613464 is in place, add * to force parameter
  # names.
  def _adaptively_balance_losses_loss_fn(loss1_fn,
                                         loss2_fn,
                                         labels,
                                         logits,
                                         weights=1.0):
    """Increasingly grow the distillation loss over time."""
    original_loss = loss1_fn(labels=labels, logits=logits, weights=weights)
    distill_loss = loss2_fn(labels=labels, logits=logits, weights=weights)

    lambda_ = distillation_spec.balance_losses_spec.minimum_lambda
    delta = (
        distillation_spec.balance_losses_spec.maximum_lambda -
        distillation_spec.balance_losses_spec.minimum_lambda)
    max_trials = distillation_spec.balance_losses_spec.num_ramp_trials

    if max_trials > 0:
      # Linearly increase the lambda toward the maximum
      lambda_ += delta * (
          max(0, my_id - distillation_spec.minimal_pool_size) / max_trials)

    return original_loss + lambda_ * distill_loss

  if (distillation_spec.distillation_type ==
      distillation_spec_pb2.DistillationSpec.MSE_LOGITS or
      distillation_spec.distillation_type ==
      distillation_spec_pb2.DistillationSpec.MSE_SOFTMAX):
    return _mse_teacher_loss_fn
  elif (distillation_spec.distillation_type ==
        distillation_spec_pb2.DistillationSpec.CROSS_ENTROPY):
    return _cross_entropy_loss_fn
  elif (distillation_spec.distillation_type ==
        distillation_spec_pb2.DistillationSpec.ADAPTIVELY_BALANCE_LOSSES):
    return functools.partial(
        _adaptively_balance_losses_loss_fn,
        loss1_fn=original_loss_fn,
        loss2_fn=_cross_entropy_loss_fn)

  else:
    raise ValueError("unsupported distillation_type.")


class Distiller(logit_bundler.LogitBundler):
  """Bundles together logits correctly for distillation."""

  def __init__(self, distillation_spec):
    """Initializes a new Distiller object.

    Args:
      distillation_spec: The spec defined in the Phoenix spec.
    """
    self._distillation_spec = distillation_spec

  def bundle_logits(self, priors_logits_specs, search_logits_specs):
    """Bundles the priors and the search candidate."""

    assert search_logits_specs, "Cannot distill with no student model."
    assert len(search_logits_specs) == 1, "Search has more than one tower."

    if not priors_logits_specs:
      return DistillationLogits(
          train_logits_specs=search_logits_specs,
          eval_logits_spec=search_logits_specs[0],
          teacher_logits_spec=None)

    with tf.compat.v1.variable_scope("Phoenix/Distiller"):
      priors_logits = tf.add_n(
          [tf.stop_gradient(spec.logits) for spec in priors_logits_specs])

      assert self._distillation_spec.distillation_type, (
          "Invalid DistillationType specified.")
      if (self._distillation_spec.distillation_type ==
          distillation_spec_pb2.DistillationSpec.DistillationType.MSE_LOGITS):
        transformed_logits = priors_logits
      else:
        transformed_logits = tf.nn.softmax(priors_logits /
                                           self._distillation_spec.temperature)

      transformed_logits_specs = architecture_utils.LogitsSpec(
          logits=transformed_logits)

      # Use the logits from the student model (search) to train and evaluate,
      # but store the logits from the teacher model (combined priors) to
      # calculate the loss.
      return DistillationLogits(
          train_logits_specs=search_logits_specs,
          eval_logits_spec=search_logits_specs[0],
          teacher_logits_spec=transformed_logits_specs)
