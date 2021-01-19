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
"""Tests for model_search.registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from model_search import registry


class Base(object):
  pass


class ValidSub(Base):
  pass


class InvalidSub(object):
  pass


class DecoratorBase(object):
  pass


@registry.register(DecoratorBase)
class DecoratorSub(DecoratorBase):
  pass


@registry.register(DecoratorBase, lookup_name="alias")
class DecoratorAlias(DecoratorBase):
  pass


def base_method(foo, bar, baz):
  return foo + bar + baz


def valid_sub_method(foo, bar, baz):
  return foo * bar * baz


def invalid_sub_method(foo, bar):
  return foo * bar


@registry.register(base_method)
def valid_decorated_sub_method(foo, bar, baz):
  return foo * bar * baz


class RegisterTest(absltest.TestCase):

  def testRegister(self):
    sub_expected = registry.register(Base)(ValidSub)
    sub_registered = registry.lookup("ValidSub", Base)
    self.assertEqual(sub_registered, sub_expected)

    alias_expected = registry.register(Base, lookup_name="alias")(ValidSub)
    alias_registered = registry.lookup("alias", Base)
    self.assertEqual(alias_registered, alias_expected)

    sub_method_expected = registry.register(base_method)(valid_sub_method)
    sub_method_registered = registry.lookup("valid_sub_method", base_method)
    self.assertEqual(sub_method_registered, sub_method_expected)

    self.assertRaises(ValueError, registry.register(Base), InvalidSub)
    self.assertRaises(ValueError, registry.register(base_method),
                      invalid_sub_method)

    # Cannot register the same class / method twice.
    self.assertRaises(ValueError, registry.register(Base), ValidSub)
    self.assertRaises(ValueError, registry.register(base_method),
                      valid_sub_method)

    # Both base and the subclass / method should be of the right type.
    self.assertRaises(ValueError, registry.register, "foo")
    self.assertRaises(ValueError, registry.register(Base), "foo")
    self.assertRaises(ValueError, registry.register(base_method), "foo")

  def testDecorator(self):
    sub_registered = registry.lookup("DecoratorSub", DecoratorBase)
    self.assertEqual(sub_registered, DecoratorSub)

    alias_registered = registry.lookup("alias", DecoratorBase)
    self.assertEqual(alias_registered, DecoratorAlias)

    sub_method_registered = registry.lookup("valid_decorated_sub_method",
                                            base_method)
    self.assertEqual(sub_method_registered, valid_decorated_sub_method)

    invalid_base = registry.lookup("InvalidSub", InvalidSub)
    self.assertIsNone(invalid_base)

    invalid_sub = registry.lookup("InvalidSub", DecoratorBase)
    self.assertIsNone(invalid_sub)

  def test_lookup_all(self):
    sub_registered = registry.lookup_all(DecoratorBase)
    self.assertLen(sub_registered, 2)


if __name__ == "__main__":
  absltest.main()
