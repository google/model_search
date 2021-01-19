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
"""Utilities for class registration.

Defines utility functions to register and look up subclasses of a base
class. See the comments in each function for more detail.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import traceback

from absl import logging

_LOCATION_TAG = "location"
_TYPE_TAG = "type"
_INIT_ARGS = "init_args"
_ENUM_ID = "enum_id"

_registries = {}


def register(base, init_args=None, lookup_name=None, enum_id=None):
  """Produces a decorator to register a subclass of a base class, or a method.

  This produces a decorator to register a subclass of a base class, or a method
  which implements the same signature as the given base method.

  For example:

  Example 1:
    class Base(object):
      ...

    @register(Base)
    class Sub(Base):
      ...

  Example 2:
    def base_foo(bar, baz):
      ...

    @register(base_foo)
    def foo(bar, baz):
      ...

  Args:
    base: A base class of the subclass to be registered, or the method whose
      signature will be copied by the methods to be registered.
    init_args: For a subclass, gives the ability to provide args for the
      __init__ function. If provided, the lookup will return an instance of the
      subclass, rather than the subclass (type).
    lookup_name: The name for lookup. If not provided, the name for lookup is
      cls_or_method.__name__. This gives the user ability to register the same
      subclass with different init_args under different names.
    enum_id: An integer. This helps creating an enum from lookup name to id for
      all subclasses / methods that belong to one base.

  Returns:
    A decorator.

  Raises:
    ValueError: If base is neither a class, nor a method.
  """
  # base should either ber a class or a function.
  if not (inspect.isclass(base) or inspect.isfunction(base)):
    raise ValueError("%s should either be a class or a function. Was: %s" %
                     (base, type(base)))

  basename = base.__name__
  if basename not in _registries:
    _registries[basename] = {}
  registry = _registries[basename]

  def _register_decorator(cls_or_method):
    """A decorator to register a subclass of the base class or a method.

    Args:
      cls_or_method: A subclass or a method to be registered.

    Raises:
      ValueError: If any of the following rules are violated, the method will
        raise a ValueError:
          (1) cls_or_method should either be a class or a method.
          (2) If cls_or_method is a class, then it should be a subclass of base.
          (3) If cls_or_method is a method, then it should have the same
            signature as base.
          (4) cls_or_method is already registered.
          (5) if init_args is provided for a method.

    Returns:
      cls_or_method itself.
    """

    if inspect.isclass(cls_or_method):
      if not issubclass(cls_or_method, base):
        raise ValueError("%s is not a sublcass of %s" % (cls_or_method, base))
    elif inspect.isfunction(cls_or_method):
      # pylint: disable=deprecated-method
      if (len(inspect.getargspec(cls_or_method).args) != len(
          inspect.getargspec(base).args)):
        # pylint: enable=deprecated-method
        raise ValueError("%s does not have the same number of arguments as %s" %
                         (cls_or_method, base))
      if init_args:
        raise ValueError("init_args is provided for a method.")
    else:
      raise ValueError("%s should either be a class or a function. Was: %s" %
                       (cls_or_method, type(cls_or_method)))

    name = lookup_name if lookup_name else cls_or_method.__name__

    if name in registry:
      (filename, line_number, function_name, _) = registry[name][_LOCATION_TAG]
      raise ValueError("Registering two %s with name '%s' !"
                       "(Previous registration was in %s %s:%d)" %
                       (basename, name, function_name, filename, line_number))

    logging.vlog(1,
                 "Registering %s (%s) as %s." % (name, cls_or_method, basename))

    # stack trace is [this_function, user_function,...]
    # so the user function is #1.
    stack = traceback.extract_stack()
    registry[name] = {
        _TYPE_TAG: cls_or_method,
        _LOCATION_TAG: stack[1],
        _INIT_ARGS: init_args,
        _ENUM_ID: enum_id
    }

    return cls_or_method

  return _register_decorator


def lookup(name, base):
  """Looks up a subclass of a base class from the registry.

  Looks up a subclass of a base class with name provided from the
  registry. Returns the registered subclass if found, None otherwise.

  Args:
    name: Name to look up from the registry.
    base: The base class of the subclass to be found.

  Returns:
    Subclass of the name if found, None otherwise.
  """
  basename = base.__name__
  if basename not in _registries:
    return None
  registry = _registries[basename]
  if name not in registry:
    return None
  init_args = registry[name][_INIT_ARGS]
  if init_args is not None:
    return registry[name][_TYPE_TAG](**init_args)
  return registry[name][_TYPE_TAG]


def lookup_all(base):
  """Looks up a subclass of a base class from the registry.

  Looks up a subclass of a base class with name provided from the
  registry. Returns a list of registered subclass if found, None otherwise.

  Args:
    base: The base class of the subclass to be found.

  Returns:
    A list of subclass of the name if found, None otherwise.
  """
  basename = base.__name__
  if basename not in _registries:
    return None
  registry = _registries[basename]
  output = []
  for name in registry.keys():
    init_args = registry[name][_INIT_ARGS]
    if init_args is not None:
      output.append(registry[name][_TYPE_TAG](**init_args))
    else:
      output.append(registry[name][_TYPE_TAG])
  return output


def get_base_enum(base):
  basename = base.__name__
  if basename not in _registries:
    return None
  registry = _registries[basename]
  output = {}
  for name, registered_instance in registry.items():
    output.update({name: registered_instance[_ENUM_ID]})
  return output
