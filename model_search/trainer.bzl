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

"""A trainer blaze binary creator."""

load("@bazel_skylib//rules:write_file.bzl", "write_file")

def parse_label(label):
    """Parse a label into (package, name).

    Args:
      label: string in relative or absolute form.

    Returns:
      Pair of strings: package, relative_name

    Raises:
      ValueError for malformed label (does not do an exhaustive validation)
    """
    if label.startswith("//"):
        label = label[2:]  # drop the leading //
        colon_split = label.split(":")
        if len(colon_split) == 1:  # no ":" in label
            pkg = label
            _, _, target = label.rpartition("/")
        else:
            pkg, target = colon_split  # fails if len(colon_split) != 2
    else:
        colon_split = label.split(":")
        if len(colon_split) == 1:  # no ":" in label
            pkg, target = native.package_name(), label
        else:
            pkg2, target = colon_split  # fails if len(colon_split) != 2
            pkg = native.package_name() + ("/" + pkg2 if pkg2 else "")
    return pkg, target

def convert_to_py_module(package_name):
    """Creates a python module name to import.

    Args:
      package_name: string - the package name in the build file

    Returns:
      a string holding the corresponding python module name.
    """
    if ":" in package_name:
        output = ("%s.%s" % parse_label(package_name)).replace("/", ".")
    else:
        pkg, _ = parse_label(package_name)
        output = ("%s" % pkg).replace("/", ".")
    if output.startswith("third_party.py."):
        output = output[len("third_party.py."):]
    return output

def internal_model_search_binary(
        name,
        dataset_dep = None,
        trainer_src = "//model_search:trainer.py",
        trainer_lib = "//model_search:trainer_lib",
        **kwargs):
    """Defines the binary rule for model search.

    Args:
      name: name of the final py_binary.
      dataset_dep: the dataset dependency as a data provider.
      trainer_src: The trainer py file (source).
      trainer_lib: The trainer library (package).
      **kwargs: kwargs.
    """
    if not dataset_dep:
        fail("Please specify the dataset dependency")
    if type(dataset_dep) == list or type(dataset_dep) == tuple:
        fail("Please provide the dataset_dep as one package:" + type(dataset_dep))
    tf2_disable_deps = []
    envelope_deps = []

    native.genrule(
        name = "get_trainer_main_{}".format(name),
        srcs = [trainer_src],
        outs = ["trainer_main_{}.py".format(name)],
        cmd = "echo \"import {}\n\" >> $@ && cat $< >> $@".format(convert_to_py_module(dataset_dep)),
    )
    write_file(
        name = "write_dataset_name_{}".format(name),
        content = [convert_to_py_module(dataset_dep)],
        out = "model_search_dataset_name_{}".format(name),
    )
    native.filegroup(
        name = "model_search_file_{}".format(name),
        srcs = [
            ":write_dataset_name_{}".format(name),
        ],
    )
    native.py_binary(
        name = name,
        srcs = [":get_trainer_main_{}".format(name)],
        data = [":model_search_file_{}".format(name)],
        main = "trainer_main_{}.py".format(name),
        deps = [
            trainer_lib,
            
        ] + [dataset_dep] + tf2_disable_deps + envelope_deps,
        srcs_version = "PY3",
        python_version = "PY3",
        **kwargs
    )

def internal_model_search_test(
        name,
        dataset_dep = None,
        problem_type = "cnn",
        extra_args = None,
        test_data = None,
        custom_spec = None,
        test_trainer_src = None,
        test_trainer_lib = None,
        open_source = False,
        **kwargs):
    """Integration test to test the provider locally.

    Args:
      name: a string holding the name of the test.
      dataset_dep: data_provider library.
      problem_type: a string. Either: cnn, dnn, rnn_all or rnn_last
      extra_args: A list of flags to provide the test. Example:
        [--filepattern=/cns/od-d/my_dummy_test_recordid]
      test_data: A list of data packages for the test.
      custom_spec: A path to a custom spec. Note, that if supplied then the
        user needs to add the file as "extra_data" dependency.
      test_trainer_src: The source file for the test trainer.
      test_trainer_lib: The library package for the test trainer.
      open_source: A boolean indicated if this is an open source binary.
      **kwargs: Addition keyvalue arguments.
    """
    native.genrule(
        name = "get_trainer_test_{}".format(name),
        srcs = [test_trainer_src],
        outs = ["trainer_test_{}.py".format(name)],
        cmd = "echo \"import {}\n\" >> $@ && cat $< >> $@".format(convert_to_py_module(dataset_dep)),
    )
    tf2_disable_deps = []
    envelope_deps = []

    user_test_flags = extra_args or []
    user_data = test_data or []
    spec = custom_spec or "model_search/configs/" + problem_type + "_config.pbtxt"
    flags_prefix = "study"
    if open_source:
        flags_prefix = "experiment"
    native.py_test(
        name = name,
        size = "large",
        srcs = [":get_trainer_test_{}".format(name)],
        main = "trainer_test_{}.py".format(name),
        srcs_version = "PY3",
        python_version = "PY3",
        data = [
            "//model_search/configs:phoenix_configs",
        ] + user_data,
        args = [
            "--phoenix_spec_filename={}".format(spec),
            "--phoenix_batch_size=2",
            "--phoenix_train_steps=20",
            "--phoenix_eval_steps=20",
            "--{}_owner=test".format(flags_prefix),
            "--{}_name=test".format(flags_prefix),
            "--{}_max_num_trials=1".format(flags_prefix),
            "--model_dir=/tmp/" + name,
        ] + user_test_flags,
        deps = [
            test_trainer_lib,
            
            
            "//model_search/proto:all_proto_py_pb2",
            
        ] + [dataset_dep] + tf2_disable_deps + envelope_deps,
        **kwargs
    )

def model_search_oss_binary(
        name,
        dataset_dep = None,
        trainer_src = "//model_search:oss_trainer.py",
        trainer_lib = "//model_search:oss_trainer_lib",
        **kwargs):
    internal_model_search_binary(
        name = name,
        dataset_dep = dataset_dep,
        trainer_src = trainer_src,
        trainer_lib = trainer_lib,
        **kwargs
    )

def model_search_oss_test(
        name,
        dataset_dep = None,
        problem_type = "cnn",
        extra_args = None,
        test_data = None,
        custom_spec = None,
        **kwargs):
    internal_model_search_test(
        name = name,
        dataset_dep = dataset_dep,
        problem_type = problem_type,
        extra_args = extra_args,
        test_data = test_data,
        custom_spec = custom_spec,
        test_trainer_src = "//model_search:oss_trainer_test.py",
        test_trainer_lib = "//model_search:oss_trainer_lib",
        open_source = True,
        **kwargs
    )
