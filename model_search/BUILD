"""Model Search is an ML framework to automatically find an ML model for you problem."""

load("@rules_python//python:defs.bzl", "py_library", "py_test")

# TODO(b/135991181): Revisit the use of pytype library once Tensorflow's
# type inference issues are resolved.



package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

exports_files(
    srcs = [
        "oss_trainer.py",
    ],
    visibility = [
        "//visibility:public",
    ],
)

exports_files(
    srcs = [
        "oss_trainer_test.py",
    ],
    visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "blocks",
    srcs = ["blocks.py"],
    srcs_version = "PY3",
    deps = [
        ":hparam",
        ":registry",
        "//model_search/ops:svdf_cell",
        "//model_search/ops:svdf_conv",
    ],
)

py_library(
    name = "blocks_builder",
    srcs = ["blocks_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":blocks",
        ":registry",
        "//model_search/hparams:hyperparameters",
    ],
)

py_test(
    name = "blocks_builder_test",
    srcs = ["blocks_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":blocks_builder",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "blocks_test",
    size = "large",
    srcs = ["blocks_test.py"],
    python_version = "PY3",
    shard_count = 15,
    srcs_version = "PY3",
    deps = [
        ":blocks",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/architecture:architecture_utils",
    ],
)

py_library(
    name = "controller",
    srcs = ["controller.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search/architecture:architecture_utils",
        "//model_search/generators:base_tower_generator",
        "//model_search/generators:prior_generator",
        "//model_search/generators:replay_generator",
        "//model_search/generators:search_candidate_generator",
        "//model_search/generators:trial_utils",
    ],
)

py_test(
    name = "controller_test",
    srcs = ["controller_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":controller",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search/metadata:ml_metadata_db",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY3",
)

py_library(
    name = "single_trainer",
    srcs = ["single_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":oss_trainer_lib",
        ":phoenix",
        "//model_search/proto:all_proto_py_pb2",
    ],
)

py_test(
    name = "single_trainer_test",
    srcs = ["single_trainer_test.py"],
    data = [
        "//model_search/configs:phoenix_configs",
        "//model_search/data/testdata:csv_random_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":single_trainer",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search/data:csv_data",
    ],
)

py_library(
    name = "hparam",
    srcs = ["hparam.py"],
    srcs_version = "PY3",
    deps = ["//model_search/proto:all_proto_py_pb2"],
)

py_library(
    name = "logit_bundler",
    srcs = ["logit_bundler.py"],
    srcs_version = "PY3",
)

py_library(
    name = "ensembler",
    srcs = ["ensembler.py"],
    srcs_version = "PY3",
    deps = [
        ":logit_bundler",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search/architecture:architecture_utils",
        "//model_search/generators:trial_utils",
    ],
)

py_test(
    name = "ensembler_test",
    srcs = ["ensembler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ensembler",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search/architecture:architecture_utils",
    ],
)

py_library(
    name = "loss_fns",
    srcs = ["loss_fns.py"],
    srcs_version = "PY3",
)

py_test(
    name = "loss_fns_test",
    srcs = ["loss_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":loss_fns",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "metric_fns",
    srcs = ["metric_fns.py"],
    srcs_version = "PY3",
)

py_test(
    name = "metric_fns_test",
    srcs = ["metric_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metric_fns",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "phoenix",
    srcs = ["phoenix.py"],
    srcs_version = "PY3",
    deps = [
        ":controller",
        ":ensembler",
        ":hparam",
        ":loss_fns",
        ":metric_fns",
        ":task_manager",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search/architecture:architecture_utils",
        "//model_search/generators:base_tower_generator",
        "//model_search/generators:prior_generator",
        "//model_search/generators:search_candidate_generator",
        "//model_search/generators:trial_utils",
        "//model_search/meta:distillation",
        "//model_search/meta:transfer_learning",
        "//model_search/metadata:ml_metadata_db",
    ],
)

py_test(
    name = "phoenix_test",
    size = "large",
    srcs = ["phoenix_test.py"],
    data = [
        "//model_search/configs:phoenix_configs",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hparam",
        ":loss_fns",
        ":phoenix",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search/architecture:architecture_utils",
        "//model_search/metadata:mock_metadata",
    ],
)

py_library(
    name = "registry",
    srcs = [
        "registry.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "registry_test",
    srcs = [
        "registry_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":registry",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "task_manager",
    srcs = ["task_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":blocks_builder",
        ":loss_fns",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "task_manager_test",
    srcs = ["task_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hparam",
        ":loss_fns",
        ":task_manager",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search/architecture:architecture_utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils"],
)

py_library(
    name = "oss_trainer_lib",
    srcs = ["oss_trainer_lib.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":hparam",
        ":loss_fns",
        ":metric_fns",
        ":phoenix",
        ":registry",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search/data",
        "//model_search/metadata:ml_metadata_db",
    ],
)
