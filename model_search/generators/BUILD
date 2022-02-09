load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "base_tower_generator",
    srcs = ["base_tower_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search/architecture:architecture_utils",
        "//model_search/architecture:tower",
    ],
)

py_library(
    name = "prior_generator",
    srcs = ["prior_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":base_tower_generator",
        ":trial_utils",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "prior_generator_test",
    srcs = ["prior_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":prior_generator",
        ":trial_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/architecture:architecture_utils",
        "//model_search/metadata:ml_metadata_db",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "replay_generator",
    srcs = ["replay_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":base_tower_generator",
        ":trial_utils",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "replay_generator_test",
    srcs = ["replay_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":replay_generator",
        ":trial_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search/metadata:ml_metadata_db",
    ],
)

py_library(
    name = "trial_utils",
    srcs = ["trial_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:hparam",
        "//model_search/architecture:architecture_utils",
        "//model_search/architecture:tower",
        "//model_search/generators:base_tower_generator",
        "//model_search/metadata:trial",
    ],
)

py_test(
    name = "trial_utils_test",
    srcs = ["trial_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":trial_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/architecture:architecture_utils",
    ],
)

py_library(
    name = "search_candidate_generator",
    srcs = ["search_candidate_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":base_tower_generator",
        ":trial_utils",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:block_builder",
        "//model_search/architecture:architecture_utils",
        "//model_search/architecture:tower",
        "//model_search/search",
    ],
)

py_test(
    name = "search_candidate_generator_test",
    size = "large",
    srcs = ["search_candidate_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":search_candidate_generator",
        ":trial_utils",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:hparam",
        "//model_search/architecture:architecture_utils",
        "//model_search/metadata:ml_metadata_db",
        "//model_search/metadata:trial",
    ],
)
