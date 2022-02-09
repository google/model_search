load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "distillation",
    srcs = ["distillation.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:logit_bundler",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "distillation_test",
    srcs = ["distillation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distillation",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "transfer_learning",
    srcs = ["transfer_learning.py"],
    srcs_version = "PY3",
    deps = ["//model_search/architecture:architecture_utils"],
)
