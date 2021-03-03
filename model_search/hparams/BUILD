# Util functions for hyperparameters.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    srcs_version = "PY3",
)

py_test(
    name = "hyperparameters_test",
    srcs = ["hyperparameters_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyperparameters",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
