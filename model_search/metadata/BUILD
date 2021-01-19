load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Handling Phoenix MetaData!

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "metadata",
    srcs = ["metadata.py"],
    srcs_version = "PY3",
    deps = [":trial"],
)

py_library(
    name = "ml_metadata_db",
    srcs = ["ml_metadata_db.py"],
    srcs_version = "PY3",
    deps = [
        ":metadata",
        ":trial",
        "//model_search/generators:trial_utils",
    ],
)

py_test(
    name = "ml_metadata_db_test",
    srcs = ["ml_metadata_db_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ml_metadata_db",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "mock_metadata",
    srcs = ["mock_metadata.py"],
    srcs_version = "PY3",
    deps = [
        ":metadata",
    ],
)

py_library(
    name = "trial",
    srcs = ["trial.py"],
    srcs_version = "PY3",
)

py_test(
    name = "trial_test",
    srcs = ["trial_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":trial",
        "@absl_py//absl/testing:absltest",
    ],
)
