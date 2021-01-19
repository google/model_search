load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//model_search:__subpackages__",
])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "svdf_cell",
    srcs = ["svdf_cell.py"],
    srcs_version = "PY3",
)

py_test(
    name = "svdf_cell_test",
    size = "small",
    srcs = ["svdf_cell_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":svdf_cell"],
)

py_library(
    name = "svdf_conv",
    srcs = ["svdf_conv.py"],
    srcs_version = "PY3",
)

py_test(
    name = "svdf_conv_test",
    size = "small",
    srcs = ["svdf_conv_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":svdf_conv"],
)
