load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "architecture_utils",
    srcs = ["architecture_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:block",
        "//model_search:block_builder",
        "//model_search:hparam",
        "//model_search:utils",
    ],
)

py_library(
    name = "tower",
    srcs = ["tower.py"],
    srcs_version = "PY3",
    deps = [
        ":architecture_utils",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:block",
        "//model_search:block_builder",
        "//model_search:hparam",
        "//model_search:utils",
    ],
)

py_test(
    name = "architecture_utils_test",
    srcs = ["architecture_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":architecture_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:block_builder",
        "//model_search:hparam",
        "//model_search/metadata:trial",
    ],
)

# py_strict_test(
#     name = "graph_architecture_test",
#     srcs = ["graph_architecture_test.py"],
#     data = [
#         "//model_search/testdata",
#     ],
#     python_version = "PY3",
#     srcs_version = "PY3",
#     tags = [
#         "notap",
#     ],
#     deps = [
#         ":architecture_utils",
#         ":graph_architecture",
#
#         "@absl_py//absl/testing:parameterized",
#         "//model_search/proto:phoenix_spec_py_pb2",
#         "//third_party/py/numpy",
#         "//third_party/py/tensorflow",
#     ],
# )

py_library(
    name = "graph_architecture",
    srcs = ["graph_architecture.py"],
    srcs_version = "PY3",
    deps = [
        ":architecture_utils",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:block_builder",
        "//model_search:utils",
    ],
)
