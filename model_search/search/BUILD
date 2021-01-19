load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//learning/adanets:__subpackages__",
    "//model_search:__subpackages__",
])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "search",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":categorical_harmonica",
        ":constrained_descent",
        ":coordinate_descent",
        ":harmonica",
        ":identity",
        ":linear_model",
        ":search_algorithm",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    deps = [
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":common",
        ":test_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "categorical_harmonica",
    srcs = ["categorical_harmonica.py"],
    srcs_version = "PY3",
    deps = [
        ":search_algorithm",
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "categorical_harmonica_test",
    srcs = ["categorical_harmonica_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":categorical_harmonica",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "coordinate_descent",
    srcs = ["coordinate_descent.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":search_algorithm",
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "coordinate_descent_test",
    srcs = ["coordinate_descent_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":coordinate_descent",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:blocks_builder",
        "//model_search:hparam",
        "//model_search/metadata:ml_metadata_db",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "linear_model",
    srcs = ["linear_model.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":search_algorithm",
        "//model_search/proto:all_proto_py_pb2",
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "linear_model_test",
    size = "small",
    timeout = "short",
    srcs = ["linear_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linear_model",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:blocks_builder",
        "//model_search:hparam",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "constrained_descent",
    srcs = ["constrained_descent.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":search_algorithm",
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "constrained_descent_test",
    srcs = ["constrained_descent_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constrained_descent",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/metadata:ml_metadata_db",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "harmonica",
    srcs = ["harmonica.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":search_algorithm",
        "//model_search:blocks_builder",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "harmonica_test",
    srcs = ["harmonica_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":harmonica",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
        "//model_search/metadata:trial",
    ],
)

py_library(
    name = "identity",
    srcs = ["identity.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":search_algorithm",
        "//model_search:blocks",
        "//model_search/architecture:architecture_utils",
    ],
)

py_test(
    name = "identity_test",
    srcs = ["identity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":identity",
        ":test_utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "//model_search:hparam",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = ["//model_search/proto:all_proto_py_pb2"],
)

py_library(
    name = "search_algorithm",
    srcs = ["search_algorithm.py"],
    srcs_version = "PY3",
)
