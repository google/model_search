load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//model_search:trainer.bzl", "model_search_oss_binary", "model_search_oss_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = ["//model_search/proto:all_proto_py_pb2"],
)

py_library(
    name = "csv_data",
    srcs = ["csv_data.py"],
    srcs_version = "PY3",
    deps = [":data"],
)

py_test(
    name = "csv_data_test",
    srcs = ["csv_data_test.py"],
    data = [
        "//model_search/configs:phoenix_configs",
        "//model_search/data/testdata:csv_random_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":csv_data",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search:constants",
        "//model_search:single_trainer",
    ],
)

py_library(
    name = "csv_data_for_binary",
    srcs = ["csv_data_for_binary.py"],
    srcs_version = "PY3",
    deps = [":data"],
)

model_search_oss_test(
    name = "csv_data_for_binary_test",
    dataset_dep = ":csv_data_for_binary",
    extra_args = [
        "--filename=$${TEST_SRCDIR}/model_search/data/testdata/csv_random_data.csv",
    ],
    problem_type = "dnn",
    test_data = [
        "//model_search/data/testdata:csv_random_data",
    ],
)

model_search_oss_binary(
    name = "csv_data_binary",
    dataset_dep = ":csv_data_for_binary",
)

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY3",
    tags = [
        "ignore_for_dep=third_party.tensorflow.python.feature_column.feature_column",
        "ignore_for_dep=third_party.tensorflow.python.feature_column.feature_column_v2",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":utils",
        "//model_search:registry",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//model_search/proto:all_proto_py_pb2",
        "@absl_py//absl/testing:parameterized",
    ],
)
