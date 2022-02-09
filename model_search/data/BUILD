load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//model_search:trainer.bzl", "model_search_oss_binary", "model_search_oss_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "--filename=$${TEST_SRCDIR}/model_search/model_search/data/testdata/csv_random_data.csv",
        "--validation_filename=$${TEST_SRCDIR}/model_search/model_search/data/testdata/csv_random_data.csv",
    ],
    problem_type = "dnn",
    test_data = [
        "//model_search/data/testdata:csv_random_data",
    ],
)

model_search_oss_test(
    name = "csv_data_for_binary_test_estimator",
    dataset_dep = ":csv_data_for_binary",
    extra_args = [
        "--filename=$${TEST_SRCDIR}/model_search/model_search/data/testdata/csv_random_data.csv",
        "--validation_filename=$${TEST_SRCDIR}/model_search/model_search/data/testdata/csv_random_data.csv",
        "--force_estimator=True",
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
    name = "image_data",
    srcs = ["image_data.py"],
    srcs_version = "PY3",
    deps = [":data"],
)

py_library(
    name = "image_data_for_binary",
    srcs = ["image_data_for_binary.py"],
    srcs_version = "PY3",
    deps = [":data"],
)

py_test(
    name = "image_data_test",
    srcs = ["image_data_test.py"],
    data = [
        "//model_search/configs:phoenix_configs",
        "//model_search/data/testdata:image_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":image_data",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//model_search:constants",
        "//model_search:single_trainer",
    ],
)

model_search_oss_test(
    name = "image_data_for_binary_test",
    dataset_dep = ":image_data_for_binary",
    extra_args = [
        "--input_dir=$${TEST_SRCDIR}/model_search/model_search/data/testdata/images",
        "--image_height=100",
        "--image_width=100",
    ],
    problem_type = "cnn",
    test_data = [
        "//model_search/data/testdata:image_data",
    ],
)

model_search_oss_test(
    name = "image_data_for_binary_test_estimator",
    dataset_dep = ":image_data_for_binary",
    extra_args = [
        "--input_dir=$${TEST_SRCDIR}/model_search/model_search/data/testdata/images",
        "--image_height=100",
        "--image_width=100",
        "--force_estimator=True",
    ],
    problem_type = "cnn",
    test_data = [
        "//model_search/data/testdata:image_data",
    ],
)

model_search_oss_binary(
    name = "image_data_binary",
    dataset_dep = ":image_data_for_binary",
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
