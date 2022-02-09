package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "phoenix_spec_proto",
    srcs = ["phoenix_spec.proto"],
    deps = [
        ":hparam_proto",
        "//model_search/proto:distillation_spec_proto",
        "//model_search/proto:ensembling_spec_proto",
        "//model_search/proto:transfer_learning_spec_proto",
    ],
)

proto_library(
    name = "hparam_proto",
    srcs = ["hparam.proto"],
)

proto_library(
    name = "ensembling_spec_proto",
    srcs = ["ensembling_spec.proto"],
)

proto_library(
    name = "distillation_spec_proto",
    srcs = ["distillation_spec.proto"],
)

proto_library(
    name = "transfer_learning_spec_proto",
    srcs = ["transfer_learning_spec.proto"],
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
py_proto_library(
    name = "all_proto_py_pb2",
    srcs = [
        "distillation_spec.proto",
        "ensembling_spec.proto",
        "hparam.proto",
        "phoenix_spec.proto",
        "transfer_learning_spec.proto",
    ],
        deps = ["@com_google_protobuf//:protobuf_python"],
)
