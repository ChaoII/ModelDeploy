# Vendored WeTextProcessing (wenet-e2e) — builds the static `md_wetext` library
# containing only the runtime pieces needed by the ModelDeploy ITN backend
# (Apache-2.0, see LICENSE). The upstream Python tn/itn tooling, bin tools and
# the gflags-dependent flags header are not vendored here; only the processor /
# string sources and the glog stub required to compile them.

cmake_minimum_required(VERSION 3.14)
project(md_we_text_processing LANGUAGES CXX)

# 与 openfst 同理：wetext 运行时为旧代码且依赖 OpenFst 头，需 C++17
# （std::allocator::pointer/rebind、shared_ptr::unique 等已被 C++20 移除）。
# 强制本子工程 C++17，SDK 其余部分保持 C++20 不受影响。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(md_wetext STATIC
    runtime/processor/wetext_processor.cc
    runtime/processor/wetext_token_parser.cc
    runtime/utils/wetext_string.cc
)

# "processor/..." and "utils/..." resolve relative to runtime/; the glog stub
# provides <glog/logging.h> without a real glog dependency.
target_include_directories(md_wetext SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/glog_stub/include
)

# OpenFst supplies <fst/fstlib.h> (and fst/config.h) transitively via the `fst`
# target built from ../../third_party/openfst.
target_link_libraries(md_wetext PUBLIC fst)
