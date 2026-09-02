# Vendored OpenFst — builds the static `fst` core library only.
#
# Only the core library sources are vendored here (include/fst/*.h + lib/*.cc,
# Apache-2.0, see LICENSE). The bin / script / extensions / test tooling from
# upstream is intentionally excluded: it is not needed by the fst core library
# and it pulls in extra dependencies. Dynamic-linking registration is disabled
# in include/fst/config.h so no <dlfcn.h> dependency is required on any
# platform (see WeTextProcessing ITN backend).

cmake_minimum_required(VERSION 3.14)
project(md_openfst LANGUAGES CXX)

add_library(fst STATIC
    lib/compat.cc
    lib/flags.cc
    lib/fst-types.cc
    lib/fst.cc
    lib/mapped-file.cc
    lib/properties.cc
    lib/symbol-table-ops.cc
    lib/symbol-table.cc
    lib/util.cc
    lib/weight.cc
)

target_include_directories(fst SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# fst/log.h uses C++ exceptions (used by LOG/CHECK); enable unwind semantics so
# MSVC does not warn C4530 throughout the build.
if (MSVC)
    target_compile_options(fst PRIVATE /EHsc)
endif ()
