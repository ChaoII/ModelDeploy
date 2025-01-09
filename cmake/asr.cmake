set(TARGET_NAME "funasr")
include(TestBigEndian)
test_big_endian(BIG_ENDIAN)

# for onnxruntime
IF (WIN32)
    add_compile_options(/wd4291 /wd4305 /wd4244 /wd4828 /wd4251 /wd4275)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src/asr/internal)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/json/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/kaldi-native-fbank)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/yaml-cpp/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/jieba/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/jieba/include/limonp/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/kaldi)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/json/include)
include_directories(${ORT_LIB_PATH}/../include)

set(ONNXRUNTIME_DIR ${ORT_LIB_PATH}/..)

# build openfst
# fst depend on glog and gflags
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/glog/src)
set(BUILD_TESTING OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/glog)
include_directories(${glog_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/gflags)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/gflags)
include_directories(${gflags_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rd_party/openfst)
include_directories(${openfst_SOURCE_DIR}/src/include)
if (WIN32)
    include_directories(${openfst_SOURCE_DIR}/src/lib)
    cmake_policy(SET CMP0077 NEW)  # 使 option() 只影响缓存变量
    set(YAML_BUILD_SHARED_LIBS ON)
    # 这俩文件开启O2 编译会报错
    set_source_files_properties("${CMAKE_SOURCE_DIR}/src/asr/internal/bias-lm.cpp" PROPERTIES COMPILE_OPTIONS "/Od")
    set_source_files_properties("${CMAKE_SOURCE_DIR}/src/asr/internal/itn-processor.cpp" PROPERTIES COMPILE_OPTIONS "/Od")
endif ()

add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/yaml-cpp)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd_party/kaldi)

file(GLOB ASR_INTERNAL_SOURCE "${CMAKE_SOURCE_DIR}/src/asr/internal/*.cpp")
if (APPLE)
    file(GLOB ITN_SOURCE "${CMAKE_SOURCE_DIR}/src/asr/internal/itn-*.cpp")
    list(REMOVE_ITEM ASR_INTERNAL_SOURCE ${ITN_SOURCE})
endif (APPLE)
list(REMOVE_ITEM ASR_INTERNAL_SOURCE "${CMAKE_SOURCE_DIR}/src/asr/internal/paraformer-torch.cpp")


add_library(${TARGET_NAME} SHARED ${ASR_INTERNAL_SOURCE})
target_compile_options(${TARGET_NAME} PRIVATE /O2)

if (WIN32)
    set(EXTRA_LIBS yaml-cpp csrc kaldi-decoder fst glog gflags avutil avcodec avformat swresample onnxruntime)
    include_directories(${FFMPEG_DIR}/include)
    target_link_directories(${TARGET_NAME} PUBLIC ${ONNXRUNTIME_DIR}/lib)
    target_link_directories(${TARGET_NAME} PUBLIC ${FFMPEG_DIR}/lib)
    target_compile_definitions(${TARGET_NAME} PUBLIC -D_FUNASR_API_EXPORT -DNOMINMAX -DYAML_CPP_DLL)
else ()
    set(EXTRA_LIBS pthread yaml-cpp csrc kaldi-decoder fst glog gflags avutil avcodec avformat swresample)
    include_directories(${FFMPEG_DIR}/include)
    if (APPLE)
        target_link_directories(${TARGET_NAME} PUBLIC ${ONNXRUNTIME_DIR}/lib)
        target_link_directories(${TARGET_NAME} PUBLIC ${FFMPEG_DIR}/lib)
    endif (APPLE)
endif ()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(${TARGET_NAME} PUBLIC ${EXTRA_LIBS})

list(APPEND DEPENDS ${TARGET_NAME})