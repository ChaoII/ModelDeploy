#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")



set(TRT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.9.0.34" CACHE PATH "TRT_DIR" FORCE)
set(TRT_LIB_DIR "${TRT_DIR}/lib")
set(TRT_INC_DIR "${TRT_DIR}/include")

# 获取编译时TensorRT主版本号
set(TRT_VERSION_HPP "${TRT_INC_DIR}/NvInferVersion.h")
if(EXISTS "${TRT_VERSION_HPP}")
    # 读取版本头文件
    file(READ "${TRT_VERSION_HPP}" TRT_VERSION_CONTENT)
    # 提取宏定义
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)"
            TRT_MAJOR_MATCH ${TRT_VERSION_CONTENT})
    if(TRT_MAJOR_MATCH)
        set(TRT_MAJOR_VERSION ${CMAKE_MATCH_1})
        message(STATUS "TensorRT 主版本: ${TRT_MAJOR_VERSION}")
    endif()
endif()



find_library(NVINFER_LIB nvinfer_10
        PATHS "${TRT_LIB_DIR}"
        NO_DEFAULT_PATH
)

find_library(NVONNXPARSER_LIB nvonnxparser_10
        PATHS "${TRT_LIB_DIR}"
        NO_DEFAULT_PATH
)

add_library(trt::nvinfer STATIC IMPORTED GLOBAL)
add_library(trt::nvonnxparser STATIC IMPORTED GLOBAL)

set_target_properties(trt::nvinfer PROPERTIES
        IMPORTED_LOCATION "${NVINFER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TRT_INC_DIR}"
)
set_target_properties(trt::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${NVONNXPARSER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TRT_INC_DIR}"
)

# 拷贝到 ${CMAKE_BINARY_DIR}/bin 或你指定的 bin 目录
if (WIN32)
    file(GLOB TRT_SHARED_LIBS "${TRT_LIB_DIR}/*.dll")
else ()
    file(GLOB TRT_SHARED_LIBS "${TRT_LIB_DIR}/*.so" "${TRT_LIB_DIR}/*.so.*")
endif ()
#file(COPY ${TRT_SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





