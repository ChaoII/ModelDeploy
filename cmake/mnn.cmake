
set(MNN_DIR "C:/Users/aichao/Desktop/MNN")
set(MNN_LIB_DIR "${MNN_DIR}/lib")
set(MNN_INC_DIR "${MNN_DIR}/include")
include_directories(${MNN_INC_DIR})
link_directories(${MNN_LIB_DIR})
list(APPEND DEPENDS MNN)
# 只有静态链接MNN才需要
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WHOLEARCHIVE:MNN")

# 拷贝到 ${CMAKE_BINARY_DIR}/bin 或你指定的 bin 目录
file(GLOB SHARED_LIBS
        "${onnxruntime_SOURCE_DIR}/lib/*.dll"
        "${onnxruntime_SOURCE_DIR}/lib/*.so"
        "${onnxruntime_SOURCE_DIR}/lib/*.dylib"
)
file(COPY ${SHARED_LIBS} DESTINATION ${CMAKE_BINARY_DIR}/bin)





