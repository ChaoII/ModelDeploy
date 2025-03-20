#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")


set(seetaface_win_x64_FILE_NAME "seetaface_win_x64.zip")
set(seetaface_linux_x64_FILE_NAME "seetaface_linux_x64.zip")
set(seetaface_linux_aarch64_FILE_NAME "seetaface_linux_aarch64.zip")
set(SEETA_FACE_BASE_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master")

set(SEETA_FACE_SHARED_LIBS "")
include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(SEETA_FCE_FILE_NAME ${seetaface_win_x64_FILE_NAME})
    set(SEETA_FACE_URL "${SEETA_FACE_BASE_URL}/${seetaface_win_x64_FILE_NAME}")
    set(SEETA_FACE_HASH "SHA256=7f2430943d46eed4d58188fdb9f6c093a8af0d93d3442f3c555daef24484a5cf")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(SEETA_FCE_FILE_NAME ${seetaface_linux_x64_FILE_NAME})
        set(SEETA_FACE_URL "${SEETA_FACE_BASE_URL}/${seetaface_linux_x64_FILE_NAME}")
        set(SEETA_FACE_HASH "SHA256=1ff14645b80ffe205ea700b1e62044f924d2c24b4e2d391bd717765c30e0a9d6")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(SEETA_FCE_FILE_NAME ${seetaface_linux_aarch64_FILE_NAME})
        set(SEETA_FACE_URL "${SEETA_FACE_BASE_URL}/${seetaface_linux_aarch64_FILE_NAME}")
        set(SEETA_FACE_HASH "SHA256=8de86770daeb2db890b0cdccac79aeb5753880dbadd053d739de48f5e6df513c")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

set(possible_file_locations
        $ENV{HOME}/Downloads/${SEETA_FCE_FILE_NAME}
        ${CMAKE_SOURCE_DIR}/${SEETA_FCE_FILE_NAME}
        ${CMAKE_BINARY_DIR}/${SEETA_FCE_FILE_NAME}
        /tmp/${SEETA_FCE_FILE_NAME})

foreach (f IN LISTS possible_file_locations)
    if (EXISTS ${f})
        set(SEETA_FACE_URL "${f}")
        file(TO_CMAKE_PATH "${SEETA_FACE_URL}" SEETA_FACE_URL)
        message(STATUS "Found local downloaded seetaface: ${SEETA_FACE_URL}")
        break()
    endif ()
endforeach ()

FetchContent_Declare(seetaface
        URL
        ${SEETA_FACE_URL}
        URL_HASH ${SEETA_FACE_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(seetaface)
if (NOT seetaface_POPULATED)
    message(STATUS "Downloading seetaface from ${SEETA_FACE_URL}")
    FetchContent_Populate(seetaface)
else ()
    message(STATUS "seetaface is already populated")
endif ()
message(STATUS "seetaface is downloaded to ${seetaface_SOURCE_DIR}")
if (NOT seetaface_SOURCE_DIR)
    message(FATAL_ERROR "seetaface_SOURCE_DIR is not set after population")
endif ()

include_directories(${seetaface_SOURCE_DIR}/include)
link_directories(${seetaface_SOURCE_DIR}/lib)
list(APPEND DEPENDS
        SeetaFaceLandmarker600
        SeetaFaceRecognizer610
        SeetaFaceDetector600
        SeetaFaceAntiSpoofingX600
        SeetaQualityAssessor300
        SeetaPoseEstimation600
        SeetaGenderPredictor600
        SeetaEyeStateDetector200
        SeetaAgePredictor600)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    file(GLOB SEETA_FACE_SHARED_LIBS "${seetaface_SOURCE_DIR}/lib/*.dll")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    file(GLOB SEETA_FACE_SHARED_LIBS "${seetaface_SOURCE_DIR}/lib/*.so")
else ()
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

# 检查目录是否存在
if (NOT EXISTS ${DIST_DIRECTORY})
    # 如果目录不存在，则创建它
    file(MAKE_DIRECTORY ${DIST_DIRECTORY})
    message(STATUS "Directory ${DIST_DIRECTORY} created.")
else ()
    message(STATUS "Directory ${DIST_DIRECTORY} already exists.")
endif ()

file(COPY ${SEETA_FACE_SHARED_LIBS} DESTINATION ${DIST_DIRECTORY})