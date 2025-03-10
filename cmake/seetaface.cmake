#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
set(seetaface_win_x64_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/seetaface_win_x64.zip")
set(seetaface_linux_x64_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/seetaface_linux_x64.zip")
set(seetaface_linux_aarch64_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/seetaface_linux_aarch64.zip")


set(seetaface_win_x64_HASH "SHA256=7f2430943d46eed4d58188fdb9f6c093a8af0d93d3442f3c555daef24484a5cf")
set(seetaface_linux_x64_HASH "SHA256=1ff14645b80ffe205ea700b1e62044f924d2c24b4e2d391bd717765c30e0a9d6")
set(seetaface_linux_aarch64_HASH "SHA256=8de86770daeb2db890b0cdccac79aeb5753880dbadd053d739de48f5e6df513c")


include(FetchContent)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(possible_file_locations
            $ENV{HOME}/Downloads/seetaface_windows_x64.zip
            ${CMAKE_SOURCE_DIR}/seetaface_windows_x64.zip
            ${CMAKE_BINARY_DIR}/seetaface_windows_x64.zip
            /tmp/seetaface_windows_x64.zip
    )
    set(SEETA_FACE_URL "${seetaface_win_x64_URL}")
    set(SEETA_FACE_HASH "${seetaface_win_x64_HASH}")
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(possible_file_locations
                $ENV{HOME}/Downloads/seetaface_linux_x64.zip
                ${CMAKE_SOURCE_DIR}/seetaface_linux_x64.zip
                ${CMAKE_BINARY_DIR}/seetaface_linux_x64.zip
                /tmp/seetaface_linux_x64.zip)
        set(SEETA_FACE_URL "${seetaface_linux_x64_URL}")
        set(SEETA_FACE_HASH "${seetaface_linux_x64_HASH}")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(possible_file_locations
                $ENV{HOME}/Downloads/seetaface_linux_aarch64.zip
                ${CMAKE_SOURCE_DIR}/seetaface_linux_aarch64.zip
                ${CMAKE_BINARY_DIR}/seetaface_linux_aarch64.zip
                /tmp/seetaface_linux_aarch64.zip)
        set(SEETA_FACE_URL "${seetaface_linux_aarch64_URL}")
        set(SEETA_FACE_HASH "${seetaface_linux_aarch64_HASH}")
    endif ()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(FATAL_ERROR "Unsupported system :" ${CMAKE_SYSTEM_NAME})
endif ()

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
)

FetchContent_GetProperties(seetaface)
if (NOT seetaface_POPULATED)
    message(STATUS "Downloading seetaface from ${SEETA_FACE_URL}")
    FetchContent_Populate(seetaface)
else ()
    message(STATUS "seetaface is already populated")
endif ()
message(STATUS "opencv is downloaded to ${seetaface_SOURCE_DIR}")
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




