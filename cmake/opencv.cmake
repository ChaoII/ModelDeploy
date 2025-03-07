#
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")


set(opencv5.x_static_mt_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/install_mt.zip")
set(opencv5.x_static_md_URL "https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/install_md.zip")
set(opencv5.x_static_mt_HASH "SHA256=fc3d1961575869267bd93cf92feb7cf1ab31cbbdcf3ef66063899e86bdd04b64")
set(opencv5.x_static_md_HASH "SHA256=8eacc41bc966829cec984775ac6a6c35267f41533cf0003bfdb8ec98dfc89639")

include(FetchContent)

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
        $ENV{HOME}/Downloads/install_mt.zip
        ${CMAKE_SOURCE_DIR}/install_mt.zip
        ${CMAKE_BINARY_DIR}/install_mt.zip
        /tmp/install_mt.zip
)

foreach (f IN LISTS possible_file_locations)
    if (EXISTS ${f})
        set(opencv5.x_static_mt_URL "${f}")
        file(TO_CMAKE_PATH "${opencv5.x_static_mt_URL}" opencv5.x_static_mt_URL)
        message(STATUS "Found local downloaded opencv: ${opencv5.x_static_mt_URL}")
        break()
    endif ()
endforeach ()

FetchContent_Declare(opencv
        URL
        ${opencv5.x_static_mt_URL}
        URL_HASH ${opencv5.x_static_mt_HASH}
)

FetchContent_GetProperties(opencv)
if (NOT opencv_POPULATED)
    message(STATUS "Downloading opencv from ${opencv5.x_static_mt_URL}")
    FetchContent_Populate(opencv)
else ()
    message(STATUS "opencv is already populated")
endif ()
message(STATUS "opencv is downloaded to ${opencv_SOURCE_DIR}")
if (NOT opencv_SOURCE_DIR)
    message(FATAL_ERROR "opencv_SOURCE_DIR is not set after population")
endif ()


set(opencv_DIR "${opencv_SOURCE_DIR}/x64/vc17/staticlib")
find_package(opencv CONFIG REQUIRED)
