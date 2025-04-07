message("================start--MD_CAPI================")

if (NOT BUILD_CAPI)
    return()
endif ()

file(GLOB_RECURSE CAPI_SRCS ${CMAKE_SOURCE_DIR}/capi/*.cpp)
if (NOT BUILD_VISION)
    file(GLOB_RECURSE VISION_CAPI_SRCS ${CMAKE_SOURCE_DIR}/capi/vision/*.cpp)
    list(REMOVE_ITEM CAPI_SRCS ${VISION_CAPI_SRCS})
endif ()

if (NOT BUILD_AUDIO)
    file(GLOB_RECURSE AUDIO_CAPI_SRCS ${CMAKE_SOURCE_DIR}/capi/audio/*.cpp)
    list(REMOVE_ITEM CAPI_SRCS ${AUDIO_CAPI_SRCS})
endif ()
list(APPEND SOURCE ${CAPI_SRCS})
message("==========================================")
message("${CAPI_SRCS}")