# paho.mqtt.c：ENABLE_MQTT 时 FetchContent 拉取并构建静态 C 客户端
include(FetchContent)

set(PAHO_MQTT_VERSION "v1.3.13" CACHE STRING "paho.mqtt.c git tag")
set(PAHO_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(PAHO_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(PAHO_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(PAHO_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(PAHO_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(PAHO_WITH_SSL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(paho_mqtt
        GIT_REPOSITORY https://github.com/eclipse/paho.mqtt.c.git
        GIT_TAG ${PAHO_MQTT_VERSION}
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(paho_mqtt)

if (TARGET paho-mqtt3c-static)
    set(PAHO_MQTT_LIBS paho-mqtt3c-static)
elseif (TARGET paho-mqtt3c)
    set(PAHO_MQTT_LIBS paho-mqtt3c)
else ()
    message(FATAL_ERROR "paho.mqtt.c target not found after FetchContent")
endif ()
set(PAHO_MQTT_INCLUDE_DIR "${paho_mqtt_SOURCE_DIR}/src")
message(STATUS "paho.mqtt.c ready: ${PAHO_MQTT_LIBS} (${PAHO_MQTT_INCLUDE_DIR})")
