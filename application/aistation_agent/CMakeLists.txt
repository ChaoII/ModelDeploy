# AIStation 布控 Agent
# 注意：BUILD_AISTATION_AGENT_TESTS 已在根 CMakeLists.txt 声明；随任务推进逐步加入源文件（当前仅 config_adapter.cpp）。
set(AGENT_SOURCES
        config_adapter.cpp
        agent_server.cpp
        main.cpp
)
list(FILTER AGENT_SOURCES EXCLUDE REGEX "main\\.cpp$")   # 供测试目标复用

add_executable(aistation_agent main.cpp)
target_link_libraries(aistation_agent PRIVATE app_common)
target_include_directories(aistation_agent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENABLE_MQTT)
    target_link_libraries(aistation_agent PRIVATE ${PAHO_MQTT_LIBS})
    target_include_directories(aistation_agent PRIVATE ${PAHO_MQTT_INCLUDE_DIR})
    target_compile_definitions(aistation_agent PRIVATE ENABLE_MQTT)
endif ()

if (BUILD_AISTATION_AGENT_TESTS)
    if (NOT TARGET Catch2::Catch2WithMain)
        add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")
    endif ()
    set(AGENT_TEST_SOURCES
            tests/test_config_adapter.cpp
            tests/test_agent_server.cpp
    )
    add_executable(aistation_agent_test ${AGENT_TEST_SOURCES} ${AGENT_SOURCES})
    target_link_libraries(aistation_agent_test PRIVATE app_common Catch2::Catch2WithMain)
    target_include_directories(aistation_agent_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENABLE_MQTT)
        target_link_libraries(aistation_agent_test PRIVATE ${PAHO_MQTT_LIBS})
        target_include_directories(aistation_agent_test PRIVATE ${PAHO_MQTT_INCLUDE_DIR})
        target_compile_definitions(aistation_agent_test PRIVATE ENABLE_MQTT)
    endif ()
    add_test(NAME aistation_agent_test COMMAND aistation_agent_test)
    set_tests_properties(aistation_agent_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif ()
