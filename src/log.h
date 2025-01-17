//
// Created by aichao on 2025/1/17.
//
#pragma once
#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h" // support for basic file logging
#include "spdlog/sinks/stdout_color_sinks.h"


class MDLog {
public:
    static std::shared_ptr<spdlog::logger>& getInstance() {
        static std::shared_ptr<spdlog::logger> logger = initializeLogger();
        return logger;
    }

private:
    static std::shared_ptr<spdlog::logger> initializeLogger() {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
#ifdef SAVE_LOG
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/log.txt", 1024 * 1024 * 100, 3);
#endif
        spdlog::sinks_init_list sink_list = {
#ifdef SAVE_LOG
            file_sink,
#endif
            console_sink
        };
        auto logger = std::make_shared<spdlog::logger>("multi_sink", sink_list.begin(), sink_list.end());
        logger->set_level(spdlog::level::info);
        // 设置日志模式，包含行号
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%l]%$ [%s:%#] [%P:%t] %v");
        return logger; // 返回文件日志记录器
    }
};

#define MD_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(MDLog::getInstance(), __VA_ARGS__)
#define MD_LOG_INFO(...) SPDLOG_LOGGER_INFO(MDLog::getInstance(), __VA_ARGS__)
#define MD_LOG_WARN(...) SPDLOG_LOGGER_WARN(MDLog::getInstance(),__VA_ARGS__)
#define MD_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(MDLog::getInstance(),__VA_ARGS__)
#define MD_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(MDLog::getInstance(),__VA_ARGS__)
