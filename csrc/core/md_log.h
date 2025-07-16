//
// Created by aichao on 2025/3/31.
//

#pragma once

#include <iostream>
#include <cstring>
#include <sstream>
#include <cstdint>
#include "core/md_decl.h"


namespace modeldeploy {
    enum class LogLevel : std::uint8_t {
        MD_LOG_NONE = 0,
        MD_LOG_D,
        MD_LOG_I,
        MD_LOG_W,
        MD_LOG_E,
        MD_LOG_F
    };

    class LogStreamWrapper {
    public:
        explicit LogStreamWrapper(std::ostream& os, const bool fatal = false) : os_(os), is_null_(false),
            fatal_(fatal) {
        }

        explicit LogStreamWrapper(std::nullptr_t) : os_(std::cout), is_null_(true), fatal_(false) {
        }

        ~LogStreamWrapper() {
            if (fatal_) {
                std::exit(EXIT_FAILURE); // fatal 日志触发立即退出
            }
        }

        template <typename T>
        LogStreamWrapper& operator<<(const T& value) {
            if (!is_null_) {
                os_ << value;
            }
            return *this;
        }

        LogStreamWrapper& operator<<(std::ostream&(*manip)(std::ostream&));

    private:
        std::ostream& os_;
        bool is_null_;
        bool fatal_;
    };

    class MODELDEPLOY_CXX_EXPORT LogManager {
    public:
        LogManager(const LogManager&) = delete;

        LogManager& operator=(const LogManager&) = delete;

        static LogManager& getInstance() {
            static LogManager instance;
            return instance;
        }

        void set_log_level(LogLevel level) {
            current_level_ = level;
        }

        LogStreamWrapper logStream(LogLevel level, const char* filename, const char* function, int line) const;

    private:
        LogLevel current_level_;
        std::ostream* log_stream_;

        // Private constructor
        LogManager() : current_level_(LogLevel::MD_LOG_I), log_stream_(&std::cout) {
        } // Default log level is INFO
    };
} // namespace modeldeploy

#ifdef _MSC_VER
#define localtime(timep, tm_ptr) localtime_s(tm_ptr, timep)
#else
#define localtime(timep, tm_ptr) localtime_r(timep, tm_ptr)
#endif

#define CUS_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1   \
: (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__))

#define MD_LOG_DEBUG modeldeploy::LogManager::getInstance() \
.logStream(modeldeploy::LogLevel::MD_LOG_D, CUS_FILENAME, __FUNCTION__, __LINE__)
#define MD_LOG_INFO modeldeploy::LogManager::getInstance() \
.logStream(modeldeploy::LogLevel::MD_LOG_I, "", "", -1)
#define MD_LOG_WARN modeldeploy::LogManager::getInstance() \
.logStream(modeldeploy::LogLevel::MD_LOG_W, "", "", -1)
#define MD_LOG_ERROR modeldeploy::LogManager::getInstance() \
.logStream(modeldeploy::LogLevel::MD_LOG_E, CUS_FILENAME, __FUNCTION__, __LINE__)
#define MD_LOG_FATAL modeldeploy::LogManager::getInstance() \
.logStream(modeldeploy::LogLevel::MD_LOG_F, "", "", -1)

#define MD_SET_LOG_LEVEL(level) modeldeploy::LogManager::getInstance().set_log_level(level)
