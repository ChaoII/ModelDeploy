//
// Created by aichao on 2025/3/31.
//

#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include "csrc/core/md_decl.h"
#include "tabulate/tabulate.hpp"

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
        explicit LogStreamWrapper(std::ostream& os) : os_(os), is_null_(false) {
        }

        explicit LogStreamWrapper(std::nullptr_t) : os_(std::cout), is_null_(true) {
        }

        template <typename T>
        LogStreamWrapper& operator<<(const T& value) {
            if (!is_null_) {
                os_ << value;
            }
            return *this;
        }

        LogStreamWrapper& operator<<(std::ostream&(*manip)(std::ostream&)) {
            if (!is_null_) {
                if (manip == static_cast<std::ostream &(*)(std::ostream&)>(std::endl)) {
                    os_ << termcolor::reset << std::endl;
                }
                else {
                    os_ << manip;
                }
            }
            return *this;
        }

    private:
        std::ostream& os_;
        bool is_null_;
    };

    class MODELDEPLOY_CXX_EXPORT LogManager {
    public:
        LogManager(const LogManager&) = delete;

        LogManager& operator=(const LogManager&) = delete;

        inline static LogManager& getInstance() {
            static LogManager instance;
            return instance;
        }

        inline void set_log_level(LogLevel level) {
            current_level_ = level;
        }

        inline LogStreamWrapper logStream(LogLevel level, const char* filename, const char* function, int line) const {
            if (current_level_ == LogLevel::MD_LOG_NONE || level < current_level_) {
                return LogStreamWrapper(nullptr);
            }

            const std::time_t now = std::time(nullptr);
            std::tm localTime{};
            localtime(&now, &localTime);

            switch (level) {
            case LogLevel::MD_LOG_D:
                *log_stream_ << termcolor::blue;
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | DEBUG | ";
                break;
            case LogLevel::MD_LOG_I:
                *log_stream_ << termcolor::green;
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | INFO  | ";
                break;
            case LogLevel::MD_LOG_W:
                *log_stream_ << termcolor::yellow;
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | WARN  | ";
                break;
            case LogLevel::MD_LOG_E:
                *log_stream_ << termcolor::red;
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ERROR | ";
                break;
            case LogLevel::MD_LOG_F:
                *log_stream_ << termcolor::on_red << termcolor::cyan;
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | FATAL | ";
                break;
            default:
                *log_stream_ << "[UNKNOWN] ";
            }

            // Build log prefix dynamically based on available data
            bool hasPrintedPrefix = false;
            if (filename && strlen(filename) > 0) {
                *log_stream_ << "[" << filename << "]";
                hasPrintedPrefix = true;
            }
            if (function && strlen(function) > 0) {
                *log_stream_ << "[" << function << "]";
                hasPrintedPrefix = true;
            }
            if (line != -1) {
                *log_stream_ << "[" << line << "]";
                hasPrintedPrefix = true;
            }
            // Only add colon and space if any prefix was printed
            if (hasPrintedPrefix) {
                *log_stream_ << ": ";
            }
            return LogStreamWrapper(*log_stream_);
        }

    private:
        LogLevel current_level_;
        std::ostream* log_stream_;

        // Private constructor
        LogManager() : current_level_(LogLevel::MD_LOG_I), log_stream_(&std::cout) {
        } // Default log level is INFO
    };
} // namespace modeldeploy
