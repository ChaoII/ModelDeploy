//
// Created by aichao on 2025/3/31.
//

#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstring>
#include "core/md_log.h"
#include "tabulate/tabulate.hpp"





namespace modeldeploy {
    LogStreamWrapper& LogStreamWrapper::operator<<(std::ostream&(*manip)(std::ostream&)) {
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


    LogStreamWrapper LogManager::logStream(LogLevel level, const char* filename, const char* function, int line) const {
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
        return LogStreamWrapper(*log_stream_, level == LogLevel::MD_LOG_F);
    }
} // namespace modeldeploy
