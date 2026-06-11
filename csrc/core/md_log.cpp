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

#ifdef _MSC_VER
#define MD_LOCALTIME(timep, tm_ptr) localtime_s(tm_ptr, timep)
#else
#define MD_LOCALTIME(timep, tm_ptr) localtime_r(timep, tm_ptr)
#endif

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

        // ERROR/FATAL → std::cerr, others → std::cout
        std::ostream& out = (level >= LogLevel::MD_LOG_E) ? std::cerr : std::cout;
        const std::time_t now = std::time(nullptr);

        // Cache localtime to avoid repeated calls within the same second
        if (now != cached_time_sec_) {
            cached_time_sec_ = now;
            MD_LOCALTIME(&now, &cached_tm_);
        }

        switch (level) {
        case LogLevel::MD_LOG_D:
            out << termcolor::blue;
            out << std::put_time(&cached_tm_, "%Y-%m-%d %H:%M:%S") << " | DEBUG | ";
            break;
        case LogLevel::MD_LOG_I:
            out << termcolor::green;
            out << std::put_time(&cached_tm_, "%Y-%m-%d %H:%M:%S") << " | INFO  | ";
            break;
        case LogLevel::MD_LOG_W:
            out << termcolor::yellow;
            out << std::put_time(&cached_tm_, "%Y-%m-%d %H:%M:%S") << " | WARN  | ";
            break;
        case LogLevel::MD_LOG_E:
            out << termcolor::red;
            out << std::put_time(&cached_tm_, "%Y-%m-%d %H:%M:%S") << " | ERROR | ";
            break;
        case LogLevel::MD_LOG_F:
            out << termcolor::on_red << termcolor::cyan;
            out << std::put_time(&cached_tm_, "%Y-%m-%d %H:%M:%S") << " | FATAL | ";
            break;
        default:
            out << "[UNKNOWN] ";
        }

        bool hasPrintedPrefix = false;
        if (filename && *filename) {
            out << "[" << filename << "]";
            hasPrintedPrefix = true;
        }
        if (function && *function) {
            out << "[" << function << "]";
            hasPrintedPrefix = true;
        }
        if (line != -1) {
            out << "[" << line << "]";
            hasPrintedPrefix = true;
        }
        if (hasPrintedPrefix) {
            out << ": ";
        }
        return LogStreamWrapper(out, level == LogLevel::MD_LOG_F);
    }
} // namespace modeldeploy
