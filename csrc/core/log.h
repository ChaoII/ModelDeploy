#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include "csrc/core/md_decl.h"
#include "tabulate/tabulate.hpp"


#ifdef _MSC_VER
#define localtime(timep, tm_ptr) localtime_s(tm_ptr, timep)
#else
    #define localtime(timep, tm_ptr) localtime_r(timep, tm_ptr)
#endif

// Macro to extract the filename from the full path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1   \
    : (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__))

#define MD_LOG_DEBUG modeldeploy::LogManager::getInstance() \
    .logStream(modeldeploy::LogLevel::MD_LOG_D, __FILENAME__, __FUNCTION__, __LINE__)
#define MD_LOG_INFO modeldeploy::LogManager::getInstance() \
    .logStream(modeldeploy::LogLevel::MD_LOG_I, "", "", -1)
#define MD_LOG_WARN modeldeploy::LogManager::getInstance() \
    .logStream(modeldeploy::LogLevel::MD_LOG_W, "", "", -1)
#define MD_LOG_ERROR modeldeploy::LogManager::getInstance() \
    .logStream(modeldeploy::LogLevel::MD_LOG_E, __FILENAME__, __FUNCTION__, __LINE__)
#define MD_LOG_FATAL modeldeploy::LogManager::getInstance() \
    .logStream(modeldeploy::LogLevel::MD_LOG_F, "", "", -1)

#define MD_SET_LOG_LEVEL(level) modeldeploy::LogManager::getInstance().set_log_level(level)

namespace modeldeploy {
    enum class LogLevel: std::uint8_t {
        MD_LOG_NONE = 0,
        MD_LOG_D,
        MD_LOG_I,
        MD_LOG_W,
        MD_LOG_E,
        MD_LOG_F
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

        std::ostream& logStream(LogLevel level, const char* filename, const char* function, int line) const {
            std::ostringstream oss;
            // Check whether the current level is LOG NONE or the log level is not enough to log
            if (current_level_ == LogLevel::MD_LOG_NONE || level < current_level_)
                return *log_stream_; // Return the log stream without any output
            const std::time_t now = std::time(nullptr);
            std::tm localTime{};
            localtime(&now, &localTime);

            switch (level) {
            case LogLevel::MD_LOG_D:
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ";

                *log_stream_ << termcolor::blue << "[DEBUG] ";
                break;
            case LogLevel::MD_LOG_I:
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ";

                *log_stream_ << termcolor::green << "[INFO] ";
                break;
            case LogLevel::MD_LOG_W:
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ";
                *log_stream_ << termcolor::yellow << "[WARN] ";
                break;
            case LogLevel::MD_LOG_E:
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ";
                *log_stream_ << termcolor::red << "[ERROR] ";
                break;
            case LogLevel::MD_LOG_F:
                *log_stream_ << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << " | ";
                *log_stream_ << termcolor::on_red << "[FATAL] ";
                break;
            default:
                *log_stream_ << "[UNKNOWN] ";
            }


            // Build log prefix dynamically based on available data
            bool hasPrintedPrefix = false;
            if (filename && strlen(filename) > 0) {
                oss << "[" << filename << "]";
                hasPrintedPrefix = true;
            }
            if (function && strlen(function) > 0) {
                oss << "[" << function << "]";
                hasPrintedPrefix = true;
            }
            if (line != -1) {
                oss << "[" << line << "]";
                hasPrintedPrefix = true;
            }
            // Only add colon and space if any prefix was printed
            if (hasPrintedPrefix) {
                oss << ": ";
            }
            // Output the prefix to the log stream
            *log_stream_ << oss.str() << termcolor::reset;
            return *log_stream_;
        }

    private:
        LogLevel current_level_;
        std::ostream* log_stream_;

        // Private constructor
        LogManager() : current_level_(LogLevel::MD_LOG_I), log_stream_(&std::cout) {
        } // Default log level is INFO
    };
} // namespace modeldeploy
