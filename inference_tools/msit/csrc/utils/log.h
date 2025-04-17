/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MSIT_ACLDUMP_LOG_H
#define MSIT_ACLDUMP_LOG_H

#include <cstdint>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include <thread>
#include <unistd.h>
#include <unordered_set>

namespace Utility {

constexpr uint8_t BUF_SIZE = 32;

enum class LogLevel {
    DEBUG = 0,
    INFO,
    WARNING,
    ERROR
};

inline uint64_t GetPid()
{
    uint64_t pid = static_cast<uint64_t>(getpid());
    return pid;
}

class Log {
public:
    Log() = default;
    ~Log() = default;
    static Log &GetInstance();
    void SetLogLevel(int level);
    int GetLogLevel();
    bool CheckLogLevel(LogLevel level);
    void PrintLog(std::string &&msg, LogLevel lv);
    void filterSpecialChar(std::string &msg);

private:
    std::mutex logMutex_;
    int logLv_ = 1;
    std::map<LogLevel, const char *> LogLevelString = {
        {LogLevel::DEBUG, "DEBUG"}, {LogLevel::INFO, "INFO"},
        {LogLevel::WARNING, "WARNING"}, {LogLevel::ERROR, "ERROR"},
    };
    
    std::unordered_set<std::string> SpecialChar = {
        "\n", "\r", "\u007f", "\b", "\f", "\t", "\v", "\u000b", "%08",
        "%09", "%0a", "%0b", "%0c", "%0d", "%7f", "//", "\\", "&",
    };
};

}

#define DEBUG_LOG(msg, ...)                                                                     \
    do                                                                                          \
    {                                                                                           \
        if (Utility::Log::GetInstance().CheckLogLevel(Utility::LogLevel::DEBUG))                \
        {                                                                                       \
            Utility::Log::GetInstance().PrintLog(std::move(msg), Utility::LogLevel::DEBUG);     \
        }                                                                                       \
    } while (0)

#define INFO_LOG(msg, ...)                                                                      \
    do                                                                                          \
    {                                                                                           \
        if (Utility::Log::GetInstance().CheckLogLevel(Utility::LogLevel::INFO))                 \
        {                                                                                       \
            Utility::Log::GetInstance().PrintLog(std::move(msg), Utility::LogLevel::INFO);      \
        }                                                                                       \
    } while (0)

#define WARNING_LOG(msg, ...)                                                                   \
    do                                                                                          \
    {                                                                                           \
        if (Utility::Log::GetInstance().CheckLogLevel(Utility::LogLevel::WARNING))              \
        {                                                                                       \
            Utility::Log::GetInstance().PrintLog(std::move(msg), Utility::LogLevel::WARNING);   \
        }                                                                                       \
    } while (0)

#define ERROR_LOG(msg, ...)                                                                     \
    do                                                                                          \
    {                                                                                           \
        if (Utility::Log::GetInstance().CheckLogLevel(Utility::LogLevel::ERROR))                \
        {                                                                                       \
            Utility::Log::GetInstance().PrintLog(std::move(msg), Utility::LogLevel::ERROR);     \
        }                                                                                       \
    } while (0)

#endif
