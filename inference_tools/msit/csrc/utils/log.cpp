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

#include "log.h"

namespace Utility {

Log &Log::GetInstance()
{
    static Log instance;
    return instance;
}

int Log::GetLogLevel()
{
    return logLv_;
}

void Log::SetLogLevel(int level)
{
    logLv_ = level;
}

bool Log::CheckLogLevel(LogLevel level)
{
    return static_cast<int>(level) >= logLv_;
}

void Log::filterSpecialChar(std::string &msg)
{
    for (const auto &s : SpecialChar)
    {
        size_t pos = 0;
        while ((pos = msg.find(s, pos)) != std::string::npos)
        {
            msg.replace(pos, s.length(), "_");
            pos += 1;
        }
    }
}

void Log::PrintLog(std::string &&msg, LogLevel lv)
{
    std::lock_guard<std::mutex> lock(logMutex_);
    filterSpecialChar(msg);
    char buf[BUF_SIZE];
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&time);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    const char *logLevel = LogLevelString[lv];
    printf("%s (PID %ld) [%s] %s\n", buf, GetPid(), logLevel, msg.c_str());
    fflush(stdout);
}

}
