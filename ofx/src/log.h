// Diagnostic logging (LTX Alpha Gen decision 39, CLAUDE.md "Logging and support"). One rotating
// per-user log: ~/.cache/matte-refine/logs/matte-refine.log
// Line format (LTX's, without the job id):
//   2026-10-21 16:35:53.383 INFO  ofx[166761/166761] node=ab12cd34 | message
// Logging never throws and never blocks rendering; a failed write is dropped.
#pragma once

#include <string>

namespace mr {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

std::string logDir();
std::string logPath();

// Level from $MR_LOG_LEVEL, else the user config key log_level, else info.
void logConfigure();
void setLogLevel(LogLevel level);
LogLevel logLevel();
bool logEnabled(LogLevel level);

// Keeps at most `keep` rotated files of about `maxBytes` each. Called at plugin load.
void rotateLogs(size_t maxBytes = 10u << 20, int keep = 5);

void logAt(LogLevel level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#define LOGD(...) ::mr::logAt(::mr::LogLevel::Debug, __VA_ARGS__)
#define LOGI(...) ::mr::logAt(::mr::LogLevel::Info, __VA_ARGS__)
#define LOGW(...) ::mr::logAt(::mr::LogLevel::Warn, __VA_ARGS__)
#define LOGE(...) ::mr::logAt(::mr::LogLevel::Error, __VA_ARGS__)

// Node id for every line logged on this thread while the scope lives.
class LogScope {
 public:
  explicit LogScope(const std::string& node);
  ~LogScope();
  LogScope(const LogScope&) = delete;
  LogScope& operator=(const LogScope&) = delete;

 private:
  std::string prevNode_;
};

}  // namespace mr
