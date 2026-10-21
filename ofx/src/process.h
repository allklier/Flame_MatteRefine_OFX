// Child processes launched from inside Flame (from LTX Alpha Gen; here only tar for Collect
// Logs): own process group, stdout/stderr piped back line by line, a cleaned environment
// (Flame's LD_LIBRARY_PATH and Python variables must not leak into the child).
#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

namespace mr {

using Env = std::vector<std::string>;  // "KEY=value"

// Current environment minus variables that would hijack our Python, plus extras.
Env cleanEnvironment(const Env& extras = {});

class WorkerProcess {
 public:
  ~WorkerProcess();
  bool start(const std::vector<std::string>& argv, const Env& env, std::string& err);
  // Waits up to timeoutMs for output; appends complete lines. Returns false once
  // the worker has closed stdout (it exited) and all lines were delivered.
  bool readLines(int timeoutMs, std::vector<std::string>& lines);
  void kill();  // SIGKILL the whole process group
  int wait();   // blocking; returns the raw waitpid status
  pid_t pid() const { return pid_; }

 private:
  pid_t pid_ = -1;
  int outFd_ = -1;
  int inFd_ = -1;
  std::string buf_;
};

}  // namespace mr
