// Linux side of platform.h.
#include <gnu/libc-version.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>

#include "platform.h"

namespace mr {

namespace {
std::string firstMatch(const char* path, const std::string& prefix) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
  return "unknown";
}

std::string gb(double bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1f GB", bytes / 1073741824.0);
  return b;
}
}  // namespace

std::vector<std::string> platformSnapshot() {
  std::vector<std::string> out;
  utsname u{};
  uname(&u);
  std::string os = firstMatch("/etc/os-release", "PRETTY_NAME=");
  out.push_back("os: " + os + ", kernel " + u.release + ", glibc " + gnu_get_libc_version());
  out.push_back("cpu: " + firstMatch("/proc/cpuinfo", "model name\t: ") + ", " +
                std::to_string(sysconf(_SC_NPROCESSORS_ONLN)) + " threads");
  struct sysinfo si {};
  sysinfo(&si);
  out.push_back("ram: " + gb(double(si.totalram) * si.mem_unit) + " total, " +
                gb(double(si.freeram) * si.mem_unit) + " free");
  return out;
}

}  // namespace mr
