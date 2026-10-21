#include "env_info.h"

#include "log.h"
#include "platform.h"
#include "user_config.h"

#ifndef MR_VERSION
#define MR_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif

namespace mr {

std::vector<std::string> environmentSnapshot(const HostInfo& host) {
  std::vector<std::string> out;
  out.push_back(std::string("plugin: Matte Refine ") + MR_VERSION + " git " + MR_GIT + " built " +
                __DATE__ + " " + __TIME__);
  out.push_back("host: " + host.name + " / " + host.label + " version " + host.version);
  for (const auto& line : platformSnapshot()) out.push_back(line);
  out.push_back("log: " + logPath());
  out.push_back("config: " + userConfigPath());
  for (const auto& kv : loadUserConfig()) out.push_back("config." + kv.first + ": " + kv.second);
  return out;
}

}  // namespace mr
