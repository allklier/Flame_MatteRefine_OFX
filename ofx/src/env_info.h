// Environment snapshot for the log and the support bundle (CLAUDE.md "Logging and support").
#pragma once

#include <string>
#include <vector>

namespace mr {

struct HostInfo {
  std::string name, label, version;
};

// One "key: value" line per fact.
std::vector<std::string> environmentSnapshot(const HostInfo& host);

}  // namespace mr
