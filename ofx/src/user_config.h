// Per-user config: ~/.config/matte-refine/config.conf, plain "key=value" lines.
// Holds per-user settings (log_level). Absent keys fall back to the built-in defaults.
#pragma once

#include <map>
#include <string>

namespace mr {

using UserConfig = std::map<std::string, std::string>;

std::string userConfigPath();
UserConfig loadUserConfig();
bool saveUserConfig(const UserConfig& cfg);  // atomic replace

}  // namespace mr
