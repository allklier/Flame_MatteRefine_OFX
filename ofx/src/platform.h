// Everything that differs between operating systems (as LTX Alpha Gen decision 64). Linux:
// platform_linux.cpp; macOS (decision 13, next): platform_macos.cpp; selected in CMakeLists.txt.
#pragma once

#include <string>
#include <vector>

namespace mr {

// OS-specific lines of the environment snapshot (os, cpu, ram).
std::vector<std::string> platformSnapshot();

}  // namespace mr
