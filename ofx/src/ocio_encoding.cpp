#include "ocio_encoding.h"

#include <map>
#include <sstream>
#include <vector>

namespace mr {

namespace {

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\""), e = s.find_last_not_of(" \t\r\"");
  return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// "key: value" on a line (after the indentation); false if the line has another key.
bool field(const std::string& line, const char* key, std::string& value) {
  const std::string t = trim(line), k = std::string(key) + ":";
  if (t.compare(0, k.size(), k) != 0) return false;
  value = trim(t.substr(k.size()));
  return true;
}

}  // namespace

std::string colourspaceEncoding(const std::string& configText, const std::string& name) {
  // Roles ("roles:" then indented "role: colourspace") and colourspace blocks ("- !<ColorSpace>"
  // then name / aliases / encoding), in the OCIO v2 layout Flame's configs use.
  std::map<std::string, std::string> roles;
  struct Space {
    std::vector<std::string> names;
    std::string encoding;
  };
  std::vector<Space> spaces;
  std::istringstream in(configText);
  std::string line;
  bool inRoles = false, inSpace = false;  // inSpace: lines belong to spaces.back()
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const bool indented = line[0] == ' ' || line[0] == '\t';
    if (!indented) {
      inRoles = trim(line) == "roles:";
      inSpace = false;  // a new top-level section
    }
    if (inRoles && indented) {
      const size_t c = line.find(':');
      if (c != std::string::npos) roles[trim(line.substr(0, c))] = trim(line.substr(c + 1));
      continue;
    }
    // A new list item of the section (indent <= 2: nested transform lists are deeper).
    const size_t indent = line.find_first_not_of(" \t");
    if (indent <= 2 && line.compare(indent, 4, "- !<") == 0) {
      inSpace = line.find("!<ColorSpace>") != std::string::npos;
      if (inSpace) spaces.emplace_back();
      continue;
    }
    if (!inSpace) continue;
    std::string v;
    if (field(line, "name", v)) {
      spaces.back().names.push_back(v);
    } else if (field(line, "aliases", v)) {  // [a, b, c]
      std::istringstream list(trim(v.substr(v.find('[') + 1, v.rfind(']') - v.find('[') - 1)));
      std::string a;
      while (std::getline(list, a, ',')) spaces.back().names.push_back(trim(a));
    } else if (field(line, "encoding", v)) {
      spaces.back().encoding = v;
    }
  }
  const auto role = roles.find(name);
  const std::string target = role != roles.end() ? role->second : name;
  for (const Space& s : spaces)
    for (const std::string& n : s.names)
      if (n == target) return s.encoding;
  return "";
}

}  // namespace mr
