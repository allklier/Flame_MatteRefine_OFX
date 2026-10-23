// A colourspace's encoding from an OCIO config's text (decision 24), without the OCIO library:
// enough to tell whether the picture is scene-linear before Foreground Colour unmixes it.
#pragma once

#include <string>

namespace mr {

// The `encoding:` of the colourspace called `name` in an OCIO v2 config (by name, alias or role,
// e.g. "scene-linear", "log", "sdr-video", "data"); "" if not found or the config has none.
std::string colourspaceEncoding(const std::string& configText, const std::string& name);

}  // namespace mr
