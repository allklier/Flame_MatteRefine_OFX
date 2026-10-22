// Synthetic test mattes (CLAUDE.md "Synthetic mattes"): a figure drawn from distance-defined
// shapes with known defects, its clean truth, and a picture to go with it. Used by the unit
// tests and, written as EXR by mr_synth, for evaluation and demos in Flame. Deterministic.
#pragma once

#include <string>
#include <vector>

namespace synth {

struct Scene {
  std::string name, about;
  int w = 0, h = 0;
  std::vector<float> matte;  // w*h, 0-1, row 0 = top: the input matte, with the scene's defects
  std::vector<float> truth;  // w*h: the clean matte the defects were added to
  std::vector<float> rgb;    // w*h*3: the picture, truth-alpha composite of fg over a background
  std::vector<float> fg;     // w*h*3: the true foreground colour (for Foreground Colour)
};

// Scene names, in demo order: holes, specks, chatter, mixed_edge, eroded, coarse, fgcolor, all.
std::vector<std::string> sceneNames();

// Geometry scales with h (designed at 1080); throws std::invalid_argument for an unknown name.
Scene make(const std::string& name, int w, int h, unsigned seed = 1);

// Uncompressed scanline OpenEXR, 32-bit float channels: 1 channel = "Y", 3 = "R G B",
// 4 = "R G B A"; data w*h*channels, interleaved, row 0 = top. Returns false on an I/O error.
bool writeExr(const std::string& path, int w, int h, int channels, const std::vector<float>& data);

}  // namespace synth
