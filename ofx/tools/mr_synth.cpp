// Writes the synthetic test scenes (synth.h) for evaluation and demos in Flame:
//   mr_synth <out dir> [WxH] [scene ...]      (default 1920x1080, all scenes)
// Per scene: <name>_matte.exr (the input matte, grey RGB: connect to the source alpha input),
// <name>_rgb.exr (the picture: the front input), <name>_truth.exr (the clean matte, to compare),
// <name>_fg.exr (fgcolor only: the true foreground colour), <name>_matte.f32 + .txt (raw, as
// Save Matte writes), and scenes.txt (what each scene contains).
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "synth.h"

namespace {
std::vector<float> grey3(const std::vector<float>& m) {
  std::vector<float> out(m.size() * 3);
  for (size_t i = 0; i < m.size(); ++i) out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = m[i];
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mr_synth <out dir> [WxH] [scene ...]\n");
    return 2;
  }
  const std::string dir = argv[1];
  int w = 1920, h = 1080, first = 2;
  if (argc > 2 && std::sscanf(argv[2], "%dx%d", &w, &h) == 2) first = 3;
  std::vector<std::string> names(argv + first, argv + argc);
  if (names.empty()) names = synth::sceneNames();
  std::ofstream list(dir + "/scenes.txt", std::ios::app);
  for (const auto& name : names) {
    const synth::Scene s = synth::make(name, w, h);
    const std::string base = dir + "/" + name;
    bool ok = synth::writeExr(base + "_matte.exr", w, h, 3, grey3(s.matte)) &&
              synth::writeExr(base + "_truth.exr", w, h, 3, grey3(s.truth)) &&
              synth::writeExr(base + "_rgb.exr", w, h, 3, s.rgb);
    if (name == "fgcolor") ok = ok && synth::writeExr(base + "_fg.exr", w, h, 3, s.fg);
    std::ofstream raw(base + "_matte.f32", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(s.matte.data()), std::streamsize(s.matte.size() * 4));
    std::ofstream(base + "_matte.txt") << "width=" << w << "\nheight=" << h
                                       << "\nformat=float32 little-endian, top row first, 0-1\n";
    list << name << " (" << w << "x" << h << "): " << s.about << "\n";
    std::printf("%-10s %s\n", name.c_str(), ok && raw ? "written" : "FAILED");
    if (!ok || !raw) return 1;
  }
  return 0;
}
