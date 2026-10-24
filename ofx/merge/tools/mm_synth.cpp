// Writes the Matte Merge test scenes (merge_synth.h) for evaluation and demos in Flame:
//   mm_synth <out dir> [WxH]      (default 1920x1080; scenes hair_flat and hair_busy)
// Per scene, as mr_synth (uncompressed float EXR, mattes as grey RGB):
//   <s>_base.exr (the base matte: the source alpha input), <s>_base_uniform.exr (a base with a uniform
//   3 px edge offset and no hair, for registration), <s>_detail.exr (the detail matte: the
//   extra clip), <s>_trimap.exr (the trimap variant of the base), <s>_truth.exr, <s>_body.exr (truth
//   body alone), <s>_fake.exr and <s>_blob.exr (what the detail made up), <s>_rgb_display.exr and
//   <s>_rgb_log.exr (the picture, sRGB-curve and Log3G10-like encodings), <s>_front.exr (display
//   picture with the base in alpha: one clip for the front input), <s>_front_trimap.exr (the same
//   with the trimap), and merge_scenes.txt (the true detail move per scene).
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "merge_synth.h"
#include "synth.h"

namespace {
std::vector<float> grey3(const std::vector<float>& m) {
  std::vector<float> out(m.size() * 3);
  for (size_t i = 0; i < m.size(); ++i) out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = m[i];
  return out;
}
std::vector<float> rgba(const std::vector<float>& rgb, const std::vector<float>& a) {
  std::vector<float> out(a.size() * 4);
  for (size_t i = 0; i < a.size(); ++i) {
    for (int c = 0; c < 3; ++c) out[i * 4 + c] = rgb[i * 3 + c];
    out[i * 4 + 3] = a[i];
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mm_synth <out dir> [WxH]\n");
    return 2;
  }
  const std::string dir = argv[1];
  int w = 1920, h = 1080;
  if (argc > 2 && std::sscanf(argv[2], "%dx%d", &w, &h) != 2) {
    std::fprintf(stderr, "bad size %s\n", argv[2]);
    return 2;
  }
  std::ofstream list(dir + "/merge_scenes.txt");
  list << "# scene WxH: base pixel p is at c + scale (p - c) + (tx, ty) in the detail, c = frame centre\n";
  for (bool busy : {false, true}) {
    const mmsynth::MergeScene m = mmsynth::makeMergeScene(w, h, busy);
    const std::string b = dir + "/" + m.name;
    const bool ok = synth::writeExr(b + "_base.exr", w, h, 3, grey3(m.base)) &&
                    synth::writeExr(b + "_base_uniform.exr", w, h, 3, grey3(m.baseUniform)) &&
                    synth::writeExr(b + "_detail.exr", w, h, 3, grey3(m.detail)) &&
                    synth::writeExr(b + "_trimap.exr", w, h, 3, grey3(m.trimap)) &&
                    synth::writeExr(b + "_truth.exr", w, h, 3, grey3(m.truth)) &&
                    synth::writeExr(b + "_body.exr", w, h, 3, grey3(m.body)) &&
                    synth::writeExr(b + "_fake.exr", w, h, 3, grey3(m.fake)) &&
                    synth::writeExr(b + "_blob.exr", w, h, 3, grey3(m.blob)) &&
                    synth::writeExr(b + "_rgb_display.exr", w, h, 3, m.rgbDisplay) &&
                    synth::writeExr(b + "_rgb_log.exr", w, h, 3, m.rgbLog) &&
                    synth::writeExr(b + "_front.exr", w, h, 4, rgba(m.rgbDisplay, m.base)) &&
                    synth::writeExr(b + "_front_trimap.exr", w, h, 4, rgba(m.rgbDisplay, m.trimap));
    char line[256];
    std::snprintf(line, sizeof line, "%s %dx%d: tx %.4f ty %.4f scale %.4f; blob at (%.1f, %.1f) r %.1f px\n",
                  m.name.c_str(), w, h, m.tx, m.ty, m.scale, m.blobX, m.blobY, m.blobR);
    list << line;
    std::printf("%-10s %s\n", m.name.c_str(), ok ? "written" : "FAILED");
    if (!ok) return 1;
  }
  return list ? 0 : 1;
}
