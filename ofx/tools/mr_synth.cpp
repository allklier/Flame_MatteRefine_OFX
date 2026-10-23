// Writes the synthetic test scenes (synth.h) for evaluation and demos in Flame:
//   mr_synth <out dir> [WxH] [scene ...]      (default 1920x1080, all scenes)
// Scenes: sceneNames(), then the ML ones (mlSceneNames(): gaps, hair_flat, hair_busy) and
// "temporal" (the 9-frame sequence, seed 1, moving background).
// Per scene: <name>_matte.exr (the input matte, grey RGB: connect to the source alpha input),
// <name>_rgb.exr (the picture: the front input), <name>_truth.exr (the clean matte, to compare),
// <name>_fg.exr (fgcolor and hair_*: the true foreground colour), <name>_label.exr (gaps: the
// label / 3 as grey: 0 background, 0.33 subject, 0.67 defect hole, 1 real gap), <name>_matte.f32
// + .txt (raw, as Save Matte writes), and scenes.txt (what each scene contains). temporal:
// temporal_{matte,truth,rgb}.NNNN.exr per frame (frame number last, so Flame sees sequences) and temporal_motion.txt (per frame: the subject's
// affine from frame 0, a b c d tx ty in pixels, the background's tx ty, whether the hole is on).
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
  if (names.empty()) {
    names = synth::sceneNames();
    for (const auto& nm : synth::mlSceneNames()) names.push_back(nm);
    names.push_back("temporal");
  }
  std::ofstream list(dir + "/scenes.txt", std::ios::app);
  for (const auto& name : names) {
    if (name == "temporal") {
      const synth::Sequence q = synth::makeSequence(w, h);
      std::ofstream motion(dir + "/temporal_motion.txt");
      motion << "# frame  subject a b c d tx ty (frame 0 -> this frame, px)  background tx ty  hole\n";
      bool ok = bool(motion);
      for (size_t t = 0; t < q.frames.size() && ok; ++t) {
        // The frame number goes last (name_kind.0004.exr) so Flame reads each kind as one sequence.
        char num[16];
        std::snprintf(num, sizeof num, ".%04d.exr", int(t));
        const std::string base = dir + "/temporal";
        ok = synth::writeExr(base + "_matte" + num, w, h, 3, grey3(q.frames[t].matte)) &&
             synth::writeExr(base + "_truth" + num, w, h, 3, grey3(q.frames[t].truth)) &&
             synth::writeExr(base + "_rgb" + num, w, h, 3, q.frames[t].rgb);
        const synth::Affine& a = q.subject[t];
        char line[256];
        std::snprintf(line, sizeof line, "%d  %.9f %.9f %.9f %.9f %.6f %.6f  %.6f %.6f  %d\n", int(t), a.a, a.b,
                      a.c, a.d, a.tx, a.ty, q.background[t].tx, q.background[t].ty, int(q.holeOn[t]));
        motion << line;
      }
      list << name << " (" << w << "x" << h << ", " << q.frames.size() << " frames): " << q.frames[0].about << "\n";
      std::printf("%-10s %s\n", name.c_str(), ok && motion ? "written" : "FAILED");
      if (!ok || !motion) return 1;
      continue;
    }
    const synth::Scene s = synth::make(name, w, h);
    const std::string base = dir + "/" + name;
    bool ok = synth::writeExr(base + "_matte.exr", w, h, 3, grey3(s.matte)) &&
              synth::writeExr(base + "_truth.exr", w, h, 3, grey3(s.truth)) &&
              synth::writeExr(base + "_rgb.exr", w, h, 3, s.rgb);
    if (name == "fgcolor" || name == "hair_flat" || name == "hair_busy")
      ok = ok && synth::writeExr(base + "_fg.exr", w, h, 3, s.fg);
    if (!s.label.empty()) {
      std::vector<float> lab(s.label.size());
      for (size_t i = 0; i < lab.size(); ++i) lab[i] = s.label[i] / 3.0f;
      ok = ok && synth::writeExr(base + "_label.exr", w, h, 3, grey3(lab));
    }
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
