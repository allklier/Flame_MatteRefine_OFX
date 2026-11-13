// Jump Repair on a real matte sequence, numbers only (not a test): reads a float32 numpy file of
// shape (frames, h, w) (alpha 0-1, row 0 = top, C order) and prints, per frame, the blobs the core
// found (direction, references, area, bounding box, guard sums, decision) and the pixels it would
// change. Never writes or prints images: footage can be confidential.
// Usage: mr_jump_eval <file.npy> [--first N] [--span 1-3] [--threshold T] [--min-size %] [--no-guard]
//        [--quiet]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "jump_repair.h"

using namespace mr::ml;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file.npy> [--first N] [--span S] [--threshold T] [--min-size P] [--no-guard]\n",
                 argv[0]);
    return 2;
  }
  JumpParams p;
  int first = 0;
  bool quiet = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--first" && i + 1 < argc) first = std::atoi(argv[++i]);
    else if (a == "--span" && i + 1 < argc) p.maxSpan = std::atoi(argv[++i]);
    else if (a == "--threshold" && i + 1 < argc) p.threshold = std::atof(argv[++i]);
    else if (a == "--min-size" && i + 1 < argc) p.minSizePercent = std::atof(argv[++i]);
    else if (a == "--no-guard") p.motionGuard = false;
    else if (a == "--quiet") quiet = true;
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  std::ifstream in(argv[1], std::ios::binary);
  char magic[8];
  if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::fprintf(stderr, "not a numpy file\n");
    return 1;
  }
  unsigned hl = 0;
  if (magic[6] == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    hl = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    hl = b[0] | (b[1] << 8) | (b[2] << 16) | (unsigned(b[3]) << 24);
  }
  std::string hdr(hl, ' ');
  in.read(&hdr[0], hl);
  if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) {
    std::fprintf(stderr, "need little-endian float32, C order: %s\n", hdr.c_str());
    return 1;
  }
  int T = 0, h = 0, w = 0;
  const size_t sp = hdr.find("'shape': (");
  if (sp == std::string::npos || std::sscanf(hdr.c_str() + sp + 10, "%d, %d, %d", &T, &h, &w) != 3) {
    std::fprintf(stderr, "need a 3-d shape: %s\n", hdr.c_str());
    return 1;
  }
  const size_t n = size_t(w) * h;
  std::vector<std::vector<float>> m(size_t(T), std::vector<float>(n, 0.0f));
  for (auto& f : m) {
    if (!in.read(reinterpret_cast<char*>(f.data()), std::streamsize(n * 4))) {
      std::fprintf(stderr, "short file\n");
      return 1;
    }
    for (float& v : f) v = v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f;  // as the plugin's extractMatte
  }
  std::printf("%d frames %dx%d, span %d threshold %.3g min size %.4g %% guard %s\n", T, w, h, p.maxSpan, p.threshold,
              p.minSizePercent, p.motionGuard ? "on" : "off");
  std::vector<float> out(n);
  int framesChanged = 0, judged = 0, repairedTotal = 0, keptTotal = 0;
  long long changedTotal = 0;
  double msTotal = 0;
  for (int t = 0; t < T; ++t) {
    const float* fr[2 * kJumpReach + 1];
    for (int k = -kJumpReach; k <= kJumpReach; ++k)
      fr[k + kJumpReach] = t + k >= 0 && t + k < T ? m[size_t(t + k)].data() : nullptr;
    JumpStats st;
    repairJumps(fr, w, h, p, out.data(), &st);
    msTotal += st.totalMs;
    judged += st.judged;
    if (st.pixelsChanged) ++framesChanged;
    changedTotal += st.pixelsChanged;
    repairedTotal += st.repaired;
    keptTotal += st.keptMoved + st.keptTurning + st.keptNotStill;
    if (quiet && st.blobs.empty()) continue;
    std::printf("frame %d: %s flagged %lld px, small blobs %d (< %lld px), blobs %d: repaired %d, kept moved %d, "
                "turning %d, not still %d | changed %lld px (%.4f %%), max change %.3f | %.1f ms\n",
                first + t, st.judged ? "judged," : "NOT judged (ends),", st.flaggedPixels, st.smallBlobs, st.minArea,
                int(st.blobs.size()), st.repaired, st.keptMoved, st.keptTurning, st.keptNotStill, st.pixelsChanged,
                100.0 * double(st.pixelsChanged) / double(n), st.maxChange, st.totalMs);
    for (const JumpBlob& b : st.blobs)
      std::printf("    %s %lld px y%d-%d x%d-%d refs %+d/%+d mass %.1f | moved %.2f: %.2f (%+d,%+d) %.2f (%+d,%+d) | turn %s %.1f %.1f | %s, "
                  "changed %lld\n",
                  b.added ? "added  " : "removed", b.area, b.y0, b.y1, b.x0, b.x1, b.before, b.after, b.mass, b.moveScore,
                  b.moveBefore, b.moveBeforeDx, b.moveBeforeDy, b.moveAfter, b.moveAfterDx, b.moveAfterDy, b.turnChecked ? "" : "(n/a)", b.turnBefore, b.turnAfter,
                  jumpDecisionName(b.decision), b.changed);
  }
  std::printf("summary: judged %d of %d, frames changed %d, blobs repaired %d kept %d, pixels changed %lld, mean %.1f "
              "ms per frame\n",
              judged, T, framesChanged, repairedTotal, keptTotal, changedTotal, msTotal / T);
  return 0;
}
