// The synthetic scenes (ofx/tools/synth): sizes, ranges, each defect where it belongs, and the
// EXR writer's layout.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "synth.h"

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

void testScenes() {
  const int w = 480, h = 270;
  for (const auto& name : synth::sceneNames()) {
    const synth::Scene s = synth::make(name, w, h);
    CHECK(s.matte.size() == size_t(w) * h && s.truth.size() == s.matte.size());
    CHECK(s.rgb.size() == s.matte.size() * 3 && s.fg.size() == s.rgb.size());
    int outOfRange = 0, higher = 0, lower = 0;
    for (size_t i = 0; i < s.matte.size(); ++i) {
      outOfRange += s.matte[i] < 0 || s.matte[i] > 1 || s.truth[i] < 0 || s.truth[i] > 1;
      higher += s.matte[i] > s.truth[i] + 1e-6f;
      lower += s.matte[i] < s.truth[i] - 1e-6f;
    }
    std::printf("%-10s higher %6d lower %6d\n", name.c_str(), higher, lower);
    CHECK(outOfRange == 0);
    if (name == "holes" || name == "eroded") CHECK(higher == 0 && lower > 100);
    if (name == "specks") CHECK(lower == 0 && higher > 1000);
    if (name == "fgcolor") CHECK(higher == 0 && lower == 0);
    if (name != "fgcolor") CHECK(higher + lower > 100);
    // The real gap inside the arm on the hip (design point (-245, -20)) is background.
    const int gx = int(w / 2.0 - 245 * h / 1080.0), gy = int(h / 2.0 - 20 * h / 1080.0);
    CHECK(s.truth[size_t(gy) * w + gx] == 0.0f);
  }
  // Deterministic.
  CHECK(synth::make("all", 64, 36).matte == synth::make("all", 64, 36).matte);
}

void testExr(const std::string& dir) {
  const int w = 5, h = 3;
  std::vector<float> rgb(size_t(w) * h * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(i) / 10;
  const std::string path = dir + "/test_synth.exr";
  CHECK(synth::writeExr(path, w, h, 3, rgb));
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  uint32_t magic = 0;
  std::memcpy(&magic, b.data(), 4);
  CHECK(magic == 20000630);
  // The last line holds B, G, R of row 2: R of pixel (4, 2) is the file's last float.
  float last = 0;
  std::memcpy(&last, b.data() + b.size() - 4, 4);
  CHECK(last == rgb[(2 * w + 4) * 3]);
  // Header + offsets (8 per line) + lines (8 + w * 3 channels * 4 bytes).
  const size_t lines = size_t(h) * (8 + size_t(w) * 12);
  const size_t headerEnd = b.size() - lines - 8 * size_t(h);
  CHECK(b[headerEnd - 1] == 0);
  uint64_t first = 0;
  std::memcpy(&first, b.data() + headerEnd, 8);
  CHECK(first == headerEnd + 8 * size_t(h));
  std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  testScenes();
  testExr(argc > 1 ? argv[1] : ".");
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
