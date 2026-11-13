// Region trimap (ops/region_trimap.h): known answers on small frames, the soft region's intermediate
// values, the band against a brute-force distance, the frame-size (render scale) behaviour of the
// band, NaN / out-of-range region values, in place.
// Mutations (2026-11-12): fg test ">=" -> ">" (threshold case fails); band seeded from w > 0 instead
// of w >= 0.5 (band case fails); the soft formula with (0.5 + L) (soft case fails).
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "ops/region_trimap.h"

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// [smoke] Known answers, one pixel each: fg / bg / grey base, region 0 / 1 / 0.5 / 0.25.
void testKnown() {
  const int w = 8, h = 1;
  const float base[w] = {1.0f, 0.95f, 0.949f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f};
  const float reg[w] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.25f};
  float out[w];
  mr::regionTrimap(base, reg, w, h, mr::RegionTrimapParams(), out);
  const float want[w] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.25f, 0.875f};
  for (int i = 0; i < w; ++i) CHECK(out[i] == want[i]);
  // Out-of-range and NaN region values are clamped (NaN = 0).
  const float base2[4] = {1.0f, 0.0f, 1.0f, 0.0f};
  const float reg2[4] = {std::numeric_limits<float>::quiet_NaN(), -1.0f, 3.0f, 2.0f};
  float out2[4];
  mr::regionTrimap(base2, reg2, 4, 1, mr::RegionTrimapParams(), out2);
  CHECK(out2[0] == 1.0f && out2[1] == 0.0f && out2[2] == 0.5f && out2[3] == 0.5f);
  // Threshold is a parameter; in place (out = base).
  mr::RegionTrimapParams p;
  p.fgThreshold = 0.5f;
  std::vector<float> b = {0.5f, 0.49f, 0.7f};
  const std::vector<float> r = {0.0f, 0.0f, 1.0f};
  mr::regionTrimap(b.data(), r.data(), 3, 1, p, b.data());
  CHECK(b[0] == 1.0f && b[1] == 0.0f && b[2] == 0.5f);
  std::printf("known answers: labels, soft values, clamping, threshold, in place\n");
}

// A disc region (soft rim) on a frame with a solid subject: unknown exactly where the region is > 0
// (and 0.5 where it is 1); with a band, also within bandPx of the region's 0.5 line (brute force).
std::vector<float> discRegion(int w, int h, double cx, double cy, double r, double soft) {
  std::vector<float> m(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double v = (r - std::hypot(x - cx, y - cy)) / soft + 0.5;
      m[size_t(y) * w + x] = float(std::min(1.0, std::max(0.0, v)));
    }
  return m;
}

void testBand() {
  const int w = 90, h = 70;
  std::vector<float> base(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) base[size_t(y) * w + x] = x < 45 ? 1.0f : x == 45 ? 0.5f : 0.0f;
  const std::vector<float> reg = discRegion(w, h, 45, 35, 12, 4);
  std::vector<float> out(base.size());
  mr::RegionTrimapParams p;
  mr::regionTrimap(base.data(), reg.data(), w, h, p, out.data());
  long bad = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const float L = base[i] >= 0.95f ? 1.0f : 0.0f;
    if (reg[i] <= 0) bad += out[i] != L;
    else if (reg[i] >= 1) bad += out[i] != 0.5f;
    else bad += !(std::fabs(out[i] - 0.5f) < std::fabs(L - 0.5f));  // between the label and unknown
  }
  CHECK(bad == 0);
  p.bandPx = 6.5;
  mr::regionTrimap(base.data(), reg.data(), w, h, p, out.data());
  bad = 0;
  long bandPx = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double best = 1e30;
      for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx)
          if (reg[size_t(yy) * w + xx] >= 0.5f) best = std::min(best, std::hypot(x - xx, y - yy));
      const size_t i = size_t(y) * w + x;
      const bool inBand = best <= p.bandPx;
      const float L = base[i] >= 0.95f ? 1.0f : 0.0f;
      if (inBand) bad += out[i] != 0.5f, bandPx += reg[i] <= 0;
      else if (reg[i] <= 0) bad += out[i] != L;
    }
  CHECK(bad == 0 && bandPx > 100);
  std::printf("band 6.5 px: matches brute force, %ld px added outside the region\n", bandPx);
}

// Render scale: the caller scales bandPx with the image; the same scene at 2x with 2x band gives
// about 4x the unknown area (within 3 %).
void testScale() {
  long count[2] = {0, 0};
  for (int k = 0; k < 2; ++k) {
    const int s = k + 1, w = 160 * s, h = 120 * s;
    std::vector<float> base(size_t(w) * h, 0.0f);
    const std::vector<float> reg = discRegion(w, h, 80.0 * s, 60.0 * s, 20.0 * s, 3.0 * s);
    std::vector<float> out(base.size());
    mr::RegionTrimapParams p;
    p.bandPx = 8.0 * s;
    mr::regionTrimap(base.data(), reg.data(), w, h, p, out.data());
    for (float v : out) count[k] += v > 0.0f;
  }
  const double ratio = double(count[1]) / (4.0 * count[0]);
  std::printf("scale: unknown px %ld (1x) / %ld (2x), ratio to 4x %.4f\n", count[0], count[1], ratio);
  CHECK(std::fabs(ratio - 1.0) < 0.03);
}

}  // namespace

int main() {
  testKnown();
  testBand();
  testScale();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
