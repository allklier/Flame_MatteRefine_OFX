// Model candidate (model_candidate.h) with fake runners: crop geometry (multiple of 32, margin, centring, frame
// border, several crops for a big region), the tensors sent (normalisation constants, trimap values, unsharp,
// low-contrast switch), never lower, bit-identity outside the region, the piece filter, Strand Gain identity,
// size scaling with the delivered height, confidence, failures; and the RunnerClient adapter against the fake
// mr_ml_runner process (shapes and placement through the real protocol).
// Usage: test_model_candidate <test_fake_runner>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_candidate.h"
#include "runner_crops.h"

using namespace mr::recipe;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// Answers each crop with a function of the FRAME position (so any crop arrangement gives the same alpha) and
// records every call.
struct Call {
  std::string model;
  int cIn, h, w, cOut, x, y;
  std::vector<float> in;
};
struct FakeRunner : CropRunner {
  std::function<float(int, int)> alphaAt = [](int, int) { return 0.0f; };
  std::vector<Call> calls;
  bool fail = false;
  void run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut, int x,
           int y) override {
    if (fail) throw std::runtime_error("fake failure");
    calls.push_back(Call{modelId, cIn, h, w, cOut, x, y, std::vector<float>(in, in + size_t(cIn) * h * w)});
    for (int ly = 0; ly < h; ++ly)
      for (int lx = 0; lx < w; ++lx) out[size_t(ly) * w + lx] = alphaAt(x + lx, y + ly);
  }
};

struct Frame {
  int w, h;
  std::vector<float> rgb, base, region;
  Frame(int w_, int h_) : w(w_), h(h_), rgb(size_t(w_) * h_ * 3, 0.5f), base(size_t(w_) * h_, 0.0f),
                          region(size_t(w_) * h_, 0.0f) {}
  void rect(std::vector<float>& m, int x0, int y0, int x1, int y1, float v) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) m[size_t(y) * w + x] = v;
  }
};

bool sameBits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

ModelCandidateParams plain() {  // no picture processing: the tensors are easy to predict
  ModelCandidateParams p;
  p.unsharpAmount = 0;
  p.lowContrast = LowContrast::Off;
  p.sizeScale = 1;
  return p;
}

void testCropGeometry() {
  // A box in the middle of a HD frame: one crop, multiple of 32, at least box + 2 margin, centred, inside.
  auto c = planCrops(900, 400, 1000, 470, 1920, 1080, 32, 1024);
  CHECK(c.size() == 1);
  const CropRect& r = c[0];
  CHECK(r.size_w % 32 == 0 && r.size_h % 32 == 0);
  CHECK(r.size_w == 192 && r.size_h == 160);  // 100 + 64 -> 192, 70 + 64 -> 160
  CHECK(r.x <= 900 - 32 && r.x + r.size_w >= 1000 + 32 && r.y <= 400 - 32 && r.y + r.size_h >= 470 + 32);
  CHECK(std::abs((r.x + r.size_w / 2) - 950) <= 1 && std::abs((r.y + r.size_h / 2) - 435) <= 1);
  CHECK(r.cx0 == 900 && r.cx1 == 1000 && r.cy0 == 400 && r.cy1 == 470);
  // At the frame border: shifted inside the frame, still containing the box.
  c = planCrops(0, 1050, 40, 1080, 1920, 1080, 32, 1024);
  CHECK(c.size() == 1 && c[0].x == 0 && c[0].y + c[0].size_h == 1080 && c[0].size_w == 128 && c[0].size_h == 96);
  // A frame smaller than the crop: the frame rounded up to 32, from 0 (the edge repeats past it).
  c = planCrops(10, 10, 180, 180, 192, 200, 32, 1024);
  CHECK(c.size() == 1 && c[0].x == 0 && c[0].y == 0 && c[0].size_w == 192 && c[0].size_h == 224);
  // Bigger than one crop: equal cells that partition the box, each crop <= 1024 and a multiple of 32.
  c = planCrops(100, 50, 2100, 650, 2400, 1200, 32, 1024);
  CHECK(c.size() == 3);
  long long area = 0;
  for (const CropRect& q : c) {
    CHECK(q.size_w <= 1024 && q.size_h <= 1024 && q.size_w % 32 == 0 && q.size_h % 32 == 0);
    CHECK(q.x <= q.cx0 && q.x + q.size_w >= q.cx1 && q.y <= q.cy0 && q.y + q.size_h >= q.cy1);
    CHECK(q.x >= 0 && q.y >= 0 && q.x + q.size_w <= 2400 && q.y + q.size_h <= 1200);
    area += (long long)(q.cx1 - q.cx0) * (q.cy1 - q.cy0);
  }
  CHECK(area == 2000LL * 600);
  CHECK(c[0].cx0 == 100 && c[2].cx1 == 2100 && c[0].cx1 == c[1].cx0 && c[1].cx1 == c[2].cx0);
  std::printf("crop geometry: ok\n");
}

void testTensors() {
  Frame f(640, 360);
  for (size_t i = 0; i < f.base.size(); ++i) {
    f.rgb[i * 3] = 0.6f, f.rgb[i * 3 + 1] = 0.4f, f.rgb[i * 3 + 2] = 1.7f;  // blue above 1: clamped to 1
  }
  f.rect(f.base, 0, 0, 200, 360, 1.0f);      // solid core on the left
  f.rect(f.region, 150, 100, 300, 200, 1.0f);  // the region overlaps the core
  f.rect(f.region, 300, 100, 310, 200, 0.2f);  // a soft edge: quantised to background (0 + 0.2 x 0.5 = 0.1)
  f.rect(f.region, 310, 100, 320, 200, 0.8f);  // 0.4: unknown
  FakeRunner r;
  std::vector<float> a(f.base.size()), conf(f.base.size());
  ModelCandidateParams p = plain();
  ModelCandidateStats st;
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), conf.data(), &st);
  CHECK(r.calls.size() == 1);
  const Call& c = r.calls[0];
  CHECK(c.model == "vitmatte-small" && c.cIn == 4 && c.cOut == 1);
  CHECK(c.w % 32 == 0 && c.h % 32 == 0 && c.w == 256 && c.h == 192);  // 170 + 64 -> 256, 100 + 64 -> 192
  const size_t plane = size_t(c.w) * c.h;
  const float want[3] = {(0.6f - 0.485f) / 0.229f, (0.4f - 0.456f) / 0.224f, (1.0f - 0.406f) / 0.225f};
  bool norm = true, triOk = true;
  for (int ch = 0; ch < 3; ++ch)
    for (size_t i = 0; i < plane; ++i) norm = norm && c.in[ch * plane + i] == want[ch];
  CHECK(norm);
  for (int ly = 0; ly < c.h; ++ly)
    for (int lx = 0; lx < c.w; ++lx) {
      const int x = c.x + lx, y = c.y + ly;
      const float t = c.in[3 * plane + size_t(ly) * c.w + lx];
      const float expect = x < 200 ? 1.0f : (y >= 100 && y < 200 && x >= 150 && x < 300) ? 0.5f
                                          : (y >= 100 && y < 200 && x >= 310 && x < 320) ? 0.5f : 0.0f;
      triOk = triOk && t == expect;
    }
  CHECK(triOk);
  // Plugin 2's normalisation, per call.
  p.norm = Norm::Half;
  r.calls.clear();
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, nullptr);
  CHECK(r.calls.size() == 1 && r.calls[0].in[0] == (0.6f - 0.5f) / 0.5f && r.calls[0].in[2 * plane] == 1.0f);
  // Unsharp on a flat picture changes nothing (the blur of a constant is the constant, up to rounding).
  p.norm = Norm::ImageNet;
  p.unsharpAmount = 1;
  r.calls.clear();
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, nullptr);
  CHECK(std::fabs(r.calls[0].in[0] - want[0]) < 1e-5f);
  // Unsharp on a step: overshoot on both sides, clamped to 0-1 (sigma 1, amount 1).
  for (int y = 0; y < f.h; ++y)
    for (int x = 0; x < f.w; ++x)
      for (int ch = 0; ch < 3; ++ch) f.rgb[(size_t(y) * f.w + x) * 3 + ch] = x < 240 ? 0.3f : 0.6f;
  r.calls.clear();
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, nullptr);
  {
    const Call& q = r.calls[0];
    auto px = [&](int x) { return q.in[size_t(50) * q.w + (x - q.x)] * 0.229f + 0.485f; };  // red, back to 0-1
    CHECK(px(239) < 0.3f - 0.05f && px(240) > 0.6f + 0.05f);  // overshoot
    CHECK(std::fabs(px(200) - 0.3f) < 1e-4f && std::fabs(px(290) - 0.6f) < 1e-4f);  // far from the step: unchanged
  }
  std::printf("tensors: ok\n");
}

void testNeverLowerAndIdentity() {
  Frame f(400, 300);
  for (int y = 0; y < f.h; ++y)
    for (int x = 0; x < f.w; ++x) {
      f.base[size_t(y) * f.w + x] = float((x * 7 + y * 13) % 101) / 100.0f;  // all values 0-1
      f.region[size_t(y) * f.w + x] = (x >= 100 && x < 250 && y >= 80 && y < 200) ? float((x - 100) % 5) / 4.0f : 0.0f;
    }
  FakeRunner r;
  std::vector<float> a(f.base.size()), conf(f.base.size());
  ModelCandidateParams p = plain();
  p.minPiecePx = 0;
  // A model that answers 0 everywhere: nothing may change anywhere.
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), conf.data(), nullptr);
  CHECK(sameBits(a.data(), f.base.data(), a.size()));
  // A model with arbitrary values: outside the region bit-identical, inside never lower, the blend exact.
  r.alphaAt = [](int x, int y) { return float((x * 3 + y * 5) % 17) / 12.0f - 0.2f; };  // also < 0 and > 1
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), conf.data(), nullptr);
  bool outside = true, lowered = false, blend = true, confOut = true;
  for (int y = 0; y < f.h; ++y)
    for (int x = 0; x < f.w; ++x) {
      const size_t i = size_t(y) * f.w + x;
      const float wt = f.region[i], b = f.base[i];
      if (wt <= 0) {
        outside = outside && std::memcmp(&a[i], &b, 4) == 0;
        confOut = confOut && conf[i] == 0.0f;
        continue;
      }
      lowered = lowered || a[i] < b;
      const float m = std::min(std::max(r.alphaAt(x, y), 0.0f), 1.0f), mx = std::max(b, m);
      const float want = wt >= 1 ? mx : b + wt * (mx - b);
      blend = blend && a[i] == want;
    }
  CHECK(outside && !lowered && blend && confOut);
  // In place (alpha = base).
  std::vector<float> inPlace = f.base;
  modelCandidate(f.rgb.data(), inPlace.data(), f.region.data(), f.w, f.h, p, r, inPlace.data(), nullptr, nullptr);
  CHECK(sameBits(inPlace.data(), a.data(), a.size()));
  // Empty region: no call, alpha = base.
  std::vector<float> none(f.base.size(), 0.0f);
  r.calls.clear();
  modelCandidate(f.rgb.data(), f.base.data(), none.data(), f.w, f.h, p, r, a.data(), conf.data(), nullptr);
  CHECK(r.calls.empty() && sameBits(a.data(), f.base.data(), a.size()));
  // NaN region weight counts as 0; NaN model output as 0.
  std::vector<float> reg = f.region;
  reg[size_t(10) * f.w + 10] = NAN;
  r.alphaAt = [](int, int) { return NAN; };
  modelCandidate(f.rgb.data(), f.base.data(), reg.data(), f.w, f.h, p, r, a.data(), nullptr, nullptr);
  CHECK(sameBits(a.data(), f.base.data(), a.size()));
  // A failing runner throws.
  r.fail = true;
  bool threw = false;
  try {
    modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, nullptr);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("never lower and identity: ok\n");
}

// Pieces: a 3x3 blob (9 px), a 4x4 blob (16 px), a diagonal line of 14 px (8-connected only) and a 2x5 piece (10
// px) inside the region; a 4 px piece inside the region that touches a long line outside it (the model's values
// outside the region do not count: dropped); the base is 0.
void testPieces() {
  auto run = [](int w, int h, double sizeScale, double minPx, ModelCandidateStats* st) {
    Frame f(w, h);
    f.rect(f.region, 20, 20, 120, 120, 1.0f);
    f.rect(f.region, 130, 20, 140, 30, 1.0f);  // widens the box past the line
    FakeRunner r;
    r.alphaAt = [](int x, int y) {
      if (x >= 118 && x < 140 && y >= 40 && y < 42) return 0.9f;
      if (x >= 30 && x < 33 && y >= 30 && y < 33) return 0.9f;
      if (x >= 50 && x < 54 && y >= 30 && y < 34) return 0.9f;
      if (x >= 60 && x < 74 && y == x - 30) return 0.5f;
      if (x >= 90 && x < 92 && y >= 80 && y < 85) return 0.25f;
      if (x >= 100 && x < 102 && y >= 80 && y < 85) return 0.15f;  // below the threshold: never a piece, kept
      return 0.0f;
    };
    ModelCandidateParams p = plain();
    p.sizeScale = sizeScale;
    p.minPiecePx = minPx;
    std::vector<float> a(size_t(w) * h);
    modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), w, h, p, r, a.data(), nullptr, st);
    return a;
  };
  ModelCandidateStats st;
  std::vector<float> a = run(200, 150, 1.0, 12, &st);
  const int w = 200;
  CHECK(a[31 * w + 31] == 0.0f);                          // 9 px: dropped
  CHECK(a[31 * w + 51] == 0.9f);                          // 16 px: kept
  CHECK(a[35 * w + 65] == 0.5f);                          // diagonal 14 px: one 8-connected piece, kept
  CHECK(a[82 * w + 90] == 0.0f);                          // 10 px: dropped
  CHECK(a[82 * w + 100] == 0.15f);                        // under the threshold: untouched
  CHECK(a[40 * w + 118] == 0.0f && a[40 * w + 125] == 0.0f);  // 4 px inside: dropped; outside: the base
  CHECK(st.piecesRemoved == 3 && st.minPieceArea == 12.0);
  // Off: everything kept.
  a = run(200, 150, 1.0, 0, &st);
  CHECK(a[31 * w + 31] == 0.9f && st.piecesRemoved == 0);
  // Auto size scale (delivered height / 1080): at 540 lines the area is 12 / 4 = 3: the 9 px blob stays.
  a = run(200, 540, 0, 12, &st);
  CHECK(st.scale == 0.5 && st.minPieceArea == 3.0 && a[31 * w + 31] == 0.9f && st.piecesRemoved == 0);
  // dropSmallPieces alone: 4- vs 8-connectivity on a diagonal.
  std::vector<float> d(16 * 16, 0.0f);
  for (int i = 0; i < 12; ++i) d[size_t(i) * 16 + i] = 1.0f;
  CHECK(dropSmallPieces(d.data(), 16, 16, 0.2f, 12) == 0 && d[5 * 16 + 5] == 1.0f);
  CHECK(dropSmallPieces(d.data(), 16, 16, 0.2f, 13) == 1 && d[5 * 16 + 5] == 0.0f);
  std::printf("pieces: ok\n");
}

void testStrandGain() {
  Frame f(300, 200);
  f.rect(f.region, 50, 50, 250, 150, 1.0f);
  f.rect(f.base, 0, 0, 60, 200, 1.0f);
  FakeRunner r;
  r.alphaAt = [](int x, int y) { return float((x * 11 + y * 7) % 23) / 22.0f; };
  ModelCandidateParams p = plain();
  p.minPiecePx = 0;
  std::vector<float> a0(f.base.size()), a1(f.base.size()), a2(f.base.size());
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a0.data(), nullptr, nullptr);
  p.strandBlack = 0.0, p.strandGain = 1.0;
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a1.data(), nullptr, nullptr);
  CHECK(sameBits(a0.data(), a1.data(), a0.size()));
  // Identity = the raw model value (max with base 0) inside the region.
  CHECK(a0[size_t(100) * f.w + 100] == r.alphaAt(100, 100));
  p.strandBlack = 0.05, p.strandGain = 1.0 / 0.75;  // the research's levels 0.05-0.8
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a2.data(), nullptr, nullptr);
  bool ok = true;
  for (int y = 50; y < 150; ++y)
    for (int x = 60; x < 250; ++x) {
      const float m = r.alphaAt(x, y), want = std::min(std::max(float((m - 0.05) * (1.0 / 0.75)), 0.0f), 1.0f);
      ok = ok && a2[size_t(y) * f.w + x] == want;
    }
  CHECK(ok);
  std::printf("strand gain: ok\n");
}

void testSeveralCrops() {
  // A region wider than one crop: the result equals a single big crop's (the fake answers by frame position),
  // every crop within the model's limit.
  Frame f(2400, 400);
  f.rect(f.region, 100, 100, 2300, 300, 1.0f);
  f.rect(f.region, 1100, 100, 1300, 300, 0.0f);  // a gap in the middle: a cell may lose its region
  FakeRunner r;
  r.alphaAt = [](int x, int y) { return float((x + 2 * y) % 50) / 49.0f; };
  ModelCandidateParams p = plain();
  ModelCandidateStats st;
  std::vector<float> a(f.base.size()), b(f.base.size());
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, &st);
  CHECK(st.crops.size() == 3 && r.calls.size() == 3);
  for (const Call& c : r.calls) CHECK(c.w <= 1024 && c.h <= 1024 && c.w % 32 == 0 && c.h % 32 == 0);
  p.maxCropSide = 4096;
  r.calls.clear();
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, b.data(), nullptr, nullptr);
  CHECK(r.calls.size() == 1 && sameBits(a.data(), b.data(), a.size()));
  // A cell without region weight is skipped.
  std::fill(f.region.begin(), f.region.end(), 0.0f);
  f.rect(f.region, 100, 100, 300, 300, 1.0f);
  f.rect(f.region, 2100, 100, 2300, 300, 1.0f);
  r.calls.clear();
  p.maxCropSide = 1024;
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, &st);
  CHECK(st.crops.size() == 3 && r.calls.size() == 2);
  std::printf("several crops: ok\n");
}

void testBorderAndScale() {
  // A region touching the frame's corner: the crop stays inside, the result is right at the border pixels.
  Frame f(1920, 1080);
  f.rect(f.region, 0, 1000, 60, 1080, 1.0f);
  FakeRunner r;
  r.alphaAt = [](int x, int y) { return (x + y) % 3 == 0 ? 0.8f : 0.0f; };
  ModelCandidateParams p = plain();
  p.sizeScale = 0;  // auto: 1080 lines = 1
  p.minPiecePx = 0;
  ModelCandidateStats st;
  std::vector<float> a(f.base.size());
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, r, a.data(), nullptr, &st);
  CHECK(r.calls.size() == 1 && r.calls[0].x == 0 && r.calls[0].y + r.calls[0].h == 1080);
  CHECK(st.scale == 1.0 && st.margin == 32.0);
  CHECK(a[size_t(1079) * 1920 + 0] == r.alphaAt(0, 1079) && a[size_t(1079) * 1920 + 59] == r.alphaAt(59, 1079));
  // A half-size proxy of the same frame: margin and piece area scale (16 px, 3 px).
  Frame g(960, 540);
  g.rect(g.region, 400, 200, 500, 260, 1.0f);
  r.calls.clear();
  p.minPiecePx = 12;
  modelCandidate(g.rgb.data(), g.base.data(), g.region.data(), g.w, g.h, p, r, a.data(), nullptr, &st);
  CHECK(st.scale == 0.5 && st.margin == 16.0 && st.minPieceArea == 3.0);
  CHECK(r.calls[0].w == 160 && r.calls[0].h == 96);  // 100 + 32 -> 160, 60 + 32 -> 96
  std::printf("border and scale: ok\n");
}

void testLowContrast() {
  // Region luma: p1 / p99 of a two-level picture (strands and background inside the region).
  auto run = [](float bg, float hair, LowContrast mode, ModelCandidateStats* st) {
    Frame f(256, 256);
    for (int y = 0; y < 256; ++y)
      for (int x = 0; x < 256; ++x)
        for (int c = 0; c < 3; ++c) f.rgb[(size_t(y) * 256 + x) * 3 + c] = (x % 8 < 2) ? hair : bg;
    f.rect(f.region, 64, 64, 192, 192, 1.0f);
    FakeRunner r;
    ModelCandidateParams p = plain();
    p.lowContrast = mode;
    std::vector<float> a(f.base.size());
    modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), 256, 256, p, r, a.data(), nullptr, st);
    return r.calls[0].in[0] * 0.229f + 0.485f;  // the first red value the model saw, back to 0-1
  };
  ModelCandidateStats st;
  // Dark on dark (the research's shadow look: display 0.18 / 0.27): stretched. The research's mild stretch maps
  // p1 -> 0.05 and averages with the original: hair 0.5 x 0.18 + 0.5 x 0.05.
  float v = run(0.27f, 0.18f, LowContrast::Auto, &st);
  CHECK(st.stretched && std::fabs(st.lumaP1 - 0.18) < 1e-6 && std::fabs(st.lumaP99 - 0.27) < 1e-6);
  CHECK(std::fabs(v - (0.5f * 0.18f + 0.5f * 0.05f)) < 1e-5f);
  // Light hair on skin (0.45 / 0.58: low range but bright): not stretched (the stretch hurt light hair).
  run(0.58f, 0.45f, LowContrast::Auto, &st);
  CHECK(!st.stretched);
  // Normal contrast (dark hair on skin 0.19 / 0.60): not stretched.
  run(0.60f, 0.19f, LowContrast::Auto, &st);
  CHECK(!st.stretched);
  // Off / On override the rule.
  v = run(0.27f, 0.18f, LowContrast::Off, &st);
  CHECK(!st.stretched && std::fabs(v - 0.18f) < 1e-5f);
  run(0.60f, 0.19f, LowContrast::On, &st);
  CHECK(st.stretched);
  std::printf("low contrast: ok\n");
}

void testConfidence() {
  Frame f(200, 200);
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 200; ++x)
      for (int c = 0; c < 3; ++c) f.rgb[(size_t(y) * 200 + x) * 3 + c] = (x >= 99 && x < 102) ? 0.1f : 0.6f;  // a ridge
  f.rect(f.region, 40, 40, 160, 160, 1.0f);
  FakeRunner r;
  r.alphaAt = [](int x, int y) { return (x >= 99 && x < 102) ? 1.0f : (y < 100 ? 0.5f : 0.0f); };
  ModelCandidateParams p = plain();
  std::vector<float> a(f.base.size()), conf(f.base.size());
  modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), 200, 200, p, r, a.data(), conf.data(), nullptr);
  CHECK(conf[size_t(120) * 200 + 100] > 0.95f);        // strand on the ridge
  CHECK(conf[size_t(60) * 200 + 60] == 0.0f);          // model at 0.5
  CHECK(std::fabs(conf[size_t(150) * 200 + 60] - 1.0f) < 1e-6f);  // nothing on a flat picture: agree
  CHECK(conf[size_t(150) * 200 + 104] < 0.8f);         // nothing next to a ridge: less sure
  CHECK(conf[size_t(20) * 200 + 20] == 0.0f);          // outside the region
  std::printf("confidence: ok\n");
}

// The adapter over RunnerClient against the fake mr_ml_runner process (ofx/ml/tests/fake_runner.cpp: output =
// f(frame position) + input plane 0 + the last input plane): the crop's shape (H, W order) and place arrive.
void testRunnerClientAdapter(const std::string& fake) {
  char tmpl[] = "/tmp/mr-wp4-XXXXXX";
  if (!mkdtemp(tmpl)) {
    CHECK(false);
    return;
  }
  const std::string dir = tmpl, home = dir + "/home";
  mkdir(home.c_str(), 0700);
  mr::ml::RunnerClientOptions o;
  o.runnerPath = fake;
  o.socketPath = dir + "/runner.sock";
  o.name = "wp4";
  o.tileTimeoutS = 5;
  o.startTimeoutS = 5;
  o.idleSeconds = 30;
  o.env = {"HOME=" + home};
  {
    mr::ml::RunnerClient client(o);
    RunnerClientCrops crops(client);
    Frame f(300, 200);
    f.rect(f.region, 120, 60, 200, 100, 1.0f);
    ModelCandidateParams p = plain();
    p.minPiecePx = 0;
    ModelCandidateStats st;
    std::vector<float> a(f.base.size());
    try {
      modelCandidate(f.rgb.data(), f.base.data(), f.region.data(), f.w, f.h, p, crops, a.data(), nullptr, &st);
      const CropRect& c = st.crops.at(0);
      CHECK(c.size_w == 160 && c.size_h == 128);  // 80 + 64 -> 160, 40 + 64 -> 128 (W != H: the order matters)
      bool ok = true;
      const float in0 = (0.5f - 0.485f) / 0.229f;
      for (int y = 60; y < 100; ++y)
        for (int x = 120; x < 200; ++x) {
          const float raw = float((x * 31 + y * 17) % 1009) / 1009.0f + in0 + 0.5f;  // the trimap is 0.5 there
          ok = ok && std::fabs(a[size_t(y) * f.w + x] - std::min(raw, 1.0f)) < 1e-6f;
        }
      CHECK(ok);
    } catch (const std::exception& e) {
      std::printf("adapter: %s\n", e.what());
      CHECK(false);
    }
    client.quit(nullptr);
  }
  if (!failures && std::system(("rm -rf '" + dir + "'").c_str()) != 0) std::printf("could not remove %s\n", dir.c_str());
  std::printf("runner client adapter: ok\n");
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  CHECK(quantiseTrimap(0.0f) == 0.0f && quantiseTrimap(0.333f) == 0.0f && quantiseTrimap(0.334f) == 0.5f &&
        quantiseTrimap(0.666f) == 0.5f && quantiseTrimap(0.667f) == 1.0f && quantiseTrimap(NAN) == 0.0f);
  testCropGeometry();
  testTensors();
  testNeverLowerAndIdentity();
  testPieces();
  testStrandGain();
  testSeveralCrops();
  testBorderAndScale();
  testLowContrast();
  testConfidence();
  if (argc > 1) testRunnerClientAdapter(argv[1]);
  std::printf(failures ? "test_model_candidate: %d FAILED\n" : "test_model_candidate: all passed\n", failures);
  return failures ? 1 : 0;
}
