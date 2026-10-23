// Smart Fill Holes (B3) core with fake segmentations: known answers on hand-made mattes (a hole
// in a segmentation-high region is filled, in a low one kept, a region touching the frame edge is
// never a hole, 4-connectivity), the colour score, Both, keepRealGaps off, the size safety, the
// runner called only when needed, abort, exceptions, the output against an independent reference
// (brute-force holes + rim) bit for bit, in place, determinism, the synthetic "gaps" scene with a
// perfect segmentation, and the BiRefNet pre / post pair against hand values. Prints HD timing.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "smart_fill.h"
#include "smart_fill_ref.h"
#include "synth.h"

using namespace mr::ml;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

struct MaskRunner : SegRunner {
  std::vector<float> mask;
  int calls = 0;
  void segment(const float*, int w, int h, float* m) override {
    ++calls;
    if (mask.size() != size_t(w) * h) throw std::runtime_error("size");
    std::copy(mask.begin(), mask.end(), m);
  }
};
struct ThrowRunner : SegRunner {
  void segment(const float*, int, int, float*) override { throw std::runtime_error("model failed"); }
};

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// A 24 x 14 frame: background border 2 px, solid 1.0 inside; hole A (3x3, 0) at the left, hole B
// (3x3, 0.2) at the right, a notch from the top edge (never a hole); subject red, background blue.
struct Tiny {
  int w = 24, h = 14;
  std::vector<float> m, rgb;
  Tiny() : m(size_t(w) * h, 0.0f), rgb(size_t(w) * h * 3, 0.0f) {
    for (int y = 2; y < h - 2; ++y)
      for (int x = 2; x < w - 2; ++x) m[size_t(y) * w + x] = 1.0f;
    for (int y = 0; y < 5; ++y) m[size_t(y) * w + 12] = 0.0f;  // the notch
    rect(4, 5, 3, 3, 0.0f);   // A
    rect(17, 5, 3, 3, 0.2f);  // B
    for (size_t i = 0; i < m.size(); ++i) paint(i, m[i] >= 0.5f ? 0.8f : 0.1f, 0.2f, m[i] >= 0.5f ? 0.1f : 0.7f);
  }
  void rect(int x0, int y0, int rw, int rh, float v) {
    for (int y = y0; y < y0 + rh; ++y)
      for (int x = x0; x < x0 + rw; ++x) m[size_t(y) * w + x] = v;
  }
  void paint(size_t i, float r, float g, float b) { rgb[i * 3] = r, rgb[i * 3 + 1] = g, rgb[i * 3 + 2] = b; }
  void paintRect(int x0, int y0, int rw, int rh, bool subject) {
    for (int y = y0; y < y0 + rh; ++y)
      for (int x = x0; x < x0 + rw; ++x)
        paint(size_t(y) * w + x, subject ? 0.8f : 0.1f, 0.2f, subject ? 0.1f : 0.7f);
  }
  // Segmentation: 1 left of column 12, 0 from it.
  std::vector<float> segLeft() const {
    std::vector<float> s(m.size());
    for (size_t i = 0; i < s.size(); ++i) s[i] = int(i % size_t(w)) < 12 ? 1.0f : 0.0f;
    return s;
  }
  bool in(int x0, int y0, int rw, int rh, size_t i) const {
    const int x = int(i % size_t(w)), y = int(i / size_t(w));
    return x >= x0 && x < x0 + rw && y >= y0 && y < y0 + rh;
  }
};

long referenceDiff(const std::vector<float>& m, int w, int h, const SmartFillParams& p, const SmartFillStats& st,
                   const std::vector<float>& out);

SmartFillParams exact(ScoreMode mode) {
  SmartFillParams p;
  p.mode = mode;
  p.rimPx = 0;
  p.ringPx = 2;
  p.bgReachPx = 8;
  return p;
}

void testTiny() {
  Tiny t;
  MaskRunner seg;
  seg.mask = t.segLeft();
  std::vector<float> out(t.m.size());
  SmartFillStats st;

  // Segmentation: A filled (seg 1), B kept (seg 0), the notch is not a hole.
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, exact(ScoreMode::Segmentation), seg, out.data(), &st);
  CHECK(st.holesFound == 2 && st.holesFilled == 1 && st.holesKept == 1 && seg.calls == 1 && st.segmented);
  CHECK(st.holes.size() == 2 && st.holes[0].area == 9 && st.holes[0].x0 == 4 && st.holes[0].y1 == 7);
  CHECK(st.holes[0].decision == HoleDecision::Filled && st.holes[1].decision == HoleDecision::KeptRealGap);
  CHECK(st.holes[0].segMean == 1.0f && st.holes[1].segMean == 0.0f && st.pixelsChanged == 9);
  long bad = 0;
  for (size_t i = 0; i < out.size(); ++i) bad += t.in(4, 5, 3, 3, i) ? out[i] != 1.0f : out[i] != t.m[i];
  CHECK(bad == 0);

  // Inverted segmentation: B filled, A kept.
  for (float& v : seg.mask) v = 1.0f - v;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, exact(ScoreMode::Segmentation), seg, out.data(), &st);
  bad = 0;
  for (size_t i = 0; i < out.size(); ++i) bad += t.in(17, 5, 3, 3, i) ? out[i] != 1.0f : out[i] != t.m[i];
  CHECK(bad == 0 && st.holesFilled == 1);

  // Threshold: mean 0.6 in both holes; 0.5 fills both, 0.7 none.
  seg.mask.assign(t.m.size(), 0.6f);
  SmartFillParams p = exact(ScoreMode::Segmentation);
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holesFilled == 2);
  p.subjectThreshold = 0.7;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holesFilled == 0 && out == t.m);

  // Mean / median / min: hole A with one pixel at 0 inside a 1.0 segmentation.
  seg.mask.assign(t.m.size(), 1.0f);
  seg.mask[size_t(6) * t.w + 5] = 0.0f;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(std::fabs(st.holes[0].segMean - 8.0f / 9) < 1e-6f && st.holes[0].segMedian == 1.0f &&
        st.holes[0].segMin == 0.0f);
  p.segStat = SegStatistic::Min;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holes[0].decision == HoleDecision::KeptRealGap && st.holes[1].decision == HoleDecision::Filled);

  // Colour: A shows subject colour, B background colour (the Tiny picture already does).
  Tiny c;
  c.paintRect(4, 5, 3, 3, true);
  c.paintRect(17, 5, 3, 3, false);
  MaskRunner none;
  smartFill(c.m.data(), c.rgb.data(), c.w, c.h, exact(ScoreMode::Colour), none, out.data(), &st);
  CHECK(none.calls == 0 && !st.segmented);
  CHECK(st.holes[0].decision == HoleDecision::Filled && st.holes[1].decision == HoleDecision::KeptRealGap);
  CHECK(st.holes[0].colour > 0.99f && st.holes[1].colour < 0.01f);
  c.paintRect(4, 5, 3, 3, false);  // A now shows the background: kept
  smartFill(c.m.data(), c.rgb.data(), c.w, c.h, exact(ScoreMode::Colour), none, out.data(), &st);
  CHECK(st.holesFilled == 0 && out == c.m);

  // Both: the segmentation says subject everywhere, the colour says B is background: only A.
  Tiny b;
  b.paintRect(4, 5, 3, 3, true);
  seg.mask.assign(b.m.size(), 1.0f);
  smartFill(b.m.data(), b.rgb.data(), b.w, b.h, exact(ScoreMode::Both), seg, out.data(), &st);
  CHECK(st.holes[0].decision == HoleDecision::Filled && st.holes[1].decision == HoleDecision::KeptRealGap);
  CHECK(st.holes[1].score == std::min(st.holes[1].segMean, st.holes[1].colour));

  // keepRealGaps off: both filled, no model call.
  seg.calls = 0;
  p = exact(ScoreMode::Segmentation);
  p.keepRealGaps = false;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holesFilled == 2 && seg.calls == 0);

  // Size safety: 9 px > 0.02 * 336 = 6.7: both too large, never filled, no model call.
  p = exact(ScoreMode::Segmentation);
  p.maxHoleFraction = 0.02;
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holesTooLarge == 2 && st.holesFilled == 0 && seg.calls == 0 && out == t.m);
  CHECK(st.holes[0].decision == HoleDecision::KeptTooLarge);
  p.maxHoleFraction = 0;  // no limit (as Max Hole % 0)
  smartFill(t.m.data(), t.rgb.data(), t.w, t.h, p, seg, out.data(), &st);
  CHECK(st.holesTooLarge == 0 && seg.calls == 1);

  // No hole: no model call.
  seg.calls = 0;
  std::vector<float> plain(t.m.size(), 1.0f);
  smartFill(plain.data(), t.rgb.data(), t.w, t.h, exact(ScoreMode::Segmentation), seg, out.data(), &st);
  CHECK(st.holesFound == 0 && seg.calls == 0 && out == plain);

  // Rim: solid 0.97 pixels around A lifted to 1.0, the 0.97 elsewhere untouched.
  Tiny r;
  r.rect(3, 4, 5, 5, 0.97f);
  r.rect(4, 5, 3, 3, 0.0f);
  r.rect(15, 10, 2, 1, 0.97f);
  seg.mask = r.segLeft();
  p = exact(ScoreMode::Segmentation);
  p.rimPx = 1;
  smartFill(r.m.data(), r.rgb.data(), r.w, r.h, p, seg, out.data(), &st);
  bad = 0;
  // Rim 1 px = the 4-neighbours of the hole: the 5x5 square without its corners (sqrt 2 away).
  auto corner = [&](size_t i) {
    const int x = int(i % size_t(r.w)), y = int(i / size_t(r.w));
    return (x == 3 || x == 7) && (y == 4 || y == 8);
  };
  for (size_t i = 0; i < out.size(); ++i)
    bad += r.in(3, 4, 5, 5, i) && !corner(i) ? out[i] != 1.0f : out[i] != r.m[i];
  CHECK(bad == 0 && st.pixelsChanged == 9 + 12);
  // Rim 0: the grey solid ring stays as it is (only the hole is written).
  p.rimPx = 0;
  smartFill(r.m.data(), r.rgb.data(), r.w, r.h, p, seg, out.data(), &st);
  bad = 0;
  for (size_t i = 0; i < out.size(); ++i) bad += r.in(4, 5, 3, 3, i) ? out[i] != 1.0f : out[i] != r.m[i];
  CHECK(bad == 0 && st.pixelsChanged == 9);
  // Rim 2 next to a kept 1-px hole C, 2 px right of A (a solid 0.97 pixel between): the rim lifts
  // solid pixels only.
  r.rect(8, 6, 1, 1, 0.3f);
  seg.mask.assign(r.m.size(), 0.0f);
  for (size_t i = 0; i < seg.mask.size(); ++i) seg.mask[i] = r.in(4, 5, 3, 3, i) ? 1.0f : 0.0f;
  p.rimPx = 2;
  smartFill(r.m.data(), r.rgb.data(), r.w, r.h, p, seg, out.data(), &st);
  CHECK(st.holesFound == 3 && st.holesFilled == 1 && out[size_t(6) * r.w + 8] == 0.3f);
  CHECK(referenceDiff(r.m, r.w, r.h, p, st, out) == 0);
}

// 4-connectivity: two hole pixels touching only diagonally are two holes; a pixel touching the
// frame-edge background only diagonally is a hole.
void testConnectivity() {
  const int w = 8, h = 8;
  std::vector<float> m(size_t(w) * h, 1.0f), rgb(size_t(w) * h * 3, 0.5f), out(m.size());
  m[size_t(3) * w + 3] = 0.0f;
  m[size_t(4) * w + 4] = 0.0f;
  m[0] = 0.0f;           // the corner: outside
  m[size_t(1) * w + 1] = 0.0f;  // diagonal to the corner: a hole
  MaskRunner seg;
  seg.mask.assign(m.size(), 1.0f);
  SmartFillStats st;
  smartFill(m.data(), rgb.data(), w, h, exact(ScoreMode::Segmentation), seg, out.data(), &st);
  CHECK(st.holesFound == 3 && st.holesFilled == 3);
  CHECK(out[0] == 0.0f && out[size_t(1) * w + 1] == 1.0f);
  const std::vector<sfref::RefHole> ref = sfref::findHoles(m.data(), w, h, 0.95f);
  CHECK(ref.size() == 3);
}

// The output against an independent reference: brute-force holes (flood fill), the decisions
// from the stats, the fill and rim recomputed per pixel. Bit for bit.
long referenceDiff(const std::vector<float>& m, int w, int h, const SmartFillParams& p, const SmartFillStats& st,
                   const std::vector<float>& out) {
  const std::vector<sfref::RefHole> ref = sfref::findHoles(m.data(), w, h, p.solidAbove);
  if (ref.size() != st.holes.size()) return -1;
  std::vector<float> e = m;
  std::vector<uint8_t> filled(m.size(), 0);
  for (size_t k = 0; k < ref.size(); ++k) {
    if (ref[k].area != st.holes[k].area) return -2;
    if (st.holes[k].decision != HoleDecision::Filled) continue;
    for (int i : ref[k].px) filled[size_t(i)] = 1, e[size_t(i)] = 1.0f;
  }
  const int R = int(std::ceil(p.rimPx));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (filled[i] || !(m[i] >= p.solidAbove) || p.rimPx <= 0) continue;
      bool near = false;
      for (int dy = -R; dy <= R && !near; ++dy)
        for (int dx = -R; dx <= R && !near; ++dx) {
          const int nx = x + dx, ny = y + dy;
          near = nx >= 0 && ny >= 0 && nx < w && ny < h && filled[size_t(ny) * w + nx] &&
                 dx * dx + dy * dy <= p.rimPx * p.rimPx;
        }
      if (near) e[i] = 1.0f;
    }
  long diff = 0;
  for (size_t i = 0; i < m.size(); ++i) diff += std::memcmp(&e[i], &out[i], 4) != 0;
  return diff;
}

// Per hole of the gaps scene: is it a real gap (label 3 inside)?
std::vector<bool> isGap(const synth::Scene& s, float thr) {
  std::vector<bool> g;
  for (const sfref::RefHole& r : sfref::findHoles(s.matte.data(), s.w, s.h, thr)) {
    bool gap = false;
    for (int i : r.px) gap = gap || s.label[size_t(i)] == synth::kRealGap;
    g.push_back(gap);
  }
  return g;
}

void testGaps() {
  const int w = 640, h = 360;
  for (unsigned seed = 1; seed <= 3; ++seed) {
    synth::Scene s = synth::makeGaps(w, h, seed);
    MaskRunner seg;
    seg.mask = sfref::fakeSegmentation(s.truth, w, h, 0, 1.0);
    const std::vector<bool> gap = isGap(s, 0.95f);
    for (ScoreMode mode : {ScoreMode::Segmentation, ScoreMode::Colour, ScoreMode::Both}) {
      SmartFillParams p;
      p.mode = mode;
      std::vector<float> out(s.matte.size());
      SmartFillStats st;
      smartFill(s.matte.data(), s.rgb.data(), w, h, p, seg, out.data(), &st);
      int wrong = 0, gaps = 0, defects = 0;
      for (size_t k = 0; k < gap.size() && k < st.holes.size(); ++k) {
        gaps += gap[k], defects += !gap[k];
        wrong += gap[k] == (st.holes[k].decision == HoleDecision::Filled);
      }
      long label2 = 0, label2Filled = 0, label3Filled = 0;
      for (size_t i = 0; i < out.size(); ++i) {
        if (s.label[i] == synth::kDefectHole) ++label2, label2Filled += out[i] >= 0.95f;
        if (s.label[i] == synth::kRealGap) label3Filled += out[i] != s.matte[i];
      }
      std::printf("gaps %dx%d seed %u mode %d: %d holes (%d gaps, %d defects), %d wrong, label-2 filled %ld/%ld, "
                  "label-3 changed %ld\n",
                  w, h, seed, int(mode), st.holesFound, gaps, defects, wrong, label2Filled, label2, label3Filled);
      CHECK(gap.size() == st.holes.size() && gaps >= 3 && defects >= 8);
      CHECK(wrong == 0 && label2Filled == label2 && label3Filled == 0);
      CHECK(referenceDiff(s.matte, w, h, p, st, out) == 0);
      SmartFillParams p0 = p;
      p0.rimPx = 0;
      std::vector<float> out0(s.matte.size());
      smartFill(s.matte.data(), s.rgb.data(), w, h, p0, seg, out0.data(), &st);
      CHECK(referenceDiff(s.matte, w, h, p0, st, out0) == 0);
      // In place: the same bits.
      std::vector<float> inPlace = s.matte;
      smartFill(inPlace.data(), s.rgb.data(), w, h, p, seg, inPlace.data());
      CHECK(sameBits(inPlace, out));
    }
  }
}

void testAbortAndErrors() {
  synth::Scene s = synth::makeGaps(320, 180, 2);
  MaskRunner seg;
  seg.mask = sfref::fakeSegmentation(s.truth, s.w, s.h, 0, 1.0);
  SmartFillParams p;
  p.mode = ScoreMode::Segmentation;
  for (int abortAt = 1; abortAt <= 2; ++abortAt) {
    int asks = 0;
    std::vector<float> out(s.matte.size(), -1.0f);
    SmartFillStats st;
    seg.calls = 0;
    smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, seg, out.data(), &st, [&]() { return ++asks >= abortAt; });
    CHECK(st.aborted && out == s.matte && seg.calls == abortAt - 1);
    std::vector<float> inPlace = s.matte;
    smartFill(inPlace.data(), s.rgb.data(), s.w, s.h, p, seg, inPlace.data(), &st, [] { return true; });
    CHECK(st.aborted && inPlace == s.matte);
  }
  ThrowRunner bad;
  bool threw = false;
  std::vector<float> out(s.matte.size());
  try {
    smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, bad, out.data());
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  // Colour mode never calls the runner, so a broken one is fine.
  p.mode = ScoreMode::Colour;
  SmartFillStats st;
  smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, bad, out.data(), &st);
  CHECK(!st.segmented && st.holesFilled > 0);
}

void testDeterminism() {
  synth::Scene s = synth::makeGaps(960, 540, 7);
  MaskRunner seg;
  seg.mask = sfref::fakeSegmentation(s.truth, s.w, s.h, 30, 1.5);
  SmartFillParams p;
  p.rimPx = 2.5;
  std::vector<float> a(s.matte.size()), b(s.matte.size());
  SmartFillStats sa, sb;
  smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, seg, a.data(), &sa);
  smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, seg, b.data(), &sb);
  CHECK(sameBits(a, b) && sa.holesFilled == sb.holesFilled);
  for (size_t k = 0; k < sa.holes.size(); ++k) CHECK(sa.holes[k].score == sb.holes[k].score);
  CHECK(referenceDiff(s.matte, s.w, s.h, p, sa, a) == 0);
}

void testPrePost() {
  // 1-D hand values (PIL BILINEAR): 4 -> 2 is a widened triangle; 2 -> 4 plain bilinear, edges held.
  const float a4[4] = {0, 1, 2, 3};
  float a2[2];
  resampleBilinear(a4, 4, 1, 1, a2, 2, 1);
  CHECK(std::fabs(a2[0] - 1.25f / 1.75f) < 1e-6f && std::fabs(a2[1] - (1 * 0.25f + 2 * 0.75f + 3 * 0.75f) / 1.75f) < 1e-6f);
  const float b2[2] = {0, 1};
  float b4[4];
  resampleBilinear(b2, 2, 1, 1, b4, 4, 1);
  CHECK(b4[0] == 0.0f && std::fabs(b4[1] - 0.25f) < 1e-7f && std::fabs(b4[2] - 0.75f) < 1e-7f && b4[3] == 1.0f);
  // Same size: the identity; vertical as horizontal; channels independent.
  std::vector<float> img(5 * 3 * 2), back(img.size());
  for (size_t i = 0; i < img.size(); ++i) img[i] = float(i % 7) / 7;
  resampleBilinear(img.data(), 5, 3, 2, back.data(), 5, 3);
  CHECK(img == back);
  const float col[4] = {0, 1, 2, 3};
  float col2[2];
  resampleBilinear(col, 1, 4, 1, col2, 1, 2);
  CHECK(col2[0] == a2[0] && col2[1] == a2[1]);

  // Prepare: the ImageNet mean gives 0; 1.0 and 2.0 (clamped) give (1 - mean) / std; NaN -> 0.
  const int w = 40, h = 30, S = kBiRefNetSize;
  std::vector<float> rgb(size_t(w) * h * 3), input(size_t(3) * S * S);
  for (size_t i = 0; i < size_t(w) * h; ++i) rgb[i * 3] = 0.485f, rgb[i * 3 + 1] = 0.456f, rgb[i * 3 + 2] = 0.406f;
  birefnetPrepare(rgb.data(), w, h, input.data());
  float mx = 0;
  for (float v : input) mx = std::max(mx, std::fabs(v));
  CHECK(mx < 1e-5f);
  for (size_t i = 0; i < size_t(w) * h; ++i) rgb[i * 3] = 1.0f, rgb[i * 3 + 1] = 2.0f, rgb[i * 3 + 2] = NAN;
  birefnetPrepare(rgb.data(), w, h, input.data());
  CHECK(std::fabs(input[0] - (1 - 0.485f) / 0.229f) < 1e-5f);
  CHECK(std::fabs(input[size_t(S) * S + 12345] - (1 - 0.456f) / 0.224f) < 1e-5f);
  CHECK(std::fabs(input[size_t(2) * S * S + S * S - 1] - (0 - 0.406f) / 0.225f) < 1e-5f);

  // Postprocess: logit 0 -> 0.5; +-10 halves -> ~1 / ~0, the boundary in the middle.
  std::vector<float> logits(size_t(S) * S, 0.0f), mask(size_t(w) * h);
  birefnetPostprocess(logits.data(), w, h, mask.data());
  CHECK(std::fabs(mask[0] - 0.5f) < 1e-6f && std::fabs(mask.back() - 0.5f) < 1e-6f);
  for (int y = 0; y < S; ++y)
    for (int x = 0; x < S; ++x) logits[size_t(y) * S + x] = x < S / 2 ? 10.0f : -10.0f;
  birefnetPostprocess(logits.data(), w, h, mask.data());
  CHECK(mask[size_t(10) * w + 5] > 0.9999f && mask[size_t(10) * w + 34] < 1e-4f);
  CHECK(std::fabs(mask[size_t(10) * w + 19] + mask[size_t(10) * w + 20] - 1.0f) < 1e-4f);  // symmetric
}

void timing() {
  synth::Scene s = synth::makeGaps(1920, 1080, 1);
  MaskRunner seg;
  seg.mask = sfref::fakeSegmentation(s.truth, s.w, s.h, 0, 1.0);
  std::vector<float> out(s.matte.size());
  for (ScoreMode mode : {ScoreMode::Segmentation, ScoreMode::Colour, ScoreMode::Both}) {
    SmartFillParams p;
    p.mode = mode;
    SmartFillStats st;
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
      smartFill(s.matte.data(), s.rgb.data(), s.w, s.h, p, seg, out.data(), &st);
      best = std::min(best, st.totalMs);
    }
    std::printf("HD mode %d: %.1f ms (label %.1f, seg copy %.1f, score %.1f, write %.1f), %d holes\n", int(mode), best,
                st.labelMs, st.segMs, st.scoreMs, st.writeMs, st.holesFound);
    CHECK(best < 200);
  }
  std::vector<float> input(size_t(3) * kBiRefNetSize * kBiRefNetSize), logits(size_t(kBiRefNetSize) * kBiRefNetSize, 1.0f);
  auto t0 = std::chrono::steady_clock::now();
  birefnetPrepare(s.rgb.data(), s.w, s.h, input.data());
  auto t1 = std::chrono::steady_clock::now();
  birefnetPostprocess(logits.data(), s.w, s.h, out.data());
  auto t2 = std::chrono::steady_clock::now();
  std::printf("HD BiRefNet prepare %.1f ms, postprocess %.1f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count());
}

}  // namespace

int main() {
  testTiny();
  testConnectivity();
  testGaps();
  testAbortAndErrors();
  testDeterminism();
  testPrePost();
  timing();
  std::printf(failures ? "%d FAILURES\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
