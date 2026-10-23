// Temporal Stabilise (B4) core: the flow on known translations, a static subject's flickering
// pixel calmed, a translating subject keeping its edge (the warp compensates), flickering holes
// (filled by a majority, kept otherwise, never punched), bit-identity outside the region,
// determinism, K and the sequence ends, rejected neighbours (forward-backward check), and the
// synthetic sequence (synth::makeSequence) against its truth.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ops/edge_field.h"
#include "synth.h"
#include "temporal.h"

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

double tex(double x, double y, double seed) {  // smooth, non-periodic enough texture, 0.2-0.8
  return 0.5 + 0.12 * std::sin(x * 0.21 + 2.0 * std::sin(y * 0.07 + seed)) +
         0.1 * std::sin(y * 0.17 + 1.5 * std::sin(x * 0.05 + 2 * seed)) + 0.06 * std::sin((x - y) * 0.33 + seed);
}

// A disc of radius r centred at (cx, cy), 2 px soft edge, its texture moving with it over a
// static background texture.
struct Frame {
  std::vector<float> rgb, matte;
};
Frame disc(int w, int h, double cx, double cy, double r, double bgSeed = 0.0) {
  Frame f;
  f.rgb.resize(size_t(w) * h * 3);
  f.matte.resize(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double a = std::min(1.0, std::max(0.0, 0.5 + (r - std::hypot(x - cx, y - cy)) / 2.0));
      const double fg = tex(x - cx, y - cy, 3.0), bg = tex(x, y, 7.0 + bgSeed);
      f.matte[i] = float(a);
      for (int c = 0; c < 3; ++c) f.rgb[i * 3 + c] = float(a * fg * (0.8 + 0.2 * c) + (1 - a) * bg * (1.1 - 0.2 * c));
    }
  return f;
}

std::vector<TemporalFrame> view(std::vector<Frame>& fr) {
  std::vector<TemporalFrame> v(fr.size());
  for (size_t i = 0; i < fr.size(); ++i) v[i] = {fr[i].rgb.data(), fr[i].matte.data()};
  return v;
}

void testFlowTranslation() {
  const int w = 200, h = 120;
  for (double t : {0.0, 1.5, 4.25, 9.0}) {
    std::vector<float> a(size_t(w) * h), b(a.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        a[size_t(y) * w + x] = float(tex(x, y, 1));
        b[size_t(y) * w + x] = float(tex(x - t, y + 0.5 * t, 1));
      }
    const Flow f = denseFlow(a.data(), b.data(), nullptr, nullptr, nullptr, w, h, FlowParams());
    double e = 0;
    long n = 0;
    for (int y = 20; y < h - 20; ++y)
      for (int x = 20; x < w - 20; ++x, ++n)
        e += std::hypot(f.u[size_t(y) * w + x] - t, f.v[size_t(y) * w + x] + 0.5 * t);
    std::printf("flow translation (%.2f, %.2f): EPE %.3f px\n", t, -0.5 * t, e / n);
    CHECK(e / n < 0.15);
  }
}

// A static disc; one band pixel flickers in the centre frame only: the median puts it back.
void testStaticFlicker() {
  const int w = 96, h = 80;
  for (int K : {1, 2}) {
    std::vector<Frame> fr(2 * K + 1, disc(w, h, 48, 40, 22));
    const size_t p = size_t(40) * w + 48 + 22;  // on the edge, value ~0.5
    const size_t q = size_t(40) * w + 48 + 20;  // 2 px inside, ~1
    const float origP = fr[K].matte[p], origQ = fr[K].matte[q];
    fr[K].matte[p] = 0.95f;
    fr[K].matte[q] = 0.1f;
    auto v = view(fr);
    StabiliseParams sp;
    sp.frames = K;
    std::vector<float> out(size_t(w) * h);
    StabiliseStats st;
    stabilise(v.data(), int(v.size()), K, w, h, sp, out.data(), &st);
    std::printf("static flicker K=%d: %.3f -> %.3f (truth %.3f), %.3f -> %.3f (truth %.3f), rejected %.1f%%\n", K,
                fr[K].matte[p], out[p], origP, fr[K].matte[q], out[q], origQ, 100 * st.rejectedFraction);
    CHECK(std::fabs(out[p] - origP) < 0.02);
    CHECK(std::fabs(out[q] - origQ) < 0.02);
    CHECK(st.neighbours == 2 * K);
    // Strength 0.5: half-way; strength 0: the input, bit for bit.
    sp.strength = 0.5;
    stabilise(v.data(), int(v.size()), K, w, h, sp, out.data());
    CHECK(std::fabs(out[p] - (0.95f + origP) / 2) < 0.02);
    sp.strength = 0;
    stabilise(v.data(), int(v.size()), K, w, h, sp, out.data());
    CHECK(out == fr[K].matte);
  }
}

// A disc translating (3, -1) px a frame, noise-free: the stabilised edge stays where the centre
// has it (median and mean: a mean of uncompensated neighbours would blur and move it), and
// flickering edge pixels of the centre are put back (a median of uncompensated neighbours keeps
// them: there the neighbours' edges are elsewhere).
void testTranslatingEdge() {
  const int w = 128, h = 96;
  for (int K : {1, 2})
    for (Combine cb : {Combine::Median, Combine::Mean}) {
    std::vector<Frame> fr;
    for (int t = -K; t <= K; ++t) fr.push_back(disc(w, h, 60 + 3.0 * t, 48 - 1.0 * t, 24));
    const std::vector<float> clean = fr[K].matte;
    const size_t px[3] = {size_t(48) * w + 60 + 24, size_t(48) * w + 60 - 24, size_t(48 - 24) * w + 60};
    auto v = view(fr);
    StabiliseParams sp;
    sp.frames = K;
    sp.combine = cb;
    std::vector<float> out(size_t(w) * h);
    for (size_t i : px) fr[K].matte[i] = fr[K].matte[i] > 0.5f ? 0.0f : 1.0f;
    stabilise(v.data(), int(v.size()), K, w, h, sp, out.data());
    double flick = 0;
    for (size_t i : px) flick = std::max(flick, double(std::fabs(out[i] - clean[i])));
    for (size_t i : px) fr[K].matte[i] = clean[i], out[i] = clean[i];
    if (cb == Combine::Median) CHECK(flick < 0.05);
    stabilise(v.data(), int(v.size()), K, w, h, sp, out.data());
    double maxDiff = 0, sumIn = 0, sumOut = 0, mxIn = 0, mxOut = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        maxDiff = std::max(maxDiff, double(std::fabs(out[i] - fr[K].matte[i])));
        sumIn += fr[K].matte[i], sumOut += out[i];
        mxIn += x * fr[K].matte[i], mxOut += x * out[i];
      }
    const double shift = mxOut / sumOut - mxIn / sumIn;
    std::printf("translating edge K=%d %s: max |out - in| %.4f, centroid shift %.4f px, area %+.2f%%, flicker pixels "
                "off by %.3f\n", K, cb == Combine::Median ? "median" : "mean", maxDiff, shift,
                100 * (sumOut - sumIn) / sumIn, flick);
    CHECK(maxDiff < 0.1);
    CHECK(std::fabs(shift) < 0.05);
    CHECK(std::fabs(sumOut - sumIn) / sumIn < 0.003);
  }
}

// Holes: a hole in the centre only is filled; one in the centre and in half the neighbours (K=2:
// 3 of 5 frames) stays; one in the neighbours only is not punched into the centre.
void testHoles() {
  const int w = 120, h = 100;
  auto punch = [&](Frame& f) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (std::hypot(x - 60, y - 50) < 7) f.matte[size_t(y) * w + x] = 0.2f;
  };
  const size_t hc = size_t(50) * w + 60;
  StabiliseParams sp;
  std::vector<float> out(size_t(w) * h);
  StabiliseStats st;
  {  // centre only, K=1
    std::vector<Frame> fr(3, disc(w, h, 60, 50, 35));
    punch(fr[1]);
    auto v = view(fr);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st);
    std::printf("hole in the centre only: %.3f -> %.3f, holes %d filled %d\n", fr[1].matte[hc], out[hc], st.holes,
                st.holesFilled);
    CHECK(st.holes == 1 && st.holesFilled == 1);
    CHECK(out[hc] > 0.98f);
    sp.fillHoles = false;  // the rule off: the hole's middle (beyond the band) stays
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st);
    CHECK(out[hc] == fr[1].matte[hc]);
    sp.fillHoles = true;
  }
  {  // K=2, hole in frames -2, 0, +2 (majority): kept; in -1, 0 only: filled
    std::vector<Frame> fr(5, disc(w, h, 60, 50, 35));
    punch(fr[0]), punch(fr[2]), punch(fr[4]);
    auto v = view(fr);
    sp.frames = 2;
    stabilise(v.data(), 5, 2, w, h, sp, out.data(), &st);
    std::printf("hole in 3 of 5: %.3f -> %.3f, filled %d\n", fr[2].matte[hc], out[hc], st.holesFilled);
    CHECK(st.holesFilled == 0 && out[hc] == fr[2].matte[hc]);
    std::vector<Frame> fr2(5, disc(w, h, 60, 50, 35));
    punch(fr2[1]), punch(fr2[2]);
    auto v2 = view(fr2);
    stabilise(v2.data(), 5, 2, w, h, sp, out.data(), &st);
    std::printf("hole in 2 of 5: %.3f -> %.3f, filled %d\n", fr2[2].matte[hc], out[hc], st.holesFilled);
    CHECK(st.holesFilled == 1 && out[hc] > 0.95f);
    sp.frames = 1;
  }
  {  // in both neighbours, not in the centre: the centre stays solid, bit for bit
    std::vector<Frame> fr(3, disc(w, h, 60, 50, 35));
    punch(fr[0]), punch(fr[2]);
    auto v = view(fr);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st);
    CHECK(out == fr[1].matte);
  }
}

// The synthetic sequence (small): bit-identity outside the region, determinism, K and the ends,
// and the scores against the truth.
void testSequence() {
  const int w = 960, h = 540;
  synth::Sequence s = synth::makeSequence(w, h, 9, 3);
  std::vector<TemporalFrame> v;
  for (auto& f : s.frames) v.push_back({f.rgb.data(), f.matte.data()});
  const size_t n = size_t(w) * h;
  for (int K : {1, 2}) {
    StabiliseParams sp;
    sp.frames = K;
    sp.bandPx = 6.0 * h / 1080;
    sp.fillHoles = false;  // so that the region is the band alone
    std::vector<std::vector<float>> out(9, std::vector<float>(n));
    for (int t = 0; t < 9; ++t) {
      const int a = std::max(0, t - K), b = std::min(8, t + K);
      stabilise(&v[a], b - a + 1, t - a, w, h, sp, out[t].data());
    }
    // Outside the band: bit-identical.
    long long outside = 0, changedInside = 0;
    for (int t = 0; t < 9; ++t) {
      const std::vector<float> sd = mr::signedEdgeDistance(s.frames[t].matte.data(), w, h, 0.5f, sp.bandPx);
      for (size_t i = 0; i < n; ++i) {
        if (std::fabs(sd[i]) >= sp.bandPx) outside += out[t][i] != s.frames[t].matte[i];
        else changedInside += out[t][i] != s.frames[t].matte[i];
      }
    }
    std::vector<std::vector<float>> in, truth;
    for (auto& f : s.frames) in.push_back(f.matte), truth.push_back(f.truth);
    const double band = 3.0 * h / 1080 + 0.5;
    const double f0 = synth::flicker(s, in, band), f1 = synth::flicker(s, out, band);
    const double e0 = synth::bandError(s, in, band), e1 = synth::bandError(s, out, band);
    std::printf("sequence 960x540 K=%d: outside changed %lld, inside changed %lld, flicker %.4f -> %.4f (truth %.4f), "
                "bandError %.4f -> %.4f\n", K, outside, changedInside, f0, f1, synth::flicker(s, truth, band), e0, e1);
    CHECK(outside == 0);
    CHECK(changedInside > 1000);
    CHECK(f1 < (K == 1 ? 0.65 : 0.5) * f0);
    CHECK(e1 < (K == 1 ? 0.85 : 0.75) * e0);
    // Determinism: the same call again, bit for bit; frames beyond K are ignored.
    std::vector<float> again(n);
    stabilise(&v[4 - K], 2 * K + 1, K, w, h, sp, again.data());
    CHECK(again == out[4]);
    if (K == 1) {
      stabilise(&v[2], 5, 2, w, h, sp, again.data());  // 5 frames given, K = 1: the middle 3 used
      CHECK(again == out[4]);
    }
  }
  // The ends: one frame = the input; two frames (centre first) = something between the two.
  StabiliseParams sp;
  sp.bandPx = 1.5;
  std::vector<float> o(n);
  StabiliseStats st;
  stabilise(&v[0], 1, 0, w, h, sp, o.data(), &st);
  CHECK(o == s.frames[0].matte && st.neighbours == 0);
  stabilise(&v[0], 2, 0, w, h, sp, o.data(), &st);
  CHECK(st.neighbours == 1 && st.pixelsChanged > 0);
  // In place (out = the centre matte) gives the same.
  std::vector<float> ref(n), m4 = s.frames[4].matte;
  stabilise(&v[3], 3, 1, w, h, sp, ref.data());
  TemporalFrame f3[3] = {v[3], {v[4].rgb, m4.data()}, v[5]};
  stabilise(f3, 3, 1, w, h, sp, m4.data());
  CHECK(m4 == ref);
}

// Holes counted against a plain flood fill (reference): random blobs and noise, 4-connectivity,
// border-connected regions excluded.
int referenceHoles(const std::vector<float>& m, int w, int h) {
  std::vector<int> seen(m.size(), 0), stack;
  int holes = 0;
  auto low = [&](size_t i) { return !(m[i] >= 0.5f); };
  for (int pass = 0; pass < 2; ++pass)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const bool border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
        const int i = y * w + x;
        if ((pass == 0) != border || seen[i] || !low(i)) continue;
        holes += pass == 1;
        stack.assign(1, i);
        seen[i] = 1;
        while (!stack.empty()) {
          const int j = stack.back();
          stack.pop_back();
          const int jx = j % w, jy = j / w;
          const int nb[4] = {jx > 0 ? j - 1 : -1, jx + 1 < w ? j + 1 : -1, jy > 0 ? j - w : -1, jy + 1 < h ? j + w : -1};
          for (int k : nb)
            if (k >= 0 && !seen[k] && low(k)) seen[k] = 1, stack.push_back(k);
        }
      }
  return holes;
}

void testHoleCount() {
  const int w = 160, h = 120;
  unsigned st = 12345;
  auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) / 16777216.0; };
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<Frame> fr(3, disc(w, h, 80, 60, 25 + trial));
    for (float& v : fr[1].matte)
      if (rnd() < 0.1 + 0.02 * trial) v = float(rnd());
    auto v = view(fr);
    StabiliseParams sp;
    std::vector<float> out(size_t(w) * h);
    StabiliseStats ss;
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &ss);
    const int ref = referenceHoles(fr[1].matte, w, h);
    if (ss.holes != ref) std::printf("trial %d: holes %d, reference %d\n", trial, ss.holes, ref);
    CHECK(ss.holes == ref);
    if (trial % 10 == 0) std::printf("hole count trial %d: %d holes (reference %d)\n", trial, ss.holes, ref);
  }
}

// A neighbour whose picture does not match (a different background, a cut): its reads fail the
// forward-backward check, so its matte (all 1 here) does not pull the band up.
void testRejectedNeighbour() {
  const int w = 120, h = 100;
  std::vector<Frame> fr(3, disc(w, h, 60, 50, 30));
  fr[2] = disc(w, h, 35, 70, 18, 5.0);
  std::fill(fr[2].matte.begin(), fr[2].matte.end(), 1.0f);
  for (size_t i = 0; i < fr[1].matte.size(); ++i) fr[1].matte[i] = std::min(1.0f, fr[1].matte[i] + 0.3f);  // grown
  auto v = view(fr);
  StabiliseParams sp;
  std::vector<float> out(size_t(w) * h);
  StabiliseStats st;
  stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st);
  // The band's mean change: towards frame 0 (lower), not towards frame 2's 1.
  double d = 0;
  long cnt = 0;
  for (size_t i = 0; i < out.size(); ++i)
    if (out[i] != fr[1].matte[i]) d += out[i] - fr[1].matte[i], ++cnt;
  std::printf("rejected neighbour: rejected %.1f%% of reads, mean change %+.4f over %ld px\n", 100 * st.rejectedFraction,
              cnt ? d / cnt : 0.0, cnt);
  CHECK(st.rejectedFraction > 0.4);
  CHECK(cnt > 100 && d / cnt < -0.05);
}

}  // namespace

int main() {
  testFlowTranslation();
  testStaticFlicker();
  testTranslatingEdge();
  testHoles();
  testSequence();
  testRejectedNeighbour();
  testHoleCount();
  if (failures) std::printf("%d failure(s)\n", failures);
  else std::printf("all temporal tests passed\n");
  return failures ? 1 : 0;
}
