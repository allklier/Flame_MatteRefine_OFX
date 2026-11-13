// Temporal Stabilise (B4) core: the flow on known translations, a static subject's flickering
// pixel calmed, a translating subject keeping its edge (the warp compensates), flickering holes
// (filled by a majority, kept otherwise, never punched), bit-identity outside the region,
// determinism, K and the sequence ends, rejected neighbours (forward-backward check), and the
// synthetic sequence (synth::makeSequence) against its truth; the weight map (incl. the Hair
// Locks strand case), precomputed flows, and golden outputs of the code before both existed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
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

// The weight map: out = in + weight (stabilised - in). Constant 0 / 0.5 / 1, a split map, NaN and
// values outside 0-1, and a flickering hole under weights 0 / 0.5 / 1 and half covered.
void testWeightMap() {
  {
    const int w = 96, h = 80;
    const size_t n = size_t(w) * h;
    std::vector<Frame> fr(3, disc(w, h, 48, 40, 22));
    const size_t p = size_t(40) * w + 70, q = size_t(40) * w + 26;  // right and left edge
    fr[1].matte[p] = 0.95f;
    fr[1].matte[q] = 0.05f;
    auto v = view(fr);
    const std::vector<float>& in = fr[1].matte;
    StabiliseParams sp;
    std::vector<float> full(n), out(n), wt(n);
    StabiliseStats s1, s;
    stabilise(v.data(), 3, 1, w, h, sp, full.data(), &s1);
    CHECK(full != in);
    std::fill(wt.begin(), wt.end(), 0.0f);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &s, wt.data());
    CHECK(out == in && s.pixelsChanged == 0);
    std::fill(wt.begin(), wt.end(), 1.0f);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), nullptr, wt.data());
    CHECK(out == full);
    std::fill(wt.begin(), wt.end(), 0.5f);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), nullptr, wt.data());
    double maxErr = 0;
    for (size_t i = 0; i < n; ++i) maxErr = std::max(maxErr, std::fabs(out[i] - (in[i] + 0.5 * (full[i] - in[i]))));
    std::printf("weight 0.5: max |out - (in + 0.5 (stabilised - in))| %.2e; p %.3f -> %.3f (full %.3f)\n", maxErr, in[p],
                out[p], full[p]);
    CHECK(maxErr < 1e-6);
    // Left half 1 (values above 1 count as 1), right half 0 (negative and NaN count as 0).
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) wt[size_t(y) * w + x] = x < w / 2 ? (x % 2 ? 1.0f : 3.0f) : (x % 2 ? -1.0f : NAN);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &s, wt.data());
    long long badL = 0, badR = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        (x < w / 2 ? badL : badR) += out[i] != (x < w / 2 ? full[i] : in[i]);
      }
    std::printf("split weight: left differs from the full result at %lld px, right from the input at %lld px; region "
                "%lld of %lld px\n", badL, badR, s.regionPixels, s1.regionPixels);
    CHECK(badL == 0 && badR == 0);
    CHECK(s.regionPixels > 0 && s.regionPixels < s1.regionPixels);
    CHECK(std::fabs(out[q] - full[q]) == 0 && out[p] == in[p]);
  }
  {  // a hole in the centre only (filled at weight 1)
    const int w = 120, h = 100;
    const size_t n = size_t(w) * h, hc = size_t(50) * w + 60;
    std::vector<Frame> fr(3, disc(w, h, 60, 50, 35));
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (std::hypot(x - 60, y - 50) < 7) fr[1].matte[size_t(y) * w + x] = 0.2f;
    auto v = view(fr);
    const std::vector<float>& in = fr[1].matte;
    StabiliseParams sp;
    std::vector<float> full(n), out(n), wt(n, 0.0f);
    StabiliseStats st;
    stabilise(v.data(), 3, 1, w, h, sp, full.data());
    CHECK(full[hc] > 0.98f);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st, wt.data());
    CHECK(out == in && st.holesFilled == 0);
    std::fill(wt.begin(), wt.end(), 0.5f);
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st, wt.data());
    std::printf("hole, weight 0.5: %.3f -> %.3f (weight 1: %.3f), filled %d\n", in[hc], out[hc], full[hc], st.holesFilled);
    CHECK(st.holesFilled == 1 && std::fabs(out[hc] - (in[hc] + 0.5 * (full[hc] - in[hc]))) < 1e-6);
    // Weight 1 on the hole's left half only: the hole is still judged as a whole; the right half stays.
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) wt[size_t(y) * w + x] = x < 60 ? 1.0f : 0.0f;
    stabilise(v.data(), 3, 1, w, h, sp, out.data(), &st, wt.data());
    CHECK(st.holesFilled == 1);
    CHECK(out[hc - 3] == full[hc - 3] && out[hc + 3] == in[hc + 3]);
  }
}

// The design problem of docs/edge_recipes.md 5.3, kept as the regression test for the Hair Locks
// recipe: a thin strand present in the centre frame's matte only (a rebuilt strand; the pictures
// show it in every frame) is erased by the median at weight 1; with weight 0 over the Focus
// Region it survives bit for bit, while a flickering band pixel elsewhere is still put back.
void testStrandUnderWeight() {
  const int w = 128, h = 96;
  const size_t n = size_t(w) * h;
  std::vector<Frame> fr(3, disc(w, h, 50, 48, 24));
  std::vector<size_t> strand;
  for (int x = 74; x < 104; ++x) strand.push_back(size_t(40 + (x - 74) / 6) * w + x);  // 1 px, outwards
  for (Frame& f : fr)
    for (size_t i : strand)
      for (int c = 0; c < 3; ++c) f.rgb[i * 3 + c] = 0.05f;
  for (size_t i : strand) fr[1].matte[i] = 1.0f;
  const size_t flick = size_t(48 - 24) * w + 50;  // top edge, far from the strand
  const float flickTruth = fr[1].matte[flick];
  fr[1].matte[flick] = flickTruth > 0.5f ? 0.0f : 1.0f;
  auto v = view(fr);
  const std::vector<float> in = fr[1].matte;
  StabiliseParams sp;
  std::vector<float> out(n), wt(n, 1.0f);
  stabilise(v.data(), 3, 1, w, h, sp, out.data());
  double kept1 = 0;
  for (size_t i : strand) kept1 += out[i];
  kept1 /= strand.size();
  for (int y = 30; y < 52; ++y)
    for (int x = 70; x < 110; ++x) wt[size_t(y) * w + x] = 0.0f;  // the Focus Region around the strand
  stabilise(v.data(), 3, 1, w, h, sp, out.data(), nullptr, wt.data());
  long long strandChanged = 0, zoneChanged = 0;
  for (size_t i : strand) strandChanged += out[i] != in[i];
  for (size_t i = 0; i < n; ++i) zoneChanged += wt[i] == 0.0f && out[i] != in[i];
  std::printf("strand (one frame only): mean alpha at weight 1 %.3f (erased), at weight 0: %lld of %zu strand px "
              "changed, %lld px changed in the zone; flicker %.3f -> %.3f (truth %.3f)\n", kept1, strandChanged,
              strand.size(), zoneChanged, in[flick], out[flick], flickTruth);
  CHECK(kept1 < 0.1);
  CHECK(strandChanged == 0 && zoneChanged == 0);
  CHECK(std::fabs(out[flick] - flickTruth) < 0.05);
}

// Precomputed flows: the same outputs bit for bit (with a weight too, and flows of a larger K),
// the content key, and stale flows rejected before out is written.
void testPrecomputedFlows() {
  const int w = 480, h = 270;
  const size_t n = size_t(w) * h;
  synth::Sequence s = synth::makeSequence(w, h, 7, 2);
  std::vector<TemporalFrame> v;
  for (auto& f : s.frames) v.push_back({f.rgb.data(), f.matte.data()});
  std::vector<float> wt(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) wt[size_t(y) * w + x] = float(std::min(1.0, std::max(0.0, (x - 150.0) / 200.0)));
  std::vector<float> a(n), b(n);
  for (int K : {1, 2}) {
    StabiliseParams sp;
    sp.frames = K;
    sp.bandPx = 3;
    const TemporalFlows fl = computeFlows(&v[3 - K], 2 * K + 1, K, w, h, sp);
    CHECK(int(fl.flows.size()) == 2 * K && int(fl.offsets.size()) == 2 * K && fl.w == w && fl.h == h);
    for (const float* wp : {(const float*)nullptr, (const float*)wt.data()}) {
      stabilise(&v[3 - K], 2 * K + 1, K, w, h, sp, a.data(), nullptr, wp);
      stabilise(&v[3 - K], 2 * K + 1, K, w, h, sp, fl, b.data(), nullptr, wp);
      CHECK(a == b);
    }
    CHECK(fl.key == flowKey(&v[3 - K], 2 * K + 1, K, w, h, sp));
    CHECK(fl.key == computeFlows(&v[3 - K], 2 * K + 1, K, w, h, sp).key);
  }
  StabiliseParams p1, p2;
  p1.bandPx = p2.bandPx = 3;
  p2.frames = 2;
  const TemporalFlows f2 = computeFlows(&v[1], 5, 2, w, h, p2);
  stabilise(&v[2], 3, 1, w, h, p1, a.data());
  stabilise(&v[1], 5, 2, w, h, p1, f2, b.data());  // K = 2 flows, a K = 1 call
  CHECK(a == b);
  // Params that do not touch the flow (band, strength, combine, holes) may differ.
  StabiliseParams p3 = p1;
  p3.strength = 0.5, p3.combine = Combine::Mean, p3.bandPx = 5, p3.fillHoles = false;
  stabilise(&v[2], 3, 1, w, h, p3, a.data());
  stabilise(&v[1], 5, 2, w, h, p3, f2, b.data());
  CHECK(a == b);
  // The key follows the content: one neighbour's picture changed in a small patch.
  std::vector<float> rgb = s.frames[4].rgb;
  for (int y = 100; y < 110; ++y)
    for (int x = 200; x < 210; ++x) rgb[(size_t(y) * w + x) * 3] += 0.2f;
  std::vector<TemporalFrame> v2(v.begin() + 2, v.begin() + 5);
  v2[2].rgb = rgb.data();
  const TemporalFlows f1 = computeFlows(&v[2], 3, 1, w, h, p1);
  CHECK(flowKey(v2.data(), 3, 1, w, h, p1) != f1.key);
  CHECK(flowKey(&v[2], 3, 1, w, h, p1) == f1.key);
  // Stale flows: other resolution, other flow settings, a missing neighbour: rejected, out untouched.
  int rejected = 0, untouched = 0;
  auto expectThrow = [&](const TemporalFlows& f, const StabiliseParams& p, int count, const TemporalFrame* fr,
                         int centre) {
    std::fill(b.begin(), b.end(), -1.0f);
    try {
      stabilise(fr, count, centre, w, h, p, f, b.data());
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
    untouched += b[0] == -1.0f;
  };
  TemporalFlows other = f1;
  other.w = w / 2;
  expectThrow(other, p1, 3, &v[2], 1);
  StabiliseParams q = p1;
  q.flowLines = 100;  // another grid scale
  expectThrow(f1, q, 3, &v[2], 1);
  q = p1;
  q.fbMaxPx = 2.0;
  expectThrow(f1, q, 3, &v[2], 1);
  q = p1;
  q.flow.radius = 4;
  expectThrow(f1, q, 3, &v[2], 1);
  const std::vector<TemporalFrame> noPic = {{nullptr, v[2].matte}, {nullptr, v[3].matte}, {nullptr, v[4].matte}};
  expectThrow(f1, p1, 3, noPic.data(), 1);  // flows on the pictures, the call on the mattes
  expectThrow(f1, p2, 5, &v[1], 2);         // K = 1 flows, a K = 2 call
  std::printf("stale flows: %d of 6 rejected, out untouched in %d\n", rejected, untouched);
  CHECK(rejected == 6 && untouched == 6);
}

// FNV-1a 64 over the bytes of a matte.
unsigned long long hashOf(const std::vector<float>& m) {
  unsigned long long x = 1469598103934665603ull;
  const unsigned char* b = reinterpret_cast<const unsigned char*>(m.data());
  for (size_t i = 0; i < m.size() * sizeof(float); ++i) x = (x ^ b[i]) * 1099511628211ull;
  return x;
}

using Run = std::function<void(const TemporalFrame*, int, int, int, int, const StabiliseParams&, float*)>;

// Golden outputs: the hashes of stabilise() on the scenes above, recorded with the code before the
// weight map and the precomputed flows existed (x86-64, glibc 2.39, GCC, Release). A null weight
// must reproduce them bit for bit. Elsewhere (another libm or compiler) the inputs themselves may
// differ in the last bit, so the comparison is only made where they were recorded.
struct Golden {
  const char* name;
  std::vector<float> out;
};
std::vector<Golden> goldenOutputs(const Run& run) {
  std::vector<Golden> g;
  {  // static disc, two flickering band pixels, K = 1 and 2
    const int w = 96, h = 80;
    for (int K : {1, 2}) {
      std::vector<Frame> fr(2 * K + 1, disc(w, h, 48, 40, 22));
      fr[K].matte[size_t(40) * w + 70] = 0.95f;
      fr[K].matte[size_t(40) * w + 68] = 0.1f;
      auto v = view(fr);
      StabiliseParams sp;
      sp.frames = K;
      std::vector<float> out(size_t(w) * h);
      run(v.data(), int(v.size()), K, w, h, sp, out.data());
      g.push_back({K == 1 ? "static K1" : "static K2", out});
    }
  }
  {  // translating disc, K = 2, median and mean
    const int w = 128, h = 96;
    std::vector<Frame> fr;
    for (int t = -2; t <= 2; ++t) fr.push_back(disc(w, h, 60 + 3.0 * t, 48 - 1.0 * t, 24));
    fr[2].matte[size_t(48) * w + 84] = 0.0f;
    auto v = view(fr);
    for (Combine cb : {Combine::Median, Combine::Mean}) {
      StabiliseParams sp;
      sp.frames = 2;
      sp.combine = cb;
      std::vector<float> out(size_t(w) * h);
      run(v.data(), 5, 2, w, h, sp, out.data());
      g.push_back({cb == Combine::Median ? "translate median" : "translate mean", out});
    }
  }
  {  // a hole in the centre only (filled), K = 1
    const int w = 120, h = 100;
    std::vector<Frame> fr(3, disc(w, h, 60, 50, 35));
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (std::hypot(x - 60, y - 50) < 7) fr[1].matte[size_t(y) * w + x] = 0.2f;
    auto v = view(fr);
    StabiliseParams sp;
    std::vector<float> out(size_t(w) * h);
    run(v.data(), 3, 1, w, h, sp, out.data());
    g.push_back({"hole", out});
  }
  {  // the synthetic sequence 480x270, frames 2-6, K = 1 / 2, holes on; K = 1 without pictures; strength 0.6
    const int w = 480, h = 270;
    synth::Sequence s = synth::makeSequence(w, h, 7, 5);
    std::vector<TemporalFrame> v, vm;
    for (auto& f : s.frames) v.push_back({f.rgb.data(), f.matte.data()}), vm.push_back({nullptr, f.matte.data()});
    for (int K : {1, 2}) {
      StabiliseParams sp;
      sp.frames = K;
      sp.bandPx = 3;
      std::vector<float> out(size_t(w) * h);
      run(&v[4 - K], 2 * K + 1, K, w, h, sp, out.data());
      g.push_back({K == 1 ? "sequence K1" : "sequence K2", out});
    }
    StabiliseParams sp;
    sp.bandPx = 3;
    sp.strength = 0.6;
    std::vector<float> out(size_t(w) * h);
    run(&vm[3], 3, 1, w, h, sp, out.data());
    g.push_back({"sequence mattes only, strength 0.6", out});
  }
  return g;
}

void testGolden() {
  const unsigned long long stored[] = {0xf4fcc549a6cbb6f2ull, 0xf4fcc549a6cbb6f2ull, 0x5b468756f2a03cb5ull,
                                       0x64d2cf1c1a3c414full, 0x179d8ea18ebd7cf2ull, 0xf803df52a62fba75ull,
                                       0x571b6ab3089e497bull, 0xfcf045420b244985ull};
  // Four ways to the same outputs: no weight, a weight of 1, precomputed flows, both.
  std::vector<float> ones;
  auto weightOne = [&](int w, int h) { ones.assign(size_t(w) * h, 1.0f); return ones.data(); };
  const Run modes[4] = {
      [](const TemporalFrame* f, int c, int ce, int w, int h, const StabiliseParams& p, float* o) {
        stabilise(f, c, ce, w, h, p, o);
      },
      [&](const TemporalFrame* f, int c, int ce, int w, int h, const StabiliseParams& p, float* o) {
        stabilise(f, c, ce, w, h, p, o, nullptr, weightOne(w, h));
      },
      [](const TemporalFrame* f, int c, int ce, int w, int h, const StabiliseParams& p, float* o) {
        stabilise(f, c, ce, w, h, p, computeFlows(f, c, ce, w, h, p), o);
      },
      [&](const TemporalFrame* f, int c, int ce, int w, int h, const StabiliseParams& p, float* o) {
        stabilise(f, c, ce, w, h, p, computeFlows(f, c, ce, w, h, p), o, nullptr, weightOne(w, h));
      }};
  const char* modeName[4] = {"no weight", "weight 1", "precomputed flows", "precomputed flows + weight 1"};
  const std::vector<Golden> ref = goldenOutputs(modes[0]);
  CHECK(ref.size() == sizeof stored / sizeof stored[0]);
#if defined(__x86_64__) && defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ == 39
  const bool compareStored = true;
#else
  const bool compareStored = false;
  std::printf("golden: stored hashes not compared (recorded on x86-64 glibc 2.39)\n");
#endif
  for (size_t i = 0; i < ref.size() && i < sizeof stored / sizeof stored[0]; ++i) {
    if (compareStored && hashOf(ref[i].out) != stored[i])
      std::printf("golden %s: 0x%016llx, stored 0x%016llx\n", ref[i].name, hashOf(ref[i].out), stored[i]);
    CHECK(!compareStored || hashOf(ref[i].out) == stored[i]);
  }
  for (int m = 1; m < 4; ++m) {
    const std::vector<Golden> g = goldenOutputs(modes[m]);
    int same = 0;
    for (size_t i = 0; i < g.size(); ++i) same += g[i].out == ref[i].out;
    std::printf("golden, %s: %d of %zu outputs bit-identical to no weight\n", modeName[m], same, g.size());
    CHECK(same == int(ref.size()));
  }
}

}  // namespace

int main() {
  testGolden();
  testWeightMap();
  testStrandUnderWeight();
  testPrecomputedFlows();
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
