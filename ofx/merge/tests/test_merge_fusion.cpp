// Focus Region and candidate fusion (merge_fusion.h, matte_merge.h FocusInput): the default fusion rule
// on known answers, fuseCandidates (no candidates, nullptr confidence, clamping, a swapped rule),
// prepareRegion (identity, inward feather, grow / shrink against brute force), merge() with a region on
// a synthetic scene (bit-identical where the weight is 0 and for an empty region, the weight blend
// exactly, never below the base, the region replaces the zone, the views), fusion inside merge() (a
// zero candidate changes nothing; a strand candidate is added inside the region only), snapInRegion.
// Mutations (2026-11-12, each made a case below fail): agreement taking min(a1, a2) instead of max;
// the penalty applied regardless of the confidences; the inward feather measured from weight < 1
// instead of weight 0 (feather case); the never-lower clamp removed (Zone Blend case); the zone from
// the Reach zone instead of the region; the alignment on the region instead of the automatic zone.
// Not caught: the region blend "wgt < 1" -> "wgt <= 1" (at
// weight 1 base + (v - base) rounds back to v on these scenes; the exact end is by construction).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "matte_merge.h"
#include "merge_fusion.h"
#include "merge_synth.h"

using namespace mr::merge;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

float rule(std::vector<float> a, std::vector<float> c, const FusionParams& p = FusionParams()) {
  return defaultFusionRule(a.data(), c.data(), int(a.size()), p);
}
bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

// [smoke] The default rule, known answers.
void testRule() {
  CHECK(rule({}, {}) == 0.0f);
  CHECK(near(rule({0.8f}, {0.5f}), 0.4f));                     // one candidate: alpha x confidence
  CHECK(near(rule({0.6f, 0.65f}, {0.5f, 0.5f}), 0.65f * 0.75f));  // agree: max alpha, c = 1 - 0.5 * 0.5
  FusionParams half;
  half.agreeBonus = 0.5f;
  CHECK(near(rule({0.6f, 0.65f}, {0.5f, 0.5f}, half), 0.65f * 0.625f));
  half.agreeBonus = 0.0f;  // no bonus: the best gated value (0.325), or max alpha x c1 (0.325)
  CHECK(near(rule({0.6f, 0.65f}, {0.5f, 0.5f}, half), 0.325f));
  CHECK(near(rule({0.9f, 0.1f}, {0.6f, 0.6f}), 0.54f * 0.5f));  // strong disagreement, low confidence
  CHECK(near(rule({0.9f, 0.1f}, {0.8f, 0.3f}), 0.72f));         // one confident: no penalty
  CHECK(near(rule({0.9f, 0.6f}, {0.6f, 0.6f}), 0.54f));         // moderate disagreement: the best
  CHECK(near(rule({0.2f, 0.9f, 0.85f}, {1.0f, 0.5f, 0.5f}), 0.9f * 0.75f));  // the top two (0.45, 0.425) agree
  // A candidate of zeros next to x with confidence 1 gives x for every x (the fusion-in-merge invariant).
  bool same = true;
  for (int k = 0; k <= 100; ++k) same = same && rule({k / 100.0f, 0.0f}, {1.0f, 1.0f}) == k / 100.0f;
  CHECK(same);
  // confFull (Hair Locks): output alpha x min(1, c / confFull); the choice still uses alpha x c; 0 = gated.
  FusionParams cf;
  cf.confFull = 0.5f;
  CHECK(near(rule({0.8f}, {0.7f}, cf), 0.8f));                 // confident: the alpha itself
  CHECK(near(rule({0.8f}, {0.25f}, cf), 0.4f));                // half way: half the alpha
  CHECK(near(rule({0.9f, 0.1f}, {0.6f, 0.6f}, cf), 0.45f));    // disagreement penalty on the output value
  CHECK(near(rule({0.6f, 0.65f}, {0.5f, 0.5f}, cf), 0.65f));   // agreement: c 0.75 -> full
  CHECK(near(rule({0.2f, 0.9f}, {1.0f, 0.1f}, cf), 0.2f));     // the choice by gated value (0.2 > 0.09)
  bool gated = true;  // confFull 0 (the default): one candidate gives alpha x confidence exactly
  for (int k = 0; k <= 20; ++k)
    for (int j = 0; j <= 20; ++j) gated = gated && rule({k / 20.0f}, {j / 20.0f}) == (k / 20.0f) * (j / 20.0f);
  CHECK(gated);
  std::printf("rule: known answers (one, agree, bonus, disagree, confident, top two, zero candidate, confFull)\n");
}

float maxRule(const float* a, const float*, int n, const FusionParams&) {
  float m = 0;
  for (int k = 0; k < n; ++k) m = std::max(m, a[k]);
  return m;
}

void testFuseMap() {
  const int w = 4, h = 2;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> a = {0.2f, 0.5f, 1.5f, nan, 0.0f, 1.0f, -1.0f, 0.3f};
  const std::vector<float> c = {1.0f, 0.5f, 1.0f, 1.0f, 1.0f, nan, 1.0f, 2.0f};
  std::vector<float> f(8, -1.0f);
  fuseCandidates(nullptr, 0, w, h, FusionParams(), f.data());
  CHECK(std::all_of(f.begin(), f.end(), [](float v) { return v == 0.0f; }));
  Candidate ca;
  ca.alpha = a.data();
  fuseCandidates(&ca, 1, w, h, FusionParams(), f.data());  // confidence nullptr = 1, alphas clamped
  const std::vector<float> want1 = {0.2f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.3f};
  CHECK(f == want1);
  ca.confidence = c.data();
  fuseCandidates(&ca, 1, w, h, FusionParams(), f.data());
  const std::vector<float> want2 = {0.2f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f};
  CHECK(f == want2);
  const std::vector<float> b(8, 0.9f);
  Candidate cs[2] = {ca, Candidate()};
  cs[1].alpha = b.data();
  fuseCandidates(cs, 2, w, h, FusionParams(), f.data(), maxRule);  // a swapped rule
  CHECK(std::all_of(f.begin(), f.end(), [](float v) { return v >= 0.9f; }) && f[2] == 1.0f);
  std::printf("fuseCandidates: none, nullptr confidence, clamping, a swapped rule\n");
}

void testPrepareRegion() {
  const int w = 40, h = 30;
  std::vector<float> m(size_t(w) * h, 0.0f), wt(m.size());
  for (int y = 5; y < 25; ++y)
    for (int x = 8; x < 32; ++x) m[size_t(y) * w + x] = (x == 8 || x == 31) ? 0.4f : 1.0f;
  m[0] = std::numeric_limits<float>::quiet_NaN();
  m[1] = 3.0f;
  prepareRegion(m.data(), w, h, 0, 0, wt.data());
  long bad = 0;
  for (size_t i = 0; i < m.size(); ++i) bad += wt[i] != (i == 0 ? 0.0f : i == 1 ? 1.0f : m[i]);
  CHECK(bad == 0);
  // Inward feather 4 px: zero stays zero, the far interior keeps the mask, the rim is lowered, never raised.
  m[0] = m[1] = 0.0f;
  prepareRegion(m.data(), w, h, 0, 4.0, wt.data());
  long outside = 0, raised = 0, lowered = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (m[i] == 0.0f) outside += wt[i] != 0.0f;
      raised += wt[i] > m[i];
      lowered += wt[i] < m[i];
      if (x >= 14 && x < 26 && y >= 11 && y < 19) CHECK(wt[i] == m[i]);
    }
  CHECK(outside == 0 && raised == 0 && lowered > 50);
  long featherBad = 0;  // brute force: m x smoothstep((d - 0.5) / 4), d = distance to the nearest m = 0 pixel
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double d = 1e30;
      for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx)
          if (m[size_t(yy) * w + xx] <= 0.0f) d = std::min(d, std::hypot(x - xx, y - yy));
      const double t = std::min(1.0, std::max(0.0, (d - 0.5) / 4.0));
      featherBad += std::fabs(wt[size_t(y) * w + x] - m[size_t(y) * w + x] * t * t * (3 - 2 * t)) > 1e-6;
    }
  CHECK(featherBad == 0);
  // Grow / shrink: a square of half-size r, against brute force.
  for (double g : {2.0, -2.0, 3.4}) {
    prepareRegion(m.data(), w, h, g, 0, wt.data());
    const int r = int(std::lround(std::fabs(g)));
    long diff = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float v = g > 0 ? 0.0f : 1.0f;
        for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
          for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx)
            v = g > 0 ? std::max(v, m[size_t(yy) * w + xx]) : std::min(v, m[size_t(yy) * w + xx]);
        diff += wt[size_t(y) * w + x] != v;
      }
    CHECK(diff == 0);
  }
  std::printf("prepareRegion: identity, inward feather, grow / shrink vs brute force\n");
}

// A soft disc region over part of the subject's edge (top of the head).
std::vector<float> discRegion(int w, int h, double cx, double cy, double r, double soft) {
  std::vector<float> m(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      m[size_t(y) * w + x] = float(std::min(1.0, std::max(0.0, (r - std::hypot(x - cx, y - cy)) / soft + 0.5)));
  return m;
}

struct Run {
  std::vector<float> out, added, zone, rejected;
  MergeInfo info;
};
Run runMerge(const mmsynth::MergeScene& s, const MergeParams& p, const FocusInput* f) {
  Run r;
  const size_t n = size_t(s.w) * s.h;
  r.out.resize(n);
  r.added.resize(n);
  r.zone.resize(n);
  r.rejected.resize(n);
  MergeViews v;
  v.added = r.added.data();
  v.zone = r.zone.data();
  v.rejected = r.rejected.data();
  merge(s.base.data(), s.detail.data(), s.rgbDisplay.data(), s.w, s.h, p, r.out.data(), v, &r.info, nullptr, f);
  return r;
}

// Where the subject's top edge is: the first row with base >= 0.5 in the middle column.
int topEdgeRow(const mmsynth::MergeScene& s) {
  for (int y = 0; y < s.h; ++y)
    if (s.base[size_t(y) * s.w + s.w / 2] >= 0.5f) return y;
  return s.h / 2;
}

// [smoke] merge() with a region.
void testMergeRegion() {
  const int W = 480, H = 270;
  const mmsynth::MergeScene s = mmsynth::makeMergeScene(W, H, false);
  const size_t n = size_t(W) * H;
  MergeParams p;
  scaleToHeight(p, HeightSizes(), H);
  const Run plain = runMerge(s, p, nullptr);
  FocusInput none;  // weight nullptr: no region
  CHECK(runMerge(s, p, &none).out == plain.out);
  // An empty region: the base, bit for bit.
  const std::vector<float> zero(n, 0.0f);
  FocusInput fz;
  fz.weight = zero.data();
  const Run rz = runMerge(s, p, &fz);
  CHECK(rz.out == s.base && rz.info.zonePixels == 0);
  // A soft disc over the top of the head.
  const int top = topEdgeRow(s);
  const std::vector<float> R = discRegion(W, H, W / 2.0, top, 0.12 * H, 6);
  std::vector<float> R1(n);
  for (size_t i = 0; i < n; ++i) R1[i] = R[i] > 0 ? 1.0f : 0.0f;
  FocusInput f, f1;
  f.weight = R.data();
  f1.weight = R1.data();
  const Run r = runMerge(s, p, &f), r1 = runMerge(s, p, &f1);
  long outsideChanged = 0, lowered = 0, inside = 0, blendBad = 0, plainOutside = 0;
  for (size_t i = 0; i < n; ++i) {
    if (R[i] <= 0) outsideChanged += r.out[i] != s.base[i], plainOutside += plain.out[i] != s.base[i];
    else inside += r.out[i] != s.base[i];
    lowered += r.out[i] < s.base[i];
    float want = R[i] >= 1 ? r1.out[i] : R[i] > 0 ? s.base[i] + R[i] * (r1.out[i] - s.base[i]) : s.base[i];
    want = std::max(want, s.base[i]);
    blendBad += r.out[i] != want;
  }
  std::printf("region: %ld px changed inside, %ld outside (no region: %ld outside the disc), %ld lowered, blend "
              "mismatches %ld\n", inside, outsideChanged, plainOutside, lowered, blendBad);
  CHECK(outsideChanged == 0 && lowered == 0 && blendBad == 0);
  CHECK(inside > 200 && plainOutside > 1000);
  CHECK(r.info.zonePixels == long(std::count_if(R.begin(), R.end(), [](float v) { return v > 0; })));
  // The alignment is the one without a region (estimated on the automatic zone).
  CHECK(r.info.align.model == plain.info.align.model && r.info.reg.tx == plain.info.reg.tx &&
        r.info.reg.ty == plain.info.reg.ty && r.info.reg.scale == plain.info.reg.scale);
  std::printf("align with the region: '%s' (without: '%s')\n", alignModelName(r.info.align.model),
              alignModelName(plain.info.align.model));
  // Views after the weight: zone = the region, added = out - base, rejected 0 outside.
  long vz = 0, va = 0, vr = 0;
  for (size_t i = 0; i < n; ++i) {
    vz += r.zone[i] != R[i];
    va += r.added[i] != r.out[i] - s.base[i];
    vr += R[i] <= 0 && r.rejected[i] != 0.0f;
  }
  CHECK(vz == 0 && va == 0 && vr == 0);
  // The region replaces the zone: with Reach 1 px (no Reach zone to speak of) the region still adds detail
  // more than 1 px beyond the base's edge.
  MergeParams tight = p;
  tight.reachPx = 1.0;
  const Run rt = runMerge(s, tight, &f);
  long far = 0;
  for (size_t i = 0; i < n; ++i) far += rt.out[i] != s.base[i] && s.base[i] <= 0.02f;
  CHECK(far > 100);
  // Zone Blend in a region (Matte edge): never below the base, unchanged outside.
  MergeParams zb = p;
  zb.method = Method::ZoneBlend;
  const Run rb = runMerge(s, zb, &f), rbPlain = runMerge(s, zb, nullptr);
  long zbLowered = 0, zbOutside = 0, plainLowered = 0;
  for (size_t i = 0; i < n; ++i) {
    zbLowered += rb.out[i] < s.base[i];
    zbOutside += R[i] <= 0 && rb.out[i] != s.base[i];
    plainLowered += rbPlain.out[i] < s.base[i];
  }
  std::printf("zone blend: lowered %ld with the region (%ld without), %ld changed outside; far detail %ld px\n",
              zbLowered, plainLowered, zbOutside, far);
  CHECK(zbLowered == 0 && zbOutside == 0 && plainLowered > 0);
}

// Fusion inside merge(): a candidate of zeros changes nothing (rule invariant, Evidence Off); a strand
// candidate rooted on the head is added inside the region and nowhere outside it.
void testFusionInMerge() {
  const int W = 480, H = 270;
  const mmsynth::MergeScene s = mmsynth::makeMergeScene(W, H, false);
  const size_t n = size_t(W) * H;
  MergeParams p;
  scaleToHeight(p, HeightSizes(), H);
  const int top = topEdgeRow(s);
  const std::vector<float> R = discRegion(W, H, W / 2.0, top, 0.12 * H, 6);
  FocusInput f;
  f.weight = R.data();
  const Run ref = runMerge(s, p, &f);
  const std::vector<float> zero(n, 0.0f);
  Candidate b;
  b.name = "zeros";
  b.alpha = zero.data();
  FocusInput fb = f;
  fb.extra = &b;
  fb.extraCount = 1;
  CHECK(runMerge(s, p, &fb).out == ref.out);
  // Two vertical 1 px strands from the head upwards, one inside the region, one far outside it.
  std::vector<float> strands(n, 0.0f);
  const int xs[2] = {W / 2 + 3, W / 2 + int(0.3 * W)};
  for (int k = 0; k < 2; ++k) {
    int y0 = 0;
    for (int y = 0; y < H; ++y)
      if (s.base[size_t(y) * W + xs[k]] >= 0.5f) {
        y0 = y;
        break;
      }
    for (int y = std::max(0, y0 - 18); y <= y0 + 2; ++y) strands[size_t(y) * W + xs[k]] = 0.9f;
  }
  b.alpha = strands.data();
  const Run rs = runMerge(s, p, &fb);
  long gainedIn = 0, outside = 0, lowered = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t i = size_t(y) * W + x;
      if (R[i] <= 0) outside += rs.out[i] != s.base[i];
      if (x == xs[0] && R[i] >= 1 && rs.out[i] > ref.out[i] + 0.2f) ++gainedIn;
      lowered += rs.out[i] < s.base[i];
    }
  std::printf("fusion: zero candidate identical; strand candidate raised %ld px inside, %ld changed outside\n",
              gainedIn, outside);
  CHECK(gainedIn >= 5 && outside == 0 && lowered == 0);
  // Evidence On with a candidate: the confidence path; still never lowers, nothing outside.
  MergeParams pe = p;
  pe.evidence = true;
  const Run re = runMerge(s, pe, &fb);
  long eo = 0, el = 0;
  for (size_t i = 0; i < n; ++i) eo += R[i] <= 0 && re.out[i] != s.base[i], el += re.out[i] < s.base[i];
  CHECK(eo == 0 && el == 0);
}

void testSnapInRegion() {
  const int W = 480, H = 270;
  const mmsynth::MergeScene s = mmsynth::makeMergeScene(W, H, false);
  const size_t n = size_t(W) * H;
  const std::vector<float> R = discRegion(W, H, W / 2.0, topEdgeRow(s), 0.12 * H, 6);
  std::vector<float> m = s.base;
  snapInRegion(s.base.data(), m.data(), s.rgbLinear.data(), R.data(), W, H, 1.0, snapParamsForHeight(H), true);
  long outside = 0, inside = 0, lowered = 0;
  for (size_t i = 0; i < n; ++i) {
    if (R[i] <= 0) outside += m[i] != s.base[i];
    else inside += m[i] != s.base[i];
    lowered += m[i] < s.base[i];
  }
  std::printf("snap in region: %ld px changed inside, %ld outside, %ld lowered\n", inside, outside, lowered);
  CHECK(outside == 0 && lowered == 0 && inside > 20);
  std::vector<float> m0 = s.base;  // strength 0: nothing
  snapInRegion(s.base.data(), m0.data(), s.rgbLinear.data(), R.data(), W, H, 0.0, snapParamsForHeight(H), true);
  CHECK(m0 == s.base);
  const mr::SnapParams sp = snapParamsForHeight(2160);
  CHECK(sp.radiusPx == 16.0 && sp.bandPx == 24.0 && snapParamsForHeight(90).radiusPx == 1.0);
}

}  // namespace

int main() {
  testRule();
  testFuseMap();
  testPrepareRegion();
  testMergeRegion();
  testFusionInMerge();
  testSnapInRegion();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
