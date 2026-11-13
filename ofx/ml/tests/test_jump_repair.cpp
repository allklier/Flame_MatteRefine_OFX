// Jump Repair core (jump_repair.h): known answers on small hand-built frames (an added blob, a
// removed one, a soft rim pulled in by the zone, the feather weight, blobs over 2 and 3 frames
// against Span, below the threshold, below the minimum size, neighbours that disagree, the sequence
// ends, in place = out of place), the guards (a fast object passing through kept, repaired with the
// guard off), the synthetic sequence (synth::makeJumpSequence: every injected jump repaired for its
// span, the moving figure, the fast disc and the swinging bar untouched, nothing changed away from
// the jumps), the same decisions at three resolutions, edge jitter, the Motion Guards switch (on = the
// pinned results, off = every blob repaired), the hysteresis growth (a soft rim taken in, the step
// limit, the threshold, a moving edge stops it; soft-rim jumps of 2-8 px), and a timing check at 2880x1620.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "jump_repair.h"
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

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// A clip of T small frames, all `bg`; rect() paints frames [f0, f1].
struct Clip {
  int w, h, T;
  std::vector<std::vector<float>> f;
  Clip(int w_, int h_, int T_, float bg) : w(w_), h(h_), T(T_), f(size_t(T_), std::vector<float>(size_t(w_) * h_, bg)) {}
  void rect(int f0, int f1, int x0, int y0, int rw, int rh, float v) {
    for (int t = f0; t <= f1; ++t)
      for (int y = y0; y < y0 + rh; ++y)
        for (int x = x0; x < x0 + rw; ++x) f[size_t(t)][size_t(y) * w + x] = v;
  }
  // The core's frame table for frame t (neighbours inside [lo, hi] only).
  void table(int t, const float** fr, int lo = 0, int hi = -1) const {
    if (hi < 0) hi = T - 1;
    for (int k = -kJumpReach; k <= kJumpReach; ++k)
      fr[k + kJumpReach] = t + k >= lo && t + k <= hi ? f[size_t(t + k)].data() : nullptr;
  }
  std::vector<float> run(int t, const JumpParams& p, JumpStats* st = nullptr, int lo = 0, int hi = -1) const {
    const float* fr[2 * kJumpReach + 1];
    table(t, fr, lo, hi);
    std::vector<float> out(size_t(w) * h);
    repairJumps(fr, w, h, p, out.data(), st);
    return out;
  }
};

JumpParams small() {  // for 64x48 frames: min size 16 px, zone 1 + 1 px
  JumpParams p;
  p.minSizePercent = 0.5;
  p.growPx = p.featherPx = 1.0;
  return p;
}

void testKnownAnswers() {
  const int w = 64, h = 48, T = 9, t = 4;
  // Added: a 5x5 square of 1 in frame 4 only, over a background of 0.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t, 20, 20, 5, 5, 1.0f);
    JumpStats st;
    const std::vector<float> out = c.run(t, small(), &st);
    CHECK(st.judged && st.minArea == 16 && st.flaggedPixels == 25 && st.blobs.size() == 1 && st.repaired == 1);
    CHECK(sameBits(out, std::vector<float>(size_t(w) * h, 0.0f)));
    CHECK(st.pixelsChanged == 25 && st.maxChange == 1.0);
    const JumpBlob& b = st.blobs[0];
    CHECK(b.added && b.area == 25 && b.x0 == 20 && b.y0 == 20 && b.x1 == 24 && b.y1 == 24 && b.before == -1 &&
          b.after == 1 && b.mass == 25.0 && b.decision == JumpDecision::Repaired);
    // The other frames: nothing to repair (frame 3 sees a removed jump? no: its neighbours 2 and 4
    // disagree at the square, so nothing is flagged).
    for (int u = 1; u < T - 1; ++u) {
      if (u == t) continue;
      JumpStats s2;
      CHECK(sameBits(c.run(u, small(), &s2), c.f[size_t(u)]) && s2.flaggedPixels == 0);
    }
  }
  // Removed: a solid 20x20 block in every frame, a 5x5 hole in frame 4: filled with 1.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(0, T - 1, 10, 10, 20, 20, 1.0f);
    c.rect(t, t, 18, 18, 5, 5, 0.0f);
    JumpStats st;
    const std::vector<float> out = c.run(t, small(), &st);
    CHECK(st.repaired == 1 && !st.blobs[0].added && st.pixelsChanged == 25);
    CHECK(sameBits(out, c.f[0]));
  }
  // A soft blob: core 0.8 (flagged), rim 0.3 one pixel around it (not flagged, but beyond both
  // references in the blob's direction and inside the zone: pulled to the reference too); a pixel
  // of 0.3 two pixels out (feather: weight (2 - 2) / 1 = 0: untouched) and with a wider feather. The
  // zone and feather rules alone: growth off here (testGrowth covers it).
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t, 19, 19, 7, 7, 0.3f);
    c.rect(t, t, 20, 20, 5, 5, 0.8f);
    c.f[size_t(t)][size_t(22) * w + 18] = 0.3f;  // 2 px left of the core's left column (x 20)
    JumpStats st;
    JumpParams p0 = small();
    p0.growMaxPx = 0;
    std::vector<float> out = c.run(t, p0, &st);
    CHECK(st.flaggedPixels == 25 && st.repaired == 1);
    std::vector<float> want(size_t(w) * h, 0.0f);
    want[size_t(22) * w + 18] = 0.3f;
    // The rim's corners are sqrt(2) from the core: weight (1 + 1 - sqrt 2) / 1.
    const float wc = float(2.0 - std::sqrt(2.0));
    for (int y : {19, 25})
      for (int x : {19, 25}) want[size_t(y) * w + x] = 0.3f + wc * (0.0f - 0.3f);
    CHECK(sameBits(out, want));
    JumpParams p = p0;
    p.featherPx = 2.0;  // distance 2: weight (1 + 2 - 2) / 2 = 0.5
    out = c.run(t, p, &st);
    CHECK(out[size_t(22) * w + 18] == 0.3f - 0.5f * 0.3f && out[size_t(22) * w + 19] == 0.0f);
  }
  // Below the threshold (0.45 < 0.5) and below the minimum size (15 < 16 px): untouched.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t, 5, 5, 6, 6, 0.45f);
    c.rect(t, t, 40, 30, 5, 3, 1.0f);
    JumpStats st;
    CHECK(sameBits(c.run(t, small(), &st), c.f[size_t(t)]) && st.flaggedPixels == 15 && st.smallBlobs == 1 &&
          st.blobs.empty() && st.pixelsChanged == 0);
    JumpParams p = small();
    p.threshold = 0.4;
    p.minSizePercent = 0.4;  // 13 px
    CHECK(!sameBits(c.run(t, p, &st), c.f[size_t(t)]) && st.repaired == 2);
  }
  // Neighbours that disagree (the square is in frame 5 too): not judged a jump.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t + 1, 20, 20, 5, 5, 1.0f);
    JumpStats st;
    CHECK(sameBits(c.run(t, small(), &st), c.f[size_t(t)]) && st.flaggedPixels == 0);
    // ... a gentle disagreement within Agree (0.1 < 0.15) is still a jump.
    Clip d(w, h, T, 0.0f);
    d.rect(t, t, 20, 20, 5, 5, 1.0f);
    d.rect(t + 1, t + 1, 20, 20, 5, 5, 0.1f);
    CHECK(std::fabs(d.run(t, small(), &st)[size_t(22) * w + 22] - 0.05f) < 1e-6f && st.repaired == 1);  // the mean of 0 and 0.1
  }
  // References that disagree by more than Agree although frame t lies beyond both: a square
  // fading in (0 before t, 1 at t, 0.4 after): not a jump.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t, 20, 20, 5, 5, 1.0f);
    c.rect(t + 1, T - 1, 20, 20, 5, 5, 0.4f);
    JumpStats st;
    CHECK(sameBits(c.run(t, small(), &st), c.f[size_t(t)]) && st.flaggedPixels == 0);
  }
  // The zone changes a pixel only beyond both references in the blob's direction: next to an
  // added blob, a strip the neighbours have at 0.4 and frame t at 0.2 (lower: the other way)
  // stays; the same strip at 0.6 in frame t (higher, not flagged: 0.2 < 0.5) is pulled to 0.4.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(0, T - 1, 25, 20, 1, 5, 0.4f);
    c.rect(t, t, 25, 20, 1, 5, 0.2f);
    c.rect(t, t, 20, 20, 5, 5, 1.0f);
    JumpStats st;
    std::vector<float> out = c.run(t, small(), &st);
    CHECK(st.repaired == 1 && out[size_t(22) * w + 25] == 0.2f && out[size_t(22) * w + 22] == 0.0f);
    c.rect(t, t, 25, 20, 1, 5, 0.6f);
    out = c.run(t, small(), &st);
    CHECK(st.repaired == 1 && out[size_t(22) * w + 25] == 0.4f);
  }
  // The ends: frame 0 (no t - 1) and the last frame (no t + 1), and a missing neighbour inside the
  // clip: not judged, the input bit for bit.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(0, 0, 20, 20, 5, 5, 1.0f);
    c.rect(T - 1, T - 1, 20, 20, 5, 5, 1.0f);
    c.rect(t, t, 20, 20, 5, 5, 1.0f);
    JumpStats st;
    CHECK(sameBits(c.run(0, small(), &st), c.f[0]) && !st.judged && st.pixelsChanged == 0);
    CHECK(sameBits(c.run(T - 1, small(), &st), c.f[size_t(T - 1)]) && !st.judged);
    CHECK(sameBits(c.run(t, small(), &st, t, T - 1), c.f[size_t(t)]) && !st.judged);  // t - 1 missing
    CHECK(sameBits(c.run(t, small(), &st, 0, t), c.f[size_t(t)]) && !st.judged);      // t + 1 missing
    c.run(t, small(), &st, t - 1, t + 1);  // only t +- 1: judged
    CHECK(st.judged && st.repaired == 1);
  }
  // In place gives the same result.
  {
    Clip c(w, h, T, 0.0f);
    c.rect(t, t, 20, 20, 5, 5, 1.0f);
    c.rect(t, t, 40, 10, 6, 6, 0.9f);
    const std::vector<float> want = c.run(t, small());
    const float* fr[2 * kJumpReach + 1];
    c.table(t, fr);
    std::vector<float> inPlace = c.f[size_t(t)];
    fr[kJumpReach] = inPlace.data();
    repairJumps(fr, w, h, small(), inPlace.data());
    CHECK(sameBits(inPlace, want));
  }
  std::printf("known answers: ok\n");
}

// Blobs lasting 2 and 3 frames: repaired only when Span allows it, at every frame of the span.
void testSpans() {
  const int w = 64, h = 48, T = 14;
  for (int len : {2, 3}) {
    Clip c(w, h, T, 0.0f);
    const int f0 = 5, f1 = f0 + len - 1;
    c.rect(f0, f1, 20, 20, 5, 5, 1.0f);       // added
    c.rect(0, T - 1, 40, 10, 15, 15, 1.0f);   // and a removed one inside a static block
    c.rect(f0, f1, 44, 14, 5, 5, 0.0f);
    for (int span = 1; span <= 3; ++span) {
      JumpParams p = small();
      p.maxSpan = span;
      for (int t = f0; t <= f1; ++t) {
        JumpStats st;
        const std::vector<float> out = c.run(t, p, &st);
        if (span >= len) {
          CHECK(st.repaired == 2 && sameBits(out, c.f[0]));
          CHECK(st.blobs.size() == 2 && st.blobs[0].after - st.blobs[0].before - 1 == len);
        } else {
          CHECK(st.repaired == 0 && sameBits(out, c.f[size_t(t)]));
        }
      }
      // The frames around the span stay as they are.
      for (int t : {f0 - 1, f1 + 1}) CHECK(sameBits(c.run(t, p), c.f[size_t(t)]));
    }
    // Unequal references: linear in time. The background after the span at 0.1 under the added
    // blob: at the span's first frame (pair (1, len)) the value is 0 + 1 / (1 + len) x 0.1.
    {
      Clip u = c;
      u.rect(f1 + 1, T - 1, 20, 20, 5, 5, 0.1f);
      JumpParams p = small();
      p.maxSpan = len;
      const std::vector<float> out = u.run(f0, p);
      CHECK(std::fabs(out[size_t(22) * w + 22] - 0.1f / float(1 + len)) < 1e-6f);
    }
    // A span of 2-3 frames needs still surroundings: next to an edge that moves 3 px a frame
    // (rows 26-31, just below the added blob), the added blob is kept, the block's hole repaired.
    Clip m = c;
    for (int t = 0; t < T; ++t) m.rect(t, t, 2, 26, 10 + 3 * t, 6, 1.0f);
    JumpParams p = small();
    p.maxSpan = 3;
    JumpStats st;
    m.run(f0, p, &st);
    CHECK(st.repaired == 1 && st.keptNotStill == 1);  // the block's hole (far from the edge) is repaired
  }
  std::printf("spans: ok\n");
}

// A small square passing through fast (12 px a frame, 6 px wide): flagged in every frame, kept as
// Moved; with the guard off it would be erased.
void testPassThrough() {
  const int w = 96, h = 96, T = 7;
  Clip c(w, h, T, 0.0f);
  for (int t = 0; t < T; ++t) c.rect(t, t, 8 + 12 * t, 20, 6, 8, 1.0f);
  JumpParams p = small();
  p.minSizePercent = 0.4;  // 37 px of 9216: the 48 px square counts
  for (int t = 1; t < T - 1; ++t) {
    JumpStats st;
    CHECK(sameBits(c.run(t, p, &st), c.f[size_t(t)]) && st.keptMoved == 1 && st.repaired == 0);
    CHECK(st.blobs.size() == 1 && st.blobs[0].moveScore >= 0.99 && st.blobs[0].moveBeforeDx == -12 &&
          st.blobs[0].moveAfterDx == 12);
  }
  p.motionGuard = false;
  JumpStats st;
  CHECK(!sameBits(c.run(3, p, &st), c.f[3]) && st.repaired == 1);
  // A blob displaced the same way against both neighbours (they agree, t is off) is a jump.
  Clip d(w, h, T, 0.0f);
  d.rect(0, T - 1, 40, 20, 6, 8, 1.0f);
  d.rect(3, 3, 40, 20, 6, 8, 0.0f);
  d.rect(3, 3, 52, 20, 6, 8, 1.0f);
  p.motionGuard = true;
  d.run(3, p, &st);
  CHECK(st.repaired == 2 && st.keptMoved == 0);
  std::printf("pass-through: ok\n");
}

// The synthetic sequence at w x h: per frame and Span, which pixels changed, against the jumps.
struct SeqResult {
  std::vector<int> repaired;     // per frame (Span 1)
  long long changed = 0;         // Span 1, all frames
  int defectsRepaired[4] = {0, 0, 0, 0};  // per Span 1-3 (index), the jump frames repaired
  long long stray = 0, legitTouched = 0;  // changed pixels away from every jump / in legit[t]
  double worst = 0, worstMean = 0;  // max and mean |out - truth| inside a jump Span covers (outside the figure's edge band)
  double msMax = 0;
};
SeqResult runSequence(const synth::JumpSequence& s, int spanMax = 3) {
  SeqResult r;
  const size_t n = size_t(s.w) * s.h;
  const double k = s.h / 1080.0, zone = 2 * std::max(1.0, s.h / 540.0) + 1.5;
  std::vector<float> out(n);
  r.repaired.assign(size_t(s.frames), 0);
  for (int span = 1; span <= spanMax; ++span) {
    JumpParams p;
    p.maxSpan = span;
    for (int t = 0; t < s.frames; ++t) {
      const float* fr[2 * kJumpReach + 1];
      for (int q = -kJumpReach; q <= kJumpReach; ++q)
        fr[q + kJumpReach] = t + q >= 0 && t + q < s.frames ? s.matte[size_t(t + q)].data() : nullptr;
      JumpStats st;
      repairJumps(fr, s.w, s.h, p, out.data(), &st);
      r.msMax = std::max(r.msMax, st.totalMs);
      if (span == 1) {
        r.repaired[size_t(t)] = st.repaired;
        r.changed += st.pixelsChanged;
      }
      for (size_t i = 0; i < n; ++i) {
        if (out[i] == s.matte[size_t(t)][i]) continue;
        const int x = int(i % size_t(s.w)), y = int(i / size_t(s.w));
        bool near = false;
        for (const synth::JumpDefect& d : s.defects)
          near = near || (t >= d.first && t <= d.last && std::hypot(x - d.cx, y - d.cy) <= d.r + zone);
        r.stray += !near;
        r.legitTouched += s.legit[size_t(t)][i] != 0;
      }
      for (const synth::JumpDefect& d : s.defects) {
        if (t < d.first || t > d.last) continue;
        double worst = 0, sum = 0;
        long long cnt = 0;
        for (int y = int(d.cy - d.r - 2); y <= int(d.cy + d.r + 2); ++y)
          for (int x = int(d.cx - d.r - 2); x <= int(d.cx + d.r + 2); ++x) {
            const size_t i = size_t(y) * s.w + x;
            const double a = s.truth[size_t(t)][i];
            // Not judged by design: pixels the moving figure's edge crosses within t +- 2 (jump 2:
            // the references there disagree or are not confirmed by the frames beyond them).
            bool moving = false;
            for (int q = std::max(0, t - 2); q <= std::min(s.frames - 1, t + 2); ++q)
              moving = moving || s.truth[size_t(q)][i] != a;
            if (moving) continue;
            worst = std::max(worst, std::fabs(double(out[i]) - a));
            sum += std::fabs(double(out[i]) - a);
            ++cnt;
          }
        const int len = d.last - d.first + 1;
        // Repaired: the jump gone up to the feathered rim (a soft rim pixel 1-2 px from the
        // flagged core keeps part of its value).
        const double mean = cnt ? sum / cnt : 0.0;
        if (worst < 0.25 && mean < 0.01) ++r.defectsRepaired[span];
        if (len <= span) {
          r.worst = std::max(r.worst, worst);
          r.worstMean = std::max(r.worstMean, mean);
        }
      }
    }
  }
  (void)k;
  return r;
}

void testSequence() {
  const synth::JumpSequence s = synth::makeJumpSequence(960, 540);
  const SeqResult r = runSequence(s);
  // Jump frames: 1 + 1 + 1 (span 1), 2 (span 2), 3 (span 3).
  std::printf("sequence 960x540: jump frames repaired per Span 1/2/3: %d %d %d of 8, worst error max %.4f mean %.4f, stray %lld, "
              "legit touched %lld, changed (Span 1) %lld px, max %.1f ms\n",
              r.defectsRepaired[1], r.defectsRepaired[2], r.defectsRepaired[3], r.worst, r.worstMean, r.stray, r.legitTouched,
              r.changed, r.msMax);
  CHECK(r.defectsRepaired[1] == 3 && r.defectsRepaired[2] == 5 && r.defectsRepaired[3] == 8);
  CHECK(r.worst < 0.25 && r.worstMean < 0.01 && r.stray == 0 && r.legitTouched == 0);
  // The bar's turning frames and the fast disc were seen and kept (Span 1).
  JumpParams p;
  int kept = 0, keptMoved = 0, keptTurning = 0;
  std::vector<float> out(size_t(s.w) * s.h);
  for (int t = 1; t + 1 < s.frames; ++t) {
    const float* fr[2 * kJumpReach + 1];
    for (int q = -kJumpReach; q <= kJumpReach; ++q)
      fr[q + kJumpReach] = t + q >= 0 && t + q < s.frames ? s.matte[size_t(t + q)].data() : nullptr;
    JumpStats st;
    repairJumps(fr, s.w, s.h, p, out.data(), &st);
    kept += st.keptMoved + st.keptTurning + st.keptNotStill;
    keptMoved += st.keptMoved;
    keptTurning += st.keptTurning;
  }
  std::printf("sequence 960x540, Span 1: blobs kept %d (moved %d, turning %d)\n", kept, keptMoved, keptTurning);
  CHECK(keptMoved >= 16 && keptTurning >= 3);  // the disc in frames 1-18 (while in the frame), the bar at 4, 8, 12, 16
}

// The same decisions at three resolutions; changed pixels scale with the area.
void testResolutions() {
  const synth::JumpSequence a = synth::makeJumpSequence(480, 270), b = synth::makeJumpSequence(960, 540),
                            c = synth::makeJumpSequence(1920, 1080);
  const SeqResult ra = runSequence(a, 1), rb = runSequence(b, 1), rc = runSequence(c, 1);
  std::printf("resolutions 480x270 / 960x540 / 1920x1080 (Span 1): changed %lld / %lld / %lld px, stray %lld %lld "
              "%lld\n",
              ra.changed, rb.changed, rc.changed, ra.stray, rb.stray, rc.stray);
  CHECK(ra.repaired == rb.repaired && rb.repaired == rc.repaired);
  CHECK(ra.stray == 0 && rb.stray == 0 && rc.stray == 0 && ra.legitTouched + rb.legitTouched + rc.legitTouched == 0);
  CHECK(rb.changed > 3.0 * ra.changed && rb.changed < 5.0 * ra.changed);
  CHECK(rc.changed > 3.0 * rb.changed && rc.changed < 5.0 * rb.changed);
}

// Edge jitter (+-1.5 px at 1080, independent per frame) on every edge: the jumps are still repaired;
// what else changes is reported (a jitter bump outside both neighbours is a small jump itself).
void testJitter() {
  const synth::JumpSequence s = synth::makeJumpSequence(960, 540, 1.5);
  const SeqResult r = runSequence(s, 1);
  std::printf("jitter 1.5 px: jump frames repaired (Span 1) %d, stray changed %lld px, legit touched %lld, changed "
              "%lld\n",
              r.defectsRepaired[1], r.stray, r.legitTouched, r.changed);
  CHECK(r.defectsRepaired[1] >= 3);
}

// Hysteresis growth on small frames: a soft rim around a flagged core is taken in up to growMaxPx
// steps where frame t lies beyond the agreeing references by more than growThreshold; it stops below
// the threshold, at the step limit, and where the references disagree (a moving edge).
void testGrowth() {
  const int w = 64, h = 48, T = 9, t = 4;
  JumpParams p = small();
  p.growMaxPx = 3;
  // A 5x5 core of 1 at x 20-24, y 20-24; rim rows y 20-24 to the right: x 25 0.3, 26 0.1, 27 0.05, 28 and
  // 29 0.01 (below growThreshold 0.02).
  Clip c(w, h, T, 0.0f);
  c.rect(t, t, 20, 20, 5, 5, 1.0f);
  const float rim[5] = {0.3f, 0.1f, 0.05f, 0.01f, 0.01f};
  for (int k = 0; k < 5; ++k) c.rect(t, t, 25 + k, 20, 1, 5, rim[k]);
  auto at = [&](const std::vector<float>& o, int x) { return o[size_t(22) * w + x]; };
  JumpStats st;
  std::vector<float> out = c.run(t, p, &st);
  // Grown: x 25-27 (15 px); x 28 is 1 px from the grown set (full weight) and x 29 2 px (weight 0).
  CHECK(st.repaired == 1 && st.grownPixels == 15 && st.blobs[0].grown == 15);
  CHECK(at(out, 25) == 0.0f && at(out, 26) == 0.0f && at(out, 27) == 0.0f && at(out, 28) == 0.0f && at(out, 29) == 0.01f);
  // No growth (the behaviour before): only the feather reaches the rim: x 25 at full weight, x 26 at 0.
  p.growMaxPx = 0;
  out = c.run(t, p, &st);
  CHECK(st.grownPixels == 0 && at(out, 25) == 0.0f && at(out, 26) == 0.1f && at(out, 27) == 0.05f);
  // The step limit: 1 step takes x 25 only.
  p.growMaxPx = 1;
  out = c.run(t, p, &st);
  CHECK(st.grownPixels == 5 && at(out, 26) == 0.0f && at(out, 27) == 0.05f);
  // A growth threshold as high as Jump Above takes nothing in.
  p.growMaxPx = 3;
  p.growThreshold = 0.5;
  out = c.run(t, p, &st);
  CHECK(st.grownPixels == 0 && at(out, 26) == 0.1f);
  // A moving edge next to the blob: rows y 20-24, x 25-29 are 0.6 in t - 1 (and t - 2), 0 in t + 1 (and
  // t + 2), 0.7 in t: beyond both by 0.1, but the references disagree: no growth; x 26-29 (2 px and
  // more from the blob: outside the feather) unchanged.
  Clip e(w, h, T, 0.0f);
  e.rect(t, t, 20, 20, 5, 5, 1.0f);
  e.rect(0, t - 1, 25, 20, 5, 5, 0.6f);
  e.rect(t, t, 25, 20, 5, 5, 0.7f);
  p.growThreshold = 0.02;
  out = e.run(t, p, &st);
  CHECK(st.repaired == 1 && st.grownPixels == 0 && at(out, 26) == 0.7f && at(out, 29) == 0.7f);
  std::printf("growth: ok\n");
}

// Soft-rim jumps (synth rimPx 4-16 at 1080 = 2-8 px at 960x540), Span 3 (all five jumps): the rim left
// after the repair (pixels near a jump where the truth is clear but the output is not: > 0.02 off),
// without growth (the behaviour before) and with it; the same blobs repaired and kept either way;
// nothing changed away from the jumps, no legit pixel touched, no pixel of the subject lowered.
void testSoftRims() {
  for (double rim : {4.0, 8.0, 12.0, 16.0}) {
    const synth::JumpSequence s = synth::makeJumpSequence(960, 540, 0.0, 1, rim);
    const size_t n = size_t(s.w) * s.h;
    const double rk = rim * s.h / 1080.0, zone = 2 * std::max(1.0, s.h / 540.0) + 1.5;
    long long residual[2] = {0, 0}, stray[2] = {0, 0}, legit[2] = {0, 0}, eaten[2] = {0, 0};
    int decisions[2][2] = {{0, 0}, {0, 0}};
    std::vector<float> out(n);
    for (int g = 0; g < 2; ++g) {
      JumpParams p;
      p.maxSpan = 3;
      if (!g) p.growMaxPx = 0;
      for (int t = 0; t < s.frames; ++t) {
        const float* fr[2 * kJumpReach + 1];
        for (int q = -kJumpReach; q <= kJumpReach; ++q)
          fr[q + kJumpReach] = t + q >= 0 && t + q < s.frames ? s.matte[size_t(t + q)].data() : nullptr;
        JumpStats st;
        repairJumps(fr, s.w, s.h, p, out.data(), &st);
        decisions[g][0] += st.repaired;
        decisions[g][1] += st.keptMoved + st.keptTurning + st.keptNotStill;
        for (size_t i = 0; i < n; ++i) {
          const int x = int(i % size_t(s.w)), y = int(i / size_t(s.w));
          const float tr = s.truth[size_t(t)][i];
          eaten[g] += out[i] < tr - 0.02f && out[i] < s.matte[size_t(t)][i];  // the subject lowered
          bool nearJump = false, inRim = false;
          for (const synth::JumpDefect& d : s.defects) {
            if (t < d.first || t > d.last) continue;
            const double dd = std::hypot(x - d.cx, y - d.cy);
            nearJump = nearJump || dd <= d.r + 1.5 * rk + zone;
            inRim = inRim || dd <= d.r + 1.5 * rk;
          }
          if (out[i] != s.matte[size_t(t)][i]) {
            stray[g] += !nearJump;
            legit[g] += s.legit[size_t(t)][i] != 0;
          }
          if (!inRim) continue;
          bool moving = false;  // the figure's edge crosses it within t +- 2: not judged by design
          for (int q = std::max(0, t - 2); q <= std::min(s.frames - 1, t + 2); ++q)
            moving = moving || s.truth[size_t(q)][i] != tr;
          if (!moving && std::fabs(out[i] - tr) > 0.02f) ++residual[g];
        }
      }
    }
    std::printf("soft rim %.0f px (at 540): rim left > 0.02 without growth %lld, with %lld | repaired / kept %d / %d "
                "and %d / %d | stray %lld %lld, legit %lld %lld, subject lowered %lld %lld\n",
                rk, residual[0], residual[1], decisions[0][0], decisions[0][1], decisions[1][0], decisions[1][1], stray[0],
                stray[1], legit[0], legit[1], eaten[0], eaten[1]);
    CHECK(decisions[0][0] == decisions[1][0] && decisions[0][1] == decisions[1][1]);
    CHECK(residual[0] > 10 && residual[1] <= residual[0] / 50);
    CHECK(stray[1] == 0 && legit[1] == 0 && eaten[1] == 0);
  }
}

// Motion Guards (motionGuard): on = the results before the switch existed (pinned: Span 1 at
// 960x540, 919 px changed with the growth, 3 repaired, 22 kept); off = the same blobs, every one repaired (the fast
// disc and the bar's turning frames now change), frames without a kept blob bit-identical to on.
void testGuardsSwitch() {
  const synth::JumpSequence s = synth::makeJumpSequence(960, 540);
  const size_t n = size_t(s.w) * s.h;
  std::vector<float> on(n), off(n);
  std::vector<uint8_t> marksOn, marksOff;
  long long changedOn = 0, changedOff = 0, legitOff = 0;
  int repairedOn = 0, keptOn = 0, repairedOff = 0, keptOff = 0, redOn = 0, redOff = 0;
  for (int t = 0; t < s.frames; ++t) {
    const float* fr[2 * kJumpReach + 1];
    for (int q = -kJumpReach; q <= kJumpReach; ++q)
      fr[q + kJumpReach] = t + q >= 0 && t + q < s.frames ? s.matte[size_t(t + q)].data() : nullptr;
    JumpParams p;
    JumpStats a, b;
    repairJumps(fr, s.w, s.h, p, on.data(), &a, &marksOn);
    p.motionGuard = false;
    repairJumps(fr, s.w, s.h, p, off.data(), &b, &marksOff);
    changedOn += a.pixelsChanged;
    changedOff += b.pixelsChanged;
    repairedOn += a.repaired;
    keptOn += a.keptMoved + a.keptTurning + a.keptNotStill;
    repairedOff += b.repaired;
    keptOff += b.keptMoved + b.keptTurning + b.keptNotStill;
    redOn += int(std::count(marksOn.begin(), marksOn.end(), uint8_t(kJumpKept)));
    redOff += int(std::count(marksOff.begin(), marksOff.end(), uint8_t(kJumpKept)));
    CHECK(a.blobs.size() == b.blobs.size() && b.repaired == int(b.blobs.size()));
    for (size_t k = 0; k < std::min(a.blobs.size(), b.blobs.size()); ++k)
      CHECK(a.blobs[k].area == b.blobs[k].area && a.blobs[k].x0 == b.blobs[k].x0 && a.blobs[k].y0 == b.blobs[k].y0 &&
            a.blobs[k].added == b.blobs[k].added);
    if (a.repaired == int(a.blobs.size())) CHECK(sameBits(on, off));
    for (size_t i = 0; i < n; ++i) legitOff += off[i] != s.matte[size_t(t)][i] && s.legit[size_t(t)][i];
  }
  std::printf("motion guards on / off (Span 1, 960x540): repaired %d / %d, kept %d / %d, changed %lld / %lld px, red "
              "pixels %d / %d, legit pixels changed off %lld\n",
              repairedOn, repairedOff, keptOn, keptOff, changedOn, changedOff, redOn, redOff, legitOff);
  // 873 px before the hysteresis growth; it takes in the jumps' 1 px soft edge (the decisions are the same).
  CHECK(changedOn == 919 && repairedOn == 3 && keptOn == 22);
  CHECK(keptOff == 0 && redOff == 0 && repairedOff == repairedOn + keptOn && changedOff > changedOn && legitOff > 0 &&
        redOn > 0);
}

// 2880x1620: the sequence at 960x540 scaled 3x (nearest), frames 1-7, the jump of frame 3.
void testTiming() {
  const synth::JumpSequence s = synth::makeJumpSequence(960, 540);
  const int W = 2880, H = 1620;
  std::vector<std::vector<float>> big(8, std::vector<float>(size_t(W) * H));
  for (int t = 0; t < 8; ++t)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) big[size_t(t)][size_t(y) * W + x] = s.matte[size_t(t)][size_t(y / 3) * 960 + x / 3];
  std::vector<float> out(size_t(W) * H);
  for (int span = 1; span <= 3; span += 2) {
    JumpParams p;
    p.maxSpan = span;
    const float* fr[2 * kJumpReach + 1];
    for (int q = -kJumpReach; q <= kJumpReach; ++q) fr[q + kJumpReach] = 3 + q >= 0 ? big[size_t(3 + q)].data() : nullptr;
    double best = 1e9;
    JumpStats st;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      repairJumps(fr, W, H, p, out.data(), &st);
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("timing 2880x1620 Span %d: %.1f ms (flag %.1f blobs %.1f repair %.1f), repaired %d kept %d\n", span,
                best, st.flagMs, st.blobMs, st.repairMs, st.repaired, st.keptMoved + st.keptTurning + st.keptNotStill);
    CHECK(st.repaired >= 1);
    CHECK(best < (span == 1 ? 150.0 : 300.0));
  }
}

}  // namespace

int main() {
  testKnownAnswers();
  testSpans();
  testPassThrough();
  testSequence();
  testResolutions();
  testJitter();
  testGuardsSwitch();
  testGrowth();
  testSoftRims();
  testTiming();
  if (failures) std::printf("%d FAILED\n", failures);
  else std::printf("all passed\n");
  return failures ? 1 : 0;
}
