// Matte Merge core on small mattes with known answers: the distance transform and the min / max
// filters against brute force, registration of a known move, each combine method on hand-made
// cases, the full Add fine detail path against a brute-force reference, bit-identity outside the
// zone, never lowering the base, dropped blobs, the trimap fill and the picture evidence
// (including its invariance to an affine and a log re-encoding of the picture).
// Mutation checks (2026-10-24): each case below names a code change that made it fail.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "matte_merge.h"
#include "merge_evidence.h"
#include "merge_morph.h"
#include "merge_register.h"
#include "ops/edge_field.h"

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

double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

// A soft-edged axis-aligned box [x0, x1] x [y0, y1] (0.5 at the bounds), edge width `soft` px.
float box(double x, double y, double x0, double y0, double x1, double y1, double soft) {
  const double d = std::min(std::min(x - x0, x1 - x), std::min(y - y0, y1 - y));
  return float(clamp01(0.5 + d / soft));
}

// Mutations: edt1d's d[q] = dq * dq + f -> dq + f; the column strips reading row[0] for every lane
// (2026-10-24, the strip version).
void testDistance() {
  std::mt19937 rng(3);
  for (int t = 0; t < 30; ++t) {
    const int w = 5 + int(rng() % 60), h = 5 + int(rng() % 50);
    const double density = t < 3 ? 0.0 : t < 6 ? 1.0 / (w * h) : 0.002 * (t % 7 + 1);
    std::vector<uint8_t> m(size_t(w) * h);
    std::uniform_real_distribution<double> u(0, 1);
    for (auto& v : m) v = u(rng) < density;
    if (t == 3) m[size_t(h / 2) * w + w / 3] = 1;
    std::vector<float> d2(m.size());
    distanceSquared(m.data(), w, h, d2.data());
    bool any = false;
    for (auto v : m) any = any || v;
    int bad = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double best = 1e30;
        for (int yy = 0; yy < h; ++yy)
          for (int xx = 0; xx < w; ++xx)
            if (m[size_t(yy) * w + xx]) best = std::min(best, double((x - xx) * (x - xx) + (y - yy) * (y - yy)));
        const double got = d2[size_t(y) * w + x];
        if (any ? std::fabs(got - best) > 1e-3 : got != kFar) ++bad;
      }
    CHECK(bad == 0);
  }
  std::printf("distance transform: 30 random masks vs brute force\n");
}

// Mutations: vhgwRow's two-block case op(hb[lo], g[hi]) -> g[hi]; its division-free lo counter
// wrapping at b + 1 instead of b (2026-10-24).
void testMinMax() {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0, 1);
  for (int t = 0; t < 24; ++t) {
    const int w = 1 + int(rng() % 40), h = 1 + int(rng() % 40), r = t % 6;
    std::vector<float> in(size_t(w) * h), mn(in.size()), mx(in.size()), op(in.size());
    for (auto& v : in) v = u(rng);
    minFilter(in.data(), w, h, r, mn.data());
    maxFilter(in.data(), w, h, r, mx.data());
    opening(in.data(), w, h, r, op.data());
    std::vector<float> e(in.size()), refOp(in.size());
    int bad = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float a = 2, b = -1;
        for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
          for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx)
            a = std::min(a, in[size_t(yy) * w + xx]), b = std::max(b, in[size_t(yy) * w + xx]);
        e[size_t(y) * w + x] = a;
        bad += mn[size_t(y) * w + x] != a || mx[size_t(y) * w + x] != b;
      }
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float b = -1;
        for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
          for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx) b = std::max(b, e[size_t(yy) * w + xx]);
        bad += op[size_t(y) * w + x] != b;
      }
    CHECK(bad == 0);
  }
  std::printf("min / max / opening: 24 random images, r 0-5, vs brute force (bit-identical)\n");
}

// Three boxes and a disc (not one circle: scale and edge offset must be separable). The detail is
// the same shape, its edges 1.5 px further out, moved by A(p) = c + s (p - c) + t.
// Mutation: solveLevel's delta column -1.0 -> 0.0 (no edge offset fitted).
void testRegistration() {
  const int w = 640, h = 400;
  auto shape = [&](double x, double y, double grow) {
    float v = box(x, y, 80 - grow, 60 - grow, 260 + grow, 300 + grow, 2.0);
    v = std::max(v, box(x, y, 380 - grow, 90 - grow, 560 + grow, 170 + grow, 2.0));
    v = std::max(v, box(x, y, 330 - grow, 250 - grow, 600 + grow, 340 + grow, 2.0));
    v = std::max(v, float(clamp01(0.5 + (40 + grow - std::hypot(x - 470, y - 210)) / 2.0)));
    return v;
  };
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  struct Case { double tx, ty, s; };
  for (const Case c : {Case{0, 0, 1}, Case{2.3, -1.7, 1}, Case{-5.6, 3.1, 1.012}, Case{11.0, -8.0, 0.99}}) {
    std::vector<float> base(size_t(w) * h), det(base.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        base[size_t(y) * w + x] = shape(x, y, 0);
        // detail(x) = shape at A^-1(x), grown 1.5 px
        det[size_t(y) * w + x] = shape(cx + (x - cx - c.tx) / c.s, cy + (y - cy - c.ty) / c.s, 1.5);
      }
    const Registration r = registerMattes(base.data(), det.data(), w, h, 0.5f, 4);
    double worst = 0;
    for (int y = 0; y < h; y += 4)
      for (int x = 0; x < w; x += 4) {
        if (!mr::isContour(base.data(), w, h, x, y)) continue;
        double qx, qy;
        r.apply(x, y, &qx, &qy);
        worst = std::max(worst, std::hypot(qx - (cx + c.s * (x - cx) + c.tx), qy - (cy + c.s * (y - cy) + c.ty)));
      }
    std::printf("registration t (%.1f, %.1f) s %.3f: got (%.3f, %.3f) s %.4f delta %.2f, max error on the contour %.3f px, "
                "rms %.3f, %d samples, sensitivity %.1f\n", c.tx, c.ty, c.s, r.tx, r.ty, r.scale, r.delta, worst, r.rmsPx,
                r.samples, r.sensitivity);
    CHECK(r.ok);
    CHECK(worst < 0.2);
    CHECK(std::fabs(r.delta - 1.5 * c.s) < 0.3);
  }
}

// Add fine detail on a hand-made case: a box base (soft 2 px edge) and a detail = the same box plus
// a 1 px line (0.8) leaving its right side and a 2 px line crossing the base's grey edge, and a
// floating blob in the zone. Align off. Known answers: off the lines out == base bit for bit; on a
// line where the ramp is 1, out = base + 0.8 (1 - base); the blob adds nothing; the zone's far
// border ramps. Mutations: combine's "f * (1 - base)" -> "f" (line over grey: 1.6 > expected);
// anchor = true (blob added); feather ramp = 1 everywhere (ramp check).
void testAddKnown() {
  const int w = 200, h = 120;
  std::vector<float> base(size_t(w) * h), det(base.size()), out(base.size()), zone(base.size()), rej(base.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      base[i] = box(x, y, 20, 20, 100, 100, 2.0);
      det[i] = base[i];
      if (y == 50 && x > 100 && x < 150) det[i] = std::max(det[i], 0.8f);           // 1 px line
      if ((y == 70 || y == 71) && x > 95 && x < 130) det[i] = std::max(det[i], 0.8f);  // crosses the grey edge
      if (std::hypot(x - 130.0, y - 25.0) < 5) det[i] = 1.0f;                        // floating blob
    }
  MergeParams p;
  p.align = false;
  p.reachPx = 40;
  p.featherPx = 4;
  p.detailWidthPx = 8;  // the known answers assume a 9x9 opening (the px default was 8 until 2026-10-24)
  p.bridgePx = 0;       // the thin-structure path alone (the bridge: testBridge)
  MergeViews v;
  v.zone = zone.data();
  v.rejected = rej.data();
  MergeInfo info;
  merge(base.data(), det.data(), nullptr, w, h, p, out.data(), v, &info);
  int offLine = 0, onLineBad = 0, blob = 0, lower = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const bool line = (y == 50 && x > 100 && x < 150) || ((y == 70 || y == 71) && x > 95 && x < 130);
      if (out[i] < base[i]) ++lower;
      if (std::hypot(x - 130.0, y - 25.0) < 7) blob += out[i] != base[i];
      if (!line) {
        offLine += out[i] != base[i];
      } else if (zone[i] == 1.0f && det[i] == 0.8f && x >= 103) {  // beyond the box's opening reach
        const float want = float(base[i] + 0.8 * (1.0 - base[i]));
        onLineBad += std::fabs(out[i] - want) > 1e-6f;
      }
    }
  // Line 1 at x = 130: inside the zone (reach 40 from the contour pixels x = 100 / 101), ramp 1.
  CHECK(std::fabs(out[size_t(50) * w + 130] - 0.8f) < 1e-6f);
  // Line 1 beyond the zone (x >= 142): dropped, base (0) bit for bit.
  CHECK(out[size_t(50) * w + 145] == 0.0f);
  // Where line 2 crosses the base's grey edge (x = 100: base 0.5, detail 0.8): the opening there is
  // 0.5 (a 9x9 square inside the box reaches it), so fine = 0.3 and out = 0.5 + 0.3 * 0.5 = 0.65.
  CHECK(std::fabs(out[size_t(70) * w + 100] - 0.65f) < 1e-6f);
  // The zone's far border: the zone ends at x = 141 (41 px from the contour pixel x = 100 is out:
  // 141 - 101 = 40); the ramp is 0 outside, smoothstep((d - 0.5) / 4) inside, d = distance to outside.
  const float r0 = zone[size_t(30) * w + 142], r1 = zone[size_t(30) * w + 139], r2 = zone[size_t(30) * w + 130];
  std::printf("add known: off-line changed %d, on-line wrong %d, blob changed %d, lowered %d; ramp %.3f %.3f %.3f; "
              "kept %ld dropped %ld\n", offLine, onLineBad, blob, lower, r0, r1, r2, info.keptPixels, info.droppedPixels);
  CHECK(offLine == 0);
  CHECK(onLineBad == 0);
  CHECK(blob == 0);
  CHECK(lower == 0);
  CHECK(r0 == 0.0f && std::fabs(r1 - 0.68359375f) < 1e-6f && r2 == 1.0f);
  CHECK(info.droppedPixels > 50);
  // Rejected shows the blob.
  CHECK(rej[size_t(25) * w + 130] > 0.9f);
}

// The Add path against a brute-force reference (zone by exhaustive distance, ramp, opening,
// connectivity by repeated relaxation, the bridge weight by exhaustive distance to the contour), on
// random mattes. Align off, evidence off.
// Mutations: zone "d2 <= reach2" -> "d2 < reach2" fails here (pixels at exactly the reach); the
// bridge's screen 1 - (1 - fine)(1 - fb) -> max(fine, fb); the strand mask's threshold 0.25 -> 0.1
// (2026-10-24).
void testAddReference() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 1);
  for (int t = 0; t < 6; ++t) {
    const int w = 90 + int(rng() % 30), h = 70 + int(rng() % 20);
    const size_t n = size_t(w) * h;
    std::vector<float> base(n), det(n), out(n);
    const double bx = 20 + 10 * u(rng), by = 15 + 10 * u(rng);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        base[i] = box(x, y, bx, by, bx + 35, by + 30, 1 + 3 * u(rng) * 0 + 2.0 * (t % 3));
        det[i] = base[i];
      }
    for (int s = 0; s < 12; ++s) {  // random lines, some attached, some floating
      double x = bx + 35 * u(rng), y = by + 30 * u(rng), a = 2 * M_PI * u(rng);
      const float val = float(0.3 + 0.7 * u(rng));
      if (s % 3 == 0) x += 50 * std::cos(a), y += 50 * std::sin(a);
      for (int j = 0; j < 40; ++j) {
        const int px = int(x + j * std::cos(a)), py = int(y + j * std::sin(a));
        if (px >= 0 && py >= 0 && px < w && py < h) det[size_t(py) * w + px] = std::max(det[size_t(py) * w + px], val);
      }
    }
    MergeParams p;
    p.align = false;
    p.reachPx = 12 + t * 3;
    p.featherPx = t % 2 ? 0.0 : 3.0;
    p.detailWidthPx = 5;
    p.strength = t == 5 ? 0.6 : t == 3 ? 1.7 : 1.0;  // 1.7: gain on the added detail
    p.bridgePx = t % 3 == 1 ? 0.0 : 2.0 + t;            // 0 = off
    merge(base.data(), det.data(), nullptr, w, h, p, out.data());

    // Reference.
    std::vector<uint8_t> g(n), zone(n), kept(n);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        g[size_t(y) * w + x] = (base[size_t(y) * w + x] > 0.02f && base[size_t(y) * w + x] < 0.98f) ||
                               mr::isContour(base.data(), w, h, x, y);
    auto nearest = [&](int x, int y, auto pred) {
      double best = 1e30;
      for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx)
          if (pred(size_t(yy) * w + xx)) best = std::min(best, double((x - xx) * (x - xx) + (y - yy) * (y - yy)));
      return best;
    };
    const double core = std::max(2.0, p.featherPx);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        zone[i] = nearest(x, y, [&](size_t j) { return g[j] != 0; }) <= p.reachPx * p.reachPx;
        if (base[i] >= 0.98f && nearest(x, y, [&](size_t j) { return base[j] < 0.98f; }) > core * core) zone[i] = 0;
      }
    for (size_t i = 0; i < n; ++i) kept[i] = zone[i] && det[i] > 0.05f && base[i] >= 0.5f;
    for (bool changed = true; changed;) {
      changed = false;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if (kept[i] || !zone[i] || det[i] <= 0.05f) continue;
          for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1) && !kept[i]; ++yy)
            for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx)
              if (kept[size_t(yy) * w + xx]) kept[i] = 1, changed = true;
        }
    }
    const int r = std::max(1, int(std::ceil((p.detailWidthPx - 1) / 2)));
    std::vector<float> e(n), op(n);
    for (int pass = 0; pass < 2; ++pass)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          float m = pass ? -1.0f : 2.0f;
          for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
            for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx)
              m = pass ? std::max(m, e[size_t(yy) * w + xx]) : std::min(m, det[size_t(yy) * w + xx]);
          (pass ? op : e)[size_t(y) * w + x] = m;
        }
    double worst = 0;
    int outsideChanged = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        double want = base[i];
        if (zone[i] && kept[i]) {
          double ramp = 1;
          if (p.featherPx > 0) {
            const double d = std::sqrt(nearest(x, y, [&](size_t j) { return !zone[j]; }));
            const double tt = clamp01((d - 0.5) / p.featherPx);
            ramp = d > 1e10 ? 1 : tt * tt * (3 - 2 * tt);
          }
          const double fine = std::max(0.0f, det[i] - op[i]);
          double comb = fine;
          if (p.bridgePx > 0) {  // the bridge: 1 - smoothstep(distance to the base's 0.5 contour / bridgePx)
            const double dc = std::sqrt(nearest(x, y, [&](size_t j) {
              return mr::isContour(base.data(), w, h, int(j % w), int(j / w));
            }));
            const double tt = clamp01((dc + 0.5) / p.bridgePx);
            // times the strand mask: distance to a kept top-hat > 0.25, 1 within bridgePx, 0 from 2 bridgePx
            const double ds = std::sqrt(nearest(x, y, [&](size_t j) { return kept[j] && det[j] - op[j] > 0.25f; }));
            const double ts = clamp01((ds - p.bridgePx) / p.bridgePx);
            const double fb = (1.0 - tt * tt * (3 - 2 * tt)) * (ds > 1e10 ? 0.0 : 1.0 - ts * ts * (3 - 2 * ts)) *
                              std::max(0.0, double(det[i]) - base[i]);
            if (fb > 0) comb = 1.0 - (1.0 - fine) * (1.0 - fb);
          }
          const double f0 = ramp * std::min(1.0, p.strength) * comb;
          const double gn = std::min(2.0, std::max(1.0, p.strength));
          const double f = gn > 1 ? 1.0 - std::pow(1.0 - std::min(1.0, f0), gn) : f0;
          if (f > 0) want = base[i] + f * (1.0 - base[i]);
        }
        if (!zone[i]) outsideChanged += out[i] != base[i];
        worst = std::max(worst, std::fabs(out[i] - want));
      }
    std::printf("add vs reference %dx%d reach %.0f feather %.0f: max diff %.2e, outside-zone changed %d\n", w, h,
                p.reachPx, p.featherPx, worst, outsideChanged);
    CHECK(worst < 1e-5);
    CHECK(outsideChanged == 0);
  }
}

// The bridge (user design 2026-10-24; strand mask after the Flame test the same day). Base: a box with
// a 2 px soft edge whose 0.5 line lies between pixels (x = 120.5 on the right: contour pixels 120 / 121,
// values 0.75 / 0.25; 40.5 on the left). Detail, scaled by s: the box grown by 4 px to the right, top
// and bottom (a thick ring the opening keeps, so Add alone never takes it), a thick shoulder below a
// 1 px strand at row 100 rooted on the ring (right), a 1 px strand at row 64 rooted on the left edge,
// next to it a thick 13 px blob that touches nothing (not connected), a detached 1 px strand 2 px right
// of the ring at row 75 (not connected: no seed), and a faint (0.2, below the 0.25 seed threshold) strand
// rooted on the top edge (no seed). Bridge 6 px:
// b(d) = 1 - smoothstep(d / 6), d = distance to the nearest contour pixel + 0.5; strand mask
// m(ds) = 1 - smoothstep((ds - 6) / 6), ds = distance to the nearest strand pixel (kept top-hat > 0.25).
// Known answers (s = 1):
//   (122, 100): base 0, ring 1, top-hat 0, strand pixel (126, 100) 4 px away: out = b(1.5);
//   (121, 100): base 0.25: out = 0.25 + 0.75 * 0.75 * b(0.5);
//   shoulder (125, 110): b(4.5) m(10) (nearest strand pixel (125, 100)); (130, 110): d 9, the base;
//   the hairless ring (row 70, all of the top edge): exactly Bridge 0; along the ring at x = 122 the
//   change fades with the distance from the strand and is 0 from 12 px on;
//   Rejected at the blob's edge (37, 72): b(3.5) m(8) (nearest strand pixel (37, 64)).
// Also: the strand has no gap from the head (min along it > 0.3; with Bridge 0 a gap < 0.05), the blob
// is never added, nothing is lowered, pixels 6+ px from the contour are those of Bridge 0 bit for bit,
// Zone Blend and Trimap ignore the bridge, and the same scene at twice the size with the sizes scaled
// (scaleToHeight) adds the same amount per unit area (within 4 %; the bridged part alone within 7 %).
// Mutations (2026-10-24, each failed here): the weight "1 - smoothstep" -> "smoothstep"; the gate
// "kept[i]" dropped for the bridge (blob added); max(0, detail - base) -> detail; the "+ 0.5" dropped
// from the distance; the bridge left out of the Rejected view; the strand mask left out (hairless ring
// bridged); its fade "(ds - bridgePx) / bridgePx" -> "ds / bridgePx"; seeds without "kept[i] &&"
// (the floater seeds the ring); the seed threshold 0.25 -> 0.1 (the faint strand seeds the top edge).
void testBridge() {
  auto scene = [](int s, std::vector<float>* base, std::vector<float>* det) {
    const int w = 200 * s, h = 160 * s;
    base->assign(size_t(w) * h, 0.0f);
    det->assign(size_t(w) * h, 0.0f);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const double X = (x + 0.5) / s - 0.5, Y = (y + 0.5) / s - 0.5;  // design coordinates
        (*base)[i] = box(X, Y, 40.5, 40.5, 120.5, 120.5, 2.0);
        float d = box(X, Y, 40.5, 36.5, 124.5, 124.5, 2.0);
        d = std::max(d, box(X, Y, 100.5, 102.5, 134.5, 118.5, 2.0));             // shoulder
        if (std::fabs(Y - 100) < 0.5 && X > 110 && X < 170) d = std::max(d, 0.8f);  // strand, right
        if (std::fabs(Y - 64) < 0.5 && X > 15 && X < 45) d = std::max(d, 0.8f);    // strand, left
        if (X >= 25.5 && X <= 37.5 && Y >= 66.5 && Y <= 78.5) d = 1.0f;           // detached thick blob
        if (std::fabs(Y - 75) < 0.5 && X > 126.5 && X < 150) d = 0.8f;            // detached strand (floater)
        if (std::fabs(X - 80) < 0.5 && Y > 10 && Y < 38) d = std::max(d, 0.2f);   // faint strand, top
        (*det)[i] = d;
      }
  };
  std::vector<float> base, det;
  scene(1, &base, &det);
  const int w = 200, h = 160;
  const size_t n = size_t(w) * h;
  auto at = [w](int x, int y) { return size_t(y) * w + x; };
  MergeParams p;
  p.align = false;
  p.reachPx = 40;
  p.featherPx = 4;
  p.detailWidthPx = 8;
  p.bridgePx = 0;
  std::vector<float> off(n), on(n), rej(n);
  merge(base.data(), det.data(), nullptr, w, h, p, off.data());
  p.bridgePx = 6;
  MergeViews v;
  v.rejected = rej.data();
  merge(base.data(), det.data(), nullptr, w, h, p, on.data(), v);
  auto ss = [](double t) { t = clamp01(t); return t * t * (3 - 2 * t); };
  auto b = [&](double d) { return 1 - ss(d / 6.0); };
  auto mk = [&](double ds) { return 1 - ss((ds - 6.0) / 6.0); };
  const double want121 = 0.25 + 0.75 * 0.75 * b(0.5), wantSh = b(4.5) * mk(10), wantRej = b(3.5) * mk(8);
  std::printf("bridge known: (122,100) %.4f (%.4f) (121,100) %.4f (%.4f) shoulder %.4f (%.4f) (130,110) %.4f (0); "
              "rejected at the blob %.4f (%.4f)\n",
              on[at(122, 100)], b(1.5), on[at(121, 100)], want121, on[at(125, 110)], wantSh, on[at(130, 110)],
              rej[at(37, 72)], wantRej);
  CHECK(std::fabs(base[at(121, 100)] - 0.25f) < 1e-6f && det[at(121, 100)] == 1.0f && base[at(122, 100)] == 0.0f);
  CHECK(std::fabs(on[at(122, 100)] - b(1.5)) < 1e-6f);
  CHECK(std::fabs(on[at(121, 100)] - want121) < 1e-6f);
  CHECK(std::fabs(on[at(125, 110)] - wantSh) < 1e-6f);
  CHECK(on[at(130, 110)] == base[at(130, 110)] && on[at(130, 110)] == 0.0f);
  CHECK(off[at(122, 100)] == base[at(122, 100)]);  // without the bridge the ring is not taken
  CHECK(std::fabs(rej[at(37, 72)] - wantRej) < 1e-6f && rej[at(30, 72)] == 0.0f);
  // The hairless ring: no strand within 12 px, exactly Bridge 0 (the halo seen in Flame).
  int hairless = 0;
  for (int x = 30; x < 160; ++x)
    for (int y = 30; y < 50; ++y) hairless += on[at(x, y)] != off[at(x, y)];  // the top edge
  for (int y = 52; y <= 86; ++y)
    for (int x = 118; x < 140; ++x) hairless += on[at(x, y)] != off[at(x, y)];  // right, 14+ px above the strand
  // Along the ring (x = 122) the change fades with the distance to the strand and is 0 from 12 px on.
  bool fades = true;
  for (int y = 88; y < 100; ++y) fades = fades && on[at(122, y)] - off[at(122, y)] <= on[at(122, y + 1)] - off[at(122, y + 1)];
  std::printf("bridge strand mask: hairless px changed %d; along the ring, change at 12 / 6 / 2 / 0 px from the strand "
              "%.3f %.3f %.3f %.3f, fades %d\n", hairless, on[at(122, 88)] - off[at(122, 88)],
              on[at(122, 94)] - off[at(122, 94)], on[at(122, 98)] - off[at(122, 98)],
              on[at(122, 100)] - off[at(122, 100)], int(fades));
  CHECK(hairless == 0 && fades && on[at(122, 88)] == off[at(122, 88)] && on[at(122, 94)] > off[at(122, 94)]);
  // The strand: from the base's edge outward, no gap with the bridge, a gap without it.
  double minOn = 1, minOff = 1;
  for (int x = 118; x <= 135; ++x) minOn = std::min(minOn, double(on[at(x, 100)]));
  for (int x = 121; x <= 135; ++x) minOff = std::min(minOff, double(off[at(x, 100)]));
  std::printf("bridge strand: min along the root %.3f (bridge 6), %.3f (bridge 0)\n", minOn, minOff);
  CHECK(minOn > 0.3 && minOff < 0.05);
  // Blob, lowering, beyond Bridge px.
  int blob = 0, lowered = 0, farDiff = 0, nearChanged = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = at(x, y);
      if (x >= 24 && x <= 39 && y >= 65 && y <= 80) blob += on[i] != base[i];
      lowered += on[i] < base[i];
      double dc = 1e30;  // distance to the nearest contour pixel (brute force)
      for (int yy = std::max(0, y - 8); yy <= std::min(h - 1, y + 8); ++yy)
        for (int xx = std::max(0, x - 8); xx <= std::min(w - 1, x + 8); ++xx)
          if (mr::isContour(base.data(), w, h, xx, yy)) dc = std::min(dc, std::hypot(xx - x, yy - y));
      if (dc >= 5.5) farDiff += on[i] != off[i];
      else nearChanged += on[i] != off[i];
    }
  std::printf("bridge: blob px changed %d, lowered %d, px >= 6 from the contour differing from bridge 0: %d, "
              "nearer px changed %d\n", blob, lowered, farDiff, nearChanged);
  CHECK(blob == 0 && lowered == 0 && farDiff == 0 && nearChanged > 50);
  // Zone Blend and Trimap: the bridge does nothing.
  for (int m = 0; m < 2; ++m) {
    std::vector<float> bb = base, a0(n), a1(n);
    if (m)
      for (auto& x : bb) x = x >= 0.99f ? 1.0f : x <= 0.01f ? 0.0f : 0.5f;
    MergeParams q = p;
    q.method = m ? Method::AddFineDetail : Method::ZoneBlend;
    q.source = m ? Source::Trimap : Source::MatteEdge;
    q.bridgePx = 0;
    merge(bb.data(), det.data(), nullptr, w, h, q, a0.data());
    q.bridgePx = 6;
    merge(bb.data(), det.data(), nullptr, w, h, q, a1.data());
    CHECK(a0 == a1);
  }
  // Scale with the height: the same scene at 160 and 320 lines, sizes from HeightSizes percentages.
  double added[2], bridged[2];
  for (int s = 1; s <= 2; ++s) {
    std::vector<float> bs, ds;
    scene(s, &bs, &ds);
    const int ws = 200 * s, hs = 160 * s;
    HeightSizes sz;
    sz.reachPct = 25;
    sz.detailWidthPct = 5;
    sz.featherPct = 2.5;
    sz.bridgePct = 3.75;  // 6 px at 160
    MergeParams q;
    q.align = false;
    scaleToHeight(q, sz, hs);
    std::vector<float> o1(bs.size()), o0(bs.size());
    merge(bs.data(), ds.data(), nullptr, ws, hs, q, o1.data());
    q.bridgePx = 0;
    merge(bs.data(), ds.data(), nullptr, ws, hs, q, o0.data());
    double a = 0, c = 0;
    for (size_t i = 0; i < bs.size(); ++i) a += o1[i] - bs[i], c += o1[i] - o0[i];
    added[s - 1] = a / (s * s);
    bridged[s - 1] = c / (s * s);
  }
  std::printf("bridge scale: added per design px^2 %.1f (160 lines) / %.1f (320), by the bridge %.1f / %.1f\n", added[0],
              added[1], bridged[0], bridged[1]);
  // 42.7 / 44.9 (5 %): with the strand mask only about 43 design px^2 are bridged, next to two 1 px strands,
  // and where a strand's top-hat first passes 0.25 is whole pixels (was 1.7 % on the whole ring before).
  CHECK(bridged[0] > 20 && std::fabs(bridged[1] / bridged[0] - 1) < 0.07 && std::fabs(added[1] / added[0] - 1) < 0.04);
}

// Random base and detail (noise and shapes), every method: outside the zone bit-identical; Add never
// lowers. The zone must be a strict part of the frame here (else "outside unchanged" is empty).
// Mutations: zone = d2 <= reach2 -> zone = 1 (zone count check); Add's "base + f (1 - base)" ->
// "base - f (1 - base)" (lowered > 0).
void testInvariants() {
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> u(0, 1);
  const int w = 160, h = 110;
  const size_t n = size_t(w) * h;
  for (int t = 0; t < 4; ++t) {
    std::vector<float> base(n), det(n), out(n), zone(n);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        base[i] = box(x, y, 40, 30, 110, 80, 3.0);
        det[i] = std::min(1.0f, std::max(0.0f, box(x, y, 38, 31, 113, 82, 3.0) + 0.3f * (u(rng) - 0.5f)));
      }
    for (int m = 0; m < 3; ++m) {
      MergeParams p;
      p.align = t % 2 == 1;
      p.reachPx = 15;
      p.detailWidthPx = 8;  // the pre-2026-10-24 px defaults, kept for this test
      p.featherPx = 4;
      p.method = m == 1 ? Method::ZoneBlend : Method::AddFineDetail;
      p.source = m == 2 ? Source::Trimap : Source::MatteEdge;
      std::vector<float> b = base;
      if (m == 2)
        for (auto& v : b) v = v >= 0.99f ? 1.0f : v <= 0.01f ? 0.0f : 0.5f;
      MergeViews v;
      v.zone = zone.data();
      MergeInfo info;
      merge(b.data(), det.data(), nullptr, w, h, p, out.data(), v, &info);
      // Zone view > 0 exactly in the zone (a zone pixel is >= 1 px from outside: ramp > 0).
      int outside = 0, lowered = 0;
      long zoneCount = 0;
      for (size_t i = 0; i < n; ++i) {
        zoneCount += zone[i] > 0.0f;
        if (zone[i] == 0.0f) outside += out[i] != b[i];
        lowered += out[i] < b[i];
      }
      std::printf("invariants t%d method %d: zone %ld of %zu px, outside changed %d, lowered %d\n", t, m, zoneCount, n,
                  outside, lowered);
      CHECK(outside == 0);
      CHECK(zoneCount > 0 && size_t(zoneCount) < n / 2);
      if (m == 0) CHECK(lowered == 0);
    }
  }
}

// Zone blend and trimap known answers. Blend: base box, detail = a box 2 px larger: where the ramp
// is 1 the output is the detail. Trimap: fg box / unknown band 12 px / bg; detail = a soft box in
// the band and a floating blob: out = detail at ramp 1, blob -> 0, known pixels bit-identical.
// Mutations: trimap prior b / (a + b) -> a / (a + b) (feather border values flip); blend's
// "base + wt * sig" -> "base" (no change).
void testBlendTrimapKnown() {
  const int w = 160, h = 120;
  const size_t n = size_t(w) * h;
  std::vector<float> base(n), det(n), out(n), zone(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      base[size_t(y) * w + x] = box(x, y, 40, 30, 110, 90, 1.0);
      det[size_t(y) * w + x] = box(x, y, 38, 28, 112, 92, 2.0);
    }
  MergeParams p;
  p.align = false;
  p.method = Method::ZoneBlend;
  p.reachPx = 10;
  p.featherPx = 3;
  p.detailWidthPx = 8;  // the pre-2026-10-24 px default
  MergeViews v;
  v.zone = zone.data();
  merge(base.data(), det.data(), nullptr, w, h, p, out.data(), v);
  int bad = 0, full = 0;
  for (size_t i = 0; i < n; ++i)
    if (zone[i] == 1.0f) ++full, bad += std::fabs(out[i] - det[i]) > 1e-6f;
  std::printf("blend known: %d full-ramp px, %d differ from the detail\n", full, bad);
  CHECK(full > 500 && bad == 0);

  std::vector<float> tri(n), det2(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const float inner = box(x, y, 50, 40, 100, 80, 0.01), outer = box(x, y, 38, 28, 112, 92, 0.01);
      tri[i] = inner >= 1.0f ? 1.0f : outer <= 0.0f ? 0.0f : 0.5f;
      det2[i] = box(x, y, 44, 34, 106, 86, 3.0);
      if (std::hypot(x - 41.0, y - 60.0) < 1.6) det2[i] = 0.0f;  // keep the blob off the subject
      if (std::hypot(x - 107.5, y - 31.5) < 2.0) det2[i] = 1.0f;  // blob in the band's corner, detached
    }
  // Detach the blob: the subject's soft edge is 3 px; clear a ring around the blob.
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double d = std::hypot(x - 107.5, y - 31.5);
      if (d >= 2.0 && d < 4.5) det2[size_t(y) * w + x] = 0.0f;
    }
  p.source = Source::Trimap;
  p.featherPx = 2;
  merge(tri.data(), det2.data(), nullptr, w, h, p, out.data(), v);
  int known = 0, fill = 0, fillBad = 0, blob = 0, border = 0, borderBad = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (tri[i] != 0.5f) {
        known += out[i] != tri[i];
        continue;
      }
      if (std::hypot(x - 107.5, y - 31.5) < 2.0) blob += out[i] > 0.0f;
      else if (zone[i] == 1.0f) ++fill, fillBad += std::fabs(out[i] - det2[i]) > 1e-6f;
      // Ramp-0 border pixels next to the known foreground: prior near 1.
      if (zone[i] < 0.2f && x > 52 && x < 98 && y == 40) ++border, borderBad += out[i] < 0.85f;
    }
  std::printf("trimap known: known changed %d, fill px %d wrong %d, blob px above 0: %d, fg-border px %d low %d\n",
              known, fill, fillBad, blob, border, borderBad);
  CHECK(known == 0);
  CHECK(fill > 500 && fillBad == 0);
  CHECK(blob == 0);
  CHECK(border > 10 && borderBad == 0);
}

// Trimap fill, detail that fails the connectivity test (user decisions 2026-10-24): a floating blob
// and a detached strand (a 2 px line) in the unknown band, apart from the subject's soft edge. By
// default both become background (0) where the ramp is 1; with keepDisconnected they are the detail
// matte's values; connected pixels and known pixels are the same in both. Matte edge (Add and Zone
// blend) ignores keepDisconnected: bit-identical outputs.
// Mutations: target()'s "keepLoose ? da[i] : 0.0f" -> "da[i]" (default keeps them: blob / strand
// checks fail); -> "0.0f" (keepDisconnected has no effect: kept checks fail); keepLoose without
// "trimap &&" and the Matte edge target -> "keepLoose ? da[i] : base[i]" (blend output changes).
void testTrimapDisconnected() {
  const int w = 160, h = 120;
  const size_t n = size_t(w) * h;
  std::vector<float> tri(n), det(n), out0(n), out1(n), zone(n);
  std::vector<uint8_t> loose(n, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const float inner = box(x, y, 50, 40, 100, 80, 0.01), outer = box(x, y, 30, 20, 120, 100, 0.01);
      tri[i] = inner >= 1.0f ? 1.0f : outer <= 0.0f ? 0.0f : 0.5f;
      det[i] = box(x, y, 46, 36, 104, 84, 2.0);  // the subject, its soft edge in the band
      if (std::hypot(x - 111.0, y - 30.0) < 3.0) det[i] = 0.9f, loose[i] = 1;  // floating blob
      if ((y == 95 || y == 96) && x >= 40 && x <= 100) det[i] = 0.6f, loose[i] = 1;  // detached strand
    }
  MergeParams p;
  p.align = false;
  p.source = Source::Trimap;
  p.featherPx = 2;
  p.detailWidthPx = 8;
  MergeViews v;
  v.zone = zone.data();
  MergeInfo i0, i1;
  merge(tri.data(), det.data(), nullptr, w, h, p, out0.data(), v, &i0);
  p.keepDisconnected = true;
  merge(tri.data(), det.data(), nullptr, w, h, p, out1.data(), v, &i1);
  int looseFull = 0, bg0 = 0, kept1 = 0, otherDiff = 0;
  for (size_t i = 0; i < n; ++i) {
    if (loose[i] && zone[i] == 1.0f) {
      ++looseFull;
      bg0 += out0[i] == 0.0f;
      kept1 += std::fabs(out1[i] - det[i]) < 1e-6f;
    } else if (!loose[i]) {
      otherDiff += out0[i] != out1[i];  // pixels away from the loose pieces: the same
    }
  }
  std::printf("trimap disconnected: %d loose px at full ramp: background by default %d, kept with keepDisconnected %d; "
              "other px differing %d; dropped %ld / %ld\n", looseFull, bg0, kept1, otherDiff, i0.droppedPixels,
              i1.droppedPixels);
  CHECK(looseFull > 100);
  CHECK(bg0 == looseFull);
  CHECK(kept1 == looseFull);
  CHECK(i0.droppedPixels > 0 && i0.droppedPixels == i1.droppedPixels);
  // The rest of the frame is the same.
  CHECK(otherDiff == 0);

  // Matte edge ignores it.
  std::vector<float> base(n), a0(n), a1(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) base[size_t(y) * w + x] = box(x, y, 46, 36, 104, 84, 2.0);
  for (int m = 0; m < 2; ++m) {
    MergeParams q;
    q.align = false;
    q.reachPx = 30;
    q.method = m ? Method::ZoneBlend : Method::AddFineDetail;
    merge(base.data(), det.data(), nullptr, w, h, q, a0.data());
    q.keepDisconnected = true;
    merge(base.data(), det.data(), nullptr, w, h, q, a1.data());
    CHECK(a0 == a1);
  }
}

// Evidence: a flat picture with a real line (in the picture) and a fake line (not). The change has
// both. Real: weight 1; fake: weight 0. The same under an affine change of the picture (3 V + 0.2)
// and under the Log3G10-like curve. Mutation: "(c - lo) / span" -> "(1.0 - lo) / span" (every
// candidate full weight: the fake line passes).
void testEvidence() {
  const int w = 120, h = 80;
  const size_t n = size_t(w) * h;
  std::vector<float> rgb(n * 3), change(n, 0.0f);
  std::vector<uint8_t> cand(n, 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const float bgc[3] = {0.22f, 0.27f, 0.33f};
      float a = 0;
      if (y == 20 && x > 10 && x < 110) a = 0.7f;  // the real strand, also in the picture
      for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = bgc[c] + a * (0.6f - bgc[c]) + 0.01f * float((x * 7 + y * 13) % 5) / 5;
      if (y == 20 && x > 10 && x < 110) change[i] = 0.7f;
      if (y == 50 && x > 10 && x < 110) change[i] = 0.7f;  // fake
      cand[i] = change[i] > 0;
    }
  auto run = [&](const std::vector<float>& pic, double* real, double* fake) {
    std::vector<float> wgt(n);
    pictureEvidence(pic.data(), change.data(), cand.data(), w, h, EvidenceParams(), wgt.data());
    double r = 0, f = 0;
    int nr = 0, nf = 0;
    for (int x = 20; x < 100; ++x) r += wgt[size_t(20) * w + x], ++nr, f += wgt[size_t(50) * w + x], ++nf;
    *real = r / nr;
    *fake = f / nf;
  };
  std::vector<float> affine(rgb), logp(rgb);
  for (size_t i = 0; i < rgb.size(); ++i) {
    affine[i] = 3.0f * rgb[i] + 0.2f;
    const float x = rgb[i] + 0.01f;
    logp[i] = 0.224282f * std::log10(x * 155.975327f + 1.0f);
  }
  double r0, f0, r1, f1, r2, f2;
  run(rgb, &r0, &f0);
  run(affine, &r1, &f1);
  run(logp, &r2, &f2);
  std::printf("evidence: real %.3f / fake %.3f (display), %.3f / %.3f (3V+0.2), %.3f / %.3f (log)\n", r0, f0, r1, f1, r2, f2);
  CHECK(r0 > 0.95 && f0 < 0.05);
  CHECK(std::fabs(r1 - r0) < 1e-3 && std::fabs(f1 - f0) < 1e-3);
  CHECK(r2 > 0.95 && f2 < 0.05);
}

}  // namespace

int main() {
  testDistance();
  testMinMax();
  testRegistration();
  testAddKnown();
  testAddReference();
  testBridge();
  testInvariants();
  testBlendTrimapKnown();
  testTrimapDisconnected();
  testEvidence();
  std::printf(failures ? "FAILED (%d failure(s))\n" : "OK (0 failure(s))\n", failures);
  return failures ? 1 : 0;
}
