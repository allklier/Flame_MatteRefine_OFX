#include "ops/analyze_frame.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <vector>

#include "ops/morphology.h"

namespace mr {

namespace {

constexpr int kLevels = 1000;  // matte values as integer levels: lv = floor(m * 1000)
constexpr int kGridLo = 500, kGridHi = 990, kGridStep = 10;  // Solid Above candidates 0.50-0.99
constexpr int kDefaultK = 950;  // the plugin's default Solid Above, 0.95

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

// Two significant digits, for percentages of the frame.
double round2(double v) {
  if (v <= 0) return 0;
  const double s = std::pow(10.0, std::floor(std::log10(v)) - 1);
  return std::round(v / s) * s;
}

std::string pct(double v) {  // a % of the frame, readable at any size
  return v >= 1 ? fmt("%.1f", v) : v >= 0.1 ? fmt("%.2f", v) : v >= 0.01 ? fmt("%.3f", v)
                                                                              : fmt("%.4f", v);
}

struct Dsu {
  std::vector<int> p;
  explicit Dsu(size_t n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  int find(int a) {
    while (p[a] != a) a = p[a] = p[p[a]];
    return a;
  }
  void unite(int a, int b) {
    a = find(a), b = find(b);
    if (a != b) p[std::max(a, b)] = std::min(a, b);
  }
};

// The escape level of every pixel: over all 4-connected paths from it to an open frame edge,
// the lowest possible highest level on the path (the pixel itself included). At Solid Above
// level k a pixel is outside (decision 9) iff esc < k, and a hole candidate iff lv < k <= esc:
// one priority flood answers every threshold at once. Only k >= kGridLo matters, so all levels
// below it are one: the background joined to an open edge below kGridLo is found first as row
// runs (fast: long runs), gets kGridLo - 1, and the flood (a bucket queue) starts at its border.
// No open edge: every pixel gets kLevels + 1.
std::vector<uint16_t> escapeLevels(const std::vector<uint16_t>& lv, int w, int h,
                                   const bool open[4]) {
  const size_t n = lv.size();
  const uint16_t unset = kLevels + 1, low = kGridLo - 1;
  // 1. Runs below kGridLo, joined between rows when they overlap (4-connected).
  struct Run {
    int y, x0, x1;
  };
  std::vector<Run> runs;
  std::vector<size_t> rowStart(size_t(h) + 1);
  for (int y = 0; y < h; ++y) {
    rowStart[y] = runs.size();
    const uint16_t* row = &lv[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (row[x] >= kGridLo) continue;
      const int x0 = x;
      while (x + 1 < w && row[x + 1] < kGridLo) ++x;
      runs.push_back({y, x0, x});
    }
  }
  rowStart[h] = runs.size();
  Dsu d(runs.size());
  for (int y = 1; y < h; ++y) {
    size_t a = rowStart[y - 1], b = rowStart[y];
    while (a < rowStart[y] && b < rowStart[y + 1]) {
      if (runs[a].x0 <= runs[b].x1 && runs[b].x0 <= runs[a].x1) d.unite(int(a), int(b));
      (runs[a].x1 < runs[b].x1 ? a : b)++;
    }
  }
  std::vector<uint8_t> outRoot(runs.size(), 0);
  for (size_t r = 0; r < runs.size(); ++r)
    if ((open[0] && runs[r].x0 == 0) || (open[1] && runs[r].x1 == w - 1) ||
        (open[2] && runs[r].y == 0) || (open[3] && runs[r].y == h - 1))
      outRoot[d.find(int(r))] = 1;
  // The flood works on a copy with a 1 px border marked done, so neighbours need no checks.
  const int pw = w + 2;
  for (size_t r = 0; r < runs.size(); ++r) outRoot[r] = outRoot[d.find(int(r))];
  std::vector<uint16_t> pe(size_t(pw) * (h + 2), 0);  // border: 0 = done
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      uint16_t* row = &pe[size_t(y + 1) * pw + 1];
      std::fill(row, row + w, unset);
      for (size_t r = rowStart[y]; r < rowStart[y + 1]; ++r)
        if (outRoot[r]) std::fill(row + runs[r].x0, row + runs[r].x1 + 1, low);
    }
  });
  // 2. Seeds: the rest of the open frame edges, and every pixel next to that background.
  std::vector<std::vector<int>> bucket(kGridHi);
  auto lvAt = [&](int p) { return lv[size_t(p / pw - 1) * w + (p % pw - 1)]; };
  std::vector<std::vector<int>> seeds(static_cast<size_t>(h));  // per row, found on threads
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const int p = (y + 1) * pw + x + 1;
        if (pe[p] != unset) continue;
        const bool edge = (open[0] && x == 0) || (open[1] && x == w - 1) || (open[2] && y == 0) ||
                          (open[3] && y == h - 1);
        if (edge || pe[p - 1] == low || pe[p + 1] == low || pe[p - pw] == low || pe[p + pw] == low)
          seeds[y].push_back(p);
      }
  });
  for (const auto& row : seeds)
    for (int p : row) {
      const uint16_t e = std::max(lvAt(p), low);
      if (e >= kGridHi) continue;
      pe[p] = e;
      bucket[e].push_back(p);
    }
  // 3. The flood, lowest level first. It stops below kGridHi: no candidate threshold tells
  // levels from there up apart, and most of a subject's interior is up there (a pixel left
  // unset keeps kLevels + 1, which reads the same).
  for (int cur = low; cur < kGridHi; ++cur) {
    std::vector<int>& b = bucket[cur];
    for (size_t k = 0; k < b.size(); ++k) {  // b grows while we walk it
      const int p = b[k];
      for (const int q : {p - 1, p + 1, p - pw, p + pw}) {
        if (pe[q] != unset) continue;
        const uint16_t e = std::max<uint16_t>(uint16_t(cur), lvAt(q));
        if (e >= kGridHi) continue;
        pe[q] = e;
        bucket[e].push_back(q);
      }
    }
    std::vector<int>().swap(b);
  }
  std::vector<uint16_t> esc(n);
  for (int y = 0; y < h; ++y) std::copy(&pe[size_t(y + 1) * pw + 1], &pe[size_t(y + 1) * pw + 1] + w, &esc[size_t(y) * w]);
  return esc;
}

struct Hole {
  long long size = 0;
  double mass = 0;  // sum of (1 - m): how much filling it adds
};

// Splits enclosed areas into defects and real gaps by size: the largest size jump (>= 4x)
// whose upper side is at least 0.1 % of the frame (t36's merged grooves reach 0.05 %). Areas of at most the returned size are
// defects. No such jump: all defects when every area is small, else the large ones are kept.
double splitSizes(std::vector<long long> s, double n) {
  std::sort(s.rbegin(), s.rend());
  const double big = 0.001 * n;
  int best = -1;
  double bestRatio = 4.0;
  for (size_t i = 0; i + 1 < s.size(); ++i)
    if (s[i] >= big && double(s[i]) / double(s[i + 1]) >= bestRatio) {
      bestRatio = double(s[i]) / double(s[i + 1]);
      best = int(i);
    }
  if (best >= 0) return std::sqrt(double(s[best]) * double(s[best + 1]));
  if (s.empty() || s[0] < big) return 1e30;
  size_t j = 0;
  while (j < s.size() && s[j] >= big) ++j;
  return j < s.size() ? std::sqrt(double(s[j - 1]) * double(s[j])) : 0.5 * double(s[j - 1]);
}

// Dip pixels (a hole candidate at some Solid Above on the grid), in pixel order, with the
// position of each one's right and lower neighbour in the list (-1: not a dip).
struct Dips {
  std::vector<int> px, right, down;
};

template <class Pred>
Dips findDips(size_t n, int w, Pred isDip) {
  Dips d;
  for (size_t i = 0; i < n; ++i)
    if (isDip(i)) d.px.push_back(int(i));
  const size_t c = d.px.size();
  d.right.assign(c, -1);
  d.down.assign(c, -1);
  parallelFor(int(c), 4096, [&](int a0, int a1) {
    for (int a = a0; a < a1; ++a) {
      const int i = d.px[a];
      if ((i + 1) % w != 0 && a + 1 < int(c) && d.px[a + 1] == i + 1) d.right[a] = a + 1;
      const auto it = std::lower_bound(d.px.begin() + a, d.px.end(), i + w);
      if (it != d.px.end() && *it == i + w) d.down[a] = int(it - d.px.begin());
    }
  });
  return d;
}

// Hole components (4-connected) at Solid Above level k, among the dip pixels.
std::vector<Hole> holesAt(int k, const Dips& dp, const std::vector<uint16_t>& lv,
                          const std::vector<uint16_t>& esc, const float* m,
                          std::vector<int>* compOf = nullptr) {
  const std::vector<int>& dips = dp.px;
  Dsu d(dips.size());
  auto active = [&](int a) { return lv[dips[a]] < k && k <= esc[dips[a]]; };
  for (size_t a = 0; a < dips.size(); ++a) {
    if (!active(int(a))) continue;
    if (dp.right[a] >= 0 && active(dp.right[a])) d.unite(int(a), dp.right[a]);
    if (dp.down[a] >= 0 && active(dp.down[a])) d.unite(int(a), dp.down[a]);
  }
  std::vector<int> slot(dips.size(), -1);
  std::vector<Hole> holes;
  if (compOf) compOf->assign(dips.size(), -1);
  for (size_t a = 0; a < dips.size(); ++a) {
    const int i = dips[a];
    if (!active(int(a))) continue;
    const int r = d.find(int(a));
    if (slot[r] < 0) {
      slot[r] = int(holes.size());
      holes.emplace_back();
    }
    holes[slot[r]].size++;
    holes[slot[r]].mass += 1.0 - m[i];
    if (compOf) (*compOf)[a] = slot[r];
  }
  return holes;
}

// The sizes of the 8-connected components of `mask` (row runs + union-find).
std::vector<long long> components8(const Mask& mask, int w, int h) {
  struct Run {
    int y, x0, x1;
  };
  std::vector<Run> runs;
  std::vector<size_t> rowStart(size_t(h) + 1);
  for (int y = 0; y < h; ++y) {
    rowStart[y] = runs.size();
    const uint8_t* row = &mask[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (!row[x]) continue;
      const int x0 = x;
      while (x + 1 < w && row[x + 1]) ++x;
      runs.push_back({y, x0, x});
    }
  }
  rowStart[h] = runs.size();
  Dsu d(runs.size());
  for (int y = 1; y < h; ++y) {
    size_t a = rowStart[y - 1], b = rowStart[y];
    while (a < rowStart[y] && b < rowStart[y + 1]) {
      if (runs[a].x0 <= runs[b].x1 + 1 && runs[b].x0 <= runs[a].x1 + 1) d.unite(int(a), int(b));
      (runs[a].x1 < runs[b].x1 ? a : b)++;
    }
  }
  std::vector<int> slot(runs.size(), -1);
  std::vector<long long> sizes;
  for (size_t r = 0; r < runs.size(); ++r) {
    const int root = d.find(int(r));
    if (slot[root] < 0) {
      slot[root] = int(sizes.size());
      sizes.push_back(0);
    }
    sizes[slot[root]] += runs[r].x1 - runs[r].x0 + 1;
  }
  return sizes;
}

}  // namespace

void analyzeFrame(const float* matte, int w, int h, AnalysisResult* out) {
  const auto t0 = std::chrono::steady_clock::now();
  AnalysisResult& r = *out;
  r = AnalysisResult();
  auto done = [&]() {
    r.measured.ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  if (n < 64 || !matte) {
    r.tooSmall = true;
    r.notes.push_back("The frame is too small to analyse; nothing changed.");
    return done();
  }
  const double nd = double(n);

  // --- The matte, clamped (decision 5), and its overall shape ---
  std::vector<float> m(n);
  float lo = 1.0f, hi = 0.0f;
  long long subject = 0, grey = 0, boundary = 0;  // boundary: subject pixels next to < 0.5
  std::mutex mu;
  parallelFor(h, 16, [&](int y0, int y1) {
    float l = 1.0f, u = 0.0f;
    long long s = 0, g = 0;
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float v = matte[i];
      m[i] = v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f;  // NaN -> 0
      l = std::min(l, m[i]);
      u = std::max(u, m[i]);
      s += m[i] >= 0.5f;
      g += m[i] > 0.15f && m[i] < 0.85f;  // above haze, below a soft interior
    }
    std::lock_guard<std::mutex> lock(mu);
    lo = std::min(lo, l);
    hi = std::max(hi, u);
    subject += s;
    grey += g;
  });
  r.measured.subjectPercent = 100.0 * subject / nd;
  r.measured.greyPercent = 100.0 * grey / nd;
  if (hi - lo < 0.01f) {
    r.constantMatte = true;
    r.notes.push_back(fmt("The matte is the same everywhere (%.2f): connect a matte to the Matte "
                          "input (or the source alpha). Nothing changed.", hi));
    return done();
  }
  if (subject < std::max<long long>(16, (long long)(1e-4 * nd))) {
    r.noSubject = true;
    r.notes.push_back(fmt("No subject to analyse: %s %% of the frame is at or above 0.5. Nothing "
                          "changed.", pct(r.measured.subjectPercent).c_str()));
    return done();
  }
  parallelFor(h, 16, [&](int y0, int y1) {
    long long b = 0;
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (m[i] < 0.5f) continue;
        b += (x > 0 && m[i - 1] < 0.5f) || (x + 1 < w && m[i + 1] < 0.5f) ||
             (y > 0 && m[i - w] < 0.5f) || (y + 1 < h && m[i + w] < 0.5f);
      }
    std::lock_guard<std::mutex> lock(mu);
    boundary += b;
  });
  if (r.measured.greyPercent > 25.0 || boundary > 0.1 * nd) {
    r.noisy = true;
    r.notes.push_back(fmt("The matte is grey or speckled almost everywhere (%.0f %% of the frame "
                          "between 0.15 and 0.85, %.0f %% on an edge): no reliable reading, "
                          "nothing changed. Clean the key first.",
                          r.measured.greyPercent, 100.0 * boundary / nd));
    return done();
  }
  // ===== Clean & Crush (runs first in the chain) =====
  std::vector<std::string> unchangedNotes;
  {
    CleanBackgroundParams& cp = r.clean;
    // Haze: the background more than 8 px from anything at or above Background Below.
    // In 8 px blocks: a pixel is far when its block and the 8 around it hold nothing at or
    // above Background Below (so it is at least 8 px away). Values as a 1000-bin histogram.
    Mask above(n);
    constexpr int kB = 8;
    const int bw = (w + kB - 1) / kB, bh = (h + kB - 1) / kB;
    std::vector<uint8_t> blk(size_t(bw) * bh, 0);
    auto markAbove = [&]() {
      parallelFor(bh, 4, [&](int b0, int b1) {  // block rows: each writes only its own
        for (int y = b0 * kB; y < std::min(h, b1 * kB); ++y)
          for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            above[i] = m[i] >= cp.backgroundBelow;
            if (above[i]) blk[size_t(y / kB) * bw + x / kB] = 1;
          }
      });
    };
    markAbove();
    std::vector<long long> hist(kLevels + 1, 0);
    long long farN = 0, hazy = 0;
    double farSum = 0;
    parallelFor(bh, 4, [&](int b0, int b1) {
      std::vector<long long> hl(kLevels + 1, 0);
      long long fn = 0, hz = 0;
      double fs = 0;
      for (int by = b0; by < b1; ++by)
        for (int bx = 0; bx < bw; ++bx) {
          bool any = false;
          for (int dy = -1; dy <= 1 && !any; ++dy)
            for (int dx = -1; dx <= 1 && !any; ++dx) {
              const int ny = by + dy, nx = bx + dx;
              any = nx >= 0 && ny >= 0 && nx < bw && ny < bh && blk[size_t(ny) * bw + nx];
            }
          if (any) continue;
          for (int y = by * kB; y < std::min(h, (by + 1) * kB); ++y)
            for (int x = bx * kB; x < std::min(w, (bx + 1) * kB); ++x) {
              const float v = m[size_t(y) * w + x];
              hl[std::min(kLevels, int(v * kLevels))]++;
              ++fn;
              hz += v > 0.02f;
              fs += v;
            }
        }
      std::lock_guard<std::mutex> lock(mu);
      for (int k = 0; k <= kLevels; ++k) hist[k] += hl[k];
      farN += fn;
      hazy += hz;
      farSum += fs;
    });
    double haze = 0;
    for (long long c = 0, k = 0; k <= kLevels; ++k) {
      c += hist[k];
      if (double(c) >= 0.999 * double(farN)) {
        haze = double(k + 1) / kLevels;
        break;
      }
    }
    r.measured.hazeLevel = haze;
    const bool hasHaze = farN > 0 && hazy > 0.01 * double(farN);
    if (hasHaze && haze > 0.8 * cp.backgroundBelow) {
      cp.backgroundBelow = float(std::min(0.5, std::ceil((haze + 0.05) * 20.0) / 20.0));
      r.changed.backgroundBelow = true;
      markAbove();
    }

    // Islands: 8-connected parts at Background Below other than the subject. Parts of at least
    // 1 % of the frame, and the largest part, are the subject or objects, never islands.
    const std::vector<long long> sizes = components8(above, w, h);
    const long long largest = sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end());
    std::vector<long long> isl;
    bool skippedLargest = false;
    for (long long s : sizes) {
      if (s == largest && !skippedLargest) {
        skippedLargest = true;
        continue;
      }
      if (s < 0.01 * nd) isl.push_back(s);
    }
    std::sort(isl.rbegin(), isl.rend());
    // The limit: the largest jump (>= 4x) whose upper side is at least 0.02 % of the frame;
    // without one, the default (0.05 %).
    double limitPx = cp.maxIslandPercent / 100.0 * nd;
    int best = -1;
    double bestRatio = 4.0;
    for (size_t i = 0; i + 1 < isl.size(); ++i)
      if (isl[i] >= 2e-4 * nd && double(isl[i]) / double(isl[i + 1]) >= bestRatio) {
        bestRatio = double(isl[i]) / double(isl[i + 1]);
        best = int(i);
      }
    if (best >= 0) {
      const double lower = double(isl[best + 1]), upper = double(isl[best]);
      if (!(limitPx >= 1.5 * lower && limitPx * 1.5 <= upper)) {
        double pc = round2(std::sqrt(lower * upper) / nd * 100.0);
        cp.maxIslandPercent = pc;
        limitPx = pc / 100.0 * nd;
        r.changed.maxIsland = true;
      }
    } else if (!isl.empty() && double(isl[0]) > limitPx && double(isl[0]) <= 1.5 * limitPx) {
      // A single size class just above the default: raise it to include them.
      cp.maxIslandPercent = round2(1.5 * double(isl[0]) / nd * 100.0);
      limitPx = cp.maxIslandPercent / 100.0 * nd;
      r.changed.maxIsland = true;
    }
    long long removed = 0, removedPx = 0, biggestRemoved = 0;
    std::vector<long long> kept;
    for (long long s : isl) {
      if (double(s) <= limitPx) {
        ++removed;
        removedPx += s;
        biggestRemoved = std::max(biggestRemoved, s);
      } else {
        kept.push_back(s);
      }
    }
    r.measured.islands = int(removed);
    r.measured.objects = int(kept.size());

    // Crush: the background's own falloff from the subject (mean value per px outside), so the
    // crush starts beyond it.
    int tail = 0;
    if (hasHaze) {
      const double floor = farSum / double(farN);
      // Rings of 4-neighbour steps out from the matte (a cheap distance: only the band).
      std::vector<double> ringSum(11, 0.0);
      std::vector<long long> ringN(11, 0);
      Mask seen = above;
      std::vector<int> ring, next;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if (seen[i]) continue;
          if ((x > 0 && above[i - 1]) || (x + 1 < w && above[i + 1]) || (y > 0 && above[i - w]) ||
              (y + 1 < h && above[i + w]))
            ring.push_back(int(i));
        }
      for (int i : ring) seen[i] = 1;
      for (int k = 1; k <= 10 && !ring.empty(); ++k) {
        next.clear();
        for (int i : ring) {
          ringSum[k] += m[i];
          ringN[k]++;
          const int x = i % w, y = i / w;
          auto visit = [&](int j) {
            if (!seen[j]) {
              seen[j] = 1;
              next.push_back(j);
            }
          };
          if (x > 0) visit(i - 1);
          if (x + 1 < w) visit(i + 1);
          if (y > 0) visit(i - w);
          if (y + 1 < h) visit(i + w);
        }
        ring.swap(next);
      }
      tail = 10;
      for (int k = 1; k <= 10; ++k)
        if (ringN[k] > 0 && ringSum[k] / double(ringN[k]) <= floor + 0.01) {
          tail = k;
          break;
        }
      if (tail > cp.crushPx) {
        cp.crushPx = double(tail);
        r.changed.crushPx = true;
      }
    }
    if (removed > 0 || hasHaze) {
      r.cleanEnable = true;
      r.changed.cleanEnable = true;
      std::string why;
      if (removed > 0)
        why = fmt("%lld small island%s outside the subject (up to %s %% of the frame)", removed,
                  removed == 1 ? "" : "s", pct(100.0 * biggestRemoved / nd).c_str());
      if (hasHaze)
        why += fmt("%sbackground haze up to %.2f in %.0f %% of the background", why.empty() ? "" : " and ",
                   haze, 100.0 * hazy / double(farN));
      r.notes.push_back("Clean & Crush on: " + why + ".");
      if (r.changed.backgroundBelow)
        r.notes.push_back(fmt("Background Below %.2f (was 0.15): the haze reaches %.2f, so it stays "
                              "below the level islands are measured at.", cp.backgroundBelow, haze));
      if (removed > 0)
        r.notes.push_back(fmt("Max Island %% %s%s: removes the %lld island%s (largest %s %%)",
                              pct(cp.maxIslandPercent).c_str(), r.changed.maxIsland ? "" : " (default)",
                              removed, removed == 1 ? "" : "s", pct(100.0 * biggestRemoved / nd).c_str()) +
                          (kept.empty() ? std::string(".")
                                        : fmt("; %zu larger piece%s (%s%s %%) %s kept as separate "
                                              "objects: raise Max Island %% above %s to remove %s too.",
                                              kept.size(), kept.size() == 1 ? "" : "s",
                                              pct(100.0 * kept.back() / nd).c_str(),
                                              kept.size() == 1 ? "" : fmt("-%s", pct(100.0 * kept.front() / nd).c_str()).c_str(),
                                              kept.size() == 1 ? "is" : "are",
                                              pct(100.0 * kept.front() / nd).c_str(),
                                              kept.size() == 1 ? "it" : "them")));
      if (hasHaze)
        r.notes.push_back(fmt("Crush Beyond px %.0f%s: everything farther outside the edge goes to "
                              "0; the subject's own falloff ends about %d px out.",
                              cp.crushPx, r.changed.crushPx ? "" : " (default)", tail));
    } else {
      // Keep the defaults on screen: nothing to clean.
      r.clean = CleanBackgroundParams();
      r.changed = AnalysisResult::Changed();
      r.measured.objects = int(kept.size());
      unchangedNotes.push_back(
          kept.empty() ? "Clean & Crush: no islands and no background haze; left off."
                       : fmt("Clean & Crush: no small islands and no haze; %zu separate piece%s "
                             "(%s %% of the frame or more) look%s like real objects; left off.",
                             kept.size(), kept.size() == 1 ? "" : "s",
                             pct(100.0 * kept.back() / nd).c_str(), kept.size() == 1 ? "s" : ""));
    }
  }
  std::vector<float> cur = m;
  if (r.cleanEnable) {
    CleanBackgroundStats st;
    cleanBackground(m.data(), cur.data(), w, h, r.clean, &st);
    r.measured.cleanChanged = st.pixelsRemoved + st.pixelsCrushed;
  }
  // ===== Fill Holes =====
  {
    // Frame edges the subject is cut by: solid along at least 0.5 % of the edge (and 3 px).
    const int nEdge[4] = {h, h, w, w};
    bool touches[4];
    for (int e = 0; e < 4; ++e) {
      long long c = 0;
      for (int k = 0; k < nEdge[e]; ++k) {
        const size_t i = e == 0 ? size_t(k) * w : e == 1 ? size_t(k) * w + w - 1
                       : e == 2 ? size_t(k) : size_t(h - 1) * w + k;
        c += cur[i] >= 0.5f;
      }
      touches[e] = c >= std::max(3.0, 0.005 * nEdge[e]);
    }
    std::vector<uint16_t> lv(n);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i)
        lv[i] = uint16_t(std::min(kLevels, int(cur[i] * kLevels)));
    });
    const bool open[4] = {!touches[0], !touches[1], !touches[2], !touches[3]};
    const std::vector<uint16_t> esc = escapeLevels(lv, w, h, open);
    // Dip pixels: a hole candidate at some Solid Above on the grid.
    const Dips dips = findDips(n, w, [&](size_t i) {
      return lv[i] < esc[i] && esc[i] >= kGridLo && lv[i] < kGridHi;
    });
    // Per candidate Solid Above: the defect holes' fill (sum of 1 - m), real gaps excluded.
    struct Cand {
      int k;
      double mass, limit;
    };
    std::vector<Cand> cands;
    double best = 0;
    for (int k = kGridLo; k <= kGridHi; k += kGridStep) cands.push_back({k, 0.0, 0.0});
    parallelFor(int(cands.size()), 1, [&](int c0, int c1) {  // independent thresholds
      for (int c = c0; c < c1; ++c) {
        const std::vector<Hole> holes = holesAt(cands[c].k, dips, lv, esc, cur.data());
        std::vector<long long> sizes;
        for (const Hole& hl : holes) sizes.push_back(hl.size);
        cands[c].limit = splitSizes(sizes, nd);
        for (const Hole& hl : holes)
          if (double(hl.size) <= cands[c].limit) cands[c].mass += hl.mass;
      }
    });
    for (const Cand& c : cands) best = std::max(best, c.mass);
    const double minMass = std::max(20.0, 1e-5 * nd);
    if (best < minMass) {
      // No defects. Still check the real gaps against the default Max Hole % (frame edges open,
      // as the defaults leave them): a gap just under it would be filled.
      const bool allOpen[4] = {true, true, true, true};
      const bool anyClosed = touches[0] || touches[1] || touches[2] || touches[3];
      const std::vector<uint16_t> esc0 = anyClosed ? escapeLevels(lv, w, h, allOpen) : esc;
      std::vector<long long> sizes;
      const Dips dips0 =
          findDips(n, w, [&](size_t i) { return lv[i] < kDefaultK && kDefaultK <= esc0[i]; });
      for (const Hole& hl : holesAt(kDefaultK, dips0, lv, esc0, cur.data()))
        sizes.push_back(hl.size);
      const double limit = splitSizes(sizes, nd);
      long long minGap = 0;
      for (long long sz : sizes)
        if (double(sz) > limit) minGap = minGap == 0 ? sz : std::min(minGap, sz);
      const double defPx = r.fill.maxHolePercent / 100.0 * nd;
      if (minGap > 0 && double(minGap) <= defPx) {
        r.fill.maxHolePercent = round2(0.5 * double(minGap) / nd * 100.0);
        r.changed.maxHole = true;
        r.notes.push_back(fmt("Max Hole %% %s: no holes to fill, but an enclosed area of %s %% of "
                              "the frame looks like a real gap (e.g. an arm on a hip) and is just "
                              "under the default 1 %%, so Fill Holes would close it.",
                              pct(r.fill.maxHolePercent).c_str(), pct(100.0 * minGap / nd).c_str()));
        unchangedNotes.push_back("Fill Holes: no holes inside the subject; the other settings "
                                 "are left at their defaults.");
      } else {
        unchangedNotes.push_back("Fill Holes: no holes inside the subject; left at its defaults.");
      }
    } else {
      // Solid Above: the default (0.95, right in both Flame tests) when it fills at least 90 %
      // of what the best setting fills (fill = sum of 1 - m over the defect holes). Otherwise the
      // middle of the settings that fill 97 % of the best: lower leaves grey holes open, higher
      // lets them leak out through the subject's own grey.
      size_t argBest = 0, def = 0;
      for (size_t c = 0; c < cands.size(); ++c) {
        if (cands[c].mass > cands[argBest].mass) argBest = c;
        if (cands[c].k == kDefaultK) def = c;
      }
      auto range = [&](double share, size_t* lo, size_t* hi) {  // contiguous, around the best
        *lo = *hi = argBest;
        while (*lo > 0 && cands[*lo - 1].mass >= share * best) --*lo;
        while (*hi + 1 < cands.size() && cands[*hi + 1].mass >= share * best) ++*hi;
      };
      size_t lo97, hi97, lo90, top;
      range(0.97, &lo97, &hi97);
      range(0.9, &lo90, &top);
      const size_t pick = cands[def].mass >= 0.9 * best ? def : (lo97 + hi97) / 2;
      const int k = cands[pick].k;
      FillHolesParams& fp = r.fill;
      fp.threshold = float(k) / kLevels;
      r.changed.threshold = std::fabs(fp.threshold - 0.95f) > 1e-4f;

      std::vector<int> compOf;
      const std::vector<Hole> holes = holesAt(k, dips, lv, esc, cur.data(), &compOf);
      const double limit = cands[pick].limit;
      long long maxDefect = 0, minGap = 0, holePx = 0;
      int nHoles = 0, nGaps = 0;
      for (const Hole& hl : holes) {
        if (double(hl.size) <= limit) {
          ++nHoles;
          holePx += hl.size;
          maxDefect = std::max(maxDefect, hl.size);
        } else {
          ++nGaps;
          minGap = minGap == 0 ? hl.size : std::min(minGap, hl.size);
        }
      }
      r.measured.holes = nHoles;
      r.measured.realGaps = nGaps;
      r.measured.holePixels = holePx;
      r.measured.largestHolePercent = 100.0 * maxDefect / nd;
      r.measured.smallestGapPercent = 100.0 * minGap / nd;
      // The holes' values (5th-95th percentile) and how many reach a Closed edge, for the notes.
      std::vector<float> hv;
      std::vector<uint8_t> atEdge(holes.size(), 0);
      for (size_t a = 0; a < dips.px.size(); ++a)
        if (compOf[a] >= 0 && double(holes[compOf[a]].size) <= limit) {
          const int i = dips.px[a], x = i % w, y = i / w;
          hv.push_back(cur[i]);
          if ((touches[0] && x == 0) || (touches[1] && x == w - 1) || (touches[2] && y == 0) ||
              (touches[3] && y == h - 1))
            atEdge[compOf[a]] = 1;
        }
      const int holesAtEdge = int(std::count(atEdge.begin(), atEdge.end(), 1));
      std::sort(hv.begin(), hv.end());
      const float hv5 = hv.empty() ? 0 : hv[hv.size() / 20], hv95 = hv.empty() ? 0 : hv[hv.size() * 19 / 20];

      // Max Hole %: between the largest defect and the smallest real gap; the default when it
      // already sits there with room on both sides (1.5x).
      const double defPx = fp.maxHolePercent / 100.0 * nd;
      if (nGaps > 0) {
        if (!(defPx >= 1.5 * double(maxDefect) && 1.5 * defPx <= double(minGap))) {
          fp.maxHolePercent = round2(std::sqrt(double(std::max<long long>(maxDefect, 1)) * double(minGap)) / nd * 100.0);
          r.changed.maxHole = true;
        }
      } else if (double(maxDefect) * 1.5 > defPx) {
        fp.maxHolePercent = round2(1.5 * double(maxDefect) / nd * 100.0);
        r.changed.maxHole = true;
      }
      // Frame edges the subject is cut by count as solid (the roadmap's rule): holes reaching
      // them fill, real gaps between the subject and the edge are kept open by Max Hole %.
      fp.closedLeft = r.changed.closedLeft = touches[0];
      fp.closedRight = r.changed.closedRight = touches[1];
      fp.closedTop = r.changed.closedTop = touches[2];
      fp.closedBottom = r.changed.closedBottom = touches[3];

      const std::string span = top > lo90 ? fmt("anything from %.2f to %.2f fills about as much",
                                                float(cands[lo90].k) / kLevels,
                                                float(cands[top].k) / kLevels)
                                          : "";
      std::string above = top + 1 < cands.size()
                              ? fmt("above %.2f %s", float(cands[top + 1].k) / kLevels,
                                    cands[top + 1].mass < 0.8 * best
                                        ? "they join the outside through the subject's own grey and stay open"
                                        : "less fills")
                              : "";
      r.notes.push_back(fmt("Solid Above %.2f%s: %d hole%s (values %.2f-%.2f) fill", fp.threshold,
                            r.changed.threshold ? "" : " (default)", nHoles, nHoles == 1 ? "" : "s",
                            hv5, hv95) +
                        (span.empty() ? "" : "; " + span) + (above.empty() ? "" : "; " + above) + ".");
      if (nGaps > 0)
        r.notes.push_back(fmt("Max Hole %% %s%s: the holes are at most %s %% of the frame; %d "
                              "enclosed area%s of %s %% or more look%s like real gaps (e.g. an arm on "
                              "a hip) and stay%s open.",
                              pct(fp.maxHolePercent).c_str(), r.changed.maxHole ? "" : " (default)",
                              pct(100.0 * maxDefect / nd).c_str(), nGaps, nGaps == 1 ? "" : "s",
                              pct(100.0 * minGap / nd).c_str(), nGaps == 1 ? "s" : "",
                              nGaps == 1 ? "s" : ""));
      else
        r.notes.push_back(fmt("Max Hole %% %s%s: every enclosed area is small (largest %s %% of the "
                              "frame), so all are filled.",
                              pct(fp.maxHolePercent).c_str(), r.changed.maxHole ? "" : " (default)",
                              pct(100.0 * maxDefect / nd).c_str()));
      const char* names[4] = {"Left", "Right", "Top", "Bottom"};
      std::string cut;
      for (int e = 0; e < 4; ++e)
        if (touches[e]) cut += std::string(cut.empty() ? "" : ", ") + names[e];
      if (!cut.empty())
        r.notes.push_back(fmt("%s Closed on: the subject is cut by %s, so holes reaching it fill "
                              "too (%s)%s.", cut.c_str(),
                              cut.find(',') != std::string::npos ? "these frame edges" : "this frame edge",
                              holesAtEdge ? fmt(holesAtEdge == 1 ? "%d hole does" : "%d of the holes do", holesAtEdge).c_str()
                                          : "none does in this frame",
                              nGaps > 0 ? "; areas it encloses between the subject and the edge are "
                                          "judged by size like the holes" : ""));
      // Bridge Gaps (decision 9): holes that leak out through a narrow gap fill only when
      // bridged. Checked with the op itself at half resolution (cost): Bridge 8 px against 0 px,
      // the rest as suggested; then 4 px, the smaller when it does most of it. Only on the
      // subject's bounding box (+16 px): everything around it is background joined to an open
      // edge below 0.5, so the box's sides act as open edges and the result is the same.
      int bx0 = w, by0 = h, bx1 = -1, by1 = -1;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          if (esc[size_t(y) * w + x] >= kGridLo) {
            bx0 = std::min(bx0, x), bx1 = std::max(bx1, x);
            by0 = std::min(by0, y), by1 = std::max(by1, y);
          }
      constexpr int kMargin = 16;
      bx0 = std::max(0, bx0 - kMargin) & ~1, by0 = std::max(0, by0 - kMargin) & ~1;
      bx1 = std::min(w - 1, bx1 + kMargin), by1 = std::min(h - 1, by1 + kMargin);
      const int hw = (bx1 - bx0 + 1) / 2, hh = (by1 - by0 + 1) / 2;
      if (hw >= 16 && hh >= 16) {
        std::vector<float> half(size_t(hw) * hh), tmp(half.size());
        parallelFor(hh, 16, [&](int y0, int y1) {
          for (int y = y0; y < y1; ++y)
            for (int x = 0; x < hw; ++x) {
              const size_t i = size_t(by0 + 2 * y) * w + bx0 + 2 * x;
              half[size_t(y) * hw + x] = 0.25f * (cur[i] + cur[i + 1] + cur[i + w] + cur[i + w + 1]);
            }
        });
        auto filled = [&](double bridgeFull) {
          FillHolesParams p = fp;
          p.bridgePx = bridgeFull / 2.0;
          p.rimPx = 0.0;
          p.maxHolePercent = fp.maxHolePercent * nd / (4.0 * double(hw) * hh);  // of the frame
          p.closedLeft = fp.closedLeft && bx0 == 0;
          p.closedRight = fp.closedRight && bx0 + 2 * hw >= w - 1;
          p.closedTop = fp.closedTop && by0 == 0;
          p.closedBottom = fp.closedBottom && by0 + 2 * hh >= h - 1;
          FillHolesStats st;
          fillHoles(half.data(), tmp.data(), hw, hh, p, &st);
          return 4 * st.pixelsChanged;  // in full-resolution pixels
        };
        const long long base = filled(0.0), gain8 = filled(8.0) - base;
        // Enough to matter: 50 px (2e-5 of the frame at UHD) and 2 % of the holes' area.
        if (double(gain8) >= std::max({50.0, 2e-5 * nd, 0.02 * double(holePx)})) {
          const long long gain4 = filled(4.0) - base;
          fp.bridgePx = gain4 >= gain8 * 4 / 5 ? 4.0 : 8.0;
          r.changed.bridge = true;
          r.notes.push_back(fmt("Bridge Gaps px %.0f: about %lld more pixels of holes open to the "
                                "outside through gaps up to about %.0f px wide fill only when "
                                "bridged.", fp.bridgePx, fp.bridgePx == 4.0 ? gain4 : gain8,
                                fp.bridgePx));
        }
      }
      if (!r.changed.bridge)
        r.notes.push_back("Bridge Gaps px 0 (default): no hole leaks out through a narrow gap. "
                          "Rim px 1 (default).");
      else
        r.notes.push_back("Rim px 1 (default).");
    }
  }
  if (r.fillEnable) {
    FillHolesStats st;
    fillHoles(cur.data(), cur.data(), w, h, r.fill, &st);
    r.measured.fillChanged = st.pixelsChanged;
  }
  // ===== Solid Core =====
  {
    Mask below(n);
    for (size_t i = 0; i < n; ++i) below[i] = cur[i] < r.core.edgeAt;
    constexpr int kRings = 12;
    const std::vector<float> d = distanceWithin(below, w, h, double(kRings), [](int) { return ContourFoot(); });
    std::vector<long long> ringN(kRings + 1, 0), ringGrey(kRings + 1, 0);
    long long deep = 0, inside = 0;
    for (size_t i = 0; i < n; ++i) {
      if (below[i]) continue;
      ++inside;
      const bool g = cur[i] < 0.9f;
      if (d[i] > 8.0f) deep += g;
      if (d[i] <= kRings) {
        const int k = int(std::ceil(d[i]));
        ringN[k]++;
        ringGrey[k] += g;
      }
    }
    r.measured.deepGreyBefore = r.measured.deepGreyAfter = deep;
    auto frac = [&](int k) { return ringN[k] ? double(ringGrey[k]) / double(ringN[k]) : 0.0; };
    double floorFrac = 0;  // grey share deeper inside (rings 9-12)
    for (int k = 9; k <= kRings; ++k) floorFrac += frac(k);
    floorFrac /= double(kRings - 8);
    // The edge's own softness: where the grey share has fallen halfway from the contour's to
    // the deep share. Depth = twice that (t36: 3 -> 6, the setting that worked in Flame).
    int edgeEnd = 0;
    for (int k = 2; k <= kRings && !edgeEnd; ++k)
      if (frac(k) <= 0.5 * (frac(1) + floorFrac)) edgeEnd = k;
    if (deep >= std::max<long long>(50, (long long)(5e-4 * double(inside)))) {
      r.coreEnable = r.changed.coreEnable = true;
      const double depth = edgeEnd ? std::min(24.0, std::max(3.0, 2.0 * edgeEnd)) : r.core.depthPx;
      r.changed.depth = std::fabs(depth - r.core.depthPx) > 1e-9;
      r.core.depthPx = depth;
      SolidCoreStats st;
      std::vector<float> after(n);
      solidCore(cur.data(), after.data(), w, h, r.core, &st);
      r.measured.coreChanged = st.pixelsChanged;
      long long deepAfter = 0;
      for (size_t i = 0; i < n; ++i) deepAfter += !below[i] && d[i] > 8.0f && after[i] < 0.9f;
      r.measured.deepGreyAfter = deepAfter;
      r.notes.push_back(fmt("Solid Core on, Depth px %.0f%s: %lld pixels below 0.9 remain more than "
                            "8 px inside the subject%s (a semi-transparent area joined to the edge); "
                            "the edge's own softness reaches about %d px in, so the outer %.0f px stay "
                            "as they are. Left: %lld. Check see-through parts (glass): they turn solid too.",
                            depth, r.changed.depth ? "" : " (default)", deep,
                            r.fillEnable ? " after Fill Holes" : "", edgeEnd, depth, deepAfter));
    } else {
      unchangedNotes.push_back(fmt("Solid Core: nothing grey deep inside (%lld pixels below 0.9 more "
                                   "than 8 px in); left off.", deep));
    }
  }
  r.notes.insert(r.notes.end(), unchangedNotes.begin(), unchangedNotes.end());
  done();
}

}  // namespace mr
