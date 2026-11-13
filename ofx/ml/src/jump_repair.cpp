#include "jump_repair.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "ops/morphology.h"

namespace mr {
namespace ml {

namespace {

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct Pair {
  int d1, d2;
};
// Shortest span first (jump_repair.h, step 1).
const Pair kPairs[] = {{1, 1}, {1, 2}, {2, 1}, {1, 3}, {2, 2}, {3, 1}};
constexpr int kPairCount = int(sizeof kPairs / sizeof kPairs[0]);

constexpr uint8_t kRemoved = 0x80;  // flag bits: pair index + 1 in the low 3 bits, the direction
constexpr uint8_t kSeen = 0x40;     // and "already in a blob"
constexpr uint8_t kKept = 0x20;     // a kept blob's pixel: growth never enters it
constexpr uint8_t kPairMask = 0x07;

// The reference value at t of pair (b, a): linear in time.
inline float refValue(const Pair& q, float b, float a) {
  return b + float(q.d1) / float(q.d1 + q.d2) * (a - b);
}

struct Entry {
  uint32_t idx;
  float w, target;
  int blob;
};

// The Moved guard's score map for one neighbour nb: for each shift v on a grid of step `step`
// (|v_x|, |v_y| <= R; (2 R / step + 1)^2 cells, row-major from (-R, -R)): sum min(e, [s (nb(p + v) -
// A_t(p + v))]+) / sum e, 0-1: how much of the blob's excess e (s (A_t - ref) per pixel) nb has, shifted
// by v, where frame t has not. At most 512 of the blob's pixels (every k-th). v = 0 always scores 0.
std::vector<float> shiftScores(const float* c, const float* nb, const std::vector<uint32_t>& px,
                               const std::vector<float>& e, float s, int w, int h, int R, int step) {
  const int cells = 2 * (R / step) + 1;
  std::vector<float> out(size_t(cells) * cells, 0.0f);
  const size_t k = (px.size() + 511) / 512;
  double total = 0;
  for (size_t j = 0; j < px.size(); j += k) total += e[j];
  if (!nb || !(total > 0)) return out;
  for (int gy = 0; gy < cells; ++gy)
    for (int gx = 0; gx < cells; ++gx) {
      const int dx = (gx - cells / 2) * step, dy = (gy - cells / 2) * step;
      double sum = 0;
      for (size_t j = 0; j < px.size(); j += k) {
        const int x = int(px[j] % uint32_t(w)) + dx, y = int(px[j] / uint32_t(w)) + dy;
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const size_t i = size_t(y) * w + x;
        const float v = s * (nb[i] - c[i]);
        if (v > 0.0f) sum += std::min(e[j], v);
      }
      out[size_t(gy) * cells + gx] = float(sum / total);
    }
  return out;
}

// Moved: a real object passing through, at about constant velocity: max over v of min(score in t - 1
// at -v, score in t + 1 at +v), each taken as the best within a tolerance of 1 cell + |v| / 4 (some
// acceleration). Returns that score; the scores and shifts of each side at the best v.
double passThrough(const std::vector<float>& sb, const std::vector<float>& sa, int R, int step, JumpBlob* b) {
  const int cells = 2 * (R / step) + 1, half = cells / 2;
  double best = 0;
  for (int gy = 0; gy < cells; ++gy)
    for (int gx = 0; gx < cells; ++gx) {
      const int vx = gx - half, vy = gy - half;
      if (!vx && !vy) continue;
      const int tol = 1 + int(std::sqrt(double(vx * vx + vy * vy)) / 4);
      auto bestNear = [&](const std::vector<float>& m, int cx, int cy, int* bx, int* by) {
        float v = 0.0f;
        *bx = cx;
        *by = cy;
        for (int y = std::max(0, cy - tol); y <= std::min(cells - 1, cy + tol); ++y)
          for (int x = std::max(0, cx - tol); x <= std::min(cells - 1, cx + tol); ++x)
            if (m[size_t(y) * cells + x] > v) {
              v = m[size_t(y) * cells + x];
              *bx = x;
              *by = y;
            }
        return v;
      };
      int ax, ay, bx, by;
      const float before = bestNear(sb, half - vx, half - vy, &bx, &by);
      const float after = bestNear(sa, half + vx, half + vy, &ax, &ay);
      const double v = std::min(before, after);
      if (v > best) {
        best = v;
        b->moveBefore = before;
        b->moveAfter = after;
        b->moveBeforeDx = (bx - half) * step;
        b->moveBeforeDy = (by - half) * step;
        b->moveAfterDx = (ax - half) * step;
        b->moveAfterDy = (ay - half) * step;
      }
    }
  return best;
}

}  // namespace

const char* jumpDecisionName(JumpDecision d) {
  switch (d) {
    case JumpDecision::Repaired: return "repaired";
    case JumpDecision::KeptMoved: return "kept (moved: found shifted in both neighbours)";
    case JumpDecision::KeptTurning: return "kept (turning: an edge at the end of a back-and-forth motion)";
    case JumpDecision::KeptNotStill: return "kept (2-3 frames, but the matte around it moves)";
  }
  return "?";
}

void repairJumps(const float* const* frames, int w, int h, const JumpParams& p, float* out, JumpStats* stp,
                 std::vector<uint8_t>* marks) {
  const auto t0 = std::chrono::steady_clock::now();
  JumpStats local;
  JumpStats& st = stp ? *stp : local;
  st = JumpStats();
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  const float* c = frames[kJumpReach];
  if (marks) marks->assign(n, kJumpNone);
  auto frame = [&](int k) -> const float* {
    return k < -kJumpReach || k > kJumpReach ? nullptr : frames[k + kJumpReach];
  };
  if (!n || !frame(-1) || !frame(1)) {  // not judged: the centre bit for bit
    if (out != c && n) std::memcpy(out, c, n * sizeof(float));
    st.totalMs = msSince(t0);
    return;
  }
  st.judged = true;

  // 1. Flag.
  const int maxSpan = std::min(std::max(p.maxSpan, 1), kJumpMaxSpan);
  int avail[kPairCount];
  int na = 0;
  for (int i = 0; i < kPairCount; ++i)
    if (kPairs[i].d1 + kPairs[i].d2 - 1 <= maxSpan && frame(-kPairs[i].d1) && frame(kPairs[i].d2)) avail[na++] = i;
  st.pairs = na;
  const float thr = float(p.threshold), agree = float(p.agree);
  std::vector<uint8_t> flag(n, 0);
  parallelFor(h, 32, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float A = c[i];
      for (int k = 0; k < na; ++k) {
        const Pair& q = kPairs[avail[k]];
        const float b = frame(-q.d1)[i], a = frame(q.d2)[i];
        if (!(std::fabs(b - a) <= agree)) continue;
        const float hi = b > a ? b : a, lo = b > a ? a : b;
        const bool added = A - hi > thr, removed = lo - A > thr;
        if (!added && !removed) continue;
        // The references confirmed by the frames beyond them (when present): a jump in a
        // reference frame next to a moving edge must not make frame t look like the jump.
        const float *bb = frame(-q.d1 - 1), *aa = frame(q.d2 + 1);
        if ((bb && !(std::fabs(bb[i] - b) <= agree)) || (aa && !(std::fabs(aa[i] - a) <= agree))) continue;
        flag[i] = uint8_t(avail[k] + 1) | (removed ? kRemoved : 0);
        break;
      }
    }
  });
  st.flagMs = msSince(t0);

  // 2. Blobs (8-connected, per direction) and 3. guards.
  const auto t1 = std::chrono::steady_clock::now();
  const double G = p.growPx < 0 ? std::max(1.0, h / 540.0) : p.growPx;
  const double F = p.featherPx < 0 ? std::max(1.0, h / 540.0) : p.featherPx;
  st.minArea = std::max<long long>(1, (long long)std::ceil(std::max(p.minSizePercent, 0.0) / 100.0 * double(n)));
  std::vector<std::vector<uint32_t>> repairedPixels;  // per entry of st.blobs (empty unless repaired)
  std::vector<int> blobPair;                          // per entry of st.blobs: its pair (kPairs index)
  std::vector<uint32_t> px, stack;
  std::vector<float> ex;
  for (size_t seed = 0; seed < n; ++seed) {
    if (!flag[seed] || (flag[seed] & kSeen)) continue;
    const uint8_t dir = flag[seed] & kRemoved;
    px.clear();
    stack.assign(1, uint32_t(seed));
    flag[seed] |= kSeen;
    while (!stack.empty()) {
      const uint32_t i = stack.back();
      stack.pop_back();
      px.push_back(i);
      const int x = int(i % uint32_t(w)), y = int(i / uint32_t(w));
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if ((!dx && !dy) || xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const uint32_t j = uint32_t(yy) * uint32_t(w) + uint32_t(xx);
          const uint8_t f = flag[j];
          if (f && !(f & kSeen) && (f & kRemoved) == dir) {
            flag[j] |= kSeen;
            stack.push_back(j);
          }
        }
    }
    st.flaggedPixels += (long long)px.size();
    if ((long long)px.size() < st.minArea) {
      ++st.smallBlobs;
      continue;
    }
    JumpBlob b;
    b.added = !dir;
    const float s = b.added ? 1.0f : -1.0f;
    b.area = (long long)px.size();
    b.x0 = w;
    b.y0 = h;
    b.x1 = b.y1 = -1;
    int count[kPairCount] = {0};
    for (uint32_t i : px) {
      const int x = int(i % uint32_t(w)), y = int(i / uint32_t(w));
      b.x0 = std::min(b.x0, x);
      b.x1 = std::max(b.x1, x);
      b.y0 = std::min(b.y0, y);
      b.y1 = std::max(b.y1, y);
      const int pi = (flag[i] & kPairMask) - 1;
      ++count[pi];
      const Pair& q = kPairs[pi];
      b.mass += s * (c[i] - refValue(q, frame(-q.d1)[i], frame(q.d2)[i]));
    }
    const int bp = int(std::max_element(count, count + kPairCount) - count);  // the blob's pair
    const Pair& q = kPairs[bp];
    b.before = -q.d1;
    b.after = q.d2;
    if (p.motionGuard) {
      const int bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1, side = std::max(bw, bh);
      // Moved: the blob found again, shifted, in t - 1 (where t vacated it) and in t + 1 (where t
      // has not reached it yet).
      ex.clear();
      for (uint32_t i : px) {
        const Pair& pq = kPairs[(flag[i] & kPairMask) - 1];
        ex.push_back(s * (c[i] - refValue(pq, frame(-pq.d1)[i], frame(pq.d2)[i])));
      }
      const int R = int(std::min(h / 6.0, std::max(4.0 * side, 8.0))), step = std::max(1, R / 32);
      b.moveScore = passThrough(shiftScores(c, frame(-1), px, ex, s, w, h, R, step),
                                shiftScores(c, frame(1), px, ex, s, w, h, R, step), R, step, &b);
      // Turning: in a window about the blob's thickness around it, the matte grows towards t before
      // the references and shrinks after them.
      const int g = int(std::ceil(std::min(bw, bh) + G + F));
      const int zx0 = std::max(0, b.x0 - g), zx1 = std::min(w - 1, b.x1 + g);
      const int zy0 = std::max(0, b.y0 - g), zy1 = std::min(h - 1, b.y1 + g);
      const float *fb = frame(-q.d1), *fa = frame(q.d2), *fbb = frame(-q.d1 - 1), *faa = frame(q.d2 + 1);
      b.turnChecked = fbb && faa;
      if (b.turnChecked) {
        double tb = 0, ta = 0;
        long long moving = 0;
        for (int y = zy0; y <= zy1; ++y)
          for (int x = zx0; x <= zx1; ++x) {
            const size_t i = size_t(y) * w + x;
            const float db = fb[i] - fbb[i], da = fa[i] - faa[i];
            tb += db;
            ta += da;
            moving += !(std::fabs(db) <= agree) || !(std::fabs(da) <= agree);
          }
        b.turnBefore = s * tb;
        b.turnAfter = s * ta;
        b.movingPixels = moving;
      }
      if (b.moveScore >= 0.5)
        b.decision = JumpDecision::KeptMoved;
      else if (q.d1 + q.d2 > 2 && !(b.turnChecked && double(b.movingPixels) <= 0.1 * double(b.area)))
        b.decision = JumpDecision::KeptNotStill;  // a span of 2-3 frames: only with still surroundings
      else if (b.turnChecked && std::min(b.turnBefore, b.turnAfter) > 0.5 * b.mass)
        b.decision = JumpDecision::KeptTurning;
    }
    st.blobs.push_back(b);
    blobPair.push_back(bp);
    repairedPixels.emplace_back();
    if (b.decision == JumpDecision::Repaired) {
      ++st.repaired;
      repairedPixels.back().swap(px);
    } else {
      for (uint32_t i : px) flag[i] |= kKept;
      b.decision == JumpDecision::KeptMoved     ? ++st.keptMoved
      : b.decision == JumpDecision::KeptTurning ? ++st.keptTurning
                                                : ++st.keptNotStill;
      if (marks)
        for (uint32_t i : px) (*marks)[i] = kJumpKept;
    }
  }
  st.blobMs = msSince(t1);

  // 4. Repair: every zone's entries first, then one write (out may be the centre).
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<Entry> entries;
  const double reach = G + F;
  const int R = int(std::ceil(reach));
  // Hysteresis growth (the blob's soft rim): at most Rg steps (8-connected) from the flagged pixels.
  const int Rg = int(std::lround(p.growMaxPx < 0 ? std::max(2.0, 16.0 * h / 1080.0) : p.growMaxPx));
  const float low = float(p.growThreshold);
  std::vector<float> d2;
  std::vector<uint8_t> in;
  std::vector<uint32_t> grown, queue;
  std::vector<uint16_t> depth;
  for (size_t bi = 0; bi < st.blobs.size(); ++bi) {
    if (repairedPixels[bi].empty()) continue;
    JumpBlob& b = st.blobs[bi];
    const float s = b.added ? 1.0f : -1.0f;
    const uint8_t dir = b.added ? 0 : kRemoved;
    const Pair& bq = kPairs[blobPair[bi]];
    const int cx0 = std::max(0, b.x0 - R - Rg), cx1 = std::min(w - 1, b.x1 + R + Rg);
    const int cy0 = std::max(0, b.y0 - R - Rg), cy1 = std::min(h - 1, b.y1 + R + Rg);
    const int cw = cx1 - cx0 + 1, ch = cy1 - cy0 + 1;
    // A pixel's references: its own pair when it is flagged in the blob's direction, else the blob's.
    auto pairOf = [&](uint32_t i) -> const Pair& {
      const uint8_t f = flag[i];
      return (f & kPairMask) && (f & kRemoved) == dir ? kPairs[(f & kPairMask) - 1] : bq;
    };
    in.assign(size_t(cw) * ch, 0);
    d2.assign(size_t(cw) * ch, 1e30f);
    depth.assign(size_t(cw) * ch, 0);
    grown = repairedPixels[bi];
    for (uint32_t i : grown) in[size_t(int(i / uint32_t(w)) - cy0) * cw + (int(i % uint32_t(w)) - cx0)] = 1;
    // Grow to connected pixels beyond the references' range in the blob's direction by more than
    // growThreshold, where the references agree (a moving edge stops it), never into a kept blob.
    queue = grown;
    for (size_t qi = 0; qi < queue.size(); ++qi) {
      const uint32_t i = queue[qi];
      const int x = int(i % uint32_t(w)), y = int(i / uint32_t(w));
      const int dep = depth[size_t(y - cy0) * cw + (x - cx0)];
      if (dep >= Rg) continue;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx < cx0 || yy < cy0 || xx > cx1 || yy > cy1) continue;
          const size_t ci = size_t(yy - cy0) * cw + (xx - cx0);
          if (in[ci]) continue;
          const uint32_t j = uint32_t(yy) * uint32_t(w) + uint32_t(xx);
          if (flag[j] & kKept) continue;
          const Pair& q = pairOf(j);
          const float fb = frame(-q.d1)[j], fa = frame(q.d2)[j];
          if (!(std::fabs(fb - fa) <= agree)) continue;
          const float ext = b.added ? std::max(fb, fa) : std::min(fb, fa);
          if (!(s * (c[j] - ext) > low)) continue;
          in[ci] = 1;
          depth[ci] = uint16_t(dep + 1);
          queue.push_back(j);
          grown.push_back(j);
        }
    }
    b.grown = (long long)(grown.size() - repairedPixels[bi].size());
    st.grownPixels += b.grown;
    const std::vector<uint32_t>& bpx = grown;
    for (uint32_t i : bpx) d2[size_t(int(i / uint32_t(w)) - cy0) * cw + (int(i % uint32_t(w)) - cx0)] = 0.0f;
    // Distance to the blob: stamp a disc of radius R around each boundary pixel of the blob.
    for (uint32_t i : bpx) {
      const int x = int(i % uint32_t(w)) - cx0, y = int(i / uint32_t(w)) - cy0;
      bool boundary = false;
      for (int dy = -1; dy <= 1 && !boundary; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= cw || yy >= ch || !in[size_t(yy) * cw + xx]) {
            boundary = true;
            break;
          }
        }
      if (!boundary) continue;
      for (int yy = std::max(0, y - R); yy <= std::min(ch - 1, y + R); ++yy)
        for (int xx = std::max(0, x - R); xx <= std::min(cw - 1, x + R); ++xx) {
          float& v = d2[size_t(yy) * cw + xx];
          const float e = float((xx - x) * (xx - x) + (yy - y) * (yy - y));
          if (e < v) v = e;
        }
    }
    for (int y = 0; y < ch; ++y)
      for (int x = 0; x < cw; ++x) {
        const size_t ci = size_t(y) * cw + x;
        const double d = std::sqrt(double(d2[ci]));
        if (d >= reach) continue;
        const float wt = d <= G ? 1.0f : float((reach - d) / F);
        if (wt <= 0.0f) continue;
        const uint32_t i = uint32_t(y + cy0) * uint32_t(w) + uint32_t(x + cx0);
        const Pair& q = pairOf(i);
        const float fb = frame(-q.d1)[i], fa = frame(q.d2)[i], A = c[i];
        const float ext = b.added ? std::max(fb, fa) : std::min(fb, fa);
        if (!(s * (A - ext) > 0.0f)) continue;  // only beyond the references, in the blob's direction
        entries.push_back({i, wt, A + wt * (refValue(q, fb, fa) - A), int(bi)});
      }
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return a.idx != b.idx ? a.idx < b.idx : a.w > b.w;  // the larger weight first
  });
  if (out != c) std::memcpy(out, c, n * sizeof(float));
  for (size_t k = 0; k < entries.size(); ++k) {
    if (k && entries[k].idx == entries[k - 1].idx) continue;
    const Entry& e = entries[k];
    const float A = c[e.idx];
    if (e.target == A) continue;
    out[e.idx] = e.target;
    ++st.pixelsChanged;
    ++st.blobs[size_t(e.blob)].changed;
    st.maxChange = std::max(st.maxChange, double(std::fabs(e.target - A)));
    if (marks) (*marks)[e.idx] = kJumpRepaired;
  }
  st.repairMs = msSince(t2);
  st.totalMs = msSince(t0);
}

}  // namespace ml
}  // namespace mr
