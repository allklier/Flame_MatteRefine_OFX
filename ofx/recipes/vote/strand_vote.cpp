// Strand presence vote: see strand_vote.h.
#include "strand_vote.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "flow.h"
#include "temporal.h"

namespace mr {
namespace vote {
namespace {

using tracer::Point;
using tracer::Piece;
using tracer::TracerResult;

float bilinear(const float* f, int w, int h, double x, double y) {
  return ml::sampleField(f, w, h, float(x), float(y));
}

bool inRegion(const float* r, size_t i) { return r[i] > 0.0f; }  // NaN = outside

struct Field {
  const Motion* m;
  int w, h;
  void at(double x, double y, double* dx, double* dy) const {
    *dx = bilinear(m->dx, w, h, x, y);
    *dy = bilinear(m->dy, w, h, x, y);
  }
  double conf(double x, double y) const { return m->conf ? std::min(std::max(double(bilinear(m->conf, w, h, x, y)), 0.0), 1.0) : 1.0; }
};

// A neighbour point q -> the centre point p with p + F(p) = q (fixed-point iteration on the smooth flow).
Point toCentre(const Field& f, Point q) {
  double dx, dy;
  f.at(q.x, q.y, &dx, &dy);
  Point p{q.x - dx, q.y - dy};
  for (int it = 0; it < 6; ++it) {
    f.at(p.x, p.y, &dx, &dy);
    p = Point{q.x - dx, q.y - dy};
  }
  return p;
}

double segDist2(const Point& p, const Point& a, const Point& b) {
  const double vx = b.x - a.x, vy = b.y - a.y, px = p.x - a.x, py = p.y - a.y, l2 = vx * vx + vy * vy;
  double t = l2 > 0 ? (px * vx + py * vy) / l2 : 0.0;
  t = std::min(std::max(t, 0.0), 1.0);
  const double ex = px - t * vx, ey = py - t * vy;
  return ex * ex + ey * ey;
}

double distTo(const std::vector<Point>& line, const Point& p) {
  if (line.empty()) return 1e30;
  if (line.size() == 1) return std::hypot(p.x - line[0].x, p.y - line[0].y);
  double best = 1e300;
  for (size_t i = 0; i + 1 < line.size(); ++i) best = std::min(best, segDist2(p, line[i], line[i + 1]));
  return std::sqrt(best);
}

// Fraction of `a`'s points within tol of any of `set`.
double cover(const std::vector<Point>& a, const std::vector<const std::vector<Point>*>& set, double tol) {
  if (a.empty() || set.empty()) return 0.0;
  int hit = 0;
  for (const Point& p : a) {
    for (const std::vector<Point>* s : set)
      if (distTo(*s, p) <= tol) {
        ++hit;
        break;
      }
  }
  return double(hit) / a.size();
}

double barOf(const VoteParams& p, const Piece& pc) {
  return (pc.anchored || pc.selfReferenced) ? p.minConf : p.minConfUnanchored;
}

// Paints max(out, 1) for pixels within radius r of the polyline (a capsule chain).
void capsule(std::vector<uint8_t>& mask, int w, int h, const std::vector<Point>& line, double r) {
  if (line.empty()) return;
  double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
  for (const Point& q : line) x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
  const int ix0 = std::max(int(std::floor(x0 - r)), 0), iy0 = std::max(int(std::floor(y0 - r)), 0);
  const int ix1 = std::min(int(std::ceil(x1 + r)), w - 1), iy1 = std::min(int(std::ceil(y1 + r)), h - 1);
  for (int y = iy0; y <= iy1; ++y)
    for (int x = ix0; x <= ix1; ++x)
      if (distTo(line, Point{double(x), double(y)}) <= r) mask[size_t(y) * w + x] = 1;
}

struct Nb {
  const Neighbour* n;
  Field f;
  bool judged = false;
  double meanConf = 1.0;
  std::vector<std::vector<Point>> moved;  // per piece of n->result: moved to t, gated by the widened region
};

}  // namespace

Motion DenseMotion::view() const {
  Motion m;
  m.offset = offset;
  m.valid = valid && !dx.empty();
  m.dx = dx.empty() ? nullptr : dx.data();
  m.dy = dy.empty() ? nullptr : dy.data();
  m.conf = conf.empty() ? nullptr : conf.data();
  return m;
}

DenseMotion motionFromFlows(const ml::TemporalFlows& flows, int offset, int x0, int y0, int w, int h) {
  DenseMotion d;
  d.offset = offset;
  size_t k = 0;
  while (k < flows.offsets.size() && flows.offsets[k] != offset) ++k;
  if (k >= flows.offsets.size() || k >= flows.flows.size() || w <= 0 || h <= 0) return d;
  const ml::NeighbourFlow& f = flows.flows[k];
  const size_t n = size_t(w) * h;
  d.dx.resize(n);
  d.dy.resize(n);
  d.conf.resize(n);
  const int s = f.scale;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double X = x0 + x, Y = y0 + y;
      double dx, dy;
      ml::flowAt(f, X, Y, &dx, &dy);  // temporal.h:75
      d.dx[i] = float(dx);
      d.dy[i] = float(dy);
      const float gx = float((X - (s - 1) * 0.5) / s), gy = float((Y - (s - 1) * 0.5) / s);  // as flowAt
      d.conf[i] = f.conf.empty() ? 1.0f : ml::sampleField(f.conf.data(), f.fwd.w, f.fwd.h, gx, gy);
    }
  d.valid = true;
  return d;
}

void setBars(VoteParams& v, const tracer::TracerParams& p) {
  v.minConf = p.keepConf;
  v.minConfUnanchored = p.minConfUnanchored;
}

tracer::TracerParams candidateTracerParams(const tracer::TracerParams& p, double margin) {
  tracer::TracerParams c = p;
  c.keepConf = std::max(p.keepConf - margin, 0.0);
  c.minConfUnanchored = std::max(p.minConfUnanchored - margin, 0.0);
  return c;
}

std::vector<float> widenRegion(const float* region, int w, int h, const Motion* motions, int count, double marginPx) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  std::vector<float> out(n, 0.0f);
  if (!region || !n) return out;
  double maxD = 0;
  for (int k = 0; k < count; ++k) {
    const Motion& m = motions[k];
    if (!m.valid || !m.dx || !m.dy) continue;
    for (size_t i = 0; i < n; ++i)
      if (inRegion(region, i)) maxD = std::max(maxD, std::hypot(double(m.dx[i]), double(m.dy[i])));
  }
  const int r = int(std::ceil(maxD + std::max(marginPx, 0.0)));
  // separable max filter (a square of side 2r + 1: covers the disc of radius r)
  std::vector<float> tmp(n, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float v = 0.0f;
      for (int j = std::max(x - r, 0); j <= std::min(x + r, w - 1); ++j) {
        const float q = region[size_t(y) * w + j];
        if (q > v) v = q;
      }
      tmp[size_t(y) * w + x] = v;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float v = 0.0f;
      for (int j = std::max(y - r, 0); j <= std::min(y + r, h - 1); ++j) v = std::max(v, tmp[size_t(j) * w + x]);
      out[size_t(y) * w + x] = v;
    }
  return out;
}

std::vector<float> movedRegion(const float* region, int w, int h, const Motion& m, double marginPx) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (!region || !n || !m.valid || !m.dx || !m.dy) return widenRegion(region, w, h, nullptr, 0, marginPx);
  const Field f{&m, w, h};
  std::vector<float> moved(n, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const Point p = toCentre(f, Point{double(x), double(y)});
      if (p.x < 0 || p.y < 0 || p.x > w - 1 || p.y > h - 1) continue;
      const float v = bilinear(region, w, h, p.x, p.y);
      moved[size_t(y) * w + x] = v > 0.0f ? v : 0.0f;
    }
  return widenRegion(moved.data(), w, h, nullptr, 0, marginPx);
}

const char* fateName(Fate f) {
  switch (f) {
    case Fate::KeptHigh: return "kept (neighbours)";
    case Fate::KeptLow: return "kept";
    case Fate::KeptConfident: return "kept (confident)";
    case Fate::KeptFallback: return "kept (own result)";
    case Fate::Dropped: return "dropped";
    case Fate::Reinstated: return "reinstated";
  }
  return "?";
}

VoteResult votePresence(const VoteInput& in, const VoteParams& p) {
  VoteResult res;
  const int w = in.width, h = in.height;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  res.alpha.assign(n, 0.0f);
  res.confidence.assign(n, 0.0f);
  if (!n || !in.region || !in.centre) return res;
  const TracerResult& C = *in.centre;
  const bool centreMaps = C.alpha.size() == n && C.confidence.size() == n;

  // neighbours: judged?, pieces moved to t
  std::vector<Motion> motions;
  for (int k = 0; k < in.count; ++k) motions.push_back(in.neighbours[k].motion);
  std::vector<float> ownWide;
  const float* wide = in.widenedRegion;
  if (!wide) {
    ownWide = widenRegion(in.region, w, h, motions.data(), int(motions.size()));
    wide = ownWide.data();
  }
  long long support = 0;
  if (in.base)
    for (size_t i = 0; i < n; ++i) support += inRegion(wide, i) && in.base[i] >= 0.5f ? 1 : 0;
  const bool useSupport = support >= p.minSupportPx;
  std::vector<Nb> nbs;
  std::string notes;
  for (int k = 0; k < in.count; ++k) {
    const Neighbour& ng = in.neighbours[k];
    Nb b{&ng, Field{&ng.motion, w, h}, false, 1.0, {}};
    char buf[96];
    if (!ng.result || !ng.motion.valid || !ng.motion.dx || !ng.motion.dy) {
      std::snprintf(buf, sizeof buf, "%s%+d: no flow", notes.empty() ? "" : "; ", ng.motion.offset);
      notes += buf;
      nbs.push_back(b);
      continue;
    }
    double sum = 0;
    long long cnt = 0;
    if (ng.motion.conf) {
      if (useSupport) {
        for (size_t i = 0; i < n; ++i)
          if (inRegion(wide, i) && in.base[i] >= 0.5f) sum += ng.motion.conf[i], ++cnt;
      } else {
        for (size_t i = 0; i < n; ++i)
          if (inRegion(in.region, i)) sum += ng.motion.conf[i], ++cnt;
      }
    }
    b.meanConf = cnt ? sum / cnt : 1.0;
    b.judged = !(p.rFlowCheck && b.meanConf < p.minFlowConf);
    if (!b.judged) {
      std::snprintf(buf, sizeof buf, "%s%+d: flow unreliable (%.2f)", notes.empty() ? "" : "; ", ng.motion.offset,
                    b.meanConf);
      notes += buf;
    }
    for (const Piece& pc : ng.result->pieces) {
      std::vector<Point> mv;
      for (const Point& q : pc.polyline) {
        const Point c = toCentre(b.f, q);
        const int ix = int(std::lround(c.x)), iy = int(std::lround(c.y));
        const bool inside = ix >= 0 && iy >= 0 && ix < w && iy < h && inRegion(wide, size_t(iy) * w + ix);
        if (!p.rRegionGate || inside) mv.push_back(c);
      }
      b.moved.push_back(mv);
    }
    nbs.push_back(b);
  }
  for (const Nb& b : nbs) res.judgedNeighbours += b.judged ? 1 : 0;
  res.fallback = res.judgedNeighbours == 0;
  if (res.fallback) notes += std::string(notes.empty() ? "" : "; ") + "no reliable neighbour: own result";
  res.note = notes;

  auto tolFor = [&](const Nb& b, const std::vector<Point>& line, double width) {
    (void)line;
    const double fc = b.meanConf;  // the neighbour's flow reliability over the support (see the flow check)
    return p.tolPx + (p.rWidthTol ? p.tolWidth * width : 0.0) + (p.rFlowTol ? p.tolFlow * (1.0 - fc) : 0.0);
  };

  // 1. the centre's pieces
  std::vector<uint8_t> keepMask(n, 0);
  std::vector<bool> keptPiece(C.pieces.size(), false);
  for (size_t i = 0; i < C.pieces.size(); ++i) {
    const Piece& pc = C.pieces[i];
    PieceVote v;
    v.index = int(i);
    v.confidence = pc.confidence;
    v.bar = barOf(p, pc);
    v.width = pc.width;
    for (const Nb& b : nbs) {
      if (!b.judged) continue;
      ++v.judged;
      std::vector<const std::vector<Point>*> set;
      for (const std::vector<Point>& m : b.moved)
        if (!m.empty()) set.push_back(&m);
      if (cover(pc.polyline, set, tolFor(b, pc.polyline, pc.width)) >= p.matchCover) ++v.matched;
    }
    v.score = v.judged ? double(v.matched) / v.judged : 0.0;
    const bool own = pc.confidence >= v.bar;
    if (res.fallback)
      v.fate = own ? Fate::KeptFallback : Fate::Dropped;
    else if (p.rHigh && v.score >= p.scoreHigh)
      v.fate = Fate::KeptHigh;
    else if (p.rLow && v.score >= p.scoreLow && own)
      v.fate = Fate::KeptLow;
    else if (p.rConfHigh && pc.confidence >= p.confHigh)
      v.fate = Fate::KeptConfident;
    else
      v.fate = Fate::Dropped;
    keptPiece[i] = v.fate != Fate::Dropped;
    if (keptPiece[i]) {
      ++res.kept;
      capsule(keepMask, w, h, pc.polyline, pc.width * 0.5 + p.capsulePad);
    } else {
      ++res.dropped;
    }
    res.pieces.push_back(v);
  }
  for (const tracer::Join& j : C.joins) {
    const bool a = j.pieceA >= 0 && size_t(j.pieceA) < keptPiece.size() && keptPiece[j.pieceA];
    const bool b = j.pieceB >= 0 && size_t(j.pieceB) < keptPiece.size() && keptPiece[j.pieceB];
    if (a && b) capsule(keepMask, w, h, j.polyline, j.width * 0.5 + p.capsulePad);
  }
  if (centreMaps)
    for (size_t i = 0; i < n; ++i)
      if (keepMask[i]) {
        res.alpha[i] = C.alpha[i];
        res.confidence[i] = C.confidence[i];
      }

  // 2. reinstatement: absent at t, present at t - 1 and t + 1 with agreeing geometry
  const Nb* prev = nullptr;
  const Nb* next = nullptr;
  for (const Nb& b : nbs) {
    if (!b.judged) continue;
    if (b.n->motion.offset == -1) prev = &b;
    if (b.n->motion.offset == 1) next = &b;
  }
  if (!res.fallback && p.rReinstate && prev && next) {
    std::vector<const std::vector<Point>*> centreSet;
    for (const Piece& pc : C.pieces) centreSet.push_back(&pc.polyline);
    const std::vector<Piece>& PA = prev->n->result->pieces;
    const std::vector<Piece>& PB = next->n->result->pieces;
    // candidate pairs (A in t - 1, B in t + 1), both at or above their bars, neither covered by a centre piece,
    // agreeing in geometry; assigned greedily by the mean distance of the shorter to the longer (closest first),
    // so the two strands of a pair are not crossed
    auto eligible = [&](const Nb& b, const Piece& pc, const std::vector<Point>& mv) {
      return mv.size() >= 2 && pc.confidence >= barOf(p, pc) &&
             cover(mv, centreSet, tolFor(b, mv, pc.width)) < p.matchCover;
    };
    struct Pair {
      double dist;
      size_t a, b;
    };
    std::vector<Pair> pairs;
    for (size_t a = 0; a < PA.size(); ++a) {
      const std::vector<Point>& A = prev->moved[a];
      if (!eligible(*prev, PA[a], A)) continue;
      for (size_t bi = 0; bi < PB.size(); ++bi) {
        const std::vector<Point>& B = next->moved[bi];
        if (!eligible(*next, PB[bi], B)) continue;
        const double tol = std::max(tolFor(*prev, A, PA[a].width), tolFor(*next, B, PB[bi].width));
        const std::vector<const std::vector<Point>*> sa{&A}, sb{&B};
        if (std::max(cover(A, sb, tol), cover(B, sa, tol)) < p.matchCover) continue;
        const std::vector<Point>& shorter = A.size() <= B.size() ? A : B;
        const std::vector<Point>& longer = A.size() <= B.size() ? B : A;
        double d = 0;
        for (const Point& q : shorter) d += distTo(longer, q);
        pairs.push_back(Pair{d / shorter.size(), a, bi});
      }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) { return x.dist < y.dist; });
    std::vector<bool> usedA(PA.size(), false), usedB(PB.size(), false);
    for (const Pair& pr : pairs) {
      if (usedA[pr.a] || usedB[pr.b]) continue;
      usedA[pr.a] = usedB[pr.b] = true;
      const size_t a = pr.a;
      const int best = int(pr.b);
      const std::vector<Point>& A = prev->moved[a];
      const Piece& pb = PB[best];
      PieceVote v;
      v.fate = Fate::Reinstated;
      v.score = 0;
      v.confidence = p.reinstateFactor * std::min(PA[a].confidence, pb.confidence);
      v.bar = std::min(barOf(p, PA[a]), barOf(p, pb));
      v.width = 0.5 * (PA[a].width + pb.width);
      v.polyline = PA[a].confidence >= pb.confidence ? A : next->moved[best];
      std::vector<uint8_t> m(n, 0);
      capsule(m, w, h, v.polyline, v.width * 0.5 + p.capsulePad);
      for (size_t i = 0; i < n; ++i) keepMask[i] |= m[i];  // the blend stays on the strands present at t
      const TracerResult& RA = *prev->n->result;
      const TracerResult& RB = *next->n->result;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if (!m[i] || !inRegion(in.region, i)) continue;
          double s = 0;
          int c = 0;
          const Nb* both[2] = {prev, next};
          const TracerResult* rs[2] = {&RA, &RB};
          for (int q = 0; q < 2; ++q) {
            if (rs[q]->alpha.size() != n) continue;
            double dx, dy;
            both[q]->f.at(x, y, &dx, &dy);
            const double sx = x + dx, sy = y + dy;
            if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) continue;
            s += bilinear(rs[q]->alpha.data(), w, h, sx, sy);
            ++c;
          }
          if (!c) continue;
          const float a = float(s / c * std::min(double(in.region[i]), 1.0));
          if (a > res.alpha[i]) res.alpha[i] = a;
          if (a > 0.0f) res.confidence[i] = std::max(res.confidence[i], float(v.confidence));
        }
      ++res.reinstated;
      res.pieces.push_back(v);
    }
  }
  // 3. temporal blend (blendNear / blendFar)
  if (!res.fallback && (p.blendNear > 0.0 || p.blendFar > 0.0)) {
    std::vector<double> sumA(n, 0.0), sumC(n, 0.0), sumP(n, 0.0);
    double wsum = 1.0;
    for (size_t i = 0; i < n; ++i) {
      sumA[i] = res.alpha[i], sumC[i] = res.confidence[i];
      sumP[i] = res.alpha[i] > p.presentAlpha ? 1.0 : 0.0;
    }
    for (const Nb& b : nbs) {
      if (!b.judged || b.n->result->alpha.size() != n) continue;
      const int off = std::abs(b.n->motion.offset);
      const double wk = off == 1 ? p.blendNear : off == 2 ? p.blendFar : 0.0;
      if (wk <= 0.0) continue;
      const TracerResult& R = *b.n->result;
      std::vector<uint8_t> m(n, 0);  // the neighbour's pieces at their real bars, with their joins
      std::vector<bool> ok(R.pieces.size(), false);
      for (size_t i = 0; i < R.pieces.size(); ++i)
        if (R.pieces[i].confidence >= barOf(p, R.pieces[i])) {
          ok[i] = true;
          capsule(m, w, h, R.pieces[i].polyline, R.pieces[i].width * 0.5 + p.capsulePad);
        }
      for (const tracer::Join& j : R.joins)
        if (j.pieceA >= 0 && size_t(j.pieceA) < ok.size() && ok[j.pieceA] && j.pieceB >= 0 &&
            size_t(j.pieceB) < ok.size() && ok[j.pieceB])
          capsule(m, w, h, j.polyline, j.width * 0.5 + p.capsulePad);
      std::vector<float> ma(n);
      for (size_t i = 0; i < n; ++i) ma[i] = m[i] ? R.alpha[i] : 0.0f;
      wsum += wk;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if (!inRegion(in.region, i)) continue;
          double dx, dy;
          b.f.at(x, y, &dx, &dy);
          const double sx = x + dx, sy = y + dy;
          if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) continue;
          const double a = bilinear(ma.data(), w, h, sx, sy);
          if (a <= 0.0) continue;
          sumA[i] += wk * a;
          if (a > p.presentAlpha) sumP[i] += wk;
          const int ix = int(std::lround(sx)), iy = int(std::lround(sy));
          sumC[i] = std::max(sumC[i], double(R.confidence[size_t(iy) * w + ix]));
        }
    }
    const bool presence = p.presenceHi > p.presenceLo;
    for (size_t i = 0; i < n; ++i) {
      if (!keepMask[i]) {  // away from the strands at t: only where most neighbours agree (no single-frame ghosts)
        const double wn = wsum - 1.0;
        double v = wn > 0.0 && p.fillHi > p.fillLo ? (sumP[i] / wn - p.fillLo) / (p.fillHi - p.fillLo) : 0.0;
        v = std::min(std::max(v, 0.0), 1.0);
        res.alpha[i] = sumP[i] > 0.0 ? float(std::min(sumA[i] / sumP[i], 1.0) * v * v * (3.0 - 2.0 * v)) : 0.0f;
      } else if (presence) {  // the mean over the frames that have the strand here x a smoothstep of the presence share
        double v = (sumP[i] / wsum - p.presenceLo) / (p.presenceHi - p.presenceLo);
        v = std::min(std::max(v, 0.0), 1.0);
        res.alpha[i] = sumP[i] > 0.0 ? float(std::min(sumA[i] / sumP[i], 1.0) * v * v * (3.0 - 2.0 * v)) : 0.0f;
      } else {
        res.alpha[i] = float(sumA[i] / wsum);
      }
      res.confidence[i] = res.alpha[i] > 0.0f ? float(sumC[i]) : 0.0f;
    }
  }
  // 4. exactly 0 outside the region
  for (size_t i = 0; i < n; ++i)
    if (!inRegion(in.region, i)) res.alpha[i] = res.confidence[i] = 0.0f;
  return res;
}

}  // namespace vote
}  // namespace mr
