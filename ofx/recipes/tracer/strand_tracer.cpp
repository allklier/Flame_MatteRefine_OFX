// Strand tracer: see strand_tracer.h and docs/strand_tracing.md section 3. Ported line by line from the Python
// prototype (tracer.py); comments name the prototype's step. Image helpers reproduce the scipy.ndimage calls the
// prototype makes (gaussian_filter truncate 4 mode reflect, grey_closing, maximum_filter, binary dilation /
// erosion with the cross structure, distance_transform_edt with indices, map_coordinates order 1 mode nearest).
#include "strand_tracer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace mr {
namespace tracer {
namespace {

const double kPi = 3.14159265358979323846;
inline double deg(double r) { return r * 180.0 / kPi; }
inline double rad(double d) { return d * kPi / 180.0; }
// Python's round() on floats: half to even (the default FE_TONEAREST mode).
inline int rnd(double v) { return (int)std::nearbyint(v); }

typedef std::vector<double> Img;
typedef std::vector<uint8_t> Mask;

// ---- scipy.ndimage equivalents -------------------------------------------------------------------------
int reflectIndex(int i, int n) {  // mode 'reflect': d c b a | a b c d | d c b a
  if (n == 1) return 0;
  const int period = 2 * n;
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - 1 - i;
}

// scipy _gaussian_kernel1d (truncate 4.0), already reversed for correlate1d (= a convolution).
std::vector<double> gaussKernel(double sigma, int order) {
  const int r = (int)(4.0 * sigma + 0.5);
  const double s2 = sigma * sigma;
  std::vector<double> k(2 * r + 1);
  double sum = 0.0;
  for (int x = -r; x <= r; ++x) sum += (k[x + r] = std::exp(-0.5 / s2 * x * x));
  for (double& v : k) v /= sum;
  for (int x = -r; x <= r; ++x) {
    if (order == 1) k[x + r] *= -x / s2;
    if (order == 2) k[x + r] *= (x * x / (s2 * s2) - 1.0 / s2);
  }
  std::reverse(k.begin(), k.end());
  return k;
}

// One axis of gaussian_filter: axis 0 = rows (y), 1 = columns (x).
Img filter1d(const Img& in, int w, int h, int axis, const std::vector<double>& k) {
  const int r = (int)k.size() / 2;
  Img out(in.size());
  if (axis == 1) {
    std::vector<int> idx(w + 2 * r);
    for (int i = 0; i < w + 2 * r; ++i) idx[i] = reflectIndex(i - r, w);
    for (int y = 0; y < h; ++y) {
      const double* row = &in[(size_t)y * w];
      for (int x = 0; x < w; ++x) {
        double s = 0.0;
        for (int j = 0; j <= 2 * r; ++j) s += k[j] * row[idx[x + j]];
        out[(size_t)y * w + x] = s;
      }
    }
  } else {
    std::vector<int> idx(h + 2 * r);
    for (int i = 0; i < h + 2 * r; ++i) idx[i] = reflectIndex(i - r, h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double s = 0.0;
        for (int j = 0; j <= 2 * r; ++j) s += k[j] * in[(size_t)idx[y + j] * w + x];
        out[(size_t)y * w + x] = s;
      }
  }
  return out;
}

Img gauss(const Img& in, int w, int h, double sigma, int orderY = 0, int orderX = 0) {
  return filter1d(filter1d(in, w, h, 0, gaussKernel(sigma, orderY)), w, h, 1, gaussKernel(sigma, orderX));
}

// Flat min / max filter of odd `size`, mode reflect, separable.
Img rankFilter(const Img& in, int w, int h, int size, bool isMax) {
  const int r = size / 2;
  Img tmp(in.size()), out(in.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double v = isMax ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
      for (int j = -r; j <= r; ++j) {
        const double q = in[(size_t)y * w + reflectIndex(x + j, w)];
        v = isMax ? std::max(v, q) : std::min(v, q);
      }
      tmp[(size_t)y * w + x] = v;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double v = isMax ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
      for (int j = -r; j <= r; ++j) {
        const double q = tmp[(size_t)reflectIndex(y + j, h) * w + x];
        v = isMax ? std::max(v, q) : std::min(v, q);
      }
      out[(size_t)y * w + x] = v;
    }
  return out;
}

// City-block distance to the nearest set pixel (exact two-pass chamfer); `outsideSet` treats the pixels beyond
// the border as set (for erosion with border_value 0).
std::vector<int> cityDistance(const Mask& m, int w, int h, bool outsideSet) {
  const int inf = 1 << 28;
  std::vector<int> d((size_t)w * h);
  for (size_t i = 0; i < d.size(); ++i) d[i] = m[i] ? 0 : inf;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int& v = d[(size_t)y * w + x];
      if (outsideSet) v = std::min(v, std::min(x + 1, y + 1));
      if (x > 0) v = std::min(v, d[(size_t)y * w + x - 1] + 1);
      if (y > 0) v = std::min(v, d[(size_t)(y - 1) * w + x] + 1);
    }
  for (int y = h - 1; y >= 0; --y)
    for (int x = w - 1; x >= 0; --x) {
      int& v = d[(size_t)y * w + x];
      if (outsideSet) v = std::min(v, std::min(w - x, h - y));
      if (x < w - 1) v = std::min(v, d[(size_t)y * w + x + 1] + 1);
      if (y < h - 1) v = std::min(v, d[(size_t)(y + 1) * w + x] + 1);
    }
  return d;
}

Mask dilateCross(const Mask& m, int w, int h, int it) {  // binary_dilation(iterations=it), border 0
  const std::vector<int> d = cityDistance(m, w, h, false);
  Mask o(m.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = d[i] <= it;
  return o;
}

Mask erodeCross(const Mask& m, int w, int h, int it) {  // binary_erosion(iterations=it), border 0
  Mask inv(m.size());
  for (size_t i = 0; i < m.size(); ++i) inv[i] = !m[i];
  const std::vector<int> d = cityDistance(inv, w, h, true);
  Mask o(m.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = d[i] > it;
  return o;
}

// Exact Euclidean distance to the nearest set pixel of `feat`, with that pixel's index (Felzenszwalb &
// Huttenlocher, separable). Ties may resolve differently from scipy's.
void edt(const Mask& feat, int w, int h, Img& dist, std::vector<int>* nearest) {
  const double inf = 1e20;
  Img colD((size_t)w * h);
  std::vector<int> colY((size_t)w * h, -1);
  for (int x = 0; x < w; ++x) {
    int last = -1;
    for (int y = 0; y < h; ++y) {
      if (feat[(size_t)y * w + x]) last = y;
      colY[(size_t)y * w + x] = last;
    }
    last = -1;
    for (int y = h - 1; y >= 0; --y) {
      if (feat[(size_t)y * w + x]) last = y;
      int& c = colY[(size_t)y * w + x];
      if (last >= 0 && (c < 0 || last - y < y - c)) c = last;
      colD[(size_t)y * w + x] = c < 0 ? inf : (double)(y - c) * (y - c);
    }
  }
  dist.assign((size_t)w * h, 0.0);
  if (nearest) nearest->assign((size_t)w * h, -1);
  std::vector<int> v(w);
  std::vector<double> z(w + 1);
  for (int y = 0; y < h; ++y) {
    const double* f = &colD[(size_t)y * w];
    int k = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    for (int q = 1; q < w; ++q) {
      double s;
      for (;;) {
        const int p = v[k];
        s = ((f[q] + (double)q * q) - (f[p] + (double)p * p)) / (2.0 * q - 2.0 * p);
        if (s <= z[k] && k > 0) {
          --k;
          continue;
        }
        break;
      }
      if (s <= z[k]) {  // k == 0 and the new parabola wins everywhere
        v[0] = q;
        z[1] = inf;
        continue;
      }
      ++k;
      v[k] = q;
      z[k] = s;
      z[k + 1] = inf;
    }
    k = 0;
    for (int q = 0; q < w; ++q) {
      while (z[k + 1] < q) ++k;
      const int p = v[k];
      const double d2 = (double)(q - p) * (q - p) + f[p];
      dist[(size_t)y * w + q] = d2 >= inf ? inf : std::sqrt(d2);
      if (nearest && f[p] < inf) (*nearest)[(size_t)y * w + q] = colY[(size_t)y * w + p] * w + p;
    }
  }
}

// np.gradient (central differences, one-sided first order at the ends).
void gradient2d(const Img& f, int w, int h, Img& gx, Img& gy) {
  gx.assign(f.size(), 0.0);
  gy.assign(f.size(), 0.0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = (size_t)y * w + x;
      if (w > 1) gx[i] = x == 0 ? f[i + 1] - f[i] : x == w - 1 ? f[i] - f[i - 1] : 0.5 * (f[i + 1] - f[i - 1]);
      if (h > 1) gy[i] = y == 0 ? f[i + w] - f[i] : y == h - 1 ? f[i] - f[i - w] : 0.5 * (f[i + w] - f[i - w]);
    }
}

std::vector<Point> gradientPts(const std::vector<Point>& P) {
  const size_t n = P.size();
  std::vector<Point> g(n, Point{0.0, 0.0});
  if (n < 2) return g;
  for (size_t i = 0; i < n; ++i) {
    if (i == 0)
      g[i] = Point{P[1].x - P[0].x, P[1].y - P[0].y};
    else if (i == n - 1)
      g[i] = Point{P[i].x - P[i - 1].x, P[i].y - P[i - 1].y};
    else
      g[i] = Point{0.5 * (P[i + 1].x - P[i - 1].x), 0.5 * (P[i + 1].y - P[i - 1].y)};
  }
  return g;
}

// np.percentile, linear interpolation.
double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * (double)(v.size() - 1);
  const size_t lo = (size_t)std::floor(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - (double)lo) * (v[hi] - v[lo]);
}
inline double median(const std::vector<double>& v) { return percentile(v, 50.0); }

// map_coordinates order 1 mode 'nearest' = clamp the coordinate, then bilinear.
inline double samp(const Img& img, int w, int h, double x, double y) {
  x = std::min(std::max(x, 0.0), (double)(w - 1));
  y = std::min(std::max(y, 0.0), (double)(h - 1));
  const int x0 = std::min((int)x, w - 1), y0 = std::min((int)y, h - 1);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const double fx = x - x0, fy = y - y0;
  const double a = img[(size_t)y0 * w + x0], b = img[(size_t)y0 * w + x1];
  const double c = img[(size_t)y1 * w + x0], d = img[(size_t)y1 * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// ---- the tracer ---------------------------------------------------------------------------------------
struct Prof {
  double x, y, lc, d, w, asym, v1, v2, B;
  int dip2;
};
struct Ref {
  double d, w, v1, v2;
};
struct Hit {
  bool valid = false;
  int ix = 0, iy = 0;
};
struct TraceOut {
  std::vector<Point> pts;
  std::vector<double> hs;
  std::vector<Prof> ds;
  bool anchored = false;
  Hit hit;
};
struct PieceIn {
  std::vector<Point> pts;
  std::vector<double> h;
  std::vector<Prof> prof;
  bool anchored = false;
  double conf = 0.0;
  bool contact = false;
};
struct JoinIn {
  std::vector<Point> pts;
  double J = 0.0, w = 3.0;
  int a = -1, b = -1;
  bool matte = false;
};

Ref refFrom(const Prof& p) {
  return Ref{std::max(p.d, 0.05), std::min(std::max(p.w, 1.0), 7.0), p.v1, p.v2};
}

// Medians of the profiles' (d, w, v1, v2).
HairReference medianRef(const std::vector<const Prof*>& ps, bool self) {
  HairReference r;
  if (ps.empty()) return r;
  std::vector<double> d, w, a, b;
  for (const Prof* p : ps) {
    d.push_back(p->d);
    w.push_back(p->w);
    a.push_back(p->v1);
    b.push_back(p->v2);
  }
  r.valid = true;
  r.depth = median(d);
  r.width = median(w);
  r.chroma1 = median(a);
  r.chroma2 = median(b);
  r.selfReferenced = self;
  r.samples = (int)ps.size();
  return r;
}
Ref toRef(const HairReference& r) { return Ref{r.depth, r.width, r.chroma1, r.chroma2}; }

// One region shape: the crop and everything the prototype's Ctx holds.
struct Shape {
  const TracerParams* c = nullptr;
  int ox = 0, oy = 0, W = 0, H = 0;  // crop origin and size in the input buffer
  Img l, lin, c1, c2, alpha, region, bth, lm, ux, uy, distOut;
  Mask search, solid;
  double budget = 0.0;
  bool hasHair = false;
  Ref hair{0, 0, 0, 0};
  bool selfref = false;
  std::vector<PieceIn> pieces;
  std::vector<JoinIn> joins;
  Mask used, usedAnc;
  HairReference contactRef, ownRef;

  size_t at(int x, int y) const { return (size_t)y * W + x; }

  bool profile(double x, double y, double nx, double ny, Prof& out) const {
    const int n = std::min((int)std::lround(4.0 * c->profHalf) + 1, 64);  // 29 samples, t = -7..7 step 0.5
    double t[64], v[64];
    for (int i = 0; i < n; ++i) {
      t[i] = -c->profHalf + 0.5 * i;
      v[i] = samp(l, W, H, x + t[i] * nx, y + t[i] * ny);
    }
    const int mid = n / 2, k = rnd(c->refine / 0.5);
    int i0 = mid - k;
    for (int i = mid - k; i <= mid + k; ++i)
      if (v[i] < v[i0]) i0 = i;
    const double lc = v[i0];
    const int nl = std::max(i0 - 3, 1), r0 = i0 + 4;
    if (nl < 2 || n - r0 < 2) return false;
    const double bl = percentile(std::vector<double>(v, v + nl), 80.0);
    const double br = percentile(std::vector<double>(v + r0, v + n), 80.0);
    const double d = std::min(bl, br) - lc;
    auto cross = [&](int side, double b) {
      const double hh = lc + 0.5 * (b - lc);
      int prev = i0;
      for (int i = i0; side < 0 ? i >= 0 : i < n; i += side) {
        if (v[i] >= hh) {
          if (i == prev) return 0.0;
          const double f = (hh - v[prev]) / std::max(v[i] - v[prev], 1e-9);
          return std::fabs((prev - i0) + f * (i - prev)) * 0.5;
        }
        prev = i;
      }
      return c->profHalf;
    };
    const double w = cross(-1, bl) + cross(1, br);
    // two strands side by side (diagnostic only): two minima 3..8 px apart, a bump >= 25 % of the depth
    int dip2 = 0;
    const int s0 = std::max(i0 - 16, 0), s1 = std::min(i0 + 17, n);
    const double* sm = v + s0;
    const int ns = s1 - s0;
    const double dd0 = std::max(std::min(bl, br) - lc, 1e-3);
    std::vector<int> mins;
    for (int i = 1; i < ns - 1; ++i)
      if (sm[i] <= sm[i - 1] && sm[i] < sm[i + 1] && std::min(bl, br) - sm[i] > 0.5 * dd0) mins.push_back(i);
    for (size_t a = 0; a < mins.size(); ++a)
      for (size_t b = a + 1; b < mins.size(); ++b) {
        const int gap = mins[b] - mins[a];
        if (gap < 6 || gap > 16) continue;
        double mx = sm[mins[a]];
        for (int i = mins[a]; i <= mins[b]; ++i) mx = std::max(mx, sm[i]);
        if (mx - std::max(sm[mins[a]], sm[mins[b]]) >= 0.25 * dd0) dip2 = 1;
      }
    const double xc = x + t[i0] * nx, yc = y + t[i0] * ny;
    const double o = w * 0.5 + 2.0;
    const double m1 = 0.5 * (samp(c1, W, H, xc - o * nx, yc - o * ny) + samp(c1, W, H, xc + o * nx, yc + o * ny));
    const double m2 = 0.5 * (samp(c2, W, H, xc - o * nx, yc - o * ny) + samp(c2, W, H, xc + o * nx, yc + o * ny));
    const double dd = std::max(d, 0.05);
    out.x = xc;
    out.y = yc;
    out.lc = lc;
    out.d = d;
    out.w = w;
    out.asym = std::fabs(bl - br);
    out.v1 = (samp(c1, W, H, xc, yc) - m1) / dd;
    out.v2 = (samp(c2, W, H, xc, yc) - m2) / dd;
    out.B = std::pow(2.0, 0.5 * (bl + br));
    out.dip2 = dip2;
    return true;
  }

  // Step score H (section 3.3).
  double score(const Prof& p, const Ref& ref, double dirx, double diry, double wsig = 0.0) const {
    if (wsig <= 0.0) wsig = c->wSig;
    double S = 1.0 / (1.0 + std::exp(-(p.d - c->d0) / c->dSoft));
    if (c->rWidth) {
      if (p.w > c->wMax || p.w < 0.5) return 0.0;
      const double q = std::log(std::max(p.w, 0.5) / ref.w) / wsig;
      S *= std::exp(-0.5 * q * q);
    }
    if (c->rProfile) {
      const double q = p.asym / (p.d + 0.05) / c->symSig;
      S *= std::exp(-0.5 * q * q);
    }
    if (c->rColour) {
      const double a = p.v1 - ref.v1, b = p.v2 - ref.v2;
      S *= std::exp(-0.5 * (a * a + b * b) / (c->colSig * c->colSig));
    }
    if (c->rContrast) {
      const double r = p.d / std::max(ref.d, 1e-3);
      if (r < c->contrastFloor) S *= (r / c->contrastFloor) * (r / c->contrastFloor);
    }
    if (c->rOrient) {
      const int ix = std::min(std::max(rnd(p.x), 0), W - 1), iy = std::min(std::max(rnd(p.y), 0), H - 1);
      const double cs = std::fabs(ux[at(ix, iy)] * dirx + uy[at(ix, iy)] * diry);
      const double ang = deg(std::acos(std::min(std::max(cs, 0.0), 1.0)));
      S *= std::exp(-0.5 * (ang / c->oriSigDeg) * (ang / c->oriSigDeg));
    }
    return S;
  }

  // One directional trace (section 3.4).
  TraceOut trace(double x, double y, double dx, double dy, const Ref& ref, const Mask* usedMask) const {
    TraceOut o;
    double C = 1.0, length = 0.0, ang = std::atan2(dy, dx);
    int low = 0;
    std::vector<double> hist(1, ang), ws;
    const double maxt = c->rTurn ? c->maxTurnDeg : 60.0;
    std::vector<double> cands;
    for (double a = -maxt; a <= maxt + 1e-6; a += 7.5) cands.push_back(rad(a));
    Ref wref = ref;
    bool anchored = false;
    for (;;) {
      bool have = false;
      double bs = 0.0, ba = 0.0;
      Prof bp{};
      for (double da : cands) {
        const double a = ang + da, ex = std::cos(a), ey = std::sin(a);
        const double qx = x + c->step * ex, qy = y + c->step * ey;
        if (!(1 <= qx && qx < W - 2 && 1 <= qy && qy < H - 2)) continue;
        Prof p;
        if (!profile(qx, qy, -ey, ex, p)) continue;
        double s = score(p, wref, ex, ey);
        if (c->rTurn) s *= std::exp(-0.5 * (deg(da) / c->turnSigmaDeg) * (deg(da) / c->turnSigmaDeg));
        if (!have || s > bs) {
          have = true;
          bs = s;
          bp = p;
          ba = a;
        }
      }
      (void)ba;
      if (!have) break;
      const double h = bs;
      ang = std::atan2(bp.y - y, bp.x - x);  // realised direction from the re-centred point
      x = bp.x;
      y = bp.y;
      length += c->step;
      hist.push_back(ang);
      if (c->rTurn && (int)hist.size() > c->winSteps) {
        const double dlt = hist.back() - hist[hist.size() - 1 - c->winSteps];
        if (deg(std::fabs(std::atan2(std::sin(dlt), std::cos(dlt)))) > c->winTurnDeg) break;
      }
      const int ix = rnd(x), iy = rnd(y);
      if (!search[at(ix, iy)]) break;
      if (alpha[at(ix, iy)] > 0.5) {
        anchored = true;
        break;
      }
      o.pts.push_back(Point{x, y});
      o.hs.push_back(h);
      ws.push_back(bp.w);
      o.ds.push_back(bp);
      if (usedMask && (*usedMask)[at(ix, iy)]) {
        o.hit.valid = true;
        o.hit.ix = ix;
        o.hit.iy = iy;
        break;
      }
      if (c->rDecay) {
        C = (1 - c->ema) * C + c->ema * h;
        low = h < c->hLow ? low + 1 : 0;
        if (C < c->cStop || low >= c->gapSteps) break;
      } else {
        const double D = 1.0 / (1.0 + std::exp(-(bp.d - c->d0) / c->dSoft));
        low = D < 0.5 ? low + 1 : 0;
        if (low >= c->gapSteps) break;
      }
      if (c->rBudget && length > budget) break;
      if (!c->rBudget && length > 4 * budget) break;
      if (c->rWidth && ws.size() >= 4) wref.w = 0.7 * wref.w + 0.3 * median(std::vector<double>(ws.end() - 4, ws.end()));
    }
    // trim the weak tail
    size_t n = o.hs.size();
    while (n > 0 && o.hs[n - 1] < c->trimH) --n;
    if (n < o.hs.size()) {
      anchored = false;
      o.hit.valid = false;
    }
    o.pts.resize(n);
    o.hs.resize(n);
    o.ds.resize(n);
    o.anchored = anchored;
    return o;
  }

  void mark(Mask& m, const std::vector<Point>& pts, const std::vector<Prof>& ds) const {
    for (size_t i = 0; i < pts.size(); ++i) {
      const int r = std::max(1, rnd(ds[i].w * 0.5));
      const int ix = rnd(pts[i].x), iy = rnd(pts[i].y);
      for (int y = std::max(iy - r, 0); y <= std::min(iy + r, H - 1); ++y)
        for (int x = std::max(ix - r, 0); x <= std::min(ix + r, W - 1); ++x) m[at(x, y)] = 1;
    }
  }
};

// Shape labels over the bounding rectangle (rx, ry, rw, rh) of region > 0.05 (the rest of the frame is never read
// twice); -1 = no shape.
struct Labels {
  int rx = 0, ry = 0, rw = 0, rh = 0;
  std::vector<int> l;
  int at(int x, int y) const {
    x -= rx;
    y -= ry;
    return x < 0 || y < 0 || x >= rw || y >= rh ? -1 : l[(size_t)y * rw + x];
  }
};

void buildShape(Shape& S, const TracerInput& in, const TracerParams& c, const Labels& label, int id,
                int x0, int y0, int x1, int y1) {
  S.c = &c;
  S.ox = x0;
  S.oy = y0;
  S.W = x1 - x0;
  S.H = y1 - y0;
  const int W = S.W, H = S.H;
  const size_t n = (size_t)W * H;
  Img lum(n), r(n), g(n), b(n);
  S.alpha.resize(n);
  S.region.resize(n);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t j = (size_t)(y + y0) * in.width + (x + x0), i = (size_t)y * W + x;
      const float* px = in.rgb + j * in.channels;
      r[i] = px[0];
      g[i] = px[1];
      b[i] = px[2];
      lum[i] = std::log2(std::max(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2], 1e-4));
      S.alpha[i] = in.base[j];
      const double rv = std::isfinite(in.region[j]) ? std::min(std::max((double)in.region[j], 0.0), 1.0) : 0.0;
      S.region[i] = (rv > 0.05 && label.at(x + x0, y + y0) != id) ? 0.0 : rv;  // only this shape's region
    }
  S.l = gauss(lum, W, H, 0.5);
  S.lin.resize(n);
  for (size_t i = 0; i < n; ++i) S.lin[i] = std::pow(2.0, S.l[i]);
  const Img rb = gauss(r, W, H, 0.7), gb = gauss(g, W, H, 0.7), bb = gauss(b, W, H, 0.7);
  S.c1.resize(n);
  S.c2.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const double gg = std::max(gb[i], 1e-4);
    S.c1[i] = std::log2(std::max(rb[i], 1e-4) / gg);
    S.c2[i] = std::log2(std::max(bb[i], 1e-4) / gg);
  }
  Mask reg(n);
  for (size_t i = 0; i < n; ++i) reg[i] = S.region[i] > 0.05;
  S.search = dilateCross(reg, W, H, c.searchDilate);
  const Img closed = rankFilter(rankFilter(S.l, W, H, c.closeSize, true), W, H, c.closeSize, false);
  S.bth.resize(n);
  for (size_t i = 0; i < n; ++i) S.bth[i] = closed[i] - S.l[i];
  // Hessian at sigma (scale-normalised), dark-line measure and the along-line unit vector
  const double s = c.hessSigma, s2 = s * s;
  const Img lxx = gauss(S.l, W, H, s, 0, 2), lyy = gauss(S.l, W, H, s, 2, 0), lxy = gauss(S.l, W, H, s, 1, 1);
  S.lm.resize(n);
  S.ux.resize(n);
  S.uy.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const double xx = lxx[i] * s2, yy = lyy[i] * s2, xy = lxy[i] * s2;
    const double tr = xx + yy, disc = std::sqrt(((xx - yy) * 0.5) * ((xx - yy) * 0.5) + xy * xy);
    const double lmax = tr * 0.5 + disc, lmin = tr * 0.5 - disc;
    double ax = xy, ay = lmax - xx, nn = std::hypot(ax, ay);
    if (nn < 1e-9) {
      ax = 1.0;
      ay = 0.0;
      nn = 1.0;
    }
    ax /= nn;
    ay /= nn;
    const double line = std::max(lmax, 0.0) - std::fabs(std::min(lmin, 0.0)) * 0.5 - std::max(lmin, 0.0);
    S.lm[i] = std::max(line, 0.0);
    S.ux[i] = -ay;
    S.uy[i] = ax;
  }
  int mnx = W, mxx = -1, mny = H, mxy = -1;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      if (S.search[S.at(x, y)]) {
        mnx = std::min(mnx, x);
        mxx = std::max(mxx, x);
        mny = std::min(mny, y);
        mxy = std::max(mxy, y);
      }
  const double diag = mxx >= 0 ? std::hypot(mxx - mnx + 1.0, mxy - mny + 1.0) : 1.0;
  S.budget = c.budgetFactor * diag;
  S.solid.resize(n);
  for (size_t i = 0; i < n; ++i) S.solid[i] = S.alpha[i] > 0.5;
  edt(S.solid, W, H, S.distOut, nullptr);
  S.used.assign(n, 0);
  S.usedAnc.assign(n, 0);
}

// Local maxima of the line measure (3x3, reflect) among `cand`, strongest first (ties: raster order).
std::vector<int> seedOrder(const Shape& S, const Mask& cand) {
  const Img mx = rankFilter(S.lm, S.W, S.H, 3, true);
  std::vector<int> idx;
  for (size_t i = 0; i < cand.size(); ++i)
    if (cand[i] && S.lm[i] == mx[i]) idx.push_back((int)i);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return S.lm[a] > S.lm[b]; });
  return idx;
}

std::vector<const Prof*> confident(const PieceIn& pc) {
  std::vector<const Prof*> o;
  for (size_t i = 0; i < pc.prof.size(); ++i)
    if (pc.h[i] > 0.5) o.push_back(&pc.prof[i]);
  return o;
}

// Section 3.5 (a): contact seeds.
void contactSeeds(Shape& S) {
  const TracerParams& c = *S.c;
  const size_t n = (size_t)S.W * S.H;
  Mask cand(n);
  for (size_t i = 0; i < n; ++i)
    cand[i] = S.bth[i] > c.seedBth && S.search[i] && !S.solid[i] && S.distOut[i] <= c.contactPx;
  Img gx, gy;
  gradient2d(S.distOut, S.W, S.H, gx, gy);
  for (int i : seedOrder(S, cand)) {
    if (S.used[i]) continue;
    const double x = i % S.W, y = i / S.W;
    double ex = S.ux[i], ey = S.uy[i];
    if (ex * gx[i] + ey * gy[i] < 0) {
      ex = -ex;
      ey = -ey;
    }
    Prof p;
    if (!S.profile(x, y, -ey, ex, p)) continue;
    const Ref ref = refFrom(p);
    TraceOut f = S.trace(x, y, ex, ey, ref, &S.used);
    if (f.pts.size() * c.step < c.minLen) continue;
    TraceOut bk = S.trace(x, y, -ex, -ey, ref, nullptr);  // root: back to the matte
    PieceIn pc;
    for (size_t k = bk.pts.size(); k-- > 0;) {
      pc.pts.push_back(bk.pts[k]);
      pc.h.push_back(bk.hs[k]);
      pc.prof.push_back(bk.ds[k]);
    }
    pc.pts.push_back(Point{p.x, p.y});
    pc.h.push_back(1.0);
    pc.prof.push_back(p);
    pc.pts.insert(pc.pts.end(), f.pts.begin(), f.pts.end());
    pc.h.insert(pc.h.end(), f.hs.begin(), f.hs.end());
    pc.prof.insert(pc.prof.end(), f.ds.begin(), f.ds.end());
    pc.anchored = true;
    pc.contact = true;
    S.mark(S.used, pc.pts, pc.prof);
    S.mark(S.usedAnc, pc.pts, pc.prof);
    S.pieces.push_back(pc);
  }
  std::vector<const Prof*> ps;
  for (const PieceIn& pc : S.pieces) {
    const std::vector<const Prof*> q = confident(pc);
    ps.insert(ps.end(), q.begin(), q.end());
  }
  S.contactRef = medianRef(ps, false);
}

// Section 3.5 (b): free seeds inside the region, traced both ways.
void freeSeeds(Shape& S) {
  const TracerParams& c = *S.c;
  const size_t n = (size_t)S.W * S.H;
  Mask cand(n);
  for (size_t i = 0; i < n; ++i) cand[i] = S.bth[i] > c.seedBth && S.search[i] && !S.solid[i] && S.region[i] > 0.05;
  for (int i : seedOrder(S, cand)) {
    const int ix = i % S.W, iy = i / S.W;
    bool near = false;  // maximum_filter(used, 3), mode reflect
    for (int dy = -1; dy <= 1 && !near; ++dy)
      for (int dx = -1; dx <= 1 && !near; ++dx)
        near = S.used[S.at(reflectIndex(ix + dx, S.W), reflectIndex(iy + dy, S.H))] != 0;
    if (near) continue;
    const double x = ix, y = iy, ex = S.ux[i], ey = S.uy[i];
    Prof p;
    if (!S.profile(x, y, -ey, ex, p)) continue;
    const Ref ref = S.hasHair ? S.hair : refFrom(p);
    const double s0 = S.score(p, ref, ex, ey, S.hasHair ? c.wSigCtx : 0.0);
    if (s0 < c.freeSeedConf) continue;
    TraceOut a = S.trace(x, y, ex, ey, ref, &S.used);
    TraceOut b = S.trace(x, y, -ex, -ey, ref, &S.used);
    PieceIn pc;
    for (size_t k = b.pts.size(); k-- > 0;) {
      pc.pts.push_back(b.pts[k]);
      pc.h.push_back(b.hs[k]);
      pc.prof.push_back(b.ds[k]);
    }
    pc.pts.push_back(Point{p.x, p.y});
    pc.h.push_back(s0);
    pc.prof.push_back(p);
    pc.pts.insert(pc.pts.end(), a.pts.begin(), a.pts.end());
    pc.h.insert(pc.h.end(), a.hs.begin(), a.hs.end());
    pc.prof.insert(pc.prof.end(), a.ds.begin(), a.ds.end());
    if (pc.pts.size() * c.step < c.minLen) continue;
    bool att = false;
    for (const Hit* e : {&a.hit, &b.hit})
      if (e->valid && S.usedAnc[S.at(e->ix, e->iy)]) att = true;
    pc.anchored = a.anchored || b.anchored || att;
    S.mark(S.used, pc.pts, pc.prof);
    if (pc.anchored) S.mark(S.usedAnc, pc.pts, pc.prof);
    S.pieces.push_back(pc);
  }
}

void pieceEnd(const PieceIn& pc, int k, Point& e, Point& t) {
  const size_t n = pc.pts.size();
  Point q;
  if (k == 0) {
    e = pc.pts[0];
    q = pc.pts[std::min<size_t>(4, n - 1)];
  } else {
    e = pc.pts[n - 1];
    q = pc.pts[n >= 5 ? n - 5 : 0];
  }
  const double tx = e.x - q.x, ty = e.y - q.y, nn = std::hypot(tx, ty);
  t = nn > 1e-6 ? Point{tx / nn, ty / nn} : Point{1.0, 0.0};
}

// Cubic Hermite between two end tangents; bend = integral of curvature^2, arc = length.
std::vector<Point> hermite(Point e1, Point t1, Point e2, Point t2, double& bend, double& arc, int n = 24) {
  const double L = std::hypot(e2.x - e1.x, e2.y - e1.y);
  std::vector<Point> P(n);
  for (int i = 0; i < n; ++i) {
    const double s = (double)i / (n - 1);
    const double h00 = 2 * s * s * s - 3 * s * s + 1, h10 = s * s * s - 2 * s * s + s;
    const double h01 = -2 * s * s * s + 3 * s * s, h11 = s * s * s - s * s;
    P[i] = Point{h00 * e1.x + h10 * L * t1.x + h01 * e2.x - h11 * L * t2.x,
                 h00 * e1.y + h10 * L * t1.y + h01 * e2.y - h11 * L * t2.y};
  }
  const std::vector<Point> d1 = gradientPts(P), d2 = gradientPts(d1);
  bend = arc = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ds = std::hypot(d1[i].x, d1[i].y);
    const double k = std::fabs(d1[i].x * d2[i].y - d1[i].y * d2[i].x) / std::max(ds * ds * ds, 1e-9);
    bend += k * k * ds;
    arc += ds;
  }
  return P;
}

double medianWidth(const PieceIn& pc) {
  std::vector<double> w;
  for (const Prof& p : pc.prof) w.push_back(p.w);
  return median(w);
}

// Section 3.6: joins (end -> matte, end <-> end), only low-cost and unambiguous.
void connect(Shape& S) {
  const TracerParams& c = *S.c;
  std::vector<PieceIn>& pieces = S.pieces;
  struct End {
    int i, k;
  };
  std::vector<End> ends;
  for (size_t i = 0; i < pieces.size(); ++i)
    if (pieces[i].pts.size() >= 3) {
      ends.push_back(End{(int)i, 0});
      ends.push_back(End{(int)i, 1});
    }
  struct Cand {
    int a, b;
    double J;
    std::vector<Point> P;
  };
  std::vector<Cand> cands;
  for (size_t a = 0; a < ends.size(); ++a)
    for (size_t b = a + 1; b < ends.size(); ++b) {
      const int i = ends[a].i, j = ends[b].i;
      if (i == j) continue;
      Point e1, t1, e2, t2;
      pieceEnd(pieces[i], ends[a].k, e1, t1);
      pieceEnd(pieces[j], ends[b].k, e2, t2);
      const double chx = e2.x - e1.x, chy = e2.y - e1.y, L = std::hypot(chx, chy);
      if (L < 1.0 || L > c.gapMax) continue;
      const double ux = chx / L, uy = chy / L;
      const double a1 = deg(std::acos(std::min(std::max(t1.x * ux + t1.y * uy, -1.0), 1.0)));
      const double a2 = deg(std::acos(std::min(std::max(-(t2.x * ux + t2.y * uy), -1.0), 1.0)));
      if (a1 > c.endAngleDeg || a2 > c.endAngleDeg) continue;
      double bend, arc;
      std::vector<Point> P = hermite(e1, t1, e2, t2, bend, arc);
      int sn = 0, sk = 0;
      for (size_t q = 2; q + 2 < P.size(); ++q) {
        const int ix = rnd(P[q].x), iy = rnd(P[q].y);
        if (ix >= 0 && ix < S.W && iy >= 0 && iy < S.H) {
          ++sn;
          sk += S.bth[S.at(ix, iy)] > 0.5 * c.seedBth;
        }
      }
      const double support = sn ? (double)sk / sn : 0.0;
      const double w1 = medianWidth(pieces[i]), w2 = medianWidth(pieces[j]);
      const double J = bend * arc * 0.5 + L / c.gapMax * 0.3 + std::fabs(std::log(w1 / w2)) * 0.5 + (1 - support) * 0.6;
      cands.push_back(Cand{(int)a, (int)b, J, P});
    }
  // end -> matte
  std::vector<JoinIn> mj;
  bool anySolid = false;
  for (uint8_t v : S.solid) anySolid = anySolid || v;
  if (anySolid) {
    Img d;
    std::vector<int> nearest;
    edt(S.solid, S.W, S.H, d, &nearest);
    for (size_t i = 0; i < pieces.size(); ++i) {
      PieceIn& pc = pieces[i];
      if (pc.anchored || pc.pts.size() < 3) continue;
      for (int k = 0; k < 2; ++k) {
        Point e, t;
        pieceEnd(pc, k, e, t);
        const int ex = std::min(std::max(rnd(e.x), 0), S.W - 1), ey = std::min(std::max(rnd(e.y), 0), S.H - 1);
        const int ni = nearest[S.at(ex, ey)];
        if (ni < 0) continue;
        // The join's cost to the solid pixel m (infinity: not allowed), and its curve.
        auto cost = [&](const Point& m, std::vector<Point>* curve) {
          const double chx = m.x - e.x, chy = m.y - e.y, L = std::hypot(chx, chy);
          if (L < 0.5 || L > c.gapMax) return std::numeric_limits<double>::infinity();
          const double ang = deg(std::acos(std::min(std::max((t.x * chx + t.y * chy) / L, -1.0), 1.0)));
          if (ang > c.endAngleDeg) return std::numeric_limits<double>::infinity();
          double bend, arc;
          std::vector<Point> P = hermite(e, t, m, Point{-chx / L, -chy / L}, bend, arc);
          int sn = 0, sk = 0;
          for (size_t q = 1; q + 1 < P.size(); ++q) {
            const int ix = std::min(std::max(rnd(P[q].x), 0), S.W - 1);
            const int iy = std::min(std::max(rnd(P[q].y), 0), S.H - 1);
            if (S.solid[S.at(ix, iy)]) continue;
            ++sn;
            sk += S.bth[S.at(ix, iy)] > 0.5 * c.seedBth;
          }
          const double support = sn ? (double)sk / sn : 1.0;
          if (curve) *curve = P;
          return bend * arc * 0.5 + L / c.gapMax * 0.3 + (1 - support) * 0.6;
        };
        Point best{(double)(ni % S.W), (double)(ni / S.W)};
        double J = cost(best, nullptr);
        if (c.coneJoin && !(J <= c.joinMax)) {  // the nearest solid pixel is off the strand's way: the cheapest
          const int r = (int)std::ceil(c.gapMax);  // matte edge pixel within gapMax in the end's direction
          for (int yy = std::max(ey - r, 0); yy <= std::min(ey + r, S.H - 1); ++yy)
            for (int xx = std::max(ex - r, 0); xx <= std::min(ex + r, S.W - 1); ++xx) {
              if (!S.solid[S.at(xx, yy)]) continue;
              const bool edgePx = (xx > 0 && !S.solid[S.at(xx - 1, yy)]) || (xx + 1 < S.W && !S.solid[S.at(xx + 1, yy)]) ||
                                  (yy > 0 && !S.solid[S.at(xx, yy - 1)]) || (yy + 1 < S.H && !S.solid[S.at(xx, yy + 1)]);
              if (!edgePx) continue;
              const Point m{(double)xx, (double)yy};
              const double Jm = cost(m, nullptr);
              if (Jm < J) {
                J = Jm;
                best = m;
              }
            }
        }
        if (J <= c.joinMax) {
          std::vector<Point> P;
          cost(best, &P);
          pc.anchored = true;
          pc.conf += c.anchorBonus;
          JoinIn jn;
          jn.pts = P;
          jn.J = J;
          jn.a = jn.b = (int)i;
          jn.w = medianWidth(pc);
          jn.matte = true;
          mj.push_back(jn);
          break;
        }
      }
    }
  }
  // end <-> end: mutual best, best < ambig x second best, else abstain
  std::vector<std::vector<std::pair<double, std::pair<int, int>>>> best(ends.size());
  for (const Cand& cd : cands)
    for (int e : {cd.a, cd.b}) best[e].push_back(std::make_pair(cd.J, std::make_pair(cd.a, cd.b)));
  for (auto& l : best) std::sort(l.begin(), l.end());
  std::vector<JoinIn> joins;
  for (const Cand& cd : cands) {
    if (cd.J > c.joinMax) continue;
    bool ok = true;
    for (int e : {cd.a, cd.b}) {
      const auto& lst = best[e];
      if (lst[0].second != std::make_pair(cd.a, cd.b))
        ok = false;
      else if (lst.size() > 1 && lst[0].first > c.ambig * lst[1].first)
        ok = false;  // ambiguous -> abstain
    }
    if (!ok) continue;
    const int i = ends[cd.a].i, j = ends[cd.b].i;
    JoinIn jn;
    jn.pts = cd.P;
    jn.J = cd.J;
    jn.a = i;
    jn.b = j;
    jn.w = 0.5 * (medianWidth(pieces[i]) + medianWidth(pieces[j]));
    joins.push_back(jn);
    if (pieces[i].anchored || pieces[j].anchored)
      for (int q : {i, j})
        if (!pieces[q].anchored) {
          pieces[q].anchored = true;
          pieces[q].conf += c.anchorBonus;
        }
  }
  joins.insert(joins.end(), mj.begin(), mj.end());
  S.joins = joins;
}

// The longest qualifying piece's reference (section 3.7, "no anchored piece").
HairReference selfRef(const Shape& S, size_t* len) {
  const TracerParams& c = *S.c;
  const PieceIn* bestPc = nullptr;
  for (const PieceIn& pc : S.pieces) {
    std::vector<double> w, sy, d;
    for (const Prof& p : pc.prof) {
      w.push_back(p.w);
      sy.push_back(p.asym / (p.d + 0.05));
      d.push_back(p.d);
    }
    if (median(w) <= c.wMax && median(sy) <= 0.3 && median(d) >= 2 * c.d0)
      if (!bestPc || pc.pts.size() > bestPc->pts.size()) bestPc = &pc;
  }
  if (len) *len = bestPc ? bestPc->pts.size() : 0;
  if (!bestPc) return HairReference();
  std::vector<const Prof*> pp = confident(*bestPc);
  if (pp.empty())
    for (const Prof& p : bestPc->prof) pp.push_back(&p);
  return medianRef(pp, true);
}

// Soft strand profile (TracerParams::softProfile): peak x blurred-bar coverage x free-end taper.
void renderSoft(const Shape& S, const std::vector<int>& keep, const std::vector<int>& keepJoins, double Fg, bool hasF, double sigma, std::vector<double>& A,
                std::vector<double>& conf) {
  const TracerParams& c = *S.c;
  const int W = S.W, H = S.H;
  sigma = std::max(sigma, 0.3);
  const double s2 = sigma * std::sqrt(2.0);
  auto prof = [&](double d, double half) {  // a bar of half-width `half` blurred by the Gaussian, at distance d
    return 0.5 * (std::erf((half - d) / s2) + std::erf((half + d) / s2));
  };
  // ends joined to something: a kept join's end point near a piece end
  auto near = [](const Point& a, const Point& b) { return std::hypot(a.x - b.x, a.y - b.y) <= 3.0; };
  std::vector<double> peakOf(S.pieces.size(), 1.0);
  struct Seg {
    const std::vector<Point>* pts;
    double w, peak, cf;
    bool free0, free1;
  };
  std::vector<Seg> segs;
  for (int k : keep) {
    const PieceIn& pc = S.pieces[k];
    std::vector<double> core;
    for (const Prof& p : pc.prof)
      if (p.w >= 3.0) core.push_back(std::pow(2.0, p.lc));
    const bool okF = core.size() >= 3 || hasF;
    const double F = core.size() >= 3 ? percentile(core, 10.0) : Fg;
    std::vector<double> pk, ws;
    for (size_t q = 0; q < pc.pts.size(); ++q) {
      ws.push_back(pc.prof[q].w);
      const double B = pc.prof[q].B;
      if (okF && B > F * 1.2) {
        double L = 1e30;  // the darkest pixel within 1 px of the centreline point (the point is sub-pixel)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int x = rnd(pc.pts[q].x) + dx, y = rnd(pc.pts[q].y) + dy;
            if (x >= 0 && y >= 0 && x < W && y < H) L = std::min(L, S.lin[S.at(x, y)]);
          }
        pk.push_back(std::min(std::max((B - L) / (B - F), 0.0), 1.0));
      }
    }
    // the centreline's matting alpha is what the picture shows through its blur; the rendered profile's centre is
    // peak x prof(0, half), so the peak is chosen to reproduce the observed centre value
    const double half0 = 0.5 * std::max(median(ws), 1.0);
    const double peak = pk.size() >= 3 ? std::min(std::max(median(pk), 0.3) / prof(0.0, half0), 1.0) : 1.0;
    peakOf[k] = peak;
    auto solidNear = [&](const Point& e) {  // the end touches the matte (within 3 px)
      for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx) {
          const int x = rnd(e.x) + dx, y = rnd(e.y) + dy;
          if (x >= 0 && y >= 0 && x < W && y < H && S.alpha[S.at(x, y)] > 0.5) return true;
        }
      return false;
    };
    Seg sg{&pc.pts, median(ws), peak, pc.conf, !solidNear(pc.pts.front()), !solidNear(pc.pts.back())};
    for (int j : keepJoins) {
      const JoinIn& jn = S.joins[j];
      if (jn.pts.empty()) continue;
      for (const Point& e : {jn.pts.front(), jn.pts.back()}) {
        if ((jn.a == k || jn.b == k) && near(e, pc.pts.front())) sg.free0 = false;
        if ((jn.a == k || jn.b == k) && near(e, pc.pts.back())) sg.free1 = false;
      }
    }
    segs.push_back(sg);
  }
  for (int j : keepJoins) {
    const JoinIn& jn = S.joins[j];
    segs.push_back(Seg{&jn.pts, jn.w, jn.a >= 0 ? peakOf[jn.a] : 1.0, 0.5, false, false});
  }
  for (const Seg& sg : segs) {
    const std::vector<Point>& P = *sg.pts;
    if (P.empty()) continue;
    // a bar of the measured FWHM blurred by sigma keeps about that FWHM (sigma <= w / 3)
    const double half = 0.5 * std::max(sg.w, 1.0), taper = std::max(c.taperW * sg.w, 1.0);
    // arc length of every point from each end
    std::vector<double> arc(P.size(), 0.0);
    for (size_t q = 1; q < P.size(); ++q) arc[q] = arc[q - 1] + std::hypot(P[q].x - P[q - 1].x, P[q].y - P[q - 1].y);
    const double total = arc.back();
    const int r = (int)std::ceil(half + 3.0 * sigma + 1.0);
    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (const Point& q : P) x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
    const int ix0 = std::max((int)std::floor(x0) - r, 0), ix1 = std::min((int)std::ceil(x1) + r, W - 1);
    const int iy0 = std::max((int)std::floor(y0) - r, 0), iy1 = std::min((int)std::ceil(y1) + r, H - 1);
    for (int yy = iy0; yy <= iy1; ++yy)
      for (int xx = ix0; xx <= ix1; ++xx) {
        // nearest point on the polyline: distance and arc position
        double best = 1e30, at = 0.0;
        if (P.size() == 1) {
          best = std::hypot(xx - P[0].x, yy - P[0].y);
        } else {
          for (size_t q = 1; q < P.size(); ++q) {
            const double vx = P[q].x - P[q - 1].x, vy = P[q].y - P[q - 1].y, l2 = vx * vx + vy * vy;
            double u = l2 > 0 ? ((xx - P[q - 1].x) * vx + (yy - P[q - 1].y) * vy) / l2 : 0.0;
            u = std::min(std::max(u, 0.0), 1.0);
            const double d = std::hypot(xx - P[q - 1].x - u * vx, yy - P[q - 1].y - u * vy);
            if (d < best) best = d, at = arc[q - 1] + u * std::sqrt(l2);
          }
        }
        if (best > half + 3.0 * sigma) continue;
        double f = 1.0;  // free ends fade out along the strand (smoothstep)
        auto ss = [](double v) { v = std::min(std::max(v, 0.0), 1.0); return v * v * (3.0 - 2.0 * v); };
        if (sg.free0) f *= ss(at / taper + 0.25);
        if (sg.free1) f *= ss((total - at) / taper + 0.25);
        const double a = sg.peak * f * prof(best, half);
        if (a < 1e-3) continue;
        const size_t i = S.at(xx, yy);
        A[i] = std::max(A[i], a);
        conf[i] = std::max(conf[i], sg.cf);
      }
  }
}

void renderShape(const Shape& S, const std::vector<int>& keep, const std::vector<int>& keepJoins,
                 std::vector<double>& A, std::vector<double>& conf, double sigma) {
  const TracerParams& c = *S.c;
  const int W = S.W, H = S.H;
  A.assign((size_t)W * H, 0.0);
  conf.assign((size_t)W * H, 0.0);
  // hair luminance F from the solid matte near its edge (10th percentile)
  Mask solid95((size_t)W * H);
  for (size_t i = 0; i < solid95.size(); ++i) solid95[i] = S.alpha[i] > 0.95;
  const Mask er = erodeCross(solid95, W, H, 12);
  std::vector<double> nearV;
  for (size_t i = 0; i < solid95.size(); ++i)
    if (solid95[i] && !er[i] && S.search[i]) nearV.push_back(S.lin[i]);
  const bool hasF = nearV.size() > 10;
  const double Fg = hasF ? percentile(nearV, 10.0) : 0.0;
  struct Seg {
    const std::vector<Point>* pts;
    std::vector<double> ws, Bs;
    bool hasB;
    double cf, F;
    bool hasF;
  };
  std::vector<Seg> segs;
  for (int k : keep) {
    const PieceIn& pc = S.pieces[k];
    std::vector<double> core;
    for (const Prof& p : pc.prof)
      if (p.w >= 3.0) core.push_back(std::pow(2.0, p.lc));
    Seg s;
    s.pts = &pc.pts;
    for (const Prof& p : pc.prof) {
      s.ws.push_back(p.w);
      s.Bs.push_back(p.B);
    }
    s.hasB = true;
    s.cf = pc.conf;
    s.hasF = core.size() >= 3 || hasF;
    s.F = core.size() >= 3 ? percentile(core, 10.0) : Fg;
    segs.push_back(s);
  }
  for (int k : keepJoins) {
    const JoinIn& jn = S.joins[k];
    Seg s;
    s.pts = &jn.pts;
    s.ws.assign(jn.pts.size(), jn.w);
    s.Bs.assign(jn.pts.size(), 0.0);
    s.hasB = false;
    s.cf = 0.5;
    s.hasF = hasF;
    s.F = Fg;
    segs.push_back(s);
  }
  if (c.softProfile) {
    renderSoft(S, keep, keepJoins, Fg, hasF, sigma, A, conf);
    return;
  }
  for (const Seg& sg : segs)
    for (size_t q = 0; q < sg.pts->size(); ++q) {
      const double x = (*sg.pts)[q].x, y = (*sg.pts)[q].y, w = sg.ws[q], B = sg.Bs[q];
      const int r = (int)std::ceil(w * 0.5 + 2);
      const int x0 = std::max((int)x - r, 0), x1 = std::min((int)x + r + 2, W);
      const int y0 = std::max((int)y - r, 0), y1 = std::min((int)y + r + 2, H);
      const double wt = std::max(std::sqrt(std::max(w * w - 1.5, 0.5)), 0.7);  // remove ~PSF from FWHM
      const bool matting = !c.capsuleAlpha && sg.hasF && sg.hasB && B > sg.F * 1.2;
      for (int yy = y0; yy < y1; ++yy)
        for (int xx = x0; xx < x1; ++xx) {
          const double dd = std::hypot(xx - x, yy - y);
          double a;
          if (matting) {
            a = std::min(std::max((B - S.lin[S.at(xx, yy)]) / (B - sg.F), 0.0), 1.0);
            if (!(dd <= wt * 0.5 + 1.0)) a = 0.0;
          } else {
            a = std::min(std::max(wt * 0.5 + 0.5 - dd, 0.0), 1.0);
          }
          const size_t i = S.at(xx, yy);
          A[i] = std::max(A[i], a);
          conf[i] = std::max(conf[i], a > 0 ? sg.cf : 0.0);
        }
    }
}

// Region shapes. A shape = a 4-connected component of the DRAWN region (weight > 0.5); the feather pixels
// (0.05 < weight <= 0.5) go to the nearest shape through the feather (breadth-first from all shapes at once), so two shapes whose feathers touch stay two shapes; a feather-only blob is a shape of its own.
// Returns the shape count, the label per pixel (-1 = none) and bounding boxes (x0, y0, x1, y1 inclusive).
int labelRegion(const TracerInput& in, Labels& lab, std::vector<int>& box) {
  int x0 = in.width, y0 = in.height, x1 = -1, y1 = -1;
  for (int y = 0; y < in.height; ++y) {
    const float* row = in.region + (size_t)y * in.width;
    for (int x = 0; x < in.width; ++x)
      if (row[x] > 0.05f) {
        x0 = std::min(x0, x);
        x1 = std::max(x1, x);
        y0 = std::min(y0, y);
        y1 = y;
      }
  }
  if (x1 < 0) return 0;
  lab.rx = x0;
  lab.ry = y0;
  const int W = lab.rw = x1 - x0 + 1, H = lab.rh = y1 - y0 + 1;
  std::vector<int>& label = lab.l;
  label.assign((size_t)W * H, -1);
  auto inside = [&](size_t i, float level) {
    return in.region[(size_t)(i / W + y0) * in.width + (i % W + x0)] > level;
  };
  int nlab = 0;
  std::vector<int> stack, queue;
  auto flood = [&](size_t seed, float level) {
    label[seed] = nlab;
    stack.assign(1, (int)seed);
    while (!stack.empty()) {
      const int j = stack.back();
      stack.pop_back();
      queue.push_back(j);
      const int jx = j % W, jy = j / W;
      const int nb[4][2] = {{jx - 1, jy}, {jx + 1, jy}, {jx, jy - 1}, {jx, jy + 1}};
      for (const auto& q : nb) {
        if (q[0] < 0 || q[0] >= W || q[1] < 0 || q[1] >= H) continue;
        const size_t k = (size_t)q[1] * W + q[0];
        if (label[k] < 0 && inside(k, level)) {
          label[k] = nlab;
          stack.push_back((int)k);
        }
      }
    }
    ++nlab;
  };
  for (size_t i = 0; i < (size_t)W * H; ++i)  // the drawn shapes
    if (label[i] < 0 && inside(i, 0.5f)) flood(i, 0.5f);
  // grow them through the feather, breadth-first from all at once (queue in raster order = deterministic)
  std::sort(queue.begin(), queue.end());
  for (size_t h = 0; h < queue.size(); ++h) {
    const int j = queue[h], jx = j % W, jy = j / W;
    const int nb[4][2] = {{jx - 1, jy}, {jx + 1, jy}, {jx, jy - 1}, {jx, jy + 1}};
    for (const auto& q : nb) {
      if (q[0] < 0 || q[0] >= W || q[1] < 0 || q[1] >= H) continue;
      const size_t k = (size_t)q[1] * W + q[0];
      if (label[k] < 0 && inside(k, 0.05f)) {
        label[k] = label[j];
        queue.push_back((int)k);
      }
    }
  }
  for (size_t i = 0; i < (size_t)W * H; ++i)  // feather-only blobs
    if (label[i] < 0 && inside(i, 0.05f)) flood(i, 0.05f);
  box.assign(4 * (size_t)nlab, 0);
  for (int k = 0; k < nlab; ++k) {
    box[4 * k] = in.width;
    box[4 * k + 1] = in.height;
    box[4 * k + 2] = box[4 * k + 3] = -1;
  }
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const int k = label[(size_t)y * W + x];
      if (k < 0) continue;
      box[4 * k] = std::min(box[4 * k], x + x0);
      box[4 * k + 1] = std::min(box[4 * k + 1], y + y0);
      box[4 * k + 2] = std::max(box[4 * k + 2], x + x0);
      box[4 * k + 3] = std::max(box[4 * k + 3], y + y0);
    }
  return nlab;
}

}  // namespace

double minPieceLength(const TracerParams& p, int frameHeight, int height, double renderScale) {
  if (p.minPieceLen1080 <= 0.0) return 0.0;
  const double rs = renderScale > 0.0 ? renderScale : 1.0;
  const double fh = frameHeight > 0 ? frameHeight : height / rs;
  return p.minPieceLen1080 * fh / 1080.0 * rs;
}

bool clearlyDiffers(const HairReference& own, const HairReference& shared, const TracerParams& p) {
  if (!own.valid || !shared.valid) return false;
  if (std::fabs(std::log(std::max(own.width, 1e-3) / std::max(shared.width, 1e-3))) > p.wSigCtx) return true;
  const double r = std::max(own.depth, 1e-3) / std::max(shared.depth, 1e-3);
  if (r < p.contrastFloor || r > 1.0 / p.contrastFloor) return true;
  return std::hypot(own.chroma1 - shared.chroma1, own.chroma2 - shared.chroma2) > 2.0 * p.colSig;
}

TracerResult traceStrands(const TracerInput& in, const TracerParams& c) {
  TracerResult res;
  const int W = in.width, H = in.height;
  if (W <= 0 || H <= 0 || !in.rgb || !in.base || !in.region || in.channels < 3) return res;
  res.alpha.assign((size_t)W * H, 0.0f);
  res.confidence.assign((size_t)W * H, 0.0f);
  Labels label;
  std::vector<int> box;
  const int nShapes = labelRegion(in, label, box);
  res.shapes = nShapes;
  if (!nShapes) return res;
  std::vector<Shape> shapes(nShapes);
  for (int k = 0; k < nShapes; ++k) {
    const int x0 = std::max(box[4 * k] - c.cropMargin, 0), y0 = std::max(box[4 * k + 1] - c.cropMargin, 0);
    const int x1 = std::min(box[4 * k + 2] + c.cropMargin + 1, W), y1 = std::min(box[4 * k + 3] + c.cropMargin + 1, H);
    buildShape(shapes[k], in, c, label, k, x0, y0, x1, y1);
  }
  // 1. contact seeds per shape; the frame's contact reference pools all shapes
  std::vector<const Prof*> pool;
  for (Shape& S : shapes) {
    contactSeeds(S);
    for (const PieceIn& pc : S.pieces) {
      const std::vector<const Prof*> q = confident(pc);
      pool.insert(pool.end(), q.begin(), q.end());
    }
  }
  const HairReference frameContact = medianRef(pool, false);
  // 2. free seeds against the contact reference, then joins
  for (Shape& S : shapes) {
    HairReference r = S.contactRef;
    // a shape with contact pieces seeds against the shared contact reference (unless it clearly differs); one
    // without seeds self-referenced as in the prototype: the shared reference decides at the keep step
    if (c.frameHairRef && frameContact.valid && r.valid && !clearlyDiffers(r, frameContact, c)) r = frameContact;
    S.hasHair = r.valid;
    if (r.valid) S.hair = toRef(r);
    freeSeeds(S);
    for (PieceIn& pc : S.pieces) pc.conf = median(pc.h);
    if (c.rConnect) connect(S);
  }
  // 3. hair reference: per frame from anchored pieces (else self-referenced), per-shape fallback (ruling 2)
  pool.clear();
  for (Shape& S : shapes) {
    std::vector<const Prof*> own;
    for (const PieceIn& pc : S.pieces)
      if (pc.anchored) {
        const std::vector<const Prof*> q = confident(pc);
        own.insert(own.end(), q.begin(), q.end());
      }
    pool.insert(pool.end(), own.begin(), own.end());
    S.ownRef = medianRef(own, false);
    if (!S.ownRef.valid) S.ownRef = selfRef(S, nullptr);
  }
  HairReference frameRef;
  if (in.reference && in.reference->valid) {
    frameRef = *in.reference;
  } else if (c.frameHairRef) {
    frameRef = medianRef(pool, false);
    if (!frameRef.valid) {  // no anchored piece anywhere: the frame's longest qualifying piece
      size_t bestLen = 0;
      for (Shape& S : shapes) {
        size_t len = 0;
        const HairReference r = selfRef(S, &len);
        if (r.valid && len > bestLen) {
          bestLen = len;
          frameRef = r;
        }
      }
    }
  }
  // frameRef stays invalid in the prototype's per-shape mode (frameHairRef off, no reference passed in)
  if (c.rHairCtx) {
    for (Shape& S : shapes) {
      HairReference r = S.ownRef;
      if (frameRef.valid && !(r.valid && clearlyDiffers(r, frameRef, c))) r = frameRef;
      S.hasHair = r.valid;
      S.selfref = r.valid && r.selfReferenced;
      if (r.valid) S.hair = toRef(r);
    }
  }
  res.reference = frameRef.valid || shapes.empty() ? frameRef : shapes[0].ownRef;
  // 4. rescore unanchored pieces, confidence, keep, rulings 1 and 4, render
  const double minLen = minPieceLength(c, in.frameHeight, H, in.renderScale);
  for (int k = 0; k < nShapes; ++k) {
    Shape& S = shapes[k];
    for (PieceIn& pc : S.pieces) {
      if (c.rHairCtx && S.hasHair && !pc.anchored) {
        const std::vector<Point> T = pc.pts.size() > 1 ? gradientPts(pc.pts) : std::vector<Point>(1, Point{1.0, 0.0});
        for (size_t i = 0; i < pc.prof.size(); ++i) {
          const double nn = std::max(std::hypot(T[i].x, T[i].y), 1e-9);
          pc.h[i] = S.score(pc.prof[i], S.hair, T[i].x / nn, T[i].y / nn, c.wSigCtx);
        }
      }
      pc.conf = median(pc.h) + (pc.anchored ? c.anchorBonus : 0.0);
      if (S.selfref && !pc.anchored) pc.conf *= 0.8;
    }
    // Ruling 4's length per piece: its own, or its chain's (chainLength, see the header).
    std::vector<double> ruleLen(S.pieces.size());
    for (size_t i = 0; i < S.pieces.size(); ++i) ruleLen[i] = S.pieces[i].pts.size() * c.step;
    if (c.chainLength && !S.pieces.empty()) {
      std::vector<int> root(S.pieces.size());
      for (size_t i = 0; i < root.size(); ++i) root[i] = (int)i;
      std::function<int(int)> find = [&](int i) { return root[i] == i ? i : (root[i] = find(root[i])); };
      auto arc = [](const std::vector<Point>& P) {
        double L = 0.0;
        for (size_t q = 1; q < P.size(); ++q) L += std::hypot(P[q].x - P[q - 1].x, P[q].y - P[q - 1].y);
        return L;
      };
      for (const JoinIn& jn : S.joins)
        if (!jn.matte && jn.a >= 0 && jn.b >= 0) root[find(jn.a)] = find(jn.b);
      std::vector<double> chain(S.pieces.size(), 0.0);
      std::vector<uint8_t> attached(S.pieces.size(), 0);  // the chain touches the matte: a contact piece or a matte join
      for (size_t i = 0; i < S.pieces.size(); ++i) {
        chain[find((int)i)] += ruleLen[i];
        if (S.pieces[i].contact) attached[find((int)i)] = 1;
      }
      for (const JoinIn& jn : S.joins)
        if (jn.a >= 0) {
          chain[find(jn.a)] += arc(jn.pts);  // end-to-end joins and matte joins
          if (jn.matte) attached[find(jn.a)] = 1;
        }
      for (size_t i = 0; i < S.pieces.size(); ++i)  // a chain attached to the matte continues it: never a speck
        if (attached[find((int)i)]) chain[find((int)i)] = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < S.pieces.size(); ++i) ruleLen[i] = chain[find((int)i)];
    }
    std::vector<int> keep;
    std::vector<int> outIndex(S.pieces.size(), -1);
    for (size_t i = 0; i < S.pieces.size(); ++i) {
      const PieceIn& pc = S.pieces[i];
      const double len = pc.pts.size() * c.step;
      bool edge = false;
      for (const Point& p : pc.pts) edge = edge || S.region[S.at(rnd(p.x), rnd(p.y))] < 0.5;
      int why = 0;
      if (pc.conf < c.keepConf) why = 1;
      else if (c.rHairCtx && !pc.anchored && !S.selfref && pc.conf < c.minConfUnanchored) why = 2;
      else if (c.rHairCtx && !pc.anchored && !S.selfref && len < c.freeLenW * (S.hasHair ? S.hair.w : 3.0)) why = 3;
      else if (c.dropEdgeTouching && edge && !pc.anchored) why = 4;  // ruling 1
      else if (c.rElongation && !pc.anchored && len < c.minElongation * medianWidth(pc)) why = 5;  // blobs (moles)
      else if (ruleLen[i] < minLen) why = 6;                        // ruling 4
      if (why) {
        Piece o;
        for (const Point& p : pc.pts) o.polyline.push_back(Point{p.x + S.ox, p.y + S.oy});
        o.width = medianWidth(pc);
        o.length = len;
        o.confidence = pc.conf;
        o.anchored = pc.anchored;
        o.touchesRegionEdge = edge;
        o.shape = k;
        res.rejected.push_back(o);
        res.rejectReason.push_back(why);
        continue;
      }
      outIndex[i] = (int)res.pieces.size();
      keep.push_back((int)i);
      Piece o;
      for (const Point& p : pc.pts) o.polyline.push_back(Point{p.x + S.ox, p.y + S.oy});
      o.width = medianWidth(pc);
      o.length = len;
      o.confidence = pc.conf;
      o.anchored = pc.anchored;
      o.touchesRegionEdge = edge;
      o.selfReferenced = S.selfref && !pc.anchored;
      o.shape = k;
      res.pieces.push_back(o);
    }
    std::vector<int> keepJoins;
    for (size_t j = 0; j < S.joins.size(); ++j) {
      const JoinIn& jn = S.joins[j];
      if (outIndex[jn.a] < 0 || outIndex[jn.b] < 0) continue;
      keepJoins.push_back((int)j);
      Join o;
      for (const Point& p : jn.pts) o.polyline.push_back(Point{p.x + S.ox, p.y + S.oy});
      o.width = jn.w;
      o.cost = jn.J;
      o.toMatte = jn.matte;
      o.pieceA = outIndex[jn.a];
      o.pieceB = outIndex[jn.b];
      res.joins.push_back(o);
    }
    std::vector<double> A, cf;
    const double fh = in.frameHeight > 0 ? in.frameHeight : H / (in.renderScale > 0.0 ? in.renderScale : 1.0);
    renderShape(S, keep, keepJoins, A, cf, c.edgeSigma1080 * fh / 1080.0 * (in.renderScale > 0.0 ? in.renderScale : 1.0));
    for (int y = 0; y < S.H; ++y)
      for (int x = 0; x < S.W; ++x) {
        const size_t i = S.at(x, y), j = (size_t)(y + S.oy) * W + (x + S.ox);
        const double rg = S.region[i];
        if (!(rg > 0.0)) continue;
        const float a = (float)(A[i] * rg);
        if (a > res.alpha[j]) res.alpha[j] = a;
        if (a > 0.0f) res.confidence[j] = std::max(res.confidence[j], (float)cf[i]);
      }
  }
  return res;
}

}  // namespace tracer
}  // namespace mr
