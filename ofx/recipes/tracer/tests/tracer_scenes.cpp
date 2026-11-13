// Synthetic scenes and metrics for the strand tracer: see tracer_scenes.h. Follows synth.py / evaluate.py of the
// prototype step by step (same geometry, colours and metric definitions; a portable random generator).
#include "tracer_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mr {
namespace tracer {
namespace scenes {
namespace {

const int N = 112;
const double kPi = 3.14159265358979323846;
const double SKIN[3] = {0.60, 0.42, 0.33}, HAIR[3] = {0.05, 0.035, 0.025}, BLACK[3] = {0.02, 0.02, 0.022};

struct Rng {  // splitmix64; uniform in [0,1); Box-Muller normals
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 12345) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal(double sd) {
    double u1 = uniform();
    while (u1 <= 0.0) u1 = uniform();
    return sd * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * uniform());
  }
  int integer(int a, int b) { return a + (int)(uniform() * (b - a)); }  // [a, b)
};

typedef std::vector<double> Img;

int reflectIndex(int i, int n) {
  const int period = 2 * n;
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - 1 - i;
}

Img gauss(const Img& in, double sigma) {  // scipy gaussian_filter, truncate 4, mode reflect, N x N
  const int r = (int)(4.0 * sigma + 0.5);
  std::vector<double> k(2 * r + 1);
  double sum = 0;
  for (int x = -r; x <= r; ++x) sum += (k[x + r] = std::exp(-0.5 * x * x / (sigma * sigma)));
  for (double& v : k) v /= sum;
  Img tmp(in.size()), out(in.size());
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      double s = 0;
      for (int j = -r; j <= r; ++j) s += k[j + r] * in[y * N + reflectIndex(x + j, N)];
      tmp[y * N + x] = s;
    }
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      double s = 0;
      for (int j = -r; j <= r; ++j) s += k[j + r] * tmp[reflectIndex(y + j, N) * N + x];
      out[y * N + x] = s;
    }
  return out;
}

std::vector<Point> bezier(const Point P[4], int n = 400) {
  std::vector<Point> o(n);
  for (int i = 0; i < n; ++i) {
    const double t = (double)i / (n - 1), u = 1 - t;
    const double a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    o[i] = Point{a * P[0].x + b * P[1].x + c * P[2].x + d * P[3].x, a * P[0].y + b * P[1].y + c * P[2].y + d * P[3].y};
  }
  return o;
}

std::vector<Point> tangents(const std::vector<Point>& p) {  // np.gradient, normalised
  const size_t n = p.size();
  std::vector<Point> t(n);
  for (size_t i = 0; i < n; ++i) {
    Point g;
    if (i == 0)
      g = Point{p[1].x - p[0].x, p[1].y - p[0].y};
    else if (i == n - 1)
      g = Point{p[i].x - p[i - 1].x, p[i].y - p[i - 1].y};
    else
      g = Point{0.5 * (p[i + 1].x - p[i - 1].x), 0.5 * (p[i + 1].y - p[i - 1].y)};
    const double l = std::hypot(g.x, g.y);
    t[i] = Point{g.x / l, g.y / l};
  }
  return t;
}

std::vector<Point> offset(const std::vector<Point>& p, double o) {
  const std::vector<Point> t = tangents(p);
  std::vector<Point> q(p.size());
  for (size_t i = 0; i < p.size(); ++i) q[i] = Point{p[i].x - o * t[i].y, p[i].y + o * t[i].x};
  return q;
}

double nearestDist(const std::vector<Point>& poly, double x, double y, size_t* idx = nullptr) {
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < poly.size(); ++i) {
    const double d = (poly[i].x - x) * (poly[i].x - x) + (poly[i].y - y) * (poly[i].y - y);
    if (d < best) {
      best = d;
      if (idx) *idx = i;
    }
  }
  return std::sqrt(best);
}

Img polyCov(const std::vector<Point>& poly, double w) {
  Img c(N * N, 0.0);
  if (poly.empty()) return c;
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) c[y * N + x] = std::min(std::max(w * 0.5 + 0.5 - nearestDist(poly, x, y), 0.0), 1.0);
  return c;
}

Img disc(Point m, double r) {
  Img c(N * N);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) c[y * N + x] = std::min(std::max(r + 0.5 - std::hypot(x - m.x, y - m.y), 0.0), 1.0);
  return c;
}

Point nearestToX(const std::vector<Point>& p, double x) {
  size_t b = 0;
  for (size_t i = 1; i < p.size(); ++i)
    if (std::fabs(p[i].x - x) < std::fabs(p[b].x - x)) b = i;
  return p[b];
}

double distToSet(const std::vector<Point>& pts, double x, double y) { return nearestDist(pts, x, y); }

}  // namespace

const std::vector<std::string>& sceneNames() {
  static const std::vector<std::string> n = {"single", "pair", "thin", "thick", "light", "crossing",
                                             "glove", "frame", "brownframe", "mole", "texture", "combo"};
  return n;
}

Scene makeScene(const std::string& name, int seed) {
  Rng r((uint64_t)seed + 1);
  Scene S;
  std::vector<double> bx(N);
  for (int y = 0; y < N; ++y) bx[y] = 30 + 3 * std::sin(y / 17.0 + seed);
  Img head(N * N);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) head[y * N + x] = std::min(std::max(bx[y] - x + 0.5, 0.0), 1.0);
  struct Dis {
    Img cov;
    double col[3];
  };
  std::vector<Strand> strands;
  std::vector<Dis> distract;
  const double y0 = 56 + r.normal(4), bend = r.normal(10);
  Point base[4] = {{26, y0}, {45, y0 + bend * 0.3}, {60, y0 + bend}, {78, y0 + 1.5 * bend}};
  base[1].y += r.normal(3);
  base[2].y += r.normal(3);
  base[3].y += r.normal(4);
  const std::vector<Point> p = bezier(base);
  const bool distractScene = name == "glove" || name == "frame" || name == "brownframe" || name == "mole" ||
                             name == "texture" || name == "combo";
  if (name == "single") {
    strands.push_back(Strand{p, 3.0});
  } else if (name == "pair") {
    strands.push_back(Strand{offset(p, -2.75), 3.5});
    strands.push_back(Strand{offset(p, 2.75), 3.5});
  } else if (name == "thin") {
    strands.push_back(Strand{offset(p, -4), 1.5});
    strands.push_back(Strand{offset(p, 4), 2.0});
  } else if (name == "thick") {
    strands.push_back(Strand{p, 6.0});
  } else if (distractScene) {
    if (name == "combo") {
      strands.push_back(Strand{offset(p, -2.5), 3.0});
      strands.push_back(Strand{offset(p, 2.5), 4.0});
    } else {
      strands.push_back(Strand{p, 3.0});
    }
    if (name == "glove" || name == "combo") {
      std::vector<Point> edge = offset(p, 6.0 + (name == "combo" ? 2.5 : 0.0));
      const Point e1 = edge.back(), e30 = edge[edge.size() - 30];
      for (int i = 0; i < 60; ++i) {
        const double f = 3.0 * i / 59.0;
        edge.push_back(Point{e1.x + (e1.x - e30.x) * f, e1.y + (e1.y - e30.y) * f});
      }
      const std::vector<Point> t = tangents(edge);
      Dis d;
      d.cov.assign(N * N, 0.0);
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
          size_t i = 0;
          nearestDist(edge, x, y, &i);
          const double side = (x - edge[i].x) * (-t[i].y) + (y - edge[i].y) * t[i].x;
          d.cov[y * N + x] = std::min(std::max(side + 0.5, 0.0), 1.0) * (x > 30 ? 1.0 : 0.0);
        }
      std::copy(BLACK, BLACK + 3, d.col);
      distract.push_back(d);
    }
    if (name == "frame" || name == "brownframe") {
      const double a = r.uniform(60, 120) * kPi / 180.0;
      const Point tip = nearestToX(p, 53);
      std::vector<Point> line(400);
      for (int i = 0; i < 400; ++i) {
        const double s = -80 + 160.0 * i / 399.0;
        line[i] = Point{tip.x + s * std::cos(a), tip.y + s * std::sin(a)};
      }
      Dis d;
      d.cov = polyCov(line, name == "frame" ? 8.0 : 6.5);
      for (int y = 0; y < N; ++y)
        for (int x = 0; x <= 30; ++x) d.cov[y * N + x] = 0.0;
      for (int k = 0; k < 3; ++k) d.col[k] = name == "frame" ? BLACK[k] : HAIR[k] * 1.3;
      distract.push_back(d);
    }
    if (name == "mole") {
      const Point tip = nearestToX(p, 53);
      Dis d;
      d.cov = disc(Point{tip.x + 4.0, tip.y + 2.0}, 3.0);
      for (int k = 0; k < 3; ++k) d.col[k] = HAIR[k] * 1.6;
      distract.push_back(d);
      const Point mid = p[p.size() / 2];
      d.cov = disc(Point{mid.x + 2.0, mid.y + 7.0}, 2.5);
      distract.push_back(d);
    }
  } else if (name == "light") {
    strands.push_back(Strand{offset(p, -2.75), 3.0});
    strands.push_back(Strand{offset(p, 2.75), 3.0});
  } else if (name == "crossing") {
    strands.push_back(Strand{p, 3.0});
    const Point c = p[p.size() / 2];
    const double a = (40 + r.normal(5)) * kPi / 180.0;
    std::vector<Point> q;
    for (int i = 0; i < 300; ++i) {
      const double s = -30 + 60.0 * i / 299.0;
      const Point v{c.x + s * std::cos(a), c.y + s * std::sin(a)};
      if (v.x > 31) q.push_back(v);
    }
    strands.push_back(Strand{q, 2.5});
  }
  // creases: thin darker skin lines (texture, combo)
  if (name == "texture" || name == "combo") {
    const std::vector<Point>& p0 = strands[0].poly;
    for (int k = 0; k < 5; ++k) {
      const Point c0 = p0[r.integer((int)p0.size() / 4, (int)p0.size() / 2)];
      const Point c{c0.x + r.normal(6), c0.y + r.normal(6)};
      const double a = r.uniform(0, kPi);
      std::vector<Point> q(100);
      for (int i = 0; i < 100; ++i) {
        const double s = -12 + 24.0 * i / 99.0;
        q[i] = Point{c.x + s * std::cos(a), c.y + s * std::sin(a)};
      }
      Img cr = polyCov(q, 1.5);
      for (double& v : cr) v *= 0.35;
      S.creases.push_back(cr);
    }
  }
  // compose (linear light)
  Img noise(N * N);
  for (double& v : noise) v = r.normal(1.0);
  Img low = gauss(noise, 12);
  {
    double m = 0, s = 0;
    for (double v : low) m += v;
    m /= low.size();
    for (double v : low) s += (v - m) * (v - m);
    s = std::sqrt(s / low.size());
    for (double& v : low) v /= s + 1e-9;
  }
  Img pr(N * N);
  for (double& v : pr) v = r.uniform() < 0.01 ? 1.0 : 0.0;
  Img pores = gauss(pr, 0.8);
  std::vector<double> img(N * N * 3);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      const int i = y * N + x;
      const double po = std::min(std::max(pores[i] * 3, 0.0), 1.0);
      for (int k = 0; k < 3; ++k) {
        double v = SKIN[k] * (1 + 0.06 * low[i]) * (1 - 0.15 * ((double)y / N)) * (1 - 0.12 * po);
        static const double CR[3] = {0.72, 0.62, 0.6};
        for (const Img& cr : S.creases) v = v * (1 - cr[i]) + v * CR[k] * cr[i];
        img[i * 3 + k] = v;
      }
    }
  std::vector<unsigned char> dmask(N * N, 0);
  for (const Dis& d : distract)
    for (int i = 0; i < N * N; ++i) {
      for (int k = 0; k < 3; ++k) img[i * 3 + k] = img[i * 3 + k] * (1 - d.cov[i]) + d.col[k] * d.cov[i];
      if (d.cov[i] > 0.2) dmask[i] = 1;
    }
  S.gt.assign(N * N, 0.0);
  const bool cutScene = name == "glove" || name == "frame" || name == "brownframe" || name == "mole" || name == "combo";
  const double cut = cutScene ? 52 : 999;
  for (const Strand& st : strands) {
    std::vector<Point> poly;
    for (const Point& q : st.poly) {
      const int yi = std::min(std::max((int)q.y, 0), N - 1);
      if (q.x > bx[yi] - 2 && q.x < cut) poly.push_back(q);
    }
    const Img cov = polyCov(poly, st.width);
    for (int i = 0; i < N * N; ++i) {
      for (int k = 0; k < 3; ++k) {
        const double hc = name == "light" ? HAIR[k] * 6 : HAIR[k];
        img[i * 3 + k] = img[i * 3 + k] * (1 - cov[i]) + hc * cov[i];
      }
      S.gt[i] = 1 - (1 - S.gt[i]) * (1 - cov[i]);
    }
    S.polys.push_back(Strand{poly, st.width});
  }
  for (int i = 0; i < N * N; ++i)
    for (int k = 0; k < 3; ++k) img[i * 3 + k] = img[i * 3 + k] * (1 - head[i]) + HAIR[k] * (1 + 0.3 * low[i]) * head[i];
  for (int k = 0; k < 3; ++k) {
    Img ch(N * N);
    for (int i = 0; i < N * N; ++i) ch[i] = img[i * 3 + k];
    ch = gauss(ch, 0.6);
    for (int i = 0; i < N * N; ++i) img[i * 3 + k] = ch[i];
  }
  for (double& v : img) v = std::max(v * (1 + r.normal(0.03)), 1e-4);
  for (int i = 0; i < N * N; ++i) S.gt[i] *= 1 - head[i];
  // region: strands within 30 px of the head, dilated 8 px (cross), feathered
  std::vector<int> dist(N * N, 1 << 20);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x)
      if (S.gt[y * N + x] > 0.3 && x < bx[y] + 30 && x > bx[y]) dist[y * N + x] = 0;
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      int& v = dist[y * N + x];
      if (x) v = std::min(v, dist[y * N + x - 1] + 1);
      if (y) v = std::min(v, dist[(y - 1) * N + x] + 1);
    }
  for (int y = N - 1; y >= 0; --y)
    for (int x = N - 1; x >= 0; --x) {
      int& v = dist[y * N + x];
      if (x < N - 1) v = std::min(v, dist[y * N + x + 1] + 1);
      if (y < N - 1) v = std::min(v, dist[(y + 1) * N + x] + 1);
    }
  Img reg(N * N);
  for (int i = 0; i < N * N; ++i) reg[i] = dist[i] <= 8 ? 1.0 : 0.0;
  reg = gauss(reg, 1.0);
  S.rgb.resize(N * N * 3);
  for (int i = 0; i < N * N * 3; ++i) S.rgb[i] = (float)img[i];
  S.alpha.resize(N * N);
  S.region.resize(N * N);
  S.dmask.resize(N * N);
  for (int i = 0; i < N * N; ++i) {
    S.alpha[i] = (float)head[i];
    S.region[i] = (float)(reg[i] > 0.5 ? 1.0 : reg[i] * 2);
    S.dmask[i] = dmask[i] && S.gt[i] < 0.05;
  }
  return S;
}

Metrics metrics(const Scene& S, const std::vector<float>& A, const std::vector<Piece>& pieces) {
  const int W = S.w, H = S.h;
  Metrics m;
  std::vector<Point> trace;
  for (const Piece& pc : pieces) trace.insert(trace.end(), pc.polyline.begin(), pc.polyline.end());
  std::vector<double> recs;
  std::vector<Point> allc;
  for (const Strand& st : S.polys) {
    std::vector<Point> P;
    for (const Point& q : st.poly) {
      const int ix = std::min(std::max((int)std::nearbyint(q.x), 0), W - 1);
      const int iy = std::min(std::max((int)std::nearbyint(q.y), 0), H - 1);
      if (S.region[iy * W + ix] > 0.5 && S.alpha[iy * W + ix] < 0.5) P.push_back(q);
    }
    allc.insert(allc.end(), st.poly.begin(), st.poly.end());
    if (P.size() < 5) continue;
    double rec = 0.0;
    if (!trace.empty()) {
      int k = 0;
      for (const Point& q : P) k += distToSet(trace, q.x, q.y) <= 1.5;
      rec = (double)k / P.size();
    }
    recs.push_back(rec);
    if (rec < 0.1)
      ++m.missed;
    else if (rec < 0.8)
      ++m.early;
  }
  for (const Piece& pc : pieces) {
    int run = 0;
    bool ev = false;
    for (const Point& q : pc.polyline) {
      run = distToSet(allc, q.x, q.y) > 2.5 ? run + 1 : 0;
      if (run >= 4) ev = true;
    }
    m.wander += ev;
  }
  // distance to the nearest truth pixel (gt >= 0.05), brute force
  std::vector<int> feat;
  for (int i = 0; i < W * H; ++i)
    if (S.gt[i] >= 0.05) feat.push_back(i);
  double gsum = 0, fals = 0, distr = 0, sad = 0;
  for (int i = 0; i < W * H; ++i) {
    const int x = i % W, y = i / W;
    double d2 = std::numeric_limits<double>::infinity();
    for (int f : feat) d2 = std::min(d2, (double)((f % W - x) * (f % W - x) + (f / W - y) * (f / W - y)));
    const bool far = d2 > 1.0;
    bool dm = S.dmask[i] != 0;
    for (const auto& cr : S.creases) dm = dm || (cr[i] > 0.1 && far);
    const double gts = S.gt[i] * (S.region[i] > 0.05 ? 1.0 : 0.0);
    gsum += gts;
    if (far) fals += A[i];
    if (dm) distr += A[i];
    sad += std::fabs(A[i] - gts);
  }
  gsum = std::max(gsum, 1e-6);
  m.fals = fals / gsum;
  m.distr = distr / gsum;
  m.sad = sad / gsum;
  if (recs.empty()) {
    m.rec = std::numeric_limits<double>::quiet_NaN();
  } else {
    double s = 0;
    for (double v : recs) s += v;
    m.rec = s / recs.size();
  }
  return m;
}

Metrics aggregate(const std::vector<Metrics>& ms) {
  Metrics a;
  int nr = 0;
  for (const Metrics& m : ms) {
    if (!std::isnan(m.rec)) {
      a.rec += m.rec;
      ++nr;
    }
    a.fals += m.fals;
    a.distr += m.distr;
    a.sad += m.sad;
    a.wander += m.wander;
    a.early += m.early;
    a.missed += m.missed;
  }
  a.rec = nr ? a.rec / nr : std::numeric_limits<double>::quiet_NaN();
  if (!ms.empty()) {
    a.fals /= ms.size();
    a.distr /= ms.size();
    a.sad /= ms.size();
  }
  return a;
}

std::vector<Metrics> evaluate(const TracerParams& p, int seeds, Metrics& mean) {
  std::vector<Metrics> rows;
  for (const std::string& n : sceneNames()) {
    std::vector<Metrics> acc;
    for (int k = 0; k < seeds; ++k) {
      const Scene S = makeScene(n, k);
      TracerInput in;
      in.rgb = S.rgb.data();
      in.base = S.alpha.data();
      in.region = S.region.data();
      in.width = S.w;
      in.height = S.h;
      in.frameHeight = 1080;
      const TracerResult r = traceStrands(in, p);
      acc.push_back(metrics(S, r.alpha, r.pieces));
    }
    rows.push_back(aggregate(acc));
  }
  mean = aggregate(rows);
  return rows;
}

}  // namespace scenes
}  // namespace tracer
}  // namespace mr
