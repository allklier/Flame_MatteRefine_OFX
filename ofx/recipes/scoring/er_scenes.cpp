#include "er_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace er {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clamp01(double v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

uint64_t mix(uint64_t z) {  // splitmix64 finaliser
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
double hash01(int x, int y, unsigned seed) {
  return double(mix((uint64_t(uint32_t(x)) << 32) ^ uint64_t(uint32_t(y)) ^ (uint64_t(seed) << 17)) >> 11) /
         double(1ull << 53);
}
// Value noise in 0..1, cell size `cell` px, smoothstep interpolation.
double vnoise(double x, double y, double cell, unsigned seed) {
  const double fx = x / cell, fy = y / cell;
  const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
  double tx = fx - ix, ty = fy - iy;
  tx = tx * tx * (3 - 2 * tx);
  ty = ty * ty * (3 - 2 * ty);
  const double a = hash01(ix, iy, seed), b = hash01(ix + 1, iy, seed), c = hash01(ix, iy + 1, seed),
               d = hash01(ix + 1, iy + 1, seed);
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

struct P {
  double x, y;
};
using Poly = std::vector<P>;

Poly bezier(const P c[4], int n) {
  Poly o(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double t = double(i) / (n - 1), u = 1 - t;
    const double a = u * u * u, b = 3 * u * u * t, cc = 3 * u * t * t, d = t * t * t;
    o[size_t(i)] = P{a * c[0].x + b * c[1].x + cc * c[2].x + d * c[3].x, a * c[0].y + b * c[1].y + cc * c[2].y + d * c[3].y};
  }
  return o;
}
P tangentAt(const Poly& p, size_t i) {
  const size_t n = p.size(), a = i ? i - 1 : 0, b = i + 1 < n ? i + 1 : n - 1;
  double tx = p[b].x - p[a].x, ty = p[b].y - p[a].y;
  const double l = std::max(std::hypot(tx, ty), 1e-12);
  return P{tx / l, ty / l};
}
Poly offset(const Poly& p, double off) {  // along the left normal (-ty, tx)
  Poly q(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    const P t = tangentAt(p, i);
    q[i] = P{p[i].x - off * t.y, p[i].y + off * t.x};
  }
  return q;
}
double distPoly(const Poly& p, double x, double y) {
  double best = 1e30;
  for (size_t i = 0; i + 1 < p.size(); ++i) {
    const double vx = p[i + 1].x - p[i].x, vy = p[i + 1].y - p[i].y, px = x - p[i].x, py = y - p[i].y;
    const double l2 = vx * vx + vy * vy;
    const double t = l2 > 0 ? clamp01((px * vx + py * vy) / l2) : 0.0;
    best = std::min(best, (px - t * vx) * (px - t * vx) + (py - t * vy) * (py - t * vy));
  }
  return std::sqrt(best);
}
// out = max(out, value(distance)) over the polyline's box grown by reach.
template <class F>
void paint(std::vector<float>& out, int w, int h, const Poly& p, double reach, F value) {
  double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
  for (const P& q : p) x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
  const int ix0 = std::max(int(std::floor(x0 - reach)), 0), iy0 = std::max(int(std::floor(y0 - reach)), 0);
  const int ix1 = std::min(int(std::ceil(x1 + reach)), w - 1), iy1 = std::min(int(std::ceil(y1 + reach)), h - 1);
  for (int y = iy0; y <= iy1; ++y)
    for (int x = ix0; x <= ix1; ++x) {
      const double v = value(distPoly(p, x, y));
      float& o = out[size_t(y) * w + x];
      if (v > o) o = float(v);
    }
}
std::vector<float> strokeCov(int w, int h, const Poly& p, double width) {
  std::vector<float> c(size_t(w) * h, 0.0f);
  paint(c, w, h, p, width, [&](double d) { return clamp01(width * 0.5 + 0.5 - d); });
  return c;
}
double sdEllipse(double x, double y, double rx, double ry) {  // > 0 inside, first order
  const double q = std::hypot(x / rx, y / ry);
  if (q < 1e-9) return std::min(rx, ry);
  const double g = std::hypot(x / (rx * rx), y / (ry * ry)) / q;
  return (1.0 - q) / g;
}
Poly moved(const Poly& p, const synth::Affine& m) {
  Poly q(p.size());
  for (size_t i = 0; i < p.size(); ++i) m.apply(p[i].x, p[i].y, &q[i].x, &q[i].y);
  return q;
}
Poly line(P c, double angle, double from, double to, int n = 200) {
  Poly q(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double s = from + (to - from) * i / (n - 1);
    q[size_t(i)] = P{c.x + s * std::cos(angle), c.y + s * std::sin(angle)};
  }
  return q;
}

}  // namespace

const char* lookName(int look) {
  static const char* n[] = {"dark_skin", "dark_texture", "dark_shadow", "light_skin"};
  return look >= 0 && look < kLookCount ? n[look] : "?";
}
const char* distractorName(int d) {
  static const char* n[] = {"none", "glove", "glasses", "mole", "crossing", "tipband"};
  return d >= 0 && d < kDistractorCount ? n[d] : "?";
}

std::string SceneSpec::name() const {
  char b[160];
  std::snprintf(b, sizeof b, "%s_w%.1f_m%.0f_%s_j%.2f%s%s%s_s%u", lookName(look), width, marginPx,
                distractorName(distractor), jitterPx, dipFrame >= 0 ? "_dip" : "", spuriousFrame >= 0 ? "_spur" : "",
                detachPx > 0 ? "_detached" : "", seed);
  return b;
}

ScoreSeq makeScoreSeq(const SceneSpec& sp) {
  ScoreSeq S;
  S.spec = sp;
  const int w = sp.w, h = sp.h;
  S.w = w;
  S.h = h;
  S.frames = sp.frames;
  const size_t n = size_t(w) * h;
  const double k = h / 562.0;
  const double hcx = w / 2.0 - 50 * k, hcy = h / 2.0, rx = 120 * k, ry = 160 * k;
  // Colours (linear, the research's look values, ofx/ml/research/vitmatte_thin_hair/exp/synth.py).
  const double SKIN[3] = {0.36, 0.22, 0.16};
  const double DARK[3] = {0.035, 0.022, 0.015}, LIGHT[3] = {0.30, 0.20, 0.10}, BLACK[3] = {0.012, 0.012, 0.013};
  const double* HAIR = sp.look == kLightSkin ? LIGHT : DARK;
  const double bgGain = sp.look == kDarkShadow ? 0.15 : 1.0;
  const double fineAmp = sp.look == kDarkTexture ? 0.12 : 0.03;
  // Motion.
  for (int t = 0; t < sp.frames; ++t) {
    const double ang = 0.8 * t * kPi / 180, sk = 0.01 * t, ca = std::cos(ang), sa = std::sin(ang);
    synth::Affine m;
    m.a = ca;
    m.b = ca * sk - sa;
    m.c = sa;
    m.d = sa * sk + ca;
    m.tx = hcx + 1.8 * k * t - (m.a * hcx + m.b * hcy);
    m.ty = hcy - 0.5 * k * t - (m.c * hcx + m.d * hcy);
    S.motion.push_back(m);
  }
  // Locks in frame-0 coordinates (as makeLocksSequence: root 4 px inside the edge, along the outward normal).
  struct Lock {
    Poly mid;
    std::vector<Poly> strands;
    double halfWidth = 0;
  };
  auto makeLock = [&](double phiDeg, double len, double bend, double detach) {
    const double phi = phiDeg * kPi / 180;
    const P E{hcx + rx * std::cos(phi), hcy + ry * std::sin(phi)};
    double nx = std::cos(phi) / rx, ny = std::sin(phi) / ry;
    const double l = std::hypot(nx, ny);
    nx /= l;
    ny /= l;
    const double tx = -ny, ty = nx;
    const double r0 = detach > 0 ? detach : -4.0;  // the root: 4 px inside the edge, or detached outside it
    const P c[4] = {{E.x + r0 * nx, E.y + r0 * ny},
                    {E.x + (r0 + len / 3) * nx, E.y + (r0 + len / 3) * ny},
                    {E.x + (r0 + 2 * len / 3) * nx + 0.4 * bend * tx, E.y + (r0 + 2 * len / 3) * ny + 0.4 * bend * ty},
                    {E.x + (r0 + len) * nx + bend * tx, E.y + (r0 + len) * ny + bend * ty}};
    Lock L;
    L.mid = bezier(c, 240);
    return L;
  };
  const double W = sp.width;
  Lock locks[2] = {makeLock(-25, 42 * k, 14 * k, sp.detachPx), makeLock(22, 38 * k, -12 * k, 0.0)};
  locks[0].strands.push_back(locks[0].mid);
  locks[0].halfWidth = W / 2;
  const double gap = 2.0;
  locks[1].strands.push_back(offset(locks[1].mid, -(W + gap) / 2));
  locks[1].strands.push_back(offset(locks[1].mid, (W + gap) / 2));
  locks[1].halfWidth = W + gap / 2;
  // Distractors in frame-0 coordinates, each a coverage polyline + width, or a filled glove side.
  const Poly& L0 = locks[0].mid;
  const P tip = L0.back(), tipT = tangentAt(L0, L0.size() - 1);
  const P mid0 = L0[L0.size() / 2], midT = tangentAt(L0, L0.size() / 2);
  struct Dis {
    Poly p;
    double width = 0;  // > 0: a stroke; 0: the glove (a filled side of p)
    const double* col = nullptr;
    double colGain = 1.0;
    double radius = 0;  // > 0: a disc at p[0]
  };
  std::vector<Dis> dis;
  Poly crossing;
  const double crossW = std::max(1.5, 0.8 * W);
  // The side of lock 0 away from lock 1: sign of the left normal pointing away from lock 1's root.
  const P l1 = locks[1].mid.front();
  const double sideSign = ((-midT.y) * (l1.x - mid0.x) + midT.x * (l1.y - mid0.y)) > 0 ? -1.0 : 1.0;
  switch (sp.distractor) {
    case kGlove: {  // a wide dark area whose edge runs 3 px from the strand and continues past its tip
      Poly e = offset(L0, sideSign * (W / 2 + 3));
      const P e1 = e.back(), e30 = e[e.size() - 30];
      for (int i = 1; i <= 60; ++i) {
        const double f = 3.0 * i / 60.0;
        e.push_back(P{e1.x + (e1.x - e30.x) * f, e1.y + (e1.y - e30.y) * f});
      }
      Dis d;
      d.p = e;
      d.col = BLACK;
      dis.push_back(d);
      break;
    }
    case kGlasses:  // an 8 px black band through the tip, across the strand
    case kTipBand: {  // a 6.5 px band in a hair colour touching the tip
      const double a = std::atan2(tipT.y, tipT.x) + (sp.distractor == kGlasses ? 80 : 100) * kPi / 180;
      Dis d;
      d.p = line(tip, a, -60 * k, 60 * k);
      d.width = sp.distractor == kGlasses ? 8.0 : 6.5;
      d.col = sp.distractor == kGlasses ? BLACK : HAIR;
      d.colGain = sp.distractor == kGlasses ? 1.0 : 1.3;
      dis.push_back(d);
      break;
    }
    case kMole: {  // two dark dots, one past the tip, one beside the middle
      Dis d;
      d.col = HAIR;
      d.colGain = 1.6;
      d.radius = 3.0;
      d.p = Poly{P{tip.x + 4 * tipT.x - 2 * tipT.y * sideSign, tip.y + 4 * tipT.y + 2 * tipT.x * sideSign}};
      dis.push_back(d);
      d.radius = 2.5;
      d.p = Poly{P{mid0.x - (W / 2 + 5.5) * midT.y * sideSign, mid0.y + (W / 2 + 5.5) * midT.x * sideSign}};
      dis.push_back(d);
      break;
    }
    case kCrossing: {  // a hair strand crossing lock 0 at 40 deg (it IS hair: in the truth)
      const double a = std::atan2(midT.y, midT.x) + 40 * kPi / 180;
      crossing = line(mid0, a, -24 * k, 24 * k);
      break;
    }
    default:
      break;
  }
  // The spurious stroke (as makeLocksSequence's): next to lock 1, 9.5 px from its middle, 35-90 % of its length.
  Poly spur;
  {
    const Poly o = offset(locks[1].mid, 9.5 + W);
    for (size_t i = 0; i < o.size(); ++i)
      if (i >= o.size() * 35 / 100 && i <= o.size() * 90 / 100) spur.push_back(o[i]);
  }
  // Static background noise (does not move): low-frequency + fine skin texture.
  std::vector<float> bgTex(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double lo = 2 * vnoise(x, y, 12 * k, sp.seed + 51) - 1;
      const double fine = 2 * vnoise(x, y, 1.3, sp.seed + 52) - 1;
      bgTex[size_t(y) * w + x] = float((1 + 0.12 * (double(x) / w - 0.5)) * (1 + 0.08 * lo) * (1 + 1.7 * fineAmp * fine));
    }
  std::mt19937 rng(sp.seed * 2654435761u + 17u);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  for (int t = 0; t < sp.frames; ++t) {
    const synth::Affine& m = S.motion[size_t(t)];
    const synth::Affine inv = m.inverse();
    double jph[3];
    for (double& p : jph) p = 2 * kPi * u01(rng);
    std::vector<float> head(n), base(n), region(n, 0.0f), cov(n, 0.0f), pic(n, 0.0f), cross;
    std::vector<float> disCov(n, 0.0f);
    std::vector<int> disIdx(n, -1);
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        const size_t i = size_t(py) * w + px;
        double qx, qy;
        inv.apply(px, py, &qx, &qy);
        const double d = sdEllipse(qx - hcx, qy - hcy, rx, ry);
        const double phi = std::atan2((qy - hcy) / ry, (qx - hcx) / rx);
        const double j = sp.jitterPx * (0.5 * std::sin(3 * phi + jph[0]) + 0.3 * std::sin(7 * phi + jph[1]) +
                                        0.2 * std::sin(13 * phi + jph[2]));
        head[i] = float(clamp01(d + 0.5));
        base[i] = float(clamp01(d + j + 0.5));
      }
    for (int L = 0; L < 2; ++L) {
      const double contrast = (L == 0 && t == sp.dipFrame) ? 0.20 : 1.0;
      for (const Poly& st : locks[L].strands) {
        const std::vector<float> c = strokeCov(w, h, moved(st, m), W);
        for (size_t i = 0; i < n; ++i) {
          cov[i] = float(1.0 - (1.0 - cov[i]) * (1.0 - c[i]));
          pic[i] = float(1.0 - (1.0 - pic[i]) * (1.0 - contrast * c[i]));
        }
      }
      const double R = 9.0 + locks[L].halfWidth + sp.marginPx, F = 5 * k;
      paint(region, w, h, moved(locks[L].mid, m), R + F, [&](double dd) { return clamp01((R - dd) / F + 0.5); });
    }
    if (!crossing.empty()) {
      cross = strokeCov(w, h, moved(crossing, m), crossW);
      for (size_t i = 0; i < n; ++i) {
        cov[i] = float(1.0 - (1.0 - cov[i]) * (1.0 - cross[i]));
        pic[i] = float(1.0 - (1.0 - pic[i]) * (1.0 - cross[i]));
      }
    }
    for (size_t di = 0; di < dis.size(); ++di) {
      const Dis& d = dis[di];
      const Poly p = moved(d.p, m);
      std::vector<float> c(n, 0.0f);
      if (d.radius > 0) {
        paint(c, w, h, p, d.radius + 1, [&](double dd) { return clamp01(d.radius + 0.5 - dd); });
      } else if (d.width > 0) {
        c = strokeCov(w, h, p, d.width);
      } else {  // glove: the far side of the edge polyline, up to 40k px deep
        const double depth = 40 * k;
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        for (const P& q : p) x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
        const int bx0 = std::max(0, int(x0 - depth - 2)), by0 = std::max(0, int(y0 - depth - 2));
        const int bx1 = std::min(w - 1, int(x1 + depth + 2)), by1 = std::min(h - 1, int(y1 + depth + 2));
        for (int y = by0; y <= by1; ++y)
          for (int x = bx0; x <= bx1; ++x) {
            // nearest point of the edge and its normal
            double best = 1e30;
            size_t bi = 0;
            for (size_t i = 0; i < p.size(); i += 2) {
              const double dd = (x - p[i].x) * (x - p[i].x) + (y - p[i].y) * (y - p[i].y);
              if (dd < best) best = dd, bi = i;
            }
            if (best > (depth + 2) * (depth + 2)) continue;
            const P tg = tangentAt(p, bi);
            const double side = sideSign * ((x - p[bi].x) * (-tg.y) + (y - p[bi].y) * tg.x);
            c[size_t(y) * w + x] = float(clamp01(side + 0.5) * clamp01(depth - side));
          }
      }
      for (size_t i = 0; i < n; ++i)
        if (c[i] > disCov[i]) disCov[i] = c[i], disIdx[i] = int(di);
    }
    std::vector<uint8_t> spurMask;
    if (t == sp.spuriousFrame) {
      const std::vector<float> c = strokeCov(w, h, moved(spur, m), 2.5);
      spurMask.assign(n, 0);
      for (size_t i = 0; i < n; ++i) {
        pic[i] = float(1.0 - (1.0 - pic[i]) * (1.0 - c[i]));
        spurMask[i] = c[i] > 0.2f ? 1 : 0;
      }
    }
    // Compose: background, distractors, strands, head on top. Sensor noise per frame.
    const unsigned fs = sp.seed * 7919u + 5000u + unsigned(t) * 104729u;
    std::vector<float> rgb(n * 3);
    std::vector<uint8_t> dmask(n, 0);
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        const size_t i = size_t(py) * w + px;
        double qx, qy;
        inv.apply(px, py, &qx, &qy);
        const double hm = 2 * vnoise(qx, qy, 4, sp.seed + 53) - 1;
        for (int c = 0; c < 3; ++c) {
          double v = SKIN[c] * bgGain * bgTex[i] * (1 - 0.15 * double(py) / h);
          if (disIdx[i] >= 0) {
            const Dis& d = dis[size_t(disIdx[i])];
            v = v * (1 - disCov[i]) + d.col[c] * d.colGain * disCov[i];
          }
          v = v * (1 - pic[i]) + HAIR[c] * pic[i];
          v = v * (1 - head[i]) + HAIR[c] * (1 + 0.25 * hm) * head[i];
          const double sigma = 0.004 * std::sqrt(std::max(v, 1e-4) / 0.2);
          v += sigma * 1.7320508 * (2 * hash01(px * 3 + c, py, fs) - 1);
          rgb[i * 3 + size_t(c)] = float(std::max(v, 0.0));
        }
        dmask[i] = disCov[i] > 0.2f && cov[i] < 0.05f && head[i] < 0.5f ? 1 : 0;
      }
    S.rgb.push_back(std::move(rgb));
    S.base.push_back(std::move(base));
    S.region.push_back(std::move(region));
    S.cov.push_back(std::move(cov));
    S.crossCov.push_back(std::move(cross));
    S.head.push_back(std::move(head));
    S.distract.push_back(std::move(dmask));
    S.spurious.push_back(std::move(spurMask));
  }
  return S;
}

ScoreSeq downsample2(const ScoreSeq& s) {
  ScoreSeq o;
  o.spec = s.spec;
  o.w = s.w / 2;
  o.h = s.h / 2;
  o.frames = s.frames;
  for (const synth::Affine& m : s.motion) {  // frame-0 px -> frame-t px at half size: scale the translation
    synth::Affine q = m;
    q.tx *= 0.5;
    q.ty *= 0.5;
    o.motion.push_back(q);
  }
  const size_t n = size_t(o.w) * o.h;
  auto plane = [&](const std::vector<float>& a) {
    std::vector<float> r;
    if (a.empty()) return r;
    r.resize(n);
    for (int y = 0; y < o.h; ++y)
      for (int x = 0; x < o.w; ++x) {
        const size_t i = size_t(2 * y) * s.w + 2 * x;
        r[size_t(y) * o.w + x] = 0.25f * (a[i] + a[i + 1] + a[i + s.w] + a[i + s.w + 1]);
      }
    return r;
  };
  auto mask = [&](const std::vector<uint8_t>& a) {
    std::vector<uint8_t> r;
    if (a.empty()) return r;
    r.resize(n);
    for (int y = 0; y < o.h; ++y)
      for (int x = 0; x < o.w; ++x) {
        const size_t i = size_t(2 * y) * s.w + 2 * x;
        r[size_t(y) * o.w + x] = (a[i] | a[i + 1] | a[i + s.w] | a[i + s.w + 1]) ? 1 : 0;
      }
    return r;
  };
  for (int t = 0; t < s.frames; ++t) {
    std::vector<float> rgb(n * 3);
    const std::vector<float>& a = s.rgb[size_t(t)];
    for (int y = 0; y < o.h; ++y)
      for (int x = 0; x < o.w; ++x)
        for (int c = 0; c < 3; ++c) {
          const size_t i = (size_t(2 * y) * s.w + 2 * x) * 3 + c, r = size_t(s.w) * 3;
          rgb[(size_t(y) * o.w + x) * 3 + c] = 0.25f * (a[i] + a[i + 3] + a[i + r] + a[i + r + 3]);
        }
    o.rgb.push_back(std::move(rgb));
    o.base.push_back(plane(s.base[size_t(t)]));
    o.region.push_back(plane(s.region[size_t(t)]));
    o.cov.push_back(plane(s.cov[size_t(t)]));
    o.crossCov.push_back(plane(s.crossCov[size_t(t)]));
    o.head.push_back(plane(s.head[size_t(t)]));
    o.distract.push_back(mask(s.distract[size_t(t)]));
    o.spurious.push_back(mask(s.spurious[size_t(t)]));
  }
  return o;
}

ScoreSeq fromLocksSequence(const synth::LocksSequence& L) {
  ScoreSeq S;
  S.spec.w = L.w;
  S.spec.h = L.h;
  S.spec.frames = L.frames;
  S.spec.dipFrame = L.flicker.dipFrame;
  S.spec.spuriousFrame = L.flicker.spuriousFrame;
  S.w = L.w;
  S.h = L.h;
  S.frames = L.frames;
  S.motion = L.motion;
  S.rgb = L.rgb;
  S.base = L.base;
  S.region = L.region;
  S.spurious = L.spurious;
  const size_t n = size_t(L.w) * L.h;
  for (int t = 0; t < L.frames; ++t) {
    std::vector<float> cov(n, 0.0f), head(n);
    for (size_t i = 0; i < n; ++i) {
      head[i] = float(clamp01(L.headDistance[size_t(t)][i] + 0.5));
      const float b = L.base[size_t(t)][i];
      cov[i] = b < 1.0f ? std::min(1.0f, L.strandTruth[size_t(t)][i] / (1.0f - b)) : 0.0f;
    }
    S.cov.push_back(std::move(cov));
    S.head.push_back(std::move(head));
    S.crossCov.emplace_back();
    S.distract.emplace_back(n, uint8_t(0));
  }
  return S;
}

}  // namespace er
