// Dense pyramidal Lucas-Kanade with a weighted data term: see flow.h.
#include "flow.h"

#include <algorithm>
#include <cmath>

#include "ops/morphology.h"

namespace mr {
namespace ml {

namespace {

// Window sums (2r+1 square, clipped to the frame) of in into out; tmp: w*h scratch. Each output is
// summed in a fixed order: the same values at any thread count.
void boxSum(const float* in, int w, int h, int r, float* out, float* tmp, Workers* wk) {
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* row = in + size_t(y) * w;
      float* o = tmp + size_t(y) * w;
      double s = 0;
      for (int x = 0; x < std::min(r, w); ++x) s += row[x];
      for (int x = 0; x < w; ++x) {
        if (x + r < w) s += row[x + r];
        if (x - r - 1 >= 0) s -= row[x - r - 1];
        o[x] = float(s);
      }
    }
  });
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      float* o = out + size_t(y) * w;
      std::fill(o, o + w, 0.0f);
      for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy) {
        const float* t = tmp + size_t(yy) * w;
        for (int x = 0; x < w; ++x) o[x] += t[x];
      }
    }
  });
}

void gradient(const float* a, int w, int h, float* gx, float* gy, Workers* wk) {
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const int ym = std::max(y - 1, 0), yp = std::min(y + 1, h - 1);
      for (int x = 0; x < w; ++x) {
        const int xm = std::max(x - 1, 0), xp = std::min(x + 1, w - 1);
        const size_t i = size_t(y) * w + x;
        gx[i] = (a[size_t(y) * w + xp] - a[size_t(y) * w + xm]) / float(std::max(xp - xm, 1));
        gy[i] = (a[size_t(yp) * w + x] - a[size_t(ym) * w + x]) / float(std::max(yp - ym, 1));
      }
    }
  });
}

// One pyramid step: out ((w + 1) / 2 x (h + 1) / 2) pixel j = the [1 3 3 1] / 8 binomial (both
// axes, border pixels repeated) centred on input 2j + 0.5: a 2x2 box mean would alias fine
// texture into false motion at the coarse levels (seen: dense LK diverging there).
void halve(const float* in, int w, int h, float* out, Workers* wk) {
  const int ow = (w + 1) / 2, oh = (h + 1) / 2;
  std::vector<float> tmp(size_t(ow) * h);
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* r = in + size_t(y) * w;
      for (int j = 0; j < ow; ++j) {
        auto at = [&](int x) { return r[std::min(std::max(x, 0), w - 1)]; };
        tmp[size_t(y) * ow + j] = (at(2 * j - 1) + 3 * at(2 * j) + 3 * at(2 * j + 1) + at(2 * j + 2)) * 0.125f;
      }
    }
  });
  parallelFor(wk, oh, 8, [&](int y0, int y1) {
    for (int j = y0; j < y1; ++j) {
      const float* r[4];
      for (int k = 0; k < 4; ++k) r[k] = &tmp[size_t(std::min(std::max(2 * j - 1 + k, 0), h - 1)) * ow];
      float* o = out + size_t(j) * ow;
      for (int x = 0; x < ow; ++x) o[x] = (r[0][x] + 3 * r[1][x] + 3 * r[2][x] + r[3][x]) * 0.125f;
    }
  });
}

struct Level {
  int w, h;
  std::vector<float> a, b, ma, mb, wt;
};

}  // namespace

float sampleField(const float* f, int w, int h, float x, float y) {
  x = std::min(std::max(x, 0.0f), float(w - 1));
  y = std::min(std::max(y, 0.0f), float(h - 1));
  const int x0 = std::min(int(x), std::max(w - 2, 0)), y0 = std::min(int(y), std::max(h - 2, 0));
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float fx = x - x0, fy = y - y0;
  const float* r0 = f + size_t(y0) * w;
  const float* r1 = f + size_t(y1) * w;
  const float top = r0[x0] + fx * (r0[x1] - r0[x0]), bot = r1[x0] + fx * (r1[x1] - r1[x0]);
  return top + fy * (bot - top);
}

void downsample(const float* in, int w, int h, int s, float* out, Workers* wk) {
  if (s <= 1) {
    std::copy(in, in + size_t(w) * h, out);
    return;
  }
  std::vector<float> a, b(in, in + size_t(w) * h);
  for (; s > 1; s /= 2) {
    a.resize(size_t((w + 1) / 2) * ((h + 1) / 2));
    halve(b.data(), w, h, a.data(), wk);
    w = (w + 1) / 2;
    h = (h + 1) / 2;
    a.swap(b);
  }
  std::copy(b.begin(), b.end(), out);
}

void luma(const float* rgb, int w, int h, int s, float* out, Workers* wk) {
  auto c01 = [](float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; };  // NaN -> 0
  std::vector<float> y(size_t(w) * h);
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i)
      y[i] = 0.2126f * c01(rgb[i * 3]) + 0.7152f * c01(rgb[i * 3 + 1]) + 0.0722f * c01(rgb[i * 3 + 2]);
  });
  downsample(y.data(), w, h, s, out, wk);
}

Flow denseFlow(const float* a, const float* b, const float* ma, const float* mb, const float* weight,
               int w, int h, const FlowParams& p, Workers* wk) {
  const bool useMatte = p.matteGain > 0 && ma && mb;
  const size_t n0 = size_t(w) * h;
  std::vector<Level> pyr(1);
  pyr[0].w = w;
  pyr[0].h = h;
  pyr[0].a.assign(a, a + n0);
  pyr[0].b.assign(b, b + n0);
  if (useMatte) {
    pyr[0].ma.assign(ma, ma + n0);
    pyr[0].mb.assign(mb, mb + n0);
  }
  pyr[0].wt = weight ? std::vector<float>(weight, weight + n0) : std::vector<float>(n0, 1.0f);
  while (int(pyr.size()) < std::max(p.levels, 1)) {
    const Level& f = pyr.back();
    const int cw = (f.w + 1) / 2, ch = (f.h + 1) / 2;
    if (cw < 16 || ch < 16) break;
    Level c;
    c.w = cw;
    c.h = ch;
    const size_t cn = size_t(cw) * ch;
    auto down = [&](const std::vector<float>& src, std::vector<float>& dst) {
      dst.resize(cn);
      halve(src.data(), f.w, f.h, dst.data(), wk);
    };
    down(f.a, c.a);
    down(f.b, c.b);
    down(f.wt, c.wt);
    if (useMatte) {
      down(f.ma, c.ma);
      down(f.mb, c.mb);
    }
    pyr.push_back(std::move(c));
  }

  Flow flow;
  for (int l = int(pyr.size()) - 1; l >= 0; --l) {
    const Level& L = pyr[l];
    const int lw = L.w, lh = L.h;
    const size_t n = size_t(lw) * lh;
    std::vector<float> u(n, 0.0f), v(n, 0.0f);
    if (!flow.u.empty()) {  // the coarser estimate, upsampled (coarse pixel j = fine 2j + 0.5)
      parallelFor(wk, lh, 16, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
          for (int x = 0; x < lw; ++x) {
            const float cx = (x - 0.5f) / 2, cy = (y - 0.5f) / 2;
            u[size_t(y) * lw + x] = 2 * sampleField(flow.u.data(), flow.w, flow.h, cx, cy);
            v[size_t(y) * lw + x] = 2 * sampleField(flow.v.data(), flow.w, flow.h, cx, cy);
          }
      });
    }
    std::vector<float> gx(n), gy(n), mx, my, tmp(n);
    gradient(L.a.data(), lw, lh, gx.data(), gy.data(), wk);
    if (useMatte) {
      mx.resize(n);
      my.resize(n);
      gradient(L.ma.data(), lw, lh, mx.data(), my.data(), wk);
    }
    const float g = p.matteGain;
    // The weighted structure tensor (fixed per level: a's gradients) and the weight sum.
    std::vector<float> sxx(n), sxy(n), syy(n), sw(n), bx(n), by(n);
    {
      std::vector<float> q(n);
      auto fill = [&](int k) {
        parallelFor(wk, lh, 16, [&](int y0, int y1) {
          for (size_t i = size_t(y0) * lw; i < size_t(y1) * lw; ++i) {
            const float wt = L.wt[i];
            float val;
            if (k == 0)
              val = gx[i] * gx[i] + (useMatte ? g * mx[i] * mx[i] : 0.0f);
            else if (k == 1)
              val = gx[i] * gy[i] + (useMatte ? g * mx[i] * my[i] : 0.0f);
            else if (k == 2)
              val = gy[i] * gy[i] + (useMatte ? g * my[i] * my[i] : 0.0f);
            else
              val = 1.0f;
            q[i] = wt * val;
          }
        });
      };
      fill(0);
      boxSum(q.data(), lw, lh, p.radius, sxx.data(), tmp.data(), wk);
      fill(1);
      boxSum(q.data(), lw, lh, p.radius, sxy.data(), tmp.data(), wk);
      fill(2);
      boxSum(q.data(), lw, lh, p.radius, syy.data(), tmp.data(), wk);
      fill(3);
      boxSum(q.data(), lw, lh, p.radius, sw.data(), tmp.data(), wk);
    }
    // Each iteration: every pixel q linearises its own data term at its own flow u(q), which gives
    // the constraint G(q) x = G(q) u(q) - grad(q) e(q) (G = grad grad^T, e = b(q + u(q)) - a(q));
    // a pixel's new flow is the weighted least-squares solution over its window, damped towards
    // its previous flow. Solving for an increment with the window's e instead (each q at its own
    // flow, one increment for the window) is not a contraction: it diverged after ~4 iterations.
    std::vector<float> qx(n), qy(n);
    for (int it = 0; it < std::max(p.iterations, 1); ++it) {
      parallelFor(wk, lh, 16, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
          for (int x = 0; x < lw; ++x) {
            const size_t i = size_t(y) * lw + x;
            const float wx = x + u[i], wy = y + v[i];
            if (wx < 0 || wy < 0 || wx > lw - 1 || wy > lh - 1 || L.wt[i] == 0.0f) {  // b has no data there
              qx[i] = qy[i] = 0.0f;
              continue;
            }
            const float et = sampleField(L.b.data(), lw, lh, wx, wy) - L.a[i];
            float gxx = gx[i] * gx[i], gxy = gx[i] * gy[i], gyy = gy[i] * gy[i];
            float px = gx[i] * et, py = gy[i] * et;
            if (useMatte) {
              const float mt = sampleField(L.mb.data(), lw, lh, wx, wy) - L.ma[i];
              px += g * mx[i] * mt;
              py += g * my[i] * mt;
              gxx += g * mx[i] * mx[i];
              gxy += g * mx[i] * my[i];
              gyy += g * my[i] * my[i];
            }
            qx[i] = L.wt[i] * (gxx * u[i] + gxy * v[i] - px);
            qy[i] = L.wt[i] * (gxy * u[i] + gyy * v[i] - py);
          }
      });
      boxSum(qx.data(), lw, lh, p.radius, bx.data(), tmp.data(), wk);
      boxSum(qy.data(), lw, lh, p.radius, by.data(), tmp.data(), wk);
      parallelFor(wk, lh, 16, [&](int y0, int y1) {
        for (size_t i = size_t(y0) * lw; i < size_t(y1) * lw; ++i) {
          // Damping: relative to the window's own gradient energy (LM-style; also settles the
          // aperture problem) plus a floor per unit of weight (textureless areas keep their flow).
          const double lam = double(p.damping) * (sxx[i] + syy[i]) + double(p.lambda) * sw[i] + 1e-20;
          const double A = sxx[i] + lam, B = sxy[i], C = syy[i] + lam;
          const double det = A * C - B * B;
          if (!(det > 0)) continue;
          const double rx = bx[i] + lam * u[i], ry = by[i] + lam * v[i];
          double du = (C * rx - B * ry) / det - u[i], dv = (-B * rx + A * ry) / det - v[i];
          const double len = std::hypot(du, dv);
          if (len > 2.0) du *= 2.0 / len, dv *= 2.0 / len;  // at most 2 px of this level per iteration
          u[i] += float(du);
          v[i] += float(dv);
        }
      });
    }
    flow.w = lw;
    flow.h = lh;
    flow.u.swap(u);
    flow.v.swap(v);
  }
  return flow;
}

}  // namespace ml
}  // namespace mr
