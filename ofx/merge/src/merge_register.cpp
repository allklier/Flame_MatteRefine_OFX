#include "merge_register.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "merge_morph.h"
#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

// A pyramid level: f = its pixel size in full-resolution px; pixel j is centred on full px
// f j + (f - 1) / 2. sb, sd: signed distances (level px) of the base's and the detail's openings.
struct Level {
  int f = 1, w = 0, h = 0;
  double band = 0;
  std::unique_ptr<float[]> sb, sd;
  std::vector<int> samples;  // base edge pixels used: |sb| < band - 2, inside the margin
};

void boxDown(const float* in, int w, int h, int f, std::vector<float>& out, int* ow, int* oh, Workers* wk) {
  *ow = w / f;
  *oh = h / f;
  const int lw = *ow;
  out.assign(size_t(lw) * *oh, 0.0f);
  parallelFor(wk, *oh, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < lw; ++x) {
        double s = 0;
        for (int j = 0; j < f; ++j)
          for (int i = 0; i < f; ++i) s += in[size_t(y * f + j) * w + x * f + i];
        out[size_t(y) * lw + x] = float(s / (f * f));
      }
  });
}


// [1 2 1] / 4 along x then y, clamped at the frame: after a box reduction a hard edge is about one
// level pixel wide, where the sub-pixel distance has a phase-dependent bias of ~0.1-0.2 px (measured:
// 0.2-0.3 px registration error on 2 px edges at half resolution); this keeps edges >= 2 px wide.
void blur121(std::vector<float>& v, int w, int h, Workers* wk) {
  std::vector<float> t(v.size());
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* r = &v[size_t(y) * w];
      for (int x = 0; x < w; ++x)
        t[size_t(y) * w + x] = 0.25f * r[std::max(0, x - 1)] + 0.5f * r[x] + 0.25f * r[std::min(w - 1, x + 1)];
    }
  });
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float *a = &t[size_t(std::max(0, y - 1)) * w], *b = &t[size_t(y) * w], *c = &t[size_t(std::min(h - 1, y + 1)) * w];
      for (int x = 0; x < w; ++x) v[size_t(y) * w + x] = 0.25f * a[x] + 0.5f * b[x] + 0.25f * c[x];
    }
  });
}

Level makeLevel(const float* base, const float* detail, int w, int h, int f, float baseLevel, int openR,
                double band, Workers* wk) {
  Level L;
  L.f = f;
  L.band = band;
  std::vector<float> b, d;
  if (f == 1) {
    L.w = w;
    L.h = h;
    b.assign(base, base + size_t(w) * h);
    d.assign(detail, detail + size_t(w) * h);
  } else {
    boxDown(base, w, h, f, b, &L.w, &L.h, wk);
    boxDown(detail, w, h, f, d, &L.w, &L.h, wk);
    blur121(b, L.w, L.h, wk);
    blur121(d, L.w, L.h, wk);
  }
  std::vector<float> ob(b.size()), od(d.size());
  opening(b.data(), L.w, L.h, openR, ob.data(), wk);
  opening(d.data(), L.w, L.h, openR, od.data(), wk);
  L.sb = signedEdgeDistanceThreaded(ob.data(), L.w, L.h, baseLevel, band);
  L.sd = signedEdgeDistanceThreaded(od.data(), L.w, L.h, 0.5f, band);
  const double lim = band - 2.0;
  const int margin = std::max(2, 8 / f);
  for (int y = margin; y < L.h - margin; ++y)
    for (int x = margin; x < L.w - margin; ++x)
      if (std::fabs(L.sb[size_t(y) * L.w + x]) < lim) L.samples.push_back(y * L.w + x);
  return L;
}

// Gauss-Newton on p = (tx, ty, u = (scale - 1) * R, delta), in full-resolution px (R = half the
// larger frame side, so u is the displacement at that distance from the centre). Residual per base
// edge pixel p (L.samples): f * sd(q) - f * sb(p) - delta, q = reg.apply(p) in level px. Robust
// weight: 1 up to k px, k / |r| beyond (Huber), 0 beyond cut. Sums per fixed chunk of samples, added
// in order: the result does not depend on the thread count.
// Tried and dropped (2026-10-24, hair scene): leaving out edges near fine detail (hair) and adding
// even harmonics of the normal angle to the edge offset; both made the estimate worse, because a
// subject cut by the frame has no edges facing down, so ty trades off against any such term.
constexpr int kP = 4;

double sensitivityOf(const double* s, int np, double count) {
  double M[kP][2 * kP];
  for (int i = 0; i < np; ++i)
    for (int j = 0; j < np; ++j) {
      M[i][j] = s[std::min(i, j) * kP + std::max(i, j)];
      M[i][np + j] = i == j;
    }
  for (int c = 0; c < np; ++c) {  // Gauss-Jordan inverse
    int p = c;
    for (int r = c + 1; r < np; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
    for (int j = 0; j < 2 * np; ++j) std::swap(M[c][j], M[p][j]);
    if (std::fabs(M[c][c]) < 1e-300) return 1e9;
    const double d = M[c][c];
    for (int j = 0; j < 2 * np; ++j) M[c][j] /= d;
    for (int r = 0; r < np; ++r)
      if (r != c) {
        const double m = M[r][c];
        for (int j = 0; j < 2 * np; ++j) M[r][j] -= m * M[c][j];
      }
  }
  // Largest eigenvalue of the symmetric 3x3 block (power iteration).
  double v[3] = {1, 1, 1}, lam = 0;
  for (int it = 0; it < 100; ++it) {
    double u[3];
    for (int i = 0; i < 3; ++i) u[i] = M[i][np] * v[0] + M[i][np + 1] * v[1] + M[i][np + 2] * v[2];
    lam = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (lam <= 0) return 0;
    for (int i = 0; i < 3; ++i) v[i] = u[i] / lam;
  }
  return std::sqrt(count * lam);
}
void solveLevel(const Level& L, Registration& reg, double R, int iterations, double k, double cut,
                bool fixScale, Workers* wk) {
  const int f = L.f, lw = L.w, lh = L.h;
  const int np = kP;
  const double off = (f - 1) / 2.0, lim = L.band - 2.0;
  const int chunk = 1024, chunks = int((L.samples.size() + chunk - 1) / chunk);
  const int nAcc = kP * kP + kP + 2;  // normal matrix, rhs, sum w r^2, count
  std::vector<double> part(size_t(chunks) * nAcc);
  for (int it = 0; it < iterations; ++it) {
    std::fill(part.begin(), part.end(), 0.0);
    parallelFor(wk, chunks, 1, [&](int c0, int c1) {
      for (int c = c0; c < c1; ++c) {
        double* acc = &part[size_t(c) * nAcc];
        const size_t e = std::min(L.samples.size(), size_t(c + 1) * chunk);
        for (size_t si = size_t(c) * chunk; si < e; ++si) {
          const int i = L.samples[si], x = i % lw, y = i / lw;
          const float sbv = L.sb[i];
          const double px = f * x + off, py = f * y + off;
          double qx, qy;
          reg.apply(px, py, &qx, &qy);
          const double lx = (qx - off) / f, ly = (qy - off) / f;
          if (lx < 1 || ly < 1 || lx > lw - 2 || ly > lh - 2) continue;
          const float* s = L.sd.get();
          const double sdv = sampleBilinear(s, lw, lh, lx, ly);
          if (std::fabs(sdv) >= lim) continue;
          const double gx = sampleBilinear(s, lw, lh, lx + 0.5, ly) - sampleBilinear(s, lw, lh, lx - 0.5, ly);
          const double gy = sampleBilinear(s, lw, lh, lx, ly + 0.5) - sampleBilinear(s, lw, lh, lx, ly - 0.5);
          const double r = f * sdv - f * double(sbv) - reg.delta;
          const double a = std::fabs(r);
          if (a > cut) continue;
          const double wt = a <= k ? 1.0 : k / a;
          // fixScale: the scale column is zero (the small Tikhonov term below then holds it).
          const double J[kP] = {gx, gy, fixScale ? 0.0 : (gx * (px - reg.cx) + gy * (py - reg.cy)) / R, -1.0};
          for (int ii = 0; ii < np; ++ii) {
            for (int j = ii; j < np; ++j) acc[ii * kP + j] += wt * J[ii] * J[j];
            acc[kP * kP + ii] -= wt * J[ii] * r;
          }
          acc[kP * kP + kP] += wt * r * r;
          acc[kP * kP + kP + 1] += 1;
        }
      }
    });
    std::vector<double> s(nAcc, 0.0);
    for (int c = 0; c < chunks; ++c)
      for (int i = 0; i < nAcc; ++i) s[i] += part[size_t(c) * nAcc + i];
    const double count = s[kP * kP + kP + 1];
    reg.samples = int(count);
    reg.rmsPx = count > 0 ? std::sqrt(s[kP * kP + kP] / count) : 0.0;
    if (count < 50) return;
    // Conditioning: sqrt(N * largest eigenvalue of the (tx, ty, u) block of (J^T W J)^-1), the
    // contour displacement (px) that a systematic edge-offset error of 1 px RMS can cause.
    if (!fixScale) reg.sensitivity = sensitivityOf(s.data(), np, count);
    // Solve the np x np system (symmetric, small Tikhonov term), Gaussian elimination with pivoting.
    double A[kP][kP + 1];
    double tr = 0;
    for (int i = 0; i < np; ++i) {
      for (int j = i; j < np; ++j) A[i][j] = A[j][i] = s[i * kP + j];
      tr += A[i][i];
    }
    for (int i = 0; i < np; ++i) {
      A[i][i] += 1e-6 * tr / np + 1e-12;
      A[i][np] = s[kP * kP + i];
    }
    for (int c = 0; c < np; ++c) {
      int p = c;
      for (int r2 = c + 1; r2 < np; ++r2)
        if (std::fabs(A[r2][c]) > std::fabs(A[p][c])) p = r2;
      for (int j = 0; j <= np; ++j) std::swap(A[c][j], A[p][j]);
      for (int r2 = c + 1; r2 < np; ++r2) {
        const double m = A[r2][c] / A[c][c];
        for (int j = c; j <= np; ++j) A[r2][j] -= m * A[c][j];
      }
    }
    double d[kP] = {0};
    for (int i = np - 1; i >= 0; --i) {
      double v = A[i][np];
      for (int j = i + 1; j < np; ++j) v -= A[i][j] * d[j];
      d[i] = v / A[i][i];
    }
    reg.tx = std::max(-64.0, std::min(64.0, reg.tx + d[0]));
    reg.ty = std::max(-64.0, std::min(64.0, reg.ty + d[1]));
    reg.scale = std::max(0.95, std::min(1.05, reg.scale + d[2] / R));
    reg.delta = std::max(-64.0, std::min(64.0, reg.delta + d[3]));
    if (std::fabs(d[0]) + std::fabs(d[1]) + std::fabs(d[2]) + std::fabs(d[3]) < 1e-3) break;
  }
}

}  // namespace

Registration registerMattes(const float* base, const float* detail, int w, int h, float baseLevel,
                            int openRadius, Workers* wk, bool fixScale, bool refine) {
  Registration reg;
  reg.cx = (w - 1) / 2.0;
  reg.cy = (h - 1) / 2.0;
  const double R = std::max(w, h) / 2.0;
  const int f = std::min(w, h) >= 256 ? 4 : 1;
  if (f > 1) {
    const Level coarse = makeLevel(base, detail, w, h, f, baseLevel, std::max(1, (openRadius + f / 2) / f), 16.0, wk);
    solveLevel(coarse, reg, R, 30, 2.0, 16.0, fixScale, wk);
    if (!refine) {
      reg.ok = reg.samples >= 50;
      if (!reg.ok) {
        const int n = reg.samples;
        reg = Registration();
        reg.cx = (w - 1) / 2.0;
        reg.cy = (h - 1) / 2.0;
        reg.samples = n;
      }
      return reg;
    }
  }
  // The refinement, at half resolution (measured on the hair scene at 2880x1620: as accurate as full
  // resolution, 30 ms instead of 100), with a band that holds the remaining edge offset.
  const double band = std::min(48.0, std::max(12.0, std::fabs(reg.delta) + 8.0));
  const int ff = f > 1 ? 2 : 1;
  const Level fine = makeLevel(base, detail, w, h, ff, baseLevel, std::max(1, (openRadius + ff / 2) / ff), band / ff, wk);
  solveLevel(fine, reg, R, 20, 0.75, 4.0, fixScale, wk);
  reg.ok = reg.samples >= 200;
  if (!reg.ok) {
    const int n = reg.samples;
    reg = Registration();
    reg.cx = (w - 1) / 2.0;
    reg.cy = (h - 1) / 2.0;
    reg.samples = n;
  }
  return reg;
}

void warpDetail(const float* detail, int w, int h, const Registration& reg, float* out, Workers* wk) {
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        double qx, qy;
        reg.apply(x, y, &qx, &qy);
        out[size_t(y) * w + x] = sampleBilinear(detail, w, h, qx, qy);
      }
  });
}

}  // namespace merge
}  // namespace mr
