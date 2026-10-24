#include "merge_synth.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "synth.h"

namespace mmsynth {

namespace {

double clamp01(double v) { return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v; }

// synth.cpp's body: head (circle) and shoulders (round box), design units, > 0 inside.
double sdBody(double x, double y) {
  const double head = 400 - std::hypot(x, y - 400);
  const double qx = std::fabs(x) - 760 + 160, qy = std::fabs(y - 560) - 200 + 160;
  const double box = 160 - (std::hypot(std::max(qx, 0.0), std::max(qy, 0.0)) + std::min(std::max(qx, qy), 0.0));
  return std::max(head, box);
}

void blur(std::vector<float>& v, int w, int h, double sigma) {
  const int r = std::max(1, int(std::ceil(3 * sigma)));
  std::vector<float> k(2 * r + 1);
  float sum = 0;
  for (int j = -r; j <= r; ++j) sum += k[j + r] = float(std::exp(-0.5 * j * j / (sigma * sigma)));
  for (float& e : k) e /= sum;
  std::vector<float> tmp(v.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float a = 0;
      for (int j = -r; j <= r; ++j) a += k[j + r] * v[size_t(y) * w + std::min(std::max(x + j, 0), w - 1)];
      tmp[size_t(y) * w + x] = a;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float a = 0;
      for (int j = -r; j <= r; ++j) a += k[j + r] * tmp[size_t(std::min(std::max(y + j, 0), h - 1)) * w + x];
      v[size_t(y) * w + x] = a;
    }
}

float bilinear(const std::vector<float>& img, int w, int h, double x, double y) {
  x = std::min(std::max(x, 0.0), double(w - 1));
  y = std::min(std::max(y, 0.0), double(h - 1));
  const int x0 = std::min(int(x), w - 2), y0 = std::min(int(y), h - 2);
  const double fx = x - x0, fy = y - y0;
  const size_t i = size_t(y0) * w + x0;
  return float((img[i] * (1 - fx) + img[i + 1] * fx) * (1 - fy) + (img[i + w] * (1 - fx) + img[i + w + 1] * fx) * fy);
}

// Strands rooted on the head as synth's makeHair draws them (polyline of 32 capsule segments,
// tapering width and opacity), composited "over" into alpha.
void drawStrands(std::vector<float>& alpha, int w, int h, double k, int count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const int segs = 32;
  std::vector<float> buf;
  for (int s = 0; s < count; ++s) {
    const double a0 = -M_PI * (0.1 + 0.8 * u(rng));
    double dir = a0 + (u(rng) - 0.5) * 1.2;
    const double len = 60 + 180 * u(rng), curl = (u(rng) - 0.5) * 2.4, width = 1 + 2 * u(rng);
    const double opacity = 0.35 + 0.65 * u(rng);
    double xs[segs + 1], ys[segs + 1];
    xs[0] = 394 * std::cos(a0);
    ys[0] = 400 + 394 * std::sin(a0);
    for (int j = 1; j <= segs; ++j) {
      dir += curl / segs;
      xs[j] = xs[j - 1] + len / segs * std::cos(dir);
      ys[j] = ys[j - 1] + len / segs * std::sin(dir);
    }
    const double pad = width * k + 2;
    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (int j = 0; j <= segs; ++j)
      x0 = std::min(x0, xs[j]), x1 = std::max(x1, xs[j]), y0 = std::min(y0, ys[j]), y1 = std::max(y1, ys[j]);
    const int bx0 = std::max(0, int(std::floor(x0 * k + w / 2.0 - pad)));
    const int bx1 = std::min(w - 1, int(std::ceil(x1 * k + w / 2.0 + pad)));
    const int by0 = std::max(0, int(std::floor(y0 * k + h / 2.0 - pad)));
    const int by1 = std::min(h - 1, int(std::ceil(y1 * k + h / 2.0 + pad)));
    if (bx0 > bx1 || by0 > by1) continue;
    const int bw = bx1 - bx0 + 1;
    buf.assign(size_t(bw) * (by1 - by0 + 1), 0.0f);
    for (int j = 0; j < segs; ++j) {
      const double t = (j + 0.5) / segs;
      const double wpx = width * k * (1.0 - 0.4 * t), op = opacity * (1.0 - 0.6 * t);
      const double ax = xs[j] * k + w / 2.0, ay = ys[j] * k + h / 2.0;
      const double cx = xs[j + 1] * k + w / 2.0, cy = ys[j + 1] * k + h / 2.0;
      for (int py = std::max(by0, int(std::floor(std::min(ay, cy) - wpx - 1)));
           py <= std::min(by1, int(std::ceil(std::max(ay, cy) + wpx + 1))); ++py)
        for (int px = std::max(bx0, int(std::floor(std::min(ax, cx) - wpx - 1)));
             px <= std::min(bx1, int(std::ceil(std::max(ax, cx) + wpx + 1))); ++px) {
          const double vx = cx - ax, vy = cy - ay, qx = px - ax, qy = py - ay;
          const double tt = clamp01((qx * vx + qy * vy) / (vx * vx + vy * vy));
          const double d = std::hypot(qx - tt * vx, qy - tt * vy);
          const double cover = op * clamp01(std::min(wpx, wpx / 2 + 0.5 - d));
          float& b = buf[size_t(py - by0) * bw + px - bx0];
          b = std::max(b, float(cover));
        }
    }
    for (int py = by0; py <= by1; ++py)
      for (int px = bx0; px <= bx1; ++px) {
        const float a = buf[size_t(py - by0) * bw + px - bx0];
        float& o = alpha[size_t(py) * w + px];
        o = a + (1 - a) * o;
      }
  }
}

}  // namespace

float srgbEncode(float v) {
  if (v <= 0.0031308f) return 12.92f * v;
  return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// RED Log3G10 (IPP2): 0.224282 log10((x + 0.01) 155.975327 + 1); linear below x = -0.01.
float log3g10Encode(float v) {
  const float x = v + 0.01f;
  if (x < 0) return x * 15.1927f;
  return 0.224282f * std::log10(x * 155.975327f + 1.0f);
}

MergeScene makeMergeScene(int w, int h, bool busy, unsigned seed, int fakeStrands) {
  const synth::Scene s = synth::makeHair(w, h, busy, seed);
  MergeScene m;
  m.name = s.name;
  m.w = w;
  m.h = h;
  m.k = h / 1080.0;
  const double k = m.k;
  const size_t n = size_t(w) * h;
  m.base = s.matte;
  m.truth = s.truth;
  m.body.resize(n);
  m.bodySd.resize(n);
  m.strands.resize(n);
  m.trimap.resize(n);
  m.baseUniform.resize(n);
  std::vector<float> thick(n);
  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
      const size_t i = size_t(py) * w + px;
      const double sd = sdBody((px - w / 2.0) / k, (py - h / 2.0) / k) * k;
      m.bodySd[i] = float(sd);
      m.body[i] = float(clamp01(0.5 + sd / (2.0 * k)));
      thick[i] = float(clamp01(0.5 + (sd + 2.0 * k) / (2.0 * k)));
      m.baseUniform[i] = float(clamp01(0.5 + (sd - 3.0 * k) / (2.0 * k)));
      const double b = m.body[i];
      m.strands[i] = b < 0.999 ? float(clamp01((m.truth[i] - b) / (1.0 - b))) : 0.0f;
      m.trimap[i] = sd > 6 * k ? 1.0f : sd < -120 * k ? 0.0f : 0.5f;
    }

  // Hallucinated strands: a different seed, the same kind of strand, not in the picture.
  m.fake.assign(n, 0.0f);
  drawStrands(m.fake, w, h, k, fakeStrands, seed * 7919u + 101u);

  // The blob: radius 14, its centre 40 outside the body edge (design units), where the real and the
  // fake strands are farthest away (candidates along the body's side and shoulder tops).
  m.blobR = 14 * k;
  double best = 1e9;
  for (double gx = -900; gx <= 900; gx += 6)
    for (double gy = -100; gy <= 540; gy += 6) {
      if (std::fabs(sdBody(gx, gy) + 40) > 1.5) continue;
      const double cx = gx * k + w / 2.0, cy = gy * k + h / 2.0;
      if (cx < 40 * k || cy < 40 * k || cx > w - 40 * k || cy > h - 40 * k) continue;
      double worst = 0;
      const int rr = int(std::ceil(m.blobR + 14 * k));
      for (int y = int(cy) - rr; y <= int(cy) + rr; ++y)
        for (int x = int(cx) - rr; x <= int(cx) + rr; ++x) {
          if (x < 0 || y < 0 || x >= w || y >= h || std::hypot(x - cx, y - cy) > rr) continue;
          worst = std::max(worst, double(std::max(m.strands[size_t(y) * w + x], m.fake[size_t(y) * w + x])));
        }
      // Prefer no strands, then the shoulder tops (gy larger), deterministic.
      const double score = worst * 1000 - gy * 1e-3;
      if (score < best) best = score, m.blobX = cx, m.blobY = cy;
    }
  m.blob.resize(n);
  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px)
      m.blob[size_t(py) * w + px] =
          float(clamp01(0.5 + (m.blobR - std::hypot(px - m.blobX, py - m.blobY)) / (2.0 * k)));

  // The detail matte before its move: thickened body, real strands, fake strands, blob; blurred.
  std::vector<float> d0(n);
  for (size_t i = 0; i < n; ++i) {
    double a = 1.0 - (1.0 - thick[i]) * (1.0 - m.strands[i]);
    a = 1.0 - (1.0 - a) * (1.0 - m.fake[i]);
    d0[i] = float(std::max(a, double(m.blob[i])));
  }
  blur(d0, w, h, 0.8 * k);
  // Moved: base pixel p is at A(p) in the detail, so detail(x) = d0(A^-1(x)).
  m.scale = 1.01;
  m.tx = 2.2 * k;
  m.ty = -1.4 * k;
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  m.detail.resize(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      m.detail[size_t(y) * w + x] =
          bilinear(d0, w, h, cx + (x - cx - m.tx) / m.scale, cy + (y - cy - m.ty) / m.scale);

  m.rgbLinear = s.rgb;
  m.rgbDisplay.resize(n * 3);
  m.rgbLog.resize(n * 3);
  for (size_t i = 0; i < n * 3; ++i) {
    m.rgbDisplay[i] = srgbEncode(s.rgb[i]);
    m.rgbLog[i] = log3g10Encode(s.rgb[i]);
  }
  return m;
}

}  // namespace mmsynth
