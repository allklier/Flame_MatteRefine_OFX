// The edge operations and their shared code as they were before the speed work (git c739311,
// ofx/src/ops: morphology, edge_field, shrink_grow, edge_width, contour_smooth, solid_core),
// pasted unchanged into namespace old_ (each operation in its own namespace: old_::sg Shrink /
// Grow, old_::ew Edge Width, old_::cs Contour Smooth, old_::sc Solid Core), so the tests can
// check the current code gives the same results bit for bit. Test-only; slow on purpose is fine.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace old_ {

// ---- morphology.h ----
// Binary morphology shared by the operations: masks are w*h bytes, row 0 = the top row.




using Mask = std::vector<uint8_t>;

// Runs fn(begin, end) over [0, n) on up to 32 threads, at least minPerThread items each.
template <class F>
void parallelFor(int n, int minPerThread, F fn) {
  const int t =
      std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, n / minPerThread}));
  if (t == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  for (int i = 0; i < t; ++i) pool.emplace_back(fn, n * i / t, n * (i + 1) / t);
  for (auto& th : pool) th.join();
}

// Pixels within Euclidean distance r of a set pixel of m (m included). Pixels outside the
// frame count as unset.
Mask dilate(const Mask& m, int w, int h, double r);

// Where the contour runs near a set pixel q, relative to q's centre (px): the foot point (fx, fy)
// and the unit normal (nx, ny) pointing away from q's side. All zero = the contour is q itself.
struct ContourFoot {
  float fx = 0.0f, fy = 0.0f, nx = 0.0f, ny = 0.0f;
};

// For each unset pixel p: the distance to the contour, modelled near each set pixel q with an
// unset 4-neighbour as a tangent segment (+-0.5 px) through q's foot point, and capped at the
// plain distance |p - q| (so never more than the grid distance). Set pixels get 0. foot(q) is
// called only for those boundary pixels. Exact (to the model) where it is at most r, larger
// than r elsewhere. When the exact search would cost more than 2 frames' worth of writes (a
// large r on a noisy matte), the plain distance (feet ignored) instead. Pixels outside the
// frame count as unset.
std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<ContourFoot(int)>& foot);

// ---- morphology.cpp ----
// Binary morphology shared by the operations (decision 9 "Implementation").




namespace {

constexpr double kFar = 1e20;

// 1-D squared distance transform of f (Felzenszwalb & Huttenlocher, "Distance Transforms of
// Sampled Functions", 2012). v, z: scratch of n and n + 1.
void edt1d(const double* f, double* d, int n, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kFar;
  z[1] = kFar;
  auto meet = [&](int q, int p) {
    return ((f[q] + double(q) * q) - (f[p] + double(p) * p)) / (2.0 * (q - p));
  };
  for (int q = 1; q < n; ++q) {
    // s > -kFar always (|f| <= kFar, q - p >= 1), so k never drops below 0.
    double s = meet(q, v[k]);
    while (s <= z[k]) s = meet(q, v[--k]);
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kFar;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const double dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

// Squared Euclidean distance of every pixel to the nearest set pixel of m (>= kFar if none).
std::vector<float> squaredDistance(const Mask& m, int w, int h) {
  std::vector<float> g(size_t(w) * h);
  // Columns, kBlock at a time so the reads and writes walk along rows.
  constexpr int kBlock = 16;
  parallelFor((w + kBlock - 1) / kBlock, 2, [&](int b0, int b1) {
    std::vector<double> f(size_t(h) * kBlock), d(h), z(h + 1);
    std::vector<int> v(h);
    for (int b = b0; b < b1; ++b) {
      const int x0 = b * kBlock, nb = std::min(kBlock, w - x0);
      for (int y = 0; y < h; ++y)
        for (int k = 0; k < nb; ++k) f[size_t(k) * h + y] = m[size_t(y) * w + x0 + k] ? 0.0 : kFar;
      for (int k = 0; k < nb; ++k) {
        edt1d(&f[size_t(k) * h], d.data(), h, v.data(), z.data());
        std::copy(d.begin(), d.end(), &f[size_t(k) * h]);
      }
      for (int y = 0; y < h; ++y)
        for (int k = 0; k < nb; ++k) g[size_t(y) * w + x0 + k] = float(f[size_t(k) * h + y]);
    }
  });
  parallelFor(h, 32, [&](int y0, int y1) {  // rows, in place
    std::vector<double> f(w), d(w), z(w + 1);
    std::vector<int> v(w);
    for (int y = y0; y < y1; ++y) {
      float* row = &g[size_t(y) * w];
      std::copy(row, row + w, f.begin());
      edt1d(f.data(), d.data(), w, v.data(), z.data());
      for (int x = 0; x < w; ++x) row[x] = float(d[x]);
    }
  });
  return g;
}

// Disc offsets within r (centre excluded) and the boundary pixels of m (set, with an unset
// 4-neighbour): the nearest set pixel of any other pixel is a boundary pixel, so stamping the
// disc on each boundary pixel is exact.
void discAndBoundary(const Mask& m, int w, int h, double r, std::vector<std::pair<int, int>>& disc,
                     std::vector<int>& boundary) {
  const int ri = int(std::floor(r));
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if ((dx || dy) && dx * dx + dy * dy <= r * r) disc.emplace_back(dx, dy);
  // Rows in bands on threads, joined in order: the list stays sorted by index.
  std::vector<std::vector<int>> part(64);
  std::atomic<int> nextPart{0};
  parallelFor(h, 32, [&](int y0, int y1) {
    std::vector<int>& out = part[size_t(nextPart++)];
    out.push_back(y0);  // the band's first row, to sort the parts
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (m[i] && ((x > 0 && !m[i - 1]) || (x + 1 < w && !m[i + 1]) || (y > 0 && !m[i - w]) ||
                     (y + 1 < h && !m[i + w])))
          out.push_back(int(i));
      }
  });
  part.resize(size_t(nextPart));
  std::sort(part.begin(), part.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
    return a[0] < b[0];
  });
  for (const auto& p : part) boundary.insert(boundary.end(), p.begin() + 1, p.end());
}

}  // namespace

// When stamping costs more than a frame's worth of writes (a large r on a noisy matte), the
// distance transform does it instead.
Mask dilate(const Mask& m, int w, int h, double r) {
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m, w, h, r, disc, boundary);
  Mask out = m;
  if (double(boundary.size()) * disc.size() > double(m.size())) {
    const std::vector<float> d = squaredDistance(m, w, h);
    const float r2 = float(r * r);
    for (size_t i = 0; i < m.size(); ++i) out[i] = d[i] <= r2;
    return out;
  }
  for (int i : boundary) {
    const int x = i % w, y = i / w;
    for (const auto& o : disc) {
      const int nx = x + o.first, ny = y + o.second;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h) out[size_t(ny) * w + nx] = 1;
    }
  }
  return out;
}

// Same two paths as dilate. A foot point lies within 1 px of q and its segment within ~1.12 px,
// so a pixel within r of the contour is within r + 1.5 of the q it is measured from; stamping
// from the boundary is exact as above. Squared distances, one sqrt per pixel at the end.
std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<ContourFoot(int)>& foot) {
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m, w, h, r + 1.5, disc, boundary);
  std::vector<float> d(m.size());
  // Measured at UHD (BFD, 2026-10-22): stamping costs ~15 ms per frame's worth of writes, the
  // distance transform 25-45 ms, so they break even near 2 frames.
  if (double(boundary.size()) * disc.size() > 2.0 * double(m.size())) {
    d = squaredDistance(m, w, h);
    for (float& v : d) v = std::sqrt(v);
    return d;
  }
  std::vector<ContourFoot> feet(boundary.size());
  parallelFor(int(boundary.size()), 256, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) feet[size_t(b)] = foot(boundary[size_t(b)]);
  });
  // Row bands on threads; each band stamps from the boundary pixels within reach of its rows
  // and writes only its own rows.
  const int reach = int(std::floor(r + 1.5));
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) d[i] = m[i] ? 0.0f : float(kFar);
    const auto first = std::lower_bound(boundary.begin(), boundary.end(), std::max(y0 - reach, 0) * w);
    const auto last = std::lower_bound(boundary.begin(), boundary.end(), std::min(y1 + reach, h) * w);
    for (auto it = first; it != last; ++it) {
      const int i = *it, x = i % w, y = i / w;
      const ContourFoot& f = feet[size_t(it - boundary.begin())];
      const bool line = f.nx != 0.0f || f.ny != 0.0f;
      for (const auto& o : disc) {
        const int px = x + o.first, py = y + o.second;
        if (px < 0 || py < y0 || px >= w || py >= y1) continue;
        const float ox = float(o.first), oy = float(o.second), grid = ox * ox + oy * oy;
        float d2 = grid;
        if (line) {
          const float vx = ox - f.fx, vy = oy - f.fy;
          const float perp = vx * f.nx + vy * f.ny;
          const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
          d2 = std::min(grid, perp * perp + along * along);
        }
        float& v = d[size_t(py) * w + px];
        v = std::min(v, d2);
      }
    }
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i)
      if (d[i] < float(kFar)) d[i] = std::sqrt(d[i]);
  });
  return d;
}

// ---- edge_field.h ----
// Sub-pixel edge geometry shared by the edge operations (decision 16, generalised to both sides
// of the edge): the signed distance to a matte's contour and resampling along the edge normal,
// which moves an edge while keeping its own softness profile.





// Where the `level` contour runs near pixel i (side = in < level): per axis, t = where level is
// crossed towards the nearest 4-neighbour on the other side (linear interpolation, 0-1); the
// contour is the line through those crossings: normal ~ (1/tx, 1/ty), distance
// s = 1 / sqrt(sum 1 / t^2) (decision 16). All zero if no neighbour is on the other side.
ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i);

// Signed distance of each pixel to the matte's `level` contour: > 0 inside (in >= level), < 0
// outside (morphology.h distanceWithin with contourFoot). Exact where |d| <= band; clamped to +-(band + 1)
// beyond. Pixels beyond the frame are neither inside nor outside: the frame border is no edge.
std::vector<float> signedEdgeDistance(const float* in, int w, int h, float level, double band);

// Bilinear sample of a w*h image at (x, y), pixel centres at integers, clamped to the frame.
float sampleBilinear(const float* img, int w, int h, double x, double y);

// out(p) = in(p + shift(p) * n(p)), n = the unit gradient of sd (pointing inward): shift > 0
// moves the edge outward (grows), < 0 inward, and the profile across the edge travels with it.
// Where sd has no usable gradient (beyond its band, |grad| < 0.5) or shift is 0, out = in.
// out may not be in.
void shiftAlongNormal(const float* in, const std::vector<float>& sd,
                      const std::vector<float>& shift, int w, int h, float* out);

// ---- edge_field.cpp ----
// Direction from central differences of the matte (exact on a ramp, whichever neighbours are
// across; using only the across neighbours tilted the normal by up to 30 deg on slants);
// distance from the one-sided crossings (exact on a hard 1 px edge), each across axis giving
// t * |n_axis|, weighted by |n_axis|.
ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i) {
  const int x = i % w, y = i / w;
  const double a = in[i];
  double g[2], t[2] = {-1.0, -1.0}, across[2] = {0.0, 0.0};
  for (int axis = 0; axis < 2; ++axis) {
    const int lo = axis ? (y > 0 ? i - w : i) : (x > 0 ? i - 1 : i);
    const int hi = axis ? (y + 1 < h ? i + w : i) : (x + 1 < w ? i + 1 : i);
    const int span = (hi - lo) / (axis ? w : 1);
    g[axis] = span > 0 ? (in[hi] - in[lo]) / span : 0.0;
    for (int sg = -1; sg <= 1; sg += 2) {
      const int nx = x + (axis ? 0 : sg), ny = y + (axis ? sg : 0);
      if (nx < 0 || ny < 0 || nx >= w || ny >= h || side[size_t(ny) * w + nx] == side[size_t(i)])
        continue;
      const double b = in[size_t(ny) * w + nx];
      const double tc = (level - a) / (b - a);  // 0-1: level lies between a and b
      if (t[axis] < 0.0 || tc < t[axis]) t[axis] = tc, across[axis] = sg;
    }
  }
  ContourFoot f;
  if (across[0] == 0.0 && across[1] == 0.0) return f;
  double len = std::hypot(g[0], g[1]);
  if (len == 0.0) {  // flat around the centre (a 1 px line): the across neighbours' directions
    g[0] = across[0];
    g[1] = across[1];
    len = std::hypot(g[0], g[1]);
  }
  // n points away from this pixel's side: up the matte for a below pixel, down for an above one.
  const double sign = side[size_t(i)] ? 1.0 : -1.0;
  const double n[2] = {sign * g[0] / len, sign * g[1] / len};
  double sum = 0.0, wsum = 0.0;
  for (int axis = 0; axis < 2; ++axis)
    if (across[axis] != 0.0 && across[axis] * n[axis] > 0.0) {
      const double wa = std::fabs(n[axis]);
      sum += t[axis] * wa * wa;
      wsum += wa;
    }
  const double s = wsum > 0.0 ? std::min(1.0, sum / wsum) : 0.0;
  f.nx = float(n[0]);
  f.ny = float(n[1]);
  f.fx = float(n[0] * s);
  f.fy = float(n[1] * s);
  return f;
}

std::vector<float> signedEdgeDistance(const float* in, int w, int h, float level, double band) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask below(n), above(n);
  for (size_t i = 0; i < n; ++i) {
    below[i] = in[i] < level;
    above[i] = !below[i];
  }
  // Inside pixels measure to the contour near the below pixels, and the reverse.
  auto foot = [&](int i) { return contourFoot(in, below, w, h, level, i); };
  const std::vector<float> dIn = distanceWithin(below, w, h, band, foot);
  const std::vector<float> dOut = distanceWithin(above, w, h, band, foot);
  const float cap = float(band + 1.0);
  std::vector<float> sd(n);
  for (size_t i = 0; i < n; ++i) sd[i] = above[i] ? std::min(dIn[i], cap) : -std::min(dOut[i], cap);
  return sd;
}

float sampleBilinear(const float* img, int w, int h, double x, double y) {
  x = std::min(std::max(x, 0.0), double(w - 1));
  y = std::min(std::max(y, 0.0), double(h - 1));
  const int x0 = std::min(int(x), w - 2 < 0 ? 0 : w - 2), y0 = std::min(int(y), h - 2 < 0 ? 0 : h - 2);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const double fx = x - x0, fy = y - y0;
  const double top = img[size_t(y0) * w + x0] * (1 - fx) + img[size_t(y0) * w + x1] * fx;
  const double bot = img[size_t(y1) * w + x0] * (1 - fx) + img[size_t(y1) * w + x1] * fx;
  return float(top * (1 - fy) + bot * fy);
}

// The value at p + shift * n is read in distance space: the matte depends mainly on the
// distance to the contour, which sd knows to ~0.05 px, while bilinear sampling of a 1 px edge
// (a ramp with kinks) is off by up to 0.25. Of the 3x3 pixels around the sample point, the two
// whose sd brackets the target sd(p) + shift most tightly are interpolated linearly.
void shiftAlongNormal(const float* in, const std::vector<float>& sd,
                      const std::vector<float>& shift, int w, int h, float* out) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      out[i] = in[i];
      if (shift[i] == 0.0f) continue;
      // Central differences, one-sided at the frame border.
      const int xl = x > 0 ? x - 1 : x, xr = x + 1 < w ? x + 1 : x;
      const int yu = y > 0 ? y - 1 : y, yd = y + 1 < h ? y + 1 : y;
      const double gx = xr > xl ? (sd[size_t(y) * w + xr] - sd[size_t(y) * w + xl]) / (xr - xl) : 0.0;
      const double gy = yd > yu ? (sd[size_t(yd) * w + x] - sd[size_t(yu) * w + x]) / (yd - yu) : 0.0;
      const double g = std::hypot(gx, gy);
      if (g < 0.5) continue;
      const double target = sd[i] + shift[i];
      const int cx = int(std::lround(x + shift[i] * gx / g)), cy = int(std::lround(y + shift[i] * gy / g));
      double loD = -1e30, hiD = 1e30;
      float loV = 0.0f, hiV = 0.0f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int qx = std::min(std::max(cx + dx, 0), w - 1), qy = std::min(std::max(cy + dy, 0), h - 1);
          const size_t j = size_t(qy) * w + qx;
          if (sd[j] <= target && sd[j] > loD) loD = sd[j], loV = in[j];
          if (sd[j] >= target && sd[j] < hiD) hiD = sd[j], hiV = in[j];
        }
      if (loD > -1e29 && hiD < 1e29)
        out[i] = hiD > loD ? float(loV + (hiV - loV) * (target - loD) / (hiD - loD)) : loV;
      else
        out[i] = loD > -1e29 ? loV : hiD < 1e29 ? hiV : in[i];  // beyond the profile: flat
    }
}

namespace sg {

// ---- shrink_grow.h ----
// Operation A3, Shrink / Grow (CLAUDE.md "Plugin 1 operations"): moves the matte's 0.5 contour
// outward (grow) or inward (shrink) by Amount px, sub-pixel, while each part of the edge keeps
// its own falloff profile (a 1 px edge stays 1 px, an 8 px edge stays 8 px). Morphological, as
// moving every contour by Amount means: a gap or hole narrower than 2 x Amount closes on grow, a
// part narrower than 2 x Amount disappears on shrink. Grow never lowers a pixel, shrink never
// raises one. Strands (parts of the >= 0.5 area under ~3 px wide) are detail and keep their
// shape; strands below 0.5 have no contour of their own: away from an edge they are not touched.




struct ShrinkGrowParams {
  double amountPx = 0.0;  // > 0 grows, < 0 shrinks; 0 = no change
};

struct ShrinkGrowStats {
  long long pixelsChanged = 0;
  int passes = 0;  // large amounts move in passes of up to 4 px
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void shrinkGrow(const float* in, float* out, int w, int h, const ShrinkGrowParams& p,
                ShrinkGrowStats* stats = nullptr);

// Edge helpers shared with Edge Width (ops/edge_width); candidates for ops/edge_field.

// fn(begin, end) over ranges of [0, n) on up to 32 threads, at least minPerThread items each.
void parallelFor(int n, int minPerThread, const std::function<void(int, int)>& fn);

// edge_field.h signedEdgeDistance, the same values, but threaded and without distanceWithin's
// switch to the plain grid distance at 2 frames of writes: at the bands these ops need (up to
// 15 px) a 1080 frame with hair already crosses it, and the edge would move by whole pixels.
// Each band of rows stamps the contour pixels within reach into its own rows. Only when the
// stamping would cost more than 12 frames of writes (a very noisy matte), signedEdgeDistance.
// w*h floats, allocated without the zero fill a std::vector does (~8 ms single-threaded at UHD).
std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band);

// The unit normal of sd at pixel (x, y), pointing inward (up the distance): central differences,
// or where they cancel (the middle of a thin part or gap) the steeper one-sided difference per
// axis. False where sd is flat (beyond its band).
bool edgeNormal(const float* sd, int w, int h, int x, int y, double& nx, double& ny);

// The matte's value at signed distance `target` along the normal through (x, y), read near
// (x, y) + (target - sd) * n as edge_field.h shiftAlongNormal does: of the pixels there, the two
// whose sd brackets target most tightly, interpolated linearly; beyond the profile, its last
// value. Two changes: the window is 3 px across the normal's main axis and 7 along the other
// (not 3x3), and a read past the middle of a thin part stops at the middle (its peak).
float profileAt(const float* in, const float* sd, int w, int h, int x, int y,
                double nx, double ny, double target);

// A pixel of the 0.5 contour: a 4-neighbour on the other side of 0.5.
bool isContour(const float* in, int w, int h, int x, int y);

// The profile's slope across the contour at contour pixel (x, y), per px of sd: from the
// 4-neighbour across 0.5 farthest in sd (exact on a linear ramp; on a hard step, the step
// itself). 0 if there is none.
double contourSlope(const float* in, const float* sd, int w, int h, int x, int y);

// The tent-weighted (1.5 px) mean of per-contour-pixel measures around (fx, fy) (a pixel's foot
// on the contour), so they vary smoothly along the edge: m holds k floats per pixel, read only
// at contour pixels of in; a first float < 0 means no measure. Returns the weight sum (0 = no
// measure nearby); mean gets k values.
double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean);

// ---- shrink_grow.cpp ----
namespace {

// Profiles up to this far each side of the contour move rigidly (edges up to 12 px wide).
constexpr double kProfilePx = 6.0;
// Larger amounts move in passes: the band, and with it the cost, grows with the step.
constexpr double kMaxStepPx = 4.0;

// out = the min (erode) or max (dilate) of m over the (2r + 1)^2 box, separable, on threads;
// straight loops over a padded row so they vectorise (12-25 ms at UHD before). Beyond the
// frame the nearest pixel counts (the frame border is no edge). out may be m.
template <bool kErode>
void boxFilter(const uint8_t* m, uint8_t* out, int w, int h, int r) {
  std::unique_ptr<uint8_t[]> rowsBuf(new uint8_t[size_t(w) * h]);
  uint8_t* rows = rowsBuf.get();
  // Plain locals in the loops: a uint8_t store may alias anything reached through a pointer
  // (the lambda's captures), which kept the loops from vectorising.
  parallelFor(h, 16, [=](int y0, int y1) {
    const int W = w, R = r;
    std::vector<uint8_t> padBuf(size_t(W) + 2 * R);
    uint8_t* pad = padBuf.data();
    for (int y = y0; y < y1; ++y) {
      const uint8_t* in = m + size_t(y) * W;
      uint8_t* o = rows + size_t(y) * W;
      std::copy(in, in + W, pad + R);
      std::fill(pad, pad + R, in[0]);
      std::fill(pad + R + W, pad + 2 * R + W, in[W - 1]);
      std::copy(in, in + W, o);
      for (int d = 0; d <= 2 * R; ++d) {
        const uint8_t* q = pad + d;
        for (int x = 0; x < W; ++x) o[x] = kErode ? std::min(o[x], q[x]) : std::max(o[x], q[x]);
      }
    }
  });
  parallelFor(h, 16, [=](int y0, int y1) {
    const int W = w, H = h, R = r;
    for (int y = y0; y < y1; ++y) {
      uint8_t* o = out + size_t(y) * W;
      std::copy(rows + size_t(y) * W, rows + size_t(y) * W + W, o);
      for (int d = -R; d <= R; ++d) {
        const uint8_t* q = rows + size_t(std::min(std::max(y + d, 0), H - 1)) * W;
        for (int x = 0; x < W; ++x) o[x] = kErode ? std::min(o[x], q[x]) : std::max(o[x], q[x]);
      }
    }
  });
}

}  // namespace

void parallelFor(int n, int minPerThread, const std::function<void(int, int)>& fn) {
  const int t = std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, n / minPerThread}));
  if (t == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  for (int i = 0; i < t; ++i) pool.emplace_back(fn, n * i / t, n * (i + 1) / t);
  for (auto& th : pool) th.join();
}

std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask below(n);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = in[i] < level;
  });
  // Contour pixels (a 4-neighbour on the other side), row-major, found in row bands on threads.
  constexpr int kRows = 64;
  const int bands = (h + kRows - 1) / kRows;
  std::vector<std::vector<int>> found(size_t(std::max(bands, 0)));
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if ((x > 0 && below[i - 1] != below[i]) || (x + 1 < w && below[i + 1] != below[i]) ||
              (y > 0 && below[i - w] != below[i]) || (y + 1 < h && below[i + w] != below[i]))
            found[b].push_back(int(i));
        }
  });
  std::vector<int> edge;
  for (const auto& f : found) edge.insert(edge.end(), f.begin(), f.end());
  // The disc of morphology.cpp distanceWithin (radius band + 1.5), as a half-width per row.
  const double r = band + 1.5;
  const int ri = int(std::floor(r));
  std::vector<int> half(size_t(2 * ri + 1));
  long long disc = 0;
  for (int dy = -ri; dy <= ri; ++dy) {
    half[size_t(dy + ri)] = int(std::floor(std::sqrt(std::max(0.0, r * r - double(dy) * dy))));
    disc += 2 * half[size_t(dy + ri)] + 1;
  }
  if (double(edge.size()) * double(disc) > 12.0 * double(n)) {
    const std::vector<float> sd = signedEdgeDistance(in, w, h, level, band);
    std::unique_ptr<float[]> out(new float[n]);
    std::copy(sd.begin(), sd.end(), out.get());
    return out;
  }
  std::vector<ContourFoot> feet(edge.size());
  parallelFor(int(edge.size()), 256, [&](int e0, int e1) {
    for (int e = e0; e < e1; ++e) feet[size_t(e)] = contourFoot(in, below, w, h, level, edge[size_t(e)]);
  });
  // Each pixel measures to the contour pixels on the other side only (signedEdgeDistance: above
  // pixels to the contour near below pixels, and the reverse); squared, as distanceWithin.
  constexpr float kFar = 1e20f;
  std::unique_ptr<float[]> d2(new float[n]);
  parallelFor(h, 16, [&](int y0, int y1) { std::fill(&d2[size_t(y0) * w], &d2[size_t(y1) * w], kFar); });
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
      const int y0 = b * kRows, y1 = std::min(h, (b + 1) * kRows);
      const auto first = std::lower_bound(edge.begin(), edge.end(), std::max(0, y0 - ri) * w);
      const auto last = std::lower_bound(edge.begin(), edge.end(), std::min(h, y1 + ri) * w);
      for (auto it = first; it != last; ++it) {
        const int q = *it, qx = q % w, qy = q / w;
        const bool qBelow = below[size_t(q)] != 0;
        const ContourFoot& f = feet[size_t(it - edge.begin())];
        const bool line = f.nx != 0.0f || f.ny != 0.0f;
        for (int py = std::max(y0, qy - ri); py < std::min(y1, qy + ri + 1); ++py) {
          const int hw = half[size_t(py - qy + ri)];
          for (int px = std::max(0, qx - hw); px <= std::min(w - 1, qx + hw); ++px) {
            const size_t j = size_t(py) * w + px;
            if ((below[j] != 0) == qBelow) continue;
            const float ox = float(px - qx), oy = float(py - qy), grid = ox * ox + oy * oy;
            float v = grid;
            if (line) {
              const float vx = ox - f.fx, vy = oy - f.fy;
              const float perp = vx * f.nx + vy * f.ny;
              const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
              v = std::min(grid, perp * perp + along * along);
            }
            d2[j] = std::min(d2[j], v);
          }
        }
      }
    }
  });
  const float cap = float(band + 1.0);
  parallelFor(h, 16, [&](int y0, int y1) {  // in place: d2 becomes sd
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float d = d2[i] < kFar ? std::sqrt(d2[i]) : kFar;
      d2[i] = below[i] ? -std::min(d, cap) : std::min(d, cap);
    }
  });
  return d2;
}

bool edgeNormal(const float* sd, int w, int h, int x, int y, double& nx, double& ny) {
  const size_t i = size_t(y) * w + x;
  double g[2];
  for (int axis = 0; axis < 2; ++axis) {
    const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
    const size_t step = axis ? size_t(w) : 1;
    const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
    g[axis] = lo && hi ? (dl + dr) / 2 : dl + dr;
  }
  double len = std::hypot(g[0], g[1]);
  if (len < 0.5) {  // a ridge: the two sides cancel, either side's direction will do
    for (int axis = 0; axis < 2; ++axis) {
      const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
      const size_t step = axis ? size_t(w) : 1;
      const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
      g[axis] = std::fabs(dl) >= std::fabs(dr) ? dl : dr;
    }
    len = std::hypot(g[0], g[1]);
    if (len < 0.5) return false;
  }
  nx = g[0] / len;
  ny = g[1] / len;
  return true;
}

float profileAt(const float* in, const float* sd, int w, int h, int x, int y, double nx, double ny,
                double target) {
  const double s0 = sd[size_t(y) * w + x], move = target - s0;
  double px = x + move * nx, py = y + move * ny;
  // Past the middle of a thin part (or gap) the distance falls again, and the window would read
  // the far side or the next strand: then walk the normal in half pixels and read where the
  // distance reaches the target, or at the middle (the part's peak) if it never does.
  if (std::fabs(sampleBilinear(sd, w, h, px, py) - target) > 0.75) {
    const double dir = move > 0 ? 1.0 : -1.0;
    const int steps = int(std::ceil(2.0 * std::fabs(move)));
    double best = s0, bestT = 0.0;
    for (int k = 1; k <= steps; ++k) {
      const double t = std::min(0.5 * k, std::fabs(move));
      const double v = sampleBilinear(sd, w, h, x + dir * t * nx, y + dir * t * ny);
      if (dir * (v - best) > 0) best = v, bestT = t;
      if (dir * (v - target) >= 0) break;
    }
    if (dir * (best - target) < 0) target = best;
    px = x + dir * bestT * nx;
    py = y + dir * bestT * ny;
  }
  const int cx = int(std::lround(px)), cy = int(std::lround(py));
  // 3 px across the normal's main axis and 7 along the other: near an axis-aligned edge the 3x3
  // pixels' sd lie ~0.8 px apart, and linear interpolation across the kink at the end of a ramp
  // was off by up to 0.065 (3 px ramp); the rows along the edge fill in between (0.03).
  const bool alongY = std::fabs(nx) >= std::fabs(ny);
  const int rx = alongY ? 1 : 3, ry = alongY ? 3 : 1;
  double loD = -1e30, hiD = 1e30;
  float loV = 0.0f, hiV = 0.0f;
  for (int dy = -ry; dy <= ry; ++dy)
    for (int dx = -rx; dx <= rx; ++dx) {
      const int qx = std::min(std::max(cx + dx, 0), w - 1), qy = std::min(std::max(cy + dy, 0), h - 1);
      const size_t j = size_t(qy) * w + qx;
      if (sd[j] <= target && sd[j] > loD) loD = sd[j], loV = in[j];
      if (sd[j] >= target && sd[j] < hiD) hiD = sd[j], hiV = in[j];
    }
  if (loD > -1e29 && hiD < 1e29)
    return hiD > loD ? float(loV + (hiV - loV) * (target - loD) / (hiD - loD)) : loV;
  return loD > -1e29 ? loV : hiD < 1e29 ? hiV : in[size_t(y) * w + x];
}

bool isContour(const float* in, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool a = in[i] >= 0.5f;
  return (x > 0 && (in[i - 1] >= 0.5f) != a) || (x + 1 < w && (in[i + 1] >= 0.5f) != a) ||
         (y > 0 && (in[i - w] >= 0.5f) != a) || (y + 1 < h && (in[i + w] >= 0.5f) != a);
}

double contourSlope(const float* in, const float* sd, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool above = in[i] >= 0.5f;
  double bestDs = 0.0, slope = 0.0;
  const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (const auto& o : nb) {
    const int qx = x + o[0], qy = y + o[1];
    if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
    const size_t j = size_t(qy) * w + qx;
    const double ds = std::fabs(double(sd[i]) - sd[j]);
    if ((in[j] >= 0.5f) != above && ds > bestDs) {
      bestDs = ds;
      slope = std::fabs(double(in[i]) - in[j]) / ds;
    }
  }
  return bestDs < 0.05 ? 0.0 : slope;
}

double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean) {
  double sum = 0.0;
  for (int j = 0; j < k; ++j) mean[j] = 0.0;
  for (int qy = std::max(0, int(std::floor(fy)) - 1); qy <= std::min(h - 1, int(std::floor(fy)) + 2); ++qy)
    for (int qx = std::max(0, int(std::floor(fx)) - 1); qx <= std::min(w - 1, int(std::floor(fx)) + 2); ++qx) {
      const float* v = m + (size_t(qy) * w + qx) * k;
      if (!isContour(in, w, h, qx, qy) || v[0] < 0.0f) continue;
      const double t = std::max(0.0, 1.5 - std::fabs(qx - fx)) * std::max(0.0, 1.5 - std::fabs(qy - fy));
      sum += t;
      for (int j = 0; j < k; ++j) mean[j] += t * v[j];
    }
  if (sum > 0.0)
    for (int j = 0; j < k; ++j) mean[j] /= sum;
  return sum;
}

// Each pass: the body = the matte without its strands. Parts of the >= 0.5 area that an opening
// by the 3x3 box (radius 1.5) removes, under ~3 px wide, are strands: detail that keeps its
// shape (moving them too made every strand of hair above 0.5 a 2 x Amount wide band, and their
// contours pulled the head's edge into lumps at the roots). out(p) = the body's profile at
// sd(p) + step along the normal (edge_field shiftAlongNormal's resampling in distance space).
// The step is full within kProfile + |step| of the contour and tapers to 0 over |step| + 1 px
// beyond, so detail inside near the edge (a grey face) is carried along without a seam and the
// mapping stays monotone. Grow: max with the input (strands stay, nothing gets lower); shrink:
// min with the input, except where the matte is more than the body (strands, hair).
void shrinkGrow(const float* in, float* out, int w, int h, const ShrinkGrowParams& p,
                ShrinkGrowStats* stats) {
  ShrinkGrowStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const double amount = p.amountPx;
  const int passes = amount == 0.0 ? 0 : int(std::ceil(std::fabs(amount) / kMaxStepPx));
  const double step = passes ? amount / passes : 0.0, a = std::fabs(step);
  const double rigid = kProfilePx + a, band = rigid + a + 1.0;
  // Each pass reads src and writes dst; the last one writes out unless out is in.
  const float* src = in;
  std::unique_ptr<float[]> prev, next, body(passes ? new float[n] : nullptr),
      falloffs(passes ? new float[n] : nullptr);
  std::unique_ptr<uint8_t[]> solid(passes ? new uint8_t[n] : nullptr), core(passes ? new uint8_t[n] : nullptr);
  for (int pass = 0; pass < passes; ++pass) {
    const bool last = pass + 1 == passes;
    if (!(last && out != in)) next.reset(new float[n]);
    float* dst = last && out != in ? out : next.get();
    // The opening: erode by the 3x3 box, dilate back; and 1 px more, since the opening also
    // takes single staircase pixels off a smooth contour (a strand keeps a stub that short).
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) solid[i] = src[i] >= 0.5f;
    });
    boxFilter<true>(solid.get(), core.get(), w, h, 1);
    boxFilter<false>(core.get(), core.get(), w, h, 2);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) body[i] = solid[i] && !core[i] ? 0.0f : src[i];
    });
    const std::unique_ptr<float[]> sdBuf = signedEdgeDistanceThreaded(body.get(), w, h, 0.5f, band);
    const float* sd = sdBuf.get();
    // Outside the contour the body's profile ends where the edge's own falloff does (0.5 /
    // its slope at the contour, + 1 px) and only falls going out (no pixel more than any nearer
    // it along the normal): so hair below 0.5 next to the edge stays detail and is not carried
    // along (it was, as a second copy Amount px out). Below 0.5: sd stays as it is.
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x)
          if (isContour(body.get(), w, h, x, y)) {
            const double slope = contourSlope(body.get(), sd, w, h, x, y);
            falloffs[size_t(y) * w + x] = slope > 0.0 ? float(0.5 / slope) : -1.0f;
          }
    });
    std::unique_ptr<float[]> mono(new float[n]);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          double v = body[i], nx, ny, falloff;
          if (v > 0.0 && sd[i] < 0.0 && sd[i] > -band && edgeNormal(sd, w, h, x, y, nx, ny)) {
            if (footMean(body.get(), falloffs.get(), 1, w, h, x - sd[i] * nx, y - sd[i] * ny, &falloff) > 0.0 &&
                -sd[i] > falloff + 1.0)
              v = 0.0;
            for (double t = 0.5; t < -sd[i] && v > 0.0; t += 0.5)
              v = std::min(v, double(sampleBilinear(body.get(), w, h, x + t * nx, y + t * ny)));
          }
          mono[i] = float(v);
        }
    });
    body.swap(mono);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          dst[i] = src[i];
          const double s = sd[i], k = std::min(1.0, (band - std::fabs(s)) / (a + 1.0));
          double nx, ny;
          if (k <= 0.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
          const float v = profileAt(body.get(), sd, w, h, x, y, nx, ny, s + step * k);
          dst[i] = step > 0 ? std::max(src[i], v) : std::min(src[i], std::max(v, src[i] - body[i]));
        }
    });
    if (!last) {
      prev.swap(next);
      src = prev.get();
    }
  }
  const float* result = passes == 0 ? in : out != in ? out : next.get();
  std::atomic<long long> changed(0);
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) c += result[i] != in[i];
    changed += c;
    if (result != out) std::copy(result + size_t(y0) * w, result + size_t(y1) * w, out + size_t(y0) * w);
  });
  st.pixelsChanged = changed;
  st.passes = passes;
  if (stats) *stats = st;
}
}  // namespace sg

namespace ew {
using namespace sg;  // edge_width.cpp included shrink_grow.h for its helpers

// ---- edge_width.h ----
// Operation A5, Edge Width (CLAUDE.md "Plugin 1 operations"): gives the matte's edge one
// falloff width along the whole contour. Hard-cut (aliased) stretches become soft, overly soft
// ones tighter; the 0.5 contour stays where it is, and the levels on either side of the edge
// (a grey face behind it, hair beyond it) are kept: only the transition between them is
// retimed, into a linear ramp. Width = the ramp's full 0-100 % width (10-90 % = 0.8 x Width).
// Parts and gaps narrower than the target width keep their own profile (hair is not smeared or
// erased); an edge already within ~3 % of the target is not touched. A hard cut (or 1 px edge)
// gets the target ramp drawn from its levels, on the distance averaged along the edge over
// +-Width / 2 (up to 4 px): it follows the line, not the pixel staircase, except within ~4 px of
// the steps at Width 2 (residual up to ~0.24); a full de-aliasing is Contour Smooth's job.



struct EdgeWidthParams {
  double widthPx = 2.0;  // target 0-100 % width of the edge; clamped to 0.5-16 px
  double amount = 1.0;   // 0 = the input, 1 = the target width; clamped to 0-1
};

struct EdgeWidthStats {
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void edgeWidth(const float* in, float* out, int w, int h, const EdgeWidthParams& p,
               EdgeWidthStats* stats = nullptr);

// ---- edge_width.cpp ----
namespace {

// Edges are measured up to this far each side of the contour: an input edge up to ~16 px wide
// is retimed whole, a softer one only over its inner 16 px.
constexpr double kBandPx = 10.0;

// What an inside contour pixel knows about its edge: how much it may be retimed (0-1; < 0: no
// usable measure), how far the transition reaches inside and outside (px), and the inside's
// share of the level span ((hi - 0.5) / (hi - lo); 0.5 for an edge from 0 to 1), and the levels.
// Only contour pixels are written (the array is not initialised); footMean reads 7 floats.
struct EdgeInfo {
  float wt, eIn, eOut, fIn, hi, lo, room;
};

}  // namespace

// sd = the signed distance to the 0.5 contour. At each contour pixel: the levels the edge runs
// between (hi inside, lo outside), read just past the transition, and how far the transition
// reaches each side, e = 2 x the area between the profile and its level / the level's distance
// from 0.5: for a linear ramp exactly its extent (the profile read along the normal from many
// pixels, so a 1 px edge's clamped neighbours don't shorten it, as the slope between two pixels
// did). The target is the linear ramp from lo to hi, Width wide, through the same contour: each
// side's share of Width by its share of the level span. Each pixel in the band reads the input's
// profile at g(sd): the transition's e px each side mapped linearly onto the target's, beyond
// that the input as it was (its level where the old transition was wider). So the levels and
// the contour stay, and nothing beyond the transition changes.
void edgeWidth(const float* in, float* out, int w, int h, const EdgeWidthParams& p,
               EdgeWidthStats* stats) {
  EdgeWidthStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  std::unique_ptr<float[]> copy;
  const float* src = in;
  if (out == in) {  // the profile is read around each pixel
    copy.reset(new float[n]);
    parallelFor(h, 16, [&](int y0, int y1) {
      std::copy(in + size_t(y0) * w, in + size_t(y1) * w, &copy[size_t(y0) * w]);
    });
    src = copy.get();
  }
  const double width = std::min(std::max(p.widthPx, 0.5), 2.0 * (kBandPx - 2.0));
  const double half = width / 2;
  const double amount = std::min(std::max(p.amount, 0.0), 1.0);
  const std::unique_ptr<float[]> sdBuf = signedEdgeDistanceThreaded(src, w, h, 0.5f, kBandPx);
  const float* sd = sdBuf.get();
  std::unique_ptr<EdgeInfo[]> info(new EdgeInfo[n]);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        if (!isContour(src, w, h, x, y)) continue;
        const size_t i = size_t(y) * w + x;
        EdgeInfo& e = info[i];
        e.wt = -1.0f;
        if (src[i] < 0.5f) continue;  // measured from the inside's contour pixels only (half the work)
        const double slope = contourSlope(src, sd, w, h, x, y);
        double nx, ny;
        if (slope <= 0.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
        const double e0 = std::min(0.5 / slope, kBandPx - 2.0);
        const double hi = profileAt(src, sd, w, h, x, y, nx, ny, e0 + 1.0);
        const double lo = profileAt(src, sd, w, h, x, y, nx, ny, -(e0 + 1.0));
        if (hi - lo < 0.02) continue;
        // Midpoint sums in 8 steps over twice the slope's estimate + 1 px.
        const double reachOld = std::min(kBandPx - 2.0, 2.0 * e0 + 1.0), step = reachOld / 8;
        double areaIn = 0.0, areaOut = 0.0;
        for (double t = step / 2; t < reachOld; t += step) {
          const double a = profileAt(src, sd, w, h, x, y, nx, ny, t);
          const double b = profileAt(src, sd, w, h, x, y, nx, ny, -t);
          areaIn += (hi - std::min(std::max(a, 0.5), hi)) * step;
          areaOut += (std::min(std::max(b, lo), 0.5) - lo) * step;
        }
        e.eIn = hi - 0.5 > 0.01 ? float(std::min(2.0 * areaIn / (hi - 0.5), kBandPx - 2.0)) : 0.0f;
        e.eOut = 0.5 - lo > 0.01 ? float(std::min(2.0 * areaOut / (0.5 - lo), kBandPx - 2.0)) : 0.0f;
        e.fIn = float((hi - 0.5) / (hi - lo));
        e.hi = float(hi);
        e.lo = float(lo);
        // Thin parts and gaps: going Width / 2 + 1 px across from here, the distance should grow
        // by as much; past the middle of a strand (or gap) at r px it falls again: 2r - reach.
        // room = the widest edge that fits, 2 x (r - 0.5) (>= Width + 1 where nothing is in the
        // way). Widening is capped to it, so a wedge's edges narrow smoothly into its tip
        // (switching the change off there left a hard stub inside soft flanks) and a strand or
        // gap too thin to widen stays as it is. wt (full from half a pixel short of the ideal
        // reach: at a staircase corner the tilted normal measures ~0.5 px less) still keeps
        // narrowing off thin parts.
        const double reach = half + 1.0;
        const double rIn = sampleBilinear(sd, w, h, x + reach * nx, y + reach * ny) - sd[i];
        const double rOut = sd[i] - sampleBilinear(sd, w, h, x - reach * nx, y - reach * ny);
        e.wt = float(std::min(std::max(std::min(rIn, rOut) - half + 0.5, 0.0), 1.0));
        e.room = float(std::max(std::min(rIn, rOut) + reach - 1.0, 0.0));
      }
  });
  std::atomic<long long> changed(0);
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        out[i] = src[i];
        const double s = sd[i];
        double nx, ny;
        if (std::fabs(s) > kBandPx - 1.0 || amount == 0.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
        // The edge's measures at this pixel's foot on the contour.
        double m[7];
        if (footMean(src, &info[0].wt, 7, w, h, x - s * nx, y - s * ny, m) <= 0.0 || m[0] < 0.0) continue;
        const double eOld = s >= 0 ? m[1] : m[2], as = std::fabs(s), share = s >= 0 ? m[3] : 1.0 - m[3];
        double eNew = std::max(width * share, 0.05), wt = m[0];
        if (eNew > eOld) {  // widening: up to the room (less 0.1 px: a 2 px gap measured a hair
          // wider than it is), never narrower than it was
          eNew = std::max(std::min(eNew, (m[6] - 0.1) * share), eOld);
          wt = 1.0;
        }
        const double widthHere = eNew / std::max(share, 1e-6);
        // An edge already within ~3 % of the target is left as it is (bit for bit): retiming it
        // would only add the measurement's noise (up to 0.04 on a near-vertical 2 px edge).
        // Full from 7 % off, so there is no seam where an edge's width crosses the line.
        const double off = eOld > 0.0 ? std::fabs(std::log(eOld / eNew)) : 1.0;
        const double k = wt * std::min(std::max((off - 0.03) / 0.04, 0.0), 1.0);
        const double g = as <= eNew ? s * eOld / eNew : std::copysign(std::max(as, eOld), s);
        if (g == s || k <= 0.0) continue;
        // A transition under ~1 px (a hard cut, a 1 px edge) has no shape worth keeping, and
        // resampling it failed: its pixels lie ~0.5 px from the contour, except at a staircase
        // corner (~0.1), so the stretched step jumped by up to 0.46 there (a comb along a near-
        // vertical hard edge, saw teeth on slants). There the target ramp is drawn from the
        // levels instead, on sd averaged along the edge over +-Width / 2 (up to 4 px), so it
        // follows the line rather than the pixel staircase; from 1 px on, the resampled profile.
        const double synth = std::min(std::max((1.0 - eOld) / 0.5, 0.0), 1.0);
        double v = synth < 1.0 ? profileAt(src, sd, w, h, x, y, nx, ny, g) : 0.0;
        if (synth > 0.0) {
          const int reach = int(std::min(widthHere / 2, 4.0));
          double sum = s;
          for (int t = 1; t <= reach; ++t)
            sum += sampleBilinear(sd, w, h, x - t * ny, y + t * nx) + sampleBilinear(sd, w, h, x + t * ny, y - t * nx);
          const double sa = sum / (2 * reach + 1);
          const double ramp = sa >= 0 ? std::min(m[4], 0.5 + (m[4] - 0.5) * sa / eNew)
                                      : std::max(m[5], 0.5 + (0.5 - m[5]) * sa / eNew);
          v = synth * ramp + (1.0 - synth) * v;
        }
        out[i] = float(src[i] + amount * k * (v - src[i]));
        c += out[i] != src[i];
      }
    changed += c;
  });
  st.pixelsChanged = changed;
  if (stats) *stats = st;
}
}  // namespace ew

namespace cs {

// ---- contour_smooth.h ----
// Operation A4, Contour Smooth (CLAUDE.md "Plugin 1 operations"): takes the wiggle and chatter
// off the matte's Edge At contour line while keeping the edge's softness profile across it and
// the real shape along it (corners, creases, thin features). Only the edge band changes.



struct ContourSmoothParams {
  double smoothPx = 30.0;   // wiggle wavelengths up to this (px) are taken off the contour
  double maxShiftPx = 1.5;  // the edge moves at most this far (px); larger deviations are shape
  float edgeAt = 0.5f;      // the contour that is smoothed
  double cornerDeg = 20.0;  // contour turning more than this is a corner: not averaged across
};

struct ContourSmoothStats {
  long long pixelsChanged = 0;
  long long contourSamples = 0;  // boundary pixels the contour was fitted on
  double largestShiftPx = 0.0;   // largest |edge move| applied
  bool skipped = false;          // contour everywhere (a noisy matte): left as it was
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void contourSmooth(const float* in, float* out, int w, int h, const ContourSmoothParams& p,
                   ContourSmoothStats* stats = nullptr);

// ---- contour_smooth.cpp ----
namespace {

// How far beyond the contour an edge's soft profile may reach and still travel with it (a
// 12 px wide soft edge); pixels farther out are never touched.
constexpr double kProfilePx = 6.0;

// Gaussian sigma per px of Smooth wavelength: the quadratic fit's response (1 + x/2) e^(-x/2),
// x = (2 pi sigma / wavelength)^2, keeps 20 % of a wiggle at the Smooth wavelength, 0.5 % at half.
constexpr double kSigmaPerWavelength = 0.4;

// Features (or gaps) up to 2 x kThinPx wide are not moved, from 2 x kThickPx on fully.
constexpr double kThinPx = 1.5, kThickPx = 3.5;

// A fit with less support than kSupportNone of a straight contour's does not move its sample,
// from kSupportFull on fully (fitMove).
constexpr double kSupportNone = 0.3, kSupportFull = 0.6;

// Every move is shrunk by this (px), smaller ones become 0: the foot points' own scatter
// (+-0.04 px on a clean slant) is no wiggle, and resampling for it only adds noise.
constexpr float kDeadPx = 0.05f;

// How far the profile across an edge may turn back (alpha) and still count as one edge.
constexpr float kMonotone = 0.05f;

// A move more than kSpike px off the median of the moves within kMedianPx is replaced by that
// median (see contourSmooth).
constexpr float kSpike = 0.25f;
constexpr int kMedianPx = 3;

// A point on the contour: a boundary pixel's foot point and its outward unit normal.
struct Sample {
  float x, y, nx, ny;
  int pix;     // the boundary pixel it was measured at
  bool below;  // that pixel's side
};

// Samples bucketed in square cells, for the neighbours within a radius.
struct Grid {
  int cell, cw, ch;
  std::vector<int> start, items;
  Grid(const std::vector<Sample>& s, int w, int h, int cellPx)
      : cell(std::max(cellPx, 1)), cw(w / cell + 1), ch(h / cell + 1), start(size_t(cw) * ch + 1, 0) {
    for (const Sample& p : s) ++start[index(p.x, p.y) + 1];
    for (size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    items.resize(s.size());
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t k = 0; k < s.size(); ++k) items[size_t(fill[index(s[k].x, s[k].y)]++)] = int(k);
  }
  size_t index(float x, float y) const {
    const int cx = std::min(std::max(int(x) / cell, 0), cw - 1), cy = std::min(std::max(int(y) / cell, 0), ch - 1);
    return size_t(cy) * cw + cx;
  }
  template <class F>
  void near(float x, float y, double r, F fn) const {
    const int x0 = std::max(int((x - r) / cell), 0), x1 = std::min(int((x + r) / cell), cw - 1);
    const int y0 = std::max(int((y - r) / cell), 0), y1 = std::min(int((y + r) / cell), ch - 1);
    for (int cy = y0; cy <= y1; ++cy)
      for (int cx = x0; cx <= x1; ++cx) {
        const size_t c = size_t(cy) * cw + cx;
        for (int k = start[c]; k < start[c + 1]; ++k) fn(items[size_t(k)]);
      }
  }
};

// Weighted least squares h(u) = a + b u + c u^2 from the sums of w u^k (k = 0-4), w h u^k
// (k = 0-2).
struct Fit {
  double m[5] = {0, 0, 0, 0, 0}, v[3] = {0, 0, 0};
  void add(double wt, double u, double hgt) {
    double p = wt;
    for (int e = 0; e < 5; ++e, p *= u) {
      m[e] += p;
      if (e < 3) v[e] += p * hgt;
    }
  }
  // The fit at u = 0: quadratic (Cramer), else linear (neighbours on one side only), else the
  // mean.
  double at0() const {
    if (m[0] <= 0.0) return 0.0;
    const double det = m[0] * (m[2] * m[4] - m[3] * m[3]) - m[1] * (m[1] * m[4] - m[3] * m[2]) +
                       m[2] * (m[1] * m[3] - m[2] * m[2]);
    if (det > 1e-6 * m[0] * m[0] * m[0])
      return (v[0] * (m[2] * m[4] - m[3] * m[3]) - m[1] * (v[1] * m[4] - m[3] * v[2]) +
              m[2] * (v[1] * m[3] - m[2] * v[2])) / det;
    const double d2 = m[0] * m[2] - m[1] * m[1];
    if (d2 > 1e-6 * m[0] * m[0]) return (v[0] * m[2] - m[1] * v[1]) / d2;
    return v[0] / m[0];
  }
};

// The smoothed contour at sample j, as an outward move of j's foot point along its normal (px):
// a weighted quadratic h(t) through the neighbours in j's tangent frame, so curves and slants
// are kept and only the wiggle is averaged away (a plain mean, like blurring the distance
// field, shrinks a 34 px arm by sigma^2 / 2r). Weights: Gaussian in t; samples whose normal turns more than the
// corner angle from j's are left out (a corner, the far side of a thin feature), tapered over
// the last 10 deg. Inside that all count fully: weighting by normal agreement (bilateral) kept
// the wiggle, which tilts the normal too. Also tried: a fit on each side of j alone, the one
// with the smaller residual winning (ENO / Kuwahara): it kept kinks but followed half a wiggle
// period, so the wiggle stayed.
double fitMove(const std::vector<Sample>& s, const Grid& grid, int j, double sigma, double cosCorner,
               double cosFull) {
  const Sample& o = s[size_t(j)];
  const double r = 2.5 * sigma;
  Fit fit;
  grid.near(o.x, o.y, r, [&](int k) {
    const Sample& q = s[size_t(k)];
    const double dot = o.nx * q.nx + o.ny * q.ny;
    if (dot <= cosCorner) return;
    const double dx = q.x - o.x, dy = q.y - o.y;
    const double t = -dx * o.ny + dy * o.nx, hgt = dx * o.nx + dy * o.ny;
    // A wiggle tilts the contour by up to ~30 deg; farther off the tangent is another contour.
    if (dx * dx + dy * dy > r * r || std::fabs(hgt) > 1.5 + 0.6 * std::fabs(t)) return;
    const double u = t / sigma;
    fit.add(std::exp(-0.5 * u * u) * std::min(1.0, (dot - cosCorner) / (cosFull - cosCorner)), u, hgt);
  });
  // Little support (a crease's apex, where both legs turn away; a tip): shape, not wiggle. A
  // straight contour has ~2 samples per px (both sides), sum of weights 2 sqrt(2 pi) sigma.
  const double support = fit.m[0] / (2.0 * std::sqrt(2.0 * M_PI) * sigma);
  const double trust = std::min(1.0, std::max(0.0, (support - kSupportNone) / (kSupportFull - kSupportNone)));
  // The fitted curve's height at t = 0 is its distance along j's own normal, the direction the
  // pixels move in (shiftAlongNormal); the distance across the fitted curve instead moved steep
  // wiggle flanks ~6 % short.
  return trust * fit.at0();
}

// The edge band, stored only in the 16 x 16 tiles it touches: full-frame buffers cost more
// than the whole smoothing (UHD, BFD: ~8 ms per fresh 33 MB buffer, 9 of them were 75 of 95
// ms).
struct Band {
  static constexpr int kTile = 16;
  int w, h, tw, th;
  std::vector<int> slot;  // per tile: its storage index, -1 = outside the band
  std::vector<float> d2;  // squared distance to the contour (pixels of the other side)
  std::vector<int> nearest;  // the sample it was measured from, -1 = none
  Band(int w, int h) : w(w), h(h), tw((w + kTile - 1) / kTile), th((h + kTile - 1) / kTile), slot(size_t(tw) * th, -1) {}
  int index(int x, int y) const {
    const int s = slot[size_t(y / kTile) * tw + x / kTile];
    return s < 0 ? -1 : s * kTile * kTile + (y % kTile) * kTile + x % kTile;
  }
};

}  // namespace

void contourSmooth(const float* in, float* out, int w, int h, const ContourSmoothParams& p,
                   ContourSmoothStats* stats) {
  ContourSmoothStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (out != in) std::copy(in, in + n, out);
  if (n == 0 || p.smoothPx <= 0.0 || p.maxShiftPx <= 0.0) {
    if (stats) *stats = st;
    return;
  }

  // Contour samples on both sides of the Edge At contour (pixels beyond the frame don't count:
  // the frame border is no edge). Hair below Edge At has no contour and gives none.
  Mask below(n);
  for (size_t i = 0; i < n; ++i) below[i] = in[i] < p.edgeAt;
  std::vector<Sample> samples;  // in scan order
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const bool b = below[i];
      if (!((x > 0 && below[i - 1] != b) || (x + 1 < w && below[i + 1] != b) ||
            (y > 0 && below[i - w] != b) || (y + 1 < h && below[i + w] != b)))
        continue;
      const ContourFoot f = contourFoot(in, below, w, h, p.edgeAt, int(i));
      const float sg = b ? -1.0f : 1.0f;  // outward: a below pixel's normal points inward
      samples.push_back({x + f.fx, y + f.fy, sg * f.nx, sg * f.ny, int(i), b});
    }
  st.contourSamples = (long long)samples.size();

  // The signed distance to the contour within the band, as signedEdgeDistance (decision 20)
  // computes it, plus which sample each pixel is nearest to: each sample stamps a disc onto the
  // pixels of the other side. Done here because signedEdgeDistance works on full frames
  // (~35 ms at UHD) and gives no nearest sample. Rows in stripes of 2 radii, even stripes then
  // odd ones, so threads never write the same pixel.
  const double band = p.maxShiftPx + kProfilePx;
  const double r = band + 1.5;  // a foot point lies within 1.5 px of its pixel (morphology.h)
  const int ri = int(std::ceil(r));
  std::vector<std::pair<int, int>> disc;
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if (dx * dx + dy * dy <= r * r) disc.emplace_back(dx, dy);
  // A noisy matte (a contour at 1 in 16 pixels or more) has no line to smooth, and would cost
  // seconds.
  if (samples.size() > n / 16) {
    st.skipped = true;
    if (stats) *stats = st;
    return;
  }
  Band bd(w, h);
  int tiles = 0;
  for (const Sample& s : samples) {
    const int x = s.pix % w, y = s.pix / w, T = Band::kTile;
    for (int ty = std::max(y - ri, 0) / T; ty <= std::min(y + ri, h - 1) / T; ++ty)
      for (int tx = std::max(x - ri, 0) / T; tx <= std::min(x + ri, w - 1) / T; ++tx)
        if (bd.slot[size_t(ty) * bd.tw + tx] < 0) bd.slot[size_t(ty) * bd.tw + tx] = tiles++;
  }
  bd.d2.assign(size_t(tiles) * Band::kTile * Band::kTile, 1e30f);
  bd.nearest.assign(bd.d2.size(), -1);
  const int stripe = 2 * ri + 1, stripes = (h + stripe - 1) / stripe;
  for (int phase = 0; phase < 2; ++phase)
    parallelFor((stripes + 1 - phase) / 2, 1, [&](int s0, int s1) {
      for (int sIdx = s0; sIdx < s1; ++sIdx) {
        const int y0 = (2 * sIdx + phase) * stripe, y1 = std::min(y0 + stripe, h);
        auto first = std::lower_bound(samples.begin(), samples.end(), size_t(y0) * w,
                                      [](const Sample& a, size_t v) { return size_t(a.pix) < v; });
        for (auto it = first; it != samples.end() && it->pix < y1 * w; ++it) {
          const int x = it->pix % w, y = it->pix / w;
          const float fx = it->x - x, fy = it->y - y;
          for (const auto& o : disc) {
            const int px = x + o.first, py = y + o.second;
            if (px < 0 || py < 0 || px >= w || py >= h || below[size_t(py) * w + px] == it->below) continue;
            const float ox = float(o.first), oy = float(o.second), vx = ox - fx, vy = oy - fy;
            const float perp = vx * it->nx + vy * it->ny;
            const float along = std::max(0.0f, std::fabs(vy * it->nx - vx * it->ny) - 0.5f);
            const float dd = std::min(ox * ox + oy * oy, perp * perp + along * along);
            const int k = bd.index(px, py), j = int(it - samples.begin());
            // Ties go to the lower sample, so the result doesn't depend on the stamping order.
            if (dd < bd.d2[size_t(k)] || (dd == bd.d2[size_t(k)] && j < bd.nearest[size_t(k)]))
              bd.d2[size_t(k)] = dd, bd.nearest[size_t(k)] = j;
          }
        }
      }
    });
  // Signed distance, > 0 inside, +-(band + 1) beyond the band (as signedEdgeDistance).
  const float cap = float(band + 1.0);
  for (size_t k = 0; k < bd.d2.size(); ++k) bd.d2[k] = bd.nearest[k] >= 0 ? std::min(std::sqrt(bd.d2[k]), cap) : cap;
  auto sd = [&](int x, int y) {
    const int k = bd.index(x, y);
    const float d = k < 0 ? cap : bd.d2[size_t(k)];
    return below[size_t(y) * w + x] ? -d : d;
  };
  auto sdNear = [&](double x, double y) {
    const int xi = int(std::lround(x)), yi = int(std::lround(y));
    return xi < 0 || yi < 0 || xi >= w || yi >= h ? 0.0f : sd(xi, yi);
  };

  const double sigma = kSigmaPerWavelength * p.smoothPx;
  const Grid grid(samples, w, h, std::max(4, int(std::ceil(sigma))));
  const double corner = std::min(std::max(p.cornerDeg, 10.0), 179.0) * M_PI / 180.0;
  const double cosCorner = std::cos(corner), cosFull = std::cos(corner - 10.0 * M_PI / 180.0);
  std::vector<float> fitted(samples.size(), 0.0f);
  std::vector<uint8_t> kept(samples.size(), 1);  // thin or not one edge: never moves
  parallelFor(int(samples.size()), 256, [&](int j0, int j1) {
    for (int j = j0; j < j1; ++j) {
      const Sample& o = samples[size_t(j)];
      // Thin features (hair strands, narrow gaps) are left alone: their contour is too poorly
      // sampled to move, and a 1.5 px move erases them. Half-thickness = the deepest |sd| along
      // the normal on either side, within kThickPx.
      double inner = 0.0, outer = 0.0;
      for (double k = 0.5; k <= kThickPx; k += 0.5) {
        inner = std::max(inner, double(sdNear(o.x - k * o.nx, o.y - k * o.ny)));
        outer = std::max(outer, -double(sdNear(o.x + k * o.nx, o.y + k * o.ny)));
      }
      const double thick =
          std::min(1.0, std::max(0.0, (std::min(inner, outer) - kThinPx) / (kThickPx - 0.5 - kThinPx)));
      if (thick == 0.0) continue;
      // Nor is an edge with something else in its profile (a strand below Edge At touching it,
      // a hair root): rising again outside or falling again inside. Its foot points are tilted
      // by the neighbour, and moving it drags the strand along.
      bool monotone = true;
      float prevIn = sampleBilinear(in, w, h, o.x, o.y), prevOut = prevIn;
      for (double k = 1.0; k <= kThickPx + 0.5 && monotone; k += 1.0) {
        const float vi = sampleBilinear(in, w, h, o.x - k * o.nx, o.y - k * o.ny);
        const float vo = sampleBilinear(in, w, h, o.x + k * o.nx, o.y + k * o.ny);
        monotone = vi >= prevIn - kMonotone && vo <= prevOut + kMonotone;
        prevIn = vi;
        prevOut = vo;
      }
      if (!monotone) continue;
      kept[size_t(j)] = 0;
      const double m = thick * fitMove(samples, grid, j, sigma, cosCorner, cosFull);
      fitted[size_t(j)] = float(std::min(std::max(m, -p.maxShiftPx), p.maxShiftPx));
    }
  });
  // A sample right at a crease (the normal turns over 1-2 px there) can still see both legs and
  // moves alone: a spike against the median of the moves within kMedianPx (same side of the
  // corner, kept samples left out) is replaced by it. Only spikes: a plain median flattened a
  // steep wiggle's extremes.
  std::vector<float> move(samples.size());
  parallelFor(int(samples.size()), 1024, [&](int j0, int j1) {
    std::vector<float> near;
    for (int j = j0; j < j1; ++j) {
      const Sample& o = samples[size_t(j)];
      move[size_t(j)] = 0.0f;
      if (kept[size_t(j)]) continue;
      near.clear();
      grid.near(o.x, o.y, kMedianPx, [&](int k) {
        const Sample& q = samples[size_t(k)];
        if (!kept[size_t(k)] && std::hypot(q.x - o.x, q.y - o.y) <= kMedianPx &&
            o.nx * q.nx + o.ny * q.ny > cosCorner)
          near.push_back(fitted[size_t(k)]);
      });
      std::nth_element(near.begin(), near.begin() + near.size() / 2, near.end());
      const float med = near[near.size() / 2];
      const float m = std::fabs(fitted[size_t(j)] - med) > kSpike ? med : fitted[size_t(j)];
      move[size_t(j)] = std::copysign(std::max(0.0f, std::fabs(m) - kDeadPx), m);
    }
  });

  // Each pixel in the band moves with its nearest contour sample along the distance's normal:
  // out(p) = in(p + s n), bilinear. Not shiftAlongNormal's distance-space interpolation: its
  // choice of the tightest-bracketing pixel pair jumps between rows on slanted edges, which put
  // a 0.13 px row-to-row jitter (aliasing) into a smoothed slant; bilinear is smooth, exact on
  // the ramp and coverage-correct on a hard edge, and moved hair strands near the edge far less
  // (synth hair crop, truth as input: max change 0.38 -> 0.07).
  std::vector<float> res(bd.d2.size());
  std::vector<int> tileList(size_t(bd.tw) * bd.th);
  for (size_t t = 0; t < tileList.size(); ++t) tileList[t] = bd.slot[t] < 0 ? -1 : int(t);
  tileList.erase(std::remove(tileList.begin(), tileList.end(), -1), tileList.end());
  std::vector<double> largest(tileList.size(), 0.0);
  parallelFor(int(tileList.size()), 16, [&](int t0, int t1) {
    for (int t = t0; t < t1; ++t) {
      const int T = Band::kTile, bx = tileList[size_t(t)] % bd.tw * T, by = tileList[size_t(t)] / bd.tw * T;
      for (int y = by; y < std::min(by + T, h); ++y)
        for (int x = bx; x < std::min(bx + T, w); ++x) {
          const int k = bd.index(x, y);
          const size_t i = size_t(y) * w + x;
          res[size_t(k)] = in[i];
          if (bd.nearest[size_t(k)] < 0 || std::fabs(sd(x, y)) > band) continue;
          const float s = move[size_t(bd.nearest[size_t(k)])];
          if (s == 0.0f) continue;
          largest[size_t(t)] = std::max(largest[size_t(t)], double(std::fabs(s)));
          const int xl = x > 0 ? x - 1 : x, xr = x + 1 < w ? x + 1 : x;
          const int yu = y > 0 ? y - 1 : y, yd = y + 1 < h ? y + 1 : y;
          const double gx = xr > xl ? (sd(xr, y) - sd(xl, y)) / (xr - xl) : 0.0;
          const double gy = yd > yu ? (sd(x, yd) - sd(x, yu)) / (yd - yu) : 0.0;
          const double g = std::hypot(gx, gy);
          if (g < 0.5) continue;
          res[size_t(k)] = sampleBilinear(in, w, h, x + s * gx / g, y + s * gy / g);
        }
    }
  });
  for (size_t t = 0; t < tileList.size(); ++t) {
    st.largestShiftPx = std::max(st.largestShiftPx, largest[t]);
    const int T = Band::kTile, bx = tileList[t] % bd.tw * T, by = tileList[t] / bd.tw * T;
    for (int y = by; y < std::min(by + T, h); ++y)
      for (int x = bx; x < std::min(bx + T, w); ++x) {
        const float v = res[size_t(bd.index(x, y))];
        st.pixelsChanged += v != in[size_t(y) * w + x];
        out[size_t(y) * w + x] = v;
      }
  }
  if (stats) *stats = st;
}
}  // namespace cs

namespace sc {

// ---- solid_core.h ----
// Operation 2, Solid Core (CLAUDE.md decisions 15, 16): every pixel more than Depth + Feather px
// inside the matte's Edge At contour becomes 1.0, with a smooth ramp over the Feather px before
// it; the outer Depth px of edge are not touched and no pixel gets darker.



struct SolidCoreParams {
  double depthPx = 6.0;  // core starts this far inside the edge
  float edgeAt = 0.5f;   // the contour the depth is measured from
  double featherPx = 2.0;  // ramp from the input to 1.0 over this many px; 0 = a hard core
};

struct SolidCoreStats {
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats = nullptr);

// ---- solid_core.cpp ----
void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats) {
  SolidCoreStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  // Near the edge = within Depth of a pixel below Edge At. Pixels beyond the frame don't
  // count: a subject cut by the frame has no edge there.
  Mask below(n);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = in[i] < p.edgeAt;
  });
  if (p.featherPx <= 0.0) {  // the hard core of decision 15
    const Mask nearEdge = dilate(below, w, h, p.depthPx);
    for (size_t i = 0; i < n; ++i) {
      const bool lift = !nearEdge[i] && in[i] < 1.0f;
      st.pixelsChanged += lift;
      out[i] = lift ? 1.0f : in[i];
    }
    if (stats) *stats = st;
    return;
  }
  // A soft core (decision 16): a binary one leaves a jagged step where its pixel staircase
  // crosses a grey area. D = the distance to the Edge At contour, refined below pixel spacing
  // (contourFoot in ops/edge_field).
  auto foot = [&](int i) { return contourFoot(in, below, w, h, p.edgeAt, i); };
  const std::vector<float> d = distanceWithin(below, w, h, p.depthPx + p.featherPx, foot);
  std::atomic<long long> changed{0};
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      if (d[i] <= p.depthPx || in[i] >= 1.0f) {  // the edge band, or already solid
        out[i] = in[i];
        continue;
      }
      // smoothstep from Depth to Depth + Feather.
      const double t = std::min(1.0, (d[i] - p.depthPx) / p.featherPx);
      const float o = std::max(in[i], float(t * t * (3.0 - 2.0 * t)));
      c += o != in[i];
      out[i] = o;
    }
    changed += c;
  });
  st.pixelsChanged = changed;
  if (stats) *stats = st;
}
}  // namespace sc

}  // namespace old_
