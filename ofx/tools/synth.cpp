#include "synth.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace synth {

namespace {

double clamp01(double v) { return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v; }

// Signed distances in 1080-line units, > 0 inside.
double sdCircle(double x, double y, double cx, double cy, double r) { return r - std::hypot(x - cx, y - cy); }

double sdCapsule(double x, double y, double ax, double ay, double bx, double by, double r) {
  const double px = x - ax, py = y - ay, vx = bx - ax, vy = by - ay;
  const double t = clamp01((px * vx + py * vy) / (vx * vx + vy * vy));
  return r - std::hypot(px - t * vx, py - t * vy);
}

double sdRoundBox(double x, double y, double cx, double cy, double hx, double hy, double r) {
  const double qx = std::fabs(x - cx) - hx + r, qy = std::fabs(y - cy) - hy + r;
  const double outside = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0));
  return r - (outside + std::min(std::max(qx, qy), 0.0));
}

// The figure: head, neck, torso, a left arm on the hip enclosing a real background gap (about
// 1.0 % of the frame (0.99 % measured: just under Max Hole %'s default 1 %, so the default fills it; Analyze Frame lowers it)), a hanging right arm, legs cut by the
// bottom of the frame. Origin = frame centre.
double sdFigure(double x, double y) {
  double d = sdCircle(x, y, 0, -330, 85);
  d = std::max(d, sdCapsule(x, y, 0, -260, 0, -190, 36));
  d = std::max(d, sdRoundBox(x, y, 0, -40, 150, 190, 60));
  d = std::max(d, sdCapsule(x, y, -140, -170, -450, 0, 38));
  d = std::max(d, sdCapsule(x, y, -450, 0, -145, 110, 34));
  d = std::max(d, sdCapsule(x, y, 145, -170, 230, 60, 36));
  d = std::max(d, sdCapsule(x, y, 230, 60, 250, 260, 32));
  d = std::max(d, sdCircle(x, y, 255, 290, 40));
  d = std::max(d, sdCapsule(x, y, -70, 140, -90, 600, 55));
  d = std::max(d, sdCapsule(x, y, 70, 140, 100, 600, 55));
  return d;
}

// Smooth value noise in 0-1, feature size `cell` (pixels), deterministic per seed.
double hash01(int ix, int iy, unsigned seed) {
  uint32_t hsh = uint32_t(ix) * 374761393u + uint32_t(iy) * 668265263u + seed * 2246822519u;
  hsh = (hsh ^ (hsh >> 13)) * 1274126177u;
  return double((hsh ^ (hsh >> 16)) & 0xffffff) / double(0xffffff);
}

double noise(double x, double y, double cell, unsigned seed) {
  const double fx = x / cell, fy = y / cell;
  const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
  double tx = fx - ix, ty = fy - iy;
  tx = tx * tx * (3 - 2 * tx);
  ty = ty * ty * (3 - 2 * ty);
  const double a = hash01(ix, iy, seed), b = hash01(ix + 1, iy, seed);
  const double c = hash01(ix, iy + 1, seed), e = hash01(ix + 1, iy + 1, seed);
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + e * tx) * ty;
}

struct Strand {
  double x[17], y[17], width, opacity;
};

// Hair: strands from the top of the head, curling outward (polylines of 16 segments).
std::vector<Strand> makeHair(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<Strand> hair(40);
  for (auto& s : hair) {
    const double a = -M_PI * (0.15 + 0.7 * u(rng));  // -27 .. -153 deg: the top of the head
    const double len = 60 + 80 * u(rng), curl = (u(rng) - 0.5) * 1.6;
    double px = 80 * std::cos(a), py = -330 + 80 * std::sin(a), dir = a;
    for (int k = 0; k <= 16; ++k) {
      s.x[k] = px;
      s.y[k] = py;
      dir += curl / 16;
      px += len / 16 * std::cos(dir);
      py += len / 16 * std::sin(dir);
    }
    s.width = 0.7 + 0.9 * u(rng);
    s.opacity = 0.35 + 0.55 * u(rng);
  }
  return hair;
}

double hairAlpha(const std::vector<Strand>& hair, double x, double y) {
  double best = 0.0;
  for (const auto& s : hair)
    for (int k = 0; k < 16; ++k) {
      const double d = -sdCapsule(x, y, s.x[k], s.y[k], s.x[k + 1], s.y[k + 1], 0.0);
      best = std::max(best, s.opacity * clamp01(1.0 - d / s.width));
    }
  return best;
}

// Which defects and edge styles a scene has.
struct Recipe {
  bool holes = false, greyFace = false, specks = false, haze = false, chatter = false,
       mixedEdge = false, coarse = false, greenBack = false;
  double erodePx = 0.0;
};

Recipe recipe(const std::string& name) {
  Recipe r;
  if (name == "holes") r.holes = r.greyFace = true;
  else if (name == "specks") r.specks = r.haze = true;
  else if (name == "chatter") r.chatter = true;
  else if (name == "mixed_edge") r.mixedEdge = true;
  else if (name == "eroded") r.erodePx = 3.0;
  else if (name == "coarse") r.coarse = true;
  else if (name == "fgcolor") r.greenBack = true;
  else if (name == "all") r.holes = r.greyFace = r.specks = r.haze = r.chatter = r.mixedEdge = true;
  else throw std::invalid_argument("unknown scene: " + name);
  return r;
}

const char* about(const std::string& name) {
  if (name == "holes") return "grey holes inside the torso and a grey (0.85) side face; a real gap (arm on hip) must stay open";
  if (name == "specks") return "islands and debris outside the figure, background haze up to ~0.1";
  if (name == "chatter") return "the left side's edge wiggles (+-1.2 px, ~15 px wavelength)";
  if (name == "mixed_edge") return "hard-cut (aliased) left side, 1 px head, 8 px soft right arm, 2 px elsewhere";
  if (name == "eroded") return "the whole matte 3 px too small (edge profile kept)";
  if (name == "coarse") return "a blocky (12 px) matte, offset 2 px right, over a picture with the true edge";
  if (name == "fgcolor") return "clean matte with soft hair over a saturated green background; fg = the true colour";
  return "all defects together: holes, grey face, specks, haze, chatter, mixed edges";
}

}  // namespace

std::vector<std::string> sceneNames() {
  return {"holes", "specks", "chatter", "mixed_edge", "eroded", "coarse", "fgcolor", "all"};
}

Scene make(const std::string& name, int w, int h, unsigned seed) {
  if (name == "gaps") return makeGaps(w, h, seed);
  if (name == "hair_flat" || name == "hair_busy") return makeHair(w, h, name == "hair_busy", seed);
  const Recipe rc = recipe(name);
  Scene sc;
  sc.name = name;
  sc.about = about(name);
  sc.w = w;
  sc.h = h;
  const size_t n = size_t(w) * h;
  sc.matte.assign(n, 0.0f);
  sc.truth.assign(n, 0.0f);
  sc.rgb.assign(n * 3, 0.0f);
  sc.fg.assign(n * 3, 0.0f);
  const double k = h / 1080.0;  // pixels per design unit
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const std::vector<Strand> hair = makeHair(rng);

  struct Blob {
    double x, y, r, v;
  };
  std::vector<Blob> holes, specks;
  if (rc.holes)
    for (int i = 0; i < 25; ++i)
      holes.push_back({-110 + 220 * u(rng), -190 + 330 * u(rng), 2 + 12 * u(rng), 0.05 + 0.75 * u(rng)});
  if (rc.specks) {
    while (specks.size() < 30) {
      const double x = -900 + 1800 * u(rng), y = -520 + 1040 * u(rng);
      const double gap = -sdFigure(x, y);
      if (gap < 8 || gap > 400) continue;  // outside, some close to the edge
      specks.push_back({x, y, 1 + 9 * u(rng), 0.2 + 0.8 * u(rng)});
    }
    specks.push_back({-700, -380, 25, 1.0});  // two large ones: island or object?
    specks.push_back({620, 300, 30, 0.9});
  }

  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
      const size_t i = size_t(py) * w + px;
      const double x = (px - w / 2.0) / k, y = (py - h / 2.0) / k;  // design units
      const double sd = sdFigure(x, y) * k;                           // pixels
      // Edge width (px): 2 in the truth; mixed_edge's input: hard left side, 1 px head, 8 px
      // right arm.
      auto profile = [&](double d) { return clamp01(0.5 + d / (2.0 * k)); };
      const bool nearHead = std::fabs(x) < 260 && y > -560 && y < -200;
      const double hairA = nearHead ? hairAlpha(hair, x, y) : 0.0;
      const double truth = std::max(profile(sd), hairA);
      double m = truth;
      if (rc.mixedEdge) {
        double width = 2.0 * k;
        if (x < -40) width = 0.0;
        else if (y < -240) width = 1.0 * k;
        else if (x > 150 && y > -150) width = 8.0 * k;
        m = std::max(width > 0 ? clamp01(0.5 + sd / width) : (sd >= 0 ? 1.0 : 0.0), hairA);
      }
      if (rc.chatter && x < 0) {
        const double wiggle = 1.2 * k * (2 * noise(px, py, 15 * k, seed + 7) - 1);
        m = std::max(profile(sd + wiggle), hairA);
      }
      if (rc.erodePx > 0) m = std::max(profile(sd - rc.erodePx * k), hairA);
      for (const Blob& b : holes) {
        const double hd = std::hypot(x - b.x, y - b.y) * k;
        const double inHole = clamp01(0.5 + (b.r * k - hd) / (1.5 * k));
        m = std::min(m, 1.0 - (1.0 - b.v) * inHole);
      }
      if (rc.greyFace && x > 60 && y > -150 && y < 100) {
        const double ramp = clamp01((x - 60) / 30.0) * clamp01((y + 150) / 30.0) * clamp01((100 - y) / 30.0);
        m = std::min(m, 1.0 - 0.15 * ramp);
      }
      if (rc.haze) {
        const double haze = 0.04 * noise(px, py, 3 * k, seed + 3) +
                            0.06 * clamp01(1.0 - std::hypot(x + 600, y + 300) / 350.0);
        m = std::max(m, haze);
      }
      for (const Blob& b : specks) {
        const double bd = std::hypot(x - b.x, y - b.y) * k;
        m = std::max(m, b.v * clamp01(0.5 + (b.r * k - bd) / (1.5 * k)));
      }
      sc.truth[i] = float(truth);
      sc.matte[i] = float(m);

      // Picture: a striped warm foreground over a grey-blue (or green) textured background.
      const double tex = noise(px, py, 40 * k, seed + 11), stripe = 0.5 + 0.5 * std::sin((x + y) * 0.08);
      const double f[3] = {0.55 + 0.25 * stripe * tex, 0.33 + 0.15 * tex, 0.22 + 0.1 * stripe};
      double bg[3];
      if (rc.greenBack) {
        bg[0] = 0.08 + 0.05 * tex;
        bg[1] = 0.65 + 0.15 * tex;
        bg[2] = 0.12;
      } else {
        const double g = 0.3 + 0.2 * (py / double(h));
        bg[0] = g * 0.8 + 0.1 * tex;
        bg[1] = g * 0.9 + 0.1 * tex;
        bg[2] = g + 0.15 * tex;
      }
      for (int c = 0; c < 3; ++c) {
        sc.fg[i * 3 + c] = float(f[c]);
        sc.rgb[i * 3 + c] = float(truth * f[c] + (1.0 - truth) * bg[c]);
      }
    }

  if (rc.coarse) {  // 12 px block averages, bilinear back up, shifted 2 px right
    const int cell = std::max(2, int(std::lround(12 * k))), shift = int(std::lround(2 * k));
    const int cw = (w + cell - 1) / cell, ch = (h + cell - 1) / cell;
    std::vector<double> blk(size_t(cw) * ch, 0.0);
    std::vector<int> cnt(blk.size(), 0);
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        blk[size_t(py / cell) * cw + px / cell] += sc.truth[size_t(py) * w + px];
        ++cnt[size_t(py / cell) * cw + px / cell];
      }
    for (size_t j = 0; j < blk.size(); ++j) blk[j] /= std::max(cnt[j], 1);
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        const double fx = std::min(std::max((px - shift + 0.5) / cell - 0.5, 0.0), cw - 1.0);
        const double fy = std::min(std::max((py + 0.5) / cell - 0.5, 0.0), ch - 1.0);
        const int x0 = int(fx), y0 = int(fy), x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
        const double tx = fx - x0, ty = fy - y0;
        const double v = (blk[size_t(y0) * cw + x0] * (1 - tx) + blk[size_t(y0) * cw + x1] * tx) * (1 - ty) +
                         (blk[size_t(y1) * cw + x0] * (1 - tx) + blk[size_t(y1) * cw + x1] * tx) * ty;
        sc.matte[size_t(py) * w + px] = float(v);
      }
  }
  return sc;
}

// ---- Scenes for Matte Refine ML (synth.h). Separate from make()'s recipes, so the scenes above
// stay bit-identical.

namespace {

double sdRing(double x, double y, double cx, double cy, double outer, double inner) {
  const double r = std::hypot(x - cx, y - cy);
  return std::min(outer - r, r - inner);
}

// The gaps figure: the figure above with a ring held in the right hand (a small ring linked
// below it), an earring, and legs whose feet touch, so the space between them is enclosed.
double sdGapFigure(double x, double y) {
  double d = sdCircle(x, y, 0, -330, 85);
  d = std::max(d, sdCapsule(x, y, 0, -260, 0, -190, 36));
  d = std::max(d, sdRoundBox(x, y, 0, -40, 150, 190, 60));
  d = std::max(d, sdCapsule(x, y, -140, -170, -450, 0, 38));
  d = std::max(d, sdCapsule(x, y, -450, 0, -145, 110, 34));
  d = std::max(d, sdCapsule(x, y, 145, -170, 230, 60, 36));
  d = std::max(d, sdCapsule(x, y, 230, 60, 250, 230, 32));
  d = std::max(d, sdCircle(x, y, 255, 260, 34));
  d = std::max(d, sdRing(x, y, 262, 330, 58, 36));
  d = std::max(d, sdRing(x, y, 262, 402, 26, 15));
  d = std::max(d, sdRing(x, y, 70, -265, 17, 8));
  d = std::max(d, sdCapsule(x, y, -85, 140, -35, 445, 44));
  d = std::max(d, sdCapsule(x, y, 85, 140, 35, 445, 44));
  d = std::max(d, sdCapsule(x, y, -75, 470, 75, 470, 28));
  return d;
}

// 4-connected flood fill from the frame border over `open` pixels; returns the reached ones.
std::vector<uint8_t> reachFromBorder(const std::vector<uint8_t>& open, int w, int h) {
  std::vector<uint8_t> seen(open.size(), 0);
  std::vector<int> stack;
  auto push = [&](int x, int y) {
    const size_t i = size_t(y) * w + x;
    if (open[i] && !seen[i]) {
      seen[i] = 1;
      stack.push_back(int(i));
    }
  };
  for (int x = 0; x < w; ++x) push(x, 0), push(x, h - 1);
  for (int y = 0; y < h; ++y) push(0, y), push(w - 1, y);
  while (!stack.empty()) {
    const int i = stack.back(), x = i % w, y = i / w;
    stack.pop_back();
    if (x > 0) push(x - 1, y);
    if (x < w - 1) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y < h - 1) push(x, y + 1);
  }
  return seen;
}

void gaussianBlur(std::vector<float>& v, int w, int h, double sigma) {
  const int r = std::max(1, int(std::ceil(3 * sigma)));
  std::vector<double> k(2 * r + 1);
  double sum = 0;
  for (int j = -r; j <= r; ++j) sum += k[j + r] = std::exp(-0.5 * j * j / (sigma * sigma));
  for (double& e : k) e /= sum;
  std::vector<float> tmp(v.size());
  for (int pass = 0; pass < 2; ++pass) {  // rows, then columns; clamped at the frame edge
    const int n = pass ? h : w, lines = pass ? w : h;
    for (int l = 0; l < lines; ++l)
      for (int i = 0; i < n; ++i) {
        double acc = 0;
        for (int j = -r; j <= r; ++j) {
          const int s = std::min(std::max(i + j, 0), n - 1);
          acc += k[j + r] * (pass ? v[size_t(s) * w + l] : v[size_t(l) * w + s]);
        }
        (pass ? tmp[size_t(i) * w + l] : tmp[size_t(l) * w + i]) = float(acc);
      }
    v.swap(tmp);
  }
}

void minFilter(std::vector<float>& v, int w, int h, int r) {  // square (2r+1)^2
  std::vector<float> tmp(v.size());
  for (int pass = 0; pass < 2; ++pass) {
    const int n = pass ? h : w, lines = pass ? w : h;
    for (int l = 0; l < lines; ++l)
      for (int i = 0; i < n; ++i) {
        float m = 1.0f;
        for (int s = std::max(i - r, 0); s <= std::min(i + r, n - 1); ++s)
          m = std::min(m, pass ? v[size_t(s) * w + l] : v[size_t(l) * w + s]);
        (pass ? tmp[size_t(i) * w + l] : tmp[size_t(l) * w + i]) = m;
      }
    v.swap(tmp);
  }
}

Scene emptyScene(const std::string& name, const std::string& about, int w, int h) {
  Scene sc;
  sc.name = name;
  sc.about = about;
  sc.w = w;
  sc.h = h;
  const size_t n = size_t(w) * h;
  sc.matte.assign(n, 0.0f);
  sc.truth.assign(n, 0.0f);
  sc.rgb.assign(n * 3, 0.0f);
  sc.fg.assign(n * 3, 0.0f);
  return sc;
}

}  // namespace

std::vector<std::string> mlSceneNames() { return {"gaps", "hair_flat", "hair_busy"}; }

Scene makeGaps(int w, int h, unsigned seed) {
  Scene sc = emptyScene("gaps",
                        "B3: defect holes (fill) and real gaps (keep) of overlapping sizes, a grey wash; "
                        "label 0 bg, 1 subject, 2 defect hole, 3 real gap",
                        w, h);
  const size_t n = size_t(w) * h;
  const double k = h / 1080.0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);

  // Defect holes: fixed sizes (largest first, so they fit), positions from the seed.
  struct Hole {
    double x, y, rx, ry, angle, v;
  };
  const double radii[] = {44, 34, 26, 20, 15, 11, 8, 6, 4};
  std::vector<Hole> holes;
  for (int j = 0; j < 10; ++j) {
    const double rx = j == 0 ? 75 : radii[j - 1], ry = j == 0 ? 42 : rx;
    for (int tries = 0;; ++tries) {
      if (tries > 100000) throw std::logic_error("makeGaps: no room for a hole");
      const Hole c{-460 + 920 * u(rng), -420 + 920 * u(rng), rx, ry, M_PI * u(rng), 0.0};
      if (sdGapFigure(c.x, c.y) < rx + 14) continue;
      bool clear = true;
      for (const Hole& o : holes) clear = clear && std::hypot(c.x - o.x, c.y - o.y) > rx + o.rx + 10;
      if (!clear) continue;
      holes.push_back(c);
      holes.back().v = j % 2 ? 0.1 + 0.3 * u(rng) : 0.0;
      break;
    }
  }

  std::vector<uint8_t> inHole(n, 0), open(n, 0);
  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
      const size_t i = size_t(py) * w + px;
      const double x = (px - w / 2.0) / k, y = (py - h / 2.0) / k;
      const double truth = clamp01(0.5 + sdGapFigure(x, y) * k / (2.0 * k));
      double m = truth;
      for (const Hole& b : holes) {
        const double dx = x - b.x, dy = y - b.y, ca = std::cos(b.angle), sa = std::sin(b.angle);
        const double ex = (dx * ca + dy * sa) / b.rx, ey = (-dx * sa + dy * ca) / b.ry;
        // Approximate distance to the ellipse (exact for circles), in px.
        const double hd = (1.0 - std::hypot(ex, ey)) * std::min(b.rx, b.ry) * k;
        const double cover = clamp01(0.5 + hd / (1.5 * k));
        m = std::min(m, 1.0 - (1.0 - b.v) * cover);
        if (cover >= 0.5) inHole[i] = 1;
      }
      const double wx = (x - 95) / 50, wy = (y + 120) / 90;  // the wash: torso, right side
      const double wash = clamp01(1.0 - (wx * wx + wy * wy));
      m = std::min(m, 1.0 - 0.42 * wash * wash * (3 - 2 * wash) * (0.8 + 0.2 * noise(px, py, 9 * k, seed + 5)));
      sc.truth[i] = float(truth);
      sc.matte[i] = float(m);
      open[i] = truth < 0.5;

      // Picture: warm red-orange striped subject, cool blue-green background, both textured.
      const double tex = noise(px, py, 25 * k, seed + 11), stripe = 0.5 + 0.5 * std::sin((x - y) * 0.06);
      const double f[3] = {0.62 + 0.2 * stripe * tex, 0.28 + 0.1 * tex, 0.16 + 0.08 * stripe};
      const double bt = noise(px, py, 60 * k, seed + 13);
      const double bg[3] = {0.1 + 0.08 * bt, 0.32 + 0.12 * bt, 0.4 + 0.15 * tex};
      for (int c = 0; c < 3; ++c) {
        sc.fg[i * 3 + c] = float(f[c]);
        sc.rgb[i * 3 + c] = float(truth * f[c] + (1.0 - truth) * bg[c]);
      }
    }
  const std::vector<uint8_t> outside = reachFromBorder(open, w, h);
  sc.label.resize(n);
  for (size_t i = 0; i < n; ++i)
    sc.label[i] = open[i] ? (outside[i] ? kBackground : kRealGap) : (inHole[i] ? kDefectHole : kSubject);
  return sc;
}

Scene makeHair(int w, int h, bool busy, unsigned seed) {
  Scene sc = emptyScene(busy ? "hair_busy" : "hair_flat",
                        std::string("B1/B2: 260 thin hair strands at the top of a head, coarse eroded matte; ") +
                            (busy ? "busy textured background" : "flat background"),
                        w, h);
  const size_t n = size_t(w) * h;
  const double k = h / 1080.0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);

  // Body: head (top at the frame centre) and shoulders, 2 px edge.
  std::vector<double> premult(n * 3);
  std::vector<float>& alpha = sc.truth;
  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
      const size_t i = size_t(py) * w + px;
      const double x = (px - w / 2.0) / k, y = (py - h / 2.0) / k;
      const double sd = std::max(sdCircle(x, y, 0, 400, 400), sdRoundBox(x, y, 0, 560, 760, 200, 160)) * k;
      const double a = clamp01(0.5 + sd / (2.0 * k)), t = noise(px, py, 30 * k, seed + 21);
      const double c[3] = {0.18 + 0.08 * t, 0.11 + 0.05 * t, 0.07 + 0.03 * t};  // dark brown
      alpha[i] = float(a);
      for (int ch = 0; ch < 3; ++ch) premult[i * 3 + ch] = a * c[ch];
    }

  // Strands: a polyline of 32 segments each, stamped into a per-strand buffer (max over its own
  // segments), then composited over everything below it.
  const int segs = 32;
  std::vector<float> buf;
  for (int s = 0; s < 260; ++s) {
    const double a0 = -M_PI * (0.1 + 0.8 * u(rng));  // root angle on the head, -18 .. -162 deg
    double dir = a0 + (u(rng) - 0.5) * 1.2;           // outward, +-34 deg
    const double len = 60 + 180 * u(rng), curl = (u(rng) - 0.5) * 2.4, width = 1 + 2 * u(rng);
    const double opacity = 0.35 + 0.65 * u(rng), tone = u(rng);
    const double col[3] = {0.45 + 0.25 * tone, 0.32 + 0.18 * tone, 0.18 + 0.1 * tone};  // lighter
    double xs[segs + 1], ys[segs + 1];
    xs[0] = 394 * std::cos(a0);
    ys[0] = 400 + 394 * std::sin(a0);
    for (int j = 1; j <= segs; ++j) {
      dir += curl / segs;
      xs[j] = xs[j - 1] + len / segs * std::cos(dir);
      ys[j] = ys[j - 1] + len / segs * std::sin(dir);
    }
    // Bounding box in px.
    double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (int j = 0; j <= segs; ++j) {
      x0 = std::min(x0, xs[j]);
      x1 = std::max(x1, xs[j]);
      y0 = std::min(y0, ys[j]);
      y1 = std::max(y1, ys[j]);
    }
    const double pad = width * k + 2;
    const int bx0 = std::max(0, int(std::floor(x0 * k + w / 2.0 - pad)));
    const int bx1 = std::min(w - 1, int(std::ceil(x1 * k + w / 2.0 + pad)));
    const int by0 = std::max(0, int(std::floor(y0 * k + h / 2.0 - pad)));
    const int by1 = std::min(h - 1, int(std::ceil(y1 * k + h / 2.0 + pad)));
    if (bx0 > bx1 || by0 > by1) continue;
    const int bw = bx1 - bx0 + 1;
    buf.assign(size_t(bw) * (by1 - by0 + 1), 0.0f);
    for (int j = 0; j < segs; ++j) {
      const double t = (j + 0.5) / segs;  // along the strand: taper width and opacity
      const double wpx = width * k * (1.0 - 0.4 * t), op = opacity * (1.0 - 0.6 * t);
      const double ax = xs[j] * k + w / 2.0, ay = ys[j] * k + h / 2.0;
      const double cx = xs[j + 1] * k + w / 2.0, cy = ys[j + 1] * k + h / 2.0;
      const int sx0 = std::max(bx0, int(std::floor(std::min(ax, cx) - wpx - 1)));
      const int sx1 = std::min(bx1, int(std::ceil(std::max(ax, cx) + wpx + 1)));
      const int sy0 = std::max(by0, int(std::floor(std::min(ay, cy) - wpx - 1)));
      const int sy1 = std::min(by1, int(std::ceil(std::max(ay, cy) + wpx + 1)));
      for (int py = sy0; py <= sy1; ++py)
        for (int px = sx0; px <= sx1; ++px) {
          const double d = -sdCapsule(px, py, ax, ay, cx, cy, 0.0);
          // Box-filtered line of width wpx: coverage ~ wpx/2 + 0.5 - d, at most min(wpx, 1).
          const double cover = op * clamp01(std::min(wpx, wpx / 2 + 0.5 - d));
          float& b = buf[size_t(py - by0) * bw + px - bx0];
          b = std::max(b, float(cover));
        }
    }
    for (int py = by0; py <= by1; ++py)
      for (int px = bx0; px <= bx1; ++px) {
        const double a = buf[size_t(py - by0) * bw + px - bx0];
        if (a <= 0) continue;
        const size_t i = size_t(py) * w + px;
        alpha[i] = float(a + (1.0 - a) * alpha[i]);
        for (int ch = 0; ch < 3; ++ch) premult[i * 3 + ch] = a * col[ch] + (1.0 - a) * premult[i * 3 + ch];
      }
  }

  // The coarse matte: blur, threshold, erode.
  sc.matte = alpha;
  gaussianBlur(sc.matte, w, h, 6.0 * k);
  for (float& v : sc.matte) v = float(clamp01((v - 0.5) / 0.08 + 0.5));
  minFilter(sc.matte, w, h, std::max(1, int(std::lround(3 * k))));

  for (int py = 0; py < h; ++py)
    for (int px = 0; px < w; ++px) {
      const size_t i = size_t(py) * w + px;
      const double a = alpha[i];
      double bg[3];
      if (busy) {  // several scales of saturated texture and thin stripes: hard for a model
        const double n1 = noise(px, py, 3 * k, seed + 31), n2 = noise(px, py, 9 * k, seed + 32);
        const double n3 = noise(px, py, 40 * k, seed + 33), st = 0.5 + 0.5 * std::sin((px * 0.7 + py * 0.3) / k);
        bg[0] = 0.15 + 0.5 * n3 * n1 + 0.15 * st;
        bg[1] = 0.1 + 0.6 * n2 * (1 - n3) + 0.1 * n1;
        bg[2] = 0.2 + 0.5 * (1 - n2) * n1 + 0.1 * st;
      } else {
        bg[0] = 0.22;
        bg[1] = 0.27;
        bg[2] = 0.33;
      }
      for (int ch = 0; ch < 3; ++ch) {
        const double p = premult[i * 3 + ch];
        sc.fg[i * 3 + ch] = float(a > 1e-6 ? p / a : 0.0);
        sc.rgb[i * 3 + ch] = float(p + (1.0 - a) * bg[ch]);
      }
    }
  return sc;
}

void Affine::apply(double x, double y, double* ox, double* oy) const {
  *ox = a * x + b * y + tx;
  *oy = c * x + d * y + ty;
}

Affine Affine::inverse() const {
  const double det = a * d - b * c;
  Affine r;
  r.a = d / det;
  r.b = -b / det;
  r.c = -c / det;
  r.d = a / det;
  r.tx = -(r.a * tx + r.b * ty);
  r.ty = -(r.c * tx + r.d * ty);
  return r;
}

Affine between(const Affine& from, const Affine& to) {
  const Affine f = from.inverse();
  Affine r;  // to(f(p))
  r.a = to.a * f.a + to.b * f.c;
  r.b = to.a * f.b + to.b * f.d;
  r.c = to.c * f.a + to.d * f.c;
  r.d = to.c * f.b + to.d * f.d;
  r.tx = to.a * f.tx + to.b * f.ty + to.tx;
  r.ty = to.c * f.tx + to.d * f.ty + to.ty;
  return r;
}

Sequence makeSequence(int w, int h, int frames, unsigned seed, bool movingBackground) {
  Sequence s;
  s.w = w;
  s.h = h;
  const size_t n = size_t(w) * h;
  const double k = h / 1080.0, cx = w / 2.0, cy = h / 2.0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double angle = 0, mx = 0, my = 0;
  for (int t = 0; t < frames; ++t) {
    if (t > 0) {
      angle += (0.3 + 0.2 * u(rng)) * M_PI / 180;
      mx += (5 + 2 * u(rng)) * k;
      my += (-1.5 + u(rng)) * k;
    }
    Affine m;  // rotate about the frame centre, then translate
    m.a = std::cos(angle);
    m.b = -std::sin(angle);
    m.c = std::sin(angle);
    m.d = std::cos(angle);
    m.tx = cx - m.a * cx - m.b * cy + mx;
    m.ty = cy - m.c * cx - m.d * cy + my;
    s.subject.push_back(m);
    Affine b;
    if (movingBackground) {
      b.tx = -2.5 * k * t;
      b.ty = 1.0 * k * t;
    }
    s.background.push_back(b);
    s.holeOn.push_back(u(rng) < 0.5);
  }
  if (frames > 1) {  // the hole must flicker: on at least once and off at least once
    bool any = false, all = true;
    for (bool on : s.holeOn) any = any || on, all = all && on;
    if (!any || all) s.holeOn[frames / 2] = !s.holeOn[frames / 2];
  }
  const double holeR = 18, holeX = 40, holeY = -60;  // design units, frame 0 (torso)
  for (int t = 0; t < frames; ++t) {
    Scene sc = emptyScene("temporal", "B4: moving figure, per-frame edge jitter, band speckle, flickering hole", w, h);
    std::vector<float> sdt(n);
    const Affine inv = s.subject[t].inverse(), binv = s.background[t].inverse();
    const unsigned fs = seed * 7919u + 1000u + unsigned(t) * 104729u;  // this frame's noise
    const double holeV = 0.1 + 0.3 * u(rng);
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        const size_t i = size_t(py) * w + px;
        double qx, qy, bx, by;
        inv.apply(px, py, &qx, &qy);  // the point in frame 0
        binv.apply(px, py, &bx, &by);
        const double x = (qx - cx) / k, y = (qy - cy) / k;
        const double sd = sdFigure(x, y) * k;
        auto profile = [&](double dd) { return clamp01(0.5 + dd / (2.0 * k)); };
        const double truth = profile(sd);
        double m = profile(sd + 2.5 * k * (2 * noise(px, py, 18 * k, fs) - 1));
        if (std::fabs(sd) < 6 * k && hash01(px, py, fs + 1) < 0.02) m = hash01(px, py, fs + 2);
        if (s.holeOn[t]) {
          const double hd = (holeR - std::hypot(x - holeX, y - holeY)) * k;
          m = std::min(m, 1.0 - (1.0 - holeV) * clamp01(0.5 + hd / (1.5 * k)));
        }
        sdt[i] = float(sd);
        sc.truth[i] = float(truth);
        sc.matte[i] = float(m);
        // Subject texture moves with the subject (qx, qy), background texture with (bx, by).
        const double st = noise(qx, qy, 20 * k, seed + 41), stripe = 0.5 + 0.5 * std::sin((x + y) * 0.08);
        const double f[3] = {0.55 + 0.25 * stripe * st, 0.33 + 0.15 * st, 0.22 + 0.1 * stripe};
        const double bt = noise(bx, by, 35 * k, seed + 43), bt2 = noise(bx, by, 7 * k, seed + 44);
        const double bg[3] = {0.25 + 0.15 * bt, 0.3 + 0.1 * bt2, 0.35 + 0.15 * bt * bt2};
        for (int c = 0; c < 3; ++c) {
          sc.fg[i * 3 + c] = float(f[c]);
          sc.rgb[i * 3 + c] = float(truth * f[c] + (1.0 - truth) * bg[c]);
        }
      }
    s.frames.push_back(std::move(sc));
    s.edgeDistance.push_back(std::move(sdt));
  }
  return s;
}

double flicker(const Sequence& s, const std::vector<std::vector<float>>& mattes, double bandPx) {
  const int w = s.w, h = s.h;
  double sum = 0;
  long long count = 0;
  for (size_t t = 1; t < mattes.size(); ++t) {
    const Affine back = between(s.subject[t], s.subject[t - 1]);
    const std::vector<float>& prev = mattes[t - 1];
    for (int py = 0; py < h; ++py)
      for (int px = 0; px < w; ++px) {
        const size_t i = size_t(py) * w + px;
        if (std::fabs(s.edgeDistance[t][i]) > bandPx) continue;
        double qx, qy;
        back.apply(px, py, &qx, &qy);
        if (qx < 0 || qy < 0 || qx > w - 1 || qy > h - 1) continue;
        const int x0 = std::min(int(qx), w - 2), y0 = std::min(int(qy), h - 2);
        const double fx = qx - x0, fy = qy - y0;
        const float* r0 = &prev[size_t(y0) * w + x0];
        const double v = (r0[0] * (1 - fx) + r0[1] * fx) * (1 - fy) + (r0[w] * (1 - fx) + r0[w + 1] * fx) * fy;
        sum += std::fabs(mattes[t][i] - v);
        ++count;
      }
  }
  return count ? sum / count : 0.0;
}

double bandError(const Sequence& s, const std::vector<std::vector<float>>& mattes, double bandPx) {
  double sum = 0;
  long long count = 0;
  for (size_t t = 0; t < mattes.size(); ++t)
    for (size_t i = 0; i < mattes[t].size(); ++i)
      if (std::fabs(s.edgeDistance[t][i]) <= bandPx) {
        sum += std::fabs(mattes[t][i] - s.frames[t].truth[i]);
        ++count;
      }
  return count ? sum / count : 0.0;
}

bool writeExr(const std::string& path, int w, int h, int channels, const std::vector<float>& data) {
  // OpenEXR file layout (openexr.com, "OpenEXR File Layout"): magic, version 2 (single-part
  // scanline), header attributes (name, type, size, value) ended by a 0 byte, the line offset
  // table, then per line: y, byte count, each channel's row in channel-list (alphabetical) order.
  std::vector<std::string> names;
  if (channels == 1) names = {"Y"};
  else if (channels == 3) names = {"B", "G", "R"};
  else if (channels == 4) names = {"A", "B", "G", "R"};
  else return false;
  auto chanIndex = [&](const std::string& nm) {  // position in the interleaved input
    return nm == "Y" || nm == "R" ? 0 : nm == "G" ? 1 : nm == "B" ? 2 : 3;
  };
  std::vector<uint8_t> f;
  auto u32 = [&](uint32_t v) { for (int b = 0; b < 4; ++b) f.push_back(uint8_t(v >> (8 * b))); };
  auto u64 = [&](uint64_t v) { for (int b = 0; b < 8; ++b) f.push_back(uint8_t(v >> (8 * b))); };
  auto f32 = [&](float v) { uint32_t b; std::memcpy(&b, &v, 4); u32(b); };
  auto str = [&](const std::string& s) { f.insert(f.end(), s.begin(), s.end()); f.push_back(0); };
  auto attr = [&](const char* name, const char* type, uint32_t size) { str(name); str(type); u32(size); };
  u32(20000630);
  u32(2);
  uint32_t chSize = 1;
  for (const auto& nm : names) chSize += uint32_t(nm.size() + 1 + 16);
  attr("channels", "chlist", chSize);
  for (const auto& nm : names) {
    str(nm);
    u32(2);  // FLOAT
    u32(0);  // pLinear + reserved
    u32(1);  // xSampling
    u32(1);  // ySampling
  }
  f.push_back(0);
  attr("compression", "compression", 1);
  f.push_back(0);  // NO_COMPRESSION
  attr("dataWindow", "box2i", 16);
  u32(0); u32(0); u32(uint32_t(w - 1)); u32(uint32_t(h - 1));
  attr("displayWindow", "box2i", 16);
  u32(0); u32(0); u32(uint32_t(w - 1)); u32(uint32_t(h - 1));
  attr("lineOrder", "lineOrder", 1);
  f.push_back(0);  // INCREASING_Y
  attr("pixelAspectRatio", "float", 4);
  f32(1.0f);
  attr("screenWindowCenter", "v2f", 8);
  f32(0.0f); f32(0.0f);
  attr("screenWindowWidth", "float", 4);
  f32(1.0f);
  f.push_back(0);  // end of header
  const uint64_t lineBytes = uint64_t(w) * 4 * names.size();
  const uint64_t table = f.size();
  for (int y = 0; y < h; ++y) u64(table + 8ull * h + uint64_t(y) * (8 + lineBytes));
  for (int y = 0; y < h; ++y) {
    u32(uint32_t(y));
    u32(uint32_t(lineBytes));
    for (const auto& nm : names)
      for (int x = 0; x < w; ++x) f32(data[(size_t(y) * w + x) * channels + chanIndex(nm)]);
  }
  FILE* out = std::fopen(path.c_str(), "wb");
  if (!out) return false;
  const bool ok = std::fwrite(f.data(), 1, f.size(), out) == f.size();
  return std::fclose(out) == 0 && ok;
}

}  // namespace synth
