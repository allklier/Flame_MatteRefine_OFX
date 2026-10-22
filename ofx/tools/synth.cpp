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
// 1.5 % of the frame, more than Max Hole %'s default 1 %), a hanging right arm, legs cut by the
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
