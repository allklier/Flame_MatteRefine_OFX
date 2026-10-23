// The pictures of USERGUIDE.md (docs/images/*.png), made from the synthetic scenes (synth.h) with
// the operations' own library code (the same functions the plugins call), so they can be remade
// after a change:
//   mr_guide_images <out dir> [<model dir> [<cuda lib dir>]]
// Every scene is made at 1920x1080, seed 1. Matte Refine's images need nothing else. The Matte
// Refine ML images (Edge Refine, Smart Fill, Temporal) are made only when this tool is built with
// MR_BUILD_ML=ON (it then links the ML cores and ONNX Runtime) and, for Edge Refine and Smart Fill,
// only when <model dir> holds vitmatte-small/model.onnx and birefnet-lite/model.onnx: the REAL
// models run in-process (CUDA, else the CPU). A missing model skips its image (printed), never a
// stand-in. Panels are 8-bit; a matte is shown as grey; pictures are shown as their values, no
// view transform. Settings per image: printed, and in USERGUIDE.md under each image.
// Build: cmake --build <build> --target mr_guide_images (not in the default build). Needs zlib.
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ops/analyze_frame.h"
#include "ops/clean_background.h"
#include "ops/contour_smooth.h"
#include "ops/edge_field.h"
#include "ops/edge_width.h"
#include "ops/fill_holes.h"
#include "ops/foreground_colour.h"
#include "ops/shrink_grow.h"
#include "ops/snap_to_image.h"
#include "ops/solid_core.h"
#include "synth.h"

#ifdef MR_GUIDE_ML
#include <unistd.h>

#include "edge_refine.h"
#include "ort_runner.h"
#include "render_chain.h"
#include "smart_fill.h"
#include "temporal.h"
#endif

namespace {

using Buf = std::vector<float>;

// ---- 8-bit RGB images and PNG output ----------------------------------------------------------
struct Img {
  int w = 0, h = 0;
  std::vector<uint8_t> px;  // w*h*3
  Img() {}
  Img(int w_, int h_, uint8_t fill) : w(w_), h(h_), px(size_t(w_) * h_ * 3, fill) {}
};

uint8_t to8(float v) { return uint8_t(std::lround(std::min(std::max(v, 0.0f), 1.0f) * 255.0f)); }

void chunk(FILE* f, const char* type, const std::vector<uint8_t>& data) {
  auto be32 = [&](uint32_t v) {
    const uint8_t b[4] = {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)};
    std::fwrite(b, 1, 4, f);
  };
  be32(uint32_t(data.size()));
  std::fwrite(type, 1, 4, f);
  if (!data.empty()) std::fwrite(data.data(), 1, data.size(), f);
  uLong crc = crc32(0L, reinterpret_cast<const Bytef*>(type), 4);
  if (!data.empty()) crc = crc32(crc, data.data(), uInt(data.size()));
  be32(uint32_t(crc));
}

// Greyscale PNG when every pixel is grey, else RGB; per row the filter with the smallest sum.
bool writePng(const std::string& path, const Img& im) {
  bool grey = true;
  for (size_t i = 0; i < im.px.size() && grey; i += 3) grey = im.px[i] == im.px[i + 1] && im.px[i] == im.px[i + 2];
  const int ch = grey ? 1 : 3, stride = im.w * ch;
  std::vector<uint8_t> raw(size_t(stride) * im.h);
  for (size_t i = 0, n = size_t(im.w) * im.h; i < n; ++i)
    for (int c = 0; c < ch; ++c) raw[i * ch + c] = im.px[i * 3 + c];
  std::vector<uint8_t> filtered;
  filtered.reserve(size_t(stride + 1) * im.h);
  std::vector<uint8_t> line(stride), best(stride);
  for (int y = 0; y < im.h; ++y) {
    const uint8_t* cur = &raw[size_t(y) * stride];
    const uint8_t* up = y ? cur - stride : nullptr;
    long bestSum = -1;
    int bestType = 0;
    for (int type = 0; type < 5; ++type) {
      long sum = 0;
      for (int x = 0; x < stride; ++x) {
        const int a = x >= ch ? cur[x - ch] : 0, b = up ? up[x] : 0, c = (up && x >= ch) ? up[x - ch] : 0;
        int pred = 0;
        if (type == 1) pred = a;
        else if (type == 2) pred = b;
        else if (type == 3) pred = (a + b) / 2;
        else if (type == 4) {
          const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
          pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        }
        line[x] = uint8_t(cur[x] - pred);
        sum += line[x] < 128 ? line[x] : 256 - line[x];
      }
      if (bestSum < 0 || sum < bestSum) bestSum = sum, bestType = type, best = line;
    }
    filtered.push_back(uint8_t(bestType));
    filtered.insert(filtered.end(), best.begin(), best.end());
  }
  uLongf zlen = compressBound(uLong(filtered.size()));
  std::vector<uint8_t> z(zlen);
  if (compress2(z.data(), &zlen, filtered.data(), uLong(filtered.size()), 9) != Z_OK) return false;
  z.resize(zlen);
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  std::fwrite(sig, 1, 8, f);
  std::vector<uint8_t> ihdr(13, 0);
  for (int k = 0; k < 4; ++k) ihdr[k] = uint8_t(im.w >> (24 - 8 * k)), ihdr[4 + k] = uint8_t(im.h >> (24 - 8 * k));
  ihdr[8] = 8;
  ihdr[9] = grey ? 0 : 2;
  chunk(f, "IHDR", ihdr);
  chunk(f, "IDAT", z);
  chunk(f, "IEND", {});
  return std::fclose(f) == 0;
}

// ---- panels -------------------------------------------------------------------------------------
// A region of a 1920x1080 frame shown at a size: an integer zoom (nearest neighbour: the pixels
// stay visible) or an integer reduction (box average).
struct View {
  int x0, y0, cw, ch;  // the crop in frame pixels
  int zoom = 1;        // > 0: output = crop * zoom
  int reduce = 1;      // > 1: output = crop / reduce (zoom must be 1)
  int ow() const { return reduce > 1 ? cw / reduce : cw * zoom; }
  int oh() const { return reduce > 1 ? ch / reduce : ch * zoom; }
};

const View kFull{0, 0, 1920, 1080, 1, 2};  // the whole frame at 960x540

// rgbAt(x, y, out[3]) gives the frame pixel's display colour (0-1).
using Shader = std::function<void(int, int, float*)>;

Img panel(const View& v, const Shader& at) {
  Img im(v.ow(), v.oh(), 0);
  for (int oy = 0; oy < im.h; ++oy)
    for (int ox = 0; ox < im.w; ++ox) {
      float c[3] = {0, 0, 0};
      if (v.reduce > 1) {
        for (int j = 0; j < v.reduce; ++j)
          for (int i = 0; i < v.reduce; ++i) {
            float s[3];
            at(v.x0 + ox * v.reduce + i, v.y0 + oy * v.reduce + j, s);
            for (int k = 0; k < 3; ++k) c[k] += s[k];
          }
        for (float& e : c) e /= float(v.reduce * v.reduce);
      } else {
        at(v.x0 + ox / v.zoom, v.y0 + oy / v.zoom, c);
      }
      for (int k = 0; k < 3; ++k) im.px[(size_t(oy) * im.w + ox) * 3 + k] = to8(c[k]);
    }
  return im;
}

constexpr int kW = 1920, kH = 1080;

Shader grey(const Buf& m, float lift = 1.0f) {  // lift < 1: v^lift, to show faint values
  return [&m, lift](int x, int y, float* c) {
    float v = m[size_t(y) * kW + x];
    if (lift != 1.0f) v = std::pow(std::max(v, 0.0f), lift);
    c[0] = c[1] = c[2] = v;
  };
}

Shader picture(const Buf& rgb) {
  return [&rgb](int x, int y, float* c) {
    for (int k = 0; k < 3; ++k) c[k] = rgb[(size_t(y) * kW + x) * 3 + k];
  };
}

// Matte Refine's QC views, as plugin.cpp draws them.
Shader showChanges(const Buf& in, const Buf& out) {
  return [&in, &out](int x, int y, float* c) {
    const size_t i = size_t(y) * kW + x;
    const float m = in[i], a = out[i], d = a - m;
    if (d >= 0.0f) {
      c[0] = c[2] = m * (1.0f - d) + d;
      c[1] = m * (1.0f - d);
    } else {
      c[0] = c[2] = a;
      c[1] = std::min(1.0f, a - d);
    }
  };
}

Shader trimapView(const Buf& a) {
  return [&a](int x, int y, float* c) {
    const float v = a[size_t(y) * kW + x];
    c[0] = c[1] = c[2] = v >= 0.98f ? 1.0f : v <= 0.02f ? 0.0f : 0.5f;
  };
}

// Grey Inside: needs the signed distance of a (plugin.cpp: signedEdgeDistance(a, 0.5, 6)).
struct GreyInside {
  Buf sd;
  explicit GreyInside(const Buf& a) : sd(mr::signedEdgeDistance(a.data(), kW, kH, 0.5f, 6.0)) {}
  Shader shader(const Buf& a) const {
    return [&a, this](int x, int y, float* c) {
      const size_t i = size_t(y) * kW + x;
      const float v = a[i];
      const bool g = v > 0.02f && v < 0.98f && sd[i] > 4.0f;
      c[0] = g ? 1.0f : v;
      c[1] = c[2] = g ? 0.15f : v;
    };
  }
};

// The truth's 0.5 contour drawn over a zoomed panel (sub-pixel, bilinear): where the true edge is.
Img withContour(Img im, const View& v, const Buf& truth, float r = 1.0f, float g = 0.15f, float b = 0.15f) {
  auto sample = [&](double fx, double fy) {
    fx = std::min(std::max(fx, 0.0), kW - 1.001), fy = std::min(std::max(fy, 0.0), kH - 1.001);
    const int x = int(fx), y = int(fy);
    const double tx = fx - x, ty = fy - y;
    const float* p = &truth[size_t(y) * kW + x];
    return (p[0] * (1 - tx) + p[1] * tx) * (1 - ty) + (p[kW] * (1 - tx) + p[kW + 1] * tx) * ty;
  };
  const double s = v.reduce > 1 ? double(v.reduce) : 1.0 / v.zoom;
  std::vector<uint8_t> in(size_t(im.w) * im.h);
  for (int oy = 0; oy < im.h; ++oy)
    for (int ox = 0; ox < im.w; ++ox)
      in[size_t(oy) * im.w + ox] = sample(v.x0 + (ox + 0.5) * s - 0.5, v.y0 + (oy + 0.5) * s - 0.5) >= 0.5;
  for (int oy = 0; oy < im.h; ++oy)
    for (int ox = 0; ox < im.w; ++ox) {
      const size_t i = size_t(oy) * im.w + ox;
      if (!in[i]) continue;
      const bool edge = (ox > 0 && !in[i - 1]) || (ox + 1 < im.w && !in[i + 1]) ||
                        (oy > 0 && !in[i - im.w]) || (oy + 1 < im.h && !in[i + im.w]);
      if (edge) im.px[i * 3] = to8(r), im.px[i * 3 + 1] = to8(g), im.px[i * 3 + 2] = to8(b);
    }
  return im;
}

// Panels side by side / grids, with a light separator.
constexpr int kGap = 6;
Img row(const std::vector<Img>& p) {
  int w = 0, h = 0;
  for (const Img& q : p) w += q.w, h = std::max(h, q.h);
  w += kGap * int(p.size() - 1);
  Img out(w, h, 200);
  int x0 = 0;
  for (const Img& q : p) {
    for (int y = 0; y < q.h; ++y)
      std::copy(&q.px[size_t(y) * q.w * 3], &q.px[size_t(y + 1) * q.w * 3], &out.px[(size_t(y) * w + x0) * 3]);
    x0 += q.w + kGap;
  }
  return out;
}
Img column(const std::vector<Img>& p) {
  int w = 0, h = 0;
  for (const Img& q : p) h += q.h, w = std::max(w, q.w);
  h += kGap * int(p.size() - 1);
  Img out(w, h, 200);
  int y0 = 0;
  for (const Img& q : p) {
    for (int y = 0; y < q.h; ++y)
      std::copy(&q.px[size_t(y) * q.w * 3], &q.px[size_t(y + 1) * q.w * 3], &out.px[(size_t(y0 + y) * w) * 3]);
    y0 += q.h + kGap;
  }
  return out;
}

std::string gOut;
int gFailed = 0;
void save(const std::string& name, const Img& im, const std::string& settings) {
  const bool ok = writePng(gOut + "/" + name, im);
  if (!ok) ++gFailed;
  std::printf("%-28s %4dx%-4d %s | %s\n", name.c_str(), im.w, im.h, ok ? "written" : "FAILED", settings.c_str());
}

double bandErr(const Buf& m, const Buf& truth, double band) {  // mean |m - truth| within band px of the true edge
  const Buf sd = mr::signedEdgeDistance(truth.data(), kW, kH, 0.5f, band + 1.0);
  double s = 0;
  long long n = 0;
  for (size_t i = 0; i < m.size(); ++i)
    if (std::fabs(sd[i]) <= band) s += std::fabs(m[i] - truth[i]), ++n;
  return n ? s / n : 0.0;
}

long long changed(const Buf& a, const Buf& b) {
  long long n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

// ---- the images --------------------------------------------------------------------------------
void sceneStrips() {
  const View v{0, 0, kW, kH, 1, 3};  // 640x360 per panel
  std::vector<std::string> names = synth::sceneNames();
  for (const auto& n : synth::mlSceneNames()) names.push_back(n);
  for (const auto& n : names) {
    const synth::Scene s = synth::make(n, kW, kH);
    save("scene_" + n + ".png", row({panel(v, picture(s.rgb)), panel(v, grey(s.matte)), panel(v, grey(s.truth))}),
         "picture | matte | truth");
  }
  const synth::Sequence q = synth::makeSequence(kW, kH);
  save("scene_temporal.png",
       row({panel(v, picture(q.frames[0].rgb)), panel(v, grey(q.frames[0].matte)), panel(v, grey(q.frames[0].truth))}),
       "frame 0: picture | matte | truth");
}

void cleanCrush() {
  const synth::Scene s = synth::make("specks", kW, kH);
  Buf out(s.matte.size());
  mr::CleanBackgroundParams p;  // the plugin's defaults
  mr::CleanBackgroundStats st;
  mr::cleanBackground(s.matte.data(), out.data(), kW, kH, p, &st);
  char b[200];
  std::snprintf(b, sizeof b, "defaults; %d islands removed, %lld px crushed", st.islandsRemoved, st.pixelsCrushed);
  save("op_clean_crush.png",
       column({row({panel(kFull, grey(s.matte)), panel(kFull, grey(out))}),
               row({panel(kFull, grey(s.matte, 0.25f)), panel(kFull, grey(out, 0.25f))})}),
       b);
}

mr::FillHolesParams fillDefaults() {
  mr::FillHolesParams p;
  p.threshold = 0.95f;  // the plugin's Solid Above (the struct's own default is 0.5)
  return p;
}

void fillHoles() {
  const synth::Scene s = synth::make("holes", kW, kH);
  Buf a(s.matte.size()), c(s.matte.size());
  mr::FillHolesParams p = fillDefaults();
  mr::FillHolesStats sa, sc;
  mr::fillHoles(s.matte.data(), a.data(), kW, kH, p, &sa);
  p.threshold = 0.79f;  // what Analyze Frame suggests for this scene
  mr::fillHoles(s.matte.data(), c.data(), kW, kH, p, &sc);
  char b[200];
  std::snprintf(b, sizeof b, "Show Changes; defaults: %d filled, %d too large | Solid Above 0.79: %d filled, %d too large",
                sa.holesFilled, sa.holesTooLarge, sc.holesFilled, sc.holesTooLarge);
  save("op_fill_holes.png", row({panel(kFull, grey(s.matte)), panel(kFull, showChanges(s.matte, a)),
                                 panel(kFull, showChanges(s.matte, c))}),
       b);
}

void solidCore() {
  const synth::Scene s = synth::make("holes", kW, kH);
  Buf f(s.matte.size()), o(s.matte.size());
  mr::FillHolesParams fp = fillDefaults();
  mr::fillHoles(s.matte.data(), f.data(), kW, kH, fp);
  mr::SolidCoreParams p;  // defaults: Depth 6, Feather 2, Edge At 0.5
  mr::SolidCoreStats st;
  mr::solidCore(f.data(), o.data(), kW, kH, p, &st);
  const View v{780, 300, 480, 270, 2, 1};
  const GreyInside gf(f), go(o);
  char b[160];
  std::snprintf(b, sizeof b, "after Fill Holes (defaults) | + Solid Core defaults (%lld px changed); Grey Inside view",
                st.pixelsChanged);
  save("op_solid_core.png", row({panel(v, gf.shader(f)), panel(v, go.shader(o))}), b);
}

void contourSmooth() {
  const synth::Scene s = synth::make("chatter", kW, kH);
  Buf o(s.matte.size());
  mr::ContourSmoothParams p;  // defaults
  mr::ContourSmoothStats st;
  mr::contourSmooth(s.matte.data(), o.data(), kW, kH, p, &st);
  const View v{598, 380, 160, 90, 6, 1};
  char b[200];
  std::snprintf(b, sizeof b, "defaults; band error (3 px) %.3f -> %.3f; %lld px changed%s",
                bandErr(s.matte, s.truth, 3), bandErr(o, s.truth, 3), st.pixelsChanged, st.skipped ? " SKIPPED" : "");
  save("op_contour_smooth.png",
       row({withContour(panel(v, grey(s.matte)), v, s.truth), withContour(panel(v, grey(o)), v, s.truth)}), b);
}

void shrinkGrow() {
  const synth::Scene s = synth::make("eroded", kW, kH);
  Buf o(s.matte.size());
  mr::ShrinkGrowParams p;
  p.amountPx = 3.0;
  mr::shrinkGrow(s.matte.data(), o.data(), kW, kH, p);
  const View v{1135, 800, 160, 90, 6, 1};
  char b[160];
  std::snprintf(b, sizeof b, "Grow (-Shrink) px 3; band error (3 px) %.3f -> %.3f", bandErr(s.matte, s.truth, 3),
                bandErr(o, s.truth, 3));
  save("op_shrink_grow.png",
       row({withContour(panel(v, grey(s.matte)), v, s.truth), withContour(panel(v, grey(o)), v, s.truth)}), b);
}

void edgeWidth() {
  const synth::Scene s = synth::make("mixed_edge", kW, kH);
  Buf o(s.matte.size());
  mr::EdgeWidthParams p;  // defaults: 2 px, mix 1
  mr::EdgeWidthStats st;
  mr::edgeWidth(s.matte.data(), o.data(), kW, kH, p, &st);
  const View soft{1185, 640, 120, 67, 8, 1}, hard{618, 392, 120, 67, 8, 1};
  char b[160];
  std::snprintf(b, sizeof b, "defaults (Edge Width px 2, Width Mix 1); %lld px changed; rows: soft arm, hard-cut side",
                st.pixelsChanged);
  save("op_edge_width.png",
       column({row({panel(soft, grey(s.matte)), panel(soft, grey(o))}),
               row({panel(hard, grey(s.matte)), panel(hard, grey(o))})}),
       b);
}

const View kHead{840, 150, 240, 135, 4, 1};

void snapToImage() {
  const synth::Scene s = synth::make("coarse", kW, kH);
  Buf o(s.matte.size());
  mr::SnapParams p;  // defaults
  mr::snapToImage(s.matte.data(), s.rgb.data(), o.data(), kW, kH, p);
  char b[160];
  std::snprintf(b, sizeof b, "defaults; band error (6 px) %.3f -> %.3f", bandErr(s.matte, s.truth, 6),
                bandErr(o, s.truth, 6));
  save("op_snap_to_image.png",
       row({panel(kHead, picture(s.rgb)), withContour(panel(kHead, grey(s.matte)), kHead, s.truth),
            withContour(panel(kHead, grey(o)), kHead, s.truth)}),
       b);
}

void edgeColourUnmix() {
  const synth::Scene s = synth::make("fgcolor", kW, kH);
  Buf fg(s.rgb.size());
  mr::FgParams p;  // the plugin's defaults: Edge Smoothness 0.005, Extend 3, Ignore Outer 3, Mix 1
  p.ignoreOuterPx = 3.0;
  mr::foregroundColour(s.rgb.data(), s.matte.data(), fg.data(), kW, kH, p);
  // Comp over a new neutral grey: a * colour + (1 - a) * grey.
  const float bg = 0.25f;
  auto comp = [&](const Buf& col) {
    return Shader([&col, &s, bg](int x, int y, float* c) {
      const size_t i = size_t(y) * kW + x;
      const float a = s.matte[i];
      for (int k = 0; k < 3; ++k) c[k] = a * col[i * 3 + k] + (1 - a) * bg;
    });
  };
  double gBefore = 0, gAfter = 0;
  long long n = 0;
  for (size_t i = 0; i < s.matte.size(); ++i)
    if (s.matte[i] > 0.02f && s.matte[i] < 0.98f) {
      gBefore += std::fabs(s.rgb[i * 3 + 1] - s.fg[i * 3 + 1]);
      gAfter += std::fabs(fg[i * 3 + 1] - s.fg[i * 3 + 1]);
      ++n;
    }
  char b[200];
  std::snprintf(b, sizeof b, "defaults; comp over grey 0.25; soft-edge green error vs true colour %.3f -> %.3f",
                gBefore / n, gAfter / n);
  const View v{860, 80, 200, 112, 4, 1};
  save("op_edge_colour_unmix.png", row({panel(v, comp(s.rgb)), panel(v, comp(fg))}), b);
}

void analyze() {
  for (const char* name : {"holes", "all"}) {
    const synth::Scene s = synth::make(name, kW, kH);
    mr::AnalysisResult r;
    mr::analyzeFrame(s.matte.data(), kW, kH, &r);
    std::printf("Analyze Frame on '%s':\n", name);
    for (const auto& n : r.notes) std::printf("  - %s\n", n.c_str());
    if (std::string(name) != "all") continue;
    Buf o = s.matte;
    if (r.cleanEnable) mr::cleanBackground(o.data(), o.data(), kW, kH, r.clean);
    if (r.fillEnable) mr::fillHoles(o.data(), o.data(), kW, kH, r.fill);
    if (r.coreEnable) mr::solidCore(o.data(), o.data(), kW, kH, r.core);
    save("op_analyze_frame.png", row({panel(kFull, grey(s.matte)), panel(kFull, showChanges(s.matte, o))}),
         "scene all: the settings Analyze Frame chose (notes above), Show Changes view");
  }
}

void views() {
  const synth::Scene s = synth::make("holes", kW, kH);
  Buf o(s.matte.size());
  mr::FillHolesParams fp = fillDefaults();
  mr::fillHoles(s.matte.data(), o.data(), kW, kH, fp);
  const GreyInside g(o);
  save("views.png",
       column({row({panel(kFull, grey(o)), panel(kFull, showChanges(s.matte, o))}),
               row({panel(kFull, trimapView(o)), panel(kFull, g.shader(o))})}),
       "holes, Fill Holes (defaults): Matte | Show Changes / Trimap | Grey Inside");
}

#ifdef MR_GUIDE_ML
using namespace mr::ml;

struct OrtSeg : SegRunner {  // as ofx/ml/tests/eval_smart_fill.cpp
  std::shared_ptr<ModelSession> s;
  void segment(const float* rgb, int w, int h, float* mask) override {
    const int S = kBiRefNetSize;
    std::vector<float> input(size_t(3) * S * S), logits(size_t(S) * S);
    birefnetPrepare(rgb, w, h, input.data());
    s->run(input.data(), 3, S, S, logits.data(), 1, S, S);
    birefnetPostprocess(logits.data(), w, h, mask);
  }
};

std::shared_ptr<TileRunner> vitmatte(const std::string& models, const std::string& cuda) {
  const std::string path = models + "/vitmatte-small/model.onnx";
  if (models.empty() || access(path.c_str(), R_OK) != 0) {
    std::printf("SKIPPED Edge Refine images: no %s\n", path.c_str());
    return nullptr;
  }
  RunnerConfig cfg;
  cfg.modelPath = path;
  cfg.modelId = "vitmatte-small";
  cfg.cudaLibDir = cuda;
  RunnerInfo info;
  std::string err;
  auto r = acquireRunner(cfg, &info, &err);
  if (!r) std::printf("SKIPPED Edge Refine images: %s\n", err.c_str());
  else std::printf("ViTMatte Small: ep %s%s%s, load %.0f ms\n", info.ep.c_str(), info.why.empty() ? "" : ", ",
                   info.why.c_str(), info.loadMs);
  return r;
}

void edgeRefine(TileRunner& run) {
  {
    const synth::Scene s = synth::make("coarse", kW, kH);
    Buf o(s.matte.size());
    RefineParams p;  // the node's defaults: Band px 10, Mix 1, Only Grey off
    RefineStats st;
    refineEdge(s.matte.data(), s.rgb.data(), o.data(), kW, kH, p, run, &st);
    Buf sn(s.matte.size());
    mr::snapToImage(s.matte.data(), s.rgb.data(), sn.data(), kW, kH, mr::SnapParams());
    char b[220];
    std::snprintf(b, sizeof b, "Model Small, Band px 10, Mix 1; band error (6 px) input %.3f, Snap %.3f, Edge Refine %.3f",
                  bandErr(s.matte, s.truth, 6), bandErr(sn, s.truth, 6), bandErr(o, s.truth, 6));
    save("ml_edge_refine.png",
         row({panel(kHead, picture(s.rgb)), withContour(panel(kHead, grey(s.matte)), kHead, s.truth),
              withContour(panel(kHead, grey(o)), kHead, s.truth)}),
         b);
  }
  {
    const synth::Scene s = synth::make("hair_flat", kW, kH);
    Buf a(s.matte.size()), c(s.matte.size());
    RefineParams p;
    refineEdge(s.matte.data(), s.rgb.data(), a.data(), kW, kH, p, run, nullptr);
    p.bandPx = 30;
    refineEdge(s.matte.data(), s.rgb.data(), c.data(), kW, kH, p, run, nullptr);
    // The strands the input lost: truth alpha where the input matte is clear (< 0.02); the share
    // of it a result brings back.
    auto strand = [&](const Buf& m) {
      double got = 0, want = 0;
      for (size_t i = 0; i < m.size(); ++i)
        if (s.matte[i] < 0.02f) want += s.truth[i], got += std::min(m[i], s.truth[i]);
      return want > 0 ? got / want : 0.0;
    };
    char b[220];
    std::snprintf(b, sizeof b,
                  "input | truth / Model Small, Band px 10 | Band px 30; lost strand alpha brought back: band 10 %.0f%%, "
                  "band 30 %.0f%%",
                  100 * strand(a), 100 * strand(c));
    const View v{800, 380, 320, 180, 3, 1};
    save("ml_edge_refine_hair.png",
         column({row({panel(v, grey(s.matte)), panel(v, grey(s.truth))}), row({panel(v, grey(a)), panel(v, grey(c))})}),
         b);
  }
}

void smartFillImage(const std::string& models, const std::string& cuda) {
  const std::string path = models + "/birefnet-lite/model.onnx";
  if (models.empty() || access(path.c_str(), R_OK) != 0) {
    std::printf("SKIPPED Smart Fill image: no %s\n", path.c_str());
    return;
  }
  RunnerConfig cfg;
  cfg.modelPath = path;
  cfg.modelId = "birefnet-lite";
  cfg.cudaLibDir = cuda;
  RunnerInfo info;
  std::string err;
  OrtSeg seg;
  seg.s = createModelSession(cfg, &info, &err);
  if (!seg.s) {
    std::printf("SKIPPED Smart Fill image: %s\n", err.c_str());
    return;
  }
  const synth::Scene s = synth::make("gaps", kW, kH);
  Buf f1(s.matte.size()), f2(s.matte.size()), sf(s.matte.size());
  mr::FillHolesParams fp = fillDefaults();
  mr::fillHoles(s.matte.data(), f1.data(), kW, kH, fp);
  fp.maxHolePercent = 0.2;
  mr::fillHoles(s.matte.data(), f2.data(), kW, kH, fp);
  const SmartFillParams p = smartFillParams(SmartFillUi(), 1.0);  // the node's defaults
  SmartFillStats st;
  smartFill(s.matte.data(), s.rgb.data(), kW, kH, p, seg, sf.data(), &st);
  std::vector<uint8_t> marks;
  markHoles(s.matte.data(), kW, kH, p.solidAbove, st, &marks);
  auto score = [&](const Buf& m, int label) {  // share of that label's pixels at >= 0.5
    long long n = 0, k = 0;
    for (size_t i = 0; i < m.size(); ++i)
      if (s.label[i] == label) ++n, k += m[i] >= 0.5f;
    return n ? 100.0 * k / n : 0.0;
  };
  char b[300];
  std::snprintf(b, sizeof b,
                "defect pixels filled / real-gap pixels filled: Fill Holes 1.0 %.0f%% / %.0f%%, 0.2 %.0f%% / %.0f%%, "
                "Smart Fill (Lite, defaults) %.0f%% / %.0f%% (%d holes: %d filled, %d kept)",
                score(f1, 2), score(f1, 3), score(f2, 2), score(f2, 3), score(sf, 2), score(sf, 3), st.holesFound,
                st.holesFilled, st.holesKept + st.holesTooLarge);
  Shader holes = [&](int x, int y, float* c) {
    const size_t i = size_t(y) * kW + x;
    holesViewRgb(marks.empty() ? uint8_t(kMarkNone) : marks[i], sf[i], c);
  };
  // The upper body at 1x (the holes are small): 960x540 crops.
  const View v{480, 200, 960, 540, 1, 1};
  save("ml_smart_fill.png",
       column({row({panel(v, picture(s.rgb)), panel(v, grey(s.matte)), panel(v, showChanges(s.matte, f1))}),
               row({panel(v, showChanges(s.matte, f2)), panel(v, showChanges(s.matte, sf)), panel(v, holes)})}),
       b);
}

void temporal() {
  const synth::Sequence q = synth::makeSequence(kW, kH);
  int t0 = 1;  // three frames t0..t0+2 on which the flickering hole changes
  for (int t = 1; t + 3 < int(q.frames.size()); ++t)
    if (q.holeOn[t] != q.holeOn[t + 1] || q.holeOn[t + 1] != q.holeOn[t + 2]) {
      t0 = t;
      break;
    }
  StabiliseParams p;  // the core's defaults (K = 1, Median, Strength 1, band 6 px)
  std::vector<Img> before, after;
  std::vector<Buf> outs;
  for (int t = t0; t < t0 + 3; ++t) {
    TemporalFrame tf[3];
    for (int k = 0; k < 3; ++k) tf[k].rgb = q.frames[t - 1 + k].rgb.data(), tf[k].matte = q.frames[t - 1 + k].matte.data();
    outs.push_back(Buf(size_t(kW) * kH));
    stabilise(tf, 3, 1, kW, kH, p, outs.back().data());
  }
  for (int k = 0; k < 3; ++k) {
    const int t = t0 + k;
    double cx, cy;  // follow one point of the subject (frame 0: the torso's right side)
    q.subject[t].apply(1055, 470, &cx, &cy);
    const View v{int(cx) - 120, int(cy) - 67, 240, 135, 3, 1};
    before.push_back(withContour(panel(v, grey(q.frames[t].matte)), v, q.frames[t].truth));
    after.push_back(withContour(panel(v, grey(outs[k])), v, q.frames[t].truth));
  }
  std::string on;  // the hole per frame of the sequence, 1 = in the input matte
  for (bool h : q.holeOn) on += h ? '1' : '0';
  char b[200];
  std::snprintf(b, sizeof b, "frames %d-%d of 0-%d (hole per frame %s), core defaults: 1 frame each side, Median, Strength 1, band 6 px",
                t0, t0 + 2, int(q.frames.size()) - 1, on.c_str());
  save("ml_temporal.png", column({row(before), row(after)}), b);
}
#endif

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mr_guide_images <out dir> [<model dir> [<cuda lib dir>]]\n");
    return 2;
  }
  gOut = argv[1];
  sceneStrips();
  cleanCrush();
  fillHoles();
  solidCore();
  contourSmooth();
  shrinkGrow();
  edgeWidth();
  snapToImage();
  edgeColourUnmix();
  analyze();
  views();
#ifdef MR_GUIDE_ML
  const std::string models = argc > 2 ? argv[2] : "", cuda = argc > 3 ? argv[3] : "";
  if (auto r = vitmatte(models, cuda)) edgeRefine(*r);
  smartFillImage(models, cuda);
  temporal();
#else
  std::printf("SKIPPED the Matte Refine ML images: built without MR_BUILD_ML\n");
#endif
  return gFailed ? 1 : 0;
}
