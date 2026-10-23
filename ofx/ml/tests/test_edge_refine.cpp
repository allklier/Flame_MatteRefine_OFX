// Edge Refine (B1) core with fake models: outside the band bit-identical (decision 17), a
// truth-returning model fixes the synthetic "coarse" and "eroded" edges, Mix 0 / 1, Only Grey,
// abort mid-way, tile coverage, in place, tiny frames, the model input (clamp only, no gamma); Outer px
// (the asymmetric band for hair: trimap, weights, untouched beyond, 0 = the earlier rule bit for bit).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "edge_refine.h"
#include "ops/edge_field.h"
#include "synth.h"

using namespace mr::ml;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// Returns a frame-sized image's values for the tile (beyond the frame: the edge repeated).
struct FrameRunner : TileRunner {
  const std::vector<float>* img;
  int w, h, calls = 0;
  FrameRunner(const std::vector<float>* i, int w, int h) : img(i), w(w), h(h) {}
  void run(const float*, int T, float* alpha, int x, int y) override {
    ++calls;
    for (int ly = 0; ly < T; ++ly)
      for (int lx = 0; lx < T; ++lx)
        alpha[size_t(ly) * T + lx] = (*img)[size_t(std::min(y + ly, h - 1)) * w + std::min(x + lx, w - 1)];
  }
};
struct ConstRunner : TileRunner {
  float v;
  int calls = 0;
  explicit ConstRunner(float v) : v(v) {}
  void run(const float* input, int T, float* alpha, int, int) override {
    ++calls;
    // The input layout: 4 planes, the trimap only 0 / 0.5 / 1, RGB normalised to [-1, 1].
    for (size_t i = 0; i < size_t(T) * T; ++i) {
      const float t = input[3 * size_t(T) * T + i];
      if (!(t == 0.0f || t == 0.5f || t == 1.0f) || input[i] < -1.0f || input[i] > 1.0f)
        throw std::runtime_error("bad input");
      alpha[i] = v;
    }
  }
};

std::vector<float> signedEdgeDistanceForTest(const std::vector<float>& m, int w, int h, double band) {
  return mr::signedEdgeDistance(m.data(), w, h, 0.5f, band);
}

double bandError(const std::vector<float>& a, const std::vector<float>& truth, const std::vector<float>& sd,
                 double band) {
  double sum = 0;
  long n = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(sd[i]) <= band) {
      sum += std::fabs(a[i] - truth[i]);
      ++n;
    }
  return n ? sum / n : 0;
}

void testScene(const char* name, double maxRatio) {
  const int w = 960, h = 540;
  synth::Scene s = synth::make(name, w, h);
  RefineParams p;
  p.bandPx = 5;  // 10 px at 1080
  std::vector<float> out(s.matte.size()), sd, tri;
  buildTrimap(s.matte.data(), w, h, p.bandPx, &sd, &tri);
  FrameRunner truth(&s.truth, w, h);
  RefineStats st;
  refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, truth, &st);
  long outside = 0, insideDiff = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::fabs(sd[i]) >= p.bandPx) outside += out[i] != s.matte[i];
    // Mix 1: the model where w = 1 (up to the float rounding of the overlap average).
    if (std::fabs(sd[i]) <= p.bandPx / 2) insideDiff += std::fabs(out[i] - s.truth[i]) > 1e-6f;
  }
  const double ein = bandError(s.matte, s.truth, sd, p.bandPx), eout = bandError(out, s.truth, sd, p.bandPx);
  std::printf("%s: %d tiles (T %d), band %.2f %%, band error %.4f -> %.4f, %lld px changed\n", name, st.tiles,
              st.tileSize, 100 * st.bandFraction, ein, eout, st.pixelsChanged);
  CHECK(outside == 0);     // decision 17
  CHECK(insideDiff == 0);
  CHECK(eout < ein * maxRatio);
  CHECK(st.tilesRun == st.tiles && st.tiles > 0 && truth.calls == st.tiles && !st.aborted);
}

// The blend of the symmetric rule as it was before Outer px (copied from edge_refine.cpp at 0.0.1):
// Outer px 0 must give exactly this.
void blendBefore(const float* in, const float* ml, const float* sd, size_t n, double band, double mix,
                 bool onlyGrey, float* out) {
  for (size_t i = 0; i < n; ++i) {
    const float v = in[i];
    out[i] = v;
    const double d = std::fabs(sd[i]);
    if (d >= band || ml[i] < 0) continue;
    if (onlyGrey && (v <= 0.02f || v >= 0.98f)) continue;
    const double wt = (d <= band / 2 ? 1.0 : (band - d) / (band / 2)) * mix;
    out[i] = wt >= 1.0 ? ml[i] : float(v + wt * (ml[i] - v));
  }
}

// Outer px (hair): the unknown band reaches band + outer outside the edge, band inside; outside the
// matte the weight is 1 up to band + outer - band / 2 and fades to 0 at band + outer; inside as
// before; beyond the band bit-identical; Outer px 0 = the symmetric rule bit for bit.
void testOuter() {
  const int w = 960, h = 540;
  // Outer 0: bit-identical to the rule before Outer px (blend and trimap), Mix 1 and 0.5, Only Grey.
  for (const char* name : {"hair_busy", "coarse", "all"}) {
    synth::Scene s = synth::make(name, w, h);
    const double band = 5;
    std::vector<float> sd, tri;
    buildTrimap(s.matte.data(), w, h, band, &sd, &tri);
    CHECK(sd == signedEdgeDistanceForTest(s.matte, w, h, band + 2));
    long triBad = 0;
    for (size_t i = 0; i < sd.size(); ++i)
      triBad += tri[i] != (sd[i] > band ? 1.0f : sd[i] < -band ? 0.0f : 0.5f);
    CHECK(triBad == 0);
    std::vector<float> ml(s.truth);
    for (size_t i = 0; i < ml.size(); i += 7) ml[i] = -1.0f;  // some pixels without a model value
    for (double mix : {1.0, 0.5})
      for (bool grey : {false, true}) {
        std::vector<float> a(ml.size()), b(ml.size());
        blendBack(s.matte.data(), ml.data(), sd.data(), ml.size(), band, mix, grey, a.data());
        blendBefore(s.matte.data(), ml.data(), sd.data(), ml.size(), band, mix, grey, b.data());
        CHECK(a == b);
        blendBack(s.matte.data(), ml.data(), sd.data(), ml.size(), band, mix, grey, a.data(), 0.0);
        CHECK(a == b);
      }
  }
  synth::Scene s = synth::make("hair_flat", w, h);
  const double band = 5, outer = 40, reach = band + outer;
  std::vector<float> sd, tri;
  buildTrimap(s.matte.data(), w, h, band, &sd, &tri, outer);
  // The trimap: unknown for -(band + outer) <= sd <= band; sd exact out to the reach (not clamped).
  long triBad = 0, farOut = 0;
  for (size_t i = 0; i < sd.size(); ++i) {
    triBad += tri[i] != (sd[i] > band ? 1.0f : sd[i] < -reach ? 0.0f : 0.5f);
    farOut += sd[i] < -(reach - 1) && sd[i] >= -reach;
  }
  CHECK(triBad == 0 && farOut > 0);
  CHECK(trimapFromDistance(std::vector<float>{-45.0f, -45.5f, 5.0f, 5.5f, -5.5f}, 5, 40) ==
        (std::vector<float>{0.5f, 0.0f, 0.5f, 1.0f, 0.5f}));
  // The weights.
  CHECK(blendWeight(0, band, outer) == 1.0 && blendWeight(2.5, band, outer) == 1.0);
  CHECK(std::fabs(blendWeight(3.75, band, outer) - 0.5) < 1e-12 && blendWeight(5, band, outer) == 0.0);
  CHECK(blendWeight(-42.5, band, outer) == 1.0 && blendWeight(-45, band, outer) == 0.0);
  CHECK(std::fabs(blendWeight(-43.75, band, outer) - 0.5) < 1e-12 && blendWeight(-46, band, outer) == 0.0);
  CHECK(blendWeight(-3.75, band, 0) == blendWeight(3.75, band, 0));
  // A constant model: the band (both reaches) gets 0.5 by the weight, the rest is untouched.
  ConstRunner c(0.5f);
  RefineParams p;
  p.bandPx = band;
  p.outerPx = outer;
  std::vector<float> out(s.matte.size()), trimapOut;
  RefineStats st;
  refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, c, &st, std::function<bool()>(), &trimapOut);
  CHECK(trimapOut == tri);
  long outside = 0, wrong = 0, changedOut = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const double wt = blendWeight(sd[i], band, outer);
    if (sd[i] > band || sd[i] < -reach) outside += out[i] != s.matte[i];
    else wrong += std::fabs(out[i] - (s.matte[i] + wt * (0.5 - s.matte[i]))) > 1e-5;
    changedOut += sd[i] < -band - 1 && out[i] != s.matte[i];
  }
  CHECK(outside == 0 && wrong == 0 && changedOut > 0);
  // Inside the matte (sd >= 0) the result does not depend on Outer px (a model that ignores its input).
  FrameRunner truth(&s.truth, w, h);
  std::vector<float> o0(out.size()), o40(out.size());
  RefineParams q = p;
  q.outerPx = 0;
  refineEdge(s.matte.data(), s.rgb.data(), o0.data(), w, h, q, truth, nullptr);
  refineEdge(s.matte.data(), s.rgb.data(), o40.data(), w, h, p, truth, nullptr);
  long insideDiff = 0, strands0 = 0, strands40 = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (sd[i] >= 0 && std::fabs(o0[i] - o40[i]) > 1e-6f) {
      if (insideDiff < 3) std::printf("  inside diff at %zu: sd %g in %g o0 %g o40 %g\n", i, sd[i], s.matte[i], o0[i], o40[i]);
      ++insideDiff;
    }
    if (s.matte[i] == 0) {
      strands0 += o0[i] > 0.1f;
      strands40 += o40[i] > 0.1f;
    }
  }
  // A truth-returning model: Outer px brings back strand pixels the symmetric band cannot reach.
  std::printf("outer px 40 @540: tiles %d, strand pixels > 0.1 where the matte was 0: %ld (outer 0) -> %ld\n",
              st.tiles, strands0, strands40);
  CHECK(insideDiff == 0 && strands40 > 2 * strands0);
  // Every unknown pixel of the asymmetric trimap lies in a selected tile.
  const std::vector<Tile> tiles = selectTiles(tri, w, h, 256, 32);
  std::vector<float> wsum(tri.size(), 0.0f);
  for (const Tile& t : tiles)
    for (int ly = 0; ly < 256 && t.y + ly < h; ++ly)
      for (int lx = 0; lx < 256 && t.x + lx < w; ++lx)
        wsum[size_t(t.y + ly) * w + t.x + lx] += tileWeight(t, 256, 32, w, h, lx, ly);
  long missed = 0;
  for (size_t i = 0; i < tri.size(); ++i) missed += tri[i] == 0.5f && !(wsum[i] > 0);
  CHECK(missed == 0);
  // A negative Outer px is 0.
  q.outerPx = -3;
  std::vector<float> oneg(out.size());
  refineEdge(s.matte.data(), s.rgb.data(), oneg.data(), w, h, q, truth, nullptr);
  CHECK(oneg == o0);
}

}  // namespace

int main() {
  testOuter();
  testScene("coarse", 0.5);
  testScene("eroded", 0.5);

  const int w = 640, h = 360;
  synth::Scene s = synth::make("all", w, h);
  std::vector<float> sd, tri;
  RefineParams p;
  p.bandPx = 6;
  p.tile = 256;
  buildTrimap(s.matte.data(), w, h, p.bandPx, &sd, &tri);

  // Mix 0 returns the input exactly; a constant model changes only the band.
  {
    ConstRunner c(0.5f);
    std::vector<float> out(s.matte.size());
    RefineParams q = p;
    q.mix = 0;
    RefineStats st;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, q, c, &st);
    CHECK(out == s.matte);
    CHECK(st.pixelsChanged == 0);
    q.mix = 1;
    std::vector<float> trimapOut;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, q, c, &st, std::function<bool()>(), &trimapOut);
    CHECK(trimapOut == tri);
    long outside = 0, wrong = 0, changed = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      const double d = std::fabs(sd[i]);
      if (d >= q.bandPx) outside += out[i] != s.matte[i];
      if (d <= q.bandPx / 2) wrong += out[i] != 0.5f;
      // Between: the linear fade from the model (|sd| = band / 2) to the input (|sd| = band).
      if (d > q.bandPx / 2 && d < q.bandPx) {
        const double wt = (q.bandPx - d) / (q.bandPx / 2);
        wrong += std::fabs(out[i] - (s.matte[i] + wt * (0.5 - s.matte[i]))) > 1e-5;
      }
      changed += out[i] != s.matte[i];
    }
    CHECK(outside == 0);
    CHECK(wrong == 0);
    CHECK(changed > 0 && changed == st.pixelsChanged);
    // Mix 0.5: half way where w = 1.
    q.mix = 0.5;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, q, c, &st);
    long bad = 0;
    for (size_t i = 0; i < out.size(); ++i)
      if (std::fabs(sd[i]) <= q.bandPx / 2) bad += std::fabs(out[i] - (s.matte[i] + 0.5f * (0.5f - s.matte[i]))) > 1e-6f;
    CHECK(bad == 0);
  }
  // Only Grey keeps the input's near-0 / near-1 pixels.
  {
    ConstRunner c(0.5f);
    std::vector<float> out(s.matte.size());
    RefineParams q = p;
    q.onlyGrey = true;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, q, c, nullptr);
    long kept = 0, solid = 0, greyChanged = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      const float v = s.matte[i];
      if (v <= 0.02f || v >= 0.98f) {
        ++solid;
        kept += out[i] == v;
      } else if (std::fabs(sd[i]) <= q.bandPx / 2) {
        greyChanged += out[i] == 0.5f;
      }
    }
    CHECK(kept == solid);
    CHECK(greyChanged > 0);
  }
  // Abort mid-way: the input, unchanged; in place too.
  {
    ConstRunner c(0.5f);
    std::vector<float> out(s.matte.size(), 7.0f);
    int asked = 0;
    RefineStats st;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, c, &st, [&]() { return ++asked > 2; });
    CHECK(st.aborted && st.tilesRun == 2 && c.calls == 2 && st.tiles > 2);
    CHECK(out == s.matte);
    std::vector<float> inPlace = s.matte;
    refineEdge(inPlace.data(), s.rgb.data(), inPlace.data(), w, h, p, c, &st, [&]() { return true; });
    CHECK(st.aborted && inPlace == s.matte);
  }
  // In place gives the same as out of place.
  {
    FrameRunner truth(&s.truth, w, h);
    std::vector<float> out(s.matte.size()), inPlace = s.matte;
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, truth, nullptr);
    refineEdge(inPlace.data(), s.rgb.data(), inPlace.data(), w, h, p, truth, nullptr);
    CHECK(out == inPlace);
  }
  // Every unknown pixel lies in a selected tile with a weight > 0 (as mr_ml_bench --selfcheck).
  for (int T : {64, 256, 512}) {
    const std::vector<Tile> tiles = selectTiles(tri, w, h, T, 32);
    std::vector<float> wsum(tri.size(), 0.0f);
    for (const Tile& t : tiles)
      for (int ly = 0; ly < T && t.y + ly < h; ++ly)
        for (int lx = 0; lx < T && t.x + lx < w; ++lx)
          wsum[size_t(t.y + ly) * w + t.x + lx] += tileWeight(t, T, 32, w, h, lx, ly);
    long missed = 0;
    for (size_t i = 0; i < tri.size(); ++i) missed += tri[i] == 0.5f && !(wsum[i] > 0);
    CHECK(missed == 0);
  }
  // Tiny frames and a frame smaller than one tile: no crash, outside the band untouched.
  const int sizes[][2] = {{1, 1}, {31, 17}, {2, 300}, {200, 120}};
  for (const auto& sz : sizes) {
    const int tw = sz[0], th = sz[1];
    std::vector<float> m(size_t(tw) * th), rgb(m.size() * 3, 0.3f), out(m.size());
    for (int y = 0; y < th; ++y)
      for (int x = 0; x < tw; ++x) m[size_t(y) * tw + x] = x < tw / 2 ? 1.0f : 0.0f;
    ConstRunner c(0.25f);
    RefineStats st;
    refineEdge(m.data(), rgb.data(), out.data(), tw, th, p, c, &st);
    std::printf("tiny %dx%d: %d tiles of %d\n", tw, th, st.tiles, st.tileSize);
    CHECK(st.tileSize >= 32 && st.tileSize % 32 == 0 && st.tileSize <= 512);
    std::vector<float> tsd, ttri;
    buildTrimap(m.data(), tw, th, p.bandPx, &tsd, &ttri);
    long outside = 0;
    for (size_t i = 0; i < m.size(); ++i) outside += std::fabs(tsd[i]) >= p.bandPx && out[i] != m[i];
    CHECK(outside == 0);
  }
  // prepareTile (rows in parallel) equals the plain single-threaded loop bit for bit, frame-edge
  // tiles included (the pixels beyond the frame repeat the edge).
  {
    synth::Scene sc = synth::make("coarse", 700, 500);
    std::vector<float> tsd, ttri;
    buildTrimap(sc.matte.data(), 700, 500, 10.0, &tsd, &ttri);
    const int T = 256;
    std::vector<float> got(size_t(4) * T * T), ref(got.size());
    long diff = 0;
    for (const Tile t : {Tile{0, 0}, Tile{444, 244}, Tile{600, 400}, Tile{123, 77}}) {
      prepareTile(sc.rgb.data(), ttri.data(), 700, 500, t, T, got.data());
      const size_t plane = size_t(T) * T;
      for (int ly = 0; ly < T; ++ly)
        for (int lx = 0; lx < T; ++lx) {
          const size_t i = size_t(std::min(t.y + ly, 499)) * 700 + std::min(t.x + lx, 699), o = size_t(ly) * T + lx;
          for (int c = 0; c < 3; ++c) ref[c * plane + o] = (clampPicture(sc.rgb[i * 3 + c]) - 0.5f) / 0.5f;
          ref[3 * plane + o] = ttri[i];
        }
      for (size_t i = 0; i < got.size(); ++i) diff += got[i] != ref[i];
    }
    std::printf("prepareTile vs the single-threaded loop: %ld values differ\n", diff);
    CHECK(diff == 0);
  }
  // The model input is the picture clamped to 0-1, no gamma (the picture is display-referred
  // already: Picture Colour Auto / Bypass): 3.1 -> 1, -0.2 -> 0, 0.5 unchanged, NaN -> 0; then
  // (x - 0.5) / 0.5, so -1 / 1 / 0 exactly in the tile.
  {
    CHECK(clampPicture(3.1f) == 1.0f);
    CHECK(clampPicture(-0.2f) == 0.0f);
    CHECK(clampPicture(0.5f) == 0.5f);
    CHECK(clampPicture(0.25f) == 0.25f);
    CHECK(clampPicture(std::nanf("")) == 0.0f);
    const int tw = 40, th = 3, T = 64;
    std::vector<float> rgb(size_t(tw) * th * 3), tt(size_t(tw) * th, 0.5f), in(size_t(4) * T * T);
    const float vals[3] = {3.1f, -0.2f, 0.5f};  // row 0: 3.1, row 1: -0.2, row 2: 0.5 (all channels)
    for (int y = 0; y < th; ++y)
      for (size_t i = 0; i < size_t(tw) * 3; ++i) rgb[size_t(y) * tw * 3 + i] = vals[y];
    prepareTile(rgb.data(), tt.data(), tw, th, Tile{0, 0}, T, in.data());
    const float want[3] = {1.0f, -1.0f, 0.0f};
    long bad = 0;
    for (int c = 0; c < 3; ++c)
      for (int ly = 0; ly < T; ++ly)
        for (int lx = 0; lx < T; ++lx) bad += in[size_t(c) * T * T + size_t(ly) * T + lx] != want[std::min(ly, th - 1)];
    std::printf("model input of 3.1 / -0.2 / 0.5: %ld values differ from 1 / -1 / 0\n", bad);
    CHECK(bad == 0);
  }
  std::printf(failures ? "test_edge_refine: %d FAILED\n" : "test_edge_refine: all passed\n", failures);
  return failures ? 1 : 0;
}
