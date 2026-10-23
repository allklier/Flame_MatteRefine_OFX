// mr_trimap_eval (B1 polish, hair trimap): Edge Refine's trimap rule against the synthetic scenes'
// truth with the real ViTMatte (in-process, ort_runner). Not a test: an evaluation tool.
//
//   mr_trimap_eval <models dir> <cuda libs dir> [--model vitmatte-small] [--sizes 1920x1080,960x540]
//                  [--scenes hair_flat,hair_busy,...] [--designs list] [--repeat 3] [--ep cuda]
//
// A design: "core:<band>:<outer>" = refineEdge itself (RefineParams bandPx / outerPx, px at 1080,
// scaled with the frame height); "sym:<B>" = core:B:0; the prototypes "proto:<in>:<out>:<fade>[:g<thr>]"
// re-implement the trimap and the blend here (fade "p" = each side fades over its second half, as
// today; "e" = the outside fades only over its last in/2 px); g<thr> = the outside beyond `in` px is
// unknown only where the picture differs from a background estimate by more than thr (picture-guided);
// "protoT:..." = the same with signedEdgeDistanceThreaded. dPrevMax / dPrevN: max |out - out of the
// previous design in the list| and the pixels where it is > 1e-3 (same scene).
//
// Metrics (the truth is known): err10 = mean |out - truth| over |sd| <= 10 px (at 1080) of the input
// matte, the same region for every design; errAll = sum |out - truth| / sum |in - truth| over the frame
// (1 = no gain); eAbs = 1000 x mean |out - truth| over the frame (for clean mattes, where errAll is 0/0); strand recovery as research/diffmatte/score.py: over pixels where the input == 0,
// rec = sum min(out, truth) / sum truth, recUnk the same over the unknown region only, spill =
// sum max(out - truth, 0) / sum truth; chgOut = pixels changed outside the unknown region (must be 0);
// tiles; ms = warm frame (trimap + tiles + blend), median of --repeat.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "ops/edge_field.h"
#include "ort_runner.h"
#include "synth.h"

using namespace mr::ml;
using Clock = std::chrono::steady_clock;

namespace {
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, c))
    if (!item.empty()) out.push_back(item);
  return out;
}

struct Design {
  std::string text;
  bool core = true;
  double in = 10, out = 0;  // core: band, outer (extra); proto: inside reach, outside reach
  char fade = 'p';
  double guide = -1;  // < 0: not guided
  bool threaded = false;  // "protoT": the distance from signedEdgeDistanceThreaded
};

Design parse(const std::string& t) {
  Design d;
  d.text = t;
  auto f = split(t, ':');
  if (f[0] == "sym") {
    d.in = std::stod(f[1]);
  } else if (f[0] == "core") {
    d.in = std::stod(f[1]);
    d.out = std::stod(f[2]);
  } else {
    d.core = false;
    d.threaded = f[0] == "protoT";
    d.in = std::stod(f[1]);
    d.out = std::stod(f[2]);
    d.fade = f.size() > 3 ? f[3][0] : 'p';
    if (f.size() > 4 && f[4][0] == 'g') d.guide = std::stod(f[4].substr(1));
  }
  return d;
}

// Picture-guided outward reach: background estimate = box mean (radius r) of the picture over the
// pixels with sd < -minOut (clearly outside); a pixel is "detail" when its colour differs from it by
// more than thr (max over channels); dilated by 3 px. Separable box sums over the masked picture.
std::vector<uint8_t> guideMask(const std::vector<float>& rgb, const std::vector<float>& sd, int w, int h,
                               double minOut, int r, double thr) {
  const size_t n = size_t(w) * h;
  std::vector<double> num(n * 4, 0.0), tmp(n * 4, 0.0);
  for (size_t i = 0; i < n; ++i)
    if (sd[i] < -minOut) {
      for (int c = 0; c < 3; ++c) num[i * 4 + c] = rgb[i * 3 + c];
      num[i * 4 + 3] = 1;
    }
  for (int y = 0; y < h; ++y) {  // horizontal running sums
    for (int c = 0; c < 4; ++c) {
      double s = 0;
      for (int x = -r; x < w + r; ++x) {
        if (x + r < w) s += num[(size_t(y) * w + std::min(x + r, w - 1)) * 4 + c] * (x + r >= 0);
        if (x - r - 1 >= 0) s -= num[(size_t(y) * w + x - r - 1) * 4 + c];
        if (x >= 0 && x < w) tmp[(size_t(y) * w + x) * 4 + c] = s;
      }
    }
  }
  for (int x = 0; x < w; ++x)
    for (int c = 0; c < 4; ++c) {
      double s = 0;
      for (int y = -r; y < h + r; ++y) {
        if (y + r < h && y + r >= 0) s += tmp[(size_t(y + r) * w + x) * 4 + c];
        if (y - r - 1 >= 0) s -= tmp[(size_t(y - r - 1) * w + x) * 4 + c];
        if (y >= 0 && y < h) num[(size_t(y) * w + x) * 4 + c] = s;
      }
    }
  std::vector<uint8_t> m(n, 0);
  for (size_t i = 0; i < n; ++i) {
    if (num[i * 4 + 3] <= 0) {
      m[i] = 1;  // no background nearby: keep it unknown
      continue;
    }
    double dmax = 0;
    for (int c = 0; c < 3; ++c)
      dmax = std::max(dmax, (double)std::fabs(clampPicture(rgb[i * 3 + c]) - clampPicture(float(num[i * 4 + c] / num[i * 4 + 3]))));
    m[i] = dmax > thr;
  }
  std::vector<uint8_t> d = mr::dilate(m, w, h, 3.0);
  return d;
}

struct Result {
  double err10 = 0, errAll = 0, eAbs = 0, rec = 0, recUnk = 0, spill = 0, unkFrac = 0, ms = 0;
  long long chgOut = 0, diffN = 0;
  int tiles = 0;
  double diffMax = -1;
};

// The prototype: trimap + tiles + blend here, from the public pieces of edge_refine.
void proto(const synth::Scene& s, const Design& d, double k, TileRunner& run, std::vector<float>* out,
           std::vector<float>* tri, int* tiles) {
  const int w = s.w, h = s.h;
  const size_t n = size_t(w) * h;
  const double in = d.in * k, ou = d.out * k;
  std::vector<float> sd;
  if (d.threaded) {
    auto t = mr::signedEdgeDistanceThreaded(s.matte.data(), w, h, 0.5f, std::max(in, ou) + 2);
    sd.assign(t.get(), t.get() + n);
  } else {
    sd = mr::signedEdgeDistance(s.matte.data(), w, h, 0.5f, std::max(in, ou) + 2);
  }
  std::vector<uint8_t> g;
  if (d.guide >= 0) g = guideMask(s.rgb, sd, w, h, ou, int(std::lround(std::max(ou, 16.0))), d.guide);
  tri->assign(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    const float v = sd[i];
    if (v > in) (*tri)[i] = 1.0f;
    else if (v >= -ou && (v >= -in || g.empty() || g[i])) (*tri)[i] = 0.5f;
  }
  const int T = 512 / 32 * 32 > (std::max(w, h) + 31) / 32 * 32 ? (std::max(w, h) + 31) / 32 * 32 : 512;
  const int overlap = std::min(32, T / 2);
  const std::vector<Tile> ts = selectTiles(*tri, w, h, T, overlap);
  *tiles = int(ts.size());
  std::vector<float> acc(n, 0.0f), ws(n, 0.0f), input(size_t(4) * T * T), alpha(size_t(T) * T);
  for (const Tile& t : ts) {
    prepareTile(s.rgb.data(), tri->data(), w, h, t, T, input.data());
    run.run(input.data(), T, alpha.data(), t.x, t.y);
    accumulateTile(alpha.data(), t, T, overlap, w, h, acc.data(), ws.data());
  }
  out->assign(s.matte.begin(), s.matte.end());
  for (size_t i = 0; i < n; ++i) {
    if ((*tri)[i] != 0.5f || !(ws[i] > 0)) continue;
    const double ml = acc[i] / ws[i], v = s.matte[i], x = sd[i];
    double wt;
    if (x >= 0) {
      wt = x <= in / 2 ? 1.0 : (in - x) / (in / 2);
    } else {
      const double dd = -x;
      if (d.fade == 'e') {
        const double f = std::min(in / 2, ou);
        wt = dd <= ou - f ? 1.0 : (ou - dd) / f;
      } else {
        wt = dd <= ou / 2 ? 1.0 : (ou - dd) / (ou / 2);
      }
    }
    wt = std::min(std::max(wt, 0.0), 1.0);
    (*out)[i] = wt >= 1.0 ? float(ml) : float(v + wt * (ml - v));
  }
}

Result evaluate(const synth::Scene& s, const Design& d, TileRunner& run, int repeat) {
  const int w = s.w, h = s.h;
  const size_t n = size_t(w) * h;
  const double k = h / 1080.0;
  std::vector<float> out, tri;
  Result r;
  std::vector<double> times;
  for (int rep = 0; rep <= repeat; ++rep) {  // the first is a warm-up
    const auto t0 = Clock::now();
    if (d.core) {
      RefineParams p;
      p.bandPx = d.in * k;
      p.outerPx = d.out * k;
      RefineStats st;
      out.resize(n);
      refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, run, &st, std::function<bool()>(), &tri);
      r.tiles = st.tiles;
    } else {
      proto(s, d, k, run, &out, &tri, &r.tiles);
    }
    if (rep) times.push_back(msSince(t0));
  }
  std::sort(times.begin(), times.end());
  r.ms = times.empty() ? 0 : times[times.size() / 2];
  const std::vector<float> sd0 = mr::signedEdgeDistance(s.matte.data(), w, h, 0.5f, 10 * k + 2);
  double e10 = 0, eIn = 0, eOut = 0, L = 0, R = 0, Lu = 0, Ru = 0, S = 0;
  long n10 = 0, unk = 0;
  for (size_t i = 0; i < n; ++i) {
    const double t = s.truth[i], a = out[i], m = s.matte[i];
    if (std::fabs(sd0[i]) <= 10 * k) {
      e10 += std::fabs(a - t);
      ++n10;
    }
    eIn += std::fabs(m - t);
    eOut += std::fabs(a - t);
    const bool u = tri[i] == 0.5f;
    unk += u;
    if (!u && a != m) ++r.chgOut;
    if (m == 0) {
      L += t;
      R += std::min(a, t);
      S += std::max(a - t, 0.0);
      if (u) {
        Lu += t;
        Ru += std::min(a, t);
      }
    }
  }
  r.err10 = n10 ? e10 / n10 : 0;
  r.errAll = eIn > 0 ? eOut / eIn : 0;
  r.eAbs = 1000.0 * eOut / double(n);
  r.rec = L > 0 ? R / L : 0;
  r.recUnk = Lu > 0 ? Ru / Lu : 0;
  r.spill = L > 0 ? S / L : 0;
  r.unkFrac = double(unk) / n;
  // Against the previous design of the list on the same scene (identical trimap rules but another
  // distance function: how much the output moves).
  static std::vector<float> prev;
  static std::string prevScene;
  r.diffMax = -1;
  if (prevScene == s.name + std::to_string(w) && prev.size() == n) {
    r.diffMax = 0;
    for (size_t i = 0; i < n; ++i) {
      const double dd = std::fabs(double(out[i]) - prev[i]);
      r.diffMax = std::max(r.diffMax, dd);
      r.diffN += dd > 1e-3;
    }
  }
  prev = out;
  prevScene = s.name + std::to_string(w);
  return r;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: mr_trimap_eval <models dir> <cuda libs dir> [options]\n");
    return 2;
  }
  std::string model = "vitmatte-small", ep = "cuda";
  std::vector<std::string> sizes{"1920x1080", "960x540"};
  std::vector<std::string> scenes{"hair_flat", "hair_busy", "coarse", "eroded", "mixed_edge", "fgcolor", "all"};
  std::vector<std::string> designs{"sym:10", "sym:20", "sym:30", "sym:45", "sym:60"};
  int repeat = 3;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--model") model = v;
    else if (k == "--sizes") sizes = split(v, ',');
    else if (k == "--scenes") scenes = split(v, ',');
    else if (k == "--designs") designs = split(v, ',');
    else if (k == "--repeat") repeat = std::atoi(v.c_str());
    else if (k == "--ep") ep = v;
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  RunnerConfig cfg;
  cfg.modelPath = std::string(argv[1]) + "/" + model + "/model.onnx";
  cfg.modelId = model;
  cfg.ep = ep;
  cfg.cudaLibDir = argv[2];
  RunnerInfo info;
  std::string err;
  auto runner = acquireRunner(cfg, &info, &err);
  if (!runner) {
    std::fprintf(stderr, "no runner: %s\n", err.c_str());
    return 1;
  }
  std::printf("model %s ep %s%s%s\n", model.c_str(), info.ep.c_str(), info.why.empty() ? "" : " because ",
              info.why.c_str());
  std::printf("%-10s %-9s %-22s %7s %7s %7s %6s %6s %6s %6s %6s %5s %8s %8s %8s\n", "scene", "size", "design", "err10",
              "errAll", "eAbs", "rec", "recUnk", "spill", "unk%", "chgOut", "tiles", "ms", "dPrevMax", "dPrevN");
  for (const std::string& sz : sizes) {
    int w = 0, h = 0;
    std::sscanf(sz.c_str(), "%dx%d", &w, &h);
    for (const std::string& sc : scenes) {
      const synth::Scene s = synth::make(sc, w, h);
      for (const std::string& dt : designs) {
        const Design d = parse(dt);
        const Result r = evaluate(s, d, *runner, repeat);
        std::printf("%-10s %-9s %-22s %7.4f %7.3f %7.3f %6.3f %6.3f %6.3f %6.2f %6lld %5d %8.1f %8.4f %8lld\n", sc.c_str(),
                    sz.c_str(), dt.c_str(), r.err10, r.errAll, r.eAbs, r.rec, r.recUnk, r.spill, 100 * r.unkFrac,
                    r.chgOut, r.tiles, r.ms, r.diffMax, r.diffN);
        std::fflush(stdout);
      }
    }
  }
  return 0;
}
