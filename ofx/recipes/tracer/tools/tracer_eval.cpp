// tracer_eval: development tool for the strand tracer (not a test; nothing here is needed by ctest).
//   tracer_eval scenes <dir> <seeds> [config,...]
//       reads scene dumps <dir>/<scene>_<seed>.bin written by the prototype's generator (format below) and prints
//       the ablation table with the prototype's metrics, the C++ port with the rulings OFF (prototype mode);
//   tracer_eval cscenes <seeds> [config,...]
//       the same on the C++ scenes (tracer_scenes.h), prototype mode; add "+rulings" to a config to turn them on;
//   config: <ablation>[+rulings|+ref|+edge|+len|+elong|+noelong][+kNN] (kNN = unanchored keep bar 0.NN)
//   tracer_eval real <rgb.f32> <base.f32> <region.f32> <w> <h> [frameHeight] [config]
//       raw little-endian float32 dumps (rgb interleaved w*h*3, base w*h, region w*h); prints NUMBERS ONLY
//       (counts, lengths, confidences), never pixels.
// Scene dump: int32 w, h, npolys; per poly int32 n, float64 width, n x (float64 x, y); float32 rgb[w*h*3],
// alpha[w*h], region[w*h]; float64 gt[w*h]; uint8 dmask[w*h]; int32 ncreases; ncreases x float64[w*h].
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strand_tracer.h"
#include "tracer_scenes.h"

using namespace mr::tracer;
using scenes::Metrics;

namespace {

bool config(const std::string& name, TracerParams& p) {
  p = TracerParams();
  // "<ablation>[+rulings|+ref|+edge|+len...]": prototype mode unless rulings are added (all, or one by one)
  std::string n = name.substr(0, name.find('+'));
  const std::string extra = name.find('+') == std::string::npos ? "" : name.substr(name.find('+'));
  const bool all = extra.find("+rulings") != std::string::npos;
  p.frameHairRef = all || extra.find("+ref") != std::string::npos;
  p.dropEdgeTouching = all || extra.find("+edge") != std::string::npos;
  p.minPieceLen1080 = all || extra.find("+len") != std::string::npos ? 12.0 : 0.0;
  // unanchored keep bar: the prototype's 0.7 in prototype mode, the default with +rulings; "+kNN" = 0.NN
  p.minConfUnanchored = all ? TracerParams().minConfUnanchored : 0.7;
  p.rElongation = all || extra.find("+elong") != std::string::npos;  // added after the prototype
  if (extra.find("+noelong") != std::string::npos) p.rElongation = false;
  const size_t kp = extra.find("+k");
  if (kp != std::string::npos) p.minConfUnanchored = std::atof(extra.substr(kp + 2, 2).c_str()) / 100.0;
  if (n == "all") return true;
  if (n == "-turn") p.rTurn = false;
  else if (n == "-width") p.rWidth = false;
  else if (n == "-profile") p.rProfile = false;
  else if (n == "-colour") p.rColour = false;
  else if (n == "-contrast") p.rContrast = false;
  else if (n == "-orient") p.rOrient = false;
  else if (n == "-decay") p.rDecay = false;
  else if (n == "-connect") p.rConnect = false;
  else if (n == "-budget") p.rBudget = false;
  else if (n == "-hairctx") p.rHairCtx = false;
  else if (n == "capsule") p.capsuleAlpha = true;
  else if (n == "none")
    p.rTurn = p.rWidth = p.rProfile = p.rColour = p.rContrast = p.rOrient = p.rDecay = p.rConnect = p.rBudget =
        p.rHairCtx = false;
  else
    return false;
  return true;
}

std::vector<std::string> split(const char* s) {
  std::vector<std::string> o;
  std::string cur;
  for (const char* c = s; *c; ++c) {
    if (*c == ',') {
      o.push_back(cur);
      cur.clear();
    } else {
      cur += *c;
    }
  }
  if (!cur.empty()) o.push_back(cur);
  return o;
}

template <class T>
bool rd(FILE* f, T* v, size_t n) { return fread(v, sizeof(T), n, f) == n; }

bool loadScene(const std::string& path, scenes::Scene& S) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  int32_t hd[3];
  bool ok = rd(f, hd, 3);
  S.w = hd[0];
  S.h = hd[1];
  const size_t n = (size_t)S.w * S.h;
  for (int k = 0; ok && k < hd[2]; ++k) {
    int32_t m;
    double w;
    ok = rd(f, &m, 1) && rd(f, &w, 1);
    std::vector<double> xy(2 * (size_t)m);
    ok = ok && rd(f, xy.data(), xy.size());
    scenes::Strand st;
    st.width = w;
    for (int i = 0; i < m; ++i) st.poly.push_back(Point{xy[2 * i], xy[2 * i + 1]});
    S.polys.push_back(st);
  }
  S.rgb.resize(3 * n);
  S.alpha.resize(n);
  S.region.resize(n);
  S.gt.resize(n);
  S.dmask.resize(n);
  ok = ok && rd(f, S.rgb.data(), 3 * n) && rd(f, S.alpha.data(), n) && rd(f, S.region.data(), n) &&
       rd(f, S.gt.data(), n) && rd(f, S.dmask.data(), n);
  int32_t nc = 0;
  ok = ok && rd(f, &nc, 1);
  for (int k = 0; ok && k < nc; ++k) {
    std::vector<double> c(n);
    ok = rd(f, c.data(), n);
    S.creases.push_back(c);
  }
  fclose(f);
  return ok;
}

void printRow(const char* a, const char* s, const Metrics& r) {
  std::printf("%-15s %-11s rec %.2f false %.2f distr %.2f sad %.2f wander %d early %d missed %d\n", a, s, r.rec, r.fals,
              r.distr, r.sad, r.wander, r.early, r.missed);
}

TracerResult run(const scenes::Scene& S, const TracerParams& p) {
  TracerInput in;
  in.rgb = S.rgb.data();
  in.base = S.alpha.data();
  in.region = S.region.data();
  in.width = S.w;
  in.height = S.h;
  in.frameHeight = 1080;
  return traceStrands(in, p);
}

int scenesMode(const char* dir, int seeds, const std::vector<std::string>& cfgs, bool dumps) {
  for (const std::string& a : cfgs) {
    TracerParams p;
    if (!config(a, p)) {
      std::fprintf(stderr, "unknown config %s\n", a.c_str());
      return 2;
    }
    std::vector<Metrics> rows;
    double ms = 0;
    for (const std::string& s : scenes::sceneNames()) {
      std::vector<Metrics> acc;
      for (int k = 0; k < seeds; ++k) {
        scenes::Scene S;
        if (dumps) {
          if (!loadScene(std::string(dir) + "/" + s + "_" + std::to_string(k) + ".bin", S)) {
            std::fprintf(stderr, "cannot read scene %s %d\n", s.c_str(), k);
            return 2;
          }
        } else {
          S = scenes::makeScene(s, k);
        }
        const auto t0 = std::chrono::steady_clock::now();
        const TracerResult r = run(S, p);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        acc.push_back(scenes::metrics(S, r.alpha, r.pieces));
      }
      rows.push_back(scenes::aggregate(acc));
      printRow(a.c_str(), s.c_str(), rows.back());
    }
    printRow(a.c_str(), "MEAN", scenes::aggregate(rows));
    std::printf("%-15s time %.2f ms per 112x112 scene\n", a.c_str(), ms / (seeds * scenes::sceneNames().size()));
  }
  return 0;
}

bool readF32(const char* path, std::vector<float>& v, size_t n) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  v.resize(n);
  const bool ok = fread(v.data(), sizeof(float), n, f) == n;
  fclose(f);
  return ok;
}

int realMode(int argc, char** argv) {
  const int w = std::atoi(argv[5]), h = std::atoi(argv[6]);
  const int fh = argc > 7 ? std::atoi(argv[7]) : 0;
  TracerParams p;
  if (argc > 8 && !config(argv[8], p)) return 2;
  std::vector<float> rgb, base, region;
  if (!readF32(argv[2], rgb, (size_t)w * h * 3) || !readF32(argv[3], base, (size_t)w * h) ||
      !readF32(argv[4], region, (size_t)w * h)) {
    std::fprintf(stderr, "cannot read inputs\n");
    return 2;
  }
  TracerInput in;
  in.rgb = rgb.data();
  in.base = base.data();
  in.region = region.data();
  in.width = w;
  in.height = h;
  in.frameHeight = fh;
  const auto t0 = std::chrono::steady_clock::now();
  TracerResult r = traceStrands(in, p);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (int k = 0; k < 4; ++k) r = traceStrands(in, p);
  const auto t1 = std::chrono::steady_clock::now();
  for (int k = 0; k < 5; ++k) r = traceStrands(in, p);
  const double ms5 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 5;
  std::printf("shapes %d, min piece length %.1f px, time first %.2f ms, warm %.2f ms\n", r.shapes,
              minPieceLength(p, fh, h, 1.0), ms, ms5);
  const HairReference& R = r.reference;
  std::printf("frame reference: valid %d self %d d %.2f w %.2f v1 %.2f v2 %.2f samples %d\n", R.valid,
              R.selfReferenced, R.depth, R.width, R.chroma1, R.chroma2, R.samples);
  std::printf("pieces %zu joins %zu\n", r.pieces.size(), r.joins.size());
  for (size_t i = 0; i < r.pieces.size(); ++i) {
    const Piece& pc = r.pieces[i];
    int inside = 0;
    for (const Point& q : pc.polyline) inside += region[(size_t)std::lround(q.y) * w + std::lround(q.x)] > 0.5f;
    std::printf("  piece %zu: shape %d len %.1f px anchored %d conf %.2f FWHM %.1f inside %.2f edge %d self %d\n", i,
                pc.shape, pc.length, pc.anchored, pc.confidence, pc.width, (double)inside / pc.polyline.size(),
                pc.touchesRegionEdge, pc.selfReferenced);
  }
  for (const Join& j : r.joins) {
    const Point a = j.polyline.front(), b = j.polyline.back();
    std::printf("  join %s pieces %d-%d J %.2f chord %.1f px\n", j.toMatte ? "matte" : "piece", j.pieceA, j.pieceB,
                j.cost, std::hypot(b.x - a.x, b.y - a.y));
  }
  double sum = 0, outside = 0;
  int n5 = 0;
  for (size_t i = 0; i < r.alpha.size(); ++i) {
    sum += r.alpha[i];
    n5 += r.alpha[i] > 0.5f;
    if (region[i] < 0.5f) outside += r.alpha[i];
  }
  std::printf("alpha sum %.1f, px > 0.5 %d, alpha where region < 0.5 %.1f\n", sum, n5, outside);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && !std::strcmp(argv[1], "scenes"))
    return scenesMode(argv[2], std::atoi(argv[3]), split(argc > 4 ? argv[4] : "all,none"), true);
  if (argc >= 3 && !std::strcmp(argv[1], "cscenes"))
    return scenesMode("", std::atoi(argv[2]), split(argc > 3 ? argv[3] : "all,none"), false);
  if (argc >= 7 && !std::strcmp(argv[1], "real")) return realMode(argc, argv);
  std::fprintf(stderr,
               "usage: tracer_eval scenes <dir> <seeds> [cfg,...] | cscenes <seeds> [cfg,...] |\n"
               "       real <rgb.f32> <base.f32> <region.f32> <w> <h> [frameHeight] [cfg]\n");
  return 2;
}
