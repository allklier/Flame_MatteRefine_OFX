// mr_ml_bench (ofx/ml/CLAUDE.md, phase 0): times an ONNX model on a matte's edge tiles with ONNX
// Runtime, cold and warm, and scores it against the synthetic scenes' truth next to plugin 1's
// Snap to Image. No plugin code: a standalone tool for the inline-vs-worker decision (28).
//
//   mr_ml_bench --model m.onnx --task trimap|segment [--ep cuda|cpu|coreml] [--tile 512]
//               [--overlap 32] [--band px@1080] [--repeat 5] [--json file] [--dump dir]
//               (--scene name [--size WxH] | --matte file.f32 --size WxH) | --info
//   mr_ml_bench --selfcheck      (trimap tiling covers every unknown pixel; no model needed)
//
// trimap (B1, ViTMatte): trimap from the matte's signed distance to its 0.5 contour (core =
//   deeper than band px, clear = beyond band px, unknown between); tiles of tile x tile (stride
//   tile - overlap, clamped inside the frame) that contain unknown pixels go through the model.
//   Input [1,4,T,T]: RGB (display-referred, clamped 0-1) normalised (x - 0.5) / 0.5,
//   trimap 0 / 0.5 / 1 not normalised (transformers VitMatteImageProcessor: rescale + normalise
//   the image, rescale only the trimap, concatenate). T a multiple of 32 (size_divisibility).
//   Tiles overlap: results are averaged with a linear ramp over `overlap` px at inner tile sides.
//   Blend back: w = 1 where |sd| <= band / 2, falling linearly to 0 at |sd| = band; out = in +
//   w (ml - in); beyond the band out = in, bit-identical.
// segment (B3, BiRefNet): the whole picture bilinear-resized to the model's size (1024 x 1024),
//   ImageNet mean / std on the clamped (display-referred) picture; output logits -> sigmoid -> bilinear back
//   to w x h (same size as the input).
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "ops/edge_field.h"
#include "ops/snap_to_image.h"
#include "synth.h"

namespace {

using mr::ml::Tile;
using mr::ml::selectTiles;
using mr::ml::tileWeight;
std::vector<float> makeTrimap(const std::vector<float>& sd, double band) { return mr::ml::trimapFromDistance(sd, band); }
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Args {
  std::string model, task = "trimap", ep = "cuda", scene, matteFile, json, dump;
  int w = 1920, h = 1080, tile = 512, overlap = 32, repeat = 5;
  double band = 10.0;  // px at 1080 lines, scaled with h
  bool info = false, selfcheck = false;
};

// --- frame input ------------------------------------------------------------------------------
struct Frame {
  std::string name;
  int w = 0, h = 0;
  std::vector<float> matte, rgb, truth;  // truth empty for real mattes
};

Frame loadFrame(const Args& a) {
  Frame f;
  f.w = a.w;
  f.h = a.h;
  if (!a.matteFile.empty()) {
    f.name = "file";
    f.matte.resize(size_t(a.w) * a.h);
    std::ifstream in(a.matteFile, std::ios::binary);
    in.read(reinterpret_cast<char*>(f.matte.data()), std::streamsize(f.matte.size() * 4));
    if (!in) throw std::runtime_error("cannot read " + std::to_string(f.matte.size() * 4) + " bytes");
    for (float& v : f.matte) v = std::min(std::max(v, 0.0f), 1.0f);
    // A synthetic grey / noise picture: only timing matters for real mattes.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0f, 0.1f);
    f.rgb.resize(f.matte.size() * 3);
    for (size_t i = 0; i < f.matte.size(); ++i)
      for (int c = 0; c < 3; ++c) f.rgb[i * 3 + c] = 0.18f + 0.5f * f.matte[i] * (0.5f + c * 0.2f) + u(rng);
  } else {
    synth::Scene s = synth::make(a.scene, a.w, a.h);
    f.name = a.scene;
    f.matte = std::move(s.matte);
    f.rgb = std::move(s.rgb);
    f.truth = std::move(s.truth);
  }
  return f;
}

// --- ONNX Runtime -----------------------------------------------------------------------------
struct Model {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mr_ml_bench"};
  Ort::Session session{nullptr};
  std::vector<std::string> inNames, outNames;
  std::vector<std::vector<int64_t>> inShapes, outShapes;
  std::vector<ONNXTensorElementDataType> inTypes, outTypes;
};

void createSession(Model& m, const Args& a) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (a.ep == "cuda") {
    OrtCUDAProviderOptions cuda;  // onnxruntime_c_api.h: device 0, default arena
    so.AppendExecutionProvider_CUDA(cuda);
  } else if (a.ep == "coreml") {
#ifdef __APPLE__
    so.AppendExecutionProvider("CoreML", {});  // onnxruntime_c_api.h: "CoreMLExecutionProvider (or CoreML)"
#else
    throw std::runtime_error("--ep coreml exists only in the macOS build");
#endif
  } else if (a.ep != "cpu") {
    throw std::runtime_error("unknown --ep " + a.ep);
  }
  m.session = Ort::Session(m.env, a.model.c_str(), so);
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < m.session.GetInputCount(); ++i) {
    m.inNames.push_back(m.session.GetInputNameAllocated(i, alloc).get());
    Ort::TypeInfo type = m.session.GetInputTypeInfo(i);  // keep alive: ti points into it
    auto ti = type.GetTensorTypeAndShapeInfo();
    m.inShapes.push_back(ti.GetShape());
    m.inTypes.push_back(ti.GetElementType());
  }
  for (size_t i = 0; i < m.session.GetOutputCount(); ++i) {
    m.outNames.push_back(m.session.GetOutputNameAllocated(i, alloc).get());
    Ort::TypeInfo type = m.session.GetOutputTypeInfo(i);  // keep alive: ti points into it
    auto ti = type.GetTensorTypeAndShapeInfo();
    m.outShapes.push_back(ti.GetShape());
    m.outTypes.push_back(ti.GetElementType());
  }
}

std::string shapeStr(const std::vector<int64_t>& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
  return r + "]";
}
const char* typeStr(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? "float32"
         : t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "float16" : "other";
}

// Runs the model on one float input (converted to fp16 if the model wants it); returns output 0
// as floats.
std::vector<float> run(Model& m, std::vector<float>& input, const std::vector<int64_t>& shape) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Float16_t> half;
  Ort::Value in{nullptr};
  if (m.inTypes[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    half.reserve(input.size());
    for (float v : input) half.push_back(Ort::Float16_t(v));
    in = Ort::Value::CreateTensor<Ort::Float16_t>(mem, half.data(), half.size(), shape.data(), shape.size());
  } else {
    in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());
  }
  const char* inName = m.inNames[0].c_str();
  const char* outName = m.outNames[0].c_str();
  auto outs = m.session.Run(Ort::RunOptions{nullptr}, &inName, &in, 1, &outName, 1);
  auto info = outs[0].GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  std::vector<float> r(n);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const Ort::Float16_t* p = outs[0].GetTensorData<Ort::Float16_t>();
    for (size_t i = 0; i < n; ++i) r[i] = p[i].ToFloat();
  } else {
    std::memcpy(r.data(), outs[0].GetTensorData<float>(), n * 4);
  }
  return r;
}

// --- memory -----------------------------------------------------------------------------------
long gpuUsedMiB() {
  FILE* p = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
  if (!p) return -1;
  long v = -1;
  if (std::fscanf(p, "%ld", &v) != 1) v = -1;
  pclose(p);
  return v;
}
long procStatusKiB(const char* key) {  // Linux /proc/self/status; -1 elsewhere
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.compare(0, std::strlen(key), key) == 0) return std::atol(line.c_str() + std::strlen(key) + 1);
  return -1;
}

// --- processing -------------------------------------------------------------------------------
struct StageMs { double pre = 0, run = 0, post = 0; };

// B1: returns the model's band result for the frame (only valid where the weight sum > 0), via
// the shared tiling in edge_refine (one implementation with the plugin).
std::vector<float> refineTiles(Model& m, const Frame& f, const std::vector<float>& tri,
                               const std::vector<Tile>& tiles, const Args& a,
                               std::vector<StageMs>* perTile) {
  const int T = a.tile, w = f.w, h = f.h;
  std::vector<float> acc(size_t(w) * h, 0.0f), wsum(size_t(w) * h, 0.0f), input(size_t(4) * T * T);
  const std::vector<int64_t> shape{1, 4, T, T};
  for (const Tile& t : tiles) {
    StageMs st;
    auto t0 = Clock::now();
    mr::ml::prepareTile(f.rgb.data(), tri.data(), w, h, t, T, input.data());
    st.pre = msSince(t0);
    t0 = Clock::now();
    std::vector<float> alpha = run(m, input, shape);
    st.run = msSince(t0);
    t0 = Clock::now();
    mr::ml::accumulateTile(alpha.data(), t, T, a.overlap, w, h, acc.data(), wsum.data());
    st.post = msSince(t0);
    if (perTile) perTile->push_back(st);
  }
  for (size_t i = 0; i < acc.size(); ++i) acc[i] = wsum[i] > 0 ? acc[i] / wsum[i] : -1.0f;
  return acc;
}

std::vector<float> blendBack(const std::vector<float>& in, const std::vector<float>& ml,
                             const std::vector<float>& sd, double band) {
  std::vector<float> out(in.size());
  mr::ml::blendBack(in.data(), ml.data(), sd.data(), in.size(), band, 1.0, false, out.data());
  return out;
}

void resizeBilinear(const float* src, int sw, int sh, int ch, float* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y)
    for (int x = 0; x < dw; ++x) {
      const double sx = (x + 0.5) * sw / dw - 0.5, sy = (y + 0.5) * sh / dh - 0.5;
      const int x0 = std::min(std::max(int(std::floor(sx)), 0), sw - 1), y0 = std::min(std::max(int(std::floor(sy)), 0), sh - 1);
      const int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
      const double fx = std::min(std::max(sx - x0, 0.0), 1.0), fy = std::min(std::max(sy - y0, 0.0), 1.0);
      for (int c = 0; c < ch; ++c) {
        auto at = [&](int xx, int yy) { return src[(size_t(yy) * sw + xx) * ch + c]; };
        dst[(size_t(y) * dw + x) * ch + c] = float((at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) +
                                                   (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy);
      }
    }
}

// B3: whole-frame segmentation at the model's input size, back to w x h.
std::vector<float> segment(Model& m, const Frame& f, int S, StageMs* st) {
  auto t0 = Clock::now();
  std::vector<float> disp(f.rgb.size()), small(size_t(S) * S * 3), input(size_t(3) * S * S);
  for (size_t i = 0; i < disp.size(); ++i) disp[i] = mr::ml::clampPicture(f.rgb[i]);
  resizeBilinear(disp.data(), f.w, f.h, 3, small.data(), S, S);
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sdv[3] = {0.229f, 0.224f, 0.225f};
  for (size_t i = 0; i < size_t(S) * S; ++i)
    for (int c = 0; c < 3; ++c) input[c * size_t(S) * S + i] = (small[i * 3 + c] - mean[c]) / sdv[c];
  st->pre = msSince(t0);
  t0 = Clock::now();
  std::vector<float> logits = run(m, input, {1, 3, S, S});
  st->run = msSince(t0);
  t0 = Clock::now();
  for (float& v : logits) v = 1.0f / (1.0f + std::exp(-v));
  std::vector<float> out(size_t(f.w) * f.h);
  resizeBilinear(logits.data(), S, S, 1, out.data(), f.w, f.h);
  st->post = msSince(t0);
  return out;
}

// Background (< 0.5) components, 4-connected, not touching the frame edge.
int countHoles(const std::vector<float>& m, int w, int h) {
  std::vector<int> lab(m.size(), 0), stack;
  int holes = 0;
  for (size_t s = 0; s < m.size(); ++s) {
    if (m[s] >= 0.5f || lab[s]) continue;
    bool edge = false;
    lab[s] = 1;
    stack.assign(1, int(s));
    while (!stack.empty()) {
      const int i = stack.back(), x = i % w, y = i / w;
      stack.pop_back();
      if (x == 0 || y == 0 || x == w - 1 || y == h - 1) edge = true;
      const int nb[4] = {x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1};
      for (int n : nb)
        if (n >= 0 && !lab[n] && m[n] < 0.5f) { lab[n] = 1; stack.push_back(n); }
    }
    if (!edge) ++holes;
  }
  return holes;
}

struct Stats { double median = 0, min = 0, p95 = 0; };
Stats stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.median = v[v.size() / 2];
  s.p95 = v[std::min(v.size() - 1, size_t(std::ceil(0.95 * v.size())) - 1)];
  return s;
}

void bandError(const std::vector<float>& a, const std::vector<float>& truth, const std::vector<float>& sd,
               double band, double* mean, double* mx) {
  double sum = 0;
  long n = 0;
  *mx = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(sd[i]) <= band) {
      const double e = std::fabs(a[i] - truth[i]);
      sum += e;
      *mx = std::max(*mx, e);
      ++n;
    }
  *mean = n ? sum / n : 0;
}

void writeF32(const std::string& path, const std::vector<float>& v) {
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}

// Every unknown pixel lies in at least one selected tile (with a weight > 0), at several sizes.
int selfcheck() {
  int fails = 0;
  for (int T : {256, 512, 1024})
    for (const char* scene : {"coarse", "all"}) {
      synth::Scene s = synth::make(scene, 1920, 1080);
      const double band = 10;
      auto sd = mr::signedEdgeDistance(s.matte.data(), 1920, 1080, 0.5f, band + 2);
      auto tri = makeTrimap(sd, band);
      auto tiles = selectTiles(tri, 1920, 1080, T, 32);
      std::vector<float> wsum(tri.size(), 0.0f);
      for (const Tile& t : tiles)
        for (int ly = 0; ly < T && t.y + ly < 1080; ++ly)
          for (int lx = 0; lx < T && t.x + lx < 1920; ++lx)
            wsum[size_t(t.y + ly) * 1920 + t.x + lx] += tileWeight(t, T, 32, 1920, 1080, lx, ly);
      long missed = 0, unknown = 0;
      for (size_t i = 0; i < tri.size(); ++i)
        if (tri[i] == 0.5f) { ++unknown; if (!(wsum[i] > 0)) ++missed; }
      std::printf("selfcheck %s tile %d: %zu tiles, %ld unknown px, %ld uncovered\n", scene, T, tiles.size(), unknown, missed);
      if (missed || !unknown) ++fails;
    }
  std::printf(fails ? "SELFCHECK FAILED\n" : "selfcheck ok\n");
  return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--model") a.model = val();
    else if (k == "--task") a.task = val();
    else if (k == "--ep") a.ep = val();
    else if (k == "--scene") a.scene = val();
    else if (k == "--matte") a.matteFile = val();
    else if (k == "--size") { const std::string s = val(); if (std::sscanf(s.c_str(), "%dx%d", &a.w, &a.h) != 2) throw std::runtime_error("--size WxH"); }
    else if (k == "--tile") a.tile = std::stoi(val());
    else if (k == "--overlap") a.overlap = std::stoi(val());
    else if (k == "--band") a.band = std::stod(val());
    else if (k == "--repeat") a.repeat = std::max(1, std::stoi(val()));
    else if (k == "--json") a.json = val();
    else if (k == "--dump") a.dump = val();
    else if (k == "--info") a.info = true;
    else if (k == "--selfcheck") a.selfcheck = true;
    else { std::fprintf(stderr, "unknown option %s (see the header of mr_ml_bench.cpp)\n", k.c_str()); return 2; }
  }
  if (a.selfcheck) return selfcheck();
  if (a.model.empty()) { std::fprintf(stderr, "--model required\n"); return 2; }
  if (a.task == "trimap" && a.tile % 32) { std::fprintf(stderr, "--tile must be a multiple of 32\n"); return 2; }

  try {
    Model m;
    const long gpu0 = gpuUsedMiB(), rss0 = procStatusKiB("VmRSS:");
    auto t0 = Clock::now();
    createSession(m, a);
    const double loadMs = msSince(t0);
    const long gpu1 = gpuUsedMiB(), rss1 = procStatusKiB("VmRSS:");
    if (a.info) {
      for (size_t i = 0; i < m.inNames.size(); ++i)
        std::printf("input  %s %s %s\n", m.inNames[i].c_str(), shapeStr(m.inShapes[i]).c_str(), typeStr(m.inTypes[i]));
      for (size_t i = 0; i < m.outNames.size(); ++i)
        std::printf("output %s %s %s\n", m.outNames[i].c_str(), shapeStr(m.outShapes[i]).c_str(), typeStr(m.outTypes[i]));
      return 0;
    }
    if (a.scene.empty() && a.matteFile.empty()) a.scene = "coarse";
    Frame f = loadFrame(a);
    const double band = a.band * f.h / 1080.0;
    const size_t N = size_t(f.w) * f.h;

    // Trimap and tiles (B1) or the model's fixed size (B3).
    auto tp = Clock::now();
    std::vector<float> sd = mr::signedEdgeDistance(f.matte.data(), f.w, f.h, 0.5f, band + 2);
    std::vector<float> tri = makeTrimap(sd, band);
    std::vector<Tile> tiles;
    long unknown = 0;
    for (float v : tri) unknown += v == 0.5f;
    int S = 1024;
    if (a.task == "trimap") tiles = selectTiles(tri, f.w, f.h, a.tile, a.overlap);
    else if (m.inShapes[0].size() == 4 && m.inShapes[0][2] > 0) S = int(m.inShapes[0][2]);
    const double trimapMs = msSince(tp);

    // Cold: the first frame (includes CUDA / cuDNN warm-up); then warm repeats.
    std::vector<double> frameMs, runMs, preMs, postMs;
    double firstInferMs = 0, firstFrameMs = 0;
    std::vector<float> result;
    for (int r = 0; r <= a.repeat; ++r) {
      std::vector<StageMs> st;
      auto tf = Clock::now();
      if (a.task == "trimap") {
        result = refineTiles(m, f, tri, tiles, a, &st);
      } else {
        StageMs s;
        result = segment(m, f, S, &s);
        st.push_back(s);
      }
      const double ms = msSince(tf);
      if (r == 0) {
        firstFrameMs = ms;
        firstInferMs = st.empty() ? 0 : st[0].run;
        continue;
      }
      frameMs.push_back(ms);
      for (const StageMs& s : st) { runMs.push_back(s.run); preMs.push_back(s.pre); postMs.push_back(s.post); }
    }
    const long gpu2 = gpuUsedMiB(), rss2 = procStatusKiB("VmRSS:"), hwm = procStatusKiB("VmHWM:");

    // Quality.
    std::map<std::string, double> q;
    std::vector<float> blended, baseline(N);
    if (a.task == "trimap") {
      blended = blendBack(f.matte, result, sd, band);
      if (!f.truth.empty()) {
        mr::SnapParams sp;  // plugin 1 defaults
        mr::snapToImage(f.matte.data(), f.rgb.data(), baseline.data(), f.w, f.h, sp);
        double mean, mx;
        bandError(f.matte, f.truth, sd, band, &mean, &mx); q["in_mean"] = mean; q["in_max"] = mx;
        bandError(blended, f.truth, sd, band, &mean, &mx); q["ml_mean"] = mean; q["ml_max"] = mx;
        bandError(baseline, f.truth, sd, band, &mean, &mx); q["snap_mean"] = mean; q["snap_max"] = mx;
      }
      long changedOutside = 0;
      for (size_t i = 0; i < N; ++i) changedOutside += std::fabs(sd[i]) >= band && blended[i] != f.matte[i];
      q["changed_outside_band"] = double(changedOutside);
    } else if (!f.truth.empty()) {
      long inter = 0, uni = 0;
      for (size_t i = 0; i < N; ++i) {
        const bool p = result[i] >= 0.5f, t = f.truth[i] >= 0.5f;
        inter += p && t; uni += p || t;
      }
      q["iou"] = uni ? double(inter) / uni : 0;
      q["holes_model"] = countHoles(result, f.w, f.h);
      q["holes_input"] = countHoles(f.matte, f.w, f.h);
      q["holes_truth"] = countHoles(f.truth, f.w, f.h);
    }
    if (!a.dump.empty()) {
      const std::string p = a.dump + "/" + f.name + "_" + std::to_string(f.w) + "x" + std::to_string(f.h);
      writeF32(p + "_in.f32", f.matte);
      writeF32(p + "_ml.f32", a.task == "trimap" ? blended : result);
      if (!f.truth.empty()) writeF32(p + "_truth.f32", f.truth);
      if (a.task == "trimap" && !f.truth.empty()) writeF32(p + "_snap.f32", baseline);
      writeF32(p + "_trimap.f32", tri);
    }

    const Stats fs = stats(frameMs), rs = stats(runMs), ps = stats(preMs), os = stats(postMs);
    // "<folder>/<file>" (e.g. birefnet-lite/model_fp16.onnx): the file names repeat across models.
    const size_t slash = a.model.find_last_of('/');
    const size_t slash2 = slash == std::string::npos || slash == 0 ? std::string::npos : a.model.find_last_of('/', slash - 1);
    const std::string modelName = a.model.substr(slash2 == std::string::npos ? 0 : slash2 + 1);
    std::printf("%s %s ep=%s scene=%s %dx%d tile=%d: %zu tiles, band %.1f px = %.2f %% of frame; load %.0f ms, "
                "first frame %.0f ms (first inference %.0f ms); warm frame median %.1f / min %.1f / p95 %.1f ms; "
                "per tile run %.2f, pre %.2f, post %.2f ms; trimap %.1f ms; GPU MiB %ld -> %ld -> %ld; RSS MiB %ld -> %ld -> %ld (peak %ld)\n",
                modelName.c_str(), a.task.c_str(), a.ep.c_str(), f.name.c_str(), f.w, f.h, a.task == "trimap" ? a.tile : S,
                a.task == "trimap" ? tiles.size() : size_t(1), band, 100.0 * unknown / N, loadMs, firstFrameMs, firstInferMs,
                fs.median, fs.min, fs.p95, rs.median, ps.median, os.median, trimapMs, gpu0, gpu1, gpu2, rss0 / 1024, rss1 / 1024,
                rss2 / 1024, hwm / 1024);
    for (const auto& kv : q) std::printf("  %s = %.5g\n", kv.first.c_str(), kv.second);
    if (!a.json.empty()) {
      FILE* j = std::fopen(a.json.c_str(), "a");
      if (!j) throw std::runtime_error("cannot open " + a.json);
      std::fprintf(j, "{\"model\":\"%s\",\"task\":\"%s\",\"ep\":\"%s\",\"scene\":\"%s\",\"w\":%d,\"h\":%d,\"tile\":%d,\"tiles\":%zu,"
                      "\"band_px\":%.2f,\"band_frac\":%.5f,\"load_ms\":%.1f,\"first_frame_ms\":%.1f,\"first_infer_ms\":%.1f,"
                      "\"frame_ms_median\":%.2f,\"frame_ms_min\":%.2f,\"frame_ms_p95\":%.2f,\"tile_run_ms_median\":%.3f,"
                      "\"tile_run_ms_min\":%.3f,\"tile_run_ms_p95\":%.3f,\"tile_pre_ms\":%.3f,\"tile_post_ms\":%.3f,\"trimap_ms\":%.2f,"
                      "\"repeat\":%d,\"gpu_mib\":[%ld,%ld,%ld],\"rss_mib\":[%ld,%ld,%ld],\"rss_peak_mib\":%ld,"
                      "\"ort\":\"%s\"",
                   modelName.c_str(), a.task.c_str(), a.ep.c_str(), f.name.c_str(), f.w, f.h, a.task == "trimap" ? a.tile : S,
                   a.task == "trimap" ? tiles.size() : size_t(1), band, double(unknown) / N, loadMs, firstFrameMs, firstInferMs,
                   fs.median, fs.min, fs.p95, rs.median, rs.min, rs.p95, ps.median, os.median, trimapMs, a.repeat, gpu0, gpu1, gpu2,
                   rss0 / 1024, rss1 / 1024, rss2 / 1024, hwm / 1024, Ort::GetVersionString().c_str());
      for (const auto& kv : q) std::fprintf(j, ",\"%s\":%.6g", kv.first.c_str(), kv.second);
      std::fprintf(j, "}\n");
      std::fclose(j);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
