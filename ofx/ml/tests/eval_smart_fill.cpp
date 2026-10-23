// Smart Fill (B3) evaluation on the synthetic "gaps" scene (synth::makeGaps: label 2 = defect hole,
// fill; label 3 = real gap, keep), seeds 1..20 at HD and 480x270. Segmentation sources: a perfect
// fake (the truth blurred), BiRefNet-like fakes that close real gaps narrower than N px (the known
// failure, PHASE0_RESULTS.md), the real BiRefNet-lite (and BiRefNet, if present) in-process
// through ort_runner; the colour score (no model); plugin 1's Fill Holes (Max Hole %) as the
// baseline. Prints per-hole accuracy and pixel fill rates, a threshold sweep, per-gap scores, and
// UHD timings of the core.
// Usage: eval_smart_fill [--quick] [--real-only] <model dir> [<cuda lib dir>]
//   --quick: 480x270 only, seeds 1..3. --real-only: only the real model rows; exit 77 without
//   <model dir>/birefnet-lite/model.onnx (the ctest ml_smart_fill_real).
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops/fill_holes.h"
#include "ort_runner.h"
#include "smart_fill.h"
#include "smart_fill_ref.h"
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

double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct MaskRunner : SegRunner {
  const std::vector<float>* mask = nullptr;
  void segment(const float*, int, int, float* m) override { std::copy(mask->begin(), mask->end(), m); }
};

struct OrtSeg : SegRunner {
  std::shared_ptr<ModelSession> s;
  float logitMin = 1e30f, logitMax = -1e30f;
  double prepMs = 0, runMs = 0, postMs = 0;
  void segment(const float* rgb, int w, int h, float* mask) override {
    const int S = kBiRefNetSize;
    std::vector<float> input(size_t(3) * S * S), logits(size_t(S) * S);
    auto t0 = std::chrono::steady_clock::now();
    birefnetPrepare(rgb, w, h, input.data());
    auto t1 = std::chrono::steady_clock::now();
    s->run(input.data(), 3, S, S, logits.data(), 1, S, S);
    auto t2 = std::chrono::steady_clock::now();
    logitMin = 1e30f, logitMax = -1e30f;  // this frame's
    for (float v : logits) logitMin = std::min(logitMin, v), logitMax = std::max(logitMax, v);
    birefnetPostprocess(logits.data(), w, h, mask);
    auto t3 = std::chrono::steady_clock::now();
    prepMs = ms(t0, t1), runMs = ms(t1, t2), postMs = ms(t2, t3);
  }
};

// One hole of the truth: is it a real gap; its label-2 / label-3 pixels.
struct TruthHole {
  bool gap;
  long long area, l2, l3;
};
// Per hole, the scores smartFill computed.
struct Scored {
  float mean, median, min, colour;
};

struct Acc {
  long long subj = 0, subjFilled = 0, gaps = 0, gapsKept = 0, l2 = 0, l2Filled = 0, l3 = 0, l3Filled = 0;
  int frames = 0;
  void add(const std::vector<TruthHole>& th, const std::vector<bool>& fill, long long l2All, long long l3All) {
    ++frames;
    l2 += l2All, l3 += l3All;
    for (size_t k = 0; k < th.size(); ++k) {
      if (th[k].gap) {
        ++gaps;
        gapsKept += !fill[k];
      } else {
        ++subj;
        subjFilled += fill[k];
      }
      if (fill[k]) l2Filled += th[k].l2, l3Filled += th[k].l3;
    }
  }
  double holeAcc() const { return double(subjFilled + gapsKept) / std::max(1LL, subj + gaps); }
};

std::map<std::string, Acc> gAcc;
std::vector<std::string> gOrder;
Acc& acc(const std::string& size, const std::string& name) {
  const std::string k = size + "|" + name;
  if (!gAcc.count(k)) gOrder.push_back(k);
  return gAcc[k];
}

double iou(const std::vector<float>& a, const std::vector<float>& b) {
  long long i = 0, u = 0;
  for (size_t k = 0; k < a.size(); ++k) {
    const bool x = a[k] >= 0.5f, y = b[k] >= 0.5f;
    i += x && y, u += x || y;
  }
  return u ? double(i) / double(u) : 1.0;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false, realOnly = false;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--quick") quick = true;
    else if (a == "--real-only") realOnly = true;
    else pos.push_back(a);
  }
  const std::string models = pos.size() > 0 ? pos[0] : "", cuda = pos.size() > 1 ? pos[1] : "";

  // The real models, if present.
  struct Real {
    std::string id;
    OrtSeg seg;
  };
  std::vector<std::unique_ptr<Real>> reals;
  for (const char* id : {"birefnet-lite", "birefnet"}) {
    if (quick && std::string(id) == "birefnet") continue;
    const std::string path = models + "/" + id + "/model.onnx";
    if (models.empty() || access(path.c_str(), R_OK) != 0) continue;
    RunnerConfig cfg;
    cfg.modelPath = path;
    cfg.modelId = id;
    cfg.ep = "cuda";
    cfg.cudaLibDir = cuda;
    RunnerInfo info;
    std::string err;
    std::unique_ptr<Real> r(new Real);
    r->id = id;
    r->seg.s = createModelSession(cfg, &info, &err);
    if (!r->seg.s) {
      std::printf("%s: no session: %s\n", id, err.c_str());
      ++failures;
      continue;
    }
    std::printf("%s: session ep %s%s%s, load %.0f ms\n", id, info.ep.c_str(), info.why.empty() ? "" : ", ",
                info.why.c_str(), info.loadMs);
    reals.push_back(std::move(r));
  }
  if (realOnly && reals.empty()) {
    std::printf("no birefnet-lite model in '%s': skipped\n", models.c_str());
    return 77;
  }

  struct Size {
    int w, h;
    std::string name;
  };
  std::vector<Size> sizes = quick ? std::vector<Size>{{480, 270, "480x270"}}
                                  : std::vector<Size>{{1920, 1080, "HD"}, {480, 270, "480x270"}};
  const unsigned seeds = quick ? 3 : 20;
  const float thr = 0.95f;
  const double sweep[] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  for (const Size& sz : sizes) {
    const double k = sz.h / 1080.0;
    for (unsigned seed = 1; seed <= seeds; ++seed)
     for (double mixFg : {0.0, 0.8, 1.0}) {
      if (quick && mixFg > 0) continue;
      synth::Scene s = synth::makeGaps(sz.w, sz.h, seed);
      // Harder pictures: rgb' = (1 - a) rgb + a fg = the background replaced by a blend of the
      // subject's colour (a) and its own (1 - a): a = 1 shows the subject colour everywhere.
      std::string sizeName = sz.name;
      if (mixFg > 0) {
        for (size_t i = 0; i < s.rgb.size(); ++i) s.rgb[i] = float((1 - mixFg) * s.rgb[i] + mixFg * s.fg[i]);
        sizeName += mixFg < 1 ? " lowc" : " nocol";
      }
      const size_t n = s.matte.size();
      long long l2All = 0, l3All = 0;
      for (uint8_t l : s.label) l2All += l == synth::kDefectHole, l3All += l == synth::kRealGap;
      std::vector<TruthHole> th;
      for (const sfref::RefHole& r : sfref::findHoles(s.matte.data(), sz.w, sz.h, thr)) {
        TruthHole t{false, r.area, 0, 0};
        for (int i : r.px) t.l2 += s.label[size_t(i)] == synth::kDefectHole, t.l3 += s.label[size_t(i)] == synth::kRealGap;
        t.gap = t.l3 > 0;
        th.push_back(t);
      }
      std::vector<bool> fill(th.size());

      // Baseline: plugin 1's Fill Holes (Solid Above 0.95, Max Hole % 1.0 = its default, and 0.2).
      if (!realOnly && mixFg == 0)
        for (double pct : {1.0, 0.2}) {
          mr::FillHolesParams fp;
          fp.threshold = thr;
          fp.maxHolePercent = pct;
          std::vector<float> o(n);
          mr::fillHoles(s.matte.data(), o.data(), sz.w, sz.h, fp);
          for (size_t q = 0; q < th.size(); ++q) fill[q] = double(th[q].area) <= pct / 100.0 * double(n);
          acc(sizeName, "Fill Holes (plugin 1), Max Hole " + std::to_string(pct).substr(0, 3) + " %")
              .add(th, fill, l2All, l3All);
          // The truth classes agree with what plugin 1 actually filled (a check of this harness).
          long long l3Changed = 0, l3Expected = 0;
          for (size_t i = 0; i < n; ++i) l3Changed += s.label[i] == synth::kRealGap && o[i] != s.matte[i];
          for (size_t q = 0; q < th.size(); ++q) l3Expected += fill[q] ? th[q].l3 : 0;
          CHECK(l3Changed == l3Expected);
        }

      // Segmentation sources.
      struct Source {
        std::string name;
        std::vector<float> mask;
        OrtSeg* real;
      };
      std::vector<Source> src;
      if (!realOnly && mixFg == 0)
        src.push_back({"perfect (truth, blur 1 px)", sfref::fakeSegmentation(s.truth, sz.w, sz.h, 0, std::max(0.7, k)), nullptr});
      if (!realOnly)
        for (int N : {24, 64, 128}) {
          if (mixFg > 0 && N != 64) continue;
          src.push_back({"closes gaps < " + std::to_string(N) + " px@1080",
                         sfref::fakeSegmentation(s.truth, sz.w, sz.h, N / 2.0 * k, std::max(0.7, 1.5 * k)), nullptr});
        }
      for (auto& r : reals) src.push_back({r->id + " (real)", {}, &r->seg});

      std::vector<float> colourScore;
      for (Source& so : src) {
        MaskRunner fake;
        fake.mask = &so.mask;
        SegRunner& runner = so.real ? static_cast<SegRunner&>(*so.real) : fake;
        SmartFillParams p;
        p.solidAbove = thr;
        p.maxHoleFraction = 1.0;  // score every hole; the safety is not what is measured
        p.mode = ScoreMode::Both;
        std::vector<float> out(n);
        SmartFillStats st;
        smartFill(s.matte.data(), s.rgb.data(), sz.w, sz.h, p, runner, out.data(), &st);
        CHECK(st.holes.size() == th.size());
        if (st.holes.size() != th.size()) continue;
        for (size_t q = 0; q < th.size(); ++q) CHECK(st.holes[q].area == th[q].area);
        std::vector<Scored> sc;
        for (const HoleInfo& hi : st.holes) sc.push_back({hi.segMean, hi.segMedian, hi.segMin, hi.colour});
        if (colourScore.empty())
          for (const Scored& x : sc) colourScore.push_back(x.colour);
        if (so.real) {
          // Read the model's mask back (the runner only fills smartFill's copy).
          std::vector<float> mask(n);
          so.real->segment(s.rgb.data(), sz.w, sz.h, mask.data());
          if (seed == 1)
            std::printf("%s %s seed 1: IoU vs truth %.4f, raw output %.2f .. %.2f (logits: sigmoid needed), "
                        "prep %.1f run %.1f post %.1f ms\n",
                        sizeName.c_str(), so.name.c_str(), iou(mask, s.truth), so.real->logitMin, so.real->logitMax,
                        so.real->prepMs, so.real->runMs, so.real->postMs);
          if (mixFg == 0) CHECK(iou(mask, s.truth) > 0.9);
          if (mixFg == 0) CHECK(so.real->logitMin < -1.0f && so.real->logitMax > 1.0f);  // logits, not 0-1
        }
        if (seed == 1) {
          std::printf("  %s %s seed 1, real gaps (area, seg mean / median / min, colour):", sizeName.c_str(),
                      so.name.c_str());
          for (size_t q = 0; q < th.size(); ++q)
            if (th[q].gap)
              std::printf(" [%lld: %.2f/%.2f/%.2f, %.2f]", th[q].area, sc[q].mean, sc[q].median, sc[q].min, sc[q].colour);
          float dmin = 1, dmed = 1;
          for (size_t q = 0; q < th.size(); ++q)
            if (!th[q].gap) dmin = std::min(dmin, sc[q].mean), dmed = std::min(dmed, sc[q].min);
          std::printf("; lowest over subject holes: mean %.2f, min %.2f\n", dmin, dmed);
        }
        for (double t : sweep) {
          const bool main = t == 0.5;
          if (!main && !so.real && so.name.find("perfect") == std::string::npos &&
              so.name.find("64") == std::string::npos)
            continue;
          auto run = [&](const std::string& what, int which) {
            for (size_t q = 0; q < th.size(); ++q) {
              const float seg = which == 1 ? sc[q].median : which == 2 ? sc[q].min : sc[q].mean;
              const float v = which == 3 ? std::min(sc[q].mean, sc[q].colour) : which == 4 ? std::min(sc[q].min, sc[q].colour) : seg;
              fill[q] = double(v) >= t;
            }
            char tb[16];
            std::snprintf(tb, sizeof tb, " @%.1f", t);
            acc(sizeName, so.name + ": " + what + tb).add(th, fill, l2All, l3All);
          };
          run("seg mean", 0);
          if (main || so.real) run("seg median", 1), run("seg min", 2);
          run("both (mean, colour)", 3);
          if (main || so.real) run("both (min, colour)", 4);
        }
      }
      // Colour alone.
      if (!colourScore.empty())
        for (double t : sweep) {
          for (size_t q = 0; q < th.size(); ++q) fill[q] = double(colourScore[q]) >= t;
          char tb[16];
          std::snprintf(tb, sizeof tb, " @%.1f", t);
          acc(sizeName, std::string("colour (no model)") + tb).add(th, fill, l2All, l3All);
        }
    }
  }

  // Phase 0 found BiRefNet "closes real gaps" on the "all" scene: the truth's enclosed gaps (truth <
  // 0.5, not reaching the frame edge) and the model's mean inside each.
  if (!quick)
    for (auto& r : reals) {
      synth::Scene s = synth::make("all", 1920, 1080, 1);
      std::vector<float> mask(s.matte.size());
      r->seg.segment(s.rgb.data(), s.w, s.h, mask.data());
      std::printf("\"all\" HD, %s: IoU %.4f; truth gaps (area: model mean / min):", r->id.c_str(), iou(mask, s.truth));
      for (const sfref::RefHole& g : sfref::findHoles(s.truth.data(), s.w, s.h, 0.5f)) {
        double sum = 0;
        float mn = 1;
        for (int i : g.px) sum += mask[size_t(i)], mn = std::min(mn, mask[size_t(i)]);
        std::printf(" [%lld: %.2f / %.2f]", g.area, sum / double(g.area), mn);
      }
      const std::vector<sfref::RefHole> mh = sfref::findHoles(mask.data(), s.w, s.h, 0.5f);
      std::printf("; enclosed gaps in the model's mask: %zu\n", mh.size());
    }

  std::printf("\n%-14s %-52s %6s %9s %9s %8s %9s %9s\n", "size", "classifier", "frames", "subj fill", "gaps kept",
              "hole acc", "l2 filled", "l3 filled");
  for (const std::string& key : gOrder) {
    const Acc& a = gAcc[key];
    const size_t bar = key.find('|');
    std::printf("%-14s %-52s %6d %4lld/%-4lld %4lld/%-4lld %7.1f%% %8.2f%% %8.2f%%\n", key.substr(0, bar).c_str(),
                key.substr(bar + 1).c_str(), a.frames, a.subjFilled, a.subj, a.gapsKept, a.gaps, 100 * a.holeAcc(),
                100.0 * double(a.l2Filled) / double(std::max(1LL, a.l2)),
                100.0 * double(a.l3Filled) / double(std::max(1LL, a.l3)));
  }

  // Timing of the core (no model): HD and UHD, the perfect fake segmentation.
  if (!quick && !realOnly)
    for (const Size& sz : std::vector<Size>{{1920, 1080, "HD"}, {3840, 2160, "UHD"}}) {
      synth::Scene s = synth::makeGaps(sz.w, sz.h, 1);
      const std::vector<float> mask = sfref::fakeSegmentation(s.truth, sz.w, sz.h, 0, 1.0);
      MaskRunner fake;
      fake.mask = &mask;
      std::vector<float> out(s.matte.size());
      for (ScoreMode mode : {ScoreMode::Segmentation, ScoreMode::Colour, ScoreMode::Both}) {
        SmartFillParams p;
        p.mode = mode;
        SmartFillStats st, best;
        best.totalMs = 1e9;
        for (int r = 0; r < 5; ++r) {
          smartFill(s.matte.data(), s.rgb.data(), sz.w, sz.h, p, fake, out.data(), &st);
          if (st.totalMs < best.totalMs) best = st;
        }
        std::printf("%s core, mode %s: %.1f ms (label %.1f, mask copy %.1f, score %.1f, write %.1f), %d holes\n",
                    sz.name.c_str(), mode == ScoreMode::Segmentation ? "seg" : mode == ScoreMode::Colour ? "colour" : "both",
                    best.totalMs, best.labelMs, best.segMs, best.scoreMs, best.writeMs, best.holesFound);
      }
      const int S = kBiRefNetSize;
      std::vector<float> input(size_t(3) * S * S), logits(size_t(S) * S, 0.5f);
      double pre = 1e9, post = 1e9;
      for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        birefnetPrepare(s.rgb.data(), sz.w, sz.h, input.data());
        auto t1 = std::chrono::steady_clock::now();
        birefnetPostprocess(logits.data(), sz.w, sz.h, out.data());
        auto t2 = std::chrono::steady_clock::now();
        pre = std::min(pre, ms(t0, t1)), post = std::min(post, ms(t1, t2));
      }
      std::printf("%s BiRefNet prepare %.1f ms, postprocess %.1f ms\n", sz.name.c_str(), pre, post);
      for (auto& r : reals) {
        std::vector<float> m(s.matte.size());
        double best = 1e9;
        for (int q = 0; q < 3; ++q) {
          r->seg.segment(s.rgb.data(), sz.w, sz.h, m.data());
          best = std::min(best, r->seg.prepMs + r->seg.runMs + r->seg.postMs);
        }
        std::printf("%s %s segment (pre + model + post, in-process): %.1f ms (model %.1f)\n", sz.name.c_str(),
                    r->id.c_str(), best, r->seg.runMs);
      }
    }
  std::printf(failures ? "%d FAILURES\n" : "done\n", failures);
  return failures ? 1 : 0;
}
