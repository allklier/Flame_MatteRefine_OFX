// Model smoke test through the plugin's own wrapper (ort_runner) and core (edge_refine): ViTMatte
// small on the synthetic "coarse" scene must bring the band error clearly below the input's
// (phase 0: 0.221 -> ~0.03 at HD). CPU at a small size; CUDA at HD and UHD when a GPU is usable
// (else that part is skipped, not failed). Prints the per-stage times of warm frames for
// comparison with ofx/ml/PHASE0_RESULTS.md. Exit 77 (skipped) without the model.
// Usage: test_ml_model <model dir> [<cuda lib dir>]
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "ort_runner.h"
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

void print(const char* what, const RefineStats& s) {
  std::printf("  %s: %d tiles (T %d), band %.2f %%: trimap %.1f, prep %.1f, run %.1f, post %.1f, blend %.1f, "
              "total %.1f ms\n",
              what, s.tilesRun, s.tileSize, 100 * s.bandFraction, s.trimapMs, s.prepMs, s.runMs, s.postMs, s.blendMs,
              s.totalMs);
}

// Runs the scene `warm` + 1 times; checks the error; returns the last frame's stats.
void runScene(const char* label, TileRunner& r, int w, int h, int warm, double maxError) {
  synth::Scene s = synth::make("coarse", w, h);
  RefineParams p;
  p.bandPx = 10.0 * h / 1080;
  std::vector<float> out(s.matte.size()), sd, tri;
  buildTrimap(s.matte.data(), w, h, p.bandPx, &sd, &tri);
  RefineStats st;
  for (int i = 0; i <= warm; ++i) {
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, r, &st);
    print(i == 0 ? (std::string(label) + " first").c_str() : (std::string(label) + " warm").c_str(), st);
  }
  const double ein = bandError(s.matte, s.truth, sd, p.bandPx), eout = bandError(out, s.truth, sd, p.bandPx);
  long outside = 0;
  for (size_t i = 0; i < out.size(); ++i) outside += std::fabs(sd[i]) >= p.bandPx && out[i] != s.matte[i];
  std::printf("  %s %dx%d: band error %.4f -> %.4f, %ld px changed outside the band\n", label, w, h, ein, eout,
              outside);
  CHECK(eout < maxError && eout < ein / 3);
  CHECK(outside == 0);
}
}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "", cudaLibs = argc > 2 ? argv[2] : "";
  const std::string model = dir + "/vitmatte-small/model.onnx";
  if (dir.empty() || access(model.c_str(), R_OK) != 0) {
    std::printf("no model (%s): skipped\n", model.c_str());
    return 77;
  }
  std::printf("ONNX Runtime %s\n", ortVersion().c_str());
  RunnerInfo ri;
  std::string err;
  // CUDA first, as in Flame: the process's first session is a CUDA one.
  RunnerConfig cuda;
  cuda.modelPath = model;
  cuda.ep = "cuda";
  cuda.cudaLibDir = cudaLibs;
  RunnerInfo gi;
  auto g = acquireRunner(cuda, &gi, &err);
  std::printf("cuda session: %s, load %.0f ms (cuda libs %d in %.0f ms), ep %s%s%s, GPU MiB %ld -> %ld, RSS MiB %ld -> %ld\n",
              g ? "ok" : err.c_str(), gi.loadMs, gi.cudaLibsLoaded, gi.cudaLibsMs, gi.ep.c_str(),
              gi.why.empty() ? "" : " because: ", gi.why.c_str(), gi.gpuMiBBefore, gi.gpuMiBAfter, gi.rssMiBBefore,
              gi.rssMiBAfter);
  RunnerConfig cpu;
  cpu.modelPath = model;
  cpu.ep = "cpu";
  auto r = acquireRunner(cpu, &ri, &err);
  std::printf("cpu session: %s, load %.0f ms, ep %s %s\n", r ? "ok" : err.c_str(), ri.loadMs, ri.ep.c_str(),
              ri.why.c_str());
  CHECK(r && ri.created && ri.ep == "cpu");
  if (r) runScene("cpu", *r, 640, 360, 0, 0.08);
  // The same config again: the shared session, not a new one.
  auto r2 = acquireRunner(cpu, &ri, &err);
  CHECK(r2 == r && !ri.created);
  // A missing model: an error, no crash, not remembered.
  RunnerConfig missing = cpu;
  missing.modelPath = dir + "/nope/model.onnx";
  CHECK(!acquireRunner(missing, &ri, &err) && !err.empty());

  CHECK(g != nullptr);  // CUDA or the CPU fallback
  // A GPU (nvidia-smi answers) and the CUDA libraries given: the fallback would be a bug.
  if (gpuUsedMiB() >= 0 && !cudaLibs.empty()) CHECK(gi.ep == "cuda");
  if (g && gi.ep == "cuda") {
    runScene("cuda HD", *g, 1920, 1080, 3, 0.06);
    runScene("cuda UHD", *g, 3840, 2160, 2, 0.06);
    std::printf("  GPU MiB after: %ld, RSS MiB %ld\n", gpuUsedMiB(), rssMiB());
  } else {
    std::printf("cuda: not usable here (%s): skipped the GPU part\n", gi.why.c_str());
  }
  std::printf(failures ? "test_ml_model: %d FAILED\n" : "test_ml_model: all passed\n", failures);
  return failures ? 1 : 0;
}
