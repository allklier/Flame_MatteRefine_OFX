// mr_model_eval: the model candidate against plugin 2's Edge Refine with the REAL mr_ml_runner, on the thin-hair
// research's synthetic scenes (ofx/ml/research/vitmatte_thin_hair/exp/synth.py; written by eval_scenes.py gen).
// Not a test (needs the model and a GPU). Scores: eval_scenes.py score (the research's metrics).
//
// Scene file scene_<i>.bin: int32 C, then float32 rgb [C, C, 3] (display encoded, gamma 1/2.4 as Flame's Auto),
// base [C, C] (the solid mass), region [C, C] (the research's R = 12 focus region, 0 / 1).
// Output: <out>/scene_<i>/pred_<variant>.f32 [C, C].
// Usage: mr_model_eval <mr_ml_runner> <model dir> <cuda lib dir> <scene dir> <out dir> <model id> [variants...]
#include <signal.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "model_candidate.h"
#include "runner_crops.h"

using namespace mr;

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: mr_model_eval runner models cuda scenes out model [variants]\n");
    return 2;
  }
  signal(SIGPIPE, SIG_IGN);
  const std::string out = argv[5], model = argv[6], home = out + "/home";
  std::vector<std::string> variants;
  for (int i = 7; i < argc; ++i) variants.push_back(argv[i]);
  if (variants.empty())
    variants = {"er", "er_outer100", "half_raw", "imnet_raw", "imnet_unsharp", "full", "full_lcunsharp", "full_gain"};
  mkdir(out.c_str(), 0755);
  mkdir(home.c_str(), 0700);
  setenv("HOME", home.c_str(), 1);  // the client's log and lock files stay in the output folder
  mkdir((home + "/.config").c_str(), 0700);
  mkdir((home + "/.config/matte-refine").c_str(), 0700);
  std::ofstream(home + "/.config/matte-refine/config.conf")
      << "ml_model_dir=" << argv[2] << "\nml_cuda_libs=" << argv[3] << "\nml_ep=cuda\nlog_level=info\n";
  ml::RunnerClientOptions o;
  o.runnerPath = argv[1];
  mkdir((out + "/sock").c_str(), 0700);  // the runner wants a private socket folder
  o.socketPath = out + "/sock/r.sock";
  o.name = "wp4-eval";
  o.tileTimeoutS = 60;
  o.startTimeoutS = 30;
  o.idleSeconds = 60;
  o.env = {"HOME=" + home};
  ml::RunnerClient client(o);
  client.setModel(model);
  recipe::RunnerClientCrops crops(client);

  std::vector<std::string> files;
  if (DIR* d = opendir(argv[4])) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.compare(0, 6, "scene_") == 0 && n.size() > 4 && n.compare(n.size() - 4, 4, ".bin") == 0) files.push_back(n);
    }
    closedir(d);
  }
  std::printf("%zu scenes, model %s\n", files.size(), model.c_str());
  double ms = 0;
  int stretched = 0, runs = 0;
  for (const std::string& fn : files) {
    std::ifstream f(std::string(argv[4]) + "/" + fn, std::ios::binary);
    int32_t C = 0;
    f.read(reinterpret_cast<char*>(&C), 4);
    const size_t n = size_t(C) * C;
    std::vector<float> rgb(n * 3), base(n), region(n), a(n), conf(n);
    f.read(reinterpret_cast<char*>(rgb.data()), n * 12);
    f.read(reinterpret_cast<char*>(base.data()), n * 4);
    f.read(reinterpret_cast<char*>(region.data()), n * 4);
    if (!f) {
      std::fprintf(stderr, "%s: short file\n", fn.c_str());
      return 3;
    }
    const std::string sd = out + "/" + fn.substr(0, fn.size() - 4);
    mkdir(sd.c_str(), 0755);
    for (const std::string& v : variants) {
      const auto t0 = std::chrono::steady_clock::now();
      try {
        if (v == "er" || v == "er_outer100") {  // plugin 2's Edge Refine, its defaults (Band 10, Outer 0, Mix 1)
          ml::RefineParams rp;
          if (v == "er_outer100") rp.outerPx = 100;
          ml::refineEdge(base.data(), rgb.data(), a.data(), C, C, rp, client, nullptr);
        } else {
          recipe::ModelCandidateParams p;
          p.modelId = model;
          p.sizeScale = 1;  // the scenes are 1080-line crops
          if (v == "half_raw" || v == "imnet_raw" || v == "imnet_unsharp") {
            p.norm = v == "half_raw" ? recipe::Norm::Half : recipe::Norm::ImageNet;
            p.unsharpAmount = v == "imnet_unsharp" ? 1.0 : 0.0;
            p.lowContrast = recipe::LowContrast::Off;
            p.minPiecePx = 0;
          } else if (v == "full_lcunsharp") {
            p.unsharpWhenStretched = true;
          } else if (v == "full_gain") {
            p.strandBlack = 0.05, p.strandGain = 1.0 / 0.75;
          } else if (v != "full") {
            std::fprintf(stderr, "unknown variant %s\n", v.c_str());
            return 2;
          }
          recipe::ModelCandidateStats st;
          recipe::modelCandidate(rgb.data(), base.data(), region.data(), C, C, p, crops, a.data(), conf.data(), &st);
          if (v == "full") stretched += st.stretched;
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "%s %s: %s\n", fn.c_str(), v.c_str(), e.what());
        return 4;
      }
      ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ++runs;
      std::ofstream(sd + "/pred_" + v + ".f32", std::ios::binary).write(reinterpret_cast<const char*>(a.data()), n * 4);
    }
  }
  std::printf("%d runs, %.1f ms mean (incl. the first, cold); low-contrast switch on in %d of %zu scenes\n", runs,
              runs ? ms / runs : 0.0, stretched, files.size());
  client.quit(nullptr);
  return 0;
}
