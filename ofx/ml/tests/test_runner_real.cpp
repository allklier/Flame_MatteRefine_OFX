// The real mr_ml_runner with ViTMatte-small (decision 32): started lazily by a client, the band
// error on the synthetic "coarse" scene as in phase 0 (0.221 -> ~0.03 at HD), two concurrent
// clients get identical results from one shared session (STATUS), the runner's library path is
// its own (ORT + ml_cuda_libs), the result equals the in-process path (CPU: bit-identical; CUDA:
// within kCudaJitter: the CUDA EP itself is not bit-deterministic, the same in-process session gives
// up to ~1.5e-3 between two runs of one frame on BFD, 2026-10-23), a client of another build replaces the runner once and
// then fails cleanly while the others recover, LRU eviction beyond two sessions, a model that cannot
// load is remembered (then retried), idle exit. Protocol 2: with MRML_BASELINE=<folder> the CPU
// results are compared bit for bit with files a pre-protocol-2 build wrote (cpu_640x360.f32,
// cpu_hd.f32, one 512 tile: tile512_in.f32 / tile512_out.f32); if birefnet-lite/model.onnx exists
// and a GPU is usable, one whole-frame BiRefNet run through the runner (the generic call) on the
// synthetic "all" scene must give IoU > 0.9 against its truth (phase 0: 0.98). Prints the timings
// for the report. Exit 77 (skipped) without the model.
// Usage: test_runner_real <mr_ml_runner> <model dir> [<cuda lib dir>]
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "edge_refine.h"
#include "ort_runner.h"
#include "runner_client.h"
#include "runner_ipc.h"
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

std::string gRunner, gDir, gSock, gHome, gModels, gCuda;
constexpr float kCudaJitter = 0.005f;  // max |diff| per pixel between two CUDA runs of a frame

RunnerClientOptions opts(const char* name) {
  RunnerClientOptions o;
  o.runnerPath = gRunner;
  o.socketPath = gSock;
  o.name = name;
  o.tileTimeoutS = 60;  // a cold CPU fallback is slow
  o.startTimeoutS = 15;
  o.idleSeconds = 120;
  o.env = {"HOME=" + gHome};
  o.extraArgs = {"--retry-s", "2"};  // the failure memory, 30 s by default
  return o;
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

struct Frame {
  int w, h;
  synth::Scene s;
  RefineParams p;
  std::vector<float> sd, tri;
  Frame(int w, int h) : w(w), h(h), s(synth::make("coarse", w, h)) {
    p.bandPx = 10.0 * h / 1080;
    buildTrimap(s.matte.data(), w, h, p.bandPx, &sd, &tri);
  }
  std::vector<float> run(TileRunner& r, RefineStats* st) const {
    std::vector<float> out(s.matte.size());
    refineEdge(s.matte.data(), s.rgb.data(), out.data(), w, h, p, r, st);
    return out;
  }
  double error(const std::vector<float>& out) const { return bandError(out, s.truth, sd, p.bandPx); }
  double inputError() const { return bandError(s.matte, s.truth, sd, p.bandPx); }
};

float maxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

bool readF32(const std::string& path, std::vector<float>* v) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  v->resize(size_t(f.tellg()) / 4);
  f.seekg(0);
  return bool(f.read(reinterpret_cast<char*>(v->data()), std::streamsize(v->size() * 4)));
}

// Bit equality (NaN-safe): the same 32 bits per value.
bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
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

// BiRefNet's pre / post processing as in mr_ml_bench (phase 0): the picture clamped, resized to
// 1024 x 1024, ImageNet mean / std, planar; the logits through a sigmoid, resized back.
std::vector<float> segmentViaRunner(RunnerClient& c, const synth::Scene& s, int w, int h) {
  const int S = 1024;
  std::vector<float> disp(s.rgb.size()), small(size_t(S) * S * 3), input(size_t(3) * S * S), logits(size_t(S) * S);
  for (size_t i = 0; i < disp.size(); ++i) disp[i] = clampPicture(s.rgb[i]);
  resizeBilinear(disp.data(), w, h, 3, small.data(), S, S);
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sdv[3] = {0.229f, 0.224f, 0.225f};
  for (size_t i = 0; i < size_t(S) * S; ++i)
    for (int ch = 0; ch < 3; ++ch) input[ch * size_t(S) * S + i] = (small[i * 3 + ch] - mean[ch]) / sdv[ch];
  c.runFrame("birefnet-lite", input.data(), 3, S, S, logits.data(), 1);
  for (float& v : logits) v = 1.0f / (1.0f + std::exp(-v));
  std::vector<float> out(size_t(w) * h);
  resizeBilinear(logits.data(), S, S, 1, out.data(), w, h);
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

void printStats(const char* what, const RefineStats& st, const RunnerClientStats& cs) {
  const double n = std::max(cs.tiles, 1);
  std::printf("  %s: %d tiles (T %d): frame %.1f ms = trimap %.1f + prep %.1f + tiles %.1f + post %.1f + blend %.1f | "
              "per tile: round trip %.2f, model %.2f, queue %.2f, transport %.2f, copy %.2f ms\n",
              what, st.tilesRun, st.tileSize, st.totalMs, st.trimapMs, st.prepMs, st.runMs, st.postMs, st.blendMs,
              cs.roundTripMs / n, cs.runMs / n, cs.queueMs / n, (cs.roundTripMs - cs.runMs - cs.queueMs) / n,
              cs.copyMs / n);
}

bool socketGone(int ms) {
  for (int i = 0; i < ms / 20; ++i) {
    if (access(gSock.c_str(), F_OK) != 0) return true;
    usleep(20000);
  }
  return false;
}

void stopRunner() {
  RunnerClient c(opts("stop"));
  std::string err;
  c.quit(&err);
}

std::string procFile(int pid, const char* name) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/" + name);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  gRunner = argv[1];
  gModels = argv[2];
  gCuda = argc > 3 ? argv[3] : "";
  if (gModels.empty() || access((gModels + "/vitmatte-small/model.onnx").c_str(), R_OK) != 0) {
    std::printf("no model in '%s': skipped\n", gModels.c_str());
    return 77;
  }
  signal(SIGPIPE, SIG_IGN);
  char tmpl[] = "/tmp/mrml-real-XXXXXX";
  if (!mkdtemp(tmpl)) return 2;
  gDir = tmpl;
  gSock = gDir + "/runner.sock";
  gHome = gDir + "/home";
  mkdir(gHome.c_str(), 0700);
  setenv("HOME", gHome.c_str(), 1);  // this process's own log lines stay out of the user's log
  mkdir((gHome + "/.config").c_str(), 0700);
  mkdir((gHome + "/.config/matte-refine").c_str(), 0700);
  std::ofstream(gHome + "/.config/matte-refine/config.conf")
      << "ml_model_dir=" << gModels << "\nml_cuda_libs=" << gCuda << "\nml_ep=cuda\nlog_level=info\n";
  const bool gpu = gpuUsedMiB() >= 0 && !gCuda.empty();
  const long gpu0 = gpuUsedMiB();

  Frame hd(1920, 1080), uhd(3840, 2160);
  std::printf("runner %s, build '%s', GPU %s\n", gRunner.c_str(), runnerBuildId(), gpu ? "yes" : "no (CPU only)");

  // 1. Cold: the first tile starts the runner, which creates the session.
  RunnerClient a(opts("A"));
  RefineStats st;
  int64_t t0 = ipc::nowMs();
  std::vector<float> first = hd.run(a, &st);
  const double cold = double(ipc::nowMs() - t0);
  std::printf("cold HD frame (runner start + session + frame): %.0f ms (connect %.0f ms, session load %.0f ms, ep %s%s%s)\n",
              cold, a.stats().connectMs, a.stats().loadMs, a.stats().ep.c_str(), a.stats().why.empty() ? "" : ": ",
              a.stats().why.c_str());
  CHECK(a.stats().runnerStarts == 1 && a.stats().sessionCreated);
  if (gpu) CHECK(a.stats().ep == "cuda");
  const double e0 = hd.inputError(), e1 = hd.error(first);
  std::printf("HD band error %.4f -> %.4f\n", e0, e1);
  CHECK(e1 < 0.06 && e1 < e0 / 3);
  long outside = 0;
  for (size_t i = 0; i < first.size(); ++i) outside += std::fabs(hd.sd[i]) >= hd.p.bandPx && first[i] != hd.s.matte[i];
  CHECK(outside == 0);

  // The runner's own library path and libraries (decision 32).
  const std::string environ = procFile(a.runnerPid(), "environ");
  const std::string maps = procFile(a.runnerPid(), "maps");
  CHECK(environ.find("LD_LIBRARY_PATH=") != std::string::npos);
  if (!gCuda.empty()) CHECK(environ.find(gCuda) != std::string::npos);
  CHECK(maps.find("libonnxruntime.so") != std::string::npos);
  char real[4096];
  if (gpu) CHECK(realpath(gCuda.c_str(), real) && maps.find(std::string(real) + "/libcudnn") != std::string::npos);

  // 2. Warm HD and UHD through the runner.
  std::vector<double> hdMs, uhdMs;
  RunnerClientStats lastHd;
  RefineStats lastHdSt;
  for (int i = 0; i < 5; ++i) {
    a.resetStats();
    std::vector<float> o = hd.run(a, &st);
    hdMs.push_back(st.totalMs);
    CHECK(maxDiff(o, first) <= (gpu ? kCudaJitter : 0.0f));  // the same session, the same answer
    lastHd = a.stats();
    lastHdSt = st;
  }
  printStats("warm HD", lastHdSt, lastHd);
  for (int i = 0; i < 3; ++i) {
    a.resetStats();
    std::vector<float> o = uhd.run(a, &st);
    uhdMs.push_back(st.totalMs);
    if (i == 2) {
      printStats("warm UHD", st, a.stats());
      std::printf("  UHD band error %.4f -> %.4f\n", uhd.inputError(), uhd.error(o));
      CHECK(uhd.error(o) < 0.06);
    }
  }
  std::printf("warm frame median: HD %.1f ms, UHD %.1f ms\n", median(hdMs), median(uhdMs));

  // 3. Two clients at once, each its own frames: identical results, one session.
  {
    RunnerClient b(opts("B"));
    std::vector<float> rb;
    hd.run(b, &st);  // connect B first (no runner start: it runs)
    CHECK(b.stats().runnerStarts == 0 && !b.stats().sessionCreated);
    std::vector<double> ta, tb;
    std::vector<std::vector<float>> oa(3), ob(3);
    double qa = 0, qb = 0;
    auto loop = [&](RunnerClient& c, std::vector<double>& t, std::vector<std::vector<float>>& outs, double& q) {
      c.resetStats();
      RefineStats s;
      for (int i = 0; i < 3; ++i) {
        outs[i] = hd.run(c, &s);
        t.push_back(s.totalMs);
      }
      q = c.stats().queueMs;
    };
    std::thread x([&] { loop(a, ta, oa, qa); }), y([&] { loop(b, tb, ob, qb); });
    x.join();
    y.join();
    float d = 0;
    for (int i = 0; i < 3; ++i) d = std::max({d, maxDiff(oa[i], first), maxDiff(ob[i], first)});
    std::printf("two concurrent clients, HD: frame median A %.1f ms, B %.1f ms (queue wait per frame A %.1f, B %.1f ms), "
                "max diff to the single-client result %g\n",
                median(ta), median(tb), qa / 3, qb / 3, d);
    CHECK(d <= (gpu ? kCudaJitter : 0.0f));
    proto::StatusReply sr;
    std::string err;
    CHECK(a.status(&sr, &err));
    std::printf("status: pid %u, %u connections, %u session(s), %llu tiles, RSS %d MiB; GPU MiB %ld -> %ld\n", sr.pid,
                sr.connections, sr.sessions, (unsigned long long)sr.runs, sr.rssMiB, gpu0, gpuUsedMiB());
    CHECK(sr.sessions == 1 && sr.connections >= 2);
    CHECK(ipc::getField(sr.session[0].model) == "vitmatte-small" &&
          ipc::getField(sr.session[0].activeEp) == (gpu ? "cuda" : "cpu"));
  }

  // 4. The same tiles in-process: CPU bit-identical, CUDA within a bound.
  {
    Frame small(640, 360);
    RunnerConfig cpu;
    cpu.modelPath = gModels + "/vitmatte-small/model.onnx";
    cpu.ep = "cpu";
    RunnerInfo ri;
    std::string err;
    auto local = acquireRunner(cpu, &ri, &err);
    CHECK(local != nullptr);
    if (local) {
      RunnerClient c(opts("cpu"));
      c.setModel("vitmatte-small", proto::kEpCpu);
      const std::vector<float> viaRunner = small.run(c, &st), inProc = small.run(*local, &st);
      std::printf("CPU EP, 640x360: runner vs in-process max diff %g\n", maxDiff(viaRunner, inProc));
      CHECK(maxDiff(viaRunner, inProc) == 0.0f);
      CHECK(c.stats().ep == "cpu");
      // Protocol 2 against a protocol 1 build's results (the same model, EP and inputs).
      if (const char* bd = std::getenv("MRML_BASELINE")) {
        const std::string dir = bd;
        std::vector<float> b640, bhd, bin, bout;
        const bool have = readF32(dir + "/cpu_640x360.f32", &b640) && readF32(dir + "/cpu_hd.f32", &bhd) &&
                          readF32(dir + "/tile512_in.f32", &bin) && readF32(dir + "/tile512_out.f32", &bout);
        CHECK(have);
        if (have) {
          const std::vector<float> nhd = hd.run(c, &st);
          const int T = 512;
          std::vector<float> tin(size_t(4) * T * T), tout(size_t(T) * T);
          prepareTile(hd.s.rgb.data(), hd.tri.data(), hd.w, hd.h, Tile{704, 268}, T, tin.data());
          c.run(tin.data(), T, tout.data(), 704, 268);
          const bool s640 = sameBits(viaRunner, b640), shd = sameBits(nhd, bhd), sin = sameBits(tin, bin),
                     sout = sameBits(tout, bout);
          std::printf("baseline (%s): 640x360 frame %s, HD frame %s (max diff %g), 512 tile input %s, output %s\n",
                      bd, s640 ? "bit-identical" : "DIFFERENT", shd ? "bit-identical" : "DIFFERENT",
                      shd ? 0.0 : double(maxDiff(nhd, bhd)), sin ? "identical" : "DIFFERENT",
                      sout ? "bit-identical" : "DIFFERENT");
          CHECK(s640 && shd && sin && sout);
        }
      } else {
        std::printf("baseline comparison: skipped (MRML_BASELINE not set)\n");
      }
    }
    if (gpu) {
      RunnerConfig cu = cpu;
      cu.ep = "cuda";
      cu.cudaLibDir = gCuda;
      auto g = acquireRunner(cu, &ri, &err);
      CHECK(g != nullptr && ri.ep == "cuda");
      if (g) {
        const std::vector<float> inProc = hd.run(*g, &st);
        const float d = maxDiff(first, inProc);
        std::printf("CUDA EP, HD: runner vs in-process (another session) max diff %g, band error %.4f vs %.4f\n", d,
                    hd.error(first), hd.error(inProc));
        CHECK(d <= kCudaJitter && std::fabs(hd.error(first) - hd.error(inProc)) < 0.001);
      }
    }
  }

  // 4b. A broken model file: the error is remembered (an instant answer), retried after --retry-s;
  // a third session evicts the least recently used one (small/cuda here).
  {
    Frame small(640, 360);
    const std::string bad = gDir + "/bad";
    mkdir(bad.c_str(), 0700);
    mkdir((bad + "/vitmatte-base").c_str(), 0700);
    std::ofstream(bad + "/vitmatte-base/model.onnx") << "not an onnx file";
    auto config = [&](const std::string& models) {
      std::ofstream(gHome + "/.config/matte-refine/config.conf")
          << "ml_model_dir=" << models << "\nml_cuda_libs=" << gCuda << "\nml_ep=cuda\nlog_level=info\n";
    };
    config(bad);
    RunnerClient c(opts("base"));
    c.setModel("vitmatte-base", proto::kEpCpu);
    std::string e1, e2;
    try {
      small.run(c, &st);
    } catch (const std::exception& e) {
      e1 = e.what();
    }
    const int64_t t1 = ipc::nowMs();
    try {
      small.run(c, &st);
    } catch (const std::exception& e) {
      e2 = e.what();
    }
    const int64_t quick = ipc::nowMs() - t1;
    std::printf("broken model: '%s'; again: '%s' (%lld ms)\n", e1.c_str(), e2.c_str(), (long long)quick);
    CHECK(e1.find("model error: cannot load the model") == 0);
    CHECK(e2.find("tried again") != std::string::npos && quick < 200);
    config(gModels);
    usleep(2200000);
    try {
      std::vector<float> ob = small.run(c, &st);  // retried: now loads
      CHECK(c.stats().ep == "cpu" && small.error(ob) < small.inputError());
    } catch (const std::exception& e) {
      std::printf("FAIL: the retry after --retry-s: %s\n", e.what());
      ++failures;
    }
    proto::StatusReply sr;
    std::string err;
    CHECK(c.status(&sr, &err));
    bool smallCuda = false, smallCpu = false, baseCpu = false;
    for (uint32_t i = 0; i < sr.sessions && i < 8; ++i) {
      const proto::SessionInfo& si = sr.session[i];
      const std::string m = ipc::getField(si.model);
      smallCuda |= m == "vitmatte-small" && si.ep == proto::kEpCuda;
      smallCpu |= m == "vitmatte-small" && si.ep == proto::kEpCpu;
      baseCpu |= m == "vitmatte-base" && si.ep == proto::kEpCpu;
    }
    std::printf("sessions after a third model: %u (small/cuda %d, small/cpu %d, base/cpu %d)\n", sr.sessions,
                int(smallCuda), int(smallCpu), int(baseCpu));
    CHECK(sr.sessions == 2 && !smallCuda && smallCpu && baseCpu);
  }

  // 4c. Another model shape through the same runner: BiRefNet-lite, one whole frame (3 -> 1 at
  // 1024 x 1024), the generic call. GPU only (the CPU run takes ~3 s and 12 GB, phase 0).
  if (access((gModels + "/birefnet-lite/model.onnx").c_str(), R_OK) != 0 || !gpu) {
    std::printf("BiRefNet whole frame: skipped (%s)\n", gpu ? "no birefnet-lite model" : "no GPU");
  } else {
    const int w = 1920, h = 1080;
    const synth::Scene sc = synth::make("all", w, h);
    RunnerClient c(opts("segment"));
    std::vector<float> seg;
    std::vector<double> ms;
    try {
      for (int i = 0; i < 4; ++i) {
        c.resetStats();
        const int64_t t1 = ipc::nowUs();
        seg = segmentViaRunner(c, sc, w, h);
        ms.push_back(double(ipc::nowUs() - t1) / 1000.0);
        if (i == 0)
          std::printf("BiRefNet-lite, first frame %.0f ms (session load %.0f ms, ep %s)\n", ms[0], c.stats().loadMs,
                      c.stats().ep.c_str());
      }
      ms.erase(ms.begin());
      long inter = 0, uni = 0;
      for (size_t i = 0; i < seg.size(); ++i) {
        const bool p = seg[i] >= 0.5f, t = sc.truth[i] >= 0.5f;
        inter += p && t;
        uni += p || t;
      }
      const double iou = uni ? double(inter) / uni : 0;
      std::printf("BiRefNet-lite via the runner, HD \"all\": IoU %.4f vs the truth, warm frame median %.1f ms "
                  "(model %.1f ms, round trip %.1f ms, copy %.1f ms), slot %zu MiB\n",
                  iou, median(ms), c.stats().runMs, c.stats().roundTripMs, c.stats().copyMs, c.slotBytes() >> 20);
      CHECK(iou > 0.9 && c.stats().ep == "cuda" && c.slotBytes() == proto::kDefaultSlotBytes);
      proto::StatusReply sr;
      std::string err;
      CHECK(c.status(&sr, &err));
      bool bir = false;
      for (uint32_t i = 0; i < sr.sessions && i < 8; ++i) bir |= ipc::getField(sr.session[i].model) == "birefnet-lite";
      CHECK(sr.sessions == 2 && bir);
    } catch (const std::exception& e) {
      std::printf("FAIL: BiRefNet via the runner: %s\n", e.what());
      ++failures;
    }
  }

  // 5. A client of another build: replaces the runner once, then fails cleanly (the real runner
  // only ever reports its own build); the right client recovers with one reconnect.
  {
    RunnerClientOptions o = opts("other");
    o.buildId = "another build";
    RunnerClient x(o);
    const int before = a.runnerPid();
    std::string e;
    try {
      hd.run(x, &st);
    } catch (const std::exception& ex) {
      e = ex.what();
    }
    std::printf("other build: '%s' (replaced %d, starts %d)\n", e.c_str(), x.stats().replaced, x.stats().runnerStarts);
    CHECK(e.find("version mismatch") != std::string::npos && x.stats().replaced == 1 && x.stats().runnerStarts == 1);
    a.resetStats();
    t0 = ipc::nowMs();
    const std::vector<float> o2 = hd.run(a, &st);
    std::printf("after the replacement: frame %.0f ms (reconnects %d, starts %d, replaced %d, new session %d)\n",
                double(ipc::nowMs() - t0), a.stats().reconnects, a.stats().runnerStarts, a.stats().replaced,
                int(a.stats().sessionCreated));
    CHECK(a.runnerPid() != before && a.stats().reconnects == 1 && hd.error(o2) < 0.06);
  }

  // 6. Cold start again (measured), then idle exit with a short idle time.
  stopRunner();
  CHECK(socketGone(5000));
  {
    RunnerClientOptions o = opts("idle");
    o.idleSeconds = 2;
    RunnerClient c(o);
    t0 = ipc::nowMs();
    hd.run(c, &st);
    std::printf("cold HD frame, second time: %.0f ms (connect incl. start %.0f ms, load %.0f ms)\n",
                double(ipc::nowMs() - t0), c.stats().connectMs, c.stats().loadMs);
  }
  CHECK(socketGone(8000));
  std::printf("idle exit: %s\n", access(gSock.c_str(), F_OK) != 0 ? "ok" : "FAILED");
  if (failures) std::printf("kept %s\n", gDir.c_str());
  else if (std::system(("rm -rf '" + gDir + "'").c_str()) != 0) std::printf("could not remove %s\n", gDir.c_str());
  std::printf(failures ? "test_runner_real: %d FAILED\n" : "test_runner_real: all passed\n", failures);
  return failures ? 1 : 0;
}
