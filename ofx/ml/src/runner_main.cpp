// mr_ml_runner (ofx/ml/CLAUDE.md decision 32): the one process per user and machine that owns ONNX
// Runtime, the CUDA libraries and the models. The plugins send model runs (runner_client.h,
// protocol 2: any model of model_table.h, validated by the server against the table); this file is
// the model engine behind runner_server.h. C++17 (ORT headers).
//
// Sessions: keyed by (model id, EP), created at the first RUN that needs one (any connection), shared
// by all connections, at most kMaxSessions alive (the least recently used is destroyed first). A
// session that cannot be created is remembered for kRetryMs (every request in that time gets the
// same error at once), then tried again; a missing model file is not remembered. A CUDA request
// that falls back to the CPU keeps the CPU session for that key until the runner exits (idle exit
// or --quit retries CUDA), and every reply says why.
//
// Config (~/.config/matte-refine/config.conf, read at each RUN, cheap): ml_model_dir (default
// ~/.cache/matte-refine-ml/models; a model at <dir>/<its table row's file>, "<id>/model.onnx"),
// ml_ep (cuda | cpu, default cuda: the EP of a request that asks for the default), ml_cuda_libs
// (the CUDA 13 / cuDNN 9 folder; "" = the system's), log_level.
//
// Library path: at start (Linux) the runner re-executes itself once with LD_LIBRARY_PATH = <the
// folder libonnxruntime was loaded from>:<ml_cuda_libs>, so anything CUDA opens lazily by soname is
// found in our folders; the client never passes a library path (it starts the runner without one).
#include <dlfcn.h>
#include <unistd.h>

#include <onnxruntime_c_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

#include "log.h"
#include "ort_runner.h"
#include "runner_client.h"
#include "runner_ipc.h"
#include "runner_server.h"
#include "user_config.h"

namespace {

using namespace mr;
using namespace mr::ml;

// Live sessions at most (LRU): two ViTMatte sessions, or one ViTMatte + one BiRefNet. Each holds
// GPU memory (ViTMatte 1.5-3 GB, BiRefNet 8-16 GB unlimited: PHASE0_RESULTS.md).
constexpr size_t kMaxSessions = 2;
constexpr int64_t kRetryMs = 30000;  // --retry-s N overrides it (tests)

std::string homeDir() {
  const char* h = std::getenv("HOME");
  return h ? h : "/tmp";
}

std::string expandHome(const std::string& p) { return p.compare(0, 2, "~/") == 0 ? homeDir() + p.substr(1) : p; }

struct Cfg {
  std::string modelDir, cudaLibs, ep;
};

Cfg loadCfg() {
  const UserConfig c = loadUserConfig();
  auto get = [&](const char* k, const std::string& def) {
    auto it = c.find(k);
    return expandHome(it == c.end() || it->second.empty() ? def : it->second);
  };
  return {get("ml_model_dir", homeDir() + "/.cache/matte-refine-ml/models"), get("ml_cuda_libs", ""),
          get("ml_ep", "cuda")};
}

class OrtEngine : public Engine {
 public:
  explicit OrtEngine(int64_t retryMs) : retryMs_(retryMs) {}
  RunResult run(const ModelSpec& m, const proto::RunRequest& q, const float* in, float* out) override {
    RunResult res;
    const Cfg cfg = loadCfg();
    const uint32_t ep = q.ep != proto::kEpDefault ? q.ep : cfg.ep == "cpu" ? proto::kEpCpu : proto::kEpCuda;
    if (q.ep == proto::kEpDefault && cfg.ep != "cpu" && cfg.ep != "cuda" && !warnedEp_) {
      warnedEp_ = true;
      LOGW("unknown ml_ep '%s': using cuda", cfg.ep.c_str());
    }
    const std::string model = m.id;
    const auto key = std::make_pair(model, ep);
    auto it = lru_.begin();
    while (it != lru_.end() && !(it->model == model && it->ep == ep)) ++it;
    if (it == lru_.end()) {
      const int64_t now = ipc::nowMs();
      auto f = failures_.find(key);
      if (f != failures_.end() && now - f->second.at < retryMs_) {
        res.status = proto::kModelError;
        res.message = f->second.error + " (tried again " + std::to_string((retryMs_ - (now - f->second.at)) / 1000 + 1) +
                      " s from now)";
        return res;
      }
      while (lru_.size() >= kMaxSessions) {  // free its GPU memory before the new one is made
        const Session& old = lru_.back();
        LOGI("session %s/%s evicted (least recently used, %llu runs)", old.model.c_str(), epName(old.ep),
             (unsigned long long)old.runs);
        lru_.pop_back();
      }
      RunnerConfig rc;
      rc.modelPath = cfg.modelDir + "/" + m.file;
      rc.modelId = model;
      rc.ep = epName(ep);
      rc.cudaLibDir = cfg.cudaLibs;
      RunnerInfo ri;
      std::string err;
      bool missing = false;
      std::shared_ptr<ModelSession> r = createModelSession(rc, &ri, &err, &missing);
      LOGI("session %s/%s: %s, model %s, ep %s%s%s, load %.0f ms (cuda libs %d in %.0f ms), GPU MiB %ld -> %ld, "
           "RSS MiB %ld -> %ld, ORT %s, cuda arena limit %u MiB%s (hint %u MiB, GPU total %ld MiB, cap only <= %ld)",
           model.c_str(), epName(ep), r ? "created" : ("FAILED: " + err).c_str(), rc.modelPath.c_str(),
           ri.ep.c_str(), ri.why.empty() ? "" : " because: ", ri.why.c_str(), ri.loadMs, ri.cudaLibsLoaded,
           ri.cudaLibsMs, ri.gpuMiBBefore, ri.gpuMiBAfter, ri.rssMiBBefore, ri.rssMiBAfter, ortVersion().c_str(),
           ri.arenaLimitMiB, ri.arenaSameAsRequested ? ", same-as-requested" : "", m.gpuMemLimitMiB, ri.gpuTotalMiB,
           kSmallGpuMiB);
      if (!r) {
        if (!missing) failures_[key] = {err, now};
        res.status = proto::kModelError;
        res.message = err;
        return res;
      }
      failures_.erase(key);
      lru_.push_front(Session{model, ep, r, ri.ep, ri.why, 0, now});
      res.created = true;
      res.loadMs = ri.loadMs;
    } else if (it != lru_.begin()) {
      lru_.splice(lru_.begin(), lru_, it);
    }
    Session& s = lru_.front();
    const int64_t t0 = ipc::nowUs();
    try {
      s.session->run(in, int(q.cIn), int(q.h), int(q.w), out, int(q.cOut), int(q.outH), int(q.outW));
    } catch (const std::exception& e) {
      res.status = proto::kModelError;
      res.message = std::string("the model failed: ") + e.what();
    }
    res.runMs = double(ipc::nowUs() - t0) / 1000.0;
    ++s.runs;
    s.lastUse = ipc::nowMs();
    res.ep = s.activeEp;
    if (res.message.empty()) res.message = s.why;
    return res;
  }

  int sessions(proto::SessionInfo* out, int cap) override {
    int n = 0;
    const int64_t now = ipc::nowMs();
    for (const Session& s : lru_) {
      if (n < cap) {
        proto::SessionInfo& i = out[n];
        ipc::setField(i.model, s.model);
        i.ep = s.ep;
        ipc::setField(i.activeEp, s.activeEp);
        i.runs = s.runs;
        i.idleMs = uint32_t(now - s.lastUse);
      }
      ++n;
    }
    return n;
  }

 private:
  static const char* epName(uint32_t ep) { return ep == proto::kEpCpu ? "cpu" : "cuda"; }
  struct Session {
    std::string model;
    uint32_t ep;
    std::shared_ptr<ModelSession> session;
    std::string activeEp, why;
    uint64_t runs;
    int64_t lastUse;
  };
  struct Failure {
    std::string error;
    int64_t at;
  };
  std::list<Session> lru_;  // front = most recently used
  std::map<std::pair<std::string, uint32_t>, Failure> failures_;
  bool warnedEp_ = false;
  int64_t retryMs_;
};

// Linux: once, LD_LIBRARY_PATH = ORT's folder + ml_cuda_libs (see the top). Returns only if no
// re-exec was needed (or it failed: then the CUDA libraries are still opened by absolute path).
void reexecWithLibraryPath(char** argv) {
#if defined(__linux__)
  Dl_info info{};
  std::string ortDir;
  if (dladdr(reinterpret_cast<void*>(&OrtGetApiBase), &info) && info.dli_fname)
    ortDir = ipc::dirName(info.dli_fname);
  const std::string cuda = loadCfg().cudaLibs;
  std::string want = ortDir;
  if (!cuda.empty()) want += (want.empty() ? "" : ":") + cuda;
  const char* have = std::getenv("LD_LIBRARY_PATH");
  if (want.empty() || (have && want == have)) return;
  if (std::getenv("MRML_RUNNER_REEXEC")) {
    LOGW("runner: LD_LIBRARY_PATH is '%s' after the re-exec, wanted '%s'", have ? have : "", want.c_str());
    return;
  }
  setenv("LD_LIBRARY_PATH", want.c_str(), 1);
  setenv("MRML_RUNNER_REEXEC", "1", 1);
  // The resolved path, not /proc/self/exe: the process name (ps, /proc/<pid>/comm) stays mr_ml_runner.
  char self[4096];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  if (n > 0) {
    self[n] = 0;
    execv(self, argv);
  }
  LOGW("runner: re-exec with LD_LIBRARY_PATH failed: %s", std::strerror(errno));
#else
  (void)argv;
#endif
}

void usage() {
  std::printf(
      "mr_ml_runner: the Matte Refine ML model process (started by the plugin; one per user).\n"
      "  mr_ml_runner [--socket PATH] [--idle-s N] [--daemon]   serve (foreground unless --daemon)\n"
      "  mr_ml_runner --status [--socket PATH]                  show the running runner\n"
      "  mr_ml_runner --quit [--socket PATH]                    stop it\n"
      "  mr_ml_runner --version\n");
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--daemon") == 0) closeInheritedFds();  // a host's fds stay out
  setLogComponent("runner");  // log lines read "runner[pid/tid]", the plugin's "ofx[...]"
  logConfigure();
  LogScope scope("runner");
  ServerOptions o;
  std::vector<std::string> rest;
  std::string err;
  if (!parseServerArgs(argc, argv, &o, &rest, &err)) {
    std::fprintf(stderr, "mr_ml_runner: %s\n", err.c_str());
    return 1;
  }
  std::string cmd;
  int64_t retryMs = kRetryMs;
  for (size_t i = 0; i < rest.size(); ++i) {
    const std::string& a = rest[i];
    if (a == "--retry-s" && i + 1 < rest.size()) {
      retryMs = int64_t(std::atof(rest[++i].c_str()) * 1000);
    } else if (a == "--status" || a == "--quit") {
      cmd = a.substr(2);
    } else if (a == "--version") {
      std::printf("mr_ml_runner %s, protocol %u, ONNX Runtime %s\n", runnerBuildId(), proto::kVersion,
                  ortVersion().c_str());
      return 0;
    } else {
      usage();
      return a == "--help" ? 0 : 1;
    }
  }
  if (!cmd.empty()) {
    if (o.socketPath.empty()) o.socketPath = ipc::defaultSocketPath(&err, runnerBuildId());
    if (o.socketPath.empty()) {
      std::fprintf(stderr, "mr_ml_runner: %s\n", err.c_str());
      return 1;
    }
    return controlMain(cmd, o.socketPath);
  }
  reexecWithLibraryPath(argv);
  OrtEngine engine(retryMs);
  return serve(o, engine);
}
