// The table's models through ONNX Runtime (ort_runner.h). C++17 (ORT 1.31 headers); only this file
// sees ORT. Every model: one float32 input [1, C, H, W] and one float32 output, named by the table
// row (else input / output 0); ViTMatte's contract as in mr_ml_bench (phase 0): "pixel_values"
// [1, 4, T, T] (RGB normalised + trimap) -> "alphas" [1, 1, T, T].
#include "ort_runner.h"

#include <dlfcn.h>
#include <unistd.h>

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "log.h"

namespace mr {
namespace ml {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// The CUDA EP's libraries in dependency order: the minimal set for ViTMatte (LD_DEBUG=files on
// a CUDA run, 2026-10-23; cuDNN's cnn / adv / ext were never opened). The provider
// (libonnxruntime_providers_cuda.so) needs the first four by soname (readelf NEEDED); cuDNN
// dlopens its sub-libraries by name (trying libX.so.9.27.0, .9.27, then .9). Opened RTLD_GLOBAL by
// absolute path, they are then found by soname: Flame's process has no loader path to them and
// LD_LIBRARY_PATH can't change after start.
const char* const kCudaLibs[] = {
    "libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10",
    "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9", "libcudnn.so.9",
};

// Loads the CUDA libraries once per process. Returns "" or what failed.
std::string loadCudaLibs(const std::string& dir, int* loaded, double* ms) {
  static bool done = false;
  static std::string result;
  static int count = 0;
  static double took = 0;
  if (!done) {
    done = true;
    const auto t0 = Clock::now();
    for (const char* name : kCudaLibs) {
      const std::string path = dir.empty() ? std::string(name) : dir + "/" + name;
      if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
        ++count;
      } else {
        const char* e = dlerror();
        LOGW("ml: cuda lib %s not loaded: %s", path.c_str(), e ? e : "?");
        if (result.empty()) result = std::string("cannot load ") + path + ": " + (e ? e : "?");
      }
    }
    took = msSince(t0);
  }
  *loaded = count;
  *ms = took;
  return result;
}

Ort::Env& env() {
  // Leaked on purpose: no teardown at process exit (see ort_runner.h).
  static Ort::Env* e = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "matte-refine-ml");
  return *e;
}

// The index of the input (output) called `name`, or 0 when name is empty; -1 if absent.
template <class NameAt>
int findTensor(size_t count, const char* name, NameAt nameAt) {
  if (!*name) return count > 0 ? 0 : -1;
  for (size_t i = 0; i < count; ++i)
    if (nameAt(i) == name) return int(i);
  return -1;
}

class OrtSession : public ModelSession {
 public:
  OrtSession(Ort::Session&& s, const ModelSpec& m) : session_(std::move(s)), id_(m.id) {
    Ort::AllocatorWithDefaultOptions alloc;
    const int in = findTensor(session_.GetInputCount(), m.inputName,
                              [&](size_t i) { return std::string(session_.GetInputNameAllocated(i, alloc).get()); });
    const int out = findTensor(session_.GetOutputCount(), m.outputName,
                               [&](size_t i) { return std::string(session_.GetOutputNameAllocated(i, alloc).get()); });
    if (in < 0 || out < 0)
      throw std::runtime_error(std::string("not a ") + m.id + " model (no input '" + m.inputName + "' or output '" +
                               m.outputName + "')");
    inName_ = session_.GetInputNameAllocated(size_t(in), alloc).get();
    outName_ = session_.GetOutputNameAllocated(size_t(out), alloc).get();
    Ort::TypeInfo type = session_.GetInputTypeInfo(size_t(in));  // keep alive: ti points into it
    auto ti = type.GetTensorTypeAndShapeInfo();
    const auto shape = ti.GetShape();
    if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 4 ||
        (shape[1] > 0 && shape[1] != m.channelsIn))
      throw std::runtime_error(std::string("not a ") + m.id + " model (input '" + inName_ + "' must be float32 [N, " +
                               std::to_string(m.channelsIn) + ", H, W])");
  }

  void run(const float* input, int cIn, int h, int w, float* out, int cOut, int oh, int ow) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t shape[4] = {1, cIn, h, w};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input), size_t(cIn) * h * w, shape, 4);
    const char* inName = inName_.c_str();
    const char* outName = outName_.c_str();
    auto outs = session_.Run(Ort::RunOptions{nullptr}, &inName, &in, 1, &outName, 1);
    auto info = outs[0].GetTensorTypeAndShapeInfo();
    const size_t n = size_t(cOut) * oh * ow;
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetElementCount() != n)
      throw std::runtime_error("unexpected output of " + id_ + " (" + std::to_string(info.GetElementCount()) +
                               " values, expected " + std::to_string(n) + ")");
    std::memcpy(out, outs[0].GetTensorData<float>(), n * sizeof(float));
  }

 private:
  Ort::Session session_;
  std::string id_, inName_, outName_;
  std::mutex mutex_;
};

// A ViTMatte tile ([4, T, T] -> [T, T]) on a session: the in-process TileRunner (acquireRunner).
class TileAdapter : public TileRunner {
 public:
  explicit TileAdapter(std::shared_ptr<ModelSession> s) : s_(std::move(s)) {}
  void run(const float* input, int T, float* alpha, int, int) override { s_->run(input, 4, T, T, alpha, 1, T, T); }

 private:
  std::shared_ptr<ModelSession> s_;
};

struct Entry {
  std::shared_ptr<TileRunner> runner;
  std::string error, ep, why;
};

Ort::Session createSession(const std::string& model, bool cuda, const CudaArena& arena) {
  // The Env first: it registers ORT's default logger, which appending the CUDA EP needs ("Attempt
  // to use DefaultLogger but none has been registered" when the first session is a CUDA one).
  Ort::Env& e = env();
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (cuda) {
    // onnxruntime_c_api.h (ORT 1.31) OrtCUDAProviderOptions (line 699): device_id 0, gpu_mem_limit
    // bytes, default SIZE_MAX = no limit (lines 704, 726-730), arena_extend_strategy 0 =
    // kNextPowerOfTwo (default) / 1 = kSameAsRequested (lines 731-738). arena = cudaArenaFor(row, GPU
    // total): 0 / false keep these defaults (decision 36: never cap a big card).
    OrtCUDAProviderOptions o;
    if (arena.limitMiB > 0) o.gpu_mem_limit = size_t(arena.limitMiB) << 20;
    if (arena.sameAsRequested) o.arena_extend_strategy = 1;
    so.AppendExecutionProvider_CUDA(o);
  }
  return Ort::Session(e, model.c_str(), so);
}

// ORT's messages end with a newline: one log line each.
std::string oneLine(std::string s) {
  for (char& c : s)
    if (c == '\n' || c == '\r') c = ' ';
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

}  // namespace

std::string ortVersion() {
  try {
    return Ort::GetVersionString();
  } catch (...) {
    return "?";
  }
}

long gpuUsedMiB() {
  const char* smi = "/usr/bin/nvidia-smi";  // absolute: no reliance on the host's PATH
  if (access(smi, X_OK) != 0) return -1;
  FILE* p = popen("/usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null", "r");
  if (!p) return -1;
  long v = -1;
  if (std::fscanf(p, "%ld", &v) != 1) v = -1;
  pclose(p);
  return v;
}

long gpuTotalMiB() {
  // nvidia-smi --help-query-gpu: "memory.total": Total installed GPU memory (driver 580, BFD).
  static long total = [] {
    const char* smi = "/usr/bin/nvidia-smi";  // absolute: no reliance on the host's PATH
    if (access(smi, X_OK) != 0) return -1L;
    FILE* p = popen("/usr/bin/nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null", "r");
    if (!p) return -1L;
    long v = -1;
    if (std::fscanf(p, "%ld", &v) != 1) v = -1;
    pclose(p);
    return v;
  }();
  return total;
}

long rssMiB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.compare(0, 6, "VmRSS:") == 0) return std::atol(line.c_str() + 6) / 1024;
  return -1;
}

std::shared_ptr<ModelSession> createModelSession(const RunnerConfig& cfg, RunnerInfo* info, std::string* error,
                                                 bool* missing) {
  RunnerInfo ri;
  Entry e;
  std::shared_ptr<ModelSession> session;
  if (missing) *missing = false;
  const ModelSpec* spec = findModel(cfg.modelId);
  if (!spec) {
    if (error) *error = "unknown model '" + cfg.modelId + "'";
    if (info) *info = ri;
    return nullptr;
  }
  if (access(cfg.modelPath.c_str(), R_OK) != 0) {
    if (error) *error = "model not found: " + cfg.modelPath;
    if (missing) *missing = true;
    if (info) *info = ri;
    return nullptr;
  }
  ri.created = true;
  ri.gpuMiBBefore = gpuUsedMiB();
  ri.rssMiBBefore = rssMiB();
  const auto t0 = Clock::now();
  bool cuda = cfg.ep == "cuda";
  if (cfg.ep != "cuda" && cfg.ep != "cpu") {
    e.why = "unknown ml_ep '" + cfg.ep + "'";
    cuda = false;
  }
  if (cuda) {
    static std::mutex libsMutex;
    std::lock_guard<std::mutex> lock(libsMutex);
    const std::string failed = loadCudaLibs(cfg.cudaLibDir, &ri.cudaLibsLoaded, &ri.cudaLibsMs);
    if (!failed.empty()) {
      e.why = failed;
      cuda = false;
    }
  }
  try {
    std::unique_ptr<Ort::Session> s;
    if (cuda) {
      ri.gpuTotalMiB = gpuTotalMiB();
      const CudaArena arena = cudaArenaFor(*spec, ri.gpuTotalMiB);
      ri.arenaLimitMiB = arena.limitMiB;
      ri.arenaSameAsRequested = arena.sameAsRequested;
      try {
        s.reset(new Ort::Session(createSession(cfg.modelPath, true, arena)));
      } catch (const std::exception& ex) {
        e.why = "CUDA EP failed: " + oneLine(ex.what());
        LOGW("ml: %s; falling back to the CPU", e.why.c_str());
        cuda = false;
      }
    }
    if (!s) s.reset(new Ort::Session(createSession(cfg.modelPath, false, CudaArena{0, false})));
    session = std::make_shared<OrtSession>(std::move(*s), *spec);
  } catch (const std::exception& ex) {
    e.error = "cannot load the model: " + oneLine(ex.what());
  } catch (...) {
    e.error = "cannot load the model (unknown error)";
  }
  ri.loadMs = msSince(t0);
  ri.gpuMiBAfter = gpuUsedMiB();
  ri.rssMiBAfter = rssMiB();
  ri.ep = session ? (cuda ? "cuda" : "cpu") : "none";
  ri.why = e.why;
  if (info) *info = ri;
  if (error) *error = e.error;
  return session;
}

std::shared_ptr<TileRunner> acquireRunner(const RunnerConfig& cfg, RunnerInfo* info, std::string* error) {
  static std::mutex mutex;
  static std::map<std::string, Entry>* sessions = new std::map<std::string, Entry>();  // leaked
  std::lock_guard<std::mutex> lock(mutex);
  const std::string key = cfg.modelPath + "|" + cfg.ep;
  auto it = sessions->find(key);
  if (it != sessions->end()) {
    RunnerInfo ri;
    ri.ep = it->second.ep;
    ri.why = it->second.why;
    if (info) *info = ri;
    if (error) *error = it->second.error;
    return it->second.runner;
  }
  RunnerInfo ri;
  std::string err;
  bool missing = false;
  Entry e;
  std::shared_ptr<ModelSession> s = createModelSession(cfg, &ri, &err, &missing);
  if (s) e.runner = std::make_shared<TileAdapter>(s);
  if (info) *info = ri;
  if (error) *error = err;
  if (missing) return nullptr;  // not remembered: the user may add it
  e.error = err;
  e.ep = ri.ep;
  e.why = ri.why;
  (*sessions)[key] = e;
  return e.runner;
}

}  // namespace ml
}  // namespace mr
