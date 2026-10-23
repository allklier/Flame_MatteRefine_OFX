// The models of the table (model_table.h) through ONNX Runtime, behind a C++14 header
// (ort_runner.cpp is compiled as C++17: ORT 1.31's headers need it). Since decision 32 this runs
// only in mr_ml_runner (its own session cache: createModelSession) and in the in-process tools and
// tests (acquireRunner, ViTMatte tiles); the plugin never links it. acquireRunner: one
// process-wide session per (model file, EP), created at the first request and never destroyed (a
// CUDA session torn down during process exit can crash a host).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "edge_refine.h"
#include "model_table.h"

namespace mr {
namespace ml {

struct RunnerConfig {
  std::string modelPath;  // .onnx file
  // The table row (model_table.h): tensor names, channels, the CUDA arena settings.
  std::string modelId = "vitmatte-small";
  std::string ep = "cuda";  // "cuda" (falls back to the CPU) or "cpu"
  // Folder with the CUDA libraries (libcudart.so.13, cuBLAS, cuRAND, cuDNN 9), loaded by absolute
  // path before the CUDA session: Flame's loader path doesn't have them. "" = the system's.
  std::string cudaLibDir;
};

struct RunnerInfo {
  bool created = false;  // this call created the session (else reused or a cached failure)
  std::string ep;        // the active EP: "cuda" or "cpu"
  std::string why;       // why the EP differs from the requested one ("" if it doesn't)
  double loadMs = 0;     // session creation (incl. the CUDA library loading)
  double cudaLibsMs = 0;
  int cudaLibsLoaded = 0;
  long gpuMiBBefore = -1, gpuMiBAfter = -1, rssMiBBefore = -1, rssMiBAfter = -1;
  // The CUDA arena actually applied (cudaArenaFor, decision 36) and the GPU total it was decided on.
  long gpuTotalMiB = -1;
  uint32_t arenaLimitMiB = 0;
  bool arenaSameAsRequested = false;
};

// One ORT session of one table model: in [cIn, h, w] -> out [cOut, oh, ow], float32, N = 1.
// Throws std::exception on failure. Thread-safe (calls are serialised).
class ModelSession {
 public:
  virtual ~ModelSession() {}
  virtual void run(const float* in, int cIn, int h, int w, float* out, int cOut, int oh, int ow) = 0;
};

// The shared ViTMatte tile runner for this config (a TileRunner over a ModelSession), created if
// needed. nullptr on failure, with *error set (a failure is remembered: the next call returns it
// at once, unless the model file was missing). Never throws. Thread-safe.
std::shared_ptr<TileRunner> acquireRunner(const RunnerConfig& cfg, RunnerInfo* info, std::string* error);

// A new session for this config, not cached (mr_ml_runner keeps its own cache). nullptr on failure
// with *error; *missing = the model file does not exist (a failure not worth remembering). The
// CUDA libraries are loaded once per process (the first CUDA request's cudaLibDir). Thread-safe.
std::shared_ptr<ModelSession> createModelSession(const RunnerConfig& cfg, RunnerInfo* info, std::string* error,
                                                 bool* missing = nullptr);

std::string ortVersion();

// GPU memory in use (MiB, whole GPU 0, nvidia-smi) and this process's RSS (MiB); -1 if unknown.
long gpuUsedMiB();
// The total memory of GPU 0 (MiB, nvidia-smi memory.total, asked once per process); -1 if unknown.
long gpuTotalMiB();
long rssMiB();

}  // namespace ml
}  // namespace mr
