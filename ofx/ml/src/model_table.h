// The models Matte Refine ML knows (runner protocol v2, ofx/ml/CLAUDE.md phase 1): one row per
// model. Shared by the plugin (RunnerClient), the runner (request validation, session creation)
// and the tests. C++14, no ONNX Runtime, no OS headers.
//
// Adding a model = one row here + its pre / post processing on the client side (the code that
// builds the input tensor and reads the output); the protocol and the runner stay as they are.
//
// Tensors are float32 NCHW with N = 1: input [1, channelsIn, H, W], output [1, channelsOut, Ho, Wo].
// Tensor names: the row names them; the runner checks at session creation that the model has
// them (a wrong file is a clear error, not a wrong answer). An empty name = the model's input /
// output 0, read from the ONNX session.
#pragma once

#include <cstdint>
#include <string>

namespace mr {
namespace ml {

enum class SizeRule : uint32_t {
  kMultipleOf = 0,  // any H, W in [minSide, maxSide] that are multiples of `multiple` (tiles)
  kFixed = 1,       // exactly fixedW x fixedH (the model's input shape is fixed; whole frame)
};

struct ModelSpec {
  const char* id;          // the name in RUN requests and the log, "vitmatte-small"
  const char* file;        // relative to the runner's ml_model_dir: "<id>/model.onnx"
  const char* inputName;   // "" = input 0 of the model
  const char* outputName;  // "" = output 0 of the model
  int channelsIn, channelsOut;
  SizeRule sizeRule;
  int multiple;            // kMultipleOf: H and W multiples of it
  int minSide, maxSide;    // kMultipleOf: the range of H and W
  int fixedW, fixedH;      // kFixed: the only accepted size
  // The output's H, W: the same as the input's (every model so far). A model with another output
  // size gets a rule here and in outputSize().
  const char* licence;     // a licence id for the setup's licence gate (P2), e.g. "vitmatte-c1k"
  const char* sha256;      // of the file; "" = unknown. Not checked yet (P2's download step will)
  // Per-model runner settings (the CUDA EP only), a hint applied by cudaArenaFor() (decision 36: only
  // on a card of kSmallGpuMiB or less). 0 / false = ORT's defaults.
  uint32_t gpuMemLimitMiB;    // OrtCUDAProviderOptions::gpu_mem_limit (the CUDA arena), 0 = no limit
  bool arenaSameAsRequested;  // arena_extend_strategy kSameAsRequested (1) instead of kNextPowerOfTwo (0)
};

// Decision 36: BiRefNet must run on a 24 GB card, a bigger card is never capped. A row's arena hint
// applies only when the GPU's total memory is known and at most this (a 32 GB card reports
// 32623 MiB: RTX PRO 4500, nvidia-smi memory.total); unknown (query failed, <= 0) = no cap.
constexpr long kSmallGpuMiB = 32768;
struct CudaArena {
  uint32_t limitMiB;     // 0 = no limit
  bool sameAsRequested;  // only together with a limit
};
CudaArena cudaArenaFor(const ModelSpec& m, long gpuTotalMiB);

// The table. Indexes are not part of the protocol (requests name the id): rows may be reordered.
int modelCount();
const ModelSpec& modelAt(int i);
const ModelSpec* findModel(const std::string& id);  // nullptr if unknown

// The largest single run any row allows (input + output floats, bytes).
uint64_t maxRunBytes();

// A request's tensor description, as on the wire (runner_protocol.h RunRequest).
struct TensorDesc {
  uint32_t channelsIn, h, w, channelsOut, outH, outW;
};
uint64_t inputBytes(const TensorDesc& d);   // channelsIn * h * w * 4
uint64_t outputBytes(const TensorDesc& d);  // channelsOut * outH * outW * 4

// Whether d is a valid run of model m (channels, size rule, output size); "" or why not.
std::string checkRequest(const ModelSpec& m, const TensorDesc& d);

}  // namespace ml
}  // namespace mr
