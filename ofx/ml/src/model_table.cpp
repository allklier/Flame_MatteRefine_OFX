// model_table.h. The rows' facts come from phase 0 (PHASE0_RESULTS.md, the mlbench models.json:
// tensor names, shapes and sha256 read from the downloaded files, 2026-10-23).
#include "model_table.h"

namespace mr {
namespace ml {

namespace {

// ViTMatte (B1): [1, 4, T, T] (RGB normalised + trimap) -> [1, 1, T, T] alpha, T a multiple of 32
// (the model's size_divisibility), tiles up to 1024. BiRefNet (B3): [1, 3, 1024, 1024] (ImageNet
// mean / std) -> [1, 1, 1024, 1024] logits, whole frame. Licence ids: LICENCES.md, decision 31.
// BiRefNet's arena hints (decision 36; applied on cards <= 32 GB, cudaArenaFor), measured on BFD (RTX
// PRO 4500 32 GB, ORT 1.31, CUDA EP, 1024 x 1024, 3 frames, kSameAsRequested), process GPU memory:
// lite unlimited 14.7 GB; limit 8192 -> 7.2 GB, 7168 runs, 6144 fails (allocation error); full
// unlimited 15.8 GB; limit 10240 / 9216 / 8704 -> 9.1 GB, 8192 fails. Hints = the smallest working
// limit + about 1-1.5 GB. Speed unchanged (lite 119 ms, full 228 ms warm). ViTMatte: no limit.
const ModelSpec kModels[] = {
    {"vitmatte-small", "vitmatte-small/model.onnx", "pixel_values", "alphas", 4, 1, SizeRule::kMultipleOf, 32, 32,
     1024, 0, 0, "vitmatte-c1k", "bf28d2e0be2c073286e88d60ad649d7123da2749a2d99133fd1098d5887e0225", 0, false},
    {"vitmatte-base", "vitmatte-base/model.onnx", "pixel_values", "alphas", 4, 1, SizeRule::kMultipleOf, 32, 32,
     1024, 0, 0, "vitmatte-c1k", "f6978437f5068849bcbf49b1f4e37b90aaca5155744e9fde4e6c689f70c2b9ee", 0, false},
    {"birefnet-lite", "birefnet-lite/model.onnx", "input_image", "output_image", 3, 1, SizeRule::kFixed, 0, 0, 0,
     1024, 1024, "birefnet-mit", "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333", 8192, true},
    {"birefnet", "birefnet/model.onnx", "input_image", "output_image", 3, 1, SizeRule::kFixed, 0, 0, 0, 1024, 1024,
     "birefnet-mit", "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c", 10240, true},
};
constexpr int kCount = int(sizeof kModels / sizeof kModels[0]);

std::string num(uint64_t v) { return std::to_string(v); }

}  // namespace

int modelCount() { return kCount; }
const ModelSpec& modelAt(int i) { return kModels[i]; }

const ModelSpec* findModel(const std::string& id) {
  for (const ModelSpec& m : kModels)
    if (id == m.id) return &m;
  return nullptr;
}

CudaArena cudaArenaFor(const ModelSpec& m, long gpuTotalMiB) {
  if (m.gpuMemLimitMiB == 0 || gpuTotalMiB <= 0 || gpuTotalMiB > kSmallGpuMiB) return CudaArena{0, false};
  return CudaArena{m.gpuMemLimitMiB, m.arenaSameAsRequested};
}

uint64_t inputBytes(const TensorDesc& d) { return uint64_t(d.channelsIn) * d.h * d.w * sizeof(float); }
uint64_t outputBytes(const TensorDesc& d) { return uint64_t(d.channelsOut) * d.outH * d.outW * sizeof(float); }

uint64_t maxRunBytes() {
  uint64_t best = 0;
  for (const ModelSpec& m : kModels) {
    const uint64_t side = m.sizeRule == SizeRule::kFixed ? 0 : uint64_t(m.maxSide);
    const uint64_t px = m.sizeRule == SizeRule::kFixed ? uint64_t(m.fixedW) * m.fixedH : side * side;
    const uint64_t b = px * uint64_t(m.channelsIn + m.channelsOut) * sizeof(float);
    if (b > best) best = b;
  }
  return best;
}

std::string checkRequest(const ModelSpec& m, const TensorDesc& d) {
  const std::string id = m.id;
  if (d.channelsIn != uint32_t(m.channelsIn))
    return "model " + id + " takes " + num(m.channelsIn) + " input channels, not " + num(d.channelsIn);
  if (d.channelsOut != uint32_t(m.channelsOut))
    return "model " + id + " gives " + num(m.channelsOut) + " output channels, not " + num(d.channelsOut);
  const std::string size = num(d.w) + "x" + num(d.h);
  if (m.sizeRule == SizeRule::kFixed) {
    if (d.w != uint32_t(m.fixedW) || d.h != uint32_t(m.fixedH))
      return "model " + id + " takes " + num(m.fixedW) + "x" + num(m.fixedH) + " only, not " + size;
  } else {
    const uint32_t lo = uint32_t(m.minSide), hi = uint32_t(m.maxSide), k = uint32_t(m.multiple);
    if (d.w < lo || d.h < lo || d.w > hi || d.h > hi || d.w % k != 0 || d.h % k != 0)
      return "size " + size + " for model " + id + " (each side a multiple of " + num(k) + ", " + num(lo) + "-" +
             num(hi) + ")";
  }
  if (d.outW != d.w || d.outH != d.h)
    return "output size " + num(d.outW) + "x" + num(d.outH) + " for model " + id + " (must be the input's, " + size + ")";
  return "";
}

}  // namespace ml
}  // namespace mr
