// gate.h
#include "gate.h"

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>

#include "model_table.h"

namespace mr {
namespace ml {

namespace {

const char* const kCudaLibs[] = {
    "libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10",
    "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9", "libcudnn.so.9",
};

bool isFile(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// SHA-256, FIPS 180-4.
class Sha256 {
 public:
  void update(const unsigned char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      buf_[used_++] = p[i];
      if (used_ == 64) {
        block(buf_);
        used_ = 0;
      }
    }
    bits_ += uint64_t(n) * 8;
  }
  std::string hex() {
    const uint64_t bits = bits_;
    const unsigned char pad = 0x80, zero = 0;
    update(&pad, 1);
    while (used_ != 56) update(&zero, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(len, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
    return std::string(out, 64);
  }

 private:
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const unsigned char* p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  unsigned char buf_[64] = {};
  size_t used_ = 0;
  uint64_t bits_ = 0;
};

GateResult blocked(GateState s, const char* what, const std::string& detail) {
  GateResult r;
  r.state = s;
  r.footer = std::string("Matte Refine ML is not set up (") + what +
             "): run setup.sh as your normal user (see the install notes), then restart Flame.";
  r.detail = detail;
  return r;
}

}  // namespace

int cudaLibCount() { return int(sizeof kCudaLibs / sizeof kCudaLibs[0]); }
const char* cudaLibName(int i) { return kCudaLibs[i]; }

std::string defaultModelDir(const std::string& home) { return home + "/.local/share/matte-refine-ml/models"; }

std::string sha256Hex(const std::string& bytes) {
  Sha256 s;
  s.update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
  return s.hex();
}

bool sha256File(const std::string& path, std::string* hex) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) return false;
  *hex = sha256Hex(bytes);
  return true;
}

const char* const kStandardModels[2] = {"vitmatte-small", "birefnet-lite"};

bool modelInstalled(const std::string& modelDir, const std::string& id) {
  const ModelSpec* m = findModel(id);
  return m && !modelDir.empty() && isFile(modelDir + "/" + m->file);
}

std::string optionLabel(const std::string& name, bool installed) {
  return installed ? name : name + " (not installed)";
}

std::string notInstalledFooter(const std::string& id) {
  const char* name = id == "vitmatte-small" ? "ViTMatte Small"
                     : id == "vitmatte-base" ? "ViTMatte Base"
                     : id == "birefnet-lite" ? "BiRefNet Lite"
                     : id == "birefnet"      ? "BiRefNet Full"
                                             : nullptr;
  const bool standard = id == kStandardModels[0] || id == kStandardModels[1];
  return (name ? std::string(name) : "The model " + id) + " is not installed: run setup.sh" +
         (standard ? "." : " --full.");
}

GateResult checkGate(const std::string& resourcesDir, const std::map<std::string, std::string>& config,
                     const std::string& modelDir, const std::string& cudaDir) {
  const std::string licences = resourcesDir + "/licences.txt";
  if (!isFile(licences)) return GateResult();  // a dev build
  std::string want;
  if (!sha256File(licences, &want))
    return blocked(GateState::kNoLicence, "licences not accepted", "cannot read " + licences);
  const auto it = config.find("ml_licence_accepted");
  if (it == config.end() || it->second.empty())
    return blocked(GateState::kNoLicence, "licences not accepted", "ml_licence_accepted is not in the config");
  if (it->second != want)
    return blocked(GateState::kNoLicence, "licences not accepted",
                   "ml_licence_accepted " + it->second + " is not the sha256 of " + licences + " (" + want + ")");
  std::string missing;
  for (const char* id : kStandardModels)
    if (!modelInstalled(modelDir, id)) missing += (missing.empty() ? "" : ", ") + modelDir + "/" + id + "/model.onnx";
  if (!missing.empty()) return blocked(GateState::kNoModels, "model files missing", "missing: " + missing);
  if (cudaDir.empty())
    return blocked(GateState::kNoCudaLibs, "CUDA libraries missing", "ml_cuda_libs is not in the config");
  for (int i = 0; i < cudaLibCount(); ++i) {
    const std::string path = cudaDir + "/" + kCudaLibs[i];
    if (!isFile(path)) missing += (missing.empty() ? "" : ", ") + path;
  }
  if (!missing.empty()) return blocked(GateState::kNoCudaLibs, "CUDA libraries missing", "missing: " + missing);
  GateResult r;
  r.state = GateState::kOpen;
  return r;
}

}  // namespace ml
}  // namespace mr
