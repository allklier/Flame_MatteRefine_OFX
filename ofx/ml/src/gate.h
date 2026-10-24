// The setup gate of Matte Refine ML (root decision 38; LTX Alpha Gen's decision 38 is the model): the
// node downloads nothing. In a RELEASE bundle (it has Contents/Resources/licences.txt) the node runs
// only when the user's setup (packaging/setup.sh) is complete:
//   - config ml_licence_accepted == the sha256 (lower-case hex) of that licences.txt
//     (setup.sh writes it after the user accepts; a changed licences.txt asks again),
//   - the standard models (kStandardModels: what setup.sh always installs) are in the model folder,
//   - every CUDA library the runner opens (cudaLibName) is in the ml_cuda_libs folder.
// Otherwise every step is off (the matte passes through) and the footer says what is missing. A DEV
// build (no licences.txt in the bundle) is never gated. Separately, in any build, a selected model
// that is not installed (the full models are optional) turns its own step off (notInstalledFooter),
// and the model choices' labels say which are installed (optionLabel). Pure: files and strings only,
// no OFX, no ONNX Runtime; C++14. Tests: ofx/ml/tests/test_gate.cpp.
#pragma once

#include <map>
#include <string>

namespace mr {
namespace ml {

// The CUDA 13 / cuDNN 9 libraries the runner dlopens from ml_cuda_libs, in dependency order
// (ort_runner.cpp; the minimal set, packaging/setup.sh installs exactly these).
int cudaLibCount();
const char* cudaLibName(int i);

// Where packaging/setup.sh puts the models by default; also the runner's default ml_model_dir.
std::string defaultModelDir(const std::string& home);  // <home>/.local/share/matte-refine-ml/models

// SHA-256 (FIPS 180-4) as 64 lower-case hex digits.
std::string sha256Hex(const std::string& bytes);
// Of a file's contents; false if it cannot be read.
bool sha256File(const std::string& path, std::string* hex);

enum class GateState {
  kDev,          // no licences.txt in the bundle: not gated
  kOpen,         // release bundle, everything in place
  kNoLicence,    // ml_licence_accepted missing, or not the hash of the bundle's licences.txt
  kNoModels,     // a standard model file is missing (kStandardModels)
  kNoCudaLibs,   // ml_cuda_libs unset, or a library missing in it
};

struct GateResult {
  GateState state = GateState::kDev;
  bool open() const { return state == GateState::kDev || state == GateState::kOpen; }
  std::string footer;  // for the node: one sentence naming the missing part ("" when open)
  std::string detail;  // for the log: what exactly is missing (paths, hashes)
};

// resourcesDir: <bundle>/Contents/Resources. config: the user config (ml_licence_accepted is read
// from it). modelDir, cudaDir: the resolved ml_model_dir and ml_cuda_libs ("" = unset). Checks in
// this order: licence, standard models, CUDA libraries; the first failing one is reported.
GateResult checkGate(const std::string& resourcesDir, const std::map<std::string, std::string>& config,
                     const std::string& modelDir, const std::string& cudaDir);

// The models setup.sh always installs; the others (vitmatte-base, birefnet) only with --full.
extern const char* const kStandardModels[2];  // vitmatte-small, birefnet-lite

// <modelDir>/<the model_table.h file of id> is a file (false for an unknown id or an empty modelDir).
bool modelInstalled(const std::string& modelDir, const std::string& id);
// A model choice's label: "Base", or "Base (not installed)".
std::string optionLabel(const std::string& name, bool installed);
// The footer when a selected model is missing, e.g. "ViTMatte Base is not installed: run setup.sh --full."
// (a standard model: "run setup.sh.").
std::string notInstalledFooter(const std::string& id);

}  // namespace ml
}  // namespace mr
