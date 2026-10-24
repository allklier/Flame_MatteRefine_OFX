// The setup gate (gate.h): SHA-256 against the FIPS 180-2 examples, and checkGate on a temporary
// bundle / model / CUDA layout: dev bundle, gate open, licence missing / empty / wrong hash, a standard
// model missing, CUDA folder unset, a CUDA library missing, the order of the checks; modelInstalled,
// the choice labels and the "not installed" footers.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "gate.h"

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

void write(const std::string& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void testSha256() {
  CHECK(sha256Hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(sha256Hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu") ==
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  CHECK(sha256Hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

}  // namespace

int main() {
  testSha256();

  char tmpl[] = "/tmp/mrml-gate-XXXXXX";
  const std::string root = mkdtemp(tmpl);
  const std::string res = root + "/Contents/Resources", models = root + "/models", cuda = root + "/cuda_libs";
  for (const std::string d : {root + "/Contents", res, models, models + "/vitmatte-small", models + "/birefnet-lite", cuda})
    mkdir(d.c_str(), 0755);
  write(models + "/vitmatte-small/model.onnx", "x");
  write(models + "/birefnet-lite/model.onnx", "x");
  for (int i = 0; i < cudaLibCount(); ++i) write(cuda + "/" + cudaLibName(i), "x");
  CHECK(cudaLibCount() == 11);

  // A dev bundle (no licences.txt): never gated, even with nothing set up.
  GateResult g = checkGate(res, {}, "", "");
  CHECK(g.state == GateState::kDev && g.open() && g.footer.empty());

  // A release bundle.
  const std::string text = "THE LICENCES\nline 2\n";
  write(res + "/licences.txt", text);
  std::string hash;
  CHECK(sha256File(res + "/licences.txt", &hash) && hash == sha256Hex(text));
  CHECK(!sha256File(root + "/nope.txt", &hash));
  std::map<std::string, std::string> cfg = {{"log_level", "debug"}, {"ml_licence_accepted", sha256Hex(text)}};

  g = checkGate(res, cfg, models, cuda);
  CHECK(g.state == GateState::kOpen && g.open() && g.footer.empty());
  // Licence: missing, empty, another file's hash (licences.txt changed since the user accepted).
  std::map<std::string, std::string> c2 = cfg;
  c2.erase("ml_licence_accepted");
  g = checkGate(res, c2, models, cuda);
  CHECK(g.state == GateState::kNoLicence && !g.open());
  CHECK(has(g.footer, "licences not accepted") && has(g.footer, "setup.sh") && has(g.footer, "restart Flame"));
  c2["ml_licence_accepted"] = "";
  g = checkGate(res, c2, models, cuda);
  CHECK(g.state == GateState::kNoLicence && has(g.detail, "not in the config"));
  c2["ml_licence_accepted"] = sha256Hex("THE LICENCES\nline 2 changed\n");
  g = checkGate(res, c2, models, cuda);
  CHECK(g.state == GateState::kNoLicence && has(g.detail, "is not the sha256"));
  c2["ml_licence_accepted"] = sha256Hex(text) + " ";  // keys and values are not trimmed (user_config.h)
  CHECK(checkGate(res, c2, models, cuda).state == GateState::kNoLicence);
  // The licence is checked first: with nothing else in place it is still the licence that is reported.
  CHECK(checkGate(res, {}, "", "").state == GateState::kNoLicence);

  // Standard models: either one missing, no model folder; the full models do not count.
  CHECK(!modelInstalled(models, "vitmatte-base") && !modelInstalled(models, "birefnet"));
  unlink((models + "/birefnet-lite/model.onnx").c_str());
  g = checkGate(res, cfg, models, cuda);
  CHECK(g.state == GateState::kNoModels && has(g.footer, "model files missing") &&
        has(g.detail, "birefnet-lite/model.onnx") && !has(g.detail, "vitmatte-small"));
  // Models before CUDA.
  CHECK(checkGate(res, cfg, models, "").state == GateState::kNoModels);
  write(models + "/birefnet-lite/model.onnx", "x");
  unlink((models + "/vitmatte-small/model.onnx").c_str());
  g = checkGate(res, cfg, models, cuda);
  CHECK(g.state == GateState::kNoModels && has(g.detail, "vitmatte-small/model.onnx"));
  write(models + "/vitmatte-small/model.onnx", "x");
  CHECK(checkGate(res, cfg, "", cuda).state == GateState::kNoModels);
  CHECK(checkGate(res, cfg, models, cuda).open());

  // CUDA libraries: folder unset, one library missing.
  g = checkGate(res, cfg, models, "");
  CHECK(g.state == GateState::kNoCudaLibs && has(g.footer, "CUDA libraries missing") &&
        has(g.detail, "ml_cuda_libs is not in the config"));
  unlink((cuda + "/libcudnn_engines_tensor_ir.so.9").c_str());
  g = checkGate(res, cfg, models, cuda);
  CHECK(g.state == GateState::kNoCudaLibs && has(g.detail, "libcudnn_engines_tensor_ir.so.9") &&
        !has(g.detail, "libcudart"));
  write(cuda + "/libcudnn_engines_tensor_ir.so.9", "x");
  CHECK(checkGate(res, cfg, models, cuda).open());

  // modelInstalled: the table's file, any build; an unknown id, an empty folder, a folder named model.onnx.
  CHECK(modelInstalled(models, "vitmatte-small") && modelInstalled(models, "birefnet-lite"));
  CHECK(!modelInstalled(models, "no-such-model") && !modelInstalled("", "vitmatte-small"));
  mkdir((models + "/birefnet").c_str(), 0755);
  mkdir((models + "/birefnet/model.onnx").c_str(), 0755);
  CHECK(!modelInstalled(models, "birefnet"));
  mkdir((models + "/vitmatte-base").c_str(), 0755);
  write(models + "/vitmatte-base/model.onnx", "x");
  CHECK(modelInstalled(models, "vitmatte-base"));
  // The choices' labels and the step's footer.
  CHECK(optionLabel("Base", true) == "Base" && optionLabel("Full", false) == "Full (not installed)");
  CHECK(notInstalledFooter("vitmatte-base") == "ViTMatte Base is not installed: run setup.sh --full.");
  CHECK(notInstalledFooter("birefnet") == "BiRefNet Full is not installed: run setup.sh --full.");
  CHECK(notInstalledFooter("vitmatte-small") == "ViTMatte Small is not installed: run setup.sh.");
  CHECK(notInstalledFooter("birefnet-lite") == "BiRefNet Lite is not installed: run setup.sh.");
  CHECK(std::string(kStandardModels[0]) == "vitmatte-small" && std::string(kStandardModels[1]) == "birefnet-lite");

  // Back to a dev bundle once licences.txt is gone.
  unlink((res + "/licences.txt").c_str());
  CHECK(checkGate(res, {}, "", "").state == GateState::kDev);

  if (std::system(("rm -rf '" + root + "'").c_str()) != 0) std::printf("note: %s not removed\n", root.c_str());
  if (failures) {
    std::printf("test_gate: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("test_gate: all passed\n");
  return 0;
}
