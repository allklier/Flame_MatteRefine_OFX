// Loads the built MatteRefineML.ofx the way a host does (dlopen RTLD_NOW + the two OFX entry
// points; catches unresolved symbols) and checks the plugin identity. Decision 32: neither ONNX
// Runtime nor any CUDA library may be mapped after loading it (they live in mr_ml_runner).
// Usage: test_load_bundle_ml <path to MatteRefineML.ofx>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "ofxCore.h"
#include "ofxImageEffect.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  auto count = reinterpret_cast<int (*)()>(dlsym(lib, "OfxGetNumberOfPlugins"));
  auto get = reinterpret_cast<OfxPlugin* (*)(int)>(dlsym(lib, "OfxGetPlugin"));
  if (!count || !get) {
    std::fprintf(stderr, "entry points missing\n");
    return 1;
  }
  const int n = count();
  OfxPlugin* p = n > 0 ? get(0) : nullptr;
  if (n != 1 || !p || std::strcmp(p->pluginIdentifier, "com.janklier.matterefineml") != 0 ||
      std::strcmp(p->pluginApi, kOfxImageEffectPluginApi) != 0) {
    std::fprintf(stderr, "unexpected plugin table (n=%d)\n", n);
    return 1;
  }
  std::ifstream maps("/proc/self/maps");
  std::string line;
  for (const char* bad : {"libcuda", "libcublas", "libcudnn", "libcurand", "libnvrtc", "providers_cuda",
                          "libonnxruntime"}) {
    maps.clear();
    maps.seekg(0);
    while (std::getline(maps, line))
      if (line.find(bad) != std::string::npos) {
        std::fprintf(stderr, "loaded with the bundle (decision 32 forbids it): %s\n", line.c_str());
        return 1;
      }
  }
  std::printf("test_load_bundle_ml: %s api=%s v%u.%u loaded OK (no ONNX Runtime, no CUDA mapped)\n",
              p->pluginIdentifier, p->pluginApi, p->pluginVersionMajor, p->pluginVersionMinor);
  return 0;
}
