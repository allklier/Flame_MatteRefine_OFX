// Loads the built EdgeRecipes.ofx the way a host does (dlopen RTLD_NOW + the two OFX entry points; catches
// unresolved symbols) and checks the plugin table: the product exports exactly Hair Locks; the test build
// exports Hair Locks + Test Dummy with distinct ids and versions. Decision 32: no ONNX Runtime or CUDA library may
// be mapped after loading.
// Usage: test_load_bundle_recipes <EdgeRecipes.ofx> <EdgeRecipesTest.ofx>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "ofxCore.h"
#include "ofxImageEffect.h"

namespace {
struct Want {
  const char* id;
  unsigned major, minor;
};

bool checkTable(const char* path, const Want* want, int n) {
  void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    std::printf("FAIL dlopen %s: %s\n", path, dlerror());
    return false;
  }
  auto count = reinterpret_cast<int (*)()>(dlsym(lib, "OfxGetNumberOfPlugins"));
  auto get = reinterpret_cast<OfxPlugin* (*)(int)>(dlsym(lib, "OfxGetPlugin"));
  if (!count || !get) {
    std::printf("FAIL entry points missing in %s\n", path);
    return false;
  }
  bool ok = count() == n;
  if (!ok) std::printf("FAIL %s: %d plugins, expected %d\n", path, count(), n);
  for (int i = 0; ok && i < n; ++i) {
    OfxPlugin* p = get(i);
    if (!p || std::strcmp(p->pluginIdentifier, want[i].id) != 0 || std::strcmp(p->pluginApi, kOfxImageEffectPluginApi) ||
        p->pluginVersionMajor != want[i].major || p->pluginVersionMinor != want[i].minor || !p->mainEntry) {
      std::printf("FAIL %s plugin %d: %s v%u.%u\n", path, i, p ? p->pluginIdentifier : "(null)",
                  p ? p->pluginVersionMajor : 0, p ? p->pluginVersionMinor : 0);
      ok = false;
    } else {
      std::printf("  %d: %s v%u.%u\n", i, p->pluginIdentifier, p->pluginVersionMajor, p->pluginVersionMinor);
    }
  }
  if (n == 2 && ok && get(0)->mainEntry == get(1)->mainEntry) {
    std::printf("FAIL %s: two plugins share one main entry (one factory class)\n", path);
    ok = false;
  }
  return ok;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const Want product[] = {{"com.janklier.edgerecipes.hairlocks", 1, 0}};
  const Want test[] = {{"com.janklier.edgerecipes.hairlocks", 1, 0}, {"com.janklier.edgerecipes.testdummy", 2, 3}};
  bool ok = checkTable(argv[1], product, 1);
  ok = checkTable(argv[2], test, 2) && ok;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line))
    for (const char* bad : {"libcuda", "libcublas", "libcudnn", "providers_cuda", "libonnxruntime"})
      if (line.find(bad) != std::string::npos) {
        std::printf("FAIL loaded with the bundle (decision 32 forbids it): %s\n", line.c_str());
        ok = false;
      }
  std::printf("test_load_bundle_recipes: %s\n", ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}
