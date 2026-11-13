// vm_run: runs every job_*.in in a folder through one ViTMatte ONNX session and writes job_*.out.
// Job file: int32 H, int32 W, then float32 [4, H, W] (RGB already normalised, trimap 0 / 0.5 / 1).
// Output: float32 [H, W] alpha. Session setup follows ofx/ml/src/ort_runner.cpp createSession.
// Build: g++ -std=c++17 -O2 vm_run.cpp -I$ORT/include -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib
// Use:   LD_LIBRARY_PATH=$ORT/lib:$CUDA_LIBS ./vm_run model.onnx jobdir [cpu]
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: vm_run model.onnx jobdir [cpu]\n");
    return 2;
  }
  const bool cpu = argc > 3 && std::string(argv[3]) == "cpu";
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "vm_run");
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (!cpu) {
    OrtCUDAProviderOptions o;
    so.AppendExecutionProvider_CUDA(o);
  }
  Ort::Session session(env, argv[1], so);
  Ort::AllocatorWithDefaultOptions alloc;
  const std::string inName = session.GetInputNameAllocated(0, alloc).get();
  const std::string outName = session.GetOutputNameAllocated(0, alloc).get();

  std::vector<std::string> jobs;
  if (DIR* d = opendir(argv[2])) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 3 && n.compare(n.size() - 3, 3, ".in") == 0) jobs.push_back(n);
    }
    closedir(d);
  }
  std::sort(jobs.begin(), jobs.end());
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> buf;
  for (const std::string& j : jobs) {
    const std::string path = std::string(argv[2]) + "/" + j;
    FILE* f = std::fopen(path.c_str(), "rb");
    int32_t hw[2];
    if (!f || std::fread(hw, 4, 2, f) != 2) return 3;
    const size_t n = size_t(4) * hw[0] * hw[1];
    buf.resize(n);
    if (std::fread(buf.data(), 4, n, f) != n) return 3;
    std::fclose(f);
    const int64_t shape[4] = {1, 4, hw[0], hw[1]};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, buf.data(), n, shape, 4);
    const char* inN = inName.c_str();
    const char* outN = outName.c_str();
    auto outs = session.Run(Ort::RunOptions{nullptr}, &inN, &in, 1, &outN, 1);
    const size_t m = size_t(hw[0]) * hw[1];
    if (outs[0].GetTensorTypeAndShapeInfo().GetElementCount() != m) return 4;
    const std::string op = path.substr(0, path.size() - 3) + ".out";
    FILE* g = std::fopen(op.c_str(), "wb");
    if (!g || std::fwrite(outs[0].GetTensorData<float>(), 4, m, g) != m) return 5;
    std::fclose(g);
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%zu jobs, %.0f ms\n", jobs.size(), ms);
  return 0;
}
