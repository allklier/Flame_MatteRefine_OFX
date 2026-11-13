// CropRunner over plugin 2's RunnerClient (ofx/ml/src/runner_client.h:83, the generic run(modelId, ...)): the
// crops go to the shared mr_ml_runner process (decision 32); no ONNX Runtime in this code (C++14).
#pragma once

#include "model_candidate.h"
#include "runner_client.h"

namespace mr {
namespace recipe {

class RunnerClientCrops : public CropRunner {
 public:
  explicit RunnerClientCrops(ml::RunnerClient& c) : c_(c) {}
  void run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut, int x,
           int y) override {
    c_.run(modelId, in, cIn, h, w, out, cOut, h, w, x, y);
  }

 private:
  ml::RunnerClient& c_;
};

}  // namespace recipe
}  // namespace mr
