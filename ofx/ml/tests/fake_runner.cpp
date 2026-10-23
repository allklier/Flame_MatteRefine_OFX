// A fake mr_ml_runner for test_runner_client: the real server (runner_server.cpp, so the real
// request validation against the model table) with a model that answers any table model's shapes:
// output plane c = a function of the run's frame position + the input's first plane + its last
// plane + c, so every run's answer is known exactly and shows that the whole input arrived and the
// output went to the right place in the slot. Behaviour switches are files in the socket's folder:
//   fake.mode     read at every RUN: normal | crash | crash-once | hang | slow:<ms> | error | seg
//                 (seg: a segmentation, see segLogit; any other mode answers as above)
//   fake.build    read at start: the build id to report (absent / empty = the real one)
//   fake.version  read at start: the protocol version to report (absent = the real one)
// --build <id> on the command line: the build id to report (overrides fake.build; two builds at once).
// Each start appends "<pid>" to fake.starts and its environment to fake.env.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"
#include "runner_ipc.h"
#include "runner_server.h"

extern char** environ;

using namespace mr::ml;

namespace {

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s;
}

// Mode seg (test_render_chain): a BiRefNet-shaped answer from the picture. The input's plane 0 is the
// red channel normalised by birefnetPrepare ((r - 0.485) / 0.229); logit = 40 (r - 0.5): subject where
// the picture is red. Every output channel the same.
float segLogit(float in0) { return 40.0f * (in0 * 0.229f + 0.485f - 0.5f); }

struct FakeEngine : Engine {
  std::string dir;
  uint64_t runs = 0;
  std::string lastModel;
  RunResult run(const ModelSpec& m, const proto::RunRequest& q, const float* in, float* out) override {
    const int64_t t0 = ipc::nowUs();
    const std::string mode = readFile(dir + "/fake.mode");
    if (mode == "crash") _exit(9);
    if (mode == "crash-once" && access((dir + "/fake.crashed").c_str(), F_OK) != 0) {
      std::ofstream(dir + "/fake.crashed") << getpid() << "\n";
      _exit(9);
    }
    if (mode == "hang") sleep(3600);
    if (mode.compare(0, 5, "slow:") == 0) usleep(useconds_t(std::atoi(mode.c_str() + 5)) * 1000);
    RunResult r;
    r.ep = "fake";
    r.runMs = double(ipc::nowUs() - t0) / 1000.0;
    if (mode == "error") {
      r.status = proto::kModelError;
      r.message = "fake model failure";
      return r;
    }
    const size_t plane = size_t(q.h) * q.w, last = (q.cIn - 1) * plane;
    if (mode == "seg") {
      for (uint32_t c = 0; c < q.cOut; ++c)
        for (size_t i = 0; i < plane; ++i) out[c * plane + i] = segLogit(in[i]);
      ++runs;
      lastModel = m.id;
      r.runMs = double(ipc::nowUs() - t0) / 1000.0;
      return r;
    }
    for (uint32_t c = 0; c < q.cOut; ++c)
      for (uint32_t ly = 0; ly < q.h; ++ly)
        for (uint32_t lx = 0; lx < q.w; ++lx) {
          const size_t i = size_t(ly) * q.w + lx;
          out[c * plane + i] = float(((q.x + int(lx)) * 31 + (q.y + int(ly)) * 17) % 1009) / 1009.0f + in[i] +
                               in[last + i] + float(c);
        }
    ++runs;
    lastModel = m.id;
    r.runMs = double(ipc::nowUs() - t0) / 1000.0;
    return r;
  }
  int sessions(proto::SessionInfo* out, int cap) override {
    if (cap > 0) {
      out[0] = proto::SessionInfo{};
      out[0].runs = runs;
      ipc::setField(out[0].model, lastModel);
      ipc::setField(out[0].activeEp, "fake");
    }
    return 1;
  }
};

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--daemon") == 0) closeInheritedFds();
  mr::setLogComponent("runner");  // as mr_ml_runner
  ServerOptions o;
  std::vector<std::string> rest;
  std::string err;
  if (!parseServerArgs(argc, argv, &o, &rest, &err) || o.socketPath.empty()) {
    std::fprintf(stderr, "fake_runner: %s\n", err.empty() ? "--socket needed" : err.c_str());
    return 1;
  }
  FakeEngine e;
  e.dir = ipc::dirName(o.socketPath);
  o.buildId = readFile(e.dir + "/fake.build");
  for (size_t i = 0; i + 1 < rest.size(); ++i)
    if (rest[i] == "--build") o.buildId = rest[i + 1];
  const std::string v = readFile(e.dir + "/fake.version");
  if (!v.empty()) o.version = uint32_t(std::atoi(v.c_str()));
  std::ofstream(e.dir + "/fake.starts", std::ios::app) << getpid() << "\n";
  {
    std::ofstream env(e.dir + "/fake.env");
    for (char** p = environ; *p; ++p) env << *p << "\n";
  }
  return serve(o, e);
}
