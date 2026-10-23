// The plugin's side of mr_ml_runner (ofx/ml/CLAUDE.md decision 32): sends model runs to the shared
// runner process over a Unix socket, the tensors in shared memory (runner_protocol.h, protocol 2).
// C++14; no ONNX Runtime, no CUDA, nothing GPU is loaded into the host.
//
// The generic call is run(modelId, in, cIn, H, W, out, cOut, Ho, Wo, x, y) for any model of the
// table (model_table.h); the caller does the model's pre / post processing. The TileRunner
// interface (edge_refine.h, ViTMatte tiles) and runFrame (whole frame, B3 later) sit on top of it.
//
// One RunnerClient per plugin instance (one connection, one slot of 20 MiB that grows to fit a
// larger run, up to 256 MiB, by reconnecting with a new one; Flame renders an instance
// on one thread at a time, so the client needs no lock; not thread-safe). The first tile connects;
// if no runner answers, it starts one under a lock file (so simultaneous nodes start one), detached
// (posix_spawn, own session; the runner daemonises itself and reports when it listens) with an
// explicit environment (no LD_LIBRARY_PATH / LD_PRELOAD from the host). A runner of another
// protocol or build is told to QUIT and replaced, once. A broken connection is retried once
// (reconnect, restarting the runner if it died); a tile without an answer within the timeout is an
// error (no retry: it would double the wait); then a STATUS on a new connection (probeTimeoutS) tells
// a slow runner from a hung one, and a hung one is killed (its pid checked against the socket's peer
// credentials), so the next frame starts a fresh runner instead of waiting out the timeout again.
// Errors throw std::runtime_error with a footer text.
#pragma once

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "runner_protocol.h"

namespace mr {
namespace ml {

// The build id both sides compare at the handshake ("<MRML_VERSION> <MR_GIT>", from CMake).
const char* runnerBuildId();

struct RunnerClientOptions {
  std::string runnerPath;     // the mr_ml_runner executable
  std::string socketPath;     // "" = ipc::defaultSocketPath(buildId): one socket per build
  std::string buildId;        // "" = runnerBuildId() (tests: another build)
  std::string name = "client";  // for the runner's log
  double tileTimeoutS = 20;   // per tile, the model call and any queue wait included
  double startTimeoutS = 15;  // starting the runner + the handshake
  // After a tile timeout: a STATUS on a new connection must be answered within this, else the
  // runner counts as hung and is stopped (SIGKILL, its pid verified by the socket's peer credentials).
  double probeTimeoutS = 1;
  int idleSeconds = 0;        // passed to a runner this client starts; 0 = its default (300 s)
  std::vector<std::string> env;        // "K=V" added to the runner's environment (tests: HOME)
  std::vector<std::string> extraArgs;  // added to the runner's command line (tests)
  uint64_t slotBytes = proto::kDefaultSlotBytes;  // the first slot's size (grows when a run needs more)
};

// Accumulated since resetStats() (the plugin resets per render).
struct RunnerClientStats {
  int tiles = 0;
  double roundTripMs = 0;  // send RUN .. reply received
  double copyMs = 0;       // the tile into and out of the slot
  double runMs = 0, queueMs = 0;  // the runner's own times
  bool sessionCreated = false;
  double loadMs = 0;
  std::string ep, why;     // the EP that ran the last tile; why it is not the requested one
  int reconnects = 0, runnerStarts = 0, replaced = 0, slotGrown = 0;
  int hungStopped = 0;     // runners stopped after a tile timeout (unanswered STATUS)
  double connectMs = 0;    // connecting (starting the runner included)
};

class RunnerClient : public TileRunner {
 public:
  explicit RunnerClient(const RunnerClientOptions& o);
  ~RunnerClient() override;
  RunnerClient(const RunnerClient&) = delete;
  RunnerClient& operator=(const RunnerClient&) = delete;

  // The model and EP of the TileRunner calls (a ViTMatte id: 4 channels in, 1 out).
  void setModel(const std::string& modelId, uint32_t ep = proto::kEpDefault) {
    model_ = modelId;
    ep_ = ep;
  }
  // One model run through the runner: in = [cIn, h, w] float32, out = [cOut, oh, ow] float32
  // (dense, NCHW without N); x, y = its place in the frame (information). The runner checks the
  // shapes against the model table. Throws std::runtime_error (the footer text).
  void run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut, int oh, int ow,
           int x, int y);
  // TileRunner: one ViTMatte tile ([4, T, T] -> [T, T]) of setModel's model.
  void run(const float* input, int size, float* alpha, int x, int y) override;
  // A whole-frame run (B3, BiRefNet: [3, H, W] -> [1, H, W] at the model's size); no tiling.
  void runFrame(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut);

  // STATUS of the runner (connects, and starts it only if start). False with *err.
  bool status(proto::StatusReply* out, std::string* err, bool start = false);
  // Asks the runner to exit and waits until it has (true also when none was running).
  bool quit(std::string* err);
  void disconnect();
  bool connected() const { return fd_ >= 0; }
  int runnerPid() const { return runnerPid_; }
  std::string runnerBuild() const { return runnerBuild_; }

  const RunnerClientStats& stats() const { return stats_; }
  void resetStats() { stats_ = RunnerClientStats(); }
  const std::string& socketPath();  // throws if there is none
  size_t slotBytes() const { return slotBytes_; }  // the current slot (0 before the first run)

 private:
  enum class Hello { Ok, Mismatch, Failed };
  void ensureConnected(bool start);
  void ensureSlot(uint64_t need);
  Hello handshake(int fd, std::string* why);
  void startRunner();
  bool waitGone(int fd, int64_t deadline);
  std::string stopIfHung();

  RunnerClientOptions o_;
  std::string socket_;
  int fd_ = -1, slotFd_ = -1;
  void* slot_ = nullptr;
  size_t slotBytes_ = 0;
  uint64_t slotWant_ = 0;
  uint32_t seq_ = 0, ep_ = proto::kEpDefault;
  std::string model_ = "vitmatte-small";
  int runnerPid_ = -1;
  std::string runnerBuild_;
  RunnerClientStats stats_;
};

}  // namespace ml
}  // namespace mr
