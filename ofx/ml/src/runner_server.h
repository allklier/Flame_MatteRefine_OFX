// The runner's server side without a model (decision 32): socket, handshake, one thread per
// connection, the shared-memory slots, the FIFO queue for the model, STATUS / QUIT, idle exit,
// SIGTERM, daemonising. C++14, no ONNX Runtime: mr_ml_runner plugs in the ORT engine
// (runner_main.cpp), the tests a fake one (tests/fake_runner.cpp).
//
// Queue: every model call (RUN, and STATUS's session listing) takes a ticket lock: callers are
// served strictly in arrival order, one at a time (one tile on the GPU at a time; a busy client
// cannot starve another: tiles of two clients interleave). A request's queueUs is its wait.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model_table.h"
#include "runner_protocol.h"

namespace mr {
namespace ml {

struct RunResult {
  uint32_t status = proto::kOk;
  bool created = false;  // the session was created for this request
  double runMs = 0, loadMs = 0;
  std::string ep;        // "cuda" / "cpu"
  std::string message;   // the error, or why the EP is not the requested one
};

// The model behind the server. Called one at a time (the queue), from connection threads.
class Engine {
 public:
  virtual ~Engine() {}
  // One validated request (validateRun): q names model m; in = [q.cIn, q.h, q.w], out =
  // [q.cOut, q.outH, q.outW] (both in the client's slot). Must not throw (the server catches and
  // reports anything that does as a model error on that connection only).
  virtual RunResult run(const ModelSpec& m, const proto::RunRequest& q, const float* in, float* out) = 0;
  // The live sessions for STATUS; returns how many there are (fills at most cap).
  virtual int sessions(proto::SessionInfo* out, int cap) {
    (void)out;
    (void)cap;
    return 0;
  }
};

struct ServerOptions {
  std::string socketPath;  // "" = ipc::defaultSocketPath(buildId)
  int idleSeconds = 300;   // exit after this long without a connection
  bool daemon = false;     // fork, setsid, report "listening" to the launcher, stderr to a log
  std::string buildId;     // "" = runnerBuildId()
  uint32_t version = proto::kVersion;  // tests: pretend another protocol
};

// The server's check of a RUN against the model table and the connection's slot (slotBytes = 0:
// no slot). Returns the model, or nullptr with *why. Never reads beyond q.
const ModelSpec* validateRun(const proto::RunRequest& q, uint64_t slotBytes, std::string* why);

// --socket P, --idle-s N, --daemon into *o; the rest into *rest. False with *err on a bad value.
bool parseServerArgs(int argc, char** argv, ServerOptions* o, std::vector<std::string>* rest, std::string* err);

// Closes every fd above 2 (a runner started by a host may inherit the host's fds). Call first.
void closeInheritedFds();

// Serves until QUIT, SIGTERM / SIGINT or idle. Exit codes: 0 done, 3 another runner already
// listens at the socket, 1 failure. With daemon: the launcher process returns the same code as
// soon as the daemon listens (or failed), the daemon keeps serving.
// The daemon's stdout / stderr go to <logDir>/runner-stderr.log, emptied at a start when larger.
constexpr uint64_t kStderrCapBytes = uint64_t(5) << 20;
int serve(const ServerOptions& o, Engine& engine);

// "status" / "quit" against the runner at socket: prints a report, returns 0 / 1.
int controlMain(const std::string& cmd, const std::string& socket);

}  // namespace ml
}  // namespace mr
