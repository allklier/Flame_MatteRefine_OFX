// runner_client.h. C++14, POSIX (posix_spawn, flock, Unix sockets); no GPU code.
#include "runner_client.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "log.h"
#include "runner_ipc.h"

#ifndef MRML_BUILD_ID  // from ofx/ml/CMakeLists.txt
#define MRML_BUILD_ID "dev"
#endif

extern char** environ;

namespace mr {
namespace ml {

using ipc::Io;
using ipc::nowMs;

const char* runnerBuildId() { return MRML_BUILD_ID; }

namespace {

double msSince(int64_t t0Us) { return double(ipc::nowUs() - t0Us) / 1000.0; }

std::runtime_error fail(const std::string& what) { return std::runtime_error(what); }

// The runner's environment: a fixed set, never the host's loader variables (decision 32: Flame's
// LD_LIBRARY_PATH would point the runner at Flame's ONNX Runtime / CUDA; LD_PRELOAD likewise).
std::vector<std::string> runnerEnv(const std::vector<std::string>& extra) {
  std::vector<std::string> env = {"PATH=/usr/bin:/bin"};
  for (const char* k : {"HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "MR_LOG_LEVEL", "CUDA_VISIBLE_DEVICES", "TMPDIR"}) {
    const char* v = std::getenv(k);
    if (v) env.push_back(std::string(k) + "=" + v);
  }
  for (const std::string& kv : extra) {
    const std::string key = kv.substr(0, kv.find('=') + 1);
    for (auto it = env.begin(); it != env.end();)
      it = it->compare(0, key.size(), key) == 0 ? env.erase(it) : it + 1;
    env.push_back(kv);
  }
  return env;
}

std::vector<char*> cstrs(std::vector<std::string>& v) {
  std::vector<char*> p;
  for (auto& s : v) p.push_back(&s[0]);
  p.push_back(nullptr);
  return p;
}

}  // namespace

RunnerClient::RunnerClient(const RunnerClientOptions& o) : o_(o) {
  if (o_.buildId.empty()) o_.buildId = runnerBuildId();
  slotWant_ = o_.slotBytes;
}

RunnerClient::~RunnerClient() {
  disconnect();
  ipc::unmapSlot(slot_, slotBytes_);
  if (slotFd_ >= 0) close(slotFd_);
}

void RunnerClient::disconnect() {
  if (fd_ >= 0) close(fd_);
  fd_ = -1;
}

const std::string& RunnerClient::socketPath() {
  if (socket_.empty()) {
    std::string err;
    socket_ = o_.socketPath.empty() ? ipc::defaultSocketPath(&err, o_.buildId) : o_.socketPath;
    if (socket_.empty()) throw fail("runner socket: " + err);
  }
  return socket_;
}

RunnerClient::Hello RunnerClient::handshake(int fd, std::string* why) {
  const int64_t deadline = nowMs() + int64_t(o_.startTimeoutS * 1000);
  proto::Hello h{};
  h.version = proto::kVersion;
  h.pid = uint32_t(getpid());
  h.slotBytes = slotBytes_;
  ipc::setField(h.buildId, o_.buildId);
  ipc::setField(h.client, o_.name);
  Io r = ipc::sendMsg(fd, proto::kHello, ++seq_, &h, sizeof h, deadline, slotFd_);
  proto::MsgHeader hdr{};
  char buf[proto::kMaxPayload];
  if (r == Io::Ok) r = ipc::recvMsg(fd, &hdr, buf, sizeof buf, deadline);
  if (r != Io::Ok) {
    *why = ipc::ioName(r);
    return Hello::Failed;
  }
  proto::HelloReply rep{};
  const size_t stable = offsetof(proto::HelloReply, status);  // the fields every version has
  if (hdr.type != proto::kHelloReply || hdr.bytes < stable) {
    *why = "unexpected answer to HELLO (type " + std::to_string(hdr.type) + ")";
    return Hello::Failed;
  }
  std::memcpy(&rep, buf, hdr.bytes < sizeof rep ? hdr.bytes : sizeof rep);
  runnerPid_ = int(rep.pid);
  runnerBuild_ = ipc::getField(rep.buildId);
  if (rep.version != proto::kVersion || runnerBuild_ != o_.buildId) {
    *why = "runner protocol " + std::to_string(rep.version) + " build '" + runnerBuild_ + "', plugin protocol " +
           std::to_string(proto::kVersion) + " build '" + o_.buildId + "'";
    return Hello::Mismatch;
  }
  if (hdr.bytes != sizeof rep || rep.status != proto::kOk) {
    *why = "runner refused the connection: " + ipc::getField(rep.message);
    return Hello::Failed;
  }
  return Hello::Ok;
}

// QUIT on fd, then wait for the runner to close it and for the socket to stop answering.
bool RunnerClient::waitGone(int fd, int64_t deadline) {
  if (ipc::sendMsg(fd, proto::kQuit, ++seq_, nullptr, 0, deadline) != Io::Ok) return false;
  proto::MsgHeader h{};
  char buf[proto::kMaxPayload];
  Io r;
  while ((r = ipc::recvMsg(fd, &h, buf, sizeof buf, deadline)) == Io::Ok) {
  }
  if (r == Io::Timeout) return false;
  for (;;) {
    int e = 0;
    const int c = ipc::connectUnix(socket_, &e, nullptr);
    if (c < 0 && (e == ENOENT || e == ECONNREFUSED)) return true;
    if (c >= 0) close(c);
    if (nowMs() > deadline) return false;
    usleep(10000);
  }
}

// After a tile timeout. A runner whose model call is merely slow answers STATUS once the call ends
// (STATUS waits in the model queue); one that is stuck holds the queue and would cost every frame of
// every node the full timeout. Not answering within probeTimeoutS = hung: SIGKILL, but only the pid
// it reported at the handshake, and only if the process listening at our socket (SO_PEERCRED of a
// new connection: the listener's credentials) is that pid and this user. Returns the footer's
// addition ("" when nothing was stopped).
std::string RunnerClient::stopIfHung() {
  const int pid = runnerPid_;
  if (pid <= 1 || socket_.empty()) return "";
  int e = 0;
  const int fd = ipc::connectUnix(socket_, &e, nullptr);
  if (fd < 0) return "";  // gone already: the next call starts a new one
  uid_t uid = 0;
  pid_t peer = -1;
  const bool cred = ipc::peerUid(fd, &uid, &peer);
  const int64_t t0 = nowMs(), deadline = t0 + int64_t(o_.probeTimeoutS * 1000);
  proto::Hello h{};
  h.version = proto::kVersion;
  h.pid = uint32_t(getpid());
  ipc::setField(h.buildId, o_.buildId);
  ipc::setField(h.client, o_.name + ":probe");
  proto::MsgHeader hdr{};
  char buf[proto::kMaxPayload];
  Io r = ipc::sendMsg(fd, proto::kHello, ++seq_, &h, sizeof h, deadline);  // no slot: STATUS needs none
  if (r == Io::Ok) r = ipc::recvMsg(fd, &hdr, buf, sizeof buf, deadline);
  if (r == Io::Ok && hdr.type == proto::kHelloReply) r = ipc::sendMsg(fd, proto::kStatus, ++seq_, nullptr, 0, deadline);
  if (r == Io::Ok) r = ipc::recvMsg(fd, &hdr, buf, sizeof buf, deadline);
  if (r == Io::Ok && hdr.type == proto::kStatusReply) {
    close(fd);
    LOGW("ml: the runner (pid %d) answered a status check %lld ms after the tile timeout: busy, not stopped", pid,
         (long long)(nowMs() - t0));
    return "";
  }
  if (!cred || uid != getuid() || peer != pid) {
    close(fd);
    LOGW("ml: the runner (pid %d) did not answer a status check, but the socket's peer is pid %d uid %d: not stopped",
         pid, int(peer), int(uid));
    return "";
  }
  kill(pid, SIGKILL);  // while our connection keeps the process (and so its pid) from being reused
  close(fd);
  ++stats_.hungStopped;
  LOGW("ml: the runner (pid %d) did not answer a status check within %.1f s (%s): stopped (SIGKILL)", pid,
       o_.probeTimeoutS, ipc::ioName(r));
  for (const int64_t end = nowMs() + 2000; nowMs() < end; usleep(10000)) {  // until nothing listens
    const int c = ipc::connectUnix(socket_, &e, nullptr);
    if (c < 0) break;
    close(c);
  }
  runnerPid_ = -1;
  return "; the runner (pid " + std::to_string(pid) + ") was stuck and has been stopped: the next frame starts a new one";
}

void RunnerClient::startRunner() {
  if (o_.runnerPath.empty() || access(o_.runnerPath.c_str(), X_OK) != 0)
    throw fail("runner not found: " + (o_.runnerPath.empty() ? std::string("(no path)") : o_.runnerPath));
  const int64_t deadline = nowMs() + int64_t(o_.startTimeoutS * 1000);
  // One start at a time per socket (two nodes rendering at once start one runner). The lock is
  // released when the fd closes, also if this process dies.
  const std::string lockPath = ipc::dirName(socket_) + "/runner.lock";
  const int lock = open(lockPath.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (lock < 0) throw fail("runner start failed: " + lockPath + ": " + std::strerror(errno));
  struct Closer {
    int fd;
    ~Closer() { close(fd); }
  } closer{lock};
  while (flock(lock, LOCK_EX | LOCK_NB) != 0) {
    if (errno != EWOULDBLOCK && errno != EINTR) throw fail("runner start failed: flock: " + std::string(std::strerror(errno)));
    if (nowMs() > deadline) throw fail("runner start timed out (another start holds " + lockPath + ")");
    usleep(10000);
  }
  int e = 0;
  const int already = ipc::connectUnix(socket_, &e, nullptr);
  if (already >= 0) {  // another node started it while this one waited for the lock
    close(already);
    return;
  }

  std::vector<std::string> args = {o_.runnerPath, "--daemon", "--socket", socket_};
  if (o_.idleSeconds > 0) {
    args.push_back("--idle-s");
    args.push_back(std::to_string(o_.idleSeconds));
  }
  args.insert(args.end(), o_.extraArgs.begin(), o_.extraArgs.end());
  std::vector<std::string> env = runnerEnv(o_.env);
  std::vector<char*> argv = cstrs(args), envp = cstrs(env);

  posix_spawn_file_actions_t fa;
  posix_spawnattr_t at;
  posix_spawn_file_actions_init(&fa);
  posix_spawnattr_init(&at);
  // stdin / stdout / stderr: /dev/null (the runner opens its own stderr log); nothing else of the
  // host's fds (its own fds are close-on-exec; the runner closes any other at its start).
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
  posix_spawn_file_actions_addclosefrom_np(&fa, 3);
#endif
  sigset_t none, defaults;
  sigemptyset(&none);
  sigemptyset(&defaults);
  for (int s : {SIGTERM, SIGINT, SIGHUP, SIGPIPE, SIGCHLD, SIGUSR1, SIGUSR2}) sigaddset(&defaults, s);
  posix_spawnattr_setsigmask(&at, &none);
  posix_spawnattr_setsigdefault(&at, &defaults);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
  pid_t pid = -1;
  const int rc = posix_spawn(&pid, o_.runnerPath.c_str(), &fa, &at, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  if (rc != 0) throw fail("runner start failed: " + o_.runnerPath + ": " + std::strerror(rc));
  ++stats_.runnerStarts;
  LOGI("ml: starting the runner %s (launcher pid %d), socket %s", o_.runnerPath.c_str(), int(pid), socket_.c_str());

  // The launcher exits once the daemon listens (0), finds one running (3) or fails (other). A host
  // that reaps children itself (SIGCHLD ignored or its own handler) hides the status: then poll the
  // socket until the deadline.
  for (;;) {
    int st = 0;
    const pid_t w = waitpid(pid, &st, WNOHANG);
    if (w == pid) {
      const int code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
      if (code == 0 || code == 3) return;
      throw fail("runner start failed (exit " + std::to_string(code) + "; see " + logDir() + "/runner-stderr.log)");
    }
    if (w < 0 && errno != EINTR) {
      const int c = ipc::connectUnix(socket_, &e, nullptr);
      if (c >= 0) {
        close(c);
        return;
      }
    }
    if (nowMs() > deadline) {
      if (w == 0) {
        kill(pid, SIGKILL);
        waitpid(pid, &st, 0);
      }
      throw fail("runner start timed out (" + std::to_string(int(o_.startTimeoutS)) + " s)");
    }
    usleep(5000);
  }
}

void RunnerClient::ensureConnected(bool start) {
  if (fd_ >= 0) return;
  const int64_t t0 = ipc::nowUs();
  socketPath();
  if (!slot_) {
    std::string err;
    slotBytes_ = size_t(slotWant_);
    slotFd_ = ipc::createSlot(slotBytes_, &err);
    if (slotFd_ >= 0) slot_ = ipc::mapSlot(slotFd_, slotBytes_, &err);
    if (!slot_) {
      if (slotFd_ >= 0) close(slotFd_);
      slotFd_ = -1;
      throw fail("runner shared memory: " + err);
    }
  }
  bool started = false, replaced = false;
  for (;;) {
    int e = 0;
    std::string err;
    const int fd = ipc::connectUnix(socket_, &e, &err);
    if (fd < 0) {
      if (e != ENOENT && e != ECONNREFUSED) throw fail("runner socket: " + err);
      if (!start) throw fail("no runner is running (" + socket_ + ")");
      if (started) throw fail("runner start failed: nothing listens at " + socket_);
      startRunner();
      started = true;
      continue;
    }
    std::string why;
    const Hello r = handshake(fd, &why);
    if (r == Hello::Ok) {
      fd_ = fd;
      stats_.connectMs += msSince(t0);
      return;
    }
    // A runner this call did not start is replaced once; one it started itself is the binary at
    // runnerPath: starting it again would not help.
    if (r == Hello::Mismatch && start && !replaced && !started) {
      LOGW("ml: replacing the runner (pid %d): %s", runnerPid_, why.c_str());
      const bool gone = waitGone(fd, nowMs() + int64_t(o_.startTimeoutS * 1000));
      close(fd);
      if (!gone) throw fail("runner version mismatch and the old runner (pid " + std::to_string(runnerPid_) + ") did not quit");
      replaced = true;
      ++stats_.replaced;
      continue;
    }
    close(fd);
    throw fail(r == Hello::Mismatch ? "runner version mismatch: " + why + " (" + o_.runnerPath + ")"
                                    : "runner handshake failed: " + why);
  }
}

// A run needing more than the current slot: a new, larger slot (whole MiB) on a new connection.
void RunnerClient::ensureSlot(uint64_t need) {
  if (need <= slotWant_) return;  // slotWant_ = the current slot's size once one exists
  const uint64_t mib = uint64_t(1) << 20;
  slotWant_ = std::max<uint64_t>(slotWant_, (need + mib - 1) / mib * mib);
  disconnect();
  ipc::unmapSlot(slot_, slotBytes_);
  if (slotFd_ >= 0) close(slotFd_);
  slot_ = nullptr;
  slotFd_ = -1;
  slotBytes_ = 0;
  ++stats_.slotGrown;
  LOGI("ml: runner slot grows to %llu MiB", (unsigned long long)(slotWant_ >> 20));
}

void RunnerClient::run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut,
                       int oh, int ow, int x, int y) {
  if (cIn <= 0 || h <= 0 || w <= 0 || cOut <= 0 || oh <= 0 || ow <= 0 || modelId.empty() ||
      modelId.size() >= size_t(proto::kModelIdBytes))
    throw fail("bad model run (" + modelId + " [" + std::to_string(cIn) + "," + std::to_string(h) + "," +
               std::to_string(w) + "] -> [" + std::to_string(cOut) + "," + std::to_string(oh) + "," + std::to_string(ow) + "])");
  proto::RunRequest q{};
  ipc::setField(q.model, modelId);
  q.ep = ep_;
  q.cIn = uint32_t(cIn);
  q.h = uint32_t(h);
  q.w = uint32_t(w);
  q.cOut = uint32_t(cOut);
  q.outH = uint32_t(oh);
  q.outW = uint32_t(ow);
  q.x = x;
  q.y = y;
  const uint64_t inBytes = proto::slotOutputOffset(q), need = proto::slotRunBytes(q);
  if (need > proto::kMaxSlotBytes)
    throw fail("model run too large for the runner's shared memory (" + std::to_string(need >> 20) + " MiB, at most " +
               std::to_string(proto::kMaxSlotBytes >> 20) + " MiB)");
  ensureSlot(need);
  ensureConnected(true);
  const int64_t c0 = ipc::nowUs();
  std::memcpy(slot_, in, size_t(inBytes));  // the same slot survives a reconnect (the fd is passed again)
  double copyMs = msSince(c0);
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (attempt == 1) {
      disconnect();
      ++stats_.reconnects;
      LOGW("ml: runner connection lost during a run: reconnecting once");
      ensureConnected(true);
    }
    const int64_t t0 = ipc::nowUs(), deadline = nowMs() + int64_t(o_.tileTimeoutS * 1000);
    Io r = ipc::sendMsg(fd_, proto::kRun, ++seq_, &q, sizeof q, deadline);
    proto::MsgHeader hd{};
    char buf[proto::kMaxPayload];
    if (r == Io::Ok) r = ipc::recvMsg(fd_, &hd, buf, sizeof buf, deadline);
    if (r == Io::Timeout) {
      disconnect();  // the answer may still come: this connection's state is unknown
      throw fail("runner timed out (no answer within " + std::to_string(int(o_.tileTimeoutS)) + " s)" + stopIfHung());
    }
    if (r == Io::Eof || r == Io::Error) continue;
    if (r == Io::Protocol || hd.seq != seq_ || (hd.type != proto::kRunReply && hd.type != proto::kError)) {
      disconnect();
      throw fail("runner protocol error (answer type " + std::to_string(hd.type) + ")");
    }
    if (hd.type == proto::kError) {
      proto::ErrorReply er{};
      std::memcpy(&er, buf, hd.bytes < sizeof er ? hd.bytes : sizeof er);
      disconnect();
      throw fail("runner: " + ipc::getField(er.message));
    }
    proto::RunReply rep{};
    if (hd.bytes != sizeof rep) {
      disconnect();
      throw fail("runner protocol error (reply size)");
    }
    std::memcpy(&rep, buf, sizeof rep);
    ++stats_.tiles;
    stats_.roundTripMs += msSince(t0);
    stats_.runMs += rep.runUs / 1000.0;
    stats_.queueMs += rep.queueUs / 1000.0;
    if (rep.sessionCreated) {
      stats_.sessionCreated = true;
      stats_.loadMs += rep.loadUs / 1000.0;
    }
    const std::string msg = ipc::getField(rep.message);
    if (rep.status != proto::kOk)
      throw fail(rep.status == proto::kModelError ? "model error: " + msg : "runner rejected the run: " + msg);
    stats_.ep = ipc::getField(rep.ep);
    stats_.why = msg;
    const int64_t c1 = ipc::nowUs();
    std::memcpy(out, static_cast<const char*>(slot_) + inBytes, size_t(need - inBytes));
    copyMs += msSince(c1);
    stats_.copyMs += copyMs;
    return;
  }
  disconnect();
  throw fail("runner connection lost (the runner exited during the run, also after a restart)");
}

void RunnerClient::run(const float* input, int T, float* alpha, int x, int y) {
  run(model_, input, 4, T, T, alpha, 1, T, T, x, y);
}

void RunnerClient::runFrame(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut) {
  run(modelId, in, cIn, h, w, out, cOut, h, w, 0, 0);
}

bool RunnerClient::status(proto::StatusReply* out, std::string* err, bool start) {
  try {
    ensureConnected(start);
    const int64_t deadline = nowMs() + int64_t(o_.startTimeoutS * 1000);
    proto::MsgHeader h{};
    char buf[proto::kMaxPayload];
    Io r = ipc::sendMsg(fd_, proto::kStatus, ++seq_, nullptr, 0, deadline);
    if (r == Io::Ok) r = ipc::recvMsg(fd_, &h, buf, sizeof buf, deadline);
    if (r != Io::Ok || h.type != proto::kStatusReply || h.bytes != sizeof *out) {
      disconnect();
      if (err) *err = std::string("status: ") + (r != Io::Ok ? ipc::ioName(r) : "unexpected answer");
      return false;
    }
    std::memcpy(out, buf, sizeof *out);
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

bool RunnerClient::quit(std::string* err) {
  disconnect();
  try {
    socketPath();
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
  int e = 0;
  const int fd = ipc::connectUnix(socket_, &e, nullptr);
  if (fd < 0) return e == ENOENT || e == ECONNREFUSED;  // none running
  const int64_t deadline = nowMs() + int64_t(o_.startTimeoutS * 1000);
  // A bare HELLO (no slot): any runner version accepts it and then QUIT.
  proto::Hello h{};
  h.version = proto::kVersion;
  h.pid = uint32_t(getpid());
  ipc::setField(h.buildId, o_.buildId);
  ipc::setField(h.client, o_.name + ":quit");
  proto::MsgHeader hdr{};
  char buf[proto::kMaxPayload];
  Io r = ipc::sendMsg(fd, proto::kHello, ++seq_, &h, sizeof h, deadline);
  if (r == Io::Ok) r = ipc::recvMsg(fd, &hdr, buf, sizeof buf, deadline);
  const bool ok = r == Io::Ok && waitGone(fd, deadline);
  close(fd);
  if (!ok && err) *err = "the runner did not quit";
  return ok;
}

}  // namespace ml
}  // namespace mr
