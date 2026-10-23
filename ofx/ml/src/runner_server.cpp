// runner_server.h. C++14, POSIX.
#include "runner_server.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#include "log.h"
#include "runner_client.h"
#include "runner_ipc.h"

namespace mr {
namespace ml {

using ipc::Io;
using ipc::nowMs;

namespace {

// Strict FIFO: tickets in arrival order (decision 32: one tile on the GPU at a time).
class TicketLock {
 public:
  void lock() {
    std::unique_lock<std::mutex> l(m_);
    const uint64_t t = next_++;
    cv_.wait(l, [&] { return serving_ == t; });
  }
  void unlock() {
    {
      std::lock_guard<std::mutex> l(m_);
      ++serving_;
    }
    cv_.notify_all();
  }
  // At exit: wait up to ms for the queue to drain (an in-flight tile), without taking a ticket.
  bool idleWithin(int ms) {
    std::unique_lock<std::mutex> l(m_);
    return cv_.wait_for(l, std::chrono::milliseconds(ms), [&] { return serving_ == next_; });
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  uint64_t next_ = 0, serving_ = 0;
};

struct Server {
  ServerOptions o;
  Engine* engine = nullptr;
  TicketLock queue;
  std::atomic<int> connections{0};
  std::atomic<uint64_t> runs{0};
  std::atomic<int64_t> lastIdleStart{0};
  std::atomic<bool> stop{false};
  int wakeFd[2] = {-1, -1};
  int64_t started = 0;
  std::atomic<uint64_t> connCounter{0};
};

Server* gServer = nullptr;

void onSignal(int) {
  if (!gServer) return;
  gServer->stop = true;
  const char b = 1;
  ssize_t ignored = write(gServer->wakeFd[1], &b, 1);
  (void)ignored;
}

void wake(Server& s) {
  s.stop = true;
  const char b = 1;
  ssize_t ignored = write(s.wakeFd[1], &b, 1);
  (void)ignored;
}

long rssMiBNow() {
  std::ifstream f("/proc/self/status");  // Linux; macOS: -1 (not built there yet)
  std::string line;
  while (std::getline(f, line))
    if (line.compare(0, 6, "VmRSS:") == 0) return std::atol(line.c_str() + 6) / 1024;
  return -1;
}

void sendError(int fd, uint32_t seq, const std::string& msg) {
  proto::ErrorReply e{};
  ipc::setField(e.message, msg);
  ipc::sendMsg(fd, proto::kError, seq, &e, sizeof e, nowMs() + 2000);
}

void fillStatus(Server& s, proto::StatusReply* r) {
  std::memset(r, 0, sizeof *r);
  r->version = s.o.version;
  r->pid = uint32_t(getpid());
  ipc::setField(r->buildId, s.o.buildId);
  r->connections = uint32_t(s.connections.load());
  r->runs = s.runs.load();
  r->uptimeS = uint32_t((nowMs() - s.started) / 1000);
  r->idleSeconds = uint32_t(s.o.idleSeconds);
  r->rssMiB = int32_t(rssMiBNow());
  s.queue.lock();
  int n = 0;
  try {
    n = s.engine->sessions(r->session, proto::kMaxSessionsReported);
  } catch (...) {
  }
  s.queue.unlock();
  r->sessions = uint32_t(n < 0 ? 0 : n);
}

// One client connection, on its own thread. Nothing here may take the server down: every failure
// ends this connection only.
void serveConnection(Server& s, int fd) {
  const uint64_t id = ++s.connCounter;
  LogScope scope("runner");
  const int64_t opened = nowMs();
  uid_t uid = 0;
  pid_t peerPid = -1;
  ipc::peerUid(fd, &uid, &peerPid);
  proto::MsgHeader h{};
  char buf[proto::kMaxPayload];
  int slotFd = -1;
  Io r = ipc::recvMsg(fd, &h, buf, sizeof buf, nowMs() + 10000, &slotFd);
  void* slot = nullptr;
  size_t slotBytes = 0;
  uint64_t myRuns = 0;
  std::string client = "?";
  auto finish = [&](const char* why) {
    LOGI("connection %llu (%s, pid %d) closed: %s; %llu tiles in %.1f s", (unsigned long long)id, client.c_str(),
         int(peerPid), why, (unsigned long long)myRuns, (nowMs() - opened) / 1000.0);
    ipc::unmapSlot(slot, slotBytes);
    if (slotFd >= 0) close(slotFd);
    close(fd);
    if (--s.connections == 0) s.lastIdleStart = nowMs();
  };
  // Any version's HELLO starts with its protocol version: another version gets a HELLO_REPLY (so
  // it can tell) and may then only send QUIT.
  proto::Hello hello{};
  if (r == Io::Ok && h.type == proto::kHello && h.bytes >= sizeof(uint32_t))
    std::memcpy(&hello, buf, h.bytes < sizeof hello ? h.bytes : sizeof hello);
  const bool sameVersion = hello.version == s.o.version;
  if (r != Io::Ok || h.type != proto::kHello || h.bytes < sizeof(uint32_t) ||
      (sameVersion && h.bytes != sizeof(proto::Hello))) {
    if (r == Io::Ok) sendError(fd, h.seq, "expected HELLO");
    finish(r == Io::Ok ? "no HELLO" : ipc::ioName(r));
    return;
  }
  if (sameVersion) client = ipc::getField(hello.client);
  proto::HelloReply rep{};
  rep.version = s.o.version;
  rep.pid = uint32_t(getpid());
  ipc::setField(rep.buildId, s.o.buildId);
  rep.idleSeconds = uint32_t(s.o.idleSeconds);
  rep.status = proto::kOk;
  std::string why;
  if (sameVersion && slotFd >= 0) {
    slotBytes = size_t(hello.slotBytes);
    if (hello.slotBytes < proto::kMinSlotBytes || hello.slotBytes > proto::kMaxSlotBytes ||
        !ipc::checkSlot(slotFd, slotBytes, &why) || !(slot = ipc::mapSlot(slotFd, slotBytes, &why))) {
      if (why.empty())
        why = "slot size " + std::to_string(hello.slotBytes) + " bytes (" + std::to_string(proto::kMinSlotBytes) + "-" +
              std::to_string(proto::kMaxSlotBytes) + ")";
      rep.status = proto::kBadSlot;
      ipc::setField(rep.message, why);
      slotBytes = 0;
    }
  }
  LOGI("connection %llu opened: client %s pid %d protocol %u build '%s'%s%s", (unsigned long long)id, client.c_str(),
       int(peerPid), hello.version, ipc::getField(hello.buildId).c_str(), slot ? ", slot mapped" : ", no slot",
       why.empty() ? "" : (" (" + why + ")").c_str());
  if (ipc::sendMsg(fd, proto::kHelloReply, h.seq, &rep, sizeof rep, nowMs() + 5000) != Io::Ok) {
    finish("HELLO reply failed");
    return;
  }

  for (;;) {
    r = ipc::recvMsg(fd, &h, buf, sizeof buf, -1);
    if (r != Io::Ok) {
      if (r == Io::Protocol) sendError(fd, h.seq, "bad message header");
      finish(ipc::ioName(r));
      return;
    }
    if (h.type == proto::kQuit) {
      LOGI("QUIT from %s (pid %d)", client.c_str(), int(peerPid));
      ipc::sendMsg(fd, proto::kQuitReply, h.seq, nullptr, 0, nowMs() + 2000);
      wake(s);
      finish("quit");
      return;
    }
    if (!sameVersion) {  // only QUIT is understood from another protocol
      sendError(fd, h.seq, "protocol mismatch: send QUIT");
      finish("protocol mismatch");
      return;
    }
    if (h.type == proto::kStatus && h.bytes == 0) {
      proto::StatusReply st;
      fillStatus(s, &st);
      if (ipc::sendMsg(fd, proto::kStatusReply, h.seq, &st, sizeof st, nowMs() + 5000) != Io::Ok) {
        finish("status reply failed");
        return;
      }
      continue;
    }
    if (h.type != proto::kRun || h.bytes != sizeof(proto::RunRequest)) {
      sendError(fd, h.seq, "bad request (type " + std::to_string(h.type) + ", " + std::to_string(h.bytes) + " bytes)");
      finish("bad request");
      return;
    }
    proto::RunRequest q{};
    std::memcpy(&q, buf, sizeof q);
    proto::RunReply out{};
    std::string bad;
    const ModelSpec* model = validateRun(q, slot ? uint64_t(slotBytes) : 0, &bad);
    if (!model) {
      out.status = proto::kBadRequest;
      ipc::setField(out.message, bad);
      LOGW("connection %llu: bad RUN: %s", (unsigned long long)id, bad.c_str());
    } else {
      const int64_t q0 = ipc::nowUs();
      s.queue.lock();
      const int64_t q1 = ipc::nowUs();
      RunResult res;
      try {
        res = s.engine->run(*model, q, static_cast<const float*>(slot),
                            reinterpret_cast<float*>(static_cast<char*>(slot) + proto::slotOutputOffset(q)));
      } catch (const std::exception& e) {
        res = RunResult();
        res.status = proto::kModelError;
        res.message = e.what();
      } catch (...) {
        res = RunResult();
        res.status = proto::kModelError;
        res.message = "unknown error";
      }
      s.queue.unlock();
      ++s.runs;
      ++myRuns;
      out.status = res.status;
      out.sessionCreated = res.created ? 1 : 0;
      out.runUs = uint32_t(res.runMs * 1000);
      out.queueUs = uint32_t(q1 - q0);
      out.loadUs = uint32_t(res.loadMs * 1000);
      ipc::setField(out.ep, res.ep);
      ipc::setField(out.message, res.message);
      LOGD("connection %llu: RUN %s ep %u [%u,%u,%u]->[%u,%u,%u] at %d,%d: status %u, queue %.1f ms, run %.1f ms%s%s",
           (unsigned long long)id, model->id, q.ep, q.cIn, q.h, q.w, q.cOut, q.outH, q.outW, q.x, q.y, res.status,
           (q1 - q0) / 1000.0, res.runMs,
           res.message.empty() ? "" : ": ", res.message.c_str());
      if (res.status != proto::kOk)
        LOGW("connection %llu: RUN failed: %s", (unsigned long long)id, res.message.c_str());
    }
    if (ipc::sendMsg(fd, proto::kRunReply, h.seq, &out, sizeof out, nowMs() + 5000) != Io::Ok) {
      finish("reply failed (client gone)");
      return;
    }
  }
}

bool sameInode(const std::string& path, ino_t ino) {
  struct stat st {};
  return lstat(path.c_str(), &st) == 0 && st.st_ino == ino;
}

}  // namespace

const ModelSpec* validateRun(const proto::RunRequest& q, uint64_t slotBytes, std::string* why) {
  // The id must be NUL-terminated inside its field (getField stops at the field's end anyway).
  const std::string id = ipc::getField(q.model);
  const ModelSpec* m = nullptr;
  std::string bad;
  if (slotBytes == 0) bad = "no shared memory on this connection";
  else if (q.slot != 0) bad = "slot " + std::to_string(q.slot) + " does not exist";
  else if (id.size() >= sizeof q.model) bad = "model id not terminated";
  else if (!(m = findModel(id))) bad = "unknown model '" + id + "'";
  else if (q.ep > proto::kEpCpu) bad = "unknown EP " + std::to_string(q.ep);
  else bad = checkRequest(*m, TensorDesc{q.cIn, q.h, q.w, q.cOut, q.outH, q.outW});
  // After the table's checks the sizes are bounded (no overflow in the byte count).
  if (bad.empty() && proto::slotRunBytes(q) > slotBytes)
    bad = "the run needs " + std::to_string(proto::slotRunBytes(q)) + " bytes of shared memory, the slot has " +
          std::to_string(slotBytes);
  if (!bad.empty()) {
    if (why) *why = bad;
    return nullptr;
  }
  return m;
}

bool parseServerArgs(int argc, char** argv, ServerOptions* o, std::vector<std::string>* rest, std::string* err) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--socket" && i + 1 < argc) {
      o->socketPath = argv[++i];
    } else if (a == "--idle-s" && i + 1 < argc) {
      o->idleSeconds = std::atoi(argv[++i]);
      if (o->idleSeconds <= 0) {
        if (err) *err = "--idle-s needs a number of seconds > 0";
        return false;
      }
    } else if (a == "--daemon") {
      o->daemon = true;
    } else {
      rest->push_back(a);
    }
  }
  return true;
}

void closeInheritedFds() {
#if defined(__linux__)
  std::vector<int> fds;
  if (DIR* d = opendir("/proc/self/fd")) {
    const int own = dirfd(d);
    while (dirent* e = readdir(d)) {
      const int fd = std::atoi(e->d_name);
      if (fd > 2 && fd != own) fds.push_back(fd);
    }
    closedir(d);
  }
  for (int fd : fds) close(fd);
#else
  const long max = sysconf(_SC_OPEN_MAX);
  for (int fd = 3; fd < (max > 0 && max < 65536 ? max : 65536); ++fd) close(fd);
#endif
}

int serve(const ServerOptions& opts, Engine& engine) {
  Server s;
  s.o = opts;
  s.engine = &engine;
  if (s.o.buildId.empty()) s.o.buildId = runnerBuildId();
  LogScope scope("runner");
  std::string err;
  if (s.o.socketPath.empty()) s.o.socketPath = ipc::defaultSocketPath(&err, s.o.buildId);
  if (s.o.socketPath.empty() || !ipc::ensurePrivateDir(ipc::dirName(s.o.socketPath), &err)) {
    LOGE("runner: no socket folder: %s", err.c_str());
    std::fprintf(stderr, "mr_ml_runner: %s\n", err.c_str());
    return 1;
  }

  int ready[2] = {-1, -1};
  if (s.o.daemon) {
    // Launcher -> daemon: the launcher waits for one status byte, exits with it; the client
    // (RunnerClient::startRunner) waits for the launcher. The daemon has its own session: a host
    // crash (or its process group being killed) does not reach it.
    if (pipe(ready) != 0) return 1;
    const pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid > 0) {
      close(ready[1]);
      pollfd p{ready[0], POLLIN, 0};
      char code = 1;
      if (poll(&p, 1, 20000) <= 0 || read(ready[0], &code, 1) != 1) code = 1;
      _exit(code);
    }
    close(ready[0]);
    setsid();
    // stdout / stderr (ORT, CUDA) to a file of their own: the main log stays line-structured.
    LOGI("runner daemon starting (pid %d)", int(getpid()));
    // Capped: started over when a runner starts and finds it above kStderrCapBytes.
    const std::string se = logDir() + "/runner-stderr.log";
    struct stat sst {};
    const bool big = stat(se.c_str(), &sst) == 0 && uint64_t(sst.st_size) > kStderrCapBytes;
    const int f = open(se.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | (big ? O_TRUNC : 0), 0644);
    if (big) LOGI("runner: %s was %lld bytes: started over", se.c_str(), (long long)sst.st_size);
    if (f >= 0) {
      dup2(f, 1);
      dup2(f, 2);
      close(f);
    }
  }
  auto report = [&](char code) {
    if (ready[1] >= 0) {
      ssize_t ignored = write(ready[1], &code, 1);
      (void)ignored;
      close(ready[1]);
      ready[1] = -1;
    }
  };

  // A runner that answers already: leave it (3). A socket nobody answers on: stale, replace it.
  int e = 0;
  const int probe = ipc::connectUnix(s.o.socketPath, &e, nullptr);
  if (probe >= 0) {
    close(probe);
    LOGI("runner: another runner already listens at %s: exiting", s.o.socketPath.c_str());
    report(3);
    return 3;
  }
  if (e == ECONNREFUSED) unlink(s.o.socketPath.c_str());
  const int lfd = ipc::listenUnix(s.o.socketPath, &err);
  if (lfd < 0) {
    LOGE("runner: %s", err.c_str());
    std::fprintf(stderr, "mr_ml_runner: %s\n", err.c_str());
    report(1);
    return 1;
  }
  struct stat st {};
  lstat(s.o.socketPath.c_str(), &st);
  const ino_t ino = st.st_ino;

  if (pipe(s.wakeFd) != 0) {
    report(1);
    return 1;
  }
  fcntl(s.wakeFd[0], F_SETFD, FD_CLOEXEC);
  fcntl(s.wakeFd[1], F_SETFD, FD_CLOEXEC);
  fcntl(s.wakeFd[1], F_SETFL, O_NONBLOCK);
  gServer = &s;
  struct sigaction sa {};
  sa.sa_handler = onSignal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGHUP, SIG_IGN);

  s.started = nowMs();
  s.lastIdleStart = nowMs();
  LOGI("runner %s (protocol %u) pid %d listening at %s, idle exit after %d s", s.o.buildId.c_str(), s.o.version,
       int(getpid()), s.o.socketPath.c_str(), s.o.idleSeconds);
  report(0);

  const char* why = "?";
  int64_t lastCheck = nowMs();
  while (!s.stop) {
    pollfd p[2] = {{lfd, POLLIN, 0}, {s.wakeFd[0], POLLIN, 0}};
    const int n = poll(p, 2, 500);
    if (s.stop) {
      why = "quit or signal";
      break;
    }
    if (n > 0 && (p[0].revents & POLLIN)) {
      const int c = accept(lfd, nullptr, nullptr);
      if (c >= 0) {
        fcntl(c, F_SETFD, FD_CLOEXEC);
        uid_t uid = 0;
        if (!ipc::peerUid(c, &uid, nullptr) || uid != getuid()) {
          LOGW("runner: refused a connection from uid %d", int(uid));
          close(c);
        } else {
          ++s.connections;
          try {
            std::thread([&s, c] { serveConnection(s, c); }).detach();
          } catch (const std::exception& ex) {
            LOGE("runner: no thread for a connection: %s", ex.what());
            close(c);
            if (--s.connections == 0) s.lastIdleStart = nowMs();
          }
        }
      }
    }
    const int64_t now = nowMs();
    if (s.connections == 0 && now - s.lastIdleStart >= int64_t(s.o.idleSeconds) * 1000) {
      why = "idle";
      break;
    }
    if (now - lastCheck > 2000) {
      lastCheck = now;
      if (!sameInode(s.o.socketPath, ino) && s.connections == 0) {
        why = "socket removed or replaced";
        break;
      }
    }
  }
  close(lfd);
  if (sameInode(s.o.socketPath, ino)) unlink(s.o.socketPath.c_str());
  const bool drained = s.queue.idleWithin(3000);
  LOGI("runner pid %d exiting (%s), %llu tiles in %.0f s%s", int(getpid()), why, (unsigned long long)s.runs.load(),
       (nowMs() - s.started) / 1000.0, drained ? "" : ", a tile was still running");
  // No teardown: sessions (CUDA) are not destroyed at exit, other threads may still be blocked.
  std::fflush(nullptr);
  _exit(0);
}

int controlMain(const std::string& cmd, const std::string& socket) {
  RunnerClientOptions o;
  o.socketPath = socket;
  o.name = "cli";
  o.startTimeoutS = 5;
  RunnerClient c(o);
  std::string err;
  if (cmd == "quit") {
    const bool ok = c.quit(&err);
    std::printf("%s\n", ok ? "runner stopped (or none was running)" : err.c_str());
    return ok ? 0 : 1;
  }
  proto::StatusReply st;
  if (!c.status(&st, &err, false)) {
    std::printf("no runner: %s\n", err.c_str());
    return 1;
  }
  std::printf("runner pid %u build '%s' protocol %u: %u connection(s), %llu tiles, up %u s, idle exit %u s, RSS %d MiB\n",
              st.pid, ipc::getField(st.buildId).c_str(), st.version, st.connections, (unsigned long long)st.runs,
              st.uptimeS, st.idleSeconds, st.rssMiB);
  for (uint32_t i = 0; i < st.sessions && i < uint32_t(proto::kMaxSessionsReported); ++i) {
    const proto::SessionInfo& si = st.session[i];
    std::printf("  session %s ep %s (active %s): %llu runs, idle %u ms\n", ipc::getField(si.model).c_str(),
                si.ep == proto::kEpCpu ? "cpu" : "cuda", ipc::getField(si.activeEp).c_str(),
                (unsigned long long)si.runs, si.idleMs);
  }
  return 0;
}

}  // namespace ml
}  // namespace mr
