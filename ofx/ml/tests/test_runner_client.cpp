// The runner protocol (v2, model-generic) and RunnerClient (decision 32) against the fake runner
// (fake_runner.cpp): framing and fd passing, the private socket folder, the model table and the
// server's request validation (every rule, in-process), two model shapes through one connection
// (ViTMatte 4->1 tiles, BiRefNet 3->1 at 1024: exact answers, the slot layout), the slot growing for
// a larger run, bad requests on the wire (wrong channels, too large, unknown model, a run that does
// not fit the slot, bad HELLO slots: clean error strings, the connection survives), lazy start under
// the lock (one start for two simultaneous clients), exact tile results for two interleaved
// clients, model errors, crash mid-tile (one reconnect, then a clean error), hang (the
// timeout, then the status probe stops the hung runner), slow past the timeout but answering (not
// stopped), a probe whose socket peer is not the reported pid (nothing killed), the runner killed
// between frames, version / protocol mismatch (replaced exactly once, no loop), idle exit, runner
// not found, two builds with their own sockets at once, the stderr cap, the "runner" log
// component, and that the runner is detached and gets neither the host's library path nor its fds.
// Usage: test_runner_client <fake runner executable>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model_table.h"
#include "runner_client.h"
#include "runner_ipc.h"
#include "runner_server.h"

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

std::string gFake, gDir, gSock, gHome;

void writeFile(const std::string& name, const std::string& text) { std::ofstream(gDir + "/" + name) << text; }
std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
int starts() {
  std::ifstream f(gDir + "/fake.starts");
  std::string l;
  int n = 0;
  while (std::getline(f, l)) n += !l.empty();
  return n;
}
bool alive(int pid) { return pid > 0 && kill(pid, 0) == 0; }
bool waitDead(int pid, int ms) {
  for (int i = 0; i < ms / 10; ++i) {
    if (!alive(pid)) return true;
    usleep(10000);
  }
  return !alive(pid);
}
bool socketGone(int ms) {
  for (int i = 0; i < ms / 10; ++i) {
    if (access(gSock.c_str(), F_OK) != 0) return true;
    usleep(10000);
  }
  return false;
}

RunnerClientOptions opts(const char* name) {
  RunnerClientOptions o;
  o.runnerPath = gFake;
  o.socketPath = gSock;
  o.name = name;
  o.tileTimeoutS = 3;
  o.startTimeoutS = 5;
  o.idleSeconds = 60;
  o.env = {"HOME=" + gHome};
  return o;
}

// The fake's answer (fake_runner.cpp), the same expression: plane c at (lx, ly).
float expected(int x, int y, int lx, int ly, float in0, float inLast, int c) {
  return float(((x + lx) * 31 + (y + ly) * 17) % 1009) / 1009.0f + in0 + inLast + float(c);
}

// True if out ([cOut, h, w]) is the fake's exact answer to in ([cIn, h, w]).
bool answerOk(const std::vector<float>& in, int cIn, int h, int w, const std::vector<float>& out, int cOut, int x,
              int y) {
  const size_t plane = size_t(h) * w;
  for (int c = 0; c < cOut; ++c)
    for (int ly = 0; ly < h; ++ly)
      for (int lx = 0; lx < w; ++lx) {
        const size_t i = size_t(ly) * w + lx;
        if (out[c * plane + i] != expected(x, y, lx, ly, in[i], in[(cIn - 1) * plane + i], c)) return false;
      }
  return true;
}

std::vector<float> pattern(size_t n, float seed) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = seed + float(i % 7) * 0.125f + float(i % 1013) * 1e-3f;
  return v;
}

// One ViTMatte tile through c (TileRunner); true if every output pixel is the fake's exact answer.
bool tileOk(RunnerClient& c, int T, int x, int y, float seed) {
  std::vector<float> in = pattern(size_t(4) * T * T, seed), out(size_t(T) * T, -1.0f);
  c.run(in.data(), T, out.data(), x, y);
  return answerOk(in, 4, T, T, out, 1, x, y);
}

// One run of any shape through the generic call.
bool runOk(RunnerClient& c, const std::string& model, int cIn, int h, int w, int cOut, int x, int y, float seed) {
  std::vector<float> in = pattern(size_t(cIn) * h * w, seed), out(size_t(cOut) * h * w, -1.0f);
  c.run(model, in.data(), cIn, h, w, out.data(), cOut, h, w, x, y);
  return answerOk(in, cIn, h, w, out, cOut, x, y);
}

std::string runErr(RunnerClient& c, const std::string& model, int cIn, int h, int w, int cOut, int oh, int ow) {
  try {
    std::vector<float> in(size_t(cIn) * h * w, 0.0f), out(size_t(cOut) * oh * ow);
    c.run(model, in.data(), cIn, h, w, out.data(), cOut, oh, ow, 0, 0);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

std::string runError(RunnerClient& c, int T = 64) {
  try {
    tileOk(c, T, 0, 0, 0.0f);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

int runnerPid() {
  RunnerClient c(opts("probe"));
  proto::StatusReply st;
  std::string err;
  return c.status(&st, &err, false) ? int(st.pid) : -1;
}

void stopRunner() {
  RunnerClient c(opts("stop"));
  std::string err;
  const int pid = runnerPid();
  c.quit(&err);
  if (pid > 0) waitDead(pid, 3000);
}

bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// ---- protocol unit tests --------------------------------------------------------------------
void testFraming() {
  int sv[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  // A message with an fd: payload intact, seq kept, the fd is the same file.
  std::string err;
  const int slot = ipc::createSlot(4096, &err);
  CHECK(slot >= 0);
  proto::RunRequest q{};
  ipc::setField(q.model, std::string("vitmatte-small"));
  q.h = 512;
  q.x = -7;
  q.y = 99;
  CHECK(ipc::sendMsg(sv[0], proto::kRun, 42, &q, sizeof q, ipc::nowMs() + 1000, slot) == ipc::Io::Ok);
  proto::MsgHeader h{};
  char buf[proto::kMaxPayload];
  int got = -1;
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 1000, &got) == ipc::Io::Ok);
  proto::RunRequest q2{};
  std::memcpy(&q2, buf, sizeof q2);
  CHECK(h.type == proto::kRun && h.seq == 42 && h.bytes == sizeof q && q2.h == 512 && q2.x == -7 && q2.y == 99 &&
        ipc::getField(q2.model) == "vitmatte-small");
  struct stat a {}, b {};
  CHECK(got >= 0 && fstat(got, &a) == 0 && fstat(slot, &b) == 0 && a.st_ino == b.st_ino);
  CHECK(got >= 0 && (fcntl(got, F_GETFD) & FD_CLOEXEC));
  CHECK(ipc::checkSlot(got, 4096, &err));
  CHECK(!ipc::checkSlot(got, 8192, &err));  // too small
  if (got >= 0) close(got);
  close(slot);
  // An unsealed memory file is refused (it could shrink under the runner).
  const int plain = memfd_create("x", MFD_CLOEXEC);
  CHECK(plain >= 0 && ftruncate(plain, 4096) == 0 && !ipc::checkSlot(plain, 4096, &err));
  close(plain);
  // A header split over two writes still arrives.
  proto::MsgHeader hh{proto::kMagic, proto::kStatus, 0, 0, 7};
  std::thread t([&] {
    CHECK(write(sv[0], &hh, 5) == 5);
    usleep(50000);
    CHECK(write(sv[0], reinterpret_cast<char*>(&hh) + 5, sizeof hh - 5) == ssize_t(sizeof hh - 5));
  });
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 2000) == ipc::Io::Ok && h.seq == 7);
  t.join();
  // Wrong magic, oversized payload: protocol errors.
  hh.magic = 0x12345678;
  CHECK(write(sv[0], &hh, sizeof hh) == ssize_t(sizeof hh));
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 1000) == ipc::Io::Protocol);
  hh.magic = proto::kMagic;
  hh.bytes = 100000;
  CHECK(write(sv[0], &hh, sizeof hh) == ssize_t(sizeof hh));
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 1000) == ipc::Io::Protocol);
  // Nothing sent: a timeout at the deadline; the peer closed: Eof.
  const int64_t t0 = ipc::nowMs();
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 150) == ipc::Io::Timeout);
  const int64_t took = ipc::nowMs() - t0;
  CHECK(took >= 140 && took < 1000);
  close(sv[0]);
  CHECK(ipc::recvMsg(sv[1], &h, buf, sizeof buf, ipc::nowMs() + 1000) == ipc::Io::Eof);
  close(sv[1]);
  // Fixed fields: NUL-terminated, truncated.
  char f[8];
  ipc::setField(f, std::string("0123456789"));
  CHECK(ipc::getField(f) == "0123456");
  std::printf("framing: ok\n");
}

void testSocketFolder() {
  const std::string xdg = gDir + "/xdg";
  mkdir(xdg.c_str(), 0700);
  const char* old = std::getenv("XDG_RUNTIME_DIR");
  const std::string saved = old ? old : "";
  setenv("XDG_RUNTIME_DIR", xdg.c_str(), 1);
  std::string err;
  const std::string p = ipc::defaultSocketPath(&err);
  CHECK(p == xdg + "/matte-refine-ml/runner.sock");
  struct stat st {};
  CHECK(stat((xdg + "/matte-refine-ml").c_str(), &st) == 0 && (st.st_mode & 0777) == 0700);
  chmod((xdg + "/matte-refine-ml").c_str(), 0755);  // group / other access: refused
  CHECK(ipc::defaultSocketPath(&err).empty() && !err.empty());
  chmod((xdg + "/matte-refine-ml").c_str(), 0700);
  // One socket per build: a short stable name per build id.
  const std::string a = ipc::defaultSocketPath(&err, "0.0.1 abc"), b = ipc::defaultSocketPath(&err, "0.0.1 abd");
  CHECK(a == xdg + "/matte-refine-ml/" + ipc::socketName("0.0.1 abc") && a != b);
  CHECK(ipc::socketName("0.0.1 abc") == ipc::socketName("0.0.1 abc") && ipc::socketName("").compare("runner.sock") == 0);
  CHECK(ipc::socketName("x").size() == std::string("runner-12345678.sock").size());
  if (old) setenv("XDG_RUNTIME_DIR", saved.c_str(), 1);
  else unsetenv("XDG_RUNTIME_DIR");
  std::printf("socket folder: ok\n");
}

// ---- against the fake runner ----------------------------------------------------------------
void testNormal() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("normal"));
  const int s0 = starts();
  CHECK(tileOk(c, 512, 0, 0, 0.5f));
  CHECK(c.stats().runnerStarts == 1 && starts() == s0 + 1);
  CHECK(tileOk(c, 32, 1000, 7, -2.0f));
  CHECK(tileOk(c, 1024, 13, 4000, 3.0f));
  CHECK(tileOk(c, 512, -5, 300, 0.25f));
  CHECK(c.stats().tiles == 4 && c.stats().reconnects == 0 && c.stats().ep == "fake");
  proto::StatusReply st;
  std::string err;
  CHECK(c.status(&st, &err));
  CHECK(int(st.pid) == c.runnerPid() && st.connections >= 1 && st.runs >= 4 && st.idleSeconds == 60);
  // The runner is detached: its own session, not this process's child.
  CHECK(getsid(c.runnerPid()) == c.runnerPid());
  std::printf("normal: ok (runner pid %d)\n", c.runnerPid());
}

void testDetachedEnvAndFds() {
  stopRunner();
  setenv("LD_LIBRARY_PATH", "/host/lib/path", 1);  // a host's loader path must not reach the runner
  setenv("LD_PRELOAD", "", 1);
  const int leak = open("/dev/zero", O_RDONLY);    // no CLOEXEC: a host fd the runner must not keep
  RunnerClient c(opts("env"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const std::string env = readFile(gDir + "/fake.env");
  CHECK(!contains(env, "LD_LIBRARY_PATH") && !contains(env, "LD_PRELOAD"));
  CHECK(contains(env, ("HOME=" + gHome + "\n").c_str()));
  int leaked = 0, total = 0;
  const std::string fdDir = "/proc/" + std::to_string(c.runnerPid()) + "/fd";
  if (DIR* d = opendir(fdDir.c_str())) {
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      char link[512] = {0};
      ++total;
      if (readlink((fdDir + "/" + e->d_name).c_str(), link, sizeof link - 1) > 0 && contains(link, "/dev/zero"))
        ++leaked;
    }
    closedir(d);
  }
  CHECK(total > 0 && leaked == 0);
  close(leak);
  unsetenv("LD_LIBRARY_PATH");
  unsetenv("LD_PRELOAD");
  std::printf("runner environment and fds: ok (%d fds open in the runner)\n", total);
}

void testBadRequestAndModelError() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("bad"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const std::string e1 = runError(c, 48);  // not a multiple of 32: the runner refuses it
  CHECK(contains(e1, "runner rejected the run") && contains(e1, "48x48"));
  CHECK(tileOk(c, 64, 64, 0, 1.0f));      // same connection
  writeFile("fake.mode", "error");
  const std::string e3 = runError(c);
  CHECK(e3 == "model error: fake model failure");
  writeFile("fake.mode", "normal");
  CHECK(tileOk(c, 64, 0, 64, 2.0f));
  CHECK(c.stats().reconnects == 0 && c.stats().runnerStarts == 0 && c.stats().slotGrown == 0);
  std::printf("bad request, model error: ok ('%s' / '%s')\n", e1.c_str(), e3.c_str());
}

// ---- protocol 2: the model table and the server's validation (in-process) ---------------------
proto::RunRequest req(const char* model, uint32_t cIn, uint32_t h, uint32_t w, uint32_t cOut, uint32_t oh, uint32_t ow) {
  proto::RunRequest q{};
  ipc::setField(q.model, std::string(model));
  q.cIn = cIn;
  q.h = h;
  q.w = w;
  q.cOut = cOut;
  q.outH = oh;
  q.outW = ow;
  return q;
}

std::string why(const proto::RunRequest& q, uint64_t slot = proto::kDefaultSlotBytes) {
  std::string w;
  const ModelSpec* m = validateRun(q, slot, &w);
  return m ? std::string("ok:") + m->id : w;
}

void testTableAndValidation() {
  // The table: today's rows, the ViTMatte ones as protocol 1 had them.
  const ModelSpec* vs = findModel("vitmatte-small");
  const ModelSpec* vb = findModel("vitmatte-base");
  CHECK(vs && vb && !findModel("small") && !findModel(""));
  CHECK(vs && std::string(vs->file) == "vitmatte-small/model.onnx" && vs->channelsIn == 4 && vs->channelsOut == 1 &&
        vs->sizeRule == SizeRule::kMultipleOf && vs->multiple == 32 && vs->minSide == 32 && vs->maxSide == 1024);
  CHECK(vb && std::string(vb->file) == "vitmatte-base/model.onnx");
  for (int i = 0; i < modelCount(); ++i) {
    const ModelSpec& m = modelAt(i);
    CHECK(std::strlen(m.id) < size_t(proto::kModelIdBytes) && findModel(m.id) == &m);
    CHECK(std::string(m.file) == std::string(m.id) + "/model.onnx");
    // Decision 36: only BiRefNet has an arena hint; ViTMatte is never limited.
    const bool bir = std::string(m.id).compare(0, 8, "birefnet") == 0;
    CHECK(bir ? m.gpuMemLimitMiB > 0 && m.arenaSameAsRequested : m.gpuMemLimitMiB == 0 && !m.arenaSameAsRequested);
    for (long total : {-1L, 0L, 16384L, 24576L, 32623L, 32768L, 32769L, 49152L, 97887L}) {
      const CudaArena a = cudaArenaFor(m, total);
      const bool cap = bir && total > 0 && total <= 32768;
      CHECK(a.limitMiB == (cap ? m.gpuMemLimitMiB : 0u) && a.sameAsRequested == cap);
    }
    CHECK(std::strlen(m.sha256) == 64 && *m.licence);
  }
  const ModelSpec* bl = findModel("birefnet-lite");
  // The hints measured on a 32 GB card (model_table.cpp): lite 8192, full 10240, both under 24 GB.
  CHECK(bl && bl->gpuMemLimitMiB == 8192 && findModel("birefnet") && findModel("birefnet")->gpuMemLimitMiB == 10240);
  CHECK(kSmallGpuMiB == 32768);
  CHECK(bl && bl->channelsIn == 3 && bl->channelsOut == 1 && bl->sizeRule == SizeRule::kFixed && bl->fixedW == 1024 &&
        bl->fixedH == 1024);
  // The default slot holds the largest run of the table: ViTMatte 1024 (20 MiB); BiRefNet 16 MiB.
  CHECK(maxRunBytes() == uint64_t(5) * 1024 * 1024 * 4 && maxRunBytes() <= proto::kDefaultSlotBytes);
  CHECK(proto::slotRunBytes(req("birefnet-lite", 3, 1024, 1024, 1, 1024, 1024)) == uint64_t(16) << 20);
  // The slot layout: [input][output], no gap.
  const proto::RunRequest t = req("vitmatte-small", 4, 512, 256, 1, 512, 256);
  CHECK(proto::slotOutputOffset(t) == uint64_t(4) * 512 * 256 * 4 && proto::slotRunBytes(t) == uint64_t(5) * 512 * 256 * 4);

  // Every rule of validateRun / checkRequest.
  CHECK(why(req("vitmatte-small", 4, 512, 512, 1, 512, 512)) == "ok:vitmatte-small");
  CHECK(why(req("vitmatte-base", 4, 32, 1024, 1, 32, 1024)) == "ok:vitmatte-base");
  CHECK(why(req("birefnet-lite", 3, 1024, 1024, 1, 1024, 1024)) == "ok:birefnet-lite");
  CHECK(why(req("birefnet", 3, 1024, 1024, 1, 1024, 1024)) == "ok:birefnet");
  CHECK(why(req("vitmatte-small", 4, 512, 512, 1, 512, 512), 0) == "no shared memory on this connection");
  proto::RunRequest q = req("vitmatte-small", 4, 512, 512, 1, 512, 512);
  q.slot = 1;
  CHECK(why(q) == "slot 1 does not exist");
  q = req("vitmatte-small", 4, 512, 512, 1, 512, 512);
  std::memset(q.model, 'a', sizeof q.model);  // no NUL inside the field
  CHECK(why(q) == "model id not terminated");
  CHECK(why(req("vitmatte-large", 4, 512, 512, 1, 512, 512)) == "unknown model 'vitmatte-large'");
  q = req("vitmatte-small", 4, 512, 512, 1, 512, 512);
  q.ep = 3;
  CHECK(why(q) == "unknown EP 3");
  q.ep = proto::kEpCpu;
  CHECK(why(q) == "ok:vitmatte-small");
  CHECK(why(req("vitmatte-small", 3, 512, 512, 1, 512, 512)) == "model vitmatte-small takes 4 input channels, not 3");
  CHECK(why(req("birefnet-lite", 4, 1024, 1024, 1, 1024, 1024)) == "model birefnet-lite takes 3 input channels, not 4");
  CHECK(why(req("vitmatte-small", 4, 512, 512, 2, 512, 512)) == "model vitmatte-small gives 1 output channels, not 2");
  CHECK(contains(why(req("vitmatte-small", 4, 512, 48, 1, 512, 48)), "size 48x512"));
  CHECK(contains(why(req("vitmatte-small", 4, 1056, 512, 1, 1056, 512)), "size 512x1056"));
  CHECK(contains(why(req("vitmatte-small", 4, 0, 0, 1, 0, 0)), "size 0x0"));
  CHECK(why(req("vitmatte-small", 4, 32, 32, 1, 32, 32)) == "ok:vitmatte-small");
  CHECK(contains(why(req("vitmatte-small", 4, 1024, 1024, 1, 1024, 1024)), "ok:"));
  CHECK(why(req("birefnet-lite", 3, 512, 512, 1, 512, 512)) == "model birefnet-lite takes 1024x1024 only, not 512x512");
  CHECK(contains(why(req("birefnet-lite", 3, 1024, 1056, 1, 1024, 1056)), "1024x1024 only"));
  CHECK(why(req("birefnet-lite", 3, 512, 1024, 1, 512, 1024)) == "model birefnet-lite takes 1024x1024 only, not 1024x512");
  CHECK(contains(why(req("vitmatte-small", 4, 512, 512, 1, 256, 512)), "output size 512x256"));
  CHECK(contains(why(req("vitmatte-small", 4, 512, 512, 1, 512, 256)), "output size 256x512"));
  // Huge values never overflow into a pass: the table's checks come first.
  CHECK(contains(why(req("vitmatte-small", 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 1, 1, 1)), "input channels"));
  CHECK(contains(why(req("vitmatte-small", 4, 0x80000000u, 0x80000000u, 1, 0x80000000u, 0x80000000u)), "size"));
  // The slot: exactly enough passes, one float less does not.
  const uint64_t need = proto::slotRunBytes(req("vitmatte-small", 4, 512, 512, 1, 512, 512));
  CHECK(why(req("vitmatte-small", 4, 512, 512, 1, 512, 512), need) == "ok:vitmatte-small");
  CHECK(contains(why(req("vitmatte-small", 4, 512, 512, 1, 512, 512), need - 4), "needs 5242880 bytes"));
  CHECK(contains(why(req("birefnet-lite", 3, 1024, 1024, 1, 1024, 1024), uint64_t(16) << 20), "ok:"));
  CHECK(contains(why(req("birefnet-lite", 3, 1024, 1024, 1, 1024, 1024), (uint64_t(16) << 20) - 1), "shared memory"));
  std::printf("model table and request validation: ok (%d models, largest run %llu bytes)\n", modelCount(),
              (unsigned long long)maxRunBytes());
}

// Two model shapes on one connection, the generic call and the whole-frame entry point.
void testTwoShapes() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("shapes"));
  CHECK(tileOk(c, 512, 100, 200, 0.5f));                              // ViTMatte tile (TileRunner)
  CHECK(runOk(c, "birefnet-lite", 3, 1024, 1024, 1, 0, 0, 0.25f));     // whole frame 3 -> 1
  CHECK(runOk(c, "vitmatte-base", 4, 64, 1024, 1, 7, -3, 2.0f));      // non-square tile
  CHECK(runOk(c, "birefnet", 3, 1024, 1024, 1, 0, 0, -1.0f));
  CHECK(tileOk(c, 1024, 13, 4000, 3.0f));                             // the largest ViTMatte tile
  {
    std::vector<float> in = pattern(size_t(3) * 1024 * 1024, 0.75f), out(size_t(1024) * 1024, -1.0f);
    c.runFrame("birefnet-lite", in.data(), 3, 1024, 1024, out.data(), 1);
    CHECK(answerOk(in, 3, 1024, 1024, out, 1, 0, 0));
  }
  CHECK(c.slotBytes() == proto::kDefaultSlotBytes && c.stats().slotGrown == 0 && c.stats().reconnects == 0);
  CHECK(c.stats().tiles == 6);
  proto::StatusReply st;
  std::string err;
  CHECK(c.status(&st, &err) && st.sessions == 1 && ipc::getField(st.session[0].model) == "birefnet-lite");
  std::printf("two model shapes on one connection: ok\n");
}

// Bad requests through the client: clean errors from the runner, the connection survives.
void testBadShapes() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("badshapes"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const int pid = c.runnerPid();
  const std::string e1 = runErr(c, "vitmatte-small", 3, 512, 512, 1, 512, 512);
  CHECK(e1 == "runner rejected the run: model vitmatte-small takes 4 input channels, not 3");
  const std::string e2 = runErr(c, "no-such-model", 4, 512, 512, 1, 512, 512);
  CHECK(e2 == "runner rejected the run: unknown model 'no-such-model'");
  const std::string e3 = runErr(c, "birefnet-lite", 3, 512, 512, 1, 512, 512);
  CHECK(contains(e3, "rejected") && contains(e3, "1024x1024 only"));
  const std::string e4 = runErr(c, "vitmatte-small", 4, 512, 512, 1, 256, 256);
  CHECK(contains(e4, "rejected") && contains(e4, "output size"));
  const std::string e5 = runErr(c, "vitmatte-small", 4, 512, 512, 2, 512, 512);
  CHECK(contains(e5, "rejected") && contains(e5, "output channels"));
  CHECK(c.stats().slotGrown == 0 && c.stats().reconnects == 0 && c.connected());
  // Too large for the slot: the client grows it (a new connection), the runner refuses the size.
  const std::string e6 = runErr(c, "vitmatte-small", 4, 2048, 2048, 1, 2048, 2048);
  CHECK(contains(e6, "rejected") && contains(e6, "size 2048x2048"));
  CHECK(c.stats().slotGrown == 1 && c.slotBytes() == size_t(80) << 20);
  // Beyond the cap: refused by the client, nothing sent.
  const std::string e7 = runErr(c, "birefnet", 3, 8192, 8192, 1, 8192, 8192);
  CHECK(contains(e7, "too large") && contains(e7, "256 MiB"));
  const std::string e8 = runErr(c, "", 4, 64, 64, 1, 64, 64);
  CHECK(contains(e8, "bad model run"));
  CHECK(runOk(c, "birefnet-lite", 3, 1024, 1024, 1, 0, 0, 0.5f));  // still fine, same runner
  CHECK(tileOk(c, 512, 0, 0, 1.0f));
  CHECK(c.runnerPid() == pid && c.stats().runnerStarts == 0 && c.stats().reconnects == 0 && c.stats().slotGrown == 1);
  std::printf("bad shapes: ok ('%s' / '%s' / '%s')\n", e1.c_str(), e2.c_str(), e6.c_str());
}

// The wire without RunnerClient: a slot that does not fit the run, bad HELLO slots.
struct Raw {
  int fd = -1, slot = -1;
  uint32_t seq = 0;
  ~Raw() {
    if (fd >= 0) close(fd);
    if (slot >= 0) close(slot);
  }
  // HELLO with a slot of `real` bytes declared as `declared` (real = 0: no fd). The reply's status.
  uint32_t hello(uint64_t real, uint64_t declared, std::string* msg) {
    int e = 0;
    fd = ipc::connectUnix(gSock, &e, nullptr);
    if (fd < 0) return 999;
    std::string err;
    if (real) slot = ipc::createSlot(size_t(real), &err);
    proto::Hello h{};
    h.version = proto::kVersion;
    h.slotBytes = declared;
    ipc::setField(h.buildId, std::string(runnerBuildId()));
    ipc::setField(h.client, std::string("raw"));
    proto::MsgHeader hd{};
    char buf[proto::kMaxPayload];
    if (ipc::sendMsg(fd, proto::kHello, ++seq, &h, sizeof h, ipc::nowMs() + 3000, slot) != ipc::Io::Ok ||
        ipc::recvMsg(fd, &hd, buf, sizeof buf, ipc::nowMs() + 3000) != ipc::Io::Ok || hd.type != proto::kHelloReply)
      return 998;
    proto::HelloReply r{};
    std::memcpy(&r, buf, sizeof r);
    *msg = ipc::getField(r.message);
    return r.status;
  }
  // One RUN; the reply's status and message (999 on a transport failure).
  uint32_t run(const proto::RunRequest& q, std::string* msg) {
    proto::MsgHeader hd{};
    char buf[proto::kMaxPayload];
    if (ipc::sendMsg(fd, proto::kRun, ++seq, &q, sizeof q, ipc::nowMs() + 3000) != ipc::Io::Ok ||
        ipc::recvMsg(fd, &hd, buf, sizeof buf, ipc::nowMs() + 3000) != ipc::Io::Ok || hd.type != proto::kRunReply)
      return 999;
    proto::RunReply r{};
    std::memcpy(&r, buf, sizeof r);
    *msg = ipc::getField(r.message);
    return r.status;
  }
};

void testRawWire() {
  writeFile("fake.mode", "normal");
  {
    RunnerClient c(opts("raw-start"));  // a runner of this build
    CHECK(tileOk(c, 64, 0, 0, 0.0f));
  }
  std::string m;
  {
    Raw r;  // a 1 MiB slot: a 512 tile (5 MiB) does not fit, a 128 tile does
    CHECK(r.hello(1 << 20, 1 << 20, &m) == proto::kOk);
    CHECK(r.run(req("vitmatte-small", 4, 512, 512, 1, 512, 512), &m) == proto::kBadRequest);
    CHECK(m == "the run needs 5242880 bytes of shared memory, the slot has 1048576");
    CHECK(r.run(req("vitmatte-small", 4, 128, 128, 1, 128, 128), &m) == proto::kOk && m.empty());
    proto::RunRequest q = req("vitmatte-small", 4, 128, 128, 1, 128, 128);
    q.slot = 2;
    CHECK(r.run(q, &m) == proto::kBadRequest && m == "slot 2 does not exist");
    CHECK(r.run(req("birefnet-lite", 3, 1024, 1024, 1, 1024, 1024), &m) == proto::kBadRequest &&
          contains(m, "needs 16777216 bytes"));
    CHECK(r.run(req("vitmatte-small", 4, 128, 128, 1, 128, 128), &m) == proto::kOk);  // still served
  }
  {
    Raw r;  // HELLO declaring more than the memory file holds
    CHECK(r.hello(1 << 20, 2 << 20, &m) == proto::kBadSlot && contains(m, "smaller than"));
    CHECK(r.run(req("vitmatte-small", 4, 32, 32, 1, 32, 32), &m) == proto::kBadRequest &&
          m == "no shared memory on this connection");
  }
  {
    Raw r;  // above the cap (the file itself is small: the size check comes first)
    CHECK(r.hello(1 << 20, proto::kMaxSlotBytes + 1, &m) == proto::kBadSlot && contains(m, "slot size"));
  }
  {
    Raw r;  // below the minimum
    CHECK(r.hello(4096, 1024, &m) == proto::kBadSlot && contains(m, "slot size 1024"));
  }
  {
    Raw r;  // exactly the cap is accepted
    CHECK(r.hello(proto::kMaxSlotBytes, proto::kMaxSlotBytes, &m) == proto::kOk);
  }
  {
    // A client asking for a slot above the cap (a slot smaller than a run grows instead).
    RunnerClientOptions o = opts("big-slot");
    o.slotBytes = proto::kMaxSlotBytes + (1 << 20);
    RunnerClient c(o);
    const std::string e = runError(c);
    CHECK(contains(e, "runner refused the connection") && contains(e, "slot size 269484032"));
  }
  std::printf("raw wire: bad slots and runs that do not fit: ok\n");
}

void testSlow() {
  writeFile("fake.mode", "slow:300");
  RunnerClient c(opts("slow"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  CHECK(c.stats().runMs >= 290 && c.stats().roundTripMs >= 290);
  writeFile("fake.mode", "normal");
  std::printf("slow: ok (round trip %.0f ms)\n", c.stats().roundTripMs);
}

void testTwoClients() {
  writeFile("fake.mode", "slow:2");
  std::atomic<int> bad{0};
  std::vector<double> queue(2, 0);
  auto work = [&](int id) {
    RunnerClient c(opts(id ? "B" : "A"));
    for (int i = 0; i < 40; ++i) {
      try {
        if (!tileOk(c, id ? 256 : 128, i * 100 + id, id * 1000 + i, float(id) + i * 0.01f)) ++bad;
      } catch (const std::exception& e) {
        std::printf("  client %d: %s\n", id, e.what());
        ++bad;
      }
    }
    queue[id] = c.stats().queueMs;
  };
  std::thread a(work, 0), b(work, 1);
  a.join();
  b.join();
  CHECK(bad == 0);
  CHECK(queue[0] + queue[1] > 0);  // they did wait for each other
  writeFile("fake.mode", "normal");
  std::printf("two clients interleaved: ok (queue wait A %.0f ms, B %.0f ms)\n", queue[0], queue[1]);
}

void testCrash() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("crash"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  writeFile("fake.mode", "crash");
  const int s0 = starts();
  const int64_t t0 = ipc::nowMs();
  const std::string e = runError(c);
  CHECK(contains(e, "connection lost"));
  CHECK(starts() == s0 + 1 && c.stats().reconnects == 1);  // one restart, then the error
  CHECK(ipc::nowMs() - t0 < 5000);
  writeFile("fake.mode", "normal");
  CHECK(tileOk(c, 64, 0, 0, 0.0f));  // the next render recovers
  // Crashing once: the retry succeeds.
  unlink((gDir + "/fake.crashed").c_str());
  writeFile("fake.mode", "crash-once");
  RunnerClient d(opts("crash-once"));
  CHECK(tileOk(d, 64, 5, 5, 0.0f));
  CHECK(d.stats().reconnects == 1 && d.stats().runnerStarts == 1);
  writeFile("fake.mode", "normal");
  std::printf("crash mid-tile: ok ('%s')\n", e.c_str());
}

void testHang() {
  writeFile("fake.mode", "normal");
  RunnerClientOptions o = opts("hang");
  o.tileTimeoutS = 1;
  RunnerClient c(o);
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const int pid = c.runnerPid();
  writeFile("fake.mode", "hang");
  const int64_t t0 = ipc::nowMs();
  const std::string e = runError(c);
  const int64_t took = ipc::nowMs() - t0;
  // The tile timeout (1 s), then the status probe (1 s) is not answered either: the client stops it.
  CHECK(contains(e, "timed out") && contains(e, "stopped") && took >= 1900 && took < 4000);
  CHECK(!c.connected() && c.stats().hungStopped == 1);
  CHECK(waitDead(pid, 3000));
  writeFile("fake.mode", "normal");
  const int s0 = starts();
  CHECK(tileOk(c, 64, 0, 0, 0.0f));  // the next frame: a fresh runner at once, no second timeout
  CHECK(starts() == s0 + 1 && c.runnerPid() != pid && c.stats().hungStopped == 1);
  std::printf("hang: ok (timed out and stopped after %lld ms: '%s')\n", (long long)took, e.c_str());
}

// A tile that answers after the timeout but within the probe: the runner is busy, not hung, and
// keeps running (no kill), the next tile goes to the same runner.
void testSlowNotKilled() {
  writeFile("fake.mode", "normal");
  RunnerClientOptions o = opts("slow-probe");
  o.tileTimeoutS = 1;
  o.probeTimeoutS = 1.5;
  RunnerClient c(o);
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const int pid = c.runnerPid();
  writeFile("fake.mode", "slow:1600");
  const std::string e = runError(c);
  writeFile("fake.mode", "normal");
  CHECK(contains(e, "timed out") && !contains(e, "stopped"));
  CHECK(c.stats().hungStopped == 0 && alive(pid));
  const int s0 = starts();
  c.resetStats();
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  CHECK(c.runnerPid() == pid && starts() == s0 && c.stats().runnerStarts == 0);
  std::printf("slow but answering: not stopped: ok ('%s')\n", e.c_str());
}

// Never a blind pid: a "runner" that reports another process's pid (a child of this test) and
// never answers: after the timeout and the failed probe the socket's peer (this process) is not the
// reported pid, so nothing is killed.
void testProbePeerCheck() {
  const pid_t child = fork();
  if (child == 0) {
    sleep(30);
    _exit(0);
  }
  const std::string sock = gDir + "/peer.sock";
  std::string err;
  const int lfd = ipc::listenUnix(sock, &err);
  CHECK(lfd >= 0);
  std::atomic<bool> stop{false};
  std::thread srv([&] {
    std::vector<int> open;
    while (!stop) {
      pollfd p{lfd, POLLIN, 0};
      if (poll(&p, 1, 50) <= 0) continue;
      const int c = accept(lfd, nullptr, nullptr);
      if (c < 0) continue;
      open.push_back(c);
      proto::MsgHeader h{};
      char buf[proto::kMaxPayload];
      int slot = -1;
      if (ipc::recvMsg(c, &h, buf, sizeof buf, ipc::nowMs() + 2000, &slot) != ipc::Io::Ok) continue;
      if (slot >= 0) close(slot);
      proto::HelloReply rep{};
      rep.version = proto::kVersion;
      rep.pid = uint32_t(child);
      ipc::setField(rep.buildId, std::string(runnerBuildId()));
      rep.status = proto::kOk;
      ipc::sendMsg(c, proto::kHelloReply, h.seq, &rep, sizeof rep, ipc::nowMs() + 2000);
      // then silence: RUN and STATUS are never answered
    }
    for (int c : open) close(c);
  });
  RunnerClientOptions o = opts("peer");
  o.socketPath = sock;
  o.tileTimeoutS = 0.5;
  o.probeTimeoutS = 0.5;
  RunnerClient c(o);
  const std::string e = runError(c);
  CHECK(contains(e, "timed out") && !contains(e, "stopped"));
  CHECK(c.runnerPid() == int(child) && c.stats().hungStopped == 0);
  usleep(100000);
  CHECK(alive(child));
  stop = true;
  srv.join();
  close(lfd);
  unlink(sock.c_str());
  kill(child, SIGKILL);
  waitpid(child, nullptr, 0);
  std::printf("probe peer check: nothing killed: ok ('%s')\n", e.c_str());
}

// Two plugin builds on one machine, each with its default (per-build) socket: two runners at once,
// neither replaces the other; one quitting removes only its own socket.
void testTwoBuilds() {
  writeFile("fake.mode", "normal");
  const char* old = std::getenv("XDG_RUNTIME_DIR");
  const std::string saved = old ? old : "";
  const std::string xdg = gDir + "/xdg2";
  mkdir(xdg.c_str(), 0700);
  setenv("XDG_RUNTIME_DIR", xdg.c_str(), 1);
  auto build = [&](const char* id) {
    RunnerClientOptions o = opts(id);
    o.socketPath = "";
    o.buildId = id;
    o.extraArgs = {"--build", id};
    return o;
  };
  {
    RunnerClient a(build("build A")), b(build("build B"));
    for (int i = 0; i < 3; ++i) {
      CHECK(tileOk(a, 64, i, 0, 0.0f));
      CHECK(tileOk(b, 64, 0, i, 1.0f));
      a.disconnect();  // a new connection each round: the handshake runs again
      b.disconnect();
    }
    CHECK(a.socketPath() != b.socketPath() && a.runnerPid() != b.runnerPid());
    CHECK(a.stats().replaced == 0 && b.stats().replaced == 0);
    CHECK(a.stats().runnerStarts == 1 && b.stats().runnerStarts == 1);
    CHECK(a.runnerBuild() == "build A" && b.runnerBuild() == "build B");
    const int pa = a.runnerPid(), pb = b.runnerPid();
    const std::string sa = a.socketPath(), sb = b.socketPath();
    std::string err;
    CHECK(a.quit(&err) && waitDead(pa, 3000));
    CHECK(access(sa.c_str(), F_OK) != 0 && access(sb.c_str(), F_OK) == 0 && alive(pb));
    CHECK(tileOk(b, 64, 0, 0, 0.0f) && b.runnerPid() == pb);
    CHECK(b.quit(&err) && waitDead(pb, 3000) && access(sb.c_str(), F_OK) != 0);
  }
  if (old) setenv("XDG_RUNTIME_DIR", saved.c_str(), 1);
  else unsetenv("XDG_RUNTIME_DIR");
  std::printf("two builds, two runners: ok\n");
}

// The runner's stderr file is started over when a runner starts and finds it above the cap; a small
// one is kept. The runner's log lines carry the component "runner", the client's "ofx".
void testStderrCapAndLogComponent() {
  writeFile("fake.mode", "normal");
  stopRunner();
  const std::string dir = gHome + "/.cache/matte-refine/logs", se = dir + "/runner-stderr.log";
  CHECK(std::system(("mkdir -p '" + dir + "'").c_str()) == 0);
  auto sizeOf = [](const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 ? (long long)st.st_size : -1LL;
  };
  {
    std::ofstream f(se, std::ios::binary | std::ios::trunc);
    f << std::string(size_t(kStderrCapBytes) + 4096, 'x');
  }
  {
    RunnerClient c(opts("cap"));
    CHECK(tileOk(c, 64, 0, 0, 0.0f));
  }
  CHECK(sizeOf(se) >= 0 && sizeOf(se) < 4096);
  stopRunner();
  { std::ofstream(se, std::ios::trunc) << "earlier line\n"; }
  {
    RunnerClient c(opts("cap2"));
    CHECK(tileOk(c, 64, 0, 0, 0.0f));
  }
  CHECK(readFile(se).compare(0, 13, "earlier line\n") == 0);
  stopRunner();
  const std::string log = readFile(dir + "/matte-refine.log");
  int runnerLines = 0, wrong = 0;
  std::istringstream in(log);
  for (std::string l; std::getline(in, l);) {
    if (!contains(l, "listening at")) continue;
    contains(l, " runner[") ? ++runnerLines : ++wrong;
  }
  CHECK(runnerLines >= 2 && wrong == 0);
  CHECK(contains(log, " ofx[") && contains(log, "starting the runner"));  // the client's own lines
  std::printf("stderr cap and log component: ok (%d runner start lines)\n", runnerLines);
}

void testKilledBetweenFrames() {
  writeFile("fake.mode", "normal");
  RunnerClient c(opts("killed"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  const int pid = c.runnerPid();
  kill(pid, SIGKILL);
  CHECK(waitDead(pid, 3000));
  CHECK(access(gSock.c_str(), F_OK) == 0);  // a stale socket is left behind
  c.resetStats();
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  CHECK(c.stats().reconnects == 1 && c.stats().runnerStarts == 1 && c.runnerPid() != pid);
  std::printf("killed between frames: ok (new runner %d)\n", c.runnerPid());
}

void testMismatch() {
  writeFile("fake.mode", "normal");
  // An old build running; the plugin's build replaces it exactly once.
  stopRunner();
  writeFile("fake.build", "old build");
  RunnerClientOptions oo = opts("old");
  oo.buildId = "old build";
  RunnerClient old(oo);
  CHECK(tileOk(old, 64, 0, 0, 0.0f));
  const int oldPid = old.runnerPid();
  writeFile("fake.build", "");
  int s0 = starts();
  RunnerClient c(opts("new"));
  CHECK(tileOk(c, 64, 0, 0, 0.0f));
  CHECK(c.stats().replaced == 1 && c.stats().runnerStarts == 1 && starts() == s0 + 1);
  CHECK(waitDead(oldPid, 3000) && c.runnerPid() != oldPid && c.runnerBuild() == runnerBuildId());
  // A runner that is still the wrong build after the replacement: an error, no second try.
  stopRunner();
  writeFile("fake.build", "old build");
  RunnerClient old2(oo);
  CHECK(tileOk(old2, 64, 0, 0, 0.0f));
  s0 = starts();
  RunnerClient d(opts("new2"));
  const std::string e = runError(d);
  CHECK(contains(e, "version mismatch") && contains(e, "old build"));
  CHECK(d.stats().replaced == 1 && d.stats().runnerStarts == 1 && starts() == s0 + 1);
  // Another protocol version: also told to quit and replaced.
  stopRunner();
  writeFile("fake.build", "");
  writeFile("fake.version", "99");
  RunnerClient v99(opts("v99"));
  const std::string e99 = runError(v99);  // the client starts a runner that speaks 99: mismatch
  std::printf("  protocol 99: '%s'\n", e99.c_str());
  CHECK(contains(e99, "version mismatch") && contains(e99, "protocol 99"));
  writeFile("fake.version", "");
  s0 = starts();
  RunnerClient f(opts("after99"));  // finds the 99 runner, quits it, starts the right one
  CHECK(tileOk(f, 64, 0, 0, 0.0f));
  CHECK(f.stats().replaced == 1 && starts() == s0 + 1);
  std::printf("mismatch: ok ('%s')\n", e.c_str());
}

void testOneStartForTwo() {
  writeFile("fake.mode", "normal");
  stopRunner();
  CHECK(access(gSock.c_str(), F_OK) != 0);
  const int s0 = starts();
  std::atomic<int> ok{0};
  auto go = [&](const char* n) {
    RunnerClient c(opts(n));
    try {
      ok += tileOk(c, 64, 0, 0, 0.0f);
    } catch (const std::exception& e) {
      std::printf("  %s: %s\n", n, e.what());
    }
  };
  std::thread a(go, "first"), b(go, "second");
  a.join();
  b.join();
  CHECK(ok == 2 && starts() == s0 + 1);
  std::printf("two clients at once start one runner: ok (%d start)\n", starts() - s0);
}

void testIdleExit() {
  stopRunner();
  RunnerClientOptions o = opts("idle");
  o.idleSeconds = 1;
  int pid;
  {
    RunnerClient c(o);
    CHECK(tileOk(c, 64, 0, 0, 0.0f));
    pid = c.runnerPid();
    usleep(1500000);  // connected: no idle exit
    CHECK(alive(pid));
  }
  CHECK(socketGone(4000) && waitDead(pid, 2000));
  std::printf("idle exit: ok\n");
}

void testNotFound() {
  stopRunner();
  RunnerClientOptions o = opts("missing");
  o.runnerPath = gDir + "/no-such-runner";
  RunnerClient c(o);
  const std::string e = runError(c);
  CHECK(e == "runner not found: " + o.runnerPath);
  std::printf("runner not found: ok ('%s')\n", e.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  gFake = argv[1];
  char tmpl[] = "/tmp/mrml-test-XXXXXX";  // short: a socket path holds at most 107 bytes
  if (!mkdtemp(tmpl)) return 2;
  gDir = tmpl;
  gSock = gDir + "/runner.sock";
  gHome = gDir + "/home";
  mkdir(gHome.c_str(), 0700);
  setenv("HOME", gHome.c_str(), 1);  // this process's own log lines stay out of the user's log
  signal(SIGPIPE, SIG_IGN);
  // An unexpected exception (a tile that should have worked) fails that test, not the run.
  auto test = [](const char* name, void (*fn)()) {
    try {
      fn();
    } catch (const std::exception& e) {
      std::printf("FAIL %s: unexpected exception: %s\n", name, e.what());
      ++failures;
    }
  };
  test("framing", testFraming);
  test("socket folder", testSocketFolder);
  test("table and validation", testTableAndValidation);
  test("normal", testNormal);
  test("two shapes", testTwoShapes);
  test("bad shapes", testBadShapes);
  test("raw wire", testRawWire);
  test("environment and fds", testDetachedEnvAndFds);
  test("bad request", testBadRequestAndModelError);
  test("slow", testSlow);
  test("two clients", testTwoClients);
  test("crash", testCrash);
  test("hang", testHang);
  test("slow not killed", testSlowNotKilled);
  test("probe peer check", testProbePeerCheck);
  test("killed between frames", testKilledBetweenFrames);
  test("mismatch", testMismatch);
  test("one start for two", testOneStartForTwo);
  test("idle exit", testIdleExit);
  test("not found", testNotFound);
  test("two builds", testTwoBuilds);
  test("stderr cap and log component", testStderrCapAndLogComponent);
  stopRunner();
  if (failures) std::printf("kept %s for inspection\n", gDir.c_str());
  else if (std::system(("rm -rf '" + gDir + "'").c_str()) != 0) std::printf("could not remove %s\n", gDir.c_str());
  std::printf(failures ? "test_runner_client: %d FAILED\n" : "test_runner_client: all passed\n", failures);
  return failures ? 1 : 0;
}
