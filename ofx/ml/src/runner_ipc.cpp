// runner_ipc.h. POSIX; Linux and macOS differences marked.
#include "runner_ipc.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mr {
namespace ml {
namespace ipc {

namespace {

#if defined(__linux__)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // macOS: SO_NOSIGPIPE on the socket instead
#endif

void setCloexec(int fd) {
  const int f = fcntl(fd, F_GETFD);
  if (f >= 0) fcntl(fd, F_SETFD, f | FD_CLOEXEC);
}

int newSocket() {
#if defined(__linux__)
  return socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd >= 0) {
    setCloexec(fd);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
  }
  return fd;
#endif
}

bool fillAddr(const std::string& path, sockaddr_un* a, std::string* err) {
  std::memset(a, 0, sizeof *a);
  a->sun_family = AF_UNIX;
  if (path.size() >= sizeof a->sun_path) {
    if (err) *err = "socket path too long (" + std::to_string(path.size()) + " bytes): " + path;
    return false;
  }
  std::memcpy(a->sun_path, path.c_str(), path.size() + 1);
  return true;
}

// Waits until fd is readable (POLLIN) or writable; false on timeout. EINTR retried.
bool waitFd(int fd, short events, int64_t deadline) {
  for (;;) {
    int timeout = -1;
    if (deadline >= 0) {
      const int64_t left = deadline - nowMs();
      if (left <= 0) return false;
      timeout = int(left > 1000000 ? 1000000 : left);
    }
    pollfd p{fd, events, 0};
    const int r = poll(&p, 1, timeout);
    if (r > 0) return true;  // also on POLLHUP / POLLERR: the read / write then reports it
    if (r < 0 && errno != EINTR) return true;
  }
}

std::string errnoStr(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

}  // namespace

int64_t nowMs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

int64_t nowUs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

std::string dirName(const std::string& path) {
  const size_t p = path.rfind('/');
  return p == std::string::npos ? "." : p == 0 ? "/" : path.substr(0, p);
}

bool ensurePrivateDir(const std::string& dir, std::string* err) {
  if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
    if (err) *err = errnoStr(("cannot create " + dir).c_str());
    return false;
  }
  struct stat st {};
  if (lstat(dir.c_str(), &st) != 0) {
    if (err) *err = errnoStr(dir.c_str());
    return false;
  }
  if (!S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & 077) != 0) {
    if (err) *err = dir + " is not a private folder of this user (owner, mode 0700): not used";
    return false;
  }
  return true;
}

std::string socketName(const std::string& buildId) {
  if (buildId.empty()) return "runner.sock";
  uint32_t h = 2166136261u;  // FNV-1a, 32 bit
  for (const unsigned char c : buildId) h = (h ^ c) * 16777619u;
  char name[32];
  std::snprintf(name, sizeof name, "runner-%08x.sock", unsigned(h));
  return name;
}

std::string defaultSocketPath(std::string* err, const std::string& buildId) {
  std::string dir;
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  if (xdg && *xdg == '/') {
    struct stat st {};
    if (stat(xdg, &st) == 0 && S_ISDIR(st.st_mode) && st.st_uid == getuid()) dir = std::string(xdg) + "/matte-refine-ml";
  }
  if (dir.empty()) dir = "/tmp/matte-refine-ml-" + std::to_string(getuid());
  if (!ensurePrivateDir(dir, err)) return "";
  return dir + "/" + socketName(buildId);
}

int connectUnix(const std::string& path, int* errnoOut, std::string* err) {
  sockaddr_un a;
  if (errnoOut) *errnoOut = 0;
  if (!fillAddr(path, &a, err)) {
    if (errnoOut) *errnoOut = ENAMETOOLONG;
    return -1;
  }
  const int fd = newSocket();
  if (fd < 0) {
    if (errnoOut) *errnoOut = errno;
    if (err) *err = errnoStr("socket");
    return -1;
  }
  int r;
  do r = connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a);
  while (r != 0 && errno == EINTR);
  if (r != 0) {
    if (errnoOut) *errnoOut = errno;
    if (err) *err = errnoStr(("connect " + path).c_str());
    close(fd);
    return -1;
  }
  return fd;
}

int listenUnix(const std::string& path, std::string* err) {
  sockaddr_un a;
  if (!fillAddr(path, &a, err)) return -1;
  const int fd = newSocket();
  if (fd < 0) {
    if (err) *err = errnoStr("socket");
    return -1;
  }
  const mode_t old = umask(077);  // the socket file itself: owner only
  const int r = bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a);
  umask(old);
  if (r != 0 || listen(fd, 64) != 0) {
    if (err) *err = errnoStr(("bind/listen " + path).c_str());
    close(fd);
    return -1;
  }
  return fd;
}

bool peerUid(int fd, uid_t* uid, pid_t* pid) {
#if defined(__linux__)
  ucred c{};
  socklen_t n = sizeof c;
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &c, &n) != 0) return false;
  *uid = c.uid;
  if (pid) *pid = c.pid;
  return true;
#else
  gid_t g;
  if (getpeereid(fd, uid, &g) != 0) return false;
  if (pid) *pid = -1;
  return true;
#endif
}

const char* ioName(Io r) {
  switch (r) {
    case Io::Ok: return "ok";
    case Io::Eof: return "connection closed";
    case Io::Timeout: return "timed out";
    case Io::Protocol: return "protocol error";
    default: return "i/o error";
  }
}

Io writeAll(int fd, const void* data, size_t n, int64_t deadline) {
  const char* p = static_cast<const char*>(data);
  while (n > 0) {
    if (!waitFd(fd, POLLOUT, deadline)) return Io::Timeout;
    const ssize_t r = send(fd, p, n, kSendFlags | MSG_DONTWAIT);
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return errno == EPIPE || errno == ECONNRESET ? Io::Eof : Io::Error;
    }
    p += r;
    n -= size_t(r);
  }
  return Io::Ok;
}

Io readAll(int fd, void* data, size_t n, int64_t deadline) {
  char* p = static_cast<char*>(data);
  while (n > 0) {
    if (!waitFd(fd, POLLIN, deadline)) return Io::Timeout;
    const ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
    if (r == 0) return Io::Eof;
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return errno == ECONNRESET ? Io::Eof : Io::Error;
    }
    p += r;
    n -= size_t(r);
  }
  return Io::Ok;
}

Io sendMsg(int fd, uint16_t type, uint32_t seq, const void* payload, uint32_t bytes, int64_t deadline,
           int passFd) {
  proto::MsgHeader h{proto::kMagic, type, 0, bytes, seq};
  if (passFd < 0) {
    const Io r = writeAll(fd, &h, sizeof h, deadline);
    return r != Io::Ok || bytes == 0 ? r : writeAll(fd, payload, bytes, deadline);
  }
  // The fd rides on the header's first byte; the rest goes as a normal write.
  if (!waitFd(fd, POLLOUT, deadline)) return Io::Timeout;
  iovec iov{&h, sizeof h};
  char ctrl[CMSG_SPACE(sizeof(int))];
  std::memset(ctrl, 0, sizeof ctrl);
  msghdr m{};
  m.msg_iov = &iov;
  m.msg_iovlen = 1;
  m.msg_control = ctrl;
  m.msg_controllen = sizeof ctrl;
  cmsghdr* c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(c), &passFd, sizeof(int));
  ssize_t r;
  do r = sendmsg(fd, &m, kSendFlags);
  while (r < 0 && errno == EINTR);
  if (r <= 0) return errno == EPIPE || errno == ECONNRESET ? Io::Eof : Io::Error;
  if (size_t(r) < sizeof h) {
    const Io w = writeAll(fd, reinterpret_cast<const char*>(&h) + r, sizeof h - size_t(r), deadline);
    if (w != Io::Ok) return w;
  }
  return bytes == 0 ? Io::Ok : writeAll(fd, payload, bytes, deadline);
}

Io recvMsg(int fd, proto::MsgHeader* h, void* buf, size_t cap, int64_t deadline, int* gotFd) {
  if (gotFd) *gotFd = -1;
  char* p = reinterpret_cast<char*>(h);
  size_t got = 0;
  while (got < sizeof *h) {
    if (!waitFd(fd, POLLIN, deadline)) return Io::Timeout;
    iovec iov{p + got, sizeof *h - got};
    char ctrl[CMSG_SPACE(sizeof(int) * 4)];
    msghdr m{};
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = ctrl;
    m.msg_controllen = sizeof ctrl;
#if defined(__linux__)
    const int flags = MSG_DONTWAIT | MSG_CMSG_CLOEXEC;
#else
    const int flags = MSG_DONTWAIT;
#endif
    const ssize_t r = recvmsg(fd, &m, flags);
    if (r == 0) return Io::Eof;
    if (r < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return errno == ECONNRESET ? Io::Eof : Io::Error;
    }
    for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
      if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
      const size_t n = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
      for (size_t i = 0; i < n; ++i) {
        int f;
        std::memcpy(&f, CMSG_DATA(c) + i * sizeof(int), sizeof f);
        setCloexec(f);
        if (gotFd && *gotFd < 0) {
          *gotFd = f;
        } else {
          close(f);  // unexpected or extra: never kept
        }
      }
    }
    got += size_t(r);
  }
  if (h->magic != proto::kMagic || h->bytes > cap || h->bytes > proto::kMaxPayload) return Io::Protocol;
  return h->bytes == 0 ? Io::Ok : readAll(fd, buf, h->bytes, deadline);
}

int createSlot(size_t bytes, std::string* err) {
#if defined(__linux__)
  const int fd = memfd_create("matte-refine-ml-slot", MFD_CLOEXEC | MFD_ALLOW_SEALING);  // glibc 2.27
  if (fd < 0) {
    if (err) *err = errnoStr("memfd_create");
    return -1;
  }
  if (ftruncate(fd, off_t(bytes)) != 0 || fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    if (err) *err = errnoStr("memfd size/seal");
    close(fd);
    return -1;
  }
  return fd;
#else
  char name[64];
  std::snprintf(name, sizeof name, "/mrml-%d-%lld", int(getpid()), (long long)nowMs());
  const int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    if (err) *err = errnoStr("shm_open");
    return -1;
  }
  shm_unlink(name);
  setCloexec(fd);
  if (ftruncate(fd, off_t(bytes)) != 0) {
    if (err) *err = errnoStr("shm size");
    close(fd);
    return -1;
  }
  return fd;
#endif
}

bool checkSlot(int fd, size_t bytes, std::string* err) {
  struct stat st {};
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || size_t(st.st_size) < bytes) {
    if (err) *err = "the shared memory is missing or smaller than " + std::to_string(bytes) + " bytes";
    return false;
  }
#if defined(__linux__)
  // A client that could shrink the file would crash the runner (SIGBUS) on the next tile.
  const int seals = fcntl(fd, F_GET_SEALS);
  if (seals < 0 || !(seals & F_SEAL_SHRINK)) {
    if (err) *err = "the shared memory is not sealed against shrinking";
    return false;
  }
#endif
  return true;
}

void* mapSlot(int fd, size_t bytes, std::string* err) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    if (err) *err = errnoStr("mmap");
    return nullptr;
  }
  return p;
}

void unmapSlot(void* p, size_t bytes) {
  if (p) munmap(p, bytes);
}

void setField(char* dst, size_t cap, const std::string& s) {
  const size_t n = s.size() < cap - 1 ? s.size() : cap - 1;
  std::memcpy(dst, s.data(), n);
  std::memset(dst + n, 0, cap - n);
}

}  // namespace ipc
}  // namespace ml
}  // namespace mr
