// Socket, message and shared-memory helpers for the runner protocol (runner_protocol.h), shared by
// the client (in the plugin) and the runner. C++14, POSIX; the Linux / macOS differences are in
// runner_ipc.cpp only (macOS: written, not built yet, decision 13). Every fd is close-on-exec.
#pragma once

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "runner_protocol.h"

namespace mr {
namespace ml {
namespace ipc {

// Monotonic milliseconds; deadlines below are absolute values of it, < 0 = no deadline.
int64_t nowMs();
int64_t nowUs();  // the same clock in microseconds, for timings

// $XDG_RUNTIME_DIR/matte-refine-ml/<name>, else /tmp/matte-refine-ml-<uid>/<name>; name =
// socketName(buildId). The folder is created (mode 0700) if missing and must be a real folder owned
// by this user with no group / other access (else "" and *err: another user could otherwise plant a
// socket there).
std::string defaultSocketPath(std::string* err, const std::string& buildId = std::string());
// "runner-<8 hex digits of a hash of buildId>.sock": one runner per plugin build, so two builds on
// one machine (two Flame versions) do not replace each other's runner. "" -> "runner.sock".
std::string socketName(const std::string& buildId);
bool ensurePrivateDir(const std::string& dir, std::string* err);
std::string dirName(const std::string& path);

// A connected stream socket, or -1 with *err (errno in *errnoOut). Linux: sockaddr_un holds at most
// 107 bytes of path.
int connectUnix(const std::string& path, int* errnoOut, std::string* err);
// A listening socket at path (the caller has removed a stale one), or -1 with *err.
int listenUnix(const std::string& path, std::string* err);
// The peer's uid (SO_PEERCRED / getpeereid); false if unknown.
bool peerUid(int fd, uid_t* uid, pid_t* pid);

enum class Io { Ok, Eof, Timeout, Error, Protocol };
const char* ioName(Io r);

Io writeAll(int fd, const void* data, size_t n, int64_t deadline);
Io readAll(int fd, void* data, size_t n, int64_t deadline);

// One message: the header + payload, with passFd (>= 0) attached as SCM_RIGHTS.
Io sendMsg(int fd, uint16_t type, uint32_t seq, const void* payload, uint32_t bytes, int64_t deadline,
           int passFd = -1);
// The next message: header into *h, payload (h->bytes, <= cap) into buf. An fd passed with it lands
// in *gotFd (if gotFd; else it is closed). A wrong magic or a payload over cap = Io::Protocol.
Io recvMsg(int fd, proto::MsgHeader* h, void* buf, size_t cap, int64_t deadline, int* gotFd = nullptr);

// Shared memory for one slot: an anonymous file of `bytes` (Linux: memfd_create, sealed against
// shrinking and growing; macOS: shm_open + shm_unlink). -1 on failure with *err.
int createSlot(size_t bytes, std::string* err);
// The runner's check of a received slot fd: a regular memory file of at least `bytes`, sealed on Linux.
bool checkSlot(int fd, size_t bytes, std::string* err);
void* mapSlot(int fd, size_t bytes, std::string* err);  // nullptr on failure
void unmapSlot(void* p, size_t bytes);

// Copies s into a fixed char field, NUL-terminated (truncated if needed).
void setField(char* dst, size_t cap, const std::string& s);
template <size_t N>
void setField(char (&dst)[N], const std::string& s) {
  setField(dst, N, s);
}
template <size_t N>
std::string getField(const char (&src)[N]) {
  size_t n = 0;
  while (n < N && src[n]) ++n;
  return std::string(src, n);
}

}  // namespace ipc
}  // namespace ml
}  // namespace mr
