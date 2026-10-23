// The wire protocol between the plugin (RunnerClient) and mr_ml_runner (ofx/ml/CLAUDE.md decision
// 32). C++14, no OS headers: shared by the client, the runner and the tests.
//
// Transport: a Unix domain stream socket (runner_ipc.h socketPath()). Every message is a
// MsgHeader followed by `bytes` of payload, one of the fixed-size structs below; a reply echoes
// the request's `seq`. Fields are little-endian: the structs are sent as they are in memory and
// the build refuses a big-endian host (both supported hosts, x86-64 and arm64, are little-endian).
// Every struct has explicit padding and a static_assert on its size: the layout is the protocol.
//
// Connection: the client's first message is HELLO, with the shared-memory slot's fd attached
// (SCM_RIGHTS, one fd). The runner answers HELLO_REPLY (the first three fields of HelloReply never
// change, so any two versions can tell each other apart). A client that finds a different
// protocol or build sends QUIT and starts the matching runner (once). Then any number of RUN /
// STATUS requests, one at a time, each answered before the next is sent.
//
// RUN (protocol 2, model-generic): the request names a model of the table (model_table.h) by id
// and describes the tensors: input [1, cIn, H, W] float32 at slot offset 0, output
// [1, cOut, Ho, Wo] float32 right after it (slotOutputOffset). The runner checks every field
// against the table and the slot's size and answers a bad request with kBadRequest and the reason
// (the connection stays usable). One slot per connection (index 0), its size chosen by the client
// at HELLO (kDefaultSlotBytes, up to kMaxSlotBytes; a client that needs more reconnects with a
// larger one).
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the Matte Refine ML runner protocol is little-endian"
#endif

namespace mr {
namespace ml {
namespace proto {

constexpr uint32_t kMagic = 0x4C4D524Du;  // "MRML" in memory order
constexpr uint32_t kVersion = 2;          // bump on any change below (2: model-generic RUN)
constexpr uint32_t kMaxPayload = 4096;    // a header announcing more is a protocol error
// The slot: 20 MiB by default = a ViTMatte tile of 1024 (4 + 1 planes of 1024^2 floats), the
// largest run of today's table (a BiRefNet run, 3 + 1 planes at 1024^2, needs 16 MiB);
// test_runner_client checks maxRunBytes() <= kDefaultSlotBytes. Cap: 256 MiB (a UHD frame with
// 4 + 1 planes is 166 MiB).
constexpr uint64_t kDefaultSlotBytes = uint64_t(20) << 20;
constexpr uint64_t kMinSlotBytes = 4096;
constexpr uint64_t kMaxSlotBytes = uint64_t(256) << 20;
constexpr int kModelIdBytes = 32;  // a model id on the wire, NUL-terminated

enum MsgType : uint16_t {
  kHello = 1,
  kHelloReply = 2,
  kRun = 3,
  kRunReply = 4,
  kStatus = 5,
  kStatusReply = 6,
  kQuit = 7,
  kQuitReply = 8,
  kError = 9,  // the runner's answer to a request it cannot parse; it then closes the connection
};

// kEpDefault = the runner's config (ml_ep, default cuda).
enum Ep : uint32_t { kEpDefault = 0, kEpCuda = 1, kEpCpu = 2 };

enum Status : uint32_t {
  kOk = 0,
  kBadRequest = 1,  // the request does not fit the model table or the slot (the message says why)
  kModelError = 2,  // the model could not be loaded or failed to run
  kBadSlot = 3,     // HELLO: the shared memory is missing, too small or not sealed
};

struct MsgHeader {
  uint32_t magic;
  uint16_t type;
  uint16_t reserved;
  uint32_t bytes;  // payload size
  uint32_t seq;
};
static_assert(sizeof(MsgHeader) == 16, "wire layout");

struct Hello {
  uint32_t version;     // kVersion of the client
  uint32_t pid;         // the client process (for the log)
  uint64_t slotBytes;   // size of the shared memory passed with this message (kMinSlotBytes..kMaxSlotBytes)
  uint32_t reserved[2];
  char buildId[64];     // the plugin build, NUL-terminated
  char client[32];      // a short name for the log ("ml:<node id>"), NUL-terminated
};
static_assert(sizeof(Hello) == 120, "wire layout");

struct HelloReply {
  uint32_t version;  // the runner's kVersion            } never change
  uint32_t pid;      // the runner process               } these three
  char buildId[64];  // the runner build, NUL-terminated }
  uint32_t status;   // kOk or kBadSlot
  uint32_t idleSeconds;
  char message[128];
};
static_assert(sizeof(HelloReply) == 208, "wire layout");

struct RunRequest {
  char model[kModelIdBytes];  // a model_table.h id, NUL-terminated ("vitmatte-small")
  uint32_t ep;                // Ep
  uint32_t slot;              // 0
  uint32_t cIn, h, w;         // input [1, cIn, h, w] float32 at slot offset 0
  uint32_t cOut, outH, outW;  // output [1, cOut, outH, outW] float32 at slotOutputOffset()
  int32_t x, y;               // the run's place in the frame (information; the test runner uses it)
  uint32_t reserved[2];
};
static_assert(sizeof(RunRequest) == 80, "wire layout");

struct RunReply {
  uint32_t status;          // Status
  uint32_t sessionCreated;  // 1: this request created the session
  uint32_t runUs;           // the model call (session.Run + the copy out)
  uint32_t queueUs;         // waiting for the GPU (other connections' tiles)
  uint32_t loadUs;          // session creation, when sessionCreated
  uint32_t reserved;
  char ep[8];               // the EP that ran: "cuda" or "cpu" ("" on failure)
  char message[256];        // the error, or why the EP differs from the requested one
};
static_assert(sizeof(RunReply) == 288, "wire layout");

struct SessionInfo {
  char model[kModelIdBytes];  // the model id
  uint32_t ep;                // as requested (resolved: kEpCuda or kEpCpu)
  uint32_t idleMs;            // since its last use
  char activeEp[8];
  uint64_t runs;
};
static_assert(sizeof(SessionInfo) == 56, "wire layout");

constexpr int kMaxSessionsReported = 8;
struct StatusReply {
  uint32_t version, pid;
  char buildId[64];
  uint32_t connections;  // open now, this one included
  uint32_t sessions;     // live sessions (<= kMaxSessionsReported reported)
  uint64_t runs;         // RUN requests answered since start
  uint32_t uptimeS, idleSeconds;
  int32_t rssMiB;
  uint32_t reserved;
  SessionInfo session[kMaxSessionsReported];
};
static_assert(sizeof(StatusReply) == 104 + 8 * 56, "wire layout");

struct ErrorReply {
  char message[256];
};
static_assert(sizeof(ErrorReply) == 256, "wire layout");

// The slot layout of a RUN: [input float32 cIn*h*w][output float32 cOut*outH*outW], no gap.
inline uint64_t slotOutputOffset(const RunRequest& q) { return uint64_t(q.cIn) * q.h * q.w * sizeof(float); }
inline uint64_t slotRunBytes(const RunRequest& q) {
  return slotOutputOffset(q) + uint64_t(q.cOut) * q.outH * q.outW * sizeof(float);
}

}  // namespace proto
}  // namespace ml
}  // namespace mr
