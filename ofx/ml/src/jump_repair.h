// Jump Repair (ofx/ml, first in the chain): removes short hallucinated jumps from a matte: a group of
// pixels that appears or vanishes in frame t for 1-3 frames while the frames around it agree with
// each other. Classical, no model, no picture; pure C++14 (no OFX, no ONNX Runtime). The centre
// frame's matte stays the source (decision 17): only pixels in a repaired blob's zone change, every
// other pixel is bit-identical. Design notes and limits: docs/jump_repair.md.
//
// 1. Flag. Reference pairs (b, a) = (t - d1, t + d2), d1, d2 >= 1, span d1 + d2 - 1 <= maxSpan, in
//    the order (1,1), (1,2), (2,1), (1,3), (2,2), (3,1) (shortest span first; a pair whose frames
//    are missing is skipped). A pixel is flagged by the first pair where |A_b - A_a| <= agree,
//    A_t - max(A_b, A_a) > threshold ("added": it appeared) or min(A_b, A_a) - A_t > threshold
//    ("removed": it vanished), and the references are confirmed by the frames beyond them where
//    those exist (|A_{b-1} - A_b| <= agree, |A_{a+1} - A_a| <= agree): otherwise a jump in t - 1
//    next to an edge that moves would make frame t look like a jump of the opposite kind. A
//    span-3 jump (frames t-1..t+1 all wrong) is seen through (2,2).
// 2. Blobs: 8-connected components of the flagged pixels, added and removed separately; blobs smaller
//    than minSizePercent of the frame area are ignored (counted only).
// 3. Guards (motionGuard): s = +1 added / -1 removed, e(p) = s (A_t(p) - ref(p)) > 0 the blob's
//    excess, M = sum of e; the blob's reference pair = the one most of its pixels used.
//    - Moved: a real object that moves more than its own size per frame is flagged whole (slower
//      ones never are: t - 1 and t + 1 cover it), and it passes through: it is found again in t - 1
//      shifted by about -v, where frame t has left it, and in t + 1 by about +v, where frame t has
//      not reached yet. Score of neighbour n at shift u: sum_p min(e(p), [s (A_n(p + u) - A_t(p +
//      u))]+) / M; Moved = max over v (|v_x|, |v_y| <= min(h / 6, max(4 x the blob's larger side,
//      8)), grid step R / 32) of min(best score of t - 1 near -v, best of t + 1 near +v), "near" =
//      within 1 grid cell + |v| / 4 (acceleration). >= 0.5 -> kept. A blob displaced the same way
//      against both neighbours (they agree, t is off) is not a pass-through: repaired.
//    - Turning: a real edge at the extreme of a back-and-forth motion overshoots both references,
//      but grows towards t before them and shrinks after: in the window of the blob's bounding box
//      grown by its smaller side (+ the repair zone), with the frames b - 1 and a + 1 (when present),
//      min(sum s (A_b - A_{b-1}), sum s (A_a - A_{a+1})) > 0.5 M -> kept.
//    - Not still (spans of 2-3 frames only): a motion with a period of a few frames (an arm swinging)
//      looks exactly like a static 2-3 frame blob in the mattes, so a longer span is repaired only
//      when the matte around it is still: in the window above, at most 0.1 x the blob's area of
//      pixels change by more than agree from b - 1 to b or from a to a + 1 (both must exist);
//      otherwise kept.
// 4. Grow (hysteresis): a repaired blob's soft rim lies below threshold but still beyond the
//    references, and a rim left behind shows as a faint outline. From the flagged pixels, 8-connected,
//    at most growMaxPx steps (16 px at 1080 lines), the blob takes in every pixel where the references
//    agree (|A_b - A_a| <= agree: a moving edge stops it) and frame t lies beyond their range in the
//    blob's direction by more than growThreshold (0.02), never a kept blob's pixel. Which blobs are
//    repaired does not change.
// 5. Repair. ref = the pair's value at t, linear in time: A_b + d1 / (d1 + d2) (A_a - A_b) (the mean
//    for span 1). Zone: pixels within growPx + featherPx of the grown blob, weight 1 up to growPx, then
//    linear to 0. A zone pixel changes only when it lies beyond the pair's range in the blob's
//    direction (added: A_t > max(A_b, A_a), removed: A_t < min); then out = A_t + w (ref - A_t), so
//    an added blob is only lowered, a removed one only raised. Flagged pixels use their own pair,
//    the rest of the zone the blob's pair. Overlapping zones: the larger weight wins.
// A frame without both t - 1 and t + 1 is not judged: out = the centre bit for bit.
#pragma once

#include <cstdint>
#include <vector>

namespace mr {
namespace ml {

constexpr int kJumpMaxSpan = 3;
// The farthest neighbour the core can use: the longest span's reference plus the turning guard's.
constexpr int kJumpReach = kJumpMaxSpan + 1;
// The neighbours needed for a span: offsets -reach .. +reach.
inline int jumpReach(int maxSpan) { return (maxSpan < 1 ? 1 : maxSpan > kJumpMaxSpan ? kJumpMaxSpan : maxSpan) + 1; }

struct JumpParams {
  double threshold = 0.5;         // "Jump Above": how far outside both references a pixel must lie
  double agree = 0.15;            // the references must agree within this
  double minSizePercent = 0.0125; // blobs smaller than this % of the frame area are ignored
                                  // (0.0125 %: about 30 px at 562 x 424, 260 px at 1920 x 1080)
  int maxSpan = 1;                // 1-3 frames
  double growPx = -1.0;           // repair zone at full weight around a blob; < 0: max(1, h / 540)
  double featherPx = -1.0;        // then the fade to 0; < 0: max(1, h / 540)
  bool motionGuard = true;        // the Moved, Turning and Not still checks ("Motion Guards")
  double growMaxPx = -1.0;        // hysteresis growth of a repaired blob, at most this many steps;
                                  // < 0: max(2, 16 h / 1080); 0: no growth
  double growThreshold = 0.02;    // a grown pixel lies beyond the references' range by more than this
};

enum class JumpDecision : uint8_t { Repaired, KeptMoved, KeptTurning, KeptNotStill };
const char* jumpDecisionName(JumpDecision d);

// A blob of at least the minimum size: where, how big, which references, the guard sums, decision.
struct JumpBlob {
  bool added = true;           // appeared in frame t (lowered by the repair); false: vanished (raised)
  int before = -1, after = 1;  // the blob's reference pair as offsets (-d1, +d2)
  long long area = 0;          // flagged pixels
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // bounding box, inclusive, row 0 = top
  double mass = 0;             // M
  double moveScore = 0;                  // the Moved score: min of the two sides at the best velocity
  double moveBefore = 0, moveAfter = 0;  // each side's score (0-1) and shift there
  int moveBeforeDx = 0, moveBeforeDy = 0, moveAfterDx = 0, moveAfterDy = 0;
  bool turnChecked = false;
  double turnBefore = 0, turnAfter = 0;  // the Turning sums (turnChecked)
  long long movingPixels = 0;            // the Not still count (turnChecked)
  JumpDecision decision = JumpDecision::Repaired;
  long long changed = 0;       // pixels this blob's zone changed (Repaired only)
  long long grown = 0;         // pixels the hysteresis growth added to it (Repaired only)
};

struct JumpStats {
  bool judged = false;         // false: t - 1 or t + 1 missing: the frame passes through
  int pairs = 0;               // reference pairs available
  long long flaggedPixels = 0;
  int smallBlobs = 0;          // blobs under the minimum size (ignored)
  long long minArea = 0;       // the minimum size in px at this frame size
  std::vector<JumpBlob> blobs; // the others, in scan order
  int repaired = 0, keptMoved = 0, keptTurning = 0, keptNotStill = 0;
  long long pixelsChanged = 0;
  long long grownPixels = 0;
  double maxChange = 0;
  double flagMs = 0, blobMs = 0, repairMs = 0, totalMs = 0;
};

// QC marks (the node's View "Jumps"): pixels the repair changed, and the pixels of blobs it kept.
enum JumpMark : uint8_t { kJumpNone = 0, kJumpRepaired = 1, kJumpKept = 2 };

// frames[k + kJumpReach] = the matte at offset k (w*h, 0-1, row 0 = top) for k in [-kJumpReach,
// kJumpReach], nullptr where not available; frames[kJumpReach] = the centre (required). out: w*h,
// may be the centre (all decisions are made before anything is written). marks (optional): resized
// to w*h.
void repairJumps(const float* const* frames, int w, int h, const JumpParams& p, float* out, JumpStats* st = nullptr,
                 std::vector<uint8_t>* marks = nullptr);

}  // namespace ml
}  // namespace mr
