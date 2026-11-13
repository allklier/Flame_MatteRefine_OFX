// Strand presence vote (Edge Recipes WP6, docs/edge_recipes.md sections 3 and 7; decision 39's "temporal
// steadiness by deciding strand presence over neighbour frames with the existing flow"). Pure C++14, no OFX
// types (decision 3).
//
// Why: the strand tracer (ofx/recipes/tracer) judges each frame alone, so a strand near its confidence bar
// flickers in and out. The vote decides per frame t whether each traced piece exists by looking for the same
// piece in the neighbour frames t-K..t+K (K = 2, Temporal's reach), aligned with the optical flow:
//   score(piece)  = matched neighbours / judged neighbours. A neighbour is judged when its flow is reliable
//                   (rule flowCheck); it matches when its pieces, moved into frame t, cover at least matchCover
//                   of the piece's points within the tolerance tolPx + tolWidth * width + tolFlow * (1 - flow
//                   confidence along the piece).
//   keep          if score >= scoreHigh                                  (rule high: kept by the neighbours,
//                                                                         even below its own bar)
//                 or score >= scoreLow and confidence >= its bar         (rule low)
//                 or confidence >= confHigh                              (rule confHigh: a very clear piece
//                                                                         survives being isolated)
//                 else dropped (an isolated piece, or one below its bar the neighbours do not confirm).
//   reinstate     a piece absent at t (no centre piece covers it) but present in t-1 AND t+1 (each at or
//                 above its bar, inside the widened region, the two agreeing in geometry once moved to t) is
//                 put back: its alpha is the mean of the two neighbours' alpha warped to t, confidence
//                 reinstateFactor x the lower of the two (rule reinstate).
// The bars: the tracer's own keep bars (keepConf for anchored and self-referenced pieces, minConfUnanchored
// for the others: setBars()). For the hysteresis to see pieces just below the bar the CALLER traces every
// frame with the bars lowered by a margin (candidateTracerParams()); the vote then applies the real bars.
//
// Inputs, all in ONE coordinate frame (a crop of the delivered image, the same box for every frame):
//   * the tracer's result for t and each neighbour, computed by the caller every render (decision 14: no
//     cross-render cache; this library only consumes results). The centre is traced with the region at t;
//     the neighbours with widenRegion() of it (user ruling Q7, 2026-11-12: the region is only known at t, it
//     is widened by the flow; it is never fetched at neighbour times).
//   * per neighbour a Motion: the displacement t -> neighbour at every crop pixel (from Temporal's flows via
//     motionFromFlows(), or any dense field), with an optional confidence.
//   * the region weight at t.
// A neighbour piece only counts where its moved position lies inside the widened region of t (rule
// regionGate). Output for t: candidate alpha + confidence (w*h) and a report per piece. Where the region
// weight at t is 0 (or NaN), alpha and confidence are exactly 0.
// Flow failure: a neighbour whose motion is marked invalid, or whose mean flow confidence over the SUPPORT is
// below minFlowConf, is not judged. Support = the pixels of the widened region where the base matte at t is
// >= 0.5 (the subject the strands hang from) when VoteInput::base is given and at least minSupportPx such
// pixels exist, else the region's pixels. Why: Temporal's subject-weighted flow (temporal.h, subjectWeighted)
// has a low forward-backward confidence on the background by construction (synthetic sequence: about 0.1-0.25
// over the region with a correct flow, 0.9 on the subject; a flash frame: 0.00 everywhere), so the region alone
// cannot tell a failed flow from a good one; with no judged neighbour the vote falls back to the frame's own result
// (the tracer's bars alone) and says so (VoteResult::fallback, note).
#pragma once

#include <string>
#include <vector>

#include "strand_tracer.h"

namespace mr {
namespace ml {
struct TemporalFlows;
}
namespace vote {

// Displacement field centre t -> neighbour t + offset at the crop's resolution (w*h each, row 0 = top):
// the centre pixel p corresponds to p + (dx, dy) in the neighbour. conf: 0-1 reliability (nullptr = 1).
struct Motion {
  int offset = 0;
  const float* dx = nullptr;
  const float* dy = nullptr;
  const float* conf = nullptr;
  bool valid = true;  // false: the caller knows the flow failed (the neighbour is not judged)
};

// Owning storage for a Motion.
struct DenseMotion {
  int offset = 0;
  bool valid = false;
  std::vector<float> dx, dy, conf;
  Motion view() const;
};

// Samples Temporal's flow centre -> the frame at `offset` (ml::computeFlows) over the crop [x0, x0 + w) x
// [y0, y0 + h) of the frame (flowAt + the forward-backward confidence on the flow grid). valid = false when
// `flows` holds no flow for that offset.
DenseMotion motionFromFlows(const ml::TemporalFlows& flows, int offset, int x0, int y0, int w, int h);

struct VoteParams {
  // the bars: always set them from the tracer params in use with setBars() (these defaults = the tracer's at
  // the time of writing: keepConf 0.45, minConfUnanchored 0.45)
  double minConf = 0.45, minConfUnanchored = 0.45;
  // hysteresis
  // confHigh 1.0: the tracer's confidence is the median step score (<= 1) + 0.15 when anchored, so an isolated
  // piece survives only when it is anchored with a median step score >= 0.85; a free piece practically never
  // (the synthetic spurious stroke, as crisp as real hair, scores 0.98)
  double scoreLow = 0.25, scoreHigh = 0.5, confHigh = 1.0;
  // matching
  double matchCover = 0.5;  // fraction of a piece's points that must lie near the other piece(s)
  double tolPx = 1.5, tolWidth = 0.5, tolFlow = 2.0;
  // flow reliability
  double minFlowConf = 0.3;  // mean over the support; below it the neighbour is not judged
  int minSupportPx = 50;
  // reinstatement
  double reinstateFactor = 0.7;
  double capsulePad = 2.0;  // px beyond half the width: the pixels that belong to a piece (alpha mask)
  // Temporal blend of the strand alpha (agent hl_flicker): the result at t is the weighted mean of the kept /
  // reinstated alpha at t and every judged neighbour's alpha moved to t by the flow (only its pieces at or
  // above their real bars, with their joins), weights 1 at t, blendNear for t+-1, blendFar for t+-2. A strand
  // traced in every frame keeps its alpha; a piece (or a join, or a piece's extra length) seen in one frame
  // only fades to a fraction instead of switching on and off. Confidence: the max over the frames that
  // contributed. 0 = off (the alpha at t only).
  // Away from the capsules of the pieces kept or reinstated at t (+ capsulePad) the neighbours alone decide:
  // smoothstep(fillLo, fillHi) of their present weight share (t excluded): t-1 and t+1 together (0.67) fill
  // (0.9), one near and one far (0.5) fill weakly (0.26), one neighbour does not (synthetic er_score: without this gate single-neighbour
  // ghosts at misaligned places raised false alpha in the region 0.011 -> 0.086; without any fill the real
  // clip's lost-lock frames came back, presence flips 3 -> 11).
  double blendNear = 0.0, blendFar = 0.0;
  // Presence mode of the blend (presenceHi > presenceLo): the alpha is the mean over the frames where the strand
  // is present at that pixel (alpha > presentAlpha) x smoothstep(presenceLo, presenceHi) of the present frames'
  // weight share: a strand in most frames keeps its full alpha, one in a single frame fades out.
  double presenceLo = 0.0, presenceHi = 0.0, presentAlpha = 0.05;
  double fillLo = 0.4, fillHi = 0.7;
  // rule switches (the ablation; all on = the vote)
  bool rHigh = true, rLow = true, rConfHigh = true, rReinstate = true, rWidthTol = true, rFlowTol = true,
       rRegionGate = true, rFlowCheck = true;
};

// minConf / minConfUnanchored from the tracer params the recipe uses (its real bars).
void setBars(VoteParams& v, const tracer::TracerParams& p);
// The tracer params to trace with when the vote follows: the keep bars lowered by `margin` (keepConf and
// minConfUnanchored; nothing else changes).
tracer::TracerParams candidateTracerParams(const tracer::TracerParams& p, double margin = 0.15);

// The region at t widened by the flow (ruling Q7): a grey dilation (max filter, disc) of the region by
// r = the largest flow displacement over the region's pixels among the given motions + marginPx. The caller
// traces the neighbours with it; the vote gates moved neighbour pieces with it. w*h.
std::vector<float> widenRegion(const float* region, int w, int h, const Motion* motions, int count,
                               double marginPx = 3.0);

// The region at t moved into the neighbour frame by the flow (each neighbour pixel q takes the region value at
// the centre point p with p + motion(p) = q), then grown by marginPx (max filter). The neighbour is traced
// with it so its tracer sees the same strands, the same matte overlap and the same region edge as the centre
// (agent hl_flicker: with widenRegion's larger window the neighbours' traces differed from their own frames'
// and the vote could not confirm or reinstate the locks). Still derived from the region at t (ruling Q7).
// An invalid motion returns widenRegion() of the region by marginPx alone.
std::vector<float> movedRegion(const float* region, int w, int h, const Motion& m, double marginPx = 3.0);

struct Neighbour {
  const tracer::TracerResult* result = nullptr;
  Motion motion;
};

struct VoteInput {
  int width = 0, height = 0;
  const float* region = nullptr;              // region weight at t (w*h)
  const float* widenedRegion = nullptr;       // widenRegion() of it; nullptr = computed inside
  const float* base = nullptr;                // base matte at t (w*h), optional: the flow check's support
  const tracer::TracerResult* centre = nullptr;
  const Neighbour* neighbours = nullptr;
  int count = 0;
};

enum class Fate { KeptHigh, KeptLow, KeptConfident, KeptFallback, Dropped, Reinstated };
const char* fateName(Fate f);

struct PieceVote {
  int index = -1;          // into centre->pieces; -1 for a reinstated piece
  double score = 0;        // matched / judged
  int matched = 0, judged = 0;
  double confidence = 0;   // own (reinstated: reinstateFactor x the lower neighbour confidence)
  double bar = 0;          // the bar this piece was held to
  Fate fate = Fate::Dropped;
  std::vector<tracer::Point> polyline;  // reinstated pieces only (moved to t)
  double width = 0;
};

struct VoteResult {
  std::vector<float> alpha, confidence;  // w*h
  std::vector<PieceVote> pieces;         // centre pieces in order, then reinstated ones
  int judgedNeighbours = 0, kept = 0, dropped = 0, reinstated = 0;
  bool fallback = false;
  std::string note;  // why a neighbour was not judged / the fallback
};

VoteResult votePresence(const VoteInput& in, const VoteParams& p = VoteParams());

}  // namespace vote
}  // namespace mr
