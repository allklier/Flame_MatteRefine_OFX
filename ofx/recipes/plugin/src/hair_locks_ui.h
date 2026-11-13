// Hair Locks (docs/edge_recipes.md 4.3) without OFX types, so it is unit-tested without a host: the node's
// controls -> the cores' params (decision 6: px controls x render scale, % controls x delivered height, as
// Matte Merge's regionPx), the Plan column's label strings (the probe of 2026-11-12: persistent messages are
// not shown as footers in Flame, a label string rewritten per render is; each <= kPlanMaxChars), and the
// colours of the views Candidates and Rebuilt.
#pragma once

#include <string>

#include "hair_locks.h"
#include "model_candidate.h"
#include "render_chain.h"
#include "strand_tracer.h"
#include "strand_vote.h"

namespace mr {
namespace recipes {

// The node's controls as read at one time (ids and defaults: recipe_def.cpp HairLocks::def()).
struct HairLocksUi {
  double featherPct = 0.25;      // rgFeather
  bool edgeModel = true;         // emEnable
  double strandGain = 1.0;       // emGain
  bool rebuild = false;          // rbEnable
  double rebuildMinConf = 0.45;  // rbMinConf: the tracer's keep bar for unanchored pieces
  double temporalAmount = 1.0;   // tmStrength
  double snapAmount = 1.0;       // rgSnapStrength
  bool strandBlend = true;       // rbBlend (TEMPORARY, hl_flicker): the presence blend over the neighbour frames
  int snapMode = 0;              // rgSnapMode (TEMPORARY, hl96): 0 Limited, 1 Off
  int model = 0;                 // mdModel: 0 Small, 1 Base
  bool jumpRepair = false;       // jrEnable (Advanced)
  ml::JumpUi jr;                 // jrThreshold, jrMinSize, jrSpanFrames (as an index), jrGuards
  double growPct = 0.0;          // rgGrow
  bool snap = true;              // rgSnap
  int lowContrast = 0;           // emLowContrast: 0 Auto, 1 Off
  bool keepLoose = true;         // rbKeepLoose
  bool vote = true;              // rbVote (the WP6 slot)
  double minPiecePx = 12.0;      // rbMinPiece, at 1080 lines
  bool temporal = true;          // tmEnable
  int temporalFrames = 0;        // tmFrames: 0 = +-1, 1 = +-2
  double temporalBandPx = 6.0;   // tmBand, full-resolution px
  bool detailConnected = false;  // the Detail socket (the third candidate)
};

// The candidate sources in the order the plugin injects them into HairLocks.
enum CandidateIndex { kCandModel = 0, kCandTracer = 1, kCandDetail = 2, kCandCount = 3 };
const char* modelId(int model);  // "vitmatte-small" / "vitmatte-base" (model_table ids, as plugin_ml.cpp)

// h: the delivered height; renderScale: the host's (decision 6).
HairLocksSettings hairLocksSettings(const HairLocksUi& ui, int h, double renderScale);
recipe::ModelCandidateParams modelParams(const HairLocksUi& ui);
tracer::TracerParams tracerParams(const HairLocksUi& ui);
// The vote's params for Hair Locks: the tracer's real bars (vote::setBars) and the temporal blend of the strand
// alpha (agent hl_flicker: t-1/t+1 weigh 1, t-2/t+2 weigh 0.5 against 1 at t; presence mode 0.15-0.5).
vote::VoteParams voteParams(const tracer::TracerParams& p, bool blend = true);  // blend: TEMPORARY test switch (rbBlend)

// --- Plan column ------------------------------------------------------------------------------------
// The model's gate state on the "Model" line: off / needs setup.sh / not installed / ready.
std::string gateText(bool edgeModel, int model, bool gateOpen, bool installed);

struct PlanInputs {
  const Recipe* recipe = nullptr;
  const PlanResult* res = nullptr;  // nullptr: the recipe did not run (a Source problem)
  HairLocksUi ui;
  std::string gate;                 // gateText(...)
  bool regionConnected = false;
  long long regionPixels = 0;       // Focus Region mask > 0
  std::string warning;              // a short Source warning (no alpha, ...); wins the Note line
  // The strand presence vote's counts (when its step ran).
  int voteKept = 0, voteDropped = 0, voteReinstated = 0;
  bool voteFallback = false;  // no neighbour judged: the frame's own result (the real bars)
};
struct PlanTexts {
  std::string plan, rebuild, model, inputs, note;  // the five label strings, each <= kPlanMaxChars
};
PlanTexts planTexts(const PlanInputs& in);
std::string fitPlan(const std::string& s);  // cut to kPlanMaxChars ("..." at the end when cut)

// --- views ------------------------------------------------------------------------------------------
// Candidates: the matte in half grey, each candidate's alpha in its colour on top (model red, tracer green,
// Detail blue; max with the grey).
void candidatesViewRgb(float base, float model, float tracer, float detail, float* rgb);
// Rebuilt: the result dimmed (0.3 grey), what Rebuild Detail added (the vote's reinstated pieces included) in
// green, the pieces the vote dropped in dim red (dropped: 0-1 coverage).
void rebuiltViewRgb(float result, float added, float dropped, float* rgb);

}  // namespace recipes
}  // namespace mr
