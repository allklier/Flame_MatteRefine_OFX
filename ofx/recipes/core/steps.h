// Edge Recipes step adapters (docs/edge_recipes.md 1.2, 2.5): thin wrappers over existing pure cores,
// no logic of their own beyond plumbing. Params are the cores' own structs, set by the plugin.
#pragma once

#include "jump_repair.h"
#include "merge_fusion.h"
#include "recipe_core.h"
#include "temporal.h"

namespace mr {
namespace recipes {

// Jump Repair (ofx/ml/src/jump_repair.h through render_chain.h runJumpRepair). Neighbours: the
// mattes at t +- jumpReach(maxSpan). First / last frame: the core passes through (summary says so).
// QC (View "Jumps"): Jump Repair's marks, and its output in QcLayer::plane (its input is the matte
// handed to it).
class JumpRepairStep : public Step {
 public:
  ml::JumpParams p;
  const char* id() const override { return "jumpRepair"; }
  const char* label() const override { return "Jump Repair"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override;
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;
};

// Region weight (evidence only): computes the Focus Region weight and its box once for the render
// (merge_fusion.h prepareRegion); the region-bound steps after it read the same map.
class RegionWeightStep : public Step {
 public:
  explicit RegionWeightStep(const RegionParams* rp) : rp_(rp) {}
  const char* id() const override { return "regionWeight"; }
  const char* label() const override { return "Region"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedRegion, 0}; }
  bool writesMatte() const override { return false; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;

 private:
  const RegionParams* rp_;
};

// A candidate source (2.1): proposes alpha + confidence on the region's crop. Implementations are
// injected: the tracer (WP3, Rebuilds), the model (WP4, needs kNeedModel), tests' fakes.
class CandidateSource {
 public:
  virtual ~CandidateSource() {}
  virtual const char* id() const = 0;     // "tracer", "model"
  virtual const char* label() const = 0;  // "Rebuild Detail", "Edge Model"
  virtual StepKind kind() const = 0;
  virtual StepNeeds needs() const = 0;    // besides the region (always needed)
  // alpha, conf: box.w() * box.h(), row-major crop at box (already sized); matte = the current
  // matte (full frame). May throw: the candidate is then not added.
  virtual void produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
                       StepReport* r) = 0;
};

// Wraps one CandidateSource as a step (evidence only: adds a candidate, never writes the matte).
class CandidateStep : public Step {
 public:
  CandidateStep(CandidateSource* src, const RegionParams* rp) : src_(src), rp_(rp) {}
  const char* id() const override { return src_->id(); }
  const char* label() const override { return src_->label(); }
  StepKind kind() const override { return src_->kind(); }
  StepNeeds needs() const override;
  bool writesMatte() const override { return false; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;

 private:
  CandidateSource* src_;
  const RegionParams* rp_;
};

// The strand presence vote (WP6, docs/edge_recipes.md section 3 step 4) is injected into HairLocks (the
// plugin's VoteStep, hair_locks_sources.h): after the candidates, before the fusion; it revises the
// "tracer" candidate in the evidence. Its condition: that candidate exists.
bool hasTracerCandidate(const StepContext& c, std::string* why);

// Fusion in the region (merge_fusion.h fuseCandidates on the region box, rule swappable), then the
// never-lower blend of docs/focus_region.md step 4 as matte_merge.cpp applies it (mixRegion):
// v = fused; w < 1: v = base + w (v - base); out = max(v, base); where w = 0: out = base bit for bit.
// base = this step's input, kept as the plane "regionBase" for Snap's never-lower.
class FusionStep : public Step {
 public:
  explicit FusionStep(const RegionParams* rp) : rp_(rp) {}
  merge::FusionParams fp;
  merge::FusionRule rule = merge::defaultFusionRule;
  const char* id() const override { return "fusion"; }
  const char* label() const override { return "Fusion"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedRegion, 0}; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;

 private:
  const RegionParams* rp_;
};
// The condition for FusionStep: at least one candidate ("no candidate").
bool hasCandidates(const StepContext& c, std::string* why);
// The condition for SnapRegionStep: the fusion ran this render ("nothing added").
bool fusionRan(const StepContext& c, std::string* why);
// Snap's weight (w*h): region x a mask of the pixels the fusion ADDED (after - before > 1e-3). grow1080 > 0: 1
// within r = max(1, round(grow1080 x h / 1080)) px (square) of an added pixel, 0.5 within 2 r, 0 beyond (h = the
// delivered height, decision 6); grow1080 <= 0: the added pixels only. Returns how many pixels were added (0:
// weight all 0).
// addedMin: the least rise that counts as added (default 1e-3, the original rule).
long long snapAddedWeight(const float* before, const float* after, const float* region, int w, int h,
                          double grow1080, float* weight, Workers* wk = nullptr, float addedMin = 1e-3f);
// Removes the pixels Snap RAISED (out > in) that form 8-connected islands touching no pixel of `in` at or above
// `touch` (the fused matte the snap started from): out = in there. Returns the pixels reset.
long long dropSnapIslands(const float* in, float* out, int w, int h, float touch);

// Snap to Image on the NEW pixels only (decision 39; WP8 scoring found Snap on the base matte's own edge in
// the region caused nearly all distractor hits): merge_fusion.h snapInRegion with the weight
// snapAddedWeight() below, Snap's sizes snapParamsForHeight(h) at the delivered height, never lower than
// "regionBase" (the matte before the fusion). Runs only after the fusion (condition fusionRan); where the
// fusion added nothing the weight is 0 and the matte passes bit for bit (the core is not called).
class SnapRegionStep : public Step {
 public:
  explicit SnapRegionStep(const RegionParams* rp) : rp_(rp) {}
  double strength = 1.0;
  bool neverLower = true;
  // The added mask's growth, px at 1080 delivered lines (snapAddedWeight). 0 = the added pixels only: tuned on
  // the WP8 distractor scenes (growth 3 / 1 / 0 px: glove distractor alpha .0151 / .0018 / .0006, Snap off
  // .0006); a growth reaches the glove edge 3 px from a strand.
  double addedGrow1080 = 0.0;
  float addedMin = 1e-3f;    // the least rise that counts as added (HairLocksSettings snapAddedMin)
  bool raiseOnly = false;    // out = max(snapped, in): Snap never lowers what the fusion made (snapRaiseOnly)
  float islandTouch = 0.0f;  // > 0: dropSnapIslands(in, out, islandTouch) after Snap (snapIslandTouch)
  const char* id() const override { return "snapRegion"; }
  const char* label() const override { return "Snap"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture | kNeedRegion, 0}; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;

 private:
  const RegionParams* rp_;
};

// Temporal behind a small local interface (tests inject their own). weight: w*h 0-1 where Temporal may
// act (nullptr = everywhere).
class TemporalRunner {
 public:
  virtual ~TemporalRunner() {}
  // flows: this render's shared flows (Evidence::flows) or nullptr.
  virtual void run(const float* matte, const float* rgb, int w, int h, const ml::StabiliseParams& p,
                   const ml::Neighbours& nb, const float* weight, const ml::TemporalFlows* flows, float* out,
                   ml::StabiliseStats* st) = 0;
};
// The implementation (WP5's weight map, temporal.h stabilise(..., weight)): the frames in time order as
// render_chain.h runTemporal builds them (neighbours within p.frames; farther ones, fetched for Jump
// Repair, ignored), weight passed into the core: where it is 0 the output is the input bit for bit and no
// neighbour is read; nullptr = the core without a weight. Throws std::invalid_argument for a neighbour
// without a w x h matte and picture. flows given (the strand presence vote computed them this render):
// stabilise()'s precomputed-flow overload, so the flow is not computed twice; note that the vote computed
// them with the matte BEFORE the fusion as the centre's subject weighting (the neighbours' raw mattes
// either way). Flows that do not fit (another setup, a neighbour missing): computed inside as without.
class WeightedTemporal : public TemporalRunner {
 public:
  bool usedSharedFlows = false;  // the last run
  void run(const float* matte, const float* rgb, int w, int h, const ml::StabiliseParams& p, const ml::Neighbours& nb,
           const float* weight, const ml::TemporalFlows* flows, float* out, ml::StabiliseStats* st) override;
};

// Temporal outside the Focus Region (3, 5.3: Temporal reads the neighbours' RAW mattes, so inside
// the region it would erase rebuilt strands): weight = 1 - region weight; no region connected:
// weight 1 everywhere. Needs neighbours within p.frames and the picture.
class TemporalOutsideStep : public Step {
 public:
  TemporalOutsideStep(const RegionParams* rp, TemporalRunner* runner) : rp_(rp), runner_(runner) {}
  ml::StabiliseParams p;
  const char* id() const override { return "temporal"; }
  const char* label() const override { return "Temporal"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture | kNeedNeighbours, p.frames}; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;

 private:
  const RegionParams* rp_;
  TemporalRunner* runner_;
};

}  // namespace recipes
}  // namespace mr
