// The three candidate sources of Hair Locks (steps.h CandidateSource; docs/edge_recipes.md 3, rows 3a-3c),
// thin adapters without OFX types. Each writes alpha + confidence on the region's box; the fusion step
// combines them and never lowers the matte.
//   ModelSource   ViTMatte on one crop centred on the region (model_candidate.h, through a CropRunner: the
//                 plugin's RunnerClientCrops); gated (kNeedModel). QC "model": the model's trimap.
//   TracerSource  the strand tracer (strand_tracer.h, public API only): Rebuilds = creates matte from the
//                 picture. The tracer reads a LINEAR picture (docs/strand_tracing.md 3); with Picture Colour
//                 Auto the picture arrives display-encoded (Rec.1886, pure gamma 2.4, plugin_ml.cpp:193-197),
//                 so `linearise` decodes it (max(v, 0)^2.4) for the tracer only. Loose Pieces off: only the
//                 anchored pieces' (and their joins') capsules keep alpha.
//   VoteStep      the strand presence vote (ofx/recipes/vote/strand_vote.h, WP6): every frame t-K..t+K is
//                 traced on the same crop with the keep bars lowered by 0.15 (candidateTracerParams); the
//                 neighbours from their OWN pictures (the fetched neighbour images) with the region at t
//                 widened by the flow (widenRegion; user ruling Q7: never the region at neighbour times);
//                 motion from Temporal's flow method (temporal.h computeFlows) on a CROP around the region
//                 (flowCrop below), or on the whole frame shared with Temporal through the evidence when that
//                 is cheaper (cost rule below); flow reliability judged on the subject
//                 (VoteInput::base); the vote applies the real bars and replaces the tracer's candidate.
//                 No cross-render cache (decision 14): the neighbours are traced again every render.
//   DetailSource  the external Detail matte (mask clip) as it arrives (no alignment); confidence as Matte
//                 Merge gives candidate (a) (matte_merge.cpp:262-280): picture evidence (merge_evidence.h) on
//                 its thin structures (detail - opening), 1 on thick parts; computed on the box + a margin.
#pragma once

#include <functional>
#include <vector>

#include "merge_evidence.h"
#include "model_candidate.h"
#include "steps.h"
#include "strand_tracer.h"
#include "strand_vote.h"
#include "temporal.h"

namespace mr {
namespace recipes {

class ModelSource : public CandidateSource {
 public:
  recipe::ModelCandidateParams p;
  RegionParams region;                           // the recipe's (same evidence key)
  std::function<recipe::CropRunner*()> runner;   // made on first use; nullptr: throws "runner not available"
  recipe::ModelCandidateStats stats;             // the last produce()
  const char* id() const override { return "model"; }
  const char* label() const override { return "Edge Model"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture | kNeedModel, 0}; }
  void produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
               StepReport* r) override;

 private:
  std::vector<float> alpha_, conf_;
};

class TracerSource : public CandidateSource {
 public:
  tracer::TracerParams p;  // the real bars
  RegionParams region;
  bool keepLoose = true;   // Loose Pieces
  bool linearise = false;  // Picture Colour Auto: decode the display-encoded picture
  // The vote follows (Temporal Vote on): trace with the bars lowered (candidateTracerParams) and give, as this
  // candidate, the vote's own-result fallback (the real bars, vote with no neighbour): what stays if the vote
  // does not run. vp: the vote's params with the real bars (setBars).
  bool forVote = false;
  vote::VoteParams vp;
  int pieces = 0, anchored = 0;  // the last produce(): pieces kept (after Loose Pieces), anchored ones
  // The last produce(), for the vote: the crop (box + the tracer's margin), its picture (as the tracer read
  // it), base matte and region weight, and the tracer's result there.
  Box crop;
  std::vector<float> rgbCrop, baseCrop, weightCrop;
  tracer::TracerResult centre;
  // Fills a crop's picture (decoded when linearise) and matte from full-frame planes (w wide).
  void fillCrop(const float* rgb, const float* matte, int w, std::vector<float>& rgbOut, std::vector<float>& baseOut) const;
  tracer::TracerResult trace(const std::vector<float>& rgb, const std::vector<float>& base, const float* region,
                             int frameW, int frameH, double renderScale, const tracer::TracerParams& tp) const;
  // Loose Pieces off: zero alpha and confidence (crop-sized) outside the anchored pieces of `centre` and their joins.
  void keepAnchoredOnly(std::vector<float>& alpha, std::vector<float>& conf);
  const char* id() const override { return "tracer"; }
  const char* label() const override { return "Rebuild Detail"; }
  StepKind kind() const override { return StepKind::Rebuilds; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture, 0}; }
  void produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
               StepReport* r) override;

 private:
  std::vector<uint8_t> keep_;
};

class VoteStep : public Step {
 public:
  explicit VoteStep(TracerSource* t) : t_(t) {}
  int K = 2;                       // neighbours on each side (Temporal's reach)
  ml::StabiliseParams flowParams;  // Temporal's flow settings (frames = K is set inside)
  int temporalFlows = 0;           // whole-frame flows Temporal computes this render (0: Temporal off), set per render
  // Flow crop (WP8 timing: whole-frame flows dominated the render): the tracer's crop (the region box + the
  // tracer's margin, which already bounds the widening by the flow) grown by the flow's support, supportGrid
  // grid px of the whole frame's flow grid (scale s = flowScale(h, flowLines)) = s x supportGrid px, the
  // origin snapped to a multiple of s, flowed at the same scale s. Cost rule (flow cost ~ grid area): crop
  // flows when crop x n + frame x temporalFlows < frame x n (n = the vote's neighbours), else whole-frame
  // flows shared with Temporal.
  int supportGrid = 32;            // = FlowParams radius 8 x 4 (the window at the third-finest level)
  bool usedCropFlows = false;      // the last run
  bool movedRegions = true;        // trace each neighbour with the region moved by its flow (movedRegion), not widened
  // The last run (Plan, log, View Rebuilt).
  vote::VoteResult last;
  bool ran = false;
  int traced = 0;  // neighbour frames traced
  double flowMs = 0, traceMs = 0, voteMs = 0;
  const char* id() const override { return "strandVote"; }
  const char* label() const override { return "Vote"; }
  StepKind kind() const override { return StepKind::Rebuilds; }
  // No kNeedNeighbours flag: without neighbours it still runs and falls back to the frame's own result
  // (the real bars); the reach K joins getFramesNeeded's union.
  StepNeeds needs() const override { return StepNeeds{kNeedPicture, K}; }
  bool writesMatte() const override { return false; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override;
  // Full frame w*h: 1 on the capsules of the pieces the last vote dropped (0 elsewhere; empty when it did not run).
  void droppedPlane(int w, int h, std::vector<float>& plane) const;

 private:
  TracerSource* t_;
  std::vector<float> nrgb_, nbase_;
  ml::TemporalFlows cropFlows_;
  std::vector<std::vector<float>> frgb_, fmatte_;  // the flow crop of each frame
  std::vector<tracer::TracerResult> nres_;
};

class DetailSource : public CandidateSource {
 public:
  RegionParams region;
  double detailWidthPx = 5.4;  // thin structures are narrower than this (Matte Merge: 0.5 % of the height)
  merge::EvidenceParams ep;
  const char* id() const override { return "detail"; }
  const char* label() const override { return "Detail"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture | kNeedDetail, 0}; }
  void produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
               StepReport* r) override;

 private:
  std::vector<float> d_, op_, fine_, ev_, rgb_, scratch_;
  std::vector<uint8_t> cand_;
};

}  // namespace recipes
}  // namespace mr
