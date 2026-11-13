#include "hair_locks.h"

namespace mr {
namespace recipes {

HairLocks::HairLocks(const std::vector<CandidateSource*>& sources, TemporalRunner* temporalRunner, Step* voteStep)
    : region(&rp_),
      fusion(&rp_),
      snap(&rp_),
      temporal(&rp_, temporalRunner ? temporalRunner : &defaultTemporal_) {
  for (CandidateSource* s : sources) candidates.emplace_back(new CandidateStep(s, &rp_));
  vote = voteStep;
}

Recipe HairLocks::recipe(const HairLocksSettings& s) {
  rp_ = s.region;
  jump.p = s.jr;
  fusion.fp = s.fusion;
  snap.strength = s.snapStrength;
  snap.addedMin = s.snapAddedMin;
  snap.raiseOnly = s.snapRaiseOnly;
  snap.islandTouch = s.snapIslandTouch;
  temporal.p = s.tm;
  Recipe r;
  r.id = "hairLocks";
  r.name = "Hair Locks";
  r.major = kMajor;
  r.minor = kMinor;
  r.rev = kRev;
  bool anyCandidate = false;
  for (size_t i = 0; i < candidates.size(); ++i) anyCandidate |= i < s.candidates.size() && s.candidates[i];
  r.entries.push_back({&jump, s.jumpRepair, {}});
  // Evidence for the region steps (on when one of them is); skips without a region.
  r.entries.push_back({&region, anyCandidate, {}});  // Snap needs a candidate too (new pixels only)
  for (size_t i = 0; i < candidates.size(); ++i)
    r.entries.push_back({candidates[i].get(), i < s.candidates.size() && s.candidates[i], {}});
  if (vote) r.entries.push_back({vote, s.strandVote, {hasTracerCandidate}});  // docs/edge_recipes.md 3, step 4
  r.entries.push_back({&fusion, anyCandidate, {hasCandidates}});
  r.entries.push_back({&snap, s.snap && anyCandidate, {fusionRan}});
  r.entries.push_back({&temporal, s.temporal, {}});
  return r;
}

}  // namespace recipes
}  // namespace mr
