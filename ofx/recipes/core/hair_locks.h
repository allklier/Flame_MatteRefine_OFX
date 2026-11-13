// The Hair Locks recipe (docs/edge_recipes.md section 3, decision 39): Jump Repair -> region weight ->
// candidates -> fusion in the region -> Snap in the region -> Temporal outside the region. The
// candidate sources are injected (tracer WP3, model WP4; fakes in the tests), in the order given.
// The Detail candidate is one more injected source (the plugin passes model, tracer, Detail). The strand
// presence vote (WP6) is an injected step after the candidates (it revises the tracer's candidate).
#pragma once

#include <memory>
#include <vector>

#include "planner.h"
#include "steps.h"

namespace mr {
namespace recipes {

// The settings the plugin derives from the node's controls (recipe_ui, WP7), in the cores' units.
struct HairLocksSettings {
  bool jumpRepair = false;  // Advanced, off by default (CLAUDE.md decision 41)
  ml::JumpParams jr;
  RegionParams region;      // px at the delivered size
  std::vector<bool> candidates;  // Enable per injected source (missing = off)
  bool strandVote = false;  // Temporal Vote (with Rebuild Detail): the injected vote step runs
  merge::FusionParams fusion;
  bool snap = true;
  double snapStrength = 1.0;
  // Snap's limits (steps.h SnapRegionStep): the least rise that counts as added, Snap never lowering what the
  // fusion made, raised islands dropped (0 = keep). Defaults = the WP8 behaviour.
  float snapAddedMin = 1e-3f;
  bool snapRaiseOnly = false;
  float snapIslandTouch = 0.0f;
  bool temporal = true;
  ml::StabiliseParams tm;
};

class HairLocks {
 public:
  static const int kMajor = 1, kMinor = 0, kRev = 1;
  // sources, vote: owned by the caller, kept for the instance's life. temporal: nullptr = WeightedTemporal.
  // vote: nullptr = no vote entry in the recipe.
  explicit HairLocks(const std::vector<CandidateSource*>& sources, TemporalRunner* temporal = nullptr,
                     Step* vote = nullptr);
  // The recipe for these settings (steps keep pointing into this object: valid while it lives and
  // until the next call).
  Recipe recipe(const HairLocksSettings& s);

  JumpRepairStep jump;
  RegionWeightStep region;
  std::vector<std::unique_ptr<CandidateStep>> candidates;
  Step* vote = nullptr;
  FusionStep fusion;
  SnapRegionStep snap;
  TemporalOutsideStep temporal;

 private:
  RegionParams rp_;
  WeightedTemporal defaultTemporal_;
};

}  // namespace recipes
}  // namespace mr
