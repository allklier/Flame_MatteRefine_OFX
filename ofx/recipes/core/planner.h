// Edge Recipes planner (docs/edge_recipes.md section 3): a recipe = an ordered list of (step,
// enabled-from-controls, condition), defined in code, one per OFX effect. The order is fixed by
// the recipe (no user-settable order, decision 19's spirit). Each render the planner decides per
// entry, deterministically from the controls, the inputs and FrameStats:
//   off                      -> not called, not listed in the Plan line
//   a need is missing        -> skipped with the reason (fixed texts, skipReason below)
//   the entry's condition    -> skipped with the condition's reason
//   otherwise                -> run; a throw = failed, input passed through, chain goes on
// and reports the plan: one Plan line, a notice when a Rebuilds step ran, a DEBUG line per render.
#pragma once

#include <string>
#include <vector>

#include "recipe_core.h"

namespace mr {
namespace recipes {

// A deterministic condition beyond the declared needs (e.g. "at least one candidate"). Empty = always.
struct Condition {
  std::function<bool(const StepContext& c, std::string* why)> fn;
};
struct RecipeEntry {
  Step* step = nullptr;  // owned by the recipe's builder (the node instance)
  bool enabled = false;  // the node's Enable for this step
  Condition when;
};
struct Recipe {
  std::string id;            // "hairLocks"
  std::string name;          // "Hair Locks" (the effect name)
  int major = 1, minor = 0;  // the descriptor's OFX version (4.2)
  int rev = 1;               // behaviour revision (hidden recipeRev param, 4.2): which code path
  std::vector<RecipeEntry> entries;
};

// getFramesNeeded: the union of the ENABLED steps' needs (flags OR, reach max).
StepNeeds unionNeeds(const Recipe& r);

// The fixed skip text for a missing need (empty: all needs met). Checked in this order: picture,
// Focus Region (not connected / empty), Detail, model gate, neighbours.
std::string skipReason(unsigned needFlags, const FrameInputs& in, const FrameStats& stats);

struct PlanResult {
  std::vector<StepReport> steps;  // one per entry, recipe order
  FrameStats stats;
  bool rebuilt = false;           // a Rebuilds step ran and succeeded
  uint64_t finalVersion = 1;      // matte versions: 1 = the input, +1 per real change
  double ms = 0;
};

// Runs the recipe on one frame. out: w*h (the result; == in.base bit for bit when nothing changed
// it). ws: the node's workspace (emptied first). wk: the render's thread pool (nullptr: one is made
// for this call). qc: optional View layer.
void runRecipe(const Recipe& r, const FrameInputs& in, Workspace& ws, float* out, PlanResult* res,
               Workers* wk = nullptr, QcLayer* qc = nullptr);

// "Hair Locks v1.0: Jump Repair, Region, Tracer (3 pieces), Fusion, Snap | skipped: Temporal (no
// neighbours) | failed: Model (runner not available)" (steps that are off are not listed).
std::string planLine(const Recipe& r, const PlanResult& res);
// "Rebuild Detail created matte from the picture." when res.rebuilt, else empty. The rebuilding
// step's label goes in front: "<label> created matte from the picture."
std::string planNotice(const PlanResult& res);
// One line for the DEBUG log: per entry id=ran|skip|fail|off ms changed note.
std::string planDebugLine(const Recipe& r, const PlanResult& res);

// --- memory (2.2) ----------------------------------------------------------------------------------
// The planes one render holds, at the delivered size (render scale already applied: a half-res
// proxy is a quarter of everything). Counts only what the recipe holds; the cores' own scratch
// buffers (Snap's guided filter, Temporal's flow pyramid) are not included.
struct FootprintInput {
  int w = 0, h = 0;
  bool region = true, detail = false;  // input planes besides picture (3) and base (1)
  int workPlanes = 3;                  // Workspace cur + next, and the plugin's output
  int evidencePlanes = 3;              // full-frame evidence: region weight, sd0.5, regionBase
  int pictureNeighbours = 0;           // neighbours with matte + picture (4 planes each)
  int matteOnlyNeighbours = 0;         // neighbours fetched matte-only (1 plane each)
  int flowNeighbours = 0;              // flows (3 planes on the flow grid each)
  int flowLines = 600;                 // temporal.h StabiliseParams::flowLines
  long long cropPixels = 0;            // the region box
  int cropPlanes = 0;                  // candidates x 2 + crop-sized maps
};
struct Footprint {
  size_t inputs = 0, work = 0, evidence = 0, neighbours = 0, flow = 0, crop = 0;
  size_t total() const { return inputs + work + evidence + neighbours + flow + crop; }
};
Footprint estimateFootprint(const FootprintInput& f);

}  // namespace recipes
}  // namespace mr
