// Edge Recipes core (decision 41, docs/edge_recipes.md sections 2-3): the step interface, the shared
// per-render evidence and the planner. Pure C++14, no OFX types, no param reads, no I/O, no static
// state (2.5). The plugin (WP1/WP7) turns controls into the cores' params and fills FrameInputs.
//
// A step is a thin adapter over one existing pure core (decision 3). Contract (2.4): out == in bit for
// bit outside the step's zone; a disabled or skipped step is not called and its input is passed on;
// a step that throws passes its input through and is reported failed, the chain goes on.
// Evidence (2.2): maps keyed (kind, matte version, params), computed once per render; version 0 =
// depends on the inputs only (stable for the render); a matte-derived map is dropped as soon as a
// step really changed the matte (the planner compares the buffers). Lifetime: ONE render of ONE node:
// Workspace::beginRender() empties it, so no content survives into the next render and nothing is
// shared between nodes (decision 14). Buffers are kept for reuse (no page faults per render).
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mr {
class Workers;
namespace ml {
struct Neighbours;
struct TemporalFlows;
}
namespace recipes {

// What a step needs, declared from the SETTINGS only (getFramesNeeded has no images, 2.3).
enum Need : unsigned {
  kNeedPicture = 1u << 0,
  kNeedRegion = 1u << 1,      // the Focus Region socket, connected and not empty
  kNeedDetail = 1u << 2,      // the Detail socket
  kNeedModel = 1u << 3,       // the runner (ML): gated per step (1.3, 4.6)
  kNeedNeighbours = 1u << 4,  // frames t +- neighbourReach
};
struct StepNeeds {
  unsigned flags = 0;
  int neighbourReach = 0;  // frames on each side (kNeedNeighbours)
};

enum class StepKind { Refines, Rebuilds };  // Rebuilds = creates matte from the picture (UI marking)

// Read-only inputs of one render; every plane w*h (rgb w*h*3), row 0 = top, as delivered (render
// scale already applied: pixel params are scaled by the plugin, decision 6).
struct FrameInputs {
  int w = 0, h = 0;
  double renderScale = 1.0;
  double time = 0;
  const float* rgb = nullptr;              // the picture as delivered (Picture Colour, decision 34)
  const float* base = nullptr;             // the matte = Source alpha (decisions 4, 35); required
  const float* region = nullptr;           // Focus Region mask; nullptr = not connected
  const float* detail = nullptr;           // Detail mask; nullptr = not connected
  const ml::Neighbours* nb = nullptr;      // render_chain.h: neighbours at t +- k; nullptr = none fetched
  bool modelGateOpen = false;              // gate.h checkGate(...).open() (dev builds: true)
};

// Per-frame statistics for the planner's conditions (one pass over the inputs).
struct FrameStats {
  long long regionPixels = 0;  // Focus Region mask > 0
  double baseMean = 0;
  double greyFraction = 0;     // base in (0.02, 0.98)
};
FrameStats computeStats(const FrameInputs& in);

// A crop of the frame: [x0, x1) x [y0, y1).
struct Box {
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  bool empty() const { return x1 <= x0 || y1 <= y0; }
  int w() const { return empty() ? 0 : x1 - x0; }
  int h() const { return empty() ? 0 : y1 - y0; }
  long long pixels() const { return (long long)w() * h(); }
};

// A candidate (2.1): alpha + confidence on the region's crop, from a candidate step; the fusion
// step reads them all at once.
struct CandidatePlanes {
  std::string id;      // "model", "tracer", "detail"
  StepKind kind = StepKind::Refines;
  Box box;
  std::vector<float> alpha, conf;  // box.w() * box.h()
};

struct EvidenceKey {
  std::string kind;      // "regionWeight", "sd0.5", ...
  uint64_t matteVersion; // 0 = inputs only
  uint64_t params;       // paramHash of the values that shape the map
  bool operator<(const EvidenceKey& o) const;
};
uint64_t paramHash(std::initializer_list<double> v);  // FNV-1a over the doubles' bits

class Evidence {
 public:
  using Compute = std::function<void(std::vector<float>* out)>;
  // The map for k, computed by fn on first use (fn gets an empty vector). A throwing fn leaves
  // nothing behind (the next get computes again).
  const std::vector<float>& get(const EvidenceKey& k, const Compute& fn);
  // Drops matte-derived maps older than v (inputs-only maps stay).
  void dropMatteVersionsBelow(uint64_t v);

  // Named full-frame planes of this render (e.g. "regionBase", the matte the fusion started from).
  std::vector<float>& setPlane(const std::string& name, const float* data, size_t n);
  const std::vector<float>* plane(const std::string& name) const;

  // Candidates of this render, in the order they were added.
  CandidatePlanes newCandidate();  // empty planes from the reuse pool; listed only by addCandidate
  void addCandidate(CandidatePlanes&& c);
  const std::vector<CandidatePlanes>& candidates() const { return cand_; }
  CandidatePlanes* findCandidate(const std::string& id);  // nullptr = none with that id (a step may revise one)

  // Temporal's flows of this render (temporal.h computeFlows), computed once by the first step that needs
  // them (the strand presence vote) and reused by Temporal's precomputed-flow overload; nullptr = none yet.
  // Emptied by beginRender like everything else (decision 14).
  std::shared_ptr<ml::TemporalFlows> flows;

  // Empties everything (content), keeps the buffers for reuse; counters restart.
  void beginRender();
  int computed() const { return computed_; }
  int hits() const { return hits_; }
  size_t bytesHeld() const;    // maps, planes and candidates of this render
  size_t bytesPooled() const;  // kept for reuse

 private:
  std::vector<float> take();
  void giveBack(std::vector<float>&& v);
  std::map<EvidenceKey, std::vector<float>> maps_;
  std::map<std::string, std::vector<float>> planes_;
  std::vector<CandidatePlanes> cand_;
  std::vector<std::vector<float>> pool_;
  int computed_ = 0, hits_ = 0;
};

// Optional QC output (the node's View): the plugin names the step whose layer it wants; that step
// fills plane and / or marks, the others never compute one (as MergeViews: nullptr = not computed).
struct QcLayer {
  std::string step;             // the wanted step's id
  std::vector<float> plane;     // w*h when filled
  std::vector<uint8_t> marks;   // w*h when filled (e.g. Jump Repair's JumpMark)
};

struct StepReport {
  std::string id, label;
  StepKind kind = StepKind::Refines;
  bool ran = false, failed = false;
  std::string note;      // skip reason or error
  std::string summary;   // short result for the Plan line ("3 pieces"); empty = none
  long long changed = 0; // pixels the step changed (the step's own count, for the report only)
  double ms = 0;
};

struct StepContext {
  const FrameInputs& in;
  const FrameStats& stats;
  Evidence& ev;
  uint64_t matteVersion;  // version of the matte handed to run()
  Workers* wk;            // one pool per render, shared by all steps
  QcLayer* qc;            // nullptr = no QC wanted
  bool wantsQc(const char* id) const { return qc && qc->step == id; }
};

// A step: a thin adapter over one pure core.
class Step {
 public:
  virtual ~Step() {}
  virtual const char* id() const = 0;     // stable, for logs and QC ("jumpRepair")
  virtual const char* label() const = 0;  // for the Plan line ("Jump Repair")
  virtual StepKind kind() const = 0;
  virtual StepNeeds needs() const = 0;    // from settings only
  // false: the step only adds evidence (region weight, a candidate); run() gets out = nullptr and
  // the matte passes on without a copy.
  virtual bool writesMatte() const { return true; }
  virtual void run(const StepContext& c, const float* in, float* out, StepReport* r) = 0;
};

// Per-instance storage (one per node): the matte buffers and the evidence, reused across renders,
// emptied at the start of each (decision 14).
struct Workspace {
  Evidence ev;
  std::vector<float> cur, next;
  size_t bytes() const { return (cur.capacity() + next.capacity()) * sizeof(float) + ev.bytesHeld(); }
};

// --- shared evidence ---------------------------------------------------------------------------
// The Focus Region weight (merge_fusion.h prepareRegion): grow / feather in px at the delivered
// size (the plugin converts the % controls, Merge's rule: merge_ui.h regionPx).
struct RegionParams {
  double growPx = 0, featherPx = 0;
};
const std::vector<float>& regionWeight(const StepContext& c, const RegionParams& p);
// The bounding box of weight > 0 (empty when none).
Box regionBox(const StepContext& c, const RegionParams& p);
// Signed distance to the 0.5 contour of the CURRENT matte (ops/edge_field.h), keyed by its version.
const std::vector<float>& signedDistance(const StepContext& c, const float* matte, double band);

}  // namespace recipes
}  // namespace mr
