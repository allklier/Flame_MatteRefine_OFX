#include "hair_locks_ui.h"

#include <algorithm>

#include "recipe_def.h"

namespace mr {
namespace recipes {

namespace {

double clamp01(double v) { return v > 0.0 ? (v < 1.0 ? v : 1.0) : 0.0; }

// The Plan line's short step names (the full labels are in the log's plan line); "" = not listed.
const char* shortName(const std::string& id) {
  if (id == "jumpRepair") return "Jump";
  if (id == "model") return "Model";
  if (id == "tracer") return "Rebuild";
  if (id == "detail") return "Detail";
  if (id == "snapRegion") return "Snap";
  if (id == "strandVote") return "Vote";
  if (id == "temporal") return "Temporal";
  return "";  // regionWeight, fusion (implied by a candidate)
}

const StepReport* find(const PlanResult* res, const char* id) {
  if (!res) return nullptr;
  for (const StepReport& r : res->steps)
    if (r.id == id) return &r;
  return nullptr;
}

}  // namespace

const char* modelId(int model) { return model == 1 ? "vitmatte-base" : "vitmatte-small"; }

HairLocksSettings hairLocksSettings(const HairLocksUi& ui, int h, double renderScale) {
  HairLocksSettings s;
  s.jumpRepair = ui.jumpRepair;
  s.jr = ml::jumpParams(ui.jr);  // shares of the frame area: no scaling (render_chain.h)
  s.region.growPx = std::max(ui.growPct, 0.0) / 100.0 * h;  // Merge's regionPx (merge_ui.cpp:22)
  s.region.featherPx = std::max(ui.featherPct, 0.0) / 100.0 * h;
  s.candidates.assign(kCandCount, false);
  s.candidates[kCandModel] = ui.edgeModel;
  s.candidates[kCandTracer] = ui.rebuild;
  s.candidates[kCandDetail] = ui.detailConnected;
  s.strandVote = ui.rebuild && ui.vote;
  s.snap = ui.snap && ui.snapMode == 0;  // TEMPORARY (hl96): Snap Mode Off switches Snap off
  s.snapAddedMin = 0.05f;                // Limited Snap: only a real rise counts as added,
  s.snapRaiseOnly = true;                // Snap never lowers what the fusion made,
  s.snapIslandTouch = 0.3f;              // raised islands that touch nothing are dropped
  s.snapStrength = clamp01(ui.snapAmount);
  s.fusion.confFull = 0.5f;  // a confident candidate keeps its alpha (agent hl96: strands came out grey)
  s.temporal = ui.temporal;
  ml::TemporalUi tu;
  tu.frames = ui.temporalFrames;
  tu.strength = ui.temporalAmount;
  tu.bandPx = ui.temporalBandPx;
  s.tm = ml::temporalParams(tu, renderScale);  // Band px x render scale (decision 6)
  return s;
}

recipe::ModelCandidateParams modelParams(const HairLocksUi& ui) {
  recipe::ModelCandidateParams p;  // sizes: s = delivered height / 1080 (sizeScale 0)
  p.modelId = modelId(ui.model);
  p.strandGain = std::max(ui.strandGain, 0.0);
  p.lowContrast = ui.lowContrast == 1 ? recipe::LowContrast::Off : recipe::LowContrast::Auto;
  return p;
}

tracer::TracerParams tracerParams(const HairLocksUi& ui) {
  tracer::TracerParams p;
  p.minConfUnanchored = clamp01(ui.rebuildMinConf);
  p.minPieceLen1080 = std::max(ui.minPiecePx, 0.0);
  p.chainLength = true;  // agent hl96: a lock's bridge to the matte is not a speck
  p.coneJoin = true;     // agent hl96: join to the matte along the strand, not only to the nearest pixel
  p.softProfile = true;  // agent hl_flicker: soft cross-section and free-end fade, not a hard capsule
  p.taperW = 0.75;       // the fade at a free end: 0.75 x the strand width (1.5 cost lock coverage on real frames)
  return p;
}

vote::VoteParams voteParams(const tracer::TracerParams& p, bool blend) {
  vote::VoteParams v;
  vote::setBars(v, p);
  if (!blend) return v;  // TEMPORARY: Strand Blend off = the vote as before the blend
  v.blendNear = 1.0;  // agent hl_flicker: the strand alpha is a flow-aligned blend over t-2..t+2
  v.blendFar = 0.5;
  v.presenceLo = 0.15;  // presence mode: full alpha where the strand is in at least half the frames' weight,
  v.presenceHi = 0.5;   // fading to 0 at 15 % (a strand of one frame alone)
  return v;
}

std::string gateText(bool edgeModel, int model, bool gateOpen, bool installed) {
  if (!edgeModel) return "Model: off";
  if (!gateOpen) return "Model: needs setup.sh";
  const std::string name = std::string("Model ") + (model == 1 ? "Base" : "Small");
  return name + (installed ? ": ready" : ": not installed");
}

std::string fitPlan(const std::string& s) {
  if (int(s.size()) <= kPlanMaxChars) return s;
  return s.substr(0, size_t(kPlanMaxChars - 3)) + "...";
}

PlanTexts planTexts(const PlanInputs& in) {
  PlanTexts t;
  const PlanResult* res = in.res;
  // Plan: the steps that ran, short names.
  std::string ran;
  if (res)
    for (const StepReport& r : res->steps) {
      const char* n = shortName(r.id);
      if (r.ran && *n) ran += (ran.empty() ? "" : "+") + std::string(n);
    }
  t.plan = fitPlan(ran.empty() ? "Nothing ran: pass-through" : ran);

  // Rebuild: the generator is marked wherever it shows (decision 39).
  const StepReport* tr = find(res, "tracer");
  const StepReport* vt = find(res, "strandVote");
  if (!in.ui.rebuild) t.rebuild = "Rebuild Detail off";
  else if (tr && tr->ran && vt && vt->ran && !in.voteFallback)
    t.rebuild = fitPlan("Vote: kept " + std::to_string(in.voteKept) + " drop " + std::to_string(in.voteDropped) +
                        " back " + std::to_string(in.voteReinstated));
  else if (tr && tr->ran) t.rebuild = fitPlan("Created matte: " + (tr->summary.empty() ? "0 pieces" : tr->summary));
  else if (tr && tr->failed) t.rebuild = "Rebuild failed: see log";
  else t.rebuild = fitPlan(tr && !tr->note.empty() ? "Rebuild: " + tr->note : std::string("Rebuild on: creates matte"));

  // Model: the gate state, or what the step did.
  const StepReport* md = find(res, "model");
  if (md && md->failed) t.model = "Model failed: see log";
  else if (md && md->ran) t.model = std::string("Model ") + (in.ui.model == 1 ? "Base" : "Small") + ": ran";
  else t.model = fitPlan(in.gate);

  // Inputs.
  std::string region = "Region off";
  if (in.regionConnected) {
    const long long n = in.regionPixels;
    region = n == 0 ? "Region empty" : n < 10000 ? "Region " + std::to_string(n) + " px"
                                                 : "Region " + std::to_string(n / 1000) + "k px";
  }
  t.inputs = fitPlan(region + ", Detail " + (in.ui.detailConnected ? "on" : "off"));

  // Note: the most important thing to say.
  std::string note;
  if (!in.warning.empty()) note = in.warning;
  if (note.empty() && res)
    for (const StepReport& r : res->steps)
      if (r.failed) {
        note = r.label + " failed: see log";
        break;
      }
  if (note.empty() && vt && vt->ran && in.voteFallback) note = "Vote: own result only";
  if (note.empty() && in.regionConnected && in.regionPixels == 0) note = "Focus Region is empty";
  const bool regionSteps = in.ui.edgeModel || in.ui.rebuild || in.ui.detailConnected || in.ui.snap;
  if (note.empty() && !in.regionConnected && regionSteps) note = "Connect a Focus Region";
  const StepReport* jr = find(res, "jumpRepair");
  if (note.empty() && jr && jr->ran && jr->summary.find("not judged") != std::string::npos)
    note = "Jump Repair: end frame";
  if (note.empty() && res)
    for (const StepReport& r : res->steps)
      if (!r.ran && !r.failed && r.note == "no neighbours") {
        note = "No neighbour frames";
        break;
      }
  t.note = fitPlan(note.empty() ? "OK" : note);
  return t;
}

void candidatesViewRgb(float base, float model, float tracer, float detail, float* rgb) {
  const float g = 0.5f * base;
  rgb[0] = std::max(g, model);
  rgb[1] = std::max(g, tracer);
  rgb[2] = std::max(g, detail);
}

void rebuiltViewRgb(float result, float added, float dropped, float* rgb) {
  const float g = 0.3f * result;
  rgb[0] = std::max(g, 0.5f * dropped);
  rgb[1] = std::max(g, added);
  rgb[2] = g;
}

}  // namespace recipes
}  // namespace mr
