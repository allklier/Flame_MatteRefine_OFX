// Hair Locks without a host (hair_locks_ui.h): the controls -> the cores' params (render scale, % of the height,
// the candidate order, the vote slot), the Plan texts for each state (every one <= kPlanMaxChars, the generator
// marked, the gate states incl. a closed gate, failures, notes) and the view colours.
#include <cmath>
#include <cstdio>
#include <string>

#include "hair_locks_ui.h"
#include "recipe_def.h"

using namespace mr;
using namespace mr::recipes;

namespace {
int failures = 0;
#define CHECK(c)                                               \
  do {                                                         \
    if (!(c)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                              \
    }                                                          \
  } while (0)

StepReport rep(const char* id, const char* label, bool ran, const std::string& note = "", const std::string& sum = "") {
  StepReport r;
  r.id = id;
  r.label = label;
  r.ran = ran;
  r.note = note;
  r.summary = sum;
  return r;
}

bool fits(const PlanTexts& t) {
  for (const std::string* s : {&t.plan, &t.rebuild, &t.model, &t.inputs, &t.note})
    if (int(s->size()) > kPlanMaxChars) {
      std::printf("too long: '%s'\n", s->c_str());
      return false;
    }
  return true;
}
}  // namespace

int main() {
  {  // controls -> params
    HairLocksUi ui;  // the node's defaults
    HairLocksSettings s = hairLocksSettings(ui, 1080, 1.0);
    CHECK(!s.jumpRepair && s.snap && s.temporal && !s.strandVote);
    CHECK(s.candidates.size() == 3 && s.candidates[kCandModel] && !s.candidates[kCandTracer] && !s.candidates[kCandDetail]);
    CHECK(std::fabs(s.region.featherPx - 2.7) < 1e-9 && s.region.growPx == 0.0);  // 0.25 % of 1080
    CHECK(s.tm.frames == 1 && s.tm.bandPx == 6.0 && s.tm.strength == 1.0);
    ui.rebuild = true;
    ui.detailConnected = true;
    ui.temporalFrames = 1;
    ui.temporalBandPx = 8;
    ui.growPct = 1.0;
    ui.jumpRepair = true;
    ui.jr.span = ml::jumpSpanIndex(3.0);
    s = hairLocksSettings(ui, 540, 0.5);  // a half-res proxy: 540 delivered lines
    CHECK(s.candidates[kCandTracer] && s.candidates[kCandDetail] && s.strandVote);
    CHECK(s.tm.frames == 2 && s.tm.bandPx == 4.0);              // Band px x render scale (decision 6)
    CHECK(std::fabs(s.region.growPx - 5.4) < 1e-9);            // % of the delivered height
    CHECK(s.jumpRepair && s.jr.maxSpan == 3);
    ui.vote = false;
    CHECK(!hairLocksSettings(ui, 540, 0.5).strandVote);
    ui.vote = true;
    ui.rebuild = false;
    CHECK(!hairLocksSettings(ui, 540, 0.5).strandVote);  // the vote only with Rebuild Detail
    ui.model = 1;
    ui.strandGain = 1.5;
    ui.lowContrast = 1;
    const recipe::ModelCandidateParams mp = modelParams(ui);
    CHECK(mp.modelId == "vitmatte-base" && mp.strandGain == 1.5 && mp.lowContrast == recipe::LowContrast::Off &&
          mp.norm == recipe::Norm::ImageNet && mp.sizeScale == 0);
    ui.rebuildMinConf = 0.6;
    ui.minPiecePx = 20;
    const tracer::TracerParams tp = tracerParams(ui);
    CHECK(tp.minConfUnanchored == 0.6 && tp.minPieceLen1080 == 20);
    CHECK(tp.softProfile && tp.taperW == 0.75 && tp.chainLength && tp.coneJoin);  // agents hl96, hl_flicker
    const vote::VoteParams vp = voteParams(tp);  // the real bars + the presence blend (agent hl_flicker)
    CHECK(vp.minConf == tp.keepConf && vp.minConfUnanchored == 0.6 && vp.blendNear == 1.0 && vp.blendFar == 0.5 &&
          vp.presenceLo == 0.15 && vp.presenceHi == 0.5);
    CHECK(std::string(modelId(0)) == "vitmatte-small" && std::string(modelId(1)) == "vitmatte-base");
  }
  {  // gate texts
    CHECK(gateText(false, 0, true, true) == "Model: off");
    CHECK(gateText(true, 0, false, true) == "Model: needs setup.sh");  // a closed gate (release bundle)
    CHECK(gateText(true, 1, true, false) == "Model Base: not installed");
    CHECK(gateText(true, 0, true, true) == "Model Small: ready");
  }
  {  // Plan texts
    Recipe r;
    PlanResult res;
    PlanInputs in;
    in.recipe = &r;
    in.res = &res;
    in.gate = "Model Small: ready";
    // nothing connected, nothing ran
    res.steps = {rep("model", "Edge Model", false, "no Focus Region"), rep("temporal", "Temporal", false, "off")};
    PlanTexts t = planTexts(in);
    CHECK(t.plan == "Nothing ran: pass-through" && t.rebuild == "Rebuild Detail off" && t.model == "Model Small: ready");
    CHECK(t.inputs == "Region off, Detail off" && t.note == "Connect a Focus Region" && fits(t));
    // everything ran, the generator marked
    in.ui.rebuild = true;
    in.ui.detailConnected = true;
    in.regionConnected = true;
    in.regionPixels = 123456;
    res.steps = {rep("jumpRepair", "Jump Repair", true), rep("regionWeight", "Region", true),
                 rep("model", "Edge Model", true, "", "1 crop"), rep("tracer", "Rebuild Detail", true, "", "3 pieces"),
                 rep("detail", "Detail", true), rep("strandVote", "Vote", false, "not built in this version"),
                 rep("fusion", "Fusion", true), rep("snapRegion", "Snap", true), rep("temporal", "Temporal", true)};
    t = planTexts(in);
    std::printf("all on: '%s' | '%s' | '%s' | '%s' | '%s'\n", t.plan.c_str(), t.rebuild.c_str(), t.model.c_str(),
                t.inputs.c_str(), t.note.c_str());
    CHECK(t.plan == "Jump+Model+Rebuild+Detail..." && t.rebuild == "Created matte: 3 pieces" &&
          t.model == "Model Small: ran" && t.inputs == "Region 123k px, Detail on" && t.note == "OK" && fits(t));
    // a failure leads the note; the model's failure shows on its line
    res.steps[2].ran = false;
    res.steps[2].failed = true;
    res.steps[2].note = "runner not available";
    t = planTexts(in);
    CHECK(t.model == "Model failed: see log" && t.note == "Edge Model failed: see log" && fits(t));
    // a Source warning wins the note; an empty region; no neighbours; end frame
    in.warning = "No alpha: use source alpha";
    CHECK(planTexts(in).note == "No alpha: use source alpha");
    in.warning.clear();
    res.steps[2].failed = false;
    in.regionPixels = 0;
    t = planTexts(in);
    CHECK(t.inputs == "Region empty, Detail on" && t.note == "Focus Region is empty");
    in.regionPixels = 500;
    res.steps[0].summary = "first/last frame: not judged";
    CHECK(planTexts(in).note == "Jump Repair: end frame" && planTexts(in).inputs == "Region 500 px, Detail on");
    res.steps[0] = rep("jumpRepair", "Jump Repair", false, "no neighbours");
    CHECK(planTexts(in).note == "No neighbour frames");
    // Rebuild on but skipped / failed
    res.steps[3] = rep("tracer", "Rebuild Detail", false, "no Focus Region");
    CHECK(planTexts(in).rebuild == "Rebuild: no Focus Region");
    res.steps[3].failed = true;
    CHECK(planTexts(in).rebuild == "Rebuild failed: see log");
    // the recipe did not run (a Source problem)
    in.res = nullptr;
    in.warning = "Source alpha is constant";
    t = planTexts(in);
    CHECK(t.plan == "Nothing ran: pass-through" && t.note == "Source alpha is constant" && fits(t));
    CHECK(fitPlan(std::string(40, 'x')).size() == size_t(kPlanMaxChars));
  }
  {  // view colours
    float c[3];
    candidatesViewRgb(1.0f, 0.0f, 0.0f, 0.0f, c);
    CHECK(c[0] == 0.5f && c[1] == 0.5f && c[2] == 0.5f);  // the base in half grey
    candidatesViewRgb(0.0f, 0.8f, 0.6f, 0.4f, c);
    CHECK(c[0] == 0.8f && c[1] == 0.6f && c[2] == 0.4f);  // model red, tracer green, Detail blue
    rebuiltViewRgb(1.0f, 0.0f, 0.0f, c);
    CHECK(c[0] == 0.3f && c[1] == 0.3f && c[2] == 0.3f);
    rebuiltViewRgb(0.5f, 0.9f, 0.0f, c);
    CHECK(c[0] == 0.15f && c[1] == 0.9f && c[2] == 0.15f);
  }
  std::printf("test_hair_locks_ui: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
