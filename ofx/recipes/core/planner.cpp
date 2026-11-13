#include "planner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "ops/morphology.h"
#include "render_chain.h"
#include "temporal.h"

namespace mr {
namespace recipes {

static double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

StepNeeds unionNeeds(const Recipe& r) {
  StepNeeds u;
  for (const auto& e : r.entries) {
    if (!e.enabled) continue;
    const StepNeeds n = e.step->needs();
    u.flags |= n.flags;
    u.neighbourReach = std::max(u.neighbourReach, n.neighbourReach);
  }
  return u;
}

std::string skipReason(unsigned f, const FrameInputs& in, const FrameStats& stats) {
  if ((f & kNeedPicture) && !in.rgb) return "no picture";
  if ((f & kNeedRegion) && !in.region) return "no Focus Region";
  if ((f & kNeedRegion) && stats.regionPixels == 0) return "Focus Region empty";
  if ((f & kNeedDetail) && !in.detail) return "no Detail";
  if ((f & kNeedModel) && !in.modelGateOpen) return "needs setup.sh";
  if ((f & kNeedNeighbours) && (!in.nb || in.nb->offsets.empty())) return "no neighbours";
  return std::string();
}

void runRecipe(const Recipe& r, const FrameInputs& in, Workspace& ws, float* out, PlanResult* res, Workers* wk,
               QcLayer* qc) {
  const auto t0 = std::chrono::steady_clock::now();
  const size_t n = size_t(std::max(in.w, 0)) * std::max(in.h, 0);
  std::unique_ptr<Workers> own;
  if (!wk) {
    own.reset(new Workers());
    wk = own.get();
  }
  ws.ev.beginRender();
  *res = PlanResult();
  res->stats = computeStats(in);
  ws.cur.assign(in.base, in.base + n);
  ws.next.resize(n);
  uint64_t version = 1;  // the input matte; 0 is reserved for inputs-only evidence
  res->steps.assign(r.entries.size(), StepReport());
  for (size_t i = 0; i < r.entries.size(); ++i) {
    const RecipeEntry& e = r.entries[i];
    StepReport& rep = res->steps[i];
    rep.id = e.step->id();
    rep.label = e.step->label();
    rep.kind = e.step->kind();
    if (!e.enabled) {
      rep.note = "off";
      continue;
    }
    rep.note = skipReason(e.step->needs().flags, in, res->stats);
    if (!rep.note.empty()) continue;
    StepContext c{in, res->stats, ws.ev, version, wk, qc};
    std::string why;
    if (e.when.fn && !e.when.fn(c, &why)) {
      rep.note = why.empty() ? std::string("condition not met") : why;
      continue;
    }
    const bool writes = e.step->writesMatte();
    const auto ts = std::chrono::steady_clock::now();
    try {
      e.step->run(c, ws.cur.data(), writes ? ws.next.data() : nullptr, &rep);
      rep.ran = true;
    } catch (const std::exception& ex) {
      rep.failed = true;
      rep.note = ex.what();
    } catch (...) {
      rep.failed = true;
      rep.note = "unknown error";
    }
    rep.ms = msSince(ts);
    if (rep.failed) {  // input passes through: next (a partial write) is discarded
      rep.changed = 0;
      continue;
    }
    if (rep.kind == StepKind::Rebuilds) res->rebuilt = true;
    // Only a real change makes matte-derived evidence stale (compare, do not trust the step's count).
    if (writes && std::memcmp(ws.cur.data(), ws.next.data(), n * sizeof(float)) != 0) {
      ws.cur.swap(ws.next);
      ws.ev.dropMatteVersionsBelow(++version);
    }
  }
  if (n) std::memcpy(out, ws.cur.data(), n * sizeof(float));
  res->finalVersion = version;
  res->ms = msSince(t0);
}

std::string planLine(const Recipe& r, const PlanResult& res) {
  std::string s = r.name + " v" + std::to_string(r.major) + "." + std::to_string(r.minor) + ":";
  std::string ran, skipped, failed;
  for (const auto& rep : res.steps) {
    if (rep.ran) {
      ran += (ran.empty() ? " " : ", ") + rep.label + (rep.summary.empty() ? "" : " (" + rep.summary + ")");
    } else if (rep.failed) {
      failed += (failed.empty() ? "" : ", ") + rep.label + " (" + rep.note + ")";
    } else if (rep.note != "off") {
      skipped += (skipped.empty() ? "" : ", ") + rep.label + " (" + rep.note + ")";
    }
  }
  s += ran.empty() ? " nothing ran" : ran;
  if (!skipped.empty()) s += " | skipped: " + skipped;
  if (!failed.empty()) s += " | failed: " + failed;
  return s;
}

std::string planNotice(const PlanResult& res) {
  for (const auto& rep : res.steps)
    if (rep.ran && rep.kind == StepKind::Rebuilds) return rep.label + " created matte from the picture.";
  return std::string();
}

std::string planDebugLine(const Recipe& r, const PlanResult& res) {
  std::string s = r.id + " rev " + std::to_string(r.rev);
  char b[96];
  for (const auto& rep : res.steps) {
    const char* st = rep.ran ? "ran" : rep.failed ? "fail" : rep.note == "off" ? "off" : "skip";
    std::snprintf(b, sizeof b, " %s=%s %.1fms %lld", rep.id.c_str(), st, rep.ms, rep.changed);
    s += b;
    if (!rep.ran && rep.note != "off") s += " [" + rep.note + "]";
  }
  std::snprintf(b, sizeof b, " | total %.1fms", res.ms);
  return s + b;
}

Footprint estimateFootprint(const FootprintInput& f) {
  const size_t plane = size_t(std::max(f.w, 0)) * std::max(f.h, 0) * sizeof(float);
  Footprint r;
  r.inputs = plane * (4 + (f.region ? 1 : 0) + (f.detail ? 1 : 0));
  r.work = plane * size_t(std::max(f.workPlanes, 0));
  r.evidence = plane * size_t(std::max(f.evidencePlanes, 0));
  r.neighbours = plane * (4 * size_t(std::max(f.pictureNeighbours, 0)) + size_t(std::max(f.matteOnlyNeighbours, 0)));
  const int s = ml::flowScale(f.h, f.flowLines);
  const size_t grid = size_t((f.w + s - 1) / s) * size_t((f.h + s - 1) / s) * sizeof(float);
  r.flow = 3 * grid * size_t(std::max(f.flowNeighbours, 0));
  r.crop = size_t(std::max(f.cropPixels, 0LL)) * size_t(std::max(f.cropPlanes, 0)) * sizeof(float);
  return r;
}

}  // namespace recipes
}  // namespace mr
