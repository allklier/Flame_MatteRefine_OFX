#include "steps.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "merge_morph.h"
#include "render_chain.h"

namespace mr {
namespace recipes {

static size_t frameSize(const StepContext& c) { return size_t(c.in.w) * c.in.h; }

// --- Jump Repair ------------------------------------------------------------------------------
StepNeeds JumpRepairStep::needs() const { return StepNeeds{kNeedNeighbours, ml::jumpReach(p.maxSpan)}; }

void JumpRepairStep::run(const StepContext& c, const float* in, float* out, StepReport* r) {
  ml::JumpStats st;
  std::vector<uint8_t>* marks = c.wantsQc(id()) ? &c.qc->marks : nullptr;
  ml::runJumpRepair(in, c.in.w, c.in.h, p, *c.in.nb, out, &st, marks);
  if (marks) c.qc->plane.assign(out, out + frameSize(c));
  r->changed = st.pixelsChanged;
  if (!st.judged) r->summary = "first/last frame: not judged";
  else if (st.repaired) r->summary = std::to_string(st.repaired) + " repaired";
}

// --- Region weight ----------------------------------------------------------------------------
void RegionWeightStep::run(const StepContext& c, const float*, float*, StepReport* r) {
  const Box b = regionBox(c, *rp_);  // computes the weight too
  if (b.empty()) r->summary = "weight empty";
}

// --- Candidates -------------------------------------------------------------------------------
StepNeeds CandidateStep::needs() const {
  StepNeeds n = src_->needs();
  n.flags |= kNeedRegion;
  return n;
}

void CandidateStep::run(const StepContext& c, const float* in, float*, StepReport* r) {
  const Box b = regionBox(c, *rp_);
  if (b.empty()) {
    r->summary = "region weight empty";
    return;
  }
  CandidatePlanes cp = c.ev.newCandidate();
  cp.id = src_->id();
  cp.kind = src_->kind();
  cp.box = b;
  cp.alpha.resize(size_t(b.pixels()));
  cp.conf.resize(size_t(b.pixels()));
  src_->produce(c, in, b, cp.alpha.data(), cp.conf.data(), r);  // a throw: nothing added
  c.ev.addCandidate(std::move(cp));
}

// --- Strand presence vote (slot) ---------------------------------------------------------------
bool hasTracerCandidate(const StepContext& c, std::string* why) {
  if (c.ev.findCandidate("tracer")) return true;
  *why = "no Rebuild Detail result";
  return false;
}

// --- Fusion -----------------------------------------------------------------------------------
bool hasCandidates(const StepContext& c, std::string* why) {
  if (!c.ev.candidates().empty()) return true;
  *why = "no candidate";
  return false;
}

void FusionStep::run(const StepContext& c, const float* in, float* out, StepReport* r) {
  const size_t n = frameSize(c);
  std::memcpy(out, in, n * sizeof(float));
  c.ev.setPlane("regionBase", in, n);
  const std::vector<float>& wgt = regionWeight(c, *rp_);
  const Box b = regionBox(c, *rp_);
  const std::vector<CandidatePlanes>& cands = c.ev.candidates();
  if (b.empty() || cands.empty()) return;
  std::vector<merge::Candidate> mc;
  for (const auto& cp : cands) {
    if (cp.box.x0 != b.x0 || cp.box.y0 != b.y0 || cp.box.x1 != b.x1 || cp.box.y1 != b.y1)
      throw std::invalid_argument("fusion: candidate '" + cp.id + "' is not on the region box");
    merge::Candidate m;
    m.name = cp.id.c_str();
    m.alpha = cp.alpha.data();
    m.confidence = cp.conf.data();
    mc.push_back(m);
  }
  std::vector<float> fused(size_t(b.pixels()));
  merge::fuseCandidates(mc.data(), int(mc.size()), b.w(), b.h(), fp, fused.data(), rule, c.wk);
  long long changed = 0;
  for (int y = b.y0; y < b.y1; ++y)
    for (int x = b.x0; x < b.x1; ++x) {
      const size_t i = size_t(y) * c.in.w + x;
      const float wt = wgt[i];
      if (wt <= 0.0f) continue;
      float v = fused[size_t(y - b.y0) * b.w() + (x - b.x0)];
      if (wt < 1.0f) v = in[i] + wt * (v - in[i]);
      out[i] = std::max(v, in[i]);
      changed += out[i] != in[i];
    }
  r->changed = changed;
  r->summary = std::to_string(cands.size()) + (cands.size() == 1 ? " candidate" : " candidates");
}

// --- Snap in the region -----------------------------------------------------------------------
bool fusionRan(const StepContext& c, std::string* why) {
  if (c.ev.plane("regionBase")) return true;
  *why = "nothing added";
  return false;
}

long long snapAddedWeight(const float* before, const float* after, const float* region, int w, int h,
                          double grow1080, float* weight, Workers* wk, float addedMin) {
  const size_t n = size_t(w) * h;
  std::vector<float> added(n), g1(n), g2(n), scratch;
  long long count = 0;
  for (size_t i = 0; i < n; ++i) count += (added[i] = after[i] - before[i] > addedMin ? 1.0f : 0.0f) > 0.0f;
  if (!count) {
    std::fill(weight, weight + n, 0.0f);
    return 0;
  }
  const int r = grow1080 > 0 ? std::max(1, int(std::lround(grow1080 * h / 1080.0))) : 0;
  if (r == 0) {
    g1 = added;
    g2 = added;
  } else {
    merge::maxFilter(added.data(), w, h, r, g1.data(), wk, &scratch);
    merge::maxFilter(added.data(), w, h, 2 * r, g2.data(), wk, &scratch);
  }
  for (size_t i = 0; i < n; ++i) {
    const float rg = region[i] > 0.0f ? (region[i] < 1.0f ? region[i] : 1.0f) : 0.0f;  // NaN -> 0
    weight[i] = rg * 0.5f * (g1[i] + g2[i]);
  }
  return count;
}

long long dropSnapIslands(const float* in, float* out, int w, int h, float touch) {
  const size_t n = size_t(w) * h;
  std::vector<int> lab(n, -1), stack;
  long long reset = 0;
  std::vector<size_t> comp;
  for (size_t s = 0; s < n; ++s) {
    if (lab[s] >= 0 || !(out[s] > in[s])) continue;
    comp.clear();
    bool touches = false;
    lab[s] = 1;
    stack.assign(1, int(s));
    while (!stack.empty()) {
      const size_t j = size_t(stack.back());
      stack.pop_back();
      comp.push_back(j);
      const int jx = int(j % size_t(w)), jy = int(j / size_t(w));
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int x = jx + dx, y = jy + dy;
          if ((dx == 0 && dy == 0) || x < 0 || y < 0 || x >= w || y >= h) continue;
          const size_t k = size_t(y) * w + x;
          if (in[k] >= touch) touches = true;
          if (lab[k] < 0 && out[k] > in[k]) {
            lab[k] = 1;
            stack.push_back(int(k));
          }
        }
      if (in[j] >= touch) touches = true;
    }
    if (touches) continue;
    for (size_t j : comp) out[j] = in[j];
    reset += (long long)comp.size();
  }
  return reset;
}

void SnapRegionStep::run(const StepContext& c, const float* in, float* out, StepReport* r) {
  const size_t n = frameSize(c);
  std::memcpy(out, in, n * sizeof(float));
  const std::vector<float>* rb = c.ev.plane("regionBase");
  if (!rb) return;  // fusionRan guards this
  std::vector<float> wgt(n);
  const long long added =
      snapAddedWeight(rb->data(), in, regionWeight(c, *rp_).data(), c.in.w, c.in.h, addedGrow1080, wgt.data(), c.wk,
                      addedMin);
  if (!added) {
    r->summary = "nothing added";
    return;
  }
  merge::snapInRegion(rb->data(), out, c.in.rgb, wgt.data(), c.in.w, c.in.h, strength,
                      merge::snapParamsForHeight(c.in.h), neverLower);
  if (raiseOnly)
    for (size_t i = 0; i < n; ++i) out[i] = std::max(out[i], in[i]);
  if (islandTouch > 0.0f) dropSnapIslands(in, out, c.in.w, c.in.h, islandTouch);
  long long changed = 0;
  for (size_t i = 0; i < n; ++i) changed += out[i] != in[i];
  r->changed = changed;
}

// --- Temporal ---------------------------------------------------------------------------------
void WeightedTemporal::run(const float* matte, const float* rgb, int w, int h, const ml::StabiliseParams& p,
                           const ml::Neighbours& nb, const float* weight, const ml::TemporalFlows* flows, float* out,
                           ml::StabiliseStats* st) {
  // The frames as render_chain.cpp runTemporal builds them (its checks, its order).
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  if (nb.offsets.size() != nb.frames.size()) throw std::invalid_argument("Temporal: neighbour list mismatch");
  auto used = [&](size_t i) { return std::abs(nb.offsets[i]) <= p.frames; };
  for (size_t i = 0; i < nb.frames.size(); ++i) {
    const ml::FetchedFrame& f = nb.frames[i];
    if (used(i) && (nb.offsets[i] == 0 || f.matte.size() != n || f.rgb.size() != 3 * n))
      throw std::invalid_argument("Temporal: neighbour " + std::to_string(nb.offsets[i]) + " has no " +
                                  std::to_string(w) + "x" + std::to_string(h) + " matte and picture");
  }
  std::vector<ml::TemporalFrame> frames;
  for (size_t i = 0; i < nb.frames.size(); ++i)
    if (used(i) && nb.offsets[i] < 0) frames.push_back({nb.frames[i].rgb.data(), nb.frames[i].matte.data()});
  const int centre = int(frames.size());
  frames.push_back({rgb, matte});
  for (size_t i = 0; i < nb.frames.size(); ++i)
    if (used(i) && nb.offsets[i] > 0) frames.push_back({nb.frames[i].rgb.data(), nb.frames[i].matte.data()});
  usedSharedFlows = false;
  if (flows) {
    try {  // throws before writing out when the flows do not fit
      ml::stabilise(frames.data(), int(frames.size()), centre, w, h, p, *flows, out, st, weight);
      usedSharedFlows = true;
      return;
    } catch (const std::invalid_argument&) {
    }
  }
  ml::stabilise(frames.data(), int(frames.size()), centre, w, h, p, out, st, weight);
}

void TemporalOutsideStep::run(const StepContext& c, const float* in, float* out, StepReport* r) {
  const float* weight = nullptr;
  if (c.in.region && rp_) {
    weight = c.ev.get({"outsideRegion", 0, paramHash({rp_->growPx, rp_->featherPx})}, [&](std::vector<float>* v) {
                   const std::vector<float>& rw = regionWeight(c, *rp_);
                   v->resize(rw.size());
                   for (size_t i = 0; i < rw.size(); ++i) (*v)[i] = 1.0f - rw[i];
                 }).data();
  }
  ml::StabiliseStats st;
  runner_->run(in, c.in.rgb, c.in.w, c.in.h, p, *c.in.nb, weight, c.ev.flows.get(), out, &st);
  long long changed = 0;
  for (size_t i = 0; i < frameSize(c); ++i) changed += out[i] != in[i];
  r->changed = changed;
  r->summary = std::to_string(st.neighbours) + (st.neighbours == 1 ? " neighbour" : " neighbours");
}

}  // namespace recipes
}  // namespace mr
