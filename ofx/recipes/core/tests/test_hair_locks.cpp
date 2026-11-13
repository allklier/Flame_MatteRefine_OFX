// The Hair Locks recipe through the planner with the real adapters (Jump Repair, region weight,
// fusion, Snap in the region, Temporal outside the region) and fake candidate sources
// (docs/edge_recipes.md 6: all off == input, == the cores chained by hand, region bit-identity,
// conditions, failing candidate, memory). Synthetic sequence only (synth::makeSequence).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "hair_locks.h"
#include "render_chain.h"
#include "synth.h"

using namespace mr;
using namespace mr::recipes;

static int failures = 0;
#define CHECK(c)                                              \
  do {                                                        \
    if (!(c)) {                                               \
      std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                             \
    }                                                         \
  } while (0)

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// A fake candidate: vertical strands at given columns (frame x), alpha 1 with a confidence, 0 elsewhere.
struct FakeSource : CandidateSource {
  const char* id_;
  const char* label_;
  StepKind kind_;
  unsigned needs_ = 0;
  std::vector<int> cols;
  float conf = 0.8f;
  bool fail = false;
  int calls = 0;
  FakeSource(const char* i, const char* l, StepKind k) : id_(i), label_(l), kind_(k) {}
  const char* id() const override { return id_; }
  const char* label() const override { return label_; }
  StepKind kind() const override { return kind_; }
  StepNeeds needs() const override { return StepNeeds{needs_, 0}; }
  float alphaAt(int x) const { return std::find(cols.begin(), cols.end(), x) != cols.end() ? 1.0f : 0.0f; }
  void produce(const StepContext&, const float*, const Box& b, float* alpha, float* cf, StepReport* r) override {
    ++calls;
    alpha[0] = 0.5f;  // a partial write before a failure must not matter
    if (fail) throw std::runtime_error("tracer lost");
    for (int y = b.y0; y < b.y1; ++y)
      for (int x = b.x0; x < b.x1; ++x) {
        const size_t i = size_t(y - b.y0) * b.w() + (x - b.x0);
        alpha[i] = alphaAt(x);
        cf[i] = conf;
      }
    r->summary = std::to_string(cols.size()) + " pieces";
  }
};

// A fake vote: zeroes the tracer candidate's alpha (the vote dropping every piece), reach 2.
struct FakeVote : Step {
  int calls = 0;
  const char* id() const override { return "strandVote"; }
  const char* label() const override { return "Vote"; }
  StepKind kind() const override { return StepKind::Rebuilds; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture, 2}; }
  bool writesMatte() const override { return false; }
  void run(const StepContext& c, const float*, float*, StepReport* r) override {
    ++calls;
    CandidatePlanes* t = c.ev.findCandidate("tracer");
    std::fill(t->alpha.begin(), t->alpha.end(), 0.0f);
    r->summary = "kept 0 dropped 3 reinstated 0";
  }
};

struct Fixture {
  int w = 256, h = 144;
  synth::Sequence seq;
  std::vector<float> base, region;
  const float* rgb = nullptr;
  ml::Neighbours nb;
  int edgeX = 0, y0 = 0, y1 = 0;
  Fixture() {
    seq = synth::makeSequence(w, h, 5, 3);
    base = seq.frames[2].matte;
    rgb = seq.frames[2].rgb.data();
    for (int y = 6; y < 18; ++y)  // a one-frame jump in the background, top left
      for (int x = 6; x < 18; ++x) base[size_t(y) * w + x] = 1.0f;
    const int yc = h / 2;
    for (int x = w - 1; x >= 0; --x)
      if (base[size_t(yc) * w + x] > 0.5f) {
        edgeX = x;
        break;
      }
    y0 = yc - 30;
    y1 = yc + 30;
    region.assign(size_t(w) * h, 0.0f);
    for (int y = y0; y < y1; ++y)
      for (int x = edgeX - 20; x < edgeX + 25; ++x) region[size_t(y) * w + x] = x < edgeX - 15 ? 0.5f : 1.0f;
    for (int k : {-2, -1, 1, 2}) nb.offsets.push_back(k);
    for (int k : nb.offsets) {
      ml::FetchedFrame f;
      const synth::Scene& s = seq.frames[2 + k];
      f.image = true;
      f.w = w;
      f.h = h;
      f.components = 4;
      f.isFloat = true;
      f.matte = s.matte;
      f.rgb = s.rgb;
      nb.frames.push_back(f);
    }
  }
  FrameInputs inputs(bool withRegion = true, bool withNb = true) const {
    FrameInputs in;
    in.w = w;
    in.h = h;
    in.rgb = rgb;
    in.base = base.data();
    in.region = withRegion ? region.data() : nullptr;
    in.nb = withNb ? &nb : nullptr;
    in.modelGateOpen = false;
    return in;
  }
};

static HairLocksSettings allOn(size_t candidates) {
  HairLocksSettings s;
  s.jumpRepair = true;
  s.jr.minSizePercent = 0.2;  // the 144-px blob is 0.39 % of 256 x 144
  s.region.featherPx = 3;
  s.candidates.assign(candidates, true);
  s.snap = true;
  s.temporal = true;
  return s;
}

// The cores chained by hand, the same order and params as the recipe.
static std::vector<float> handChain(const Fixture& f, const HairLocksSettings& s, const std::vector<FakeSource*>& src,
                                    std::vector<float>* beforeTemporal = nullptr, std::vector<float>* weightOut = nullptr) {
  const int w = f.w, h = f.h;
  const size_t n = size_t(w) * h;
  std::vector<float> a(n);
  ml::JumpStats js;
  if (s.jumpRepair) ml::runJumpRepair(f.base.data(), w, h, s.jr, f.nb, a.data(), &js);
  else a = f.base;
  std::vector<float> wgt(n);
  merge::prepareRegion(f.region.data(), w, h, s.region.growPx, s.region.featherPx, wgt.data());
  int bx0 = w, by0 = h, bx1 = 0, by1 = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (wgt[size_t(y) * w + x] > 0) {
        bx0 = std::min(bx0, x), by0 = std::min(by0, y), bx1 = std::max(bx1, x + 1), by1 = std::max(by1, y + 1);
      }
  const int bw = bx1 - bx0, bh = by1 - by0;
  std::vector<std::vector<float>> al, cf;
  std::vector<merge::Candidate> mc;
  for (size_t k = 0; k < src.size(); ++k) {
    if (!s.candidates[k] || src[k]->fail) continue;
    al.emplace_back(size_t(bw) * bh);
    cf.emplace_back(size_t(bw) * bh, src[k]->conf);
    for (int y = 0; y < bh; ++y)
      for (int x = 0; x < bw; ++x) al.back()[size_t(y) * bw + x] = src[k]->alphaAt(bx0 + x);
  }
  for (size_t k = 0; k < al.size(); ++k) mc.push_back(merge::Candidate{"c", al[k].data(), cf[k].data()});
  std::vector<float> b = a;
  if (!mc.empty()) {
    std::vector<float> fused(size_t(bw) * bh);
    merge::fuseCandidates(mc.data(), int(mc.size()), bw, bh, s.fusion, fused.data());
    for (int y = by0; y < by1; ++y)
      for (int x = bx0; x < bx1; ++x) {
        const size_t i = size_t(y) * w + x;
        if (wgt[i] <= 0) continue;
        float v = fused[size_t(y - by0) * bw + (x - bx0)];
        if (wgt[i] < 1.0f) v = a[i] + wgt[i] * (v - a[i]);
        b[i] = std::max(v, a[i]);
      }
  }
  if (s.snap && !mc.empty()) {  // Snap on the new pixels only (the added mask, steps.h snapAddedWeight)
    std::vector<float> sw(n);
    if (snapAddedWeight(a.data(), b.data(), wgt.data(), w, h, 0.0, sw.data()) > 0)
      merge::snapInRegion(a.data(), b.data(), f.rgb, sw.data(), w, h, s.snapStrength, merge::snapParamsForHeight(h), true);
  }
  if (beforeTemporal) *beforeTemporal = b;
  if (weightOut) *weightOut = wgt;
  if (!s.temporal) return b;
  // Temporal with the weight 1 - region inside the core (temporal.h stabilise(..., weight)), the frames
  // in time order as render_chain.cpp runTemporal builds them.
  std::vector<float> d(n), outside(n);
  for (size_t i = 0; i < n; ++i) outside[i] = 1.0f - wgt[i];
  std::vector<ml::TemporalFrame> fr;
  for (size_t i = 0; i < f.nb.offsets.size(); ++i)
    if (f.nb.offsets[i] < 0 && -f.nb.offsets[i] <= s.tm.frames) fr.push_back({f.nb.frames[i].rgb.data(), f.nb.frames[i].matte.data()});
  const int centre = int(fr.size());
  fr.push_back({f.rgb, b.data()});
  for (size_t i = 0; i < f.nb.offsets.size(); ++i)
    if (f.nb.offsets[i] > 0 && f.nb.offsets[i] <= s.tm.frames) fr.push_back({f.nb.frames[i].rgb.data(), f.nb.frames[i].matte.data()});
  ml::stabilise(fr.data(), int(fr.size()), centre, w, h, s.tm, d.data(), nullptr, outside.data());
  return d;
}

int main() {
  Fixture f;
  const size_t n = size_t(f.w) * f.h;
  FakeSource model("model", "Edge Model", StepKind::Refines), tracer("tracer", "Rebuild Detail", StepKind::Rebuilds);
  model.cols = {f.edgeX + 6, f.edgeX + 7};
  model.conf = 0.6f;
  tracer.cols = {f.edgeX + 6, f.edgeX + 12, f.edgeX + 24};  // + the region's last column (box edge)
  std::vector<FakeSource*> src = {&model, &tracer};
  FakeVote fakeVote;
  HairLocks hl({&model, &tracer}, nullptr, &fakeVote);
  Workspace ws;
  std::vector<float> out(n);
  PlanResult res;
  std::printf("scene %dx%d, edge x %d\n", f.w, f.h, f.edgeX);

  {  // 1. every step off: the input bit for bit, nothing ran, nothing computed
    HairLocksSettings s;
    s.snap = false;
    s.temporal = false;
    Recipe r = hl.recipe(s);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(same(out, f.base));
    CHECK(ws.ev.computed() == 0 && model.calls == 0);
    CHECK(planLine(r, res) == "Hair Locks v1.0: nothing ran");
    CHECK(unionNeeds(r).flags == 0);
  }
  {  // 2. all on == the cores chained by hand, bit for bit; plan text; Rebuilds notice
    const HairLocksSettings s = allOn(2);
    Recipe r = hl.recipe(s);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    std::vector<float> pre, wgt;
    const std::vector<float> ref = handChain(f, s, src, &pre, &wgt);
    CHECK(same(out, ref));
    for (const auto& rep : res.steps) CHECK(rep.ran || (rep.id == "strandVote" && rep.note == "off"));
    CHECK(res.steps[0].changed > 0);  // Jump Repair repaired the blob
    for (size_t i = 5; i < res.steps.size(); ++i) CHECK(res.steps[i].changed > 0);  // fusion, Snap, Temporal
    const std::string line = planLine(r, res);
    std::printf("plan: %s\nnotice: %s\ndebug: %s\n", line.c_str(), planNotice(res).c_str(), planDebugLine(r, res).c_str());
    CHECK(line == "Hair Locks v1.0: Jump Repair (1 repaired), Region, Edge Model (2 pieces), Rebuild Detail (3 pieces), "
                  "Fusion (2 candidates), Snap, Temporal (2 neighbours)");
    CHECK(planNotice(res) == "Rebuild Detail created matte from the picture.");
    const StepNeeds u = unionNeeds(r);
    CHECK(u.flags == (kNeedNeighbours | kNeedRegion | kNeedPicture) && u.neighbourReach == 2);
    // 3. region bit-identity: inside the region (weight 1) Temporal left the matte as Snap made it;
    // where the weight is 0 the region steps left Jump Repair's output (checked in 4)
    long long inside = 0, insideDiff = 0, outsideDiff = 0;
    for (size_t i = 0; i < n; ++i) {
      if (wgt[i] >= 1.0f) ++inside, insideDiff += out[i] != pre[i];
      if (wgt[i] <= 0.0f) outsideDiff += out[i] != pre[i];
    }
    CHECK(inside > 0 && insideDiff == 0);
    CHECK(outsideDiff > 0);  // Temporal acted outside (the check is not vacuous)
    // memory: what the evidence holds = 3 full planes (regionWeight, outsideRegion, regionBase),
    // the box (4 floats), 2 candidates x 2 crop planes; the estimate agrees
    const FrameInputs fin = f.inputs();
    const Box b = regionBox(StepContext{fin, res.stats, ws.ev, 1, nullptr, nullptr}, s.region);  // a hit
    const size_t expect = (3 * n + 4 + 4 * size_t(b.pixels())) * sizeof(float);
    CHECK(ws.ev.bytesHeld() == expect);
    FootprintInput fi;
    fi.w = f.w;
    fi.h = f.h;
    fi.evidencePlanes = 3;
    fi.cropPixels = b.pixels();
    fi.cropPlanes = 4;
    const Footprint fp = estimateFootprint(fi);
    CHECK(fp.evidence + fp.crop + 4 * sizeof(float) == ws.ev.bytesHeld());
    CHECK(ws.cur.size() == n && ws.next.size() == n);
  }
  {  // 4. region steps only (no Jump Repair, no Temporal): outside the region == the input bit for
    // bit, never below the input anywhere, and something changed inside
    HairLocksSettings s = allOn(2);
    s.jumpRepair = false;
    s.temporal = false;
    Recipe r = hl.recipe(s);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(same(out, handChain(f, s, src)));
    long long outside = 0, below = 0, inside = 0;
    for (size_t i = 0; i < n; ++i) {
      outside += f.region[i] == 0.0f && out[i] != f.base[i];
      below += out[i] < f.base[i];
      inside += f.region[i] > 0.0f && out[i] != f.base[i];
    }
    CHECK(outside == 0 && below == 0 && inside > 0);
  }
  {  // 5. conditions: no neighbours -> Jump Repair and Temporal skipped, the rest runs
    const HairLocksSettings s = allOn(2);
    Recipe r = hl.recipe(s);
    runRecipe(r, f.inputs(true, false), ws, out.data(), &res);
    HairLocksSettings s2 = s;
    s2.jumpRepair = s2.temporal = false;
    CHECK(same(out, handChain(f, s2, src)));
    CHECK(res.steps[0].note == "no neighbours" && res.steps.back().note == "no neighbours");
    CHECK(planLine(r, res) == "Hair Locks v1.0: Region, Edge Model (2 pieces), Rebuild Detail (3 pieces), Fusion (2 "
                              "candidates), Snap | skipped: Jump Repair (no neighbours), Temporal (no neighbours)");
    // the model needs the gate: closed -> skipped, open -> runs
    model.needs_ = kNeedModel;
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[2].note == "needs setup.sh" && res.steps[3].ran && res.steps[5].summary == "1 candidate");
    CHECK(unionNeeds(r).flags & kNeedModel);
    FrameInputs in = f.inputs();
    in.modelGateOpen = true;
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(res.steps[2].ran && same(out, handChain(f, s, src)));
    model.needs_ = 0;
    // no Focus Region: the region steps skip; Jump Repair and Temporal (everywhere) run
    runRecipe(r, f.inputs(false), ws, out.data(), &res);
    for (size_t i : {1, 2, 3, 5, 6}) CHECK(res.steps[i].note == "no Focus Region");
    CHECK(res.steps[4].note == "off");  // the vote slot
    std::vector<float> a(n), d(n);
    ml::runJumpRepair(f.base.data(), f.w, f.h, s.jr, f.nb, a.data(), nullptr);
    bool ab = false;
    ml::runTemporal(a.data(), f.rgb, f.w, f.h, s.tm, f.nb, d.data(), nullptr, &ab);
    CHECK(same(out, d));
    // an empty region
    std::vector<float> empty(n, 0.0f);
    in = f.inputs();
    in.region = empty.data();
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(res.steps[1].note == "Focus Region empty" && res.steps[5].note == "Focus Region empty");
    // a region the Grow erodes away: candidates add nothing, the fusion has no candidate
    HairLocksSettings s3 = s;
    s3.region.growPx = -40;
    r = hl.recipe(s3);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[1].summary == "weight empty" && res.steps[2].summary == "region weight empty");
    CHECK(res.steps[5].note == "no candidate");
    // one candidate off: only the other is fused
    HairLocksSettings s4 = s;
    s4.candidates = {false, true};
    r = hl.recipe(s4);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[2].note == "off" && same(out, handChain(f, s4, src)));
  }
  {  // 6. a failing candidate: reported, nothing added, the other candidate fused, the chain goes on
    const HairLocksSettings s = allOn(2);
    Recipe r = hl.recipe(s);
    tracer.fail = true;
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[3].failed && res.steps[3].note == "tracer lost");
    CHECK(ws.ev.candidates().size() == 1 && res.steps[5].summary == "1 candidate");
    CHECK(same(out, handChain(f, s, src)));  // handChain skips failing sources
    CHECK(!res.rebuilt && planNotice(res).empty());
    // every candidate failing: the fusion is skipped, Snap and Temporal still run
    model.fail = true;
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[5].note == "no candidate" && res.steps[6].note == "nothing added" && res.steps[7].ran);
    CHECK(same(out, handChain(f, s, src)));
    model.fail = tracer.fail = false;
  }
  {  // 7. first frame: Jump Repair not judged (passes through); QC marks only when asked
    ml::Neighbours fwd;
    for (size_t i = 0; i < f.nb.offsets.size(); ++i)
      if (f.nb.offsets[i] > 0) fwd.offsets.push_back(f.nb.offsets[i]), fwd.frames.push_back(f.nb.frames[i]);
    HairLocksSettings s = allOn(0);
    s.snap = s.temporal = false;
    Recipe r = hl.recipe(s);
    FrameInputs in = f.inputs();
    in.nb = &fwd;
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(res.steps[0].ran && res.steps[0].summary == "first/last frame: not judged" && same(out, f.base));
    QcLayer qc;
    qc.step = "jumpRepair";
    runRecipe(r, f.inputs(), ws, out.data(), &res, nullptr, &qc);
    CHECK(qc.marks.size() == n && std::count(qc.marks.begin(), qc.marks.end(), uint8_t(ml::kJumpRepaired)) > 0);
    QcLayer other;
    other.step = "snapRegion";
    runRecipe(r, f.inputs(), ws, out.data(), &res, nullptr, &other);
    CHECK(other.marks.empty() && other.plane.empty());
  }
  {  // 8. nothing survives a render: a reused workspace gives what a fresh one gives
    HairLocksSettings s = allOn(2);
    s.region.featherPx = 6;
    Recipe r = hl.recipe(s);
    Workspace fresh;
    std::vector<float> o2(n);
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    runRecipe(r, f.inputs(), fresh, o2.data(), &res);
    CHECK(same(out, o2));
  }
  {  // 9. the injected vote: after the candidates, before the fusion; it revises the tracer's candidate
    HairLocksSettings s = allOn(2);
    s.jumpRepair = false;
    s.temporal = false;
    s.candidates = {true, false};  // the model only
    std::vector<float> o2(n);
    runRecipe(hl.recipe(s), f.inputs(), ws, o2.data(), &res);
    s.candidates = {true, true};
    s.strandVote = true;
    Recipe r = hl.recipe(s);
    CHECK(unionNeeds(r).neighbourReach == 2);  // the vote's reach joins the union
    runRecipe(r, f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[4].id == "strandVote" && res.steps[4].ran && fakeVote.calls == 1);
    CHECK(same(out, o2));  // the tracer's candidate zeroed by the vote: as if only the model ran
    CHECK(planLine(r, res).find("Vote (kept 0 dropped 3 reinstated 0), Fusion") != std::string::npos);
    s.candidates = {true, false};  // no tracer candidate: the vote is skipped
    runRecipe(hl.recipe(s), f.inputs(), ws, out.data(), &res);
    CHECK(res.steps[4].note == "no Rebuild Detail result" && fakeVote.calls == 1);
    HairLocks noVote({&model, &tracer});
    CHECK(noVote.recipe(s).entries.size() == hl.recipe(s).entries.size() - 1);
  }
  {  // 10. Snap acts on the new pixels only (decision 39; WP8 scoring): no candidate -> Snap is a no-op (the
    // same as Snap off, bit for bit); with a candidate it changes pixels only inside the grown added mask
    HairLocksSettings s = allOn(2);
    s.jumpRepair = s.temporal = false;
    s.candidates = {false, false};
    std::vector<float> o2(n);
    runRecipe(hl.recipe(s), f.inputs(), ws, out.data(), &res);
    s.snap = false;
    runRecipe(hl.recipe(s), f.inputs(), ws, o2.data(), &res);
    CHECK(same(out, o2) && same(out, f.base));
    s = allOn(2);
    s.jumpRepair = s.temporal = false;
    s.snap = false;
    runRecipe(hl.recipe(s), f.inputs(), ws, o2.data(), &res);  // the fusion's result
    s.snap = true;
    runRecipe(hl.recipe(s), f.inputs(), ws, out.data(), &res);
    std::vector<float> wgt(n), sw(n);
    merge::prepareRegion(f.region.data(), f.w, f.h, s.region.growPx, s.region.featherPx, wgt.data());
    snapAddedWeight(f.base.data(), o2.data(), wgt.data(), f.w, f.h, 0.0, sw.data());  // the default: added px only
    long long changed = 0, outsideMask = 0;
    for (size_t i = 0; i < n; ++i) {
      changed += out[i] != o2[i];
      outsideMask += sw[i] == 0.0f && out[i] != o2[i];
    }
    std::printf("snap on new pixels: changed %lld, outside the added mask %lld\n", changed, outsideMask);
    CHECK(changed > 0 && outsideMask == 0);
    // the mask: brute force on a single added pixel (r = 3 at 1080 lines -> 1 at 144)
    std::vector<float> b0(n, 0.0f), a0(n, 0.0f), one(n, 1.0f), m(n);
    a0[size_t(70) * f.w + 100] = 0.5f;
    CHECK(snapAddedWeight(b0.data(), a0.data(), one.data(), f.w, f.h, 3.0, m.data()) == 1);
    long long bad = 0;
    for (int y = 0; y < f.h; ++y)
      for (int x = 0; x < f.w; ++x) {
        const int d = std::max(std::abs(x - 100), std::abs(y - 70));
        bad += m[size_t(y) * f.w + x] != (d <= 1 ? 1.0f : d <= 2 ? 0.5f : 0.0f);
      }
    CHECK(bad == 0);
    CHECK(snapAddedWeight(b0.data(), b0.data(), one.data(), f.w, f.h, 3.0, m.data()) == 0);
    CHECK(snapAddedWeight(b0.data(), a0.data(), one.data(), f.w, f.h, 0.0, m.data()) == 1);  // growth 0: the pixel only
    long long on = 0;
    for (float v : m) on += v != 0.0f;
    CHECK(on == 1 && m[size_t(70) * f.w + 100] == 1.0f);
    // addedMin: a rise of 0.02 is not added at 0.05, is at the default 1e-3
    std::vector<float> a1(n, 0.0f);
    a1[size_t(70) * f.w + 100] = 0.02f;
    CHECK(snapAddedWeight(b0.data(), a1.data(), one.data(), f.w, f.h, 0.0, m.data()) == 1);
    CHECK(snapAddedWeight(b0.data(), a1.data(), one.data(), f.w, f.h, 0.0, m.data(), nullptr, 0.05f) == 0);
  }
  {  // dropSnapIslands: a raised blob touching nothing >= 0.3 goes back to the input; one touching stays
    const int w = 20, h = 10;
    std::vector<float> in(size_t(w) * h, 0.0f), out;
    for (int y = 0; y < h; ++y) in[size_t(y) * w + 0] = 1.0f;  // the matte: column 0
    out = in;
    out[size_t(5) * w + 1] = out[size_t(5) * w + 2] = 0.6f;  // raised, touches column 0
    out[size_t(5) * w + 12] = out[size_t(6) * w + 13] = 0.7f;  // raised island (8-connected pair)
    out[size_t(2) * w + 0] = 0.5f;  // lowered: never touched
    const long long reset = dropSnapIslands(in.data(), out.data(), w, h, 0.3f);
    CHECK(reset == 2);
    CHECK(out[size_t(5) * w + 12] == 0.0f && out[size_t(6) * w + 13] == 0.0f);
    CHECK(out[size_t(5) * w + 1] == 0.6f && out[size_t(5) * w + 2] == 0.6f && out[size_t(2) * w + 0] == 0.5f);
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
