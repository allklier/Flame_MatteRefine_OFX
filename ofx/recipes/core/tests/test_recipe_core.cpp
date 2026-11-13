// Edge Recipes core: step interface, evidence, planner (docs/edge_recipes.md 2-3, 6). Synthetic data
// only. Steps here are test-local adapters over plugin 1's cores (Solid Core, Snap to Image).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ops/snap_to_image.h"
#include "ops/solid_core.h"
#include "planner.h"
#include "render_chain.h"

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

struct SolidCoreStep : Step {
  SolidCoreParams p;
  const char* id() const override { return "solidCore"; }
  const char* label() const override { return "Solid Core"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds(); }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override {
    SolidCoreStats st;
    solidCore(in, out, c.in.w, c.in.h, p, &st);
    r->changed = st.pixelsChanged;
  }
};
struct SnapStep : Step {
  SnapParams p;
  RegionParams rp;
  const char* id() const override { return "snap"; }
  const char* label() const override { return "Snap"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture | kNeedRegion, 0}; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override {
    SnapStats st;
    snapToImage(in, c.in.rgb, out, c.in.w, c.in.h, p, &st, regionWeight(c, rp).data());
    r->changed = st.pixelsChanged;
  }
};
// Reads matte-derived evidence and changes nothing.
struct ProbeStep : Step {
  const char* name;
  explicit ProbeStep(const char* n) : name(n) {}
  const char* id() const override { return name; }
  const char* label() const override { return name; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds(); }
  void run(const StepContext& c, const float* in, float* out, StepReport*) override {
    (void)signedDistance(c, in, 12.0);
    std::memcpy(out, in, sizeof(float) * c.in.w * c.in.h);
  }
};
struct ThrowStep : Step {
  bool unknown = false;
  const char* id() const override { return "model"; }
  const char* label() const override { return "Edge Model"; }
  StepKind kind() const override { return StepKind::Refines; }
  StepNeeds needs() const override { return StepNeeds{kNeedPicture, 0}; }
  void run(const StepContext&, const float*, float* out, StepReport* r) override {
    out[0] = 0.123f;  // a partial write must not leak
    r->changed = 99;
    if (unknown) throw 7;
    throw std::runtime_error("runner not available");
  }
};
struct NeedStep : Step {
  StepNeeds n;
  StepKind k = StepKind::Refines;
  const char* nm = "need";
  const char* id() const override { return nm; }
  const char* label() const override { return nm; }
  StepKind kind() const override { return k; }
  StepNeeds needs() const override { return n; }
  void run(const StepContext& c, const float* in, float* out, StepReport* r) override {
    std::memcpy(out, in, sizeof(float) * c.in.w * c.in.h);
    out[0] = in[0] > 0.5f ? 0.0f : 1.0f;  // a real change of one pixel
    r->changed = 1;
  }
};

struct Scene {
  int w = 320, h = 240;
  std::vector<float> base, rgb, region;
  Scene() {
    base.resize(w * h);
    rgb.resize(w * h * 3);
    region.assign(w * h, 0.0f);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double d = std::hypot(x - 160.0, y - 120.0);
        float a = float(std::min(1.0, std::max(0.0, (80.0 - d) / 3.0 + 0.5)));
        if (std::hypot(x - 150.0, y - 110.0) < 12) a = 0.7f;  // grey defect deep inside
        base[y * w + x] = a;
        const float pic = float(std::min(1.0, std::max(0.0, (82.0 - d) / 1.5 + 0.5)));
        for (int ch = 0; ch < 3; ++ch) rgb[(y * w + x) * 3 + ch] = 0.05f + 0.8f * pic * (ch == 1 ? 0.9f : 1.0f);
        if (x >= 220 && x < 260 && y >= 80 && y < 160) region[y * w + x] = (x < 225) ? 0.5f : 1.0f;
      }
  }
  FrameInputs inputs(bool withRegion = true) const {
    FrameInputs in;
    in.w = w;
    in.h = h;
    in.rgb = rgb.data();
    in.base = base.data();
    in.region = withRegion ? region.data() : nullptr;
    return in;
  }
};

static Recipe make(std::initializer_list<RecipeEntry> e) {
  Recipe r;
  r.id = "test";
  r.name = "Test";
  r.entries = e;
  return r;
}

int main() {
  Scene s;
  const size_t n = size_t(s.w) * s.h;
  SolidCoreStep core;
  SnapStep snap;
  Workspace ws;
  std::vector<float> out(n);
  PlanResult res;

  {  // 1. every step off: the input bit for bit, nothing computed, nothing listed
    Recipe r = make({{&core, false, {}}, {&snap, false, {}}});
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    CHECK(same(out, s.base));
    CHECK(ws.ev.computed() == 0);
    CHECK(res.finalVersion == 1);
    CHECK(planLine(r, res) == "Test v1.0: nothing ran");
  }
  {  // 2. independent steps == the solo cores chained by hand, bit for bit; 3. outside the region untouched
    Recipe r = make({{&core, true, {}}, {&snap, true, {}}});
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    std::vector<float> a(n), b(n), wgt(n);
    solidCore(s.base.data(), a.data(), s.w, s.h, core.p);
    for (size_t i = 0; i < n; ++i) wgt[i] = std::min(1.0f, std::max(0.0f, s.region[i]));
    snapToImage(a.data(), s.rgb.data(), b.data(), s.w, s.h, snap.p, nullptr, wgt.data());
    CHECK(same(out, b));
    CHECK(res.steps[0].ran && res.steps[1].ran && res.steps[0].changed > 0 && res.steps[1].changed > 0);
    long long diffOutside = 0, diffInside = 0;
    for (size_t i = 0; i < n; ++i) {
      diffOutside += (s.region[i] == 0.0f && out[i] != a[i]);
      diffInside += (s.region[i] > 0.0f && out[i] != a[i]);
    }
    CHECK(diffOutside == 0);
    CHECK(diffInside > 0);  // the region test is not vacuous
    CHECK(res.finalVersion == 3);
  }
  {  // 4. evidence: computed once, shared, recomputed only after a real change; inputs-only maps stay
    ProbeStep p1("probe1"), p2("probe2"), p3("probe3");
    NeedStep same1;  // copies: no change -> no new version
    same1.nm = "copy";
    struct CopyStep : NeedStep {
      void run(const StepContext& c, const float* in, float* out, StepReport* r) override {
        std::memcpy(out, in, sizeof(float) * c.in.w * c.in.h);
        r->changed = 5;  // a wrong count: must not bump the version
      }
    } copy;
    Recipe r = make({{&p1, true, {}}, {&copy, true, {}}, {&p2, true, {}}, {&core, true, {}}, {&p3, true, {}},
                     {&snap, true, {}}});
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    // sd: probe1 computes, probe2 hits (copy changed nothing), core changes, probe3 recomputes; regionWeight once.
    CHECK(ws.ev.computed() == 3);
    CHECK(ws.ev.hits() == 1);
    CHECK(res.finalVersion == 3);
    // 4b. the next render on the same workspace: nothing carried over (same counts, not hits)
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    CHECK(ws.ev.computed() == 3 && ws.ev.hits() == 1);
    CHECK(ws.ev.bytesPooled() > 0 || ws.ev.bytesHeld() > 0);
    // 4c. a throwing compute leaves no entry behind
    Evidence ev;
    bool threw = false;
    try {
      ev.get({"x", 0, 1}, [](std::vector<float>* v) {
        v->assign(4, 1.0f);
        throw std::runtime_error("x");
      });
    } catch (const std::exception&) {
      threw = true;
    }
    const std::vector<float>& v = ev.get({"x", 0, 1}, [](std::vector<float>* v) { v->assign(2, 3.0f); });
    CHECK(threw && v.size() == 2 && v[0] == 3.0f && ev.computed() == 1 && ev.hits() == 0);
    // 4d. beginRender: content gone, buffers kept
    ev.setPlane("p", s.base.data(), n);
    ev.beginRender();
    CHECK(ev.plane("p") == nullptr && ev.bytesHeld() == 0 && ev.bytesPooled() >= n * sizeof(float));
    const std::vector<float>& v2 = ev.get({"x", 0, 1}, [](std::vector<float>* v) { CHECK(v->empty()); v->assign(1, 0.0f); });
    CHECK(ev.computed() == 1 && v2.size() == 1);
  }
  {  // 5. a failing step passes its input through (no partial write) and the chain goes on
    ThrowStep th, th2;
    th2.unknown = true;
    Recipe r = make({{&th, true, {}}, {&core, true, {}}, {&th2, true, {}}});
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    std::vector<float> a(n);
    solidCore(s.base.data(), a.data(), s.w, s.h, core.p);
    CHECK(same(out, a));
    CHECK(res.steps[0].failed && !res.steps[0].ran && res.steps[0].note == "runner not available");
    CHECK(res.steps[0].changed == 0);
    CHECK(res.steps[2].failed && res.steps[2].note == "unknown error");
    CHECK(planLine(r, res) ==
          "Test v1.0: Solid Core | failed: Edge Model (runner not available), Edge Model (unknown error)");
  }
  {  // 6. frames needed = the union over ENABLED steps
    NeedStep a, b, c;
    a.n = StepNeeds{kNeedNeighbours, 1};
    b.n = StepNeeds{kNeedNeighbours | kNeedRegion, 3};
    c.n = StepNeeds{kNeedModel | kNeedPicture, 0};
    Recipe r = make({{&a, true, {}}, {&b, false, {}}, {&c, true, {}}});
    StepNeeds u = unionNeeds(r);
    CHECK(u.neighbourReach == 1);
    CHECK(u.flags == (kNeedNeighbours | kNeedModel | kNeedPicture));
    r.entries[1].enabled = true;
    u = unionNeeds(r);
    CHECK(u.neighbourReach == 3 && (u.flags & kNeedRegion));
    r.entries.assign(1, RecipeEntry{&a, false, {}});
    CHECK(unionNeeds(r).flags == 0 && unionNeeds(r).neighbourReach == 0);
  }
  {  // 7. skip reasons per need, and a condition; the rest runs; skipped steps leave the matte alone
    NeedStep pic, reg, det, mdl, nbr;
    pic.n.flags = kNeedPicture, pic.nm = "pic";
    reg.n.flags = kNeedRegion, reg.nm = "reg";
    det.n.flags = kNeedDetail, det.nm = "det";
    mdl.n.flags = kNeedModel, mdl.nm = "mdl";
    nbr.n = StepNeeds{kNeedNeighbours, 1}, nbr.nm = "nbr";
    Condition never{[](const StepContext& c, std::string* why) {
      *why = "grey " + std::to_string(int(c.stats.greyFraction > 0));
      return false;
    }};
    NeedStep cond;
    cond.nm = "cond";
    Recipe r = make({{&pic, true, {}}, {&reg, true, {}}, {&det, true, {}}, {&mdl, true, {}}, {&nbr, true, {}},
                     {&cond, true, never}});
    FrameInputs in = s.inputs(false);
    in.rgb = nullptr;
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(same(out, s.base));
    CHECK(res.steps[0].note == "no picture" && res.steps[1].note == "no Focus Region" && res.steps[2].note == "no Detail");
    CHECK(res.steps[3].note == "needs setup.sh" && res.steps[4].note == "no neighbours" && res.steps[5].note == "grey 1");
    CHECK(planLine(r, res) == "Test v1.0: nothing ran | skipped: pic (no picture), reg (no Focus Region), det (no "
                              "Detail), mdl (needs setup.sh), nbr (no neighbours), cond (grey 1)");
    // the needs met: each runs
    std::vector<float> emptyRegion(n, 0.0f), detail(n, 1.0f);
    ml::Neighbours nb;
    in = s.inputs();
    in.detail = detail.data();
    in.modelGateOpen = true;
    in.nb = &nb;
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(res.steps[4].note == "no neighbours");  // fetched, but none usable
    nb.offsets.push_back(1);
    nb.frames.resize(1);
    runRecipe(r, in, ws, out.data(), &res);
    for (int i = 0; i < 5; ++i) CHECK(res.steps[i].ran);
    CHECK(!res.steps[5].ran);
    in.region = emptyRegion.data();
    runRecipe(r, in, ws, out.data(), &res);
    CHECK(res.steps[1].note == "Focus Region empty");
  }
  {  // 8. report texts: summaries, the Rebuilds notice, the debug line
    NeedStep a, b;
    a.nm = "Tracer";
    a.k = StepKind::Rebuilds;
    struct SumStep : NeedStep {
      void run(const StepContext& c, const float* in, float* out, StepReport* r) override {
        NeedStep::run(c, in, out, r);
        r->summary = "3 pieces";
      }
    } sum;
    sum.nm = "Snap";
    b.nm = "Temporal";
    b.n = StepNeeds{kNeedNeighbours, 1};
    Recipe r = make({{&sum, true, {}}, {&a, true, {}}, {&b, true, {}}, {&core, false, {}}});
    r.name = "Hair Locks";
    r.id = "hairLocks";
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    CHECK(planLine(r, res) == "Hair Locks v1.0: Snap (3 pieces), Tracer | skipped: Temporal (no neighbours)");
    CHECK(res.rebuilt);
    CHECK(planNotice(res) == "Tracer created matte from the picture.");
    const std::string dbg = planDebugLine(r, res);
    CHECK(dbg.compare(0, 16, "hairLocks rev 1 ") == 0);
    CHECK(dbg.find(" Snap=ran ") != std::string::npos && dbg.find(" Temporal=skip ") != std::string::npos);
    CHECK(dbg.find("[no neighbours]") != std::string::npos && dbg.find(" solidCore=off ") != std::string::npos);
    std::printf("plan: %s\ndebug: %s\n", planLine(r, res).c_str(), dbg.c_str());
    r.entries[1].enabled = false;
    runRecipe(r, s.inputs(), ws, out.data(), &res);
    CHECK(!res.rebuilt && planNotice(res).empty());
  }
  {  // 9. memory at 4096x2304 (2.2): K = 1, Span 1 about 0.8 GB; K = 2 about 1.1 GB; half-res = 1/4
    const double GB = 1e9;
    FootprintInput f;
    f.w = 4096;
    f.h = 2304;
    f.cropPixels = 400LL * 400;
    f.cropPlanes = 2 * 2 + 3 * 4;  // 2 candidates + ridge/unsharp crop maps
    f.pictureNeighbours = 2;       // t +- 1 with picture (Temporal K = 1)
    f.matteOnlyNeighbours = 2;     // t +- 2 matte-only (Jump Repair Span 1 reach 2)
    f.flowNeighbours = 2;
    const Footprint a = estimateFootprint(f);
    const double plane = 4096.0 * 2304 * 4;
    CHECK(a.inputs == size_t(5 * plane) && a.work == size_t(3 * plane) && a.neighbours == size_t(10 * plane));
    CHECK(a.flow == size_t(3 * 1024 * 576 * 4 * 2));  // flow grid 1024x576 (scale 4 at <= 600 lines)
    std::printf("footprint 4096x2304 K=1: %.2f GB\n", a.total() / GB);
    CHECK(a.total() > 0.75 * GB && a.total() < 0.9 * GB);
    f.pictureNeighbours = 4;
    f.matteOnlyNeighbours = 0;
    f.flowNeighbours = 4;
    const Footprint b = estimateFootprint(f);
    std::printf("footprint 4096x2304 K=2: %.2f GB\n", b.total() / GB);
    CHECK(b.total() > 1.0 * GB && b.total() < 1.2 * GB);
    f.w /= 2;
    f.h /= 2;
    const Footprint c = estimateFootprint(f);
    CHECK(c.inputs * 4 == b.inputs && c.work * 4 == b.work && c.neighbours * 4 == b.neighbours);
    CHECK(c.flow == b.flow);  // the flow grid is capped at 600 lines: 1024x576 at both sizes
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
