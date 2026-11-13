// Matte Refine ML (plugin 2) system test, Jump Repair part: loads MatteRefineML.ofx (a dev bundle: not
// gated) in the Flame-like host of Matte Merge (ofx/merge/systest/ofx_host.*) and checks the params
// (ids, defaults, the View option "Jumps" appended last), the layout rules of ofx/CLAUDE.md ("Flame host
// behaviour": one group per column of at most 5 rows, choice 2, secret params count; groups in chain
// order), and renders the synthetic jump sequence (synth::makeJumpSequence) with no model step on:
// Jump Repair off = the input matte bit for bit and frame t fetched alone; on = the core run directly
// on the same mattes, neighbours t +- 2 fetched; frame 0 passes through with the footer; the View
// "Jumps" colours (a blob kept as motion red; with Motion Guards off it is repaired, green); Span 3
// widens the fetch to t +- 4; user edits do not recurse.
// Usage: ml_systest <MatteRefineML.ofx> <scratch HOME folder> (never the real HOME).
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "jump_repair.h"
#include "ofx_host.h"
#include "synth.h"

namespace {
int failures = 0;
#define CHECK(c)                                                \
  do {                                                          \
    if (!(c)) {                                                 \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      ++failures;                                               \
    }                                                           \
  } while (0)

using namespace systest;

int rowsOf(const Param& p) {
  if (p.type == kOfxParamTypeChoice) return 2;
  if (p.type == kOfxParamTypeGroup || p.type == kOfxParamTypePage) return 0;
  return 1;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: ml_systest <MatteRefineML.ofx> <scratch home>\n");
    return 2;
  }
  const std::string home = argv[2];
  mkdir(home.c_str(), 0755);
  setenv("HOME", home.c_str(), 1);
  unsetenv("MR_LOG_LEVEL");
  std::remove((home + "/.config/matte-refine/config.conf").c_str());

  Host host(argv[1]);
  if (!host.loaded()) {
    std::printf("FAIL load: %s\n", host.loadError().c_str());
    return 1;
  }
  CHECK(std::strcmp(host.plugin()->pluginIdentifier, "com.janklier.matterefineml") == 0);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();

  // 1. Params: ids, defaults, the View option appended last (indexes stored in setups).
  std::map<std::string, const Param*> by;
  for (const auto& p : d.params.list) by[p->name] = p.get();
  for (const char* id : {"jrEnable", "jrThreshold", "jrMinSize", "jrSpanFrames", "jrGuards", "grpJumpRepair", "view"})
    CHECK(by.count(id));
  CHECK(!by.count("jrSpan"));  // the 2-row choice it replaced (never released) is not reused
  if (by.count("jrSpanFrames") && by.count("jrGuards") && by.count("view") && by.count("jrEnable")) {
    CHECK(by["jrEnable"]->props.getInt(kOfxParamPropDefault) == 0);
    CHECK(by["jrThreshold"]->props.getDbl(kOfxParamPropDefault) == 0.5);
    CHECK(by["jrMinSize"]->props.getDbl(kOfxParamPropDefault) == 0.0125);
    CHECK(by["jrSpanFrames"]->type == kOfxParamTypeDouble && by["jrSpanFrames"]->props.getDbl(kOfxParamPropDefault) == 1.0);
    CHECK(by["jrGuards"]->type == kOfxParamTypeBoolean && by["jrGuards"]->props.getInt(kOfxParamPropDefault) == 1);
    CHECK(by["jrGuards"]->props.getStr(kOfxParamPropParent) == "grpJumpRepair");
    const PropSet& vp = by["view"]->props;
    CHECK(vp.props.at(kOfxParamPropChoiceOption).size() == 6 && vp.getStr(kOfxParamPropChoiceOption, 4) == "Holes" &&
          vp.getStr(kOfxParamPropChoiceOption, 5) == "Jumps");
  }
  // 2. Layout: groups in definition order = chain order, each param right after its group's earlier
  // params (one column each), at most 5 rows per column, at most 6 columns on the tab.
  {
    std::vector<std::string> groups;
    std::map<std::string, int> rows;
    std::string current;
    bool contiguous = true, orphan = false;
    for (const auto& p : d.params.list) {
      if (p->type == kOfxParamTypeGroup) {
        groups.push_back(p->name);
        current = p->name;
        continue;
      }
      if (p->type == kOfxParamTypePage) continue;
      const std::string parent = p->props.getStr(kOfxParamPropParent);
      if (parent.empty()) orphan = true;
      else if (parent != current) contiguous = false;
      rows[parent] += rowsOf(*p);
    }
    const std::vector<std::string> want = {"grpMatte", "grpJumpRepair", "grpTemporal", "grpSmartFill", "grpEdgeRefine",
                                           "grpSupport"};
    CHECK(groups == want && contiguous && !orphan);
    std::printf("layout:");
    for (const std::string& g : groups) {
      std::printf(" %s %d", g.c_str(), rows[g]);
      CHECK(rows[g] <= 5);
    }
    std::printf(" rows (%zu columns)\n", groups.size());
    CHECK(groups.size() <= 6);
  }

  // 3. Render the synthetic jump sequence (480x270, 20 frames): picture grey, matte = alpha.
  const int W = 480, H = 270;
  const synth::JumpSequence s = synth::makeJumpSequence(W, H);
  std::map<int, int> fetched;  // frame -> fetches since the last clear
  Feeds feeds;
  feeds[kOfxImageEffectSimpleSourceClipName] = Feed{[&](int f, int w, int h, std::vector<float>& o) {
                                                      ++fetched[f];
                                                      for (size_t i = 0; i < size_t(w) * h; ++i) {
                                                        o[i * 4] = o[i * 4 + 1] = o[i * 4 + 2] = 0.5f;
                                                        o[i * 4 + 3] = s.matte[size_t(f)][i];
                                                      }
                                                    },
                                                    4};
  auto inst = host.createInstance(W, H, s.frames, feeds);
  inst->setBool("erEnable", false);  // no model step: no runner, no model needed
  inst->setInt("view", 1);           // Matte
  auto alphaOf = [&](const Rendered& r) {
    std::vector<float> a(size_t(W) * H);
    for (int row = 0; row < H; ++row)
      for (int x = 0; x < W; ++x) a[size_t(row) * W + x] = r.px(x, row)[3];
    return a;
  };
  auto byHand = [&](int t, int span, bool guards = true) {
    const float* fr[2 * mr::ml::kJumpReach + 1];
    const int reach = mr::ml::jumpReach(span);
    for (int k = -mr::ml::kJumpReach; k <= mr::ml::kJumpReach; ++k)
      fr[k + mr::ml::kJumpReach] =
          std::abs(k) <= reach && t + k >= 0 && t + k < s.frames ? s.matte[size_t(t + k)].data() : nullptr;
    mr::ml::JumpParams p;
    p.maxSpan = span;
    p.motionGuard = guards;
    std::vector<float> out(size_t(W) * H);
    mr::ml::JumpStats st;
    mr::ml::repairJumps(fr, W, H, p, out.data(), &st);
    return out;
  };
  const int t = 3;  // jump (0): an added disc
  // Off: the input bit for bit, only frame t fetched (no neighbour cost: the Temporal-off rule).
  fetched.clear();
  Rendered r = inst->render(t);
  CHECK(r.status == kOfxStatOK && alphaOf(r) == s.matte[size_t(t)]);
  CHECK(fetched.size() == 1 && fetched.count(t));
  std::printf("off: status %d, fetched %zu frame(s), footer '%s'\n", int(r.status), fetched.size(),
              inst->persistent.c_str());
  // On: = the core directly, neighbours t +- 2 fetched once each.
  inst->setBool("jrEnable", true);
  fetched.clear();
  r = inst->render(t);
  const std::vector<float> want = byHand(t, 1);
  CHECK(r.status == kOfxStatOK && alphaOf(r) == want && want != s.matte[size_t(t)]);
  CHECK(fetched.size() == 5 && fetched[t - 2] == 1 && fetched[t - 1] == 1 && fetched[t + 1] == 1 && fetched[t + 2] == 1);
  CHECK(inst->persistent.empty());
  long changed = 0;
  for (size_t i = 0; i < want.size(); ++i) changed += want[i] != s.matte[size_t(t)][i];
  std::printf("on: status %d, fetched %zu frames, changed %ld px, footer '%s'\n", int(r.status), fetched.size(), changed,
              inst->persistent.c_str());
  // A frame without a jump: unchanged (the moving figure, fast disc and bar are left alone).
  r = inst->render(9);
  CHECK(r.status == kOfxStatOK && alphaOf(r) == s.matte[9]);
  // Frame 0: not judged, the footer says so.
  r = inst->render(0);
  CHECK(r.status == kOfxStatOK && alphaOf(r) == s.matte[0]);
  CHECK(inst->persistent == "Jump Repair: this frame needs both neighbour frames (t-1 and t+1); it passes through.");
  std::printf("frame 0 footer: '%s'\n", inst->persistent.c_str());
  // View Jumps: the jump's centre (lowered 1 -> 0) green, a pixel far from it grey.
  inst->setInt("view", 5);
  r = inst->render(t);
  {
    const synth::JumpDefect& j = s.defects[0];
    const float* c = r.px(int(j.cx), int(j.cy));
    CHECK(c[0] == 0.0f && c[1] == 1.0f && c[2] == 0.0f && c[3] == 0.0f);
    const float* g = r.px(5, 5);
    const float m = s.matte[size_t(t)][size_t(5) * W + 5];
    CHECK(g[0] == m && g[1] == m && g[2] == m);
  }
  // Motion Guards: frame 9 has the fast disc (passing through, kept as motion): red in View Jumps and
  // the alpha unchanged with the guards on; off, the disc is repaired (green, alpha lowered, = the core
  // with motionGuard false).
  {
    const int t9 = 9, dx = int(W / 2.0 + (-900 + 80.0 * t9) * H / 1080.0), dy = int(H / 2.0 - 480.0 * H / 1080.0);
    r = inst->render(t9);
    CHECK(alphaOf(r) == s.matte[t9]);
    const float* c = r.px(dx, dy);
    CHECK(c[0] == 1.0f && c[1] == 0.0f && c[2] == 0.0f && c[3] == 1.0f);
    inst->setBool("jrGuards", false);
    r = inst->render(t9);
    CHECK(alphaOf(r) == byHand(t9, 1, false) && alphaOf(r) != s.matte[t9]);
    c = r.px(dx, dy);
    CHECK(c[0] == 0.0f && c[1] == 1.0f && c[2] == 0.0f && c[3] == 0.0f);
    // Guards off at frame 3: the injected jump still repaired as with the guards on.
    inst->setInt("view", 1);
    r = inst->render(t);
    CHECK(alphaOf(r) == byHand(t, 1, false));
    inst->setBool("jrGuards", true);
    r = inst->render(t9);
    CHECK(alphaOf(r) == s.matte[t9]);
    std::printf("motion guards: disc at %d,%d kept (red) when on, repaired (green) when off\n", dx, dy);
  }
  // Span 3 frames: neighbours t +- 4 (the guards look one frame beyond the references), = the core.
  inst->setInt("view", 1);
  inst->setDouble("jrSpanFrames", 3.0);
  fetched.clear();
  const int t2 = 15;  // jump (4): removed, frames 14-16
  r = inst->render(t2);
  CHECK(r.status == kOfxStatOK && alphaOf(r) == byHand(t2, 3) && alphaOf(r) != s.matte[size_t(t2)]);
  CHECK(fetched.size() == 9);
  std::printf("span 3: fetched %zu frames\n", fetched.size());
  // No recursion from user edits; no host violations.
  CHECK(inst->maxEchoDepth <= 1 && host.errors.empty());
  for (const std::string& e : host.errors) std::printf("host error: %s\n", e.c_str());
  inst.reset();
  if (failures) std::printf("%d FAILED\n", failures);
  else std::printf("ml_systest: all passed\n");
  return failures ? 1 : 0;
}
