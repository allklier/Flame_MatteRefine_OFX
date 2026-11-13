// Edge Recipes system test: the bundle in the Flame-like host (ofx/merge/systest/ofx_host.*), one host per plugin
// index. Each effect must describe and render as ITSELF (the factory-per-recipe rule, docs/edge_recipes.md 4.1):
// its label, grouping, sockets, and a descriptor whose params are exactly its RecipeDef's (same ids, same labels,
// laid out by Flame's rules: tab 1 = 6 columns, then "Advanced"). Then Hair Locks with its steps (WP7), driven
// through the Source, Detail and FocusRegion clips on synthetic scenes only (ofx/tools/synth, the tracer scenes):
//   nothing enabled -> the Source alpha bit for bit (odd values included); Rebuild Detail -> base + strand alpha,
//   never below the base, bit-identical outside the region (linear picture on Bypass, display-encoded on Auto);
//   the model step through the fake mr_ml_runner process (installed / not installed / failing runner); Temporal
//   off = bit-identical, on = outside the region only; Jump Repair = the core run directly (bit for bit);
//   getFramesNeeded = the union of the enabled steps' reach; the Detail clip as a candidate; every View; the Plan
//   strings (<= 28 characters) and footers; Collect Logs; no setValue echo recursion.
// Usage: recipes_systest <EdgeRecipes.ofx> <EdgeRecipesTest.ofx> <scratch HOME> <test_fake_runner>
//        recipes_systest --mutant <EdgeRecipesMutant.ofx> <scratch HOME>   (must FAIL: ctest WILL_FAIL)
//        recipes_systest --timing <EdgeRecipes.ofx> <scratch HOME> <test_fake_runner>   (not a test: per-step ms)
#include <signal.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "hair_locks_ui.h"
#include "jump_repair.h"
#include "merge_fusion.h"
#include "ofxParam.h"
#include "ofx_host.h"
#include "recipe_def.h"
#include "flow.h"
#include "render_chain.h"
#include "synth.h"
#include "tracer_scenes.h"

namespace {
int failures = 0;
#define CHECK(c)                                               \
  do {                                                         \
    if (!(c)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                              \
    }                                                          \
  } while (0)

using namespace systest;
using namespace mr::recipes;

namespace scenes = mr::tracer::scenes;

constexpr int W = 40, H = 30;
std::string gHome, gXdg;  // gXdg: a short folder for the runner socket (a Unix socket path is <= 107 bytes)

// Alpha with values a clamp or a copy through 0-1 would change: negative, above 1, NaN, a denormal.
float alphaAt(int x, int y) {
  const int k = (x + 3 * y) % 7;
  return k == 0 ? -0.25f : k == 1 ? 1.5f : k == 2 ? std::numeric_limits<float>::quiet_NaN()
                         : k == 3 ? std::numeric_limits<float>::denorm_min() : float(x * y) / (W * H);
}
Feed sourceFeed(int comps) {
  return Feed{[comps](int, int w, int h, std::vector<float>& o) {
                for (int y = 0; y < h; ++y)
                  for (int x = 0; x < w; ++x) {
                    float* p = &o[(size_t(y) * w + x) * comps];
                    p[0] = float(x) / w;
                    p[1] = float(y) / h;
                    p[2] = 0.3f + 0.01f * float(x % 5);
                    if (comps == 4) p[3] = alphaAt(x, y);
                  }
              },
              comps};
}
Feed constFeed(float v) {
  return Feed{[v](int, int w, int h, std::vector<float>& o) { std::fill(o.begin(), o.begin() + size_t(w) * h, v); }, 1};
}
// Per frame: picture (w*h*3) and matte (w*h), top-down, the same size as the host's frames.
Feed rgbaFeed(std::function<const float*(int)> rgb, std::function<const float*(int)> matte) {
  return Feed{[rgb, matte](int f, int w, int h, std::vector<float>& o) {
                const float* c = rgb(f);
                const float* m = matte(f);
                for (size_t i = 0; i < size_t(w) * h; ++i) {
                  o[i * 4] = c[i * 3];
                  o[i * 4 + 1] = c[i * 3 + 1];
                  o[i * 4 + 2] = c[i * 3 + 2];
                  o[i * 4 + 3] = m[i];
                }
              },
              4};
}
Feed planeFeed(std::function<const float*(int)> plane) {
  return Feed{[plane](int f, int w, int h, std::vector<float>& o) {
                const float* p = plane(f);
                std::copy(p, p + size_t(w) * h, o.begin());
              },
              1};
}

bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

void writeConfig(const std::map<std::string, std::string>& kv) {
  const std::string dir = gHome + "/.config/matte-refine";
  if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) std::printf("mkdir %s failed\n", dir.c_str());
  std::ofstream f(dir + "/config.conf");
  for (const auto& e : kv) f << e.first << "=" << e.second << "\n";
}

// Every Plan label string fits Flame's ~30 characters (kPlanMaxChars).
void checkPlanFits(Instance& inst) {
  for (const char* id : {kPPlan, kPPlanRebuild, kPPlanGate, kPPlanInputs, kPPlanNote}) {
    const std::string s = inst.getString(id);
    if (int(s.size()) > kPlanMaxChars) std::printf("FAIL plan %s too long: '%s'\n", id, s.c_str()), ++failures;
  }
}
void printPlan(const char* what, Instance& inst) {
  std::printf("%s: '%s' | '%s' | '%s' | '%s' | '%s'\n", what, inst.getString(kPPlan).c_str(),
              inst.getString(kPPlanRebuild).c_str(), inst.getString(kPPlanGate).c_str(),
              inst.getString(kPPlanInputs).c_str(), inst.getString(kPPlanNote).c_str());
  checkPlanFits(inst);
}

// Hair Locks with every step off (the node's defaults have Edge Model, Snap and Temporal on).
void allOff(Instance& inst) {
  for (const char* id : {"emEnable", "rbEnable", "rgSnap", "tmEnable", "jrEnable"}) inst.setBool(id, false);
}

std::vector<float> alphaOf(const Rendered& r) {
  std::vector<float> a(size_t(r.w) * r.h);
  for (size_t i = 0; i < a.size(); ++i) a[i] = r.rgba[i * 4 + 3];
  return a;
}
bool sameAlpha(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The descriptor against the RecipeDef: params in definition order with the def's labels, groups as
// columns, each column on the def's page; Flame's tab rule (6 columns on tab 1, the rest on the next page).
void checkDescriptor(Host& host, const RecipeDef& def) {
  std::vector<const ParamDef*> want;
  std::vector<std::string> wantGroups, wantPages;
  for (const PageDef& pg : def.pages)
    for (const ColumnDef& c : pg.columns) {
      wantGroups.push_back(c.group);
      wantPages.push_back(pg.name);
      for (const ParamDef& p : c.params) want.push_back(&p);
    }
  std::vector<std::string> groups, pages;
  std::vector<int> rows;
  std::string page;
  size_t i = 0;
  for (const auto& p : host.descriptor().params.list) {
    if (p->type == kOfxParamTypePage) {
      page = p->name;
      continue;
    }
    if (p->type == kOfxParamTypeGroup) {
      groups.push_back(p->name);
      pages.push_back(page);
      rows.push_back(0);
      continue;
    }
    CHECK(!rows.empty());
    if (rows.empty()) return;
    CHECK(p->props.getStr(kOfxParamPropParent) == groups.back());
    rows.back() += p->type == kOfxParamTypeChoice ? 2 : 1;
    if (i >= want.size() || p->name != want[i]->id) {
      std::printf("FAIL param %zu is %s, the def has %s\n", i, p->name.c_str(), i < want.size() ? want[i]->id : "-");
      ++failures;
      return;
    }
    const ParamDef& d = *want[i++];
    const bool hidden = d.kind == Kind::HiddenString || d.kind == Kind::HiddenInt;
    CHECK(p->props.getInt(kOfxParamPropSecret) == (hidden ? 1 : 0));
    if (!hidden && p->props.getStr(kOfxPropLabel) != d.label) {
      std::printf("FAIL %s label '%s', the def has '%s'\n", d.id, p->props.getStr(kOfxPropLabel).c_str(), d.label);
      ++failures;
    }
    if (p->type == kOfxParamTypeDouble) CHECK(p->props.getInt(kOfxParamPropDigits) == d.digits);
    if (p->type == kOfxParamTypeChoice)
      CHECK(p->props.props.count(kOfxParamPropChoiceOption) &&
            p->props.props.at(kOfxParamPropChoiceOption).size() ==
                int(((d.flags & kViewOptions) ? def.views : d.options).size()));
  }
  CHECK(i == want.size());
  CHECK(groups == wantGroups && pages == wantPages);
  for (size_t c = 0; c < rows.size(); ++c) {
    CHECK(rows[c] <= kMaxRows);
    if (c < size_t(kMaxColumns)) CHECK(pages[c] == def.label);  // tab 1
    else CHECK(pages[c] == "Advanced");                         // overflow tab, named after the page
  }
}

// One effect of a binary: describe, layout, pass-through render, Plan, footer, colour. Returns false if it did
// not load.
bool runOne(const std::string& path, int index, int count, const RecipeDef& def) {
  std::printf("-- %s plugin %d (%s)\n", path.c_str(), index, def.label);
  Host host(path, index);
  if (!host.loaded()) {
    std::printf("FAIL load %d: %s\n", index, host.loadError().c_str());
    ++failures;
    return false;
  }
  CHECK(host.pluginCount() == count);
  CHECK(pluginId(def) == host.plugin()->pluginIdentifier);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();
  CHECK(d.props.getStr(kOfxPropLabel) == def.label);
  CHECK(d.props.getStr(kOfxImageEffectPluginPropGrouping) == "Edge Recipes");
  CHECK(d.clips.size() == 2u + (def.detailSocket ? 1 : 0) + (def.regionSocket ? 1 : 0));
  for (const char* m : {"Detail", "FocusRegion"}) {
    const bool want = std::string(m) == "Detail" ? def.detailSocket : def.regionSocket;
    CHECK(d.clips.count(m) == (want ? 1u : 0u));
    if (!want || !d.clips.count(m)) continue;
    CHECK(d.clips[m]->props.getInt(kOfxImageClipPropIsMask) == 1);
    CHECK(d.clips[m]->props.getInt(kOfxImageClipPropOptional) == 1);
  }
  const int before = failures;
  checkDescriptor(host, def);
  if (failures != before) return true;  // described as another effect: the instance checks would not apply

  const bool hair = findParam(def, "rbEnable") != nullptr;
  auto inst = host.createInstance(W, H, 10, Feeds{{kOfxImageEffectSimpleSourceClipName, sourceFeed(4)}});
  CHECK(inst->getString(kPNodeId).size() == 8);
  CHECK(inst->getInt(kPRecipeRev) == def.recipeRev);
  const std::string version = inst->getString(kPVersion);
  CHECK(version.find(std::string(def.label) + " " + std::to_string(def.versionMajor) + "." +
                     std::to_string(def.versionMinor)) != std::string::npos);
  if (hair) allOff(*inst);

  // View Result Matte (default): alpha bit for bit, RGB = the matte.
  CHECK(inst->getInt(kPView) == kViewResult);
  Rendered r = inst->render(3);
  CHECK(r.status == kOfxStatOK);
  long long badA = 0, badRgb = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float* p = r.px(x, y);
      badA += !sameBits(p[3], alphaAt(x, y));
      for (int c = 0; c < 3; ++c) badRgb += !sameBits(p[c], alphaAt(x, y));
    }
  CHECK(badA == 0 && badRgb == 0);
  CHECK(inst->persistent.empty());
  // View Picture: RGB = the Source bit for bit, alpha unchanged.
  inst->setInt(kPView, kViewPicture);
  r = inst->render(4);
  std::vector<float> src(size_t(W) * H * 4);
  sourceFeed(4).fn(4, W, H, src);
  badA = badRgb = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float* p = r.px(x, y);
      const float* s = &src[(size_t(y) * W + x) * 4];
      badA += !sameBits(p[3], s[3]);
      for (int c = 0; c < 3; ++c) badRgb += !sameBits(p[c], s[c]);
    }
  CHECK(badA == 0 && badRgb == 0);

  printPlan("plan", *inst);
  if (hair) {
    CHECK(inst->getString(kPPlan) == "Nothing ran: pass-through");
    CHECK(inst->getString(kPPlanRebuild) == "Rebuild Detail off");
    CHECK(inst->getString(kPPlanGate) == "Model: off");
    CHECK(inst->getString(kPPlanInputs) == "Region off, Detail off");
    CHECK(inst->getString(kPPlanNote) == "OK");
    // Every view with nothing enabled: the alpha stays the Source's bit for bit.
    for (int v = 2; v <= kViewTrimap; ++v) {
      inst->setInt(kPView, v);
      r = inst->render(5);
      long long bad = 0;
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) bad += !sameBits(r.px(x, y)[3], alphaAt(x, y));
      CHECK(bad == 0);
    }
    // The defaults (Edge Model, Snap, Temporal on), no Focus Region: Temporal runs on the whole matte.
    for (const char* id : {"emEnable", "rgSnap", "tmEnable"}) inst->setBool(id, true);
    inst->setInt(kPView, kViewResult);
    inst->render(5);
    printPlan("defaults", *inst);
    CHECK(inst->getString(kPPlan) == "Temporal" && inst->getString(kPPlanNote) == "Connect a Focus Region");
    CHECK(inst->getString(kPPlanGate) == "Model Small: not installed");  // a scratch HOME: no models (dev build)
    inst->setBool("rbEnable", true);
    inst->setFeed("FocusRegion", constFeed(0.5f));
    inst->render(5);
    printPlan("rebuild", *inst);
    CHECK(inst->getString(kPPlanInputs) == "Region 1200 px, Detail off");
    CHECK(inst->getString(kPPlanRebuild).find("Vote: kept") == 0);  // Temporal Vote is on by default
  } else {
    CHECK(inst->getString(kPPlan) == std::string(def.label) + " " + std::to_string(def.versionMajor) + "." +
                                         std::to_string(def.versionMinor) + ": pass-through");
    CHECK(inst->getString(kPPlanNote) == "Skeleton: no steps yet");
    CHECK(inst->getString(kPPlanRebuild).empty() && inst->getString(kPPlanGate).empty());
  }

  // Colour answers (decisions 23, 34): View Picture = the requested picture colourspace, or Source on Bypass.
  inst->setInt(kPView, kViewPicture);
  CHECK(inst->outputColourspace() == "Rec.1886 Rec.709 - Display");
  inst->setInt(kPPicColour, kColourBypass);
  CHECK(inst->outputColourspace() == "OfxColourspace_Source");

  // Collect Logs writes a support bundle into HOME.
  const size_t nMessages = inst->messages.size();
  inst->press(kPCollectLogs);
  CHECK(inst->messages.size() == nMessages + 1 &&
        inst->messages.back().text.find("Support bundle written") != std::string::npos);

  // An RGB-only picture: no matte, a footer (never a '%': the host formats the text); every view renders.
  inst->setFeed(kOfxImageEffectSimpleSourceClipName, sourceFeed(3));
  for (int v = 0; v < int(def.views.size()); ++v) {
    inst->setInt(kPView, v);
    CHECK(inst->render(7).status == kOfxStatOK);
  }
  CHECK(inst->persistent.find("no alpha") != std::string::npos);
  CHECK(inst->persistent.find('%') == std::string::npos);
  if (hair)
    CHECK(inst->getString(kPPlanNote) == "No alpha: use source alpha" &&
          inst->getString(kPPlan) == "Nothing ran: pass-through");
  checkPlanFits(*inst);
  // A constant alpha: the footer as plugin 2's.
  if (hair) {
    inst->setFeed(kOfxImageEffectSimpleSourceClipName,
                  Feed{[](int, int w, int h, std::vector<float>& o) {
                         for (size_t i = 0; i < size_t(w) * h; ++i) o[i * 4] = o[i * 4 + 1] = o[i * 4 + 2] = o[i * 4 + 3] = 0.5f;
                       },
                       4});
    inst->render(7);
    CHECK(inst->persistent.find("constant") != std::string::npos && inst->getString(kPPlanNote) == "Source alpha is constant");
  }
  CHECK(inst->maxEchoDepth <= 2);
  for (const auto& err : host.errors) std::printf("FAIL host: %s\n", err.c_str());
  failures += int(host.errors.size());
  return true;
}

// ---- Hair Locks scenarios -------------------------------------------------------------------------------

// The prepared region weight as the plugin computes it (Merge's prepareRegion, Feather % of the height).
std::vector<float> weightOf(const std::vector<float>& mask, int w, int h, double featherPct) {
  std::vector<float> wt(mask.size());
  mr::merge::prepareRegion(mask.data(), w, h, 0.0, featherPct / 100.0 * h, wt.data());
  return wt;
}

// A tracer scene ("single": a 3 px strand leaving a solid head across a soft region); linear picture.
struct TracerScene {
  scenes::Scene s;
  std::vector<float> display;  // the picture display-encoded (x^(1/2.4)): what Picture Colour Auto delivers
  std::vector<float> gtf;      // truth strand alpha as float (an external Detail matte)
  explicit TracerScene(const char* name = "single") {
    s = scenes::makeScene(name, 1);
    display.resize(s.rgb.size());
    for (size_t i = 0; i < s.rgb.size(); ++i) display[i] = s.rgb[i] > 0 ? std::pow(s.rgb[i], 1.0f / 2.4f) : 0.0f;
    gtf.assign(s.gt.begin(), s.gt.end());
  }
  Feeds feeds(bool displayEncoded) const {
    const float* rgb = displayEncoded ? display.data() : s.rgb.data();
    const float* a = s.alpha.data();
    const float* reg = s.region.data();
    return Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed([rgb](int) { return rgb; }, [a](int) { return a; })},
                 {"FocusRegion", planeFeed([reg](int) { return reg; })}};
  }
};

// out against the base: never below it, the base bit for bit where the region mask is 0. Returns the summed
// rise on the truth strand pixels.
double checkRegionContract(const std::vector<float>& out, const scenes::Scene& s, const char* what) {
  long long below = 0, outside = 0;
  double rise = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    below += out[i] < s.alpha[i];
    outside += s.region[i] == 0.0f && !sameBits(out[i], s.alpha[i]);
    if (s.gt[i] > 0.5) rise += out[i] - s.alpha[i];
  }
  std::printf("%s: below %lld, changed outside %lld, rise on strands %.1f\n", what, below, outside, rise);
  CHECK(below == 0 && outside == 0);
  return rise;
}

void rebuildScenario(Host& host) {
  std::printf("-- Hair Locks: Rebuild Detail (tracer scene)\n");
  TracerScene ts;
  const scenes::Scene& s = ts.s;
  auto inst = host.createInstance(s.w, s.h, 5, ts.feeds(false));
  allOff(*inst);
  inst->setBool("rbVote", false);  // the tracer alone here; the vote: voteScenario
  inst->setInt(kPPicColour, kColourBypass);  // the scene is linear: the tracer reads it as delivered
  Rendered r = inst->render(2);
  CHECK(sameAlpha(alphaOf(r), s.alpha));  // nothing on, a region connected: the base bit for bit
  CHECK(inst->getString(kPPlan) == "Nothing ran: pass-through");
  inst->setBool("rbEnable", true);
  r = inst->render(2);
  printPlan("rebuild", *inst);
  const std::vector<float> rebuilt = alphaOf(r);
  CHECK(checkRegionContract(rebuilt, s, "rebuild (Bypass)") > 20.0);
  CHECK(inst->getString(kPPlan) == "Rebuild");
  CHECK(inst->getString(kPPlanRebuild).find("Created matte: ") == 0 &&
        inst->getString(kPPlanRebuild) != "Created matte: 0 pieces");
  // Views: Changes, Region, Candidates (tracer green), Rebuilt (green = what was added).
  const std::vector<float> wt = weightOf(s.region, s.w, s.h, 0.25);
  long long badChanges = 0, badRegion = 0, badCand = 0, badRebuilt = 0, green = 0, addedPx = 0;
  inst->setInt(kPView, kViewChanges);
  r = inst->render(2);
  for (size_t i = 0; i < rebuilt.size(); ++i) {
    float c[3];
    mr::ml::changesViewRgb(s.alpha[i], rebuilt[i], c);
    badChanges += !(sameBits(c[0], r.rgba[i * 4]) && sameBits(c[1], r.rgba[i * 4 + 1]) && sameBits(c[2], r.rgba[i * 4 + 2]));
  }
  inst->setInt(kPView, kViewRegion);
  r = inst->render(2);
  for (size_t i = 0; i < wt.size(); ++i) badRegion += !sameBits(r.rgba[i * 4], wt[i]);
  inst->setInt(kPView, kViewCandidates);
  r = inst->render(2);
  for (size_t i = 0; i < wt.size(); ++i) {
    const float* p = &r.rgba[i * 4];
    badCand += p[0] != 0.5f * s.alpha[i] || p[2] != 0.5f * s.alpha[i];  // no model, no Detail
    green += p[1] > 0.5f * s.alpha[i];
  }
  CHECK(green > 20);
  inst->setInt(kPView, kViewRebuilt);
  r = inst->render(2);
  for (size_t i = 0; i < wt.size(); ++i) {
    const float* p = &r.rgba[i * 4];
    const float added = std::max(0.0f, rebuilt[i] - s.alpha[i]);  // the tracer is the only candidate here
    float c[3];
    rebuiltViewRgb(rebuilt[i], added, 0.0f, c);  // the vote is off: nothing dropped
    badRebuilt += !(sameBits(p[0], c[0]) && sameBits(p[1], c[1]) && sameBits(p[2], c[2]));
    addedPx += added > 0.0f;
  }
  std::printf("views: changes %lld, region %lld, candidates %lld (green %lld), rebuilt %lld (added %lld) bad px\n",
              badChanges, badRegion, badCand, green, badRebuilt, addedPx);
  CHECK(badChanges == 0 && badRegion == 0 && badCand == 0 && badRebuilt == 0 && addedPx > 20);
  // Loose Pieces off keeps at most what was kept with it on.
  inst->setInt(kPView, kViewResult);
  inst->setBool("rbKeepLoose", false);
  const std::vector<float> anchoredOnly = alphaOf(inst->render(2));
  long long more = 0;
  for (size_t i = 0; i < rebuilt.size(); ++i) more += anchoredOnly[i] > rebuilt[i];
  CHECK(more == 0);
  checkRegionContract(anchoredOnly, s, "rebuild, Loose Pieces off");
  inst->setBool("rbKeepLoose", true);
  // Snap without a candidate (Rebuild off): a no-op, the same as Snap off (decision 39: new pixels only).
  inst->setBool("rbEnable", false);
  inst->setBool("rgSnap", true);
  CHECK(sameAlpha(alphaOf(inst->render(2)), s.alpha) && inst->getString(kPPlan) == "Nothing ran: pass-through");
  inst->setBool("rgSnap", false);
  inst->setBool("rbEnable", true);
  // Temporal Vote on, the 5 frames identical: the neighbours confirm the piece; the contract holds.
  inst->setBool("rbVote", true);
  CHECK(checkRegionContract(alphaOf(inst->render(2)), s, "rebuild + vote (static frames)") > 20.0);
  CHECK(inst->getString(kPPlan) == "Rebuild+Vote" && inst->getString(kPPlanRebuild).find("Vote: kept 1") == 0);
  inst->setBool("rbVote", false);
  // Snap on top: still never below, still bit-identical outside.
  inst->setBool("rgSnap", true);
  checkRegionContract(alphaOf(inst->render(2)), s, "rebuild + Snap");
  CHECK(inst->getString(kPPlan) == "Rebuild+Snap");
  inst->setBool("rgSnap", false);
  // Picture Colour Auto: the picture arrives display-encoded; the tracer decodes it and finds the strand.
  auto autoInst = host.createInstance(s.w, s.h, 5, ts.feeds(true));
  allOff(*autoInst);
  autoInst->setBool("rbVote", false);
  autoInst->setBool("rbEnable", true);
  CHECK(checkRegionContract(alphaOf(autoInst->render(2)), s, "rebuild (Auto, display-encoded)") > 20.0);
  // An empty Focus Region: nothing changes, the footer and the Note say so.
  inst->setFeed("FocusRegion", constFeed(0.0f));
  CHECK(sameAlpha(alphaOf(inst->render(2)), s.alpha));
  printPlan("empty region", *inst);
  CHECK(inst->getString(kPPlanNote) == "Focus Region is empty" &&
        inst->getString(kPPlanInputs) == "Region empty, Detail off");
  CHECK(inst->persistent.find("black everywhere") != std::string::npos);
  CHECK(inst->getString(kPPlanRebuild) == "Rebuild: Focus Region empty");
}

void detailScenario(Host& host) {
  std::printf("-- Hair Locks: Detail clip as a candidate\n");
  TracerScene ts;
  const scenes::Scene& s = ts.s;
  Feeds f = ts.feeds(false);
  const float* gt = ts.gtf.data();
  f["Detail"] = planeFeed([gt](int) { return gt; });
  auto inst = host.createInstance(s.w, s.h, 5, f);
  allOff(*inst);
  inst->setInt(kPPicColour, kColourBypass);
  const std::vector<float> out = alphaOf(inst->render(2));
  printPlan("detail", *inst);
  CHECK(checkRegionContract(out, s, "detail") > 20.0);
  CHECK(inst->getString(kPPlan) == "Detail" && inst->getString(kPPlanInputs).find("Detail on") != std::string::npos);
  inst->setInt(kPView, kViewCandidates);
  Rendered r = inst->render(2);
  long long blue = 0;
  for (size_t i = 0; i < out.size(); ++i) blue += r.rgba[i * 4 + 2] > 0.5f * s.alpha[i] && s.gt[i] > 0.5;
  CHECK(blue > 20);
  // Disconnected: the base passes through.
  inst->setFeed("Detail", Feed{});
  inst->setInt(kPView, kViewResult);
  CHECK(sameAlpha(alphaOf(inst->render(2)), s.alpha));
}

void killFakeRunners() {
  std::ifstream f(gXdg + "/matte-refine-ml/fake.starts");
  int pid = 0;
  while (f >> pid)
    if (pid > 1) kill(pid, SIGKILL);
}

// The fake runner answers with the build id in fake.build (ofx/ml/tests/fake_runner.cpp): Edge Recipes' own id
// (ER_RUNNER_BUILD_ID, docs/edge_recipes.md 1.3), else the handshake mismatches. The folder must be private (0700).
void writeFakeBuild() {
  if (std::system(("mkdir -p -m 700 '" + gXdg + "/matte-refine-ml'").c_str()) != 0) CHECK(false);
  std::ofstream(gXdg + "/matte-refine-ml/fake.build") << ER_RUNNER_BUILD_ID;
}

void modelScenario(Host& host, const std::string& fake) {
  std::printf("-- Hair Locks: Edge Model through the fake runner\n");
  const std::string models = gHome + "/models";
  if (std::system(("mkdir -p '" + models + "/vitmatte-small'").c_str()) != 0) CHECK(false);
  std::ofstream(models + "/vitmatte-small/model.onnx") << "fake";
  writeConfig({{"er_runner", fake}, {"ml_model_dir", models}});
  writeFakeBuild();
  TracerScene ts;
  const scenes::Scene& s = ts.s;
  auto inst = host.createInstance(s.w, s.h, 5, ts.feeds(true));
  allOff(*inst);
  inst->setBool("emEnable", true);
  std::vector<float> out = alphaOf(inst->render(2));
  printPlan("model", *inst);
  CHECK(inst->getString(kPPlanGate) == "Model Small: ran" && inst->getString(kPPlan) == "Model");
  checkRegionContract(out, s, "model (fake runner)");
  long long raised = 0;
  for (size_t i = 0; i < out.size(); ++i) raised += out[i] > s.alpha[i];
  CHECK(raised > 0);
  // View Trimap: the model's trimap (0 / 0.5 / 1; 1 on the solid matte, unknown in the region).
  inst->setInt(kPView, kViewTrimap);
  Rendered r = inst->render(2);
  long long other = 0, unknown = 0, fgWrong = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const float t = r.rgba[i * 4];
    other += t != 0.0f && t != 0.5f && t != 1.0f;
    unknown += t == 0.5f;
    fgWrong += s.alpha[i] >= 0.95f && t != 1.0f;
  }
  CHECK(other == 0 && unknown > 0 && fgWrong == 0);
  // Candidates: the model in red.
  inst->setInt(kPView, kViewCandidates);
  r = inst->render(2);
  long long red = 0;
  for (size_t i = 0; i < out.size(); ++i) red += r.rgba[i * 4] > 0.5f * s.alpha[i];
  CHECK(red > 0);
  inst->setInt(kPView, kViewResult);
  // A failing runner: the model step fails, the matte passes through it, the other steps still run.
  std::ofstream(gXdg + "/matte-refine-ml/fake.mode") << "error";
  inst->setBool("rbEnable", true);
  out = alphaOf(inst->render(2));
  printPlan("model fails", *inst);
  CHECK(inst->getString(kPPlanGate) == "Model failed: see log" &&
        inst->getString(kPPlanNote) == "Edge Model failed: see log");
  CHECK(inst->persistent.find("Edge Model") != std::string::npos && inst->persistent.find('%') == std::string::npos);
  CHECK(checkRegionContract(out, s, "model failing, tracer on") > 20.0);
  std::remove((gXdg + "/matte-refine-ml/fake.mode").c_str());
  inst->setBool("rbEnable", false);
  // Not installed: the step is gated, the base passes through, the Model line says why.
  std::remove((models + "/vitmatte-small/model.onnx").c_str());
  CHECK(sameAlpha(alphaOf(inst->render(2)), s.alpha));
  printPlan("not installed", *inst);
  CHECK(inst->getString(kPPlanGate) == "Model Small: not installed" &&
        inst->getString(kPPlan) == "Nothing ran: pass-through");
  writeConfig({});
}

void temporalScenario(Host& host) {
  std::printf("-- Hair Locks: Temporal outside the region\n");
  const int w = 192, h = 108;
  const synth::Sequence seq = synth::makeSequence(w, h, 7, 3);
  auto rgb = [&seq](int f) { return seq.frames[size_t(f)].rgb.data(); };
  auto mat = [&seq](int f) { return seq.frames[size_t(f)].matte.data(); };
  auto inst = host.createInstance(w, h, 7, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(rgb, mat)}});
  allOff(*inst);
  const std::vector<float>& base = seq.frames[3].matte;
  CHECK(sameAlpha(alphaOf(inst->render(3)), base));  // Temporal off: bit for bit
  inst->setBool("tmEnable", true);
  const std::vector<float> everywhere = alphaOf(inst->render(3));
  printPlan("temporal", *inst);
  long long changed = 0;
  for (size_t i = 0; i < base.size(); ++i) changed += !sameBits(everywhere[i], base[i]);
  CHECK(changed > 0 && inst->getString(kPPlan) == "Temporal");
  // With a Focus Region over the right half: inside (weight 1) the base bit for bit; where the mask is 0 the
  // whole-frame result bit for bit (the weight polarity). Every other step off.
  std::vector<float> mask(base.size(), 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = w / 2; x < w; ++x) mask[size_t(y) * w + x] = 1.0f;
  inst->setFeed("FocusRegion", planeFeed([&mask](int) { return mask.data(); }));
  const std::vector<float> outside = alphaOf(inst->render(3));
  const std::vector<float> wt = weightOf(mask, w, h, 0.25);
  long long in1 = 0, in1Diff = 0, out0Diff = 0, out0Changed = 0;
  for (size_t i = 0; i < base.size(); ++i) {
    if (wt[i] >= 1.0f) ++in1, in1Diff += !sameBits(outside[i], base[i]);
    if (mask[i] == 0.0f) out0Diff += !sameBits(outside[i], everywhere[i]), out0Changed += !sameBits(outside[i], base[i]);
  }
  std::printf("temporal with region: inside %lld px (%lld changed), outside differs from the whole-frame run %lld, "
              "changed outside %lld\n", in1, in1Diff, out0Diff, out0Changed);
  CHECK(in1 > 0 && in1Diff == 0 && out0Diff == 0 && out0Changed > 0);
  // First and last frame: one neighbour each; no failure.
  inst->render(0);
  CHECK(inst->getString(kPPlan) == "Temporal");
  inst->render(6);
  CHECK(inst->getString(kPPlan) == "Temporal");
}

void jumpScenario(Host& host) {
  std::printf("-- Hair Locks: Jump Repair\n");
  const int w = 384, h = 216;
  const synth::JumpSequence js = synth::makeJumpSequence(w, h);
  std::vector<float> grey(size_t(w) * h * 3, 0.5f);
  auto rgb = [&grey](int) { return grey.data(); };
  auto mat = [&js](int f) { return js.matte[size_t(f)].data(); };
  auto inst = host.createInstance(w, h, js.frames, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(rgb, mat)}});
  allOff(*inst);
  inst->setBool("jrEnable", true);
  const int t = js.defects[0].first;  // an added disc on one frame
  const std::vector<float> out = alphaOf(inst->render(t));
  printPlan("jump", *inst);
  // The core on the same mattes (t +- 2) == the node, bit for bit; the blob is gone.
  mr::ml::Neighbours nb;
  for (int k : {-2, -1, 1, 2}) {
    mr::ml::FetchedFrame f;
    f.image = true;
    f.w = w;
    f.h = h;
    f.components = 4;
    f.isFloat = true;
    f.matte = js.matte[size_t(t + k)];
    nb.offsets.push_back(k);
    nb.frames.push_back(f);
  }
  std::vector<float> ref(out.size());
  std::vector<uint8_t> marks;
  const mr::ml::JumpUi ju;
  mr::ml::runJumpRepair(js.matte[size_t(t)].data(), w, h, mr::ml::jumpParams(ju), nb, ref.data(), nullptr, &marks);
  CHECK(sameAlpha(out, ref));
  const size_t c = size_t(std::lround(js.defects[0].cy)) * w + size_t(std::lround(js.defects[0].cx));
  std::printf("jump blob centre: matte %.2f -> %.2f\n", js.matte[size_t(t)][c], out[c]);
  CHECK(js.matte[size_t(t)][c] > 0.5f && out[c] < 0.5f);
  CHECK(inst->getString(kPPlan) == "Jump");
  // View Jumps: the core's marks and colours.
  inst->setInt(kPView, kViewJumps);
  Rendered r = inst->render(t);
  long long bad = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    float q[3];
    mr::ml::jumpsViewRgb(marks[i], js.matte[size_t(t)][i], ref[i], q);
    bad += !(sameBits(q[0], r.rgba[i * 4]) && sameBits(q[1], r.rgba[i * 4 + 1]) && sameBits(q[2], r.rgba[i * 4 + 2]));
  }
  CHECK(bad == 0);
  // Off: bit for bit. Frame 0: not judged, the Note says so.
  inst->setInt(kPView, kViewResult);
  inst->setBool("jrEnable", false);
  CHECK(sameAlpha(alphaOf(inst->render(t)), js.matte[size_t(t)]));
  inst->setBool("jrEnable", true);
  CHECK(sameAlpha(alphaOf(inst->render(0)), js.matte[0]));
  CHECK(inst->getString(kPPlanNote) == "Jump Repair: end frame");
}

// The strand presence vote through the host on the synthetic hair-locks sequence (synth::makeLocksSequence,
// WP6's injected flicker: a dip, a vanished lock, a spurious one-frame stroke, a flow-failure frame).
double locksFlicker(const synth::LocksSequence& s, const std::vector<std::vector<float>>& a) {
  double sum = 0;
  long long cnt = 0;
  for (int t = 1; t < s.frames; ++t) {
    const synth::Affine back = synth::between(s.motion[size_t(t)], s.motion[size_t(t - 1)]);
    for (int y = 0; y < s.h; ++y)
      for (int x = 0; x < s.w; ++x) {
        const size_t i = size_t(y) * s.w + x;
        if (!(s.region[size_t(t)][i] > 0)) continue;
        double qx, qy;
        back.apply(x, y, &qx, &qy);
        if (qx < 0 || qy < 0 || qx > s.w - 1 || qy > s.h - 1) continue;
        sum += std::fabs(a[size_t(t)][i] - mr::ml::sampleField(a[size_t(t - 1)].data(), s.w, s.h, float(qx), float(qy)));
        ++cnt;
      }
  }
  return cnt ? sum / cnt : 0.0;
}
double locksCoverage(const synth::LocksSequence& s, const std::vector<float>& a, int t) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (s.region[size_t(t)][i] > 0) num += std::min(a[i], s.strandTruth[size_t(t)][i]), den += s.strandTruth[size_t(t)][i];
  return den > 0 ? num / den : 0.0;
}

void voteScenario(Host& host) {
  std::printf("-- Hair Locks: strand presence vote (hair-locks sequence)\n");
  synth::LockFlicker fl;
  fl.dipFrame = 4;
  fl.vanishFrame = 8;
  fl.spuriousFrame = 11;
  fl.flowFailFrame = 14;
  const synth::LocksSequence s = synth::makeLocksSequence(424, 562, 16, 1, fl);
  auto rgb = [&s](int f) { return s.rgb[size_t(f)].data(); };
  auto mat = [&s](int f) { return s.base[size_t(f)].data(); };
  auto reg = [&s](int f) { return s.region[size_t(f)].data(); };
  auto inst = host.createInstance(s.w, s.h, s.frames, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(rgb, mat)},
                                                           {"FocusRegion", planeFeed(reg)}});
  allOff(*inst);
  inst->setInt(kPPicColour, kColourBypass);  // linear picture
  inst->setBool("rbEnable", true);
  std::vector<std::vector<float>> on(size_t(s.frames)), off(size_t(s.frames));
  inst->setBool("rbVote", false);
  for (int t = 0; t < s.frames; ++t) off[size_t(t)] = alphaOf(inst->render(t));
  inst->setBool("rbVote", true);
  long long outside = 0, below = 0;
  for (int t = 0; t < s.frames; ++t) {
    on[size_t(t)] = alphaOf(inst->render(t));
    const std::string rb = inst->getString(kPPlanRebuild), note = inst->getString(kPPlanNote);
    std::printf("  t %2d: '%s' | '%s' | cov off %.3f on %.3f\n", t, rb.c_str(), note.c_str(),
                locksCoverage(s, off[size_t(t)], t), locksCoverage(s, on[size_t(t)], t));
    if (t == fl.flowFailFrame) CHECK(note == "Vote: own result only" && rb.find("Created matte") == 0);
    else CHECK(rb.find("Vote: kept") == 0);  // first / last frame too: t+1, t+2 (t-1, t-2) are there
    checkPlanFits(*inst);
    for (size_t i = 0; i < on[size_t(t)].size(); ++i) {
      outside += s.region[size_t(t)][i] == 0.0f && !sameBits(on[size_t(t)][i], s.base[size_t(t)][i]);
      below += on[size_t(t)][i] < s.base[size_t(t)][i];
    }
  }
  CHECK(outside == 0 && below == 0);
  const double flOff = locksFlicker(s, off), flOn = locksFlicker(s, on);
  std::printf("  flicker of the effect's matte in the region: vote off %.4f, on %.4f\n", flOff, flOn);
  CHECK(flOn < flOff);
  double a0 = 0, a1 = 0;  // the spurious stroke: gone with the vote
  for (size_t i = 0; i < s.spurious[size_t(fl.spuriousFrame)].size(); ++i)
    if (s.spurious[size_t(fl.spuriousFrame)][i])
      a0 += off[size_t(fl.spuriousFrame)][i] - s.base[size_t(fl.spuriousFrame)][i],
          a1 += on[size_t(fl.spuriousFrame)][i] - s.base[size_t(fl.spuriousFrame)][i];
  std::printf("  spurious frame %d: added alpha on the stroke off %.1f, on %.1f\n", fl.spuriousFrame, a0, a1);
  CHECK(a0 > 5.0 && a1 <= 0.1 * a0 + 1.0);
  for (int f : {fl.dipFrame, fl.vanishFrame}) {  // a lost lock is reinstated; a kept one stays
    const double c0 = locksCoverage(s, off[size_t(f)], f), c1 = locksCoverage(s, on[size_t(f)], f);
    std::printf("  frame %d: coverage off %.3f on %.3f\n", f, c0, c1);
    CHECK(c1 >= c0 - 0.02);
    if (c0 < 0.8) CHECK(c1 > c0 + 0.1);
  }
  {  // flow failure: the frame's own result
    const int f = fl.flowFailFrame;
    CHECK(std::fabs(locksCoverage(s, on[size_t(f)], f) - locksCoverage(s, off[size_t(f)], f)) <= 0.02);
  }
  // View Rebuilt on the spurious frame: the dropped stroke dim red, nothing added there.
  inst->setInt(kPView, kViewRebuilt);
  Rendered r = inst->render(fl.spuriousFrame);
  long long red = 0;
  for (size_t i = 0; i < on[0].size(); ++i)
    if (s.spurious[size_t(fl.spuriousFrame)][i]) red += r.rgba[i * 4] >= 0.5f - 1e-6f && r.rgba[i * 4 + 1] < 0.5f;
  std::printf("  View Rebuilt: dropped stroke pixels shown dim red %lld\n", red);
  CHECK(red > 10);
  inst->setInt(kPView, kViewResult);
  // Temporal on: it reuses the vote's flows (the log says so at debug level).
  inst->setBool("debugLog", true);
  inst->setBool("tmEnable", true);
  inst->render(6);
  CHECK(inst->getString(kPPlan) == "Rebuild+Vote+Temporal");
  inst->setBool("debugLog", false);
  {
    std::ifstream log(gHome + "/.cache/matte-refine/logs/matte-refine.log");
    std::string line, lastVote;
    while (std::getline(log, line))
      if (line.find(" vote: neighbours traced") != std::string::npos) lastVote = line;
    // a small region: the vote flows a crop, Temporal its own whole frame (its output as without the vote)
    CHECK(lastVote.find("(crop flows) | Temporal computed its own flows") != std::string::npos);
  }
  {  // a region over the whole frame: the crop is the frame, one whole-frame flow set shared with Temporal
    auto full = host.createInstance(s.w, s.h, s.frames, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(rgb, mat)},
                                                             {"FocusRegion", constFeed(1.0f)}});
    allOff(*full);
    full->setInt(kPPicColour, kColourBypass);
    full->setBool("rbEnable", true);
    full->setBool("tmEnable", true);
    full->setBool("debugLog", true);
    full->render(6);
    full->setBool("debugLog", false);
    std::ifstream log(gHome + "/.cache/matte-refine/logs/matte-refine.log");
    std::string line, lastVote;
    while (std::getline(log, line))
      if (line.find(" vote: neighbours traced") != std::string::npos) lastVote = line;
    CHECK(lastVote.find("(whole-frame flows) | Temporal used the shared flows") != std::string::npos);
  }
  // No neighbours (a one-frame clip): the frame's own result, with the Plan note.
  auto single = host.createInstance(s.w, s.h, 1, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(rgb, mat)},
                                                       {"FocusRegion", planeFeed(reg)}});
  allOff(*single);
  single->setInt(kPPicColour, kColourBypass);
  single->setBool("rbEnable", true);
  const std::vector<float> own = alphaOf(single->render(0));
  printPlan("one-frame clip", *single);
  CHECK(single->getString(kPPlanNote) == "Vote: own result only");
  CHECK(std::fabs(locksCoverage(s, own, 0) - locksCoverage(s, off[0], 0)) <= 0.02);
}

// getFramesNeeded: the union of the enabled steps' reach on the Source, clamped to the clip (frames 0-9).
void framesNeededScenario(Host& host) {
  std::printf("-- Hair Locks: getFramesNeeded\n");
  auto inst = host.createInstance(W, H, 10, Feeds{{kOfxImageEffectSimpleSourceClipName, sourceFeed(4)}});
  // The out property the Support library writes (ofxsImageEffect.cpp:755).
  const std::string prop = std::string("OfxImageClipPropFrameRange_") + kOfxImageEffectSimpleSourceClipName;
  auto range = [&](double t, double* lo, double* hi) {
    PropSet in, out;
    in.setDbl(kOfxPropTime, t);
    inst->dispatch(kOfxImageEffectActionGetFramesNeeded, &in, &out);
    if (!out.props.count(prop)) return false;
    *lo = out.getDbl(prop.c_str(), 0);
    *hi = out.getDbl(prop.c_str(), 1);
    return true;
  };
  double lo = 0, hi = 0;
  allOff(*inst);
  CHECK(!range(5, &lo, &hi));  // nothing needs neighbours: the host's default
  inst->setBool("emEnable", true);
  inst->setBool("rbEnable", true);  // Temporal Vote on (default): its reach K = 2
  CHECK(range(5, &lo, &hi) && lo == 3 && hi == 7);
  inst->setBool("tmEnable", true);  // Temporal +-1: the union stays +-2
  CHECK(range(5, &lo, &hi) && lo == 3 && hi == 7);
  inst->setBool("rbVote", false);  // the vote off: Temporal's +-1
  CHECK(range(5, &lo, &hi) && lo == 4 && hi == 6);
  inst->setBool("rbEnable", false);
  inst->setBool("rbVote", true);  // the vote without Rebuild Detail does not run: no reach
  CHECK(range(5, &lo, &hi) && lo == 4 && hi == 6);
  inst->setInt("tmFrames", 1);
  CHECK(range(5, &lo, &hi) && lo == 3 && hi == 7);
  inst->setBool("tmEnable", false);
  inst->setBool("jrEnable", true);  // Span 1: t +- 2
  CHECK(range(5, &lo, &hi) && lo == 3 && hi == 7);
  inst->setDouble("jrSpanFrames", 3);  // Span 3: t +- 4
  CHECK(range(5, &lo, &hi) && lo == 1 && hi == 9);
  inst->setBool("tmEnable", true);  // the union: the larger reach
  CHECK(range(5, &lo, &hi) && lo == 1 && hi == 9);
  inst->setDouble("jrSpanFrames", 1);
  inst->setInt("tmFrames", 0);
  CHECK(range(5, &lo, &hi) && lo == 3 && hi == 7);
  CHECK(range(0, &lo, &hi) && lo == 0 && hi == 2);  // clamped to the clip
  CHECK(range(9, &lo, &hi) && lo == 7 && hi == 9);
}

// Not a test: everything on (Jump Repair, the model through the fake runner, tracer, Detail, Snap, Temporal) on a
// synthetic HD sequence and its 2x (UHD) upscale; the per-step ms from the log's DEBUG render line.
int timing(const std::string& bundle, const std::string& fake) {
  setenv("MR_LOG_LEVEL", "debug", 1);
  const std::string models = gHome + "/models";
  if (std::system(("mkdir -p '" + models + "/vitmatte-small'").c_str()) != 0) return 1;
  std::ofstream(models + "/vitmatte-small/model.onnx") << "fake";
  writeConfig({{"er_runner", fake}, {"ml_model_dir", models}});
  writeFakeBuild();
  const synth::Sequence seq = synth::makeSequence(1920, 1080, 5, 3);
  Host host(bundle);
  if (!host.loaded() || host.describe() != kOfxStatOK) return 1;
  for (int scale : {1, 2}) {
    const int w = 1920 * scale, h = 1080 * scale;
    // The region: a 360 x 240 box (at HD) across the subject's right edge at mid height.
    const std::vector<float>& m2 = seq.frames[2].matte;
    int edge = 0;
    for (int x = 1919; x >= 0; --x)
      if (m2[size_t(540) * 1920 + x] > 0.5f) {
        edge = x;
        break;
      }
    std::vector<float> mask(size_t(w) * h, 0.0f);
    for (int y = (540 - 120) * scale; y < (540 + 120) * scale; ++y)
      for (int x = std::max(0, edge - 180) * scale; x < std::min(1920, edge + 180) * scale; ++x) mask[size_t(y) * w + x] = 1.0f;
    std::vector<std::vector<float>> rgb(5), mat(5), det(5);
    for (int f = 0; f < 5; ++f) {
      const synth::Scene& sc = seq.frames[size_t(f)];
      rgb[size_t(f)].resize(size_t(w) * h * 3);
      mat[size_t(f)].resize(size_t(w) * h);
      det[size_t(f)].resize(size_t(w) * h);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x, j = size_t(y / scale) * 1920 + size_t(x / scale);
          for (int k = 0; k < 3; ++k) rgb[size_t(f)][i * 3 + k] = sc.rgb[j * 3 + k];
          mat[size_t(f)][i] = sc.matte[j];
          det[size_t(f)][i] = sc.truth[j];
        }
    }
    Feeds feeds{{kOfxImageEffectSimpleSourceClipName,
                 rgbaFeed([&rgb](int f) { return rgb[size_t(f)].data(); }, [&mat](int f) { return mat[size_t(f)].data(); })},
                {"FocusRegion", planeFeed([&mask](int) { return mask.data(); })},
                {"Detail", planeFeed([&det](int f) { return det[size_t(f)].data(); })}};
    auto inst = host.createInstance(w, h, 5, feeds);
    for (const char* id : {"emEnable", "rbEnable", "rgSnap", "tmEnable", "jrEnable"}) inst->setBool(id, true);
    for (int i = 0; i < 3; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      inst->render(2);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("%dx%d render %d: %.0f ms through the host\n", w, h, i, ms);
    }
    printPlan("timing", *inst);
  }
  std::ifstream log(gHome + "/.cache/matte-refine/logs/matte-refine.log");
  std::string line;
  while (std::getline(log, line))
    if (line.find(" vote: neighbours traced") != std::string::npos) {
      std::printf("  %s\n", line.substr(line.find("render t=")).c_str());
    } else if (line.find("hairLocks rev") != std::string::npos) {
      const size_t a = line.find("render t="), b = line.find(" | inputs");
      if (a != std::string::npos) std::printf("%s\n", line.substr(a, b == std::string::npos ? b : b - a).c_str());
    }
  killFakeRunners();
  writeConfig({});
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const bool mutant = argc == 4 && mode == "--mutant";
  if (!(mutant || argc == 5)) {
    std::fprintf(stderr, "usage: recipes_systest <EdgeRecipes.ofx> <EdgeRecipesTest.ofx> <scratch home> <fake runner>\n"
                         "       recipes_systest --mutant <EdgeRecipesMutant.ofx> <scratch home>\n"
                         "       recipes_systest --timing <EdgeRecipes.ofx> <scratch home> <fake runner>\n");
    return 2;
  }
  signal(SIGPIPE, SIG_IGN);
  gHome = argv[3];
  if (std::system(("rm -rf '" + gHome + "' && mkdir -p '" + gHome + "'").c_str()) != 0) return 2;
  char xdg[] = "/tmp/er-systest-XXXXXX";  // mode 0700, ours: the runner socket here, never next to a real runner
  if (!mkdtemp(xdg)) return 2;
  gXdg = xdg;
  setenv("HOME", gHome.c_str(), 1);
  setenv("XDG_RUNTIME_DIR", xdg, 1);
  unsetenv("MR_LOG_LEVEL");
  if (mutant) {
    // The shared-factory build: both effects dispatch to one uid, so Hair Locks describes as the dummy (or the
    // dummy as Hair Locks). Exit 0 = the checks did NOT notice (ctest WILL_FAIL turns that into a failure);
    // a bundle that does not load also exits 0, so a missing mutant cannot pass silently.
    bool loaded = runOne(argv[2], 0, 2, HairLocksTag::def());
    loaded = runOne(argv[2], 1, 2, TestDummy::def()) && loaded;
    if (std::system(("rm -rf '" + gXdg + "'").c_str()) != 0) std::printf("could not remove %s\n", gXdg.c_str());
    std::printf("recipes_systest --mutant: %d failures (expected > 0)\n", failures);
    return loaded && failures > 0 ? 1 : 0;
  }
  if (mode == "--timing") {
    const int rc = timing(argv[2], argv[4]);
    if (std::system(("rm -rf '" + gXdg + "'").c_str()) != 0) std::printf("could not remove %s\n", gXdg.c_str());
    return rc;
  }
  runOne(argv[1], 0, 1, HairLocksTag::def());
  runOne(argv[2], 0, 2, HairLocksTag::def());
  runOne(argv[2], 1, 2, TestDummy::def());
  {
    Host host(argv[1]);
    if (host.loaded() && host.describe() == kOfxStatOK) {
      rebuildScenario(host);
      detailScenario(host);
      modelScenario(host, argv[4]);
      temporalScenario(host);
      jumpScenario(host);
      voteScenario(host);
      framesNeededScenario(host);
      for (const auto& err : host.errors) std::printf("FAIL host: %s\n", err.c_str());
      failures += int(host.errors.size());
    } else {
      std::printf("FAIL load %s\n", argv[1]);
      ++failures;
    }
  }
  killFakeRunners();
  if (std::system(("rm -rf '" + gXdg + "'").c_str()) != 0) std::printf("could not remove %s\n", gXdg.c_str());
  std::printf("recipes_systest: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
