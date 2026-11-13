// Edge Recipes probe system test: loads EdgeRecipesProbe.ofx in the Flame-like host
// (ofx/merge/systest/ofx_host.*) once per plugin index and checks that the binary exports three
// effects with distinct ids under "Edge Recipes", each described and rendered as ITSELF (the
// factory-per-recipe catch, docs/edge_recipes.md 4.1): its own constant in the footer, its own tint
// and matte, its own sockets; Gamma's layout (tab 1 = 6 columns, then the "Advanced" page); the
// Plan label string and the relabelled boolean after a render; the mask clips reported Alpha-only.
// Usage: test_recipes_probe <EdgeRecipesProbe.ofx> <scratch HOME folder>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "ofxParam.h"
#include "ofx_host.h"

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

constexpr int W = 64, H = 48;

struct Expect {
  const char* id;
  const char* label;
  const char* constant;
  int tintChannel;  // the channel kept at full gain
  int matte;        // 0 alpha, 1 1 - alpha, 2 0.5 alpha
  size_t clips;     // incl. Output
  bool advanced;
};
const Expect kExpect[3] = {
    {"com.janklier.edgerecipesprobe.alpha", "Probe Alpha", "K=101", 0, 0, 2, false},
    {"com.janklier.edgerecipesprobe.beta", "Probe Beta", "K=202", 1, 1, 4, false},
    {"com.janklier.edgerecipesprobe.gamma", "Probe Gamma", "K=303", 2, 2, 4, true},
};

// Picture: flat grey 0.5 RGB, alpha ramp left to right (0 .. 1).
Feed sourceFeed() {
  return Feed{[](int, int w, int h, std::vector<float>& o) {
                for (int y = 0; y < h; ++y)
                  for (int x = 0; x < w; ++x) {
                    float* p = &o[(size_t(y) * w + x) * 4];
                    p[0] = p[1] = p[2] = 0.5f;
                    p[3] = float(x) / (w - 1);
                  }
              },
              4};
}
Feed maskFeed() {
  return Feed{[](int, int w, int h, std::vector<float>& o) { std::fill(o.begin(), o.begin() + size_t(w) * h, 0.25f); },
              1};
}

// Flame's layout rules (ofx/CLAUDE.md "Flame host behaviour"): each group starts a column, rows
// choice 2 / others 1 (hidden ones too), <= 5 rows, 6 columns per tab; overflow columns go to a tab
// named after the page defined last before them. Returns the columns' page names.
std::vector<std::string> checkLayout(Host& host, const std::string& firstPage) {
  std::vector<int> rows;
  std::vector<std::string> pages;
  std::string page;
  for (const auto& p : host.descriptor().params.list) {
    if (p->type == kOfxParamTypePage) {
      page = p->name;
      continue;
    }
    if (p->type == kOfxParamTypeGroup) {
      rows.push_back(0);
      pages.push_back(page);
      continue;
    }
    CHECK(!p->props.getStr(kOfxParamPropParent).empty());
    CHECK(!rows.empty());
    if (rows.empty()) return pages;
    rows.back() += p->type == kOfxParamTypeChoice ? 2 : 1;
    CHECK(p->props.getStr(kOfxPropLabel).size() <= 30);
    if (p->type == kOfxParamTypeDouble) CHECK(p->props.getInt(kOfxParamPropDigits) >= 2);
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    CHECK(rows[i] <= 5);
    if (i < 6) CHECK(pages[i] == firstPage);
  }
  return pages;
}

std::string runOne(int index, const std::string& path) {
  const Expect& e = kExpect[index];
  Host host(path, index);
  if (!host.loaded()) {
    std::printf("FAIL load %d: %s\n", index, host.loadError().c_str());
    ++failures;
    return "";
  }
  CHECK(host.pluginCount() == 3);
  CHECK(std::strcmp(host.plugin()->pluginIdentifier, e.id) == 0);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();
  CHECK(d.props.getStr(kOfxPropLabel) == e.label);
  CHECK(d.props.getStr(kOfxImageEffectPluginPropGrouping) == "Edge Recipes");
  CHECK(d.clips.size() == e.clips);
  for (const char* m : {"Detail", "FocusRegion"}) {
    if (e.clips != 4) break;
    CHECK(d.clips.count(m) == 1);
    if (!d.clips.count(m)) continue;
    CHECK(d.clips[m]->props.getInt(kOfxImageClipPropIsMask) == 1);
    CHECK(d.clips[m]->props.getInt(kOfxImageClipPropOptional) == 1);
  }
  // Describe-time labels are per descriptor.
  CHECK(d.params.byName.count("strength") &&
        d.params.byName["strength"]->props.getStr(kOfxPropLabel) == std::string(e.label) + " Strength");
  const std::vector<std::string> pages = checkLayout(host, e.label);
  if (e.advanced) {
    CHECK(pages.size() == 8);
    for (size_t i = 6; i < pages.size(); ++i) CHECK(pages[i] == "Advanced");
  } else {
    CHECK(pages.size() == 4);
  }

  Feeds feeds{{kOfxImageEffectSimpleSourceClipName, sourceFeed()}};
  if (e.clips == 4) feeds["Detail"] = maskFeed();
  auto inst = host.createInstance(W, H, 10, feeds);
  const std::string nodeId = inst->getString("nodeId");
  CHECK(nodeId.size() == 8);
  Rendered r = inst->render(3);
  CHECK(r.status == kOfxStatOK);
  const std::string& footer = inst->persistent;
  std::printf("footer %d: %s\n", index, footer.c_str());
  CHECK(footer.find(std::string(e.label) + " " + e.constant) == 0);
  CHECK(footer.find("| f3 |") != std::string::npos);
  CHECK(footer.find("Source 4ch 0,0-64,48") != std::string::npos);
  CHECK(footer.find("id " + nodeId) != std::string::npos);
  if (e.clips == 4) {
    CHECK(footer.find("Detail 1ch 0,0-64,48") != std::string::npos);
    CHECK(footer.find("Focus off") != std::string::npos);
  } else {
    CHECK(footer.find("Detail") == std::string::npos);
  }
  CHECK(footer.find('%') == std::string::npos);  // host message text is a format string
  // Output: grey 0.5 tinted towards the effect's colour; alpha per the effect's matte.
  for (int x : {0, W / 2, W - 1}) {
    const float* p = r.px(x, H / 2);
    const float a = float(x) / (W - 1);
    const float want = e.matte == 0 ? a : e.matte == 1 ? 1.f - a : 0.5f * a;
    CHECK(std::fabs(p[3] - want) < 1e-6f);
    for (int c = 0; c < 3; ++c) {
      if (c == e.tintChannel) CHECK(std::fabs(p[c] - 0.5f) < 1e-6f);
      else CHECK(p[c] < 0.45f);
    }
  }
  // Runtime UI text: Plan value and the relabelled boolean.
  const std::string plan = inst->getString("plan");
  std::printf("plan %d: '%s'\n", index, plan.c_str());
  CHECK(plan.compare(0, 4, std::string(1, e.label[6]) + " f3") == 0);
  CHECK(plan.size() <= 28);
  CHECK(inst->effect().params.byName["labelProbe"]->props.getStr(kOfxPropLabel) == "Seen at f3");
  // Focus Region connected: reported, Alpha-only.
  if (e.clips == 4) {
    inst->setFeed("FocusRegion", maskFeed());
    inst->render(4);
    CHECK(inst->persistent.find("Focus 1ch 0,0-64,48") != std::string::npos);
    CHECK(inst->getString("plan").find("D1 F1") != std::string::npos);
  }
  // A saved value shows in the footer (the Flame round trip compares these).
  inst->setDouble("strength", 0.0);
  Rendered r0 = inst->render(5);
  CHECK(inst->persistent.find("S 0.00") != std::string::npos);
  CHECK(std::fabs(r0.px(1, 1)[0] - 0.5f) < 1e-6f && std::fabs(r0.px(1, 1)[1] - 0.5f) < 1e-6f);
  CHECK(inst->maxEchoDepth <= 2);
  for (const auto& err : host.errors) std::printf("FAIL host: %s\n", err.c_str());
  failures += int(host.errors.size());
  return footer;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: test_recipes_probe <EdgeRecipesProbe.ofx> <scratch home>\n");
    return 2;
  }
  mkdir(argv[2], 0755);
  setenv("HOME", argv[2], 1);
  unsetenv("MR_LOG_LEVEL");
  std::set<std::string> footers, ids;
  for (int i = 0; i < 3; ++i) {
    footers.insert(runOne(i, argv[1]));
    ids.insert(kExpect[i].id);
  }
  CHECK(footers.size() == 3 && ids.size() == 3);
  std::printf("test_recipes_probe: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
