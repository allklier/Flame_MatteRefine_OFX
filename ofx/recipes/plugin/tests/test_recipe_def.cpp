// The recipe table against Flame's layout rules and the append-only id rule (docs/edge_recipes.md 4.3, 4.7):
// every param exactly once, <= 5 rows per column (choice 2, others 1, hidden ones too), tab 1 filled to exactly 6
// columns so the next page's columns land on a tab named "Advanced", labels <= 15 characters (Flame cuts them),
// 5-8 exposed recipe controls, and the Hair Locks ids pinned: an id may be added (append it to kHairLocksIds) but
// never removed or renamed.
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "recipe_def.h"

namespace {
int failures = 0;
#define CHECK(c)                                               \
  do {                                                         \
    if (!(c)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                              \
    }                                                          \
  } while (0)

using namespace mr::recipes;

// Hair Locks' param and group ids as released (append-only: new ids go at the end).
const char* const kHairLocksIds[] = {
    "grpMatte", "view", "version",
    "grpHairLocks", "rgFeather", "emEnable", "emGain", "rbEnable", "rbMinConf",
    "grpHairLocks2", "tmStrength", "rgSnapStrength", "rgSnapMode",
    "grpPlan", "plan", "planRebuild", "planGate", "planInputs", "planNote",
    "grpModelColour", "mdModel", "picColour",
    "grpSupport", "debugLog", "collectLogs", "saveMatte", "nodeId", "recipeRev",
    "grpJumpRepair", "jrEnable", "jrThreshold", "jrMinSize", "jrSpanFrames", "jrGuards",
    "grpFocusRegion", "rgGrow", "rgSnap",
    "grpEdgeModel", "emLowContrast",
    "grpRebuild", "rbKeepLoose", "rbVote", "rbBlend", "rbMinPiece",
    "grpTemporal", "tmEnable", "tmFrames", "tmBand",
};
// The View options as released (index append-only, 4.3).
const char* const kHairLocksViews[] = {"Result Matte", "Picture", "Changes", "Region",
                                       "Jumps",        "Candidates", "Rebuilt", "Trimap"};
// Columns shared by every recipe (not recipe controls).
const std::set<std::string> kCommonGroups = {"grpMatte", "grpPlan", "grpModelColour", "grpSupport"};

// Checks every recipe must pass; returns the ids in definition order.
std::set<std::string> checkLayout(const RecipeDef& d, bool advanced) {
  std::set<std::string> ids;
  int exposed = 0;
  CHECK(!d.pages.empty());
  CHECK(std::strlen(d.label) <= size_t(kMaxLabel));
  for (size_t pg = 0; pg < d.pages.size(); ++pg) {
    const PageDef& page = d.pages[pg];
    CHECK(!page.columns.empty() && page.columns.size() <= size_t(kMaxColumns));
    if (pg == 0) CHECK(page.name == std::string(d.label));
    // A later page only starts its own tab when the tab before it is full (Flame: overflow only).
    if (pg + 1 < d.pages.size()) CHECK(page.columns.size() == size_t(kMaxColumns));
    if (pg == 1) CHECK(page.name == std::string("Advanced"));
    for (const ColumnDef& c : page.columns) {
      CHECK(ids.insert(c.group).second);
      CHECK(std::strlen(c.label) <= size_t(kMaxLabel));
      int rows = 0;
      for (const ParamDef& p : c.params) {
        if (!ids.insert(p.id).second) std::printf("FAIL duplicate id %s\n", p.id), ++failures;
        rows += rowsOf(p.kind);
        if (std::strlen(p.label) > size_t(kMaxLabel)) std::printf("FAIL label too long: %s\n", p.label), ++failures;
        if (p.kind == Kind::Double) CHECK(p.digits >= 0 && p.lo <= p.def && p.def <= p.hi);
        if (p.kind == Kind::Choice) {
          const size_t n = (p.flags & kViewOptions) ? d.views.size() : p.options.size();
          CHECK(n >= 2 && p.def >= 0 && p.def < double(n));
        }
        if (pg == 0 && !kCommonGroups.count(c.group)) ++exposed;
      }
      if (rows > kMaxRows) std::printf("FAIL column %s has %d rows\n", c.group, rows), ++failures;
    }
  }
  CHECK(d.pages.size() == (advanced ? 2u : 1u));
  if (advanced) CHECK(exposed >= 5 && exposed <= 8);
  // Every recipe has the shared params the plugin layer reads.
  for (const char* id : {kPView, kPVersion, kPPicColour, kPModel, kPDebugLog, kPCollectLogs, kPSaveMatte, kPNodeId,
                         kPRecipeRev, kPPlan, kPPlanRebuild, kPPlanGate, kPPlanInputs, kPPlanNote})
    CHECK(findParam(d, id) != nullptr);
  CHECK(findParam(d, kPNodeId)->kind == Kind::HiddenString && findParam(d, kPRecipeRev)->kind == Kind::HiddenInt);
  CHECK(findParam(d, "noSuchParam") == nullptr);
  CHECK(d.views.size() >= 2 && d.views[kViewResult] == "Result Matte" && d.views[kViewPicture] == "Picture");
  CHECK(d.versionMajor >= 1 && d.recipeRev >= 1);
  return ids;
}

}  // namespace

int main() {
  const RecipeDef& h = HairLocksTag::def();
  CHECK(pluginId(h) == "com.janklier.edgerecipes.hairlocks");
  CHECK(std::string(h.label) == "Hair Locks");
  CHECK(h.detailSocket && h.regionSocket && !h.requiresModel);
  CHECK(h.pages.size() == 2 && h.pages[0].columns.size() == 6);
  const std::set<std::string> ids = checkLayout(h, true);
  std::set<std::string> pinned(std::begin(kHairLocksIds), std::end(kHairLocksIds));
  for (const auto& id : pinned)
    if (!ids.count(id)) std::printf("FAIL pinned id removed or renamed: %s\n", id.c_str()), ++failures;
  for (const auto& id : ids)
    if (!pinned.count(id)) std::printf("FAIL new id %s: append it to kHairLocksIds\n", id.c_str()), ++failures;
  CHECK(h.views.size() == sizeof kHairLocksViews / sizeof kHairLocksViews[0]);
  for (size_t i = 0; i < h.views.size() && i < sizeof kHairLocksViews / sizeof kHairLocksViews[0]; ++i)
    CHECK(h.views[i] == kHairLocksViews[i]);
  // Tab 1 order (4.3): Matte, the recipe's columns, Plan, Model & Colour, Support last (the hidden params).
  const auto& cols = h.pages[0].columns;
  CHECK(std::string(cols[0].group) == "grpMatte" && std::string(cols[3].group) == "grpPlan" &&
        std::string(cols[4].group) == "grpModelColour" && std::string(cols[5].group) == "grpSupport");
  CHECK(findParam(h, "rbEnable")->def == 0 && findParam(h, "jrEnable")->def == 0);  // generator and Jump Repair off

  const RecipeDef& t = TestDummy::def();
  checkLayout(t, false);
  CHECK(pluginId(t) != pluginId(h));

  std::printf("test_recipe_def: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
