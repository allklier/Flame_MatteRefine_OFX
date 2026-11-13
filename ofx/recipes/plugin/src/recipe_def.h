// Edge Recipes (decision 41, docs/edge_recipes.md 4): the static definition of one recipe effect, without
// OFX types, so the layout is unit-tested without a host. plugin_recipes.cpp turns a RecipeDef into an OFX
// descriptor (pages, groups = columns, params) and one RecipeFactory<Tag> per recipe.
// Flame layout rules this table must follow (ofx/CLAUDE.md "Flame host behaviour", probe 2026-11-12): each
// group starts a column of <= 5 rows (choice 2, every other param 1, a hidden param too), 6 columns per tab, a
// new tab only on overflow, named after the page defined before it; labels cut at about 15 characters.
// Param ids are append-only per recipe (stored in setups): never rename or reuse; choice options append-only.
#pragma once

#include <string>
#include <vector>

namespace mr {
namespace recipes {

// Owner part of every plugin id (as plugin.cpp:60, plugin_ml.cpp:60, plugin_merge.cpp:59).
constexpr const char* kIdPrefix = "com.janklier.edgerecipes.";
constexpr const char* kGrouping = "Edge Recipes";  // flat: Flame's OFX menu is group, then effect
constexpr int kMaxLabel = 15;                      // Flame truncates labels at about 15 characters (probe)
constexpr int kMaxRows = 5, kMaxColumns = 6;

enum class Kind { Bool, Double, Choice, LabelString, PushButton, HiddenString, HiddenInt };

enum Flags : unsigned {
  kNoAnimate = 1u << 0,       // setAnimates(false)
  kClipPrefSlave = 1u << 1,   // addClipPreferencesSlaveParam (Picture Colour, decision 34)
  kNotPersistent = 1u << 2,   // not stored in setups (Debug Logging: per user)
  kNoEvaluate = 1u << 3,      // setEvaluateOnChange(false)
  kModelOptions = 1u << 4,    // options get "(not installed)" from the model folder (gate.h optionLabel)
  kViewOptions = 1u << 5,     // options = RecipeDef::views
};

struct ParamDef {
  ParamDef(const char* id, const char* label, const char* hint, Kind kind) : id(id), label(label), hint(hint), kind(kind) {}
  const char* id;
  const char* label;
  const char* hint;
  Kind kind;
  double def = 0;  // Bool 0/1, Choice index, HiddenInt value
  double lo = 0, hi = 1;
  int digits = 2;  // Flame showed doubles without digits as integers
  double inc = 0.05;
  std::vector<std::string> options;  // Choice
  unsigned flags = 0;
};

struct ColumnDef {
  const char* group;  // group param id
  const char* label;
  std::vector<ParamDef> params;
};

struct PageDef {
  const char* name;  // tab name (the first tab: the recipe's name; the overflow tab: "Advanced")
  std::vector<ColumnDef> columns;
};

struct RecipeDef {
  const char* idSuffix;  // plugin id = kIdPrefix + idSuffix; never changed after a release
  const char* label;     // effect name in Flame's menu
  int versionMajor, versionMinor;  // OFX plugin version of this descriptor (docs/edge_recipes.md 4.2)
  int recipeRev;                   // set on new nodes in the hidden recipeRev param (behaviour pinning)
  std::vector<PageDef> pages;
  std::vector<std::string> views;  // View choice options, index append-only; 0 Result Matte, 1 Picture
  bool detailSocket, regionSocket;  // optional mask clips "Detail", "FocusRegion" (4.4)
  bool requiresModel;               // false: classical steps run with the model gate closed (4.6)
};

int rowsOf(Kind k);  // Flame rows a param takes in its column
std::string pluginId(const RecipeDef& d);
const ParamDef* findParam(const RecipeDef& d, const std::string& id);

// One tag type per recipe: each gets its own RecipeFactory<Tag> class (docs/edge_recipes.md 4.1).
struct HairLocksTag {  // (the core's recipe class is mr::recipes::HairLocks, hair_locks.h)
  static const RecipeDef& def();
};
// Test builds only (ER_TEST_RECIPES): a second, minimal recipe that proves ids and descriptors stay distinct.
struct TestDummy {
  static const RecipeDef& def();
};

// Param ids shared by every recipe (the plugin layer reads them; append-only).
constexpr const char* kPView = "view";
constexpr const char* kPVersion = "version";
constexpr const char* kPPicColour = "picColour";
constexpr const char* kPModel = "mdModel";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPSaveMatte = "saveMatte";
constexpr const char* kPNodeId = "nodeId";
constexpr const char* kPRecipeRev = "recipeRev";
// Plan label strings, rewritten per render (only when the text changes).
constexpr const char* kPPlan = "plan";
constexpr const char* kPPlanRebuild = "planRebuild";
constexpr const char* kPPlanGate = "planGate";
constexpr const char* kPPlanInputs = "planInputs";
constexpr const char* kPPlanNote = "planNote";
constexpr int kPlanMaxChars = 28;  // Flame cuts label-string text at about 30 (ofx/CLAUDE.md)

// View indexes (append-only, stored in setups). 0-1 every recipe; 2-7 Hair Locks (docs/edge_recipes.md 4.3).
enum ViewIndex {
  kViewResult = 0,
  kViewPicture = 1,
  kViewChanges = 2,
  kViewRegion = 3,
  kViewJumps = 4,
  kViewCandidates = 5,
  kViewRebuilt = 6,
  kViewTrimap = 7
};
enum PicColour { kColourAuto = 0, kColourBypass = 1 };

}  // namespace recipes
}  // namespace mr
