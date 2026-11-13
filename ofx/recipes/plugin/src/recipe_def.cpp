// The recipe table (docs/edge_recipes.md 4.3). Hair Locks v1 exposes the controls of the user's ruling of
// 2026-11-12 (CLAUDE.md decision 41: Region Feather %, Edge Model, Strand Gain, Rebuild Detail, Rebuild Min
// Confidence, Temporal Strength, Snap Strength; Jump Repair and the Temporal switch moved to Advanced, Jump
// Repair off), every other control once on the Advanced tab. Labels <= 15 characters (Flame cuts them).
#include "recipe_def.h"

namespace mr {
namespace recipes {

int rowsOf(Kind k) { return k == Kind::Choice ? 2 : 1; }

std::string pluginId(const RecipeDef& d) { return std::string(kIdPrefix) + d.idSuffix; }

const ParamDef* findParam(const RecipeDef& d, const std::string& id) {
  for (const PageDef& pg : d.pages)
    for (const ColumnDef& c : pg.columns)
      for (const ParamDef& p : c.params)
        if (id == p.id) return &p;
  return nullptr;
}

namespace {

ParamDef boolean(const char* id, const char* label, const char* hint, bool def, unsigned flags = 0) {
  ParamDef p{id, label, hint, Kind::Bool};
  p.def = def ? 1 : 0;
  p.flags = flags;
  return p;
}

ParamDef number(const char* id, const char* label, const char* hint, double def, double lo, double hi, int digits,
                double inc) {
  ParamDef p{id, label, hint, Kind::Double};
  p.def = def;
  p.lo = lo;
  p.hi = hi;
  p.digits = digits;
  p.inc = inc;
  return p;
}

ParamDef choice(const char* id, const char* label, const char* hint, std::vector<std::string> options, int def,
                unsigned flags = 0) {
  ParamDef p{id, label, hint, Kind::Choice};
  p.options = std::move(options);
  p.def = def;
  p.flags = flags;
  return p;
}

ParamDef labelString(const char* id, const char* label) {
  ParamDef p{id, label, "", Kind::LabelString};
  p.flags = kNotPersistent | kNoEvaluate;
  return p;
}

// Columns every recipe has (the Matte column holds only View and Version, so a recipe may add rows to it).
ColumnDef matteColumn() {
  return {"grpMatte",
          "Matte",
          {choice(kPView, "View",
                  "What the image output shows (the matte output is always the result). Result Matte: the result "
                  "in grey. Picture: the Source input.",
                  {}, kViewResult, kNoAnimate | kViewOptions),
           labelString(kPVersion, "Version")}};
}

ColumnDef planColumn() {
  return {"grpPlan",
          "Plan",
          {labelString(kPPlan, "Plan"), labelString(kPPlanRebuild, "Rebuild"), labelString(kPPlanGate, "Model"),
           labelString(kPPlanInputs, "Inputs"), labelString(kPPlanNote, "Note")}};
}

ColumnDef modelColourColumn() {
  return {"grpModelColour",
          "Model & Colour",
          {choice(kPModel, "Model", "ViTMatte Small (faster) or Base (about 1.2x slower).", {"Small", "Base"}, 0,
                  kNoAnimate | kModelOptions),
           choice(kPPicColour, "Picture Colour",
                  "Auto: Flame converts the picture to Rec.709 display for the steps that read it. Bypass: the "
                  "picture is used as it arrives.",
                  {"Auto", "Bypass"}, kColourAuto, kNoAnimate | kClipPrefSlave)}};
}

ColumnDef supportColumn() {
  ParamDef nodeId{kPNodeId, "Node Id", "", Kind::HiddenString};
  nodeId.flags = kNoEvaluate;
  ParamDef rev{kPRecipeRev, "Recipe Rev", "", Kind::HiddenInt};
  rev.flags = kNoEvaluate;
  return {"grpSupport",
          "Support",
          {boolean(kPDebugLog, "Debug Logging", "Verbose logging for troubleshooting (per user).", false,
                   kNotPersistent | kNoEvaluate),
           ParamDef{kPCollectLogs, "Collect Logs", "Write a support bundle (logs and system details) to your home folder.",
                    Kind::PushButton},
           ParamDef{kPSaveMatte, "Save Matte", "Save this frame's input matte for a support case.", Kind::PushButton},
           nodeId, rev}};
}

const std::vector<std::string> kHairLocksViews = {"Result Matte", "Picture", "Changes", "Region",
                                                  "Jumps",        "Candidates", "Rebuilt", "Trimap"};

}  // namespace

const RecipeDef& HairLocksTag::def() {
  static const RecipeDef d{
      "hairlocks",
      "Hair Locks",
      1,
      0,
      1,
      {{"Hair Locks",
        {matteColumn(),
         {"grpHairLocks",
          "Hair Locks",
          {number("rgFeather", "Feather %",
                  "Soft edge of the Focus Region, in percent of the picture height. Outside the region nothing changes.",
                  0.25, 0.0, 5.0, 2, 0.05),
           boolean("emEnable", "Edge Model", "Re-solve the hair inside the Focus Region with the ViTMatte model.", true),
           number("emGain", "Strand Gain", "How strongly faint strands the model finds are lifted. 1 = as found.", 1.0,
                  0.0, 2.0, 2, 0.05),
           boolean("rbEnable", "Rebuild Detail",
                   "Creates matte from the picture: traces thin strands the matte lost inside the Focus Region.", false),
           number("rbMinConf", "Min Confidence", "Rebuilt strands below this confidence (0-1) are dropped.", 0.45, 0.0,
                  1.0, 2, 0.05)}},
         {"grpHairLocks2",
          "Hair Locks 2",
          {number("tmStrength", "Temporal Amount", "0 = the input matte, 1 = the neighbours' combined edge.", 1.0, 0.0,
                  1.0, 2, 0.05),
           number("rgSnapStrength", "Snap Amount", "How strongly new pixels snap to the picture's edges (0-1).", 1.0, 0.0,
                  1.0, 2, 0.05),
           choice("rgSnapMode", "Snap Mode",
                  "TEMPORARY test control. Limited: Snap only raises, drops islands. Off: no Snap.",
                  {"Limited", "Off"}, 0, kNoAnimate)}},
         planColumn(),
         modelColourColumn(),
         supportColumn()}},
       {"Advanced",
        {{"grpJumpRepair",
          "Jump Repair",
          {boolean("jrEnable", "Enable",
                   "Remove short jumps of the whole matte: a patch that appears or vanishes for 1-3 frames.", false),
           number("jrThreshold", "Jump Above", "How far (0-1) the matte must jump against both neighbour frames.", 0.5,
                  0.05, 1.0, 2, 0.05),
           number("jrMinSize", "Min Size %", "Jumps smaller than this percentage of the frame are left alone.", 0.0125,
                  0.0, 1.0, 4, 0.0025),
           number("jrSpanFrames", "Span", "How many frames a jump may last (1-3).", 1.0, 1.0, 3.0, 0, 1.0),
           boolean("jrGuards", "Motion Guards", "On: patches that move like a real object are kept.", true)}},
         {"grpFocusRegion",
          "Focus Region",
          {number("rgGrow", "Region Grow %", "Grow the Focus Region by this percentage of the picture height.", 0.0, 0.0,
                  5.0, 2, 0.05),
           boolean("rgSnap", "Snap to Picture", "Snap the new pixels inside the region to the picture's edges.", true)}},
         {"grpEdgeModel",
          "Edge Model",
          {choice("emLowContrast", "Low Contrast",
                  "Auto: stretch the contrast of a flat region before the model. Off: never.", {"Auto", "Off"}, 0,
                  kNoAnimate)}},
         {"grpRebuild",
          "Rebuild Detail",
          {boolean("rbKeepLoose", "Loose Pieces", "Keep rebuilt pieces that do not touch the matte.", true),
           boolean("rbVote", "Temporal Vote", "Keep a rebuilt strand only when the neighbour frames agree.", true),
           boolean("rbBlend", "Strand Blend", "TEMPORARY test switch. On: a strand's alpha is blended over the neighbour frames.", true),
           number("rbMinPiece", "Min Piece px", "Rebuilt pieces shorter than this are dropped (at 1080 lines).", 12.0, 1.0,
                  100.0, 0, 1.0)}},
         {"grpTemporal",
          "Temporal",
          {boolean("tmEnable", "Enable",
                   "Calm the edge flicker outside the Focus Region with the neighbour frames.", true),
           choice("tmFrames", "Frames", "Neighbour frames on each side.", {"+-1 frame", "+-2 frames"}, 0, kNoAnimate),
           number("tmBand", "Band px", "Only pixels this close to the matte's edge change.", 6.0, 1.0, 32.0, 1,
                  0.5)}}}}},
      kHairLocksViews,
      true,
      true,
      false};
  return d;
}

const RecipeDef& TestDummy::def() {
  static const RecipeDef d{"testdummy",
                           "Test Dummy",
                           2,
                           3,
                           7,
                           {{"Test Dummy", {matteColumn(), planColumn(), modelColourColumn(), supportColumn()}}},
                           {"Result Matte", "Picture"},
                           false,
                           false,
                           false};
  return d;
}

}  // namespace recipes
}  // namespace mr
