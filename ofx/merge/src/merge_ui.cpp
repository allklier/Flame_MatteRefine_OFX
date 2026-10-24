#include "merge_ui.h"

#include <cstdio>

#include "ofxColour.h"  // colour management style names and kOfxColourspaceRaw / OfxRaw (OpenFX 1.5)

namespace mr {
namespace merge {

const char* const kViewNames[6] = {"Result Matte", "Picture", "Added", "Rejected", "Zone", "Residual"};
const char* const kSourceNames[2] = {"Matte Edge", "Trimap"};
const char* const kAlignNames[2] = {"Auto", "Off"};
const char* const kMethodNames[2] = {"Add Fine Detail", "Zone Blend"};
const char* const kEvidenceNames[2] = {"Off", "On"};

const char* const kNoDetailFooter = "Connect the detail matte to the Detail input.";

MergeParams toMergeParams(const UiValues& v, int h) {
  MergeParams p;
  p.source = v.source == 1 ? Source::Trimap : Source::MatteEdge;
  p.method = v.method == 1 ? Method::ZoneBlend : Method::AddFineDetail;
  p.strength = v.strength;
  p.align = v.align != 1;
  p.evidence = v.evidence == 1;
  p.keepDisconnected = v.keepDisconnected;
  HeightSizes s;
  s.reachPct = v.reachPct;
  s.detailWidthPct = v.detailWidthPct;
  s.featherPct = v.featherPct;
  s.bridgePct = v.bridgePct;
  scaleToHeight(p, s, h);
  return p;
}

int clampView(int view) { return view >= kViewResult && view <= kViewResidual ? view : kViewResult; }

std::string constantBaseFooter(float value) {
  char b[200];
  std::snprintf(b, sizeof b,
                "The source alpha is constant %.2f: connect the base matte or trimap to the node's second "
                "socket (source alpha).",
                value);
  return b;
}

std::string noAlphaFooter() {
  return "The picture has no alpha: connect the base matte or trimap to the node's second socket (source alpha).";
}

std::string sizeMismatchFooter(int dw, int dh, int w, int h) {
  char b[200];
  std::snprintf(b, sizeof b, "The detail matte is %dx%d, the picture %dx%d: the base matte passes through.", dw, dh,
                w, h);
  return b;
}

std::string failedFooter(const std::string& what) {
  return "Matte Merge failed (" + what + "): the base matte passes through.";
}

std::string shortCount(long n) {
  char b[32];
  if (n < 10000)
    std::snprintf(b, sizeof b, "%ld", n);
  else if (n < 1000000)
    std::snprintf(b, sizeof b, "%.1fk", n / 1000.0);
  else
    std::snprintf(b, sizeof b, "%.2fM", n / 1e6);
  return b;
}

std::string infoFooter(const MergeInfo& info) {
  // None = Align Auto found no usable estimate: said plainly (the detail is used as given).
  std::string a = info.align.model == AlignModel::None ? std::string("Align: no alignment (could not register)")
                                                       : std::string("Align: ") + alignModelName(info.align.model);
  if (info.align.model != AlignModel::Off && info.align.model != AlignModel::None) {
    char b[48];
    std::snprintf(b, sizeof b, ", +-%.2f px", info.align.uncertaintyPx);
    a += b;
  }
  return a + ". Detail: " + shortCount(info.keptPixels) + " px kept, " + shortCount(info.droppedPixels) +
         " dropped.";
}

std::string outputColourspaceAnswer(bool viewPicture, bool sourceConnected, const std::string& style,
                                    bool configHasMatte) {
  if (viewPicture && sourceConnected) return "OfxColourspace_Source";
  if (style == kOfxImageEffectColourManagementOCIO) return configHasMatte ? "Matte" : kOfxColourspaceRaw;
  if (style == kOfxImageEffectColourManagementFull || style == kOfxImageEffectColourManagementCore)
    return kOfxColourspaceRaw;
  if (style == kOfxImageEffectColourManagementBasic) return kOfxColourspaceOfxRaw;
  return "";
}

}  // namespace merge
}  // namespace mr
