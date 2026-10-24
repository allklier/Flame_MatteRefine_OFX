// Matte Refine OFX (CLAUDE.md): one node, matte operations as sections (decision 1), all
// rendered inline in render(), no cache, no Process button (decision 14).
// - A filter: the picture on the front (Source, mandatory), the matte in its alpha (Flame's second
//   socket, source alpha; decision 4, as Matte Refine ML's decision 35); enabled ops run in a fixed
//   order (3).
// - Output: RGB = source (or the Show Filled view, 11), alpha = the refined matte.
// - Layout follows Flame's rules: one group = one column of <= 5 rows (CLAUDE.md "Flame host").

#include <sys/stat.h>
#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include "env_info.h"
#include "log.h"
#include "ocio_encoding.h"
#include "ofxColour.h"
#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"
#include "ops/analyze_frame.h"
#include "ops/clean_background.h"
#include "ops/contour_smooth.h"
#include "ops/edge_field.h"
#include "ops/edge_width.h"
#include "ops/fill_holes.h"
#include "ops/foreground_colour.h"
#include "ops/shrink_grow.h"
#include "ops/snap_to_image.h"
#include "ops/solid_core.h"
#include "process.h"
#include "user_config.h"

#ifndef MR_VERSION  // from CMakeLists.txt
#define MR_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif
#ifndef MR_VERSION_MAJOR
#define MR_VERSION_MAJOR 0
#define MR_VERSION_MINOR 0
#endif

namespace {

using namespace mr;

constexpr const char* kPluginId = "com.janklier.matterefine";
constexpr const char* kPluginLabel = "Matte Refine";
constexpr const char* kPView = "view";
constexpr const char* kPView2 = "view2";  // the same View on tab 2, kept equal (decision 23)
constexpr const char* kPFhEnable = "fhEnable";
constexpr const char* kPFhThreshold = "fhThreshold";
constexpr const char* kPFhMaxHole = "fhMaxHole";
constexpr const char* kPFhBridge = "fhBridge";
constexpr const char* kPFhRim = "fhRim";
constexpr const char* kPFhClosedLeft = "fhClosedLeft";
constexpr const char* kPFhClosedRight = "fhClosedRight";
constexpr const char* kPFhClosedTop = "fhClosedTop";
constexpr const char* kPFhClosedBottom = "fhClosedBottom";
constexpr const char* kPScEnable = "scEnable";
constexpr const char* kPScDepth = "scDepth";
constexpr const char* kPScEdgeAt = "scEdgeAt";
constexpr const char* kPScFeather = "scFeather";
constexpr const char* kPCbEnable = "cbEnable";
constexpr const char* kPCbMaxIsland = "cbMaxIsland";
constexpr const char* kPCbBelow = "cbBackgroundBelow";
constexpr const char* kPCbCrush = "cbCrush";
constexpr const char* kPCbFeather = "cbCrushFeather";
constexpr const char* kPCsEnable = "csEnable";
constexpr const char* kPCsSmooth = "csSmooth";
constexpr const char* kPCsMaxShift = "csMaxShift";
constexpr const char* kPCsCorner = "csCorner";
constexpr const char* kPCsEdgeAt = "csEdgeAt";
constexpr const char* kPSgEnable = "sgEnable";
constexpr const char* kPSgAmount = "sgAmount";
constexpr const char* kPEwEnable = "ewEnable";
constexpr const char* kPEwWidth = "ewWidth";
constexpr const char* kPEwMix = "ewMix";
constexpr const char* kPSnEnable = "snEnable";
constexpr const char* kPSnRadius = "snRadius";
constexpr const char* kPSnBand = "snBand";
constexpr const char* kPSnDetail = "snDetail";
constexpr const char* kPSnFromPicture = "snFromPicture";
constexpr const char* kPFgEnable = "fgEnable";
constexpr const char* kPFgSmoothness = "fgSmoothness";
constexpr const char* kPFgExtend = "fgExtend";
constexpr const char* kPFgIgnore = "fgIgnore";
constexpr const char* kPFgMix = "fgMix";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPSaveMatte = "saveMatte";
constexpr const char* kPAnalyze = "analyzeFrame";
constexpr const char* kPNodeId = "nodeId";

// Stored as indexes in setups: append only. 0 was "Result", then "Source" (the front input);
// 1 was "Show Filled" (raised pixels only).
enum View { kViewPicture = 0, kViewShowChanges = 1, kViewMatte = 2, kViewTrimap = 3, kViewGreyInside = 4 };
const char* const kViewNames[] = {"Picture", "Show Changes", "Matte", "Trimap", "Grey Inside"};

std::string homeDir() {
  const char* h = std::getenv("HOME");
  return h ? h : "/tmp";
}

std::string timeStr(const char* fmt) {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char b[32];
  std::strftime(b, sizeof b, fmt, &tm);
  return b;
}

std::string randomId() {
  std::random_device rd;
  char b[9];
  std::snprintf(b, sizeof b, "%08x", rd());
  return b;
}

HostInfo hostInfo() {
  HostInfo h;
  if (const auto* d = OFX::getImageEffectHostDescription()) {
    h.name = d->hostName;
    h.label = d->hostLabel;
    h.version = d->versionLabel + " (" + std::to_string(d->versionMajor) + "." +
                std::to_string(d->versionMinor) + "." + std::to_string(d->versionMicro) + ")";
  }
  return h;
}

int componentCount(OFX::PixelComponentEnum c) {
  return c == OFX::ePixelComponentRGBA ? 4 : c == OFX::ePixelComponentRGB ? 3 : 1;
}

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

// The input matte (decision 4), top-down (OFX rows are bottom-up), clamped to 0-1 (decision 5).
std::vector<float> extractMatte(const OFX::Image& src) {
  const OfxRectI sb = src.getBounds();
  const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1, sc = componentCount(src.getPixelComponents());
  std::vector<float> matte(size_t(w) * h);
  for (int y = sb.y1; y < sb.y2; ++y) {
    const auto* p = static_cast<const float*>(src.getPixelAddress(sb.x1, y));
    float* m = &matte[size_t(sb.y2 - 1 - y) * w];
    for (int x = 0; x < w; ++x, p += sc) {
      m[x] = clamp01(sc == 1 ? p[0] : sc == 4 ? p[3] : 1.0f);
    }
  }
  return matte;
}

// The picture (the front input), top-down, w*h*3 (decision 4): for Snap to Image and
// Edge Colour Unmix.
std::vector<float> extractRgb(const OFX::Image& src) {
  const OfxRectI sb = src.getBounds();
  const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1, sc = componentCount(src.getPixelComponents());
  std::vector<float> rgb(size_t(w) * h * 3);
  for (int y = sb.y1; y < sb.y2; ++y) {
    const auto* p = static_cast<const float*>(src.getPixelAddress(sb.x1, y));
    float* q = &rgb[size_t(sb.y2 - 1 - y) * w * 3];
    for (int x = 0; x < w; ++x, p += sc, q += 3) {
      q[0] = p[0];
      q[1] = sc >= 3 ? p[1] : p[0];
      q[2] = sc >= 3 ? p[2] : p[0];
    }
  }
  return rgb;
}

class MatteRefineEffect : public OFX::ImageEffect {
 public:
  explicit MatteRefineEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    view_ = fetchChoiceParam(kPView);
    view2_ = fetchChoiceParam(kPView2);
    fhEnable_ = fetchBooleanParam(kPFhEnable);
    fhThreshold_ = fetchDoubleParam(kPFhThreshold);
    fhMaxHole_ = fetchDoubleParam(kPFhMaxHole);
    fhBridge_ = fetchDoubleParam(kPFhBridge);
    fhRim_ = fetchDoubleParam(kPFhRim);
    fhClosedLeft_ = fetchBooleanParam(kPFhClosedLeft);
    fhClosedRight_ = fetchBooleanParam(kPFhClosedRight);
    fhClosedTop_ = fetchBooleanParam(kPFhClosedTop);
    fhClosedBottom_ = fetchBooleanParam(kPFhClosedBottom);
    scEnable_ = fetchBooleanParam(kPScEnable);
    scDepth_ = fetchDoubleParam(kPScDepth);
    scEdgeAt_ = fetchDoubleParam(kPScEdgeAt);
    scFeather_ = fetchDoubleParam(kPScFeather);
    cbEnable_ = fetchBooleanParam(kPCbEnable);
    cbMaxIsland_ = fetchDoubleParam(kPCbMaxIsland);
    cbBelow_ = fetchDoubleParam(kPCbBelow);
    cbCrush_ = fetchDoubleParam(kPCbCrush);
    cbFeather_ = fetchDoubleParam(kPCbFeather);
    csEnable_ = fetchBooleanParam(kPCsEnable);
    csSmooth_ = fetchDoubleParam(kPCsSmooth);
    csMaxShift_ = fetchDoubleParam(kPCsMaxShift);
    csCorner_ = fetchDoubleParam(kPCsCorner);
    csEdgeAt_ = fetchDoubleParam(kPCsEdgeAt);
    sgEnable_ = fetchBooleanParam(kPSgEnable);
    sgAmount_ = fetchDoubleParam(kPSgAmount);
    ewEnable_ = fetchBooleanParam(kPEwEnable);
    ewWidth_ = fetchDoubleParam(kPEwWidth);
    ewMix_ = fetchDoubleParam(kPEwMix);
    snEnable_ = fetchBooleanParam(kPSnEnable);
    snRadius_ = fetchDoubleParam(kPSnRadius);
    snBand_ = fetchDoubleParam(kPSnBand);
    snDetail_ = fetchDoubleParam(kPSnDetail);
    snFromPicture_ = fetchDoubleParam(kPSnFromPicture);
    fgEnable_ = fetchBooleanParam(kPFgEnable);
    fgSmoothness_ = fetchDoubleParam(kPFgSmoothness);
    fgExtend_ = fetchDoubleParam(kPFgExtend);
    fgIgnore_ = fetchDoubleParam(kPFgIgnore);
    fgMix_ = fetchDoubleParam(kPFgMix);
    debugLog_ = fetchBooleanParam(kPDebugLog);
    OFX::StringParam* nodeIdParam = fetchStringParam(kPNodeId);
    try {
      nodeIdParam->getValue(nodeId_);
      if (nodeId_.empty()) {
        nodeId_ = randomId();  // saved with the setup; copies of a node share it
        nodeIdParam->setValue(nodeId_);
      }
      LogScope scope(nodeId_);
      LOGI("createInstance %p context=%d", static_cast<void*>(this), int(getContext()));
      debugLog_->setValue(logLevel() == LogLevel::Debug);  // per user, not stored in setups
      int v = 0, v2 = 0;
      view_->getValue(v);
      view2_->getValue(v2);
      if (v2 != v) view2_->setValue(v);
    } catch (const std::exception& e) {
      LOGE("createInstance: setting param values FAILED: %s [%s]", e.what(), typeid(e).name());
    }
  }

  ~MatteRefineEffect() override {
    LogScope scope(nodeId_);
    LOGI("destroyInstance %p", static_cast<void*>(this));
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    // Alpha carries the result, also for an RGB source (Flame forces RGBA anyway).
    // Output size and PAR: the host's filter default, Source's.
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    // Flame reports our own setValue() calls back synchronously as changedParam (reason =
    // plugin edit); reacting to those recursed until Flame crashed (LTX Flame test 2). Only
    // user edits are handled, and never re-entrantly.
    if (args.reason != OFX::eChangeUserEdit || inChangedParam_) return;
    struct Guard {
      bool& b;
      explicit Guard(bool& b) : b(b) { b = true; }
      ~Guard() { b = false; }
    } guard(inChangedParam_);
    LogScope scope(nodeId_);
    LOGI("changedParam %s (user) time=%g", name.c_str(), args.time);
    if (name == kPCollectLogs) {
      collectLogs();
    } else if (name == kPSaveMatte) {
      saveMatte(args.time);
    } else if (name == kPAnalyze) {
      analyzeFrameButton(args.time, args.renderScale.x);
    } else if (name == kPView || name == kPView2) {  // keep both tabs' View equal
      int v = 0;
      (name == kPView ? view_ : view2_)->getValue(v);
      (name == kPView ? view2_ : view_)->setValue(v);
    } else if (name == kPDebugLog) {
      bool on = false;
      debugLog_->getValue(on);
      UserConfig cfg = loadUserConfig();
      cfg["log_level"] = on ? "debug" : "info";
      const bool ok = saveUserConfig(cfg);
      setLogLevel(on ? LogLevel::Debug : LogLevel::Info);
      LOGI("debug logging %s (config %s)", on ? "on" : "off", ok ? "saved" : "NOT saved");
    }
  }

  void render(const OFX::RenderArguments& args) override {
    LogScope scope(nodeId_);
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<OFX::Image> out(dst_->fetchImage(args.time));
    if (!out) {
      LOGE("render t=%g: host returned no output image", args.time);
      OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    // A filter (decision 4, as Matte Refine ML's 35): the picture is Source, the matte its alpha.
    std::string warning;
    std::unique_ptr<OFX::Image> src(src_->isConnected() ? src_->fetchImage(args.time) : nullptr);
    if (src && src->getPixelDepth() != OFX::eBitDepthFloat) {
      warning = "Only 32-bit float images are supported.";
      src.reset();
    }
    const OFX::Image* mi = src.get();
    if (!mi && warning.empty()) warning = "No picture.";

    const OfxRectI sb = mi ? mi->getBounds() : out->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    const int sc = src ? componentCount(src->getPixelComponents()) : 0;
    int view = kViewPicture;
    view_->getValueAtTime(args.time, view);
    if (src && sc == 3)
      warning = "The picture has no alpha, so there is no matte: connect the matte to the node's second "
                "socket (source alpha).";

    std::vector<float> matte = mi ? extractMatte(*mi) : std::vector<float>(size_t(w) * h, 0.0f);
    std::string inLog;  // what actually arrived
    if (mi && !matte.empty()) {
      const auto mm = std::minmax_element(matte.begin(), matte.end());
      if (*mm.first == *mm.second && warning.empty()) {
        char b[160];
        // Flame delivers alpha 1.0 when nothing is on the source alpha input (probe, 2026-10-22).
        std::snprintf(b, sizeof b,
                      "The input matte is constant %.2f: no matte on the source alpha. Connect the "
                      "matte to the node's second socket (source alpha).",
                      *mm.first);
        warning = b;
      }
      if (logEnabled(LogLevel::Debug)) {
        double sum = 0.0;
        for (float v : matte) sum += v;
        char b[120];
        std::snprintf(b, sizeof b, " src %d ch, matte min %.3f mean %.3f max %.3f,", sc, *mm.first,
                      sum / matte.size(), *mm.second);
        inLog = b;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // The operations, in their fixed order (decision 3).
    std::vector<float> result = matte;
    std::string opLog;
    const double scale = args.renderScale.x;  // pixel params scale (decision 6)
    auto on = [&](OFX::BooleanParam* b) {
      bool v = false;
      b->getValueAtTime(args.time, v);
      return mi && v;
    };
    auto val = [&](OFX::DoubleParam* d) { return d->getValueAtTime(args.time); };
    auto since = [](std::chrono::steady_clock::time_point t) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    if (on(cbEnable_)) {
      CleanBackgroundParams p;
      p.maxIslandPercent = val(cbMaxIsland_);
      p.backgroundBelow = float(val(cbBelow_));
      p.crushPx = val(cbCrush_) * scale;
      p.featherPx = val(cbFeather_) * scale;
      p.fringePx *= scale;
      CleanBackgroundStats st;
      const auto c0 = std::chrono::steady_clock::now();
      cleanBackground(result.data(), result.data(), w, h, p, &st);
      char b[200];
      std::snprintf(b, sizeof b, " clean [island=%.3g%% below=%.3g crush=%.3g/%.3g] %.1f ms (%d islands, %lld px removed, %lld crushed)",
                    p.maxIslandPercent, p.backgroundBelow, p.crushPx, p.featherPx, since(c0),
                    st.islandsRemoved, st.pixelsRemoved, st.pixelsCrushed);
      opLog += b;
    }
    bool fhOn = true;
    fhEnable_->getValueAtTime(args.time, fhOn);
    if (mi && fhOn) {
      FillHolesParams p;
      p.threshold = float(fhThreshold_->getValueAtTime(args.time));
      p.maxHolePercent = fhMaxHole_->getValueAtTime(args.time);
      p.bridgePx = fhBridge_->getValueAtTime(args.time) * args.renderScale.x;  // decision 6
      p.rimPx = fhRim_->getValueAtTime(args.time) * args.renderScale.x;
      p.closedLeft = fhClosedLeft_->getValueAtTime(args.time);
      p.closedRight = fhClosedRight_->getValueAtTime(args.time);
      p.closedTop = fhClosedTop_->getValueAtTime(args.time);
      p.closedBottom = fhClosedBottom_->getValueAtTime(args.time);
      FillHolesStats st;
      const auto f0 = std::chrono::steady_clock::now();
      fillHoles(result.data(), result.data(), w, h, p, &st);
      char b[400];
      std::snprintf(b, sizeof b,
                    " fillHoles [thr=%.3g max=%.3g%% bridge=%.3g rim=%.3g closed=%s%s%s%s] %.1f ms "
                    "(%d filled, %d too large, %d notches kept, %lld px changed; %.2f%% below "
                    "thr, %.2f%% outside)",
                    p.threshold, p.maxHolePercent, p.bridgePx, p.rimPx, p.closedLeft ? "L" : "",
                    p.closedRight ? "R" : "", p.closedTop ? "T" : "", p.closedBottom ? "B" : "",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count(),
                    st.holesFilled, st.holesTooLarge, st.notchesKept, st.pixelsChanged,
                    100.0 * st.belowThreshold / std::max<size_t>(matte.size(), 1),
                    100.0 * st.outsidePixels / std::max<size_t>(matte.size(), 1));
      opLog += b;
    }
    bool scOn = false;
    scEnable_->getValueAtTime(args.time, scOn);
    if (mi && scOn) {
      SolidCoreParams p;
      p.depthPx = scDepth_->getValueAtTime(args.time) * args.renderScale.x;  // decision 6
      p.edgeAt = float(scEdgeAt_->getValueAtTime(args.time));
      p.featherPx = scFeather_->getValueAtTime(args.time) * args.renderScale.x;
      SolidCoreStats st;
      const auto c0 = std::chrono::steady_clock::now();
      solidCore(result.data(), result.data(), w, h, p, &st);
      char b[160];
      std::snprintf(b, sizeof b,
                    " solidCore [depth=%.3g feather=%.3g edge=%.3g] %.1f ms (%lld px changed)",
                    p.depthPx, p.featherPx, p.edgeAt,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count(),
                    st.pixelsChanged);
      opLog += b;
    }
    if (on(csEnable_)) {
      ContourSmoothParams p;
      p.smoothPx = val(csSmooth_) * scale;
      p.maxShiftPx = val(csMaxShift_) * scale;
      p.cornerDeg = val(csCorner_);
      p.edgeAt = float(val(csEdgeAt_));
      ContourSmoothStats st;
      const auto c0 = std::chrono::steady_clock::now();
      contourSmooth(result.data(), result.data(), w, h, p, &st);
      char b[200];
      std::snprintf(b, sizeof b, " smooth [%.3g px, max %.3g, corner %.3g, edge %.3g] %.1f ms (%lld px changed%s)",
                    p.smoothPx, p.maxShiftPx, p.cornerDeg, p.edgeAt, since(c0), st.pixelsChanged,
                    st.skipped ? ", SKIPPED: noisy matte" : "");
      opLog += b;
    }
    double growPx = 0.0;  // the matte was grown this much (user units): Edge Colour Unmix ignores it
    if (on(sgEnable_)) {
      ShrinkGrowParams p;
      p.amountPx = val(sgAmount_) * scale;
      growPx = std::max(0.0, val(sgAmount_));
      ShrinkGrowStats st;
      const auto c0 = std::chrono::steady_clock::now();
      shrinkGrow(result.data(), result.data(), w, h, p, &st);
      char b[160];
      std::snprintf(b, sizeof b, " shrinkGrow [%.3g px] %.1f ms (%d passes, %lld px changed)",
                    p.amountPx, since(c0), st.passes, st.pixelsChanged);
      opLog += b;
    }
    if (on(ewEnable_)) {
      EdgeWidthParams p;
      p.widthPx = val(ewWidth_) * scale;
      p.amount = val(ewMix_);
      EdgeWidthStats st;
      const auto c0 = std::chrono::steady_clock::now();
      edgeWidth(result.data(), result.data(), w, h, p, &st);
      char b[160];
      std::snprintf(b, sizeof b, " edgeWidth [%.3g px, mix %.3g] %.1f ms (%lld px changed)",
                    p.widthPx, p.amount, since(c0), st.pixelsChanged);
      opLog += b;
    }
    const bool snOn = on(snEnable_), fgOn = on(fgEnable_);  // on() is false without Source
    const std::vector<float> rgb = snOn || fgOn ? extractRgb(*src) : std::vector<float>();
    if (snOn) {
      SnapParams p;
      p.radiusPx = val(snRadius_) * scale;
      p.bandPx = val(snBand_) * scale;
      p.detail = val(snDetail_);
      p.fromPicture = val(snFromPicture_);
      SnapStats st;
      const auto c0 = std::chrono::steady_clock::now();
      snapToImage(result.data(), rgb.data(), result.data(), w, h, p, &st);
      char b[200];
      std::snprintf(b, sizeof b, " snap [r %.3g, band %.3g, detail %.3g, picture %.3g] %.1f ms (%lld px changed, max %.3f)",
                    p.radiusPx, p.bandPx, p.detail, p.fromPicture, since(c0), st.pixelsChanged, st.maxChange);
      opLog += b;
    }
    // Edge Colour Unmix (foreground_colour) changes the picture, not the matte: View Picture.
    std::vector<float> fg;
    bool error = false;
    if (fgOn) {
      const std::string problem = fgColourspaceProblem();
      if (!problem.empty()) {
        warning = problem;
        error = true;
      }
      FgParams p;
      p.smoothness = val(fgSmoothness_);
      p.extendPx = val(fgExtend_) * scale;
      // Decision 26: a grown matte holds old-background pixels as "solid": ignore that much more.
      p.ignoreOuterPx = (val(fgIgnore_) + growPx) * scale;
      p.mix = float(val(fgMix_));
      FgStats st;
      fg.resize(rgb.size());
      const auto c0 = std::chrono::steady_clock::now();
      foregroundColour(rgb.data(), result.data(), fg.data(), w, h, p, &st);
      char b[160];
      std::snprintf(b, sizeof b,
                    " fgColour [smooth %.3g, extend %.3g, ignore %.3g, mix %.3g] %.1f ms (%lld edge px, %lld ignored)",
                    p.smoothness, p.extendPx, p.ignoreOuterPx, p.mix, since(c0), st.edgePixels,
                    st.ignoredPixels);
      opLog += b;
    }
    // Grey Inside: semi-transparent pixels more than 4 px inside the 0.5 contour.
    const std::vector<float> sd = view == kViewGreyInside && mi
                                      ? signedEdgeDistance(result.data(), w, h, 0.5f, 6.0)
                                      : std::vector<float>();

    // Output (decision 11): RGB per View, A = the refined matte.
    const OfxRectI db = out->getBounds();
    const OfxRectI& win = args.renderWindow;
    const int dc = componentCount(dst_->getPixelComponents());
    for (int y = std::max(win.y1, db.y1); y < std::min(win.y2, db.y2); ++y) {
      auto* px = static_cast<float*>(out->getPixelAddress(std::max(win.x1, db.x1), y));
      const int row = sb.y2 - 1 - y;
      for (int x = std::max(win.x1, db.x1); x < std::min(win.x2, db.x2); ++x, px += dc) {
        const int col = x - sb.x1;
        const bool inside = mi && row >= 0 && row < h && col >= 0 && col < w;
        const size_t i = inside ? size_t(row) * w + col : 0;
        const float a = inside ? result[i] : 0.0f;
        if (dc == 1) {
          px[0] = a;
          continue;
        }
        if (!inside) {
          px[0] = px[1] = px[2] = 0.0f;
        } else if (view == kViewShowChanges) {
          // The input matte in grey; raised pixels tinted magenta, lowered ones green.
          const float m = matte[i], d = a - m;
          if (d >= 0.0f) {
            px[0] = px[2] = m * (1.0f - d) + d;
            px[1] = m * (1.0f - d);
          } else {
            px[0] = px[2] = a;
            px[1] = std::min(1.0f, a - d);
          }
        } else if (view == kViewMatte) {
          px[0] = px[1] = px[2] = a;
        } else if (view == kViewTrimap) {
          px[0] = px[1] = px[2] = a >= 0.98f ? 1.0f : a <= 0.02f ? 0.0f : 0.5f;
        } else if (view == kViewGreyInside) {
          const bool grey = a > 0.02f && a < 0.98f && sd[i] > 4.0f;
          px[0] = grey ? 1.0f : a;
          px[1] = px[2] = grey ? 0.15f : a;
        } else if (!fg.empty()) {
          px[0] = fg[i * 3];
          px[1] = fg[i * 3 + 1];
          px[2] = fg[i * 3 + 2];
        } else {
          const auto* sp = static_cast<const float*>(src->getPixelAddress(x, y));
          px[0] = sp[0];
          px[1] = sc >= 3 ? sp[1] : sp[0];
          px[2] = sc >= 3 ? sp[2] : sp[0];
        }
        if (dc == 4) px[3] = a;
      }
    }
    try {
      if (warning.empty())
        clearPersistentMessage();
      else
        setPersistentMessage(error ? OFX::Message::eMessageError : OFX::Message::eMessageWarning, "",
                             warning);
    } catch (const std::exception& e) {
      LOGW("render: persistent message FAILED: %s", e.what());
    }
    const auto t2 = std::chrono::steady_clock::now();
    LOGD("render t=%g %dx%d scale=%g view=%s:%s read %.1f ms,%s total %.1f ms%s%s",
         args.time, w, h, args.renderScale.x, kViewNames[std::min(std::max(view, 0), 4)], inLog.c_str(),
         std::chrono::duration<double, std::milli>(t1 - t0).count(), opLog.c_str(),
         std::chrono::duration<double, std::milli>(t2 - t0).count(), warning.empty() ? "" : " | ",
         warning.c_str());
  }

 private:
  // Writes this frame's input matte (what the ops see) for offline analysis on this machine:
  // ~/.cache/matte-refine/dumps/<node>_t<time>.f32 (w*h floats, top row first) + .txt.
  // Analyze Frame (roadmap, 2026-10-23): analyse this frame's input matte and set Clean & Crush,
  // Fill Holes and Solid Core for it; the popup says what was chosen and why. Pixel values come
  // back at the frame's own resolution and are stored in full-resolution units (decision 6).
  void analyzeFrameButton(double time, double scale) {
    std::unique_ptr<OFX::Image> img(src_->fetchImage(time));
    if (!img || img->getPixelDepth() != OFX::eBitDepthFloat) {
      sendMessage(OFX::Message::eMessageError, "", "Analyze Frame: no 32-bit float matte.");
      return;
    }
    const OfxRectI b = img->getBounds();
    const int w = b.x2 - b.x1, h = b.y2 - b.y1;
    const std::vector<float> matte = extractMatte(*img);
    AnalysisResult r;
    analyzeFrame(matte.data(), w, h, &r);
    std::string text = "Analyze Frame:";
    for (const std::string& n : r.notes) text += "\n- " + n;
    LOGI("analyzeFrame %dx%d: %.0f ms, clean %d fill %d (thr %.3g max %.3g bridge %.3g) core %d%s%s%s%s", w, h,
         r.measured.ms, r.cleanEnable, r.fillEnable, r.fill.threshold, r.fill.maxHolePercent,
         r.fill.bridgePx, r.coreEnable, r.constantMatte ? " [constant]" : "", r.noSubject ? " [no subject]" : "",
         r.noisy ? " [noisy]" : "", r.tooSmall ? " [too small]" : "");
    if (!r.constantMatte && !r.noSubject && !r.noisy && !r.tooSmall) {  // else nothing is changed
      const double sc = scale > 0.0 ? scale : 1.0;
      cbEnable_->setValue(r.cleanEnable);
      cbMaxIsland_->setValue(r.clean.maxIslandPercent);
      cbBelow_->setValue(r.clean.backgroundBelow);
      cbCrush_->setValue(r.clean.crushPx / sc);
      fhEnable_->setValue(r.fillEnable);
      fhThreshold_->setValue(r.fill.threshold);
      fhMaxHole_->setValue(r.fill.maxHolePercent);
      fhBridge_->setValue(r.fill.bridgePx / sc);
      fhRim_->setValue(r.fill.rimPx / sc);
      fhClosedLeft_->setValue(r.fill.closedLeft);
      fhClosedRight_->setValue(r.fill.closedRight);
      fhClosedTop_->setValue(r.fill.closedTop);
      fhClosedBottom_->setValue(r.fill.closedBottom);
      scEnable_->setValue(r.coreEnable);
      scDepth_->setValue(r.core.depthPx / sc);
      scEdgeAt_->setValue(r.core.edgeAt);
      scFeather_->setValue(r.core.featherPx / sc);
    }
    sendMessage(OFX::Message::eMessageMessage, "", text);
  }

  void saveMatte(double time) {
    std::unique_ptr<OFX::Image> src(src_->fetchImage(time));
    if (!src || src->getPixelDepth() != OFX::eBitDepthFloat) {
      sendMessage(OFX::Message::eMessageError, "", "Save Matte: no 32-bit float matte.");
      return;
    }
    const std::vector<float> matte = extractMatte(*src);
    const OfxRectI b = src->getBounds();
    const std::string logs = logDir();
    const std::string dir = logs.substr(0, logs.rfind('/')) + "/dumps";
    mkdir(dir.c_str(), 0700);
    char base[64];
    std::snprintf(base, sizeof base, "/%s_t%g", nodeId_.c_str(), time);
    const std::string path = dir + base;
    std::ofstream raw(path + ".f32", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(matte.data()), std::streamsize(matte.size() * sizeof(float)));
    std::ofstream txt(path + ".txt");
    txt << "width=" << (b.x2 - b.x1) << "\nheight=" << (b.y2 - b.y1)
        << "\nformat=float32 little-endian, top row first, 0-1\nmatte_from=source alpha"
        << "\ntime=" << time << "\n";
    const bool ok = raw.good() && txt.good();
    LOGI("saveMatte t=%g %dx%d -> %s.f32 %s", time, b.x2 - b.x1, b.y2 - b.y1, path.c_str(),
         ok ? "ok" : "FAILED");
    sendMessage(ok ? OFX::Message::eMessageMessage : OFX::Message::eMessageError, "",
                ok ? "Matte saved to " + path + ".f32 (stays on this machine)."
                   : "Could not write " + path + ".f32");
  }

  // Bundles the logs and an environment snapshot for a support request.
  void collectLogs() {
    const std::string dir = logDir();
    {
      std::ofstream envf(dir + "/environment.txt");
      for (const auto& line : environmentSnapshot(hostInfo())) envf << line << "\n";
    }
    const std::string bundle =
        homeDir() + "/matte-refine-support-" + timeStr("%Y%m%d-%H%M%S") + ".tar.gz";
    const std::vector<std::string> argv = {"/usr/bin/tar", "-czf", bundle, "-C",
                                           dir.substr(0, dir.rfind('/')), "logs"};
    std::string err;
    WorkerProcess tar;
    std::vector<std::string> lines;
    const bool started = tar.start(argv, cleanEnvironment(), err);
    if (started)
      while (tar.readLines(200, lines)) {
      }
    const int status = started ? tar.wait() : -1;
    const bool ok = started && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    for (const auto& l : lines) LOGW("collectLogs: tar: %s", l.c_str());
    LOGI("collectLogs: %s -> %s %s", ok ? "ok" : "FAILED", bundle.c_str(), err.c_str());
    sendMessage(ok ? OFX::Message::eMessageMessage : OFX::Message::eMessageError, "",
                ok ? "Support bundle written to " + bundle +
                         ". Please review it before sending: it contains your logs, system "
                         "details and folder paths, but no images."
                   : "Could not write the support bundle: " + err);
  }

  OFX::Clip* src_;
  OFX::Clip* dst_;
  OFX::ChoiceParam* view_;
  OFX::BooleanParam* fhEnable_;
  OFX::DoubleParam* fhThreshold_;
  OFX::DoubleParam* fhMaxHole_;
  OFX::DoubleParam* fhBridge_;
  OFX::DoubleParam* fhRim_;
  OFX::BooleanParam* fhClosedLeft_;
  OFX::BooleanParam* fhClosedRight_;
  OFX::BooleanParam* fhClosedTop_;
  OFX::BooleanParam* fhClosedBottom_;
  OFX::BooleanParam* scEnable_;
  OFX::DoubleParam* scDepth_;
  OFX::DoubleParam* scEdgeAt_;
  OFX::DoubleParam* scFeather_;
  OFX::BooleanParam *cbEnable_, *csEnable_, *sgEnable_, *ewEnable_, *snEnable_, *fgEnable_;
  OFX::DoubleParam *cbMaxIsland_, *cbBelow_, *cbCrush_, *cbFeather_;
  OFX::DoubleParam *csSmooth_, *csMaxShift_, *csCorner_, *csEdgeAt_;
  OFX::DoubleParam* sgAmount_;
  OFX::DoubleParam *ewWidth_, *ewMix_;
  OFX::DoubleParam *snRadius_, *snBand_, *snDetail_, *snFromPicture_;
  OFX::DoubleParam *fgSmoothness_, *fgExtend_, *fgIgnore_, *fgMix_;
  OFX::BooleanParam* debugLog_;
  std::string nodeId_;
  bool inChangedParam_ = false;
  OFX::ChoiceParam* view2_;

 public:
  // Decision 23 (OpenFX 1.5 colour management): the picture passing through keeps the input's
  // colourspace; a matte on RGB is tagged as a matte (the config's "Matte", else "Raw"). Empty
  // = no colour management negotiated: the host's default.
  std::string outputColourspace() {
    int view = kViewPicture;
    view_->getValue(view);
    if (view == kViewPicture && src_->isConnected()) return "OfxColourspace_Source";
    const std::string style =
        getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false);
    if (style == kOfxImageEffectColourManagementOCIO) return configHasMatte() ? "Matte" : "Raw";
    if (style == kOfxImageEffectColourManagementFull || style == kOfxImageEffectColourManagementCore)
      return kOfxColourspaceRaw;
    if (style == kOfxImageEffectColourManagementBasic) return kOfxColourspaceOfxRaw;
    return "";
  }

 private:
  // The active OCIO config's text, read once (no OCIO library; decisions 23, 24).
  const std::string& configText() {
    if (!configRead_) {
      configRead_ = true;
      const std::string path = getPropertySet().propGetString(kOfxImageEffectPropOCIOConfig, false);
      std::ifstream in(path);
      configText_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
      LOGI("colour: OCIO config '%s' (%zu bytes)", path.c_str(), configText_.size());
    }
    return configText_;
  }

  // Flame's configs define a "Matte" colourspace (alias adsk:matte_data, encoding data; 2026.0
  // aces2.0 config); others may not.
  bool configHasMatte() {
    if (!matteChecked_) {
      matteChecked_ = true;
      hasMatte_ = colourspaceEncoding(configText(), "Matte") == "data";
      LOGI("colour: Matte colourspace %s", hasMatte_ ? "found" : "not found (Raw)");
    }
    return hasMatte_;
  }

  // Decision 24: Edge Colour Unmix solves the compositing equation, which holds only in linear
  // light. "" = the picture is scene-linear, else the footer error.
  std::string fgColourspaceProblem() {
    const std::string cs = src_->getPropertySet().propGetString(kOfxImageClipPropColourspace, false);
    if (cs != lastPictureCs_) {
      lastPictureCs_ = cs;
      LOGI("colour: picture colourspace '%s'", cs.c_str());
    }
    if (cs.empty())
      return "Edge Colour Unmix: the picture's colour space is unknown (no colour management); "
             "it must be scene-linear.";
    const std::string enc = colourspaceEncoding(configText(), cs);
    if (enc == "scene-linear") return "";
    return "Edge Colour Unmix needs a scene-linear picture; the front input is " + cs +
           (enc.empty() ? " (not in the colour config)." : " (" + enc + "). Convert it to linear first.");
  }
  std::string configText_, lastPictureCs_;
  bool configRead_ = false;
  bool matteChecked_ = false, hasMatte_ = false;
};

template <class D>
D* labelled(D* p, const char* label, const char* hint) {
  p->setLabels(label, label, label);
  if (hint && *hint) p->setHint(hint);
  return p;
}

class MatteRefineFactory : public OFX::PluginFactoryHelper<MatteRefineFactory> {
 public:
  // The 1.4 Support library doesn't know OpenFX 1.5's GetOutputColourspace action (it answers
  // "not trapped"), so this entry point answers it and passes every other action on.
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in,
                         OfxPropertySetHandle out) {
    if (std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0) {
      try {
        auto* fx = dynamic_cast<MatteRefineEffect*>(
            OFX::Private::retrieveImageEffectPointer((OfxImageEffectHandle)handle));
        const std::string cs = fx ? fx->outputColourspace() : "";
        if (cs.empty()) return kOfxStatReplyDefault;
        OFX::PropertySet(out).propSetString(kOfxImageClipPropColourspace, cs);
        LOGD("colour: output colourspace %s", cs.c_str());
        return kOfxStatOK;
      } catch (const std::exception& e) {
        LOGE("colour: GetOutputColourspace FAILED: %s", e.what());
        return kOfxStatFailed;
      }
    }
    return FactoryMainEntryHelper<MatteRefineFactory>::mainEntry(action, handle, in, out);
  }
  OfxPluginEntryPoint* getMainEntry() override { return entry; }

  MatteRefineFactory()
      : PluginFactoryHelper<MatteRefineFactory>(kPluginId, MR_VERSION_MAJOR, MR_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LOGI("==== plugin load (describe) ====");
    for (const auto& line : environmentSnapshot(hostInfo())) LOGI("env %s", line.c_str());
    desc.setLabels(kPluginLabel, kPluginLabel, kPluginLabel);
    desc.setPluginGrouping("Matte Refine");
    desc.addSupportedContext(OFX::eContextFilter);  // decision 4: as Matte Refine ML (its decision 35)
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);
    // Decision 23: OpenFX 1.5 colour management (Flame 2026.1+); older hosts ignore these.
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementStyle,
                                        kOfxImageEffectColourManagementOCIO, false);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementAvailableConfigs,
                                        "ofx-native-v1.5_aces-v1.3_ocio-v2.3", false);
  }

  void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum ctx) override {
    try {
      describeParams(desc);
      LOGI("describeInContext ctx=%d done", int(ctx));
    } catch (const std::exception& e) {
      LOGE("describeInContext ctx=%d FAILED: %s [%s]", int(ctx), e.what(), typeid(e).name());
      throw;
    }
  }

  void describeParams(OFX::ImageEffectDescriptor& desc) {
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);  // mandatory: the picture, the matte in its alpha (decision 4)
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    // Flame: params flow into columns of <= 5 rows, 6 columns per tab, each group starts a
    // column, pages are ignored (CLAUDE.md "Planned layout"). Groups in column order.
    // Decision 19: tab 1 "Fill & Core", tab 2 "Edge" (Flame opens it when tab 1 is full and
    // names it after the next page).
    OFX::PageParamDescriptor* page = desc.definePageParam("Fill & Core");
    auto group = [&](const char* name, const char* text) {
      auto* g = desc.defineGroupParam(name);
      g->setLabels(text, text, text);
      g->setOpen(true);
      page->addChild(*g);
      return g;
    };
    auto boolean = [&](const char* name, const char* text, const char* hint, bool def,
                       OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineBooleanParam(name), text, hint);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    // Digits and increment: Flame showed the first build's doubles as integers (2026-10-21).
    auto number = [&](const char* name, const char* text, const char* hint, double def,
                      double maxRange, double maxDisplay, int digits, double increment,
                      OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text, hint);
      p->setRange(0.0, maxRange);
      p->setDisplayRange(0.0, maxDisplay);
      p->setDigits(digits);
      p->setIncrement(increment);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };

    // --- Matte (3 rows): view, version. The matte is the source alpha (decision 4). ---
    auto* gMatte = group("grpMatte", "Matte");
    auto* view = labelled(desc.defineChoiceParam(kPView), "View",
                          "What the image output shows (the matte output is always the refined "
                          "matte). Matte: the refined matte. Show Changes: the input matte in "
                          "grey, raised pixels magenta, lowered green. Picture: the front input "
                          "(its edges unmixed with Edge Colour Unmix on). Trimap: "
                          "solid / edge / clear. Grey Inside: semi-transparent pixels deep "
                          "inside the matte, in red.");
    for (const char* c : kViewNames) view->appendOption(c);
    view->setDefault(kViewPicture);  // the picture passes through, the matte refined (as Matte Refine ML)
    view->setAnimates(false);        // decision 23: two Views kept equal, no keyframes
    view->setParent(*gMatte);
    auto* version = labelled(desc.defineStringParam("version"), "Version", "");
    version->setStringType(OFX::eStringTypeLabel);
    version->setDefault(MR_VERSION " (" MR_GIT ")");
    version->setIsPersistant(false);
    version->setEvaluateOnChange(false);
    version->setParent(*gMatte);

    // --- Clean & Crush (5 rows; A1 + A2, runs first) ---
    auto* gCb = group("grpCleanBackground", "Clean & Crush");
    boolean(kPCbEnable, "Enable",
            "Remove small islands outside the subject and crush background haze beyond the edge.",
            false, gCb);
    number(kPCbMaxIsland, "Max Island %",
           "Islands up to this % of the frame area are removed with their fringe. 0 = keep all.",
           0.05, 10.0, 1.0, 3, 0.01, gCb);
    number(kPCbBelow, "Background Below",
           "Matte below this is background or haze; islands and the crush edge are measured at it.",
           0.15, 0.5, 0.5, 3, 0.01, gCb)
        ->setRange(0.01, 0.5);
    number(kPCbCrush, "Crush Beyond px",
           "Everything farther than this outside the edge becomes 0. 0 = no crush.", 4.0, 100.0,
           30.0, 1, 0.5, gCb);
    number(kPCbFeather, "Crush Feather px", "The crush fades in over this many pixels.", 2.0,
           20.0, 10.0, 1, 0.5, gCb);

    labelled(desc.definePushButtonParam(kPAnalyze), "Analyze Frame",
             "Look at this frame's matte and set Clean & Crush, Fill Holes and Solid Core for it; "
             "a message says what was chosen and why. Other operations are not touched.")
        ->setParent(*gMatte);

    // --- Fill Holes (5 rows; decisions 8-10) ---
    auto* gFh = group("grpFillHoles", "Fill Holes");
    boolean(kPFhEnable, "Enable", "Fill holes enclosed by solid matte with white.", true, gFh);
    number(kPFhThreshold, "Solid Above", "Matte at or above this value counts as solid.", 0.95, 1.0,
           1.0, 3, 0.01, gFh);
    number(kPFhMaxHole, "Max Hole %",
           "Holes larger than this percentage of the frame stay open (real gaps, e.g. an arm "
           "on a hip). 0 = fill every enclosed hole.",
           1.0, 100.0, 10.0, 2, 0.1, gFh);
    number(kPFhBridge, "Bridge Gaps px",
           "Also fill holes whose solid edge has a gap up to this many pixels wide. 0 = only "
           "fully enclosed holes.",
           0.0, 200.0, 20.0, 1, 0.5, gFh);
    number(kPFhRim, "Rim px",
           "Also lift the soft rim around a filled hole to white, up to this many pixels into "
           "the solid matte.",
           1.0, 50.0, 5.0, 1, 0.5, gFh);

    // --- Fill Holes Edges (4 rows) ---
    auto* gEdges = group("grpFillHolesEdges", "Fill Holes Edges");
    const char* edgeHint = "Treat this frame edge as solid, e.g. a subject cut by the frame: "
                           "holes reaching it are filled.";
    boolean(kPFhClosedLeft, "Left Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedRight, "Right Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedTop, "Top Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedBottom, "Bottom Closed", edgeHint, false, gEdges);

    // --- Solid Core (4 rows; decisions 15, 16) ---
    auto* gSc = group("grpSolidCore", "Solid Core");
    boolean(kPScEnable, "Enable",
            "Make everything deeper than Depth inside the matte solid white; the outer Depth "
            "pixels of edge are not touched.",
            false, gSc);
    number(kPScDepth, "Depth px",
           "The core starts this many pixels inside the edge. Smaller = more of the soft edge "
           "becomes solid.",
           6.0, 200.0, 50.0, 1, 0.5, gSc);
    number(kPScFeather, "Feather px",
           "The core fades in over this many pixels after Depth, so its edge is smooth. 0 = a "
           "hard core.",
           2.0, 50.0, 10.0, 1, 0.5, gSc);
    // Not 0: nothing is below 0, so the whole frame would be core and turn white (Flame test 3).
    number(kPScEdgeAt, "Edge At", "The matte value where the edge is measured from.", 0.5, 1.0,
           1.0, 3, 0.01, gSc)
        ->setRange(0.01, 1.0);

    // --- Support (3 rows, per user) ---
    auto* gSupport = group("grpSupport", "Support");
    auto* dbg = boolean(kPDebugLog, "Debug Logging",
                        "Verbose logging for troubleshooting (per user).", false, gSupport);
    dbg->setIsPersistant(false);
    dbg->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPCollectLogs), "Collect Logs",
             "Write a support bundle (logs and system details) to your home folder.")
        ->setParent(*gSupport);
    labelled(desc.definePushButtonParam(kPSaveMatte), "Save Matte",
             "Save this frame's input matte to ~/.cache/matte-refine/dumps for analysis. It "
             "stays on this machine.")
        ->setParent(*gSupport);
    // Hidden, but Flame still gives it a slot: defined last outside a group it took a column
    // of its own and, with tab 2 full, a third tab (Flame test, 2026-10-23). In Support (4 rows).
    auto* nid = desc.defineStringParam(kPNodeId);
    nid->setDefault("");
    nid->setIsSecret(true);
    nid->setEvaluateOnChange(false);
    nid->setParent(*gSupport);

    // --- Tab 2 "Edge" (decision 19), in chain order ---
    page = desc.definePageParam("Edge");
    // The Matte column again (decision 23): View kept equal to tab 1's by changedParam.
    auto* gMatte2 = group("grpMatte2", "Matte");
    auto* view2 = labelled(desc.defineChoiceParam(kPView2), "View",
                           "The same View as on the first tab.");
    for (const char* c : kViewNames) view2->appendOption(c);
    view2->setDefault(kViewPicture);
    view2->setAnimates(false);
    view2->setEvaluateOnChange(false);  // render reads tab 1's View, which changes with it
    view2->setParent(*gMatte2);
    auto* version2 = labelled(desc.defineStringParam("version2"), "Version", "");
    version2->setStringType(OFX::eStringTypeLabel);
    version2->setDefault(MR_VERSION " (" MR_GIT ")");
    version2->setIsPersistant(false);
    version2->setEvaluateOnChange(false);
    version2->setParent(*gMatte2);
    auto* gCs = group("grpContourSmooth", "Contour Smooth");
    boolean(kPCsEnable, "Enable",
            "Take wiggle and chatter off the edge line; the edge's softness and corners are kept.",
            false, gCs);
    number(kPCsSmooth, "Smooth px", "Wiggles up to this wavelength are removed.", 30.0, 100.0,
           100.0, 1, 1.0, gCs)
        ->setRange(5.0, 100.0);
    number(kPCsMaxShift, "Max Shift px", "The edge never moves farther than this.", 1.5, 3.0, 3.0,
           2, 0.25, gCs)
        ->setRange(0.25, 3.0);
    number(kPCsCorner, "Corner deg", "Turns sharper than this are corners and kept.", 20.0, 90.0,
           90.0, 0, 1.0, gCs)
        ->setRange(10.0, 90.0);
    number(kPCsEdgeAt, "Edge At", "The matte value of the edge line that is smoothed.", 0.5, 0.95,
           0.95, 3, 0.01, gCs)
        ->setRange(0.05, 0.95);
    auto* gSg = group("grpShrinkGrow", "Shrink / Grow");
    boolean(kPSgEnable, "Enable",
            "Move the edge out (grow) or in (shrink); each part of the edge keeps its softness.",
            false, gSg);
    auto* amount = number(kPSgAmount, "Grow (-Shrink) px",
                          "Pixels to move the edge outward; negative moves it inward.", 1.0, 20.0,
                          10.0, 2, 0.25, gSg);
    amount->setRange(-20.0, 20.0);
    amount->setDisplayRange(-10.0, 10.0);
    auto* gEw = group("grpEdgeWidth", "Edge Width");
    boolean(kPEwEnable, "Enable", "Give the whole edge one consistent softness.", false, gEw);
    number(kPEwWidth, "Edge Width px", "The edge's target width, from clear to solid.", 2.0, 16.0,
           16.0, 2, 0.25, gEw)
        ->setRange(0.5, 16.0);
    number(kPEwMix, "Width Mix", "0 = the input edge, 1 = the target width.", 1.0, 1.0, 1.0, 2,
           0.05, gEw);
    auto* gSn = group("grpSnapToImage", "Snap to Image");
    boolean(kPSnEnable, "Enable",
            "Pull the edge onto the picture's edges (front input); only near the edge.", false, gSn);
    number(kPSnRadius, "Radius px", "The window the picture is compared in.", 8.0, 32.0, 32.0, 1,
           0.5, gSn)
        ->setRange(1.0, 32.0);
    number(kPSnBand, "Band px", "Changes reach this far from the edge or a soft matte area.", 12.0, 64.0, 64.0, 1, 0.5,
           gSn);
    number(kPSnDetail, "Detail", "Higher follows weaker picture edges.", 0.5, 1.0, 1.0, 2, 0.05,
           gSn);
    number(kPSnFromPicture, "Edge From Picture",
           "0 = keep the matte's own softness, 1 = take the softness from the picture.", 0.35, 1.0,
           1.0, 2, 0.05, gSn);
    auto* gFg = group("grpForegroundColour", "Edge Colour Unmix");  // decision 25
    boolean(kPFgEnable, "Enable",
            "Removes the old background's colour from soft edges (RGB only, the matte is unchanged); "
            "View Picture shows it.", false, gFg);
    number(kPFgSmoothness, "Edge Smoothness", "Higher = flatter foreground colour in the edge.",
           0.005, 0.5, 0.2, 4, 0.001, gFg)
        ->setRange(0.0001, 0.5);
    number(kPFgExtend, "Extend px", "Also solve this far beyond the soft edge.", 3.0, 20.0, 20.0,
           1, 0.5, gFg);
    number(kPFgIgnore, "Ignore Outer px",
           "Estimate the colour only from pixels this deep inside the edge, so a matte that is "
           "too big (green edge) still unmixes. Shrink / Grow's amount is added when it is on.",
           3.0, 20.0, 10.0, 1, 0.5, gFg);
    number(kPFgMix, "Mix", "0 = the picture, 1 = the clean foreground colour.", 1.0, 1.0, 1.0, 2,
           0.05, gFg);

  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
    return new MatteRefineEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static MatteRefineFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
