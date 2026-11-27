// Plate Tools bundle, first effect "Plate Merge" (docs/plate_merge.md, root decision 43): two aligned plates
// of the same camera, each with a tight matte of the person in it, merged into one plate (plate_merge.h).
// Classical, inline in render() (decision 14), synchronous per frame, no temporal access (decision 5).
// Inputs: a filter; Source = plate A on the front with A's matte in its alpha (decisions 4, 35); "PlateB" an
// optional RGBA clip (plate B, its matte in its alpha; NOT a mask clip: it carries a picture); "KeepA" an
// optional mask clip (a protect roto: plate A is kept there, bit for bit). Output: RGB = the merged plate
// (or a QC view), alpha = the Swap Matte or the Residual (Matte Out). Colour: no conversion requested; the
// output keeps the source's colourspace for picture views (decision 23). Any failure: a footer and plate A
// passing through; never an empty frame.
// More two-plate / clean-plate effects go into this bundle as further factories (one factory class per
// effect: the Support library keeps the plugin id in a static member per factory type, ofxsImageEffect.h
// PluginFactoryHelper; root decision 41).

#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
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
#include "plate_merge.h"
#include "plate_ui.h"
#include "process.h"
#include "user_config.h"

#ifndef PT_VERSION  // from ofx/plate/CMakeLists.txt
#define PT_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif
#ifndef PT_VERSION_MAJOR
#define PT_VERSION_MAJOR 0
#define PT_VERSION_MINOR 0
#endif

namespace {

using namespace mr;
namespace pm = mr::plate;

constexpr const char* kPluginId = "com.janklier.platemerge";
constexpr const char* kPluginLabel = "Plate Merge";
constexpr const char* kPlateBClip = "PlateB";  // stored in setups: never rename
constexpr const char* kKeepClip = "KeepA";

// Param ids: stored in setups, append only.
constexpr const char* kPView = "view";
constexpr const char* kPMatteOut = "matteOut";
constexpr const char* kPMode = "mode";
constexpr const char* kPGrow = "growPx";
constexpr const char* kPFeather = "featherPx";
constexpr const char* kPMatch = "levelMatch";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPNodeId = "nodeId";

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
    h.version = d->versionLabel + " (" + std::to_string(d->versionMajor) + "." + std::to_string(d->versionMinor) +
                "." + std::to_string(d->versionMicro) + ")";
  }
  return h;
}

int componentCount(OFX::PixelComponentEnum c) {
  return c == OFX::ePixelComponentRGBA ? 4 : c == OFX::ePixelComponentRGB ? 3 : 1;
}

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// A picture (RGB, w*h*3) and its alpha (w*h, 1 when it has none), top-down (OFX rows are bottom-up). No
// clamp: the plates' values (> 1, < 0) go to the core unchanged.
void extractPlate(const OFX::Image& img, std::vector<float>& rgb, std::vector<float>& alpha) {
  const OfxRectI b = img.getBounds();
  const int w = b.x2 - b.x1, h = b.y2 - b.y1, c = componentCount(img.getPixelComponents());
  rgb.resize(size_t(w) * h * 3);
  alpha.resize(size_t(w) * h);
  for (int y = b.y1; y < b.y2; ++y) {
    const auto* p = static_cast<const float*>(img.getPixelAddress(b.x1, y));
    const size_t row = size_t(b.y2 - 1 - y) * w;
    for (int x = 0; x < w; ++x, p += c) {
      float* q = &rgb[(row + x) * 3];
      q[0] = p[0];
      q[1] = c >= 3 ? p[1] : p[0];
      q[2] = c >= 3 ? p[2] : p[0];
      alpha[row + x] = c == 4 ? p[3] : c == 1 ? p[0] : 1.0f;
    }
  }
}

// A mask clip (Alpha-only in Flame; RGBA -> A, RGB -> R), top-down.
void extractMask(const OFX::Image& img, std::vector<float>& m) {
  const OfxRectI b = img.getBounds();
  const int w = b.x2 - b.x1, h = b.y2 - b.y1, c = componentCount(img.getPixelComponents());
  m.resize(size_t(w) * h);
  for (int y = b.y1; y < b.y2; ++y) {
    const auto* p = static_cast<const float*>(img.getPixelAddress(b.x1, y));
    float* q = &m[size_t(b.y2 - 1 - y) * w];
    for (int x = 0; x < w; ++x, p += c) q[x] = c == 4 ? p[3] : p[0];
  }
}

bool constantOne(const std::vector<float>& m) {
  return !m.empty() && std::all_of(m.begin(), m.end(), [](float v) { return v == 1.0f; });
}

bool sameBounds(const OfxRectI& a, const OfxRectI& b) {
  return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
}

class PlateMergeEffect : public OFX::ImageEffect {
 public:
  explicit PlateMergeEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    plateB_ = fetchClip(kPlateBClip);
    keep_ = fetchClip(kKeepClip);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    view_ = fetchChoiceParam(kPView);
    matteOut_ = fetchChoiceParam(kPMatteOut);
    mode_ = fetchChoiceParam(kPMode);
    grow_ = fetchDoubleParam(kPGrow);
    feather_ = fetchDoubleParam(kPFeather);
    match_ = fetchBooleanParam(kPMatch);
    debugLog_ = fetchBooleanParam(kPDebugLog);
    OFX::StringParam* nodeIdParam = fetchStringParam(kPNodeId);
    try {
      nodeIdParam->getValue(nodeId_);
      if (nodeId_.empty()) {
        nodeId_ = randomId();  // saved with the setup; copies of a node share it
        nodeIdParam->setValue(nodeId_);
      }
      logNode_ = "pm:" + nodeId_;
      LogScope scope(logNode_);
      LOGI("createInstance %p context=%d", static_cast<void*>(this), int(getContext()));
      debugLog_->setValue(logLevel() == LogLevel::Debug);  // per user, not stored in setups
    } catch (const std::exception& e) {
      LOGE("createInstance: setting param values FAILED: %s [%s]", e.what(), typeid(e).name());
    }
  }

  ~PlateMergeEffect() override {
    LogScope scope(logNode_);
    LOGI("destroyInstance %p", static_cast<void*>(this));
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    // Flame echoes our own setValue() as changedParam: only user edits, never re-entrantly
    // (ofx/CLAUDE.md "Flame host behaviour").
    if (args.reason != OFX::eChangeUserEdit || inChangedParam_) return;
    struct Guard {
      bool& b;
      explicit Guard(bool& b) : b(b) { b = true; }
      ~Guard() { b = false; }
    } guard(inChangedParam_);
    LogScope scope(logNode_);
    LOGI("changedParam %s (user) time=%g", name.c_str(), args.time);
    if (name == kPCollectLogs) {
      collectLogs();
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
    LogScope scope(logNode_);
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<OFX::Image> out(dst_->fetchImage(args.time));
    if (!out) {
      LOGE("render t=%g: host returned no output image", args.time);
      OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    int view = pm::kViewResult, matteOut = pm::kMatteSwap;
    view_->getValueAtTime(args.time, view);
    view = pm::clampView(view);
    matteOut_->getValueAtTime(args.time, matteOut);
    pm::UiValues ui;
    mode_->getValueAtTime(args.time, ui.mode);
    ui.growPx = grow_->getValueAtTime(args.time);
    ui.featherPx = feather_->getValueAtTime(args.time);
    match_->getValueAtTime(args.time, ui.levelMatch);

    // Plate A with its matte in the alpha (mandatory in a filter).
    std::string warning;  // a footer that means the merge did not run (plate A passes through)
    bool error = false;
    std::unique_ptr<OFX::Image> src(src_->isConnected() ? src_->fetchImage(args.time) : nullptr);
    if (src && src->getPixelDepth() != OFX::eBitDepthFloat) {
      warning = "Only 32-bit float images are supported.";
      error = true;
      src.reset();
    }
    if (!src && warning.empty()) warning = "No plate A.";
    const OfxRectI sb = src ? src->getBounds() : out->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    if (src) {
      extractPlate(*src, aRgb_, aMatte_);
      if (componentCount(src->getPixelComponents()) != 4 || constantOne(aMatte_)) warning = pm::constantMatteFooter("A");
    }

    // Plate B (RGBA, its matte in the alpha).
    std::string bLog = "not connected";
    if (warning.empty()) {
      if (!plateB_->isConnected()) {
        warning = pm::kNoPlateBFooter;
      } else {
        std::unique_ptr<OFX::Image> bi;
        try {
          bi.reset(plateB_->fetchImage(args.time));
        } catch (const std::exception& e) {
          bLog = std::string("fetch failed: ") + e.what();
        }
        if (!bi) {
          warning = pm::failedFooter("no plate B image from the host");
          error = true;
        } else if (bi->getPixelDepth() != OFX::eBitDepthFloat) {
          warning = pm::failedFooter("plate B is not 32-bit float");
          error = true;
        } else {
          const OfxRectI bb = bi->getBounds();
          const int bc = componentCount(bi->getPixelComponents());
          char b[96];
          std::snprintf(b, sizeof b, "%dch %d,%d-%d,%d", bc, bb.x1, bb.y1, bb.x2, bb.y2);
          bLog = b;
          if (!sameBounds(bb, sb)) {
            warning = pm::sizeMismatchFooter(bb.x2 - bb.x1, bb.y2 - bb.y1, w, h);
            error = true;
          } else if (bc == 1) {
            warning = pm::failedFooter("plate B arrived as a matte only (connect the picture to Plate B's front)");
            error = true;
          } else {
            extractPlate(*bi, bRgb_, bMatte_);
            if (bc != 4 || constantOne(bMatte_)) warning = pm::constantMatteFooter("B");
          }
        }
      }
    }
    // Keep A (protect roto, optional mask).
    std::string keepLog = "not connected";
    bool haveKeep = false;
    if (warning.empty() && keep_->isConnected()) {
      std::unique_ptr<OFX::Image> ki;
      try {
        ki.reset(keep_->fetchImage(args.time));
      } catch (const std::exception& e) {
        keepLog = std::string("fetch failed: ") + e.what();
      }
      if (!ki || ki->getPixelDepth() != OFX::eBitDepthFloat) {
        warning = pm::failedFooter("no 32-bit float Keep A image from the host");
        error = true;
      } else if (!sameBounds(ki->getBounds(), sb)) {
        warning = pm::failedFooter("Keep A is not the size of plate A");
        error = true;
      } else {
        extractMask(*ki, keep_buf_);
        keepLog = std::to_string(componentCount(ki->getPixelComponents())) + "ch";
        haveKeep = true;
      }
    }
    const double readMs = msSince(t0);

    // The merge (or plate A passing through, alpha 0).
    const pm::PlateParams p = pm::toPlateParams(ui, args.renderScale.x);
    pm::PlateInfo info;
    bool merged = false;
    double mergeMs = 0;
    const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
    if (warning.empty()) {
      const auto tm = std::chrono::steady_clock::now();
      try {
        result_.resize(n * 3);
        swap_.resize(n);
        residual_.resize(n);
        ring_.assign(view == pm::kViewRing ? n : 0, 0.0f);
        matchedB_.resize(view == pm::kViewPlateB ? n * 3 : 0);
        pm::PlateViews v;
        v.swap = swap_.data();
        v.residual = residual_.data();
        v.ring = view == pm::kViewRing ? ring_.data() : nullptr;
        v.matchedB = view == pm::kViewPlateB ? matchedB_.data() : nullptr;
        pm::mergePlates(aRgb_.data(), aMatte_.data(), bRgb_.data(), bMatte_.data(), haveKeep ? keep_buf_.data() : nullptr,
                        w, h, p, result_.data(), v, &info);
        merged = true;
      } catch (const std::exception& e) {
        warning = pm::failedFooter(e.what());
        error = true;
        LOGE("render t=%g: merge FAILED: %s [%s]", args.time, e.what(), typeid(e).name());
      }
      mergeMs = msSince(tm);
    }

    // Output: RGB per View, alpha per Matte Out, inside the render window.
    const auto tw = std::chrono::steady_clock::now();
    const OfxRectI ob = out->getBounds();
    const OfxRectI& win = args.renderWindow;
    const int oc = componentCount(out->getPixelComponents());
    for (int y = std::max(win.y1, ob.y1); y < std::min(win.y2, ob.y2); ++y) {
      const int x0 = std::max(win.x1, ob.x1), x1 = std::min(win.x2, ob.x2);
      if (x1 <= x0) continue;
      auto* px = static_cast<float*>(out->getPixelAddress(x0, y));
      const int row = sb.y2 - 1 - y;
      for (int x = x0; x < x1; ++x, px += oc) {
        const int col = x - sb.x1;
        const bool inside = src && row >= 0 && row < h && col >= 0 && col < w;
        const size_t i = inside ? size_t(row) * w + col : 0;
        const float a = !inside || !merged ? 0.0f : matteOut == pm::kMatteResidual ? residual_[i] : swap_[i];
        if (oc == 1) {
          px[0] = a;
          continue;
        }
        const float* rgb = nullptr;
        float grey = 0.0f;
        if (!inside) {
          grey = 0.0f;
        } else if (!merged || view == pm::kViewPlateA) {
          rgb = &aRgb_[3 * i];
        } else if (view == pm::kViewResult) {
          rgb = &result_[3 * i];
        } else if (view == pm::kViewPlateB) {
          rgb = &matchedB_[3 * i];
        } else {
          grey = view == pm::kViewSwap ? swap_[i] : view == pm::kViewResidual ? residual_[i] : ring_[i];
        }
        px[0] = rgb ? rgb[0] : grey;
        px[1] = rgb ? rgb[1] : grey;
        px[2] = rgb ? rgb[2] : grey;
        if (oc == 4) px[3] = a;
      }
    }
    const double writeMs = msSince(tw);

    // Footer: a warning / error every render; the info line only when it changes.
    const std::string footer = warning.empty() ? pm::infoFooter(info, ui.levelMatch) : warning;
    try {
      if (!warning.empty()) {
        setPersistentMessage(error ? OFX::Message::eMessageError : OFX::Message::eMessageWarning, "", footer);
      } else if (footer != lastFooter_) {
        setPersistentMessage(OFX::Message::eMessageMessage, "", footer);
      }
      if (footer != lastFooter_) LOGI("footer: %s", footer.c_str());
      lastFooter_ = footer;
    } catch (const std::exception& e) {
      LOGW("render: persistent message FAILED: %s", e.what());
    }
    // One DEBUG line per render: settings, counts, where the time went. No image data.
    LOGD("render t=%g %dx%d scale=%g window=%d,%d-%d,%d view=%s matteOut=%s | mode=%s grow=%.3g px feather=%.3g px "
         "match=%d | plateB %s keep %s | %s: swap %ld residual %ld ring %ld matched %d gain %+.3f %+.3f %+.3f | read "
         "%.1f merge %.1f write %.1f render %.1f ms%s%s",
         args.time, w, h, args.renderScale.x, win.x1, win.y1, win.x2, win.y2, pm::kViewNames[view],
         pm::kMatteOutNames[matteOut == pm::kMatteResidual], pm::kModeNames[ui.mode == 1], p.growPx, p.featherPx,
         int(p.levelMatch), bLog.c_str(), keepLog.c_str(), merged ? "merged" : "passed through", info.swapPixels,
         info.residualPixels, info.ringPixels, int(info.matched), info.globalGainStops[0], info.globalGainStops[1],
         info.globalGainStops[2], readMs, mergeMs, writeMs, msSince(t0), warning.empty() ? "" : " | ",
         warning.c_str());
  }

  // Decision 23: the picture views keep the source's colourspace; a matte on RGB is tagged as a matte.
  std::string outputColourspace() {
    LogScope scope(logNode_);
    int view = pm::kViewResult;
    view_->getValue(view);
    const std::string style = getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false);
    const bool picture = pm::viewIsPicture(pm::clampView(view));
    const bool matte = !picture && style == kOfxImageEffectColourManagementOCIO && configHasMatte();
    const std::string cs = pm::outputColourspaceAnswer(picture, src_->isConnected(), style, matte);
    std::lock_guard<std::mutex> lock(csMutex_);
    if (!outputLogged_ || cs != lastOutputCs_) {
      LOGI("colour: output colourspace answer '%s'%s (style '%s')", cs.c_str(), cs.empty() ? " (reply default)" : "",
           style.c_str());
      lastOutputCs_ = cs;
      outputLogged_ = true;
    }
    return cs;
  }

 private:
  bool configHasMatte() {
    if (!matteChecked_) {
      matteChecked_ = true;
      const std::string path = getPropertySet().propGetString(kOfxImageEffectPropOCIOConfig, false);
      std::ifstream in(path);
      const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      hasMatte_ = colourspaceEncoding(text, "Matte") == "data";
      LOGI("colour: OCIO config '%s' (%zu bytes), Matte colourspace %s", path.c_str(), text.size(),
           hasMatte_ ? "found" : "not found (Raw)");
    }
    return hasMatte_;
  }

  // A support bundle (as the other plugins: the log is shared).
  void collectLogs() {
    const std::string dir = logDir();
    {
      std::ofstream envf(dir + "/environment.txt");
      for (const auto& line : environmentSnapshot(hostInfo())) envf << line << "\n";
    }
    const std::string bundle = homeDir() + "/matte-refine-support-" + timeStr("%Y%m%d-%H%M%S") + ".tar.gz";
    const std::vector<std::string> argv = {"/usr/bin/tar", "-czf", bundle, "-C", dir.substr(0, dir.rfind('/')), "logs"};
    std::string err;
    WorkerProcess tar;
    std::vector<std::string> lines;
    const bool started = tar.start(argv, cleanEnvironment(), err);
    if (started)
      while (tar.readLines(200, lines)) {
      }
    const int status = started ? tar.wait() : -1;
    const bool ok = started && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    LOGI("collectLogs: %s -> %s %s", ok ? "ok" : "FAILED", bundle.c_str(), err.c_str());
    sendMessage(ok ? OFX::Message::eMessageMessage : OFX::Message::eMessageError, "",
                ok ? "Support bundle written to " + bundle + ". Please review it before sending."
                   : "Could not write the support bundle: " + err);
  }

  OFX::Clip *src_, *plateB_, *keep_, *dst_;
  OFX::ChoiceParam *view_, *matteOut_, *mode_;
  OFX::BooleanParam *match_, *debugLog_;
  OFX::DoubleParam *grow_, *feather_;
  std::string nodeId_, logNode_;
  bool inChangedParam_ = false;
  bool matteChecked_ = false, hasMatte_ = false;
  std::mutex csMutex_;
  std::string lastOutputCs_;
  bool outputLogged_ = false;
  std::string lastFooter_;
  std::vector<float> aRgb_, aMatte_, bRgb_, bMatte_, keep_buf_, result_, swap_, residual_, ring_, matchedB_;
};

template <class D>
D* labelled(D* p, const char* label, const char* hint) {
  p->setLabels(label, label, label);
  if (hint && *hint) p->setHint(hint);
  return p;
}

class PlateMergeFactory : public OFX::PluginFactoryHelper<PlateMergeFactory> {
 public:
  // GetOutputColourspace (OpenFX 1.5) is unknown to the 1.4 Support library: answered here (as the
  // other plugins), everything else goes to the Support library.
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    if (std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0) {
      try {
        auto* fx = dynamic_cast<PlateMergeEffect*>(OFX::Private::retrieveImageEffectPointer((OfxImageEffectHandle)handle));
        const std::string cs = fx ? fx->outputColourspace() : "";
        if (cs.empty()) return kOfxStatReplyDefault;
        OFX::PropertySet(out).propSetString(kOfxImageClipPropColourspace, cs);
        return kOfxStatOK;
      } catch (const std::exception& e) {
        LOGE("colour: GetOutputColourspace FAILED: %s", e.what());
        return kOfxStatFailed;
      }
    }
    return FactoryMainEntryHelper<PlateMergeFactory>::mainEntry(action, handle, in, out);
  }
  OfxPluginEntryPoint* getMainEntry() override { return entry; }

  PlateMergeFactory() : PluginFactoryHelper<PlateMergeFactory>(kPluginId, PT_VERSION_MAJOR, PT_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LogScope scope("pm");
    LOGI("==== Plate Tools %s (%s) load (describe): Plate Merge ====", PT_VERSION, MR_GIT);
    for (const auto& line : environmentSnapshot(hostInfo())) LOGI("env %s", line.c_str());
    desc.setLabels(kPluginLabel, kPluginLabel, kPluginLabel);
    desc.setPluginGrouping("Matte Refine");
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);
    // OCIO style (decision 23): the host then reports the config and asks GetOutputColourspace.
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementStyle, kOfxImageEffectColourManagementOCIO,
                                        false);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementAvailableConfigs,
                                        "ofx-native-v1.5_aces-v1.3_ocio-v2.3", false);
  }

  void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum ctx) override {
    LogScope scope("pm");
    try {
      describeParams(desc);
      LOGI("describeInContext ctx=%d done", int(ctx));
    } catch (const std::exception& e) {
      LOGE("describeInContext ctx=%d FAILED: %s [%s]", int(ctx), e.what(), typeid(e).name());
      throw;
    }
  }

  void describeParams(OFX::ImageEffectDescriptor& desc) {
    // Source = plate A (front) + A's matte (its alpha); Plate B an RGBA clip (front + its alpha: how Flame
    // shows a non-mask RGBA clip's sockets is UNVERIFIED for this case, docs/plate_merge.md); Keep A a mask.
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);
    OFX::ClipDescriptor* b = desc.defineClip(kPlateBClip);
    b->addSupportedComponent(OFX::ePixelComponentRGBA);
    b->addSupportedComponent(OFX::ePixelComponentRGB);
    b->setOptional(true);
    b->setSupportsTiles(false);
    b->setLabels("Plate B", "Plate B", "Plate B");
    OFX::ClipDescriptor* k = desc.defineClip(kKeepClip);
    k->addSupportedComponent(OFX::ePixelComponentAlpha);
    k->addSupportedComponent(OFX::ePixelComponentRGBA);
    k->addSupportedComponent(OFX::ePixelComponentRGB);
    k->setIsMask(true);
    k->setOptional(true);
    k->setSupportsTiles(false);
    k->setLabels("Keep A", "Keep A", "Keep A");
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    // Flame: one group = one column of <= 5 rows (choice 2, others 1; a hidden param takes a slot too).
    // Columns: Plate Merge (View 2, Matte Out 2, Version 1) · Merge (Mode 2, Grow 1, Feather 1, Level Match 1)
    // · Support (Debug Logging, Collect Logs, hidden node id).
    OFX::PageParamDescriptor* page = desc.definePageParam("Plate Merge");
    auto group = [&](const char* name, const char* text) {
      auto* g = desc.defineGroupParam(name);
      g->setLabels(text, text, text);
      g->setOpen(true);
      page->addChild(*g);
      return g;
    };
    auto choice = [&](const char* name, const char* text, const char* hint, const char* const* opts, int n, int def,
                      OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineChoiceParam(name), text, hint);
      for (int i = 0; i < n; ++i) p->appendOption(opts[i]);  // stored as an index: append only
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto number = [&](const char* name, const char* text, const char* hint, double def, double lo, double hi,
                      OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text, hint);
      p->setRange(lo, hi);
      p->setDisplayRange(lo, hi);
      p->setDigits(1);  // Flame showed doubles without digits as integers
      p->setIncrement(0.5);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    const pm::UiValues def;

    auto* gMain = group("grpPlateMerge", "Plate Merge");
    auto* view = choice(kPView, "View",
                        "What the image output shows. Result: the merged plate. Plate A: the front input. Plate B "
                        "Matched: plate B after Level Match. Swap Matte: where plate B is taken (white). Residual: "
                        "where both persons are, so neither plate has a clean pixel. Ring: the pixels Level Match "
                        "compares.",
                        pm::kViewNames, pm::kViewCount, pm::kViewResult, gMain);
    view->setAnimates(false);
    choice(kPMatteOut, "Matte Out",
           "What the node's matte output carries. Swap Matte: where plate B was taken (for a comp in Batch). "
           "Residual: where both persons are (for an inpaint or a check).",
           pm::kMatteOutNames, 2, pm::kMatteSwap, gMain);
    auto* version = labelled(desc.defineStringParam("version"), "Version", "");
    version->setStringType(OFX::eStringTypeLabel);
    version->setDefault(PT_VERSION " (" MR_GIT ")");
    version->setIsPersistant(false);
    version->setEvaluateOnChange(false);
    version->setParent(*gMain);

    auto* gMerge = group("grpMerge", "Merge");
    choice(kPMode, "Mode",
           "B Over A: plate B through both mattes over plate A (B's person on top; A's person replaced by B's clean "
           "background). Clean Background: both persons removed where the other plate is clean.",
           pm::kModeNames, 2, def.mode, gMerge);
    number(kPGrow, "Grow px",
           "How far past each matte the swap reaches (soft shadows, defocus, halos), in pixels at full resolution.",
           def.growPx, 0.0, 200.0, gMerge);
    number(kPFeather, "Feather px", "Soft fade after Grow, in pixels at full resolution. 0 = hard edge.",
           def.featherPx, 0.0, 200.0, gMerge);
    auto* match = labelled(desc.defineBooleanParam(kPMatch), "Level Match",
                           "Match plate B's level and colour to plate A on a ring of clean pixels around the swap, "
                           "so the seam disappears (assumes scene-linear plates).");
    match->setDefault(def.levelMatch);
    match->setParent(*gMerge);

    auto* gSupport = group("grpSupport", "Support");
    auto* dbg = labelled(desc.defineBooleanParam(kPDebugLog), "Debug Logging",
                         "Verbose logging for troubleshooting (per user).");
    dbg->setDefault(false);
    dbg->setIsPersistant(false);
    dbg->setEvaluateOnChange(false);
    dbg->setParent(*gSupport);
    labelled(desc.definePushButtonParam(kPCollectLogs), "Collect Logs",
             "Write a support bundle (logs and system details) to your home folder.")
        ->setParent(*gSupport);
    auto* nid = desc.defineStringParam(kPNodeId);  // hidden, but takes a slot: inside a group
    nid->setDefault("");
    nid->setIsSecret(true);
    nid->setEvaluateOnChange(false);
    nid->setParent(*gSupport);
  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
    return new PlateMergeEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static PlateMergeFactory merge;  // more Plate Tools effects: one factory each, appended here
  ids.push_back(&merge);
}
}  // namespace Plugin
}  // namespace OFX
