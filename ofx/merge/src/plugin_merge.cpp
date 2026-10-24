// Matte Merge, the third plugin (ofx/merge/CLAUDE.md, root decision 37): a base matte (or trimap) plus
// a detail matte in, one combined matte out (matte_merge.h). render() stays synchronous per frame
// (decision 14), the classical core runs inline.
// Inputs (probe in Flame 2027.x, ofx/merge/CLAUDE.md "Probe result"): a filter; Source = the picture
// on the front with the base matte / trimap in its alpha (Flame's source alpha socket, decisions 4
// and 35); one optional mask clip "Detail" (arrives Alpha-only). Output: alpha = the combined matte,
// RGB per View (decision 11). Colour: no conversion requested (the core is encoding-invariant);
// output tags as decision 23 (outputColourspaceAnswer, merge_ui.h). Any failure is a footer and the
// base matte passing through; never an empty frame.

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
#include "matte_merge.h"
#include "merge_ui.h"
#include "ocio_encoding.h"
#include "ofxColour.h"
#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"
#include "process.h"
#include "user_config.h"

#ifndef MM_VERSION  // from ofx/merge/CMakeLists.txt
#define MM_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif
#ifndef MM_VERSION_MAJOR
#define MM_VERSION_MAJOR 0
#define MM_VERSION_MINOR 0
#endif

namespace {

using namespace mr;
namespace mm = mr::merge;

constexpr const char* kPluginId = "com.janklier.mattemerge";
constexpr const char* kPluginLabel = "Matte Merge";
constexpr const char* kDetailClip = "Detail";

// Param ids: stored in setups, append only.
constexpr const char* kPView = "view";
constexpr const char* kPSource = "zoneSource";
constexpr const char* kPReach = "zoneReach";
constexpr const char* kPKeepDisc = "zoneKeepDisconnected";
constexpr const char* kPAlign = "align";
constexpr const char* kPMethod = "method";
constexpr const char* kPDetailWidth = "detailWidth";
constexpr const char* kPStrength = "strength";
constexpr const char* kPFeather = "feather";
constexpr const char* kPEvidence = "evidence";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPNodeId = "nodeId";
constexpr const char* kPBridge = "zoneBridge";

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

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// A matte, top-down (OFX rows are bottom-up), clamped to 0-1: Alpha-only -> its channel, RGBA -> A,
// RGB -> R (as Matte Refine's mask clip; an RGB picture's "alpha" is 1, as Flame's empty socket).
void extractMatte(const OFX::Image& img, bool rgbIsMatte, std::vector<float>& m) {
  const OfxRectI b = img.getBounds();
  const int w = b.x2 - b.x1, h = b.y2 - b.y1, c = componentCount(img.getPixelComponents());
  m.resize(size_t(w) * h);
  for (int y = b.y1; y < b.y2; ++y) {
    const auto* p = static_cast<const float*>(img.getPixelAddress(b.x1, y));
    float* q = &m[size_t(b.y2 - 1 - y) * w];
    for (int x = 0; x < w; ++x, p += c) q[x] = clamp01(c == 1 ? p[0] : c == 4 ? p[3] : rgbIsMatte ? p[0] : 1.0f);
  }
}

// The picture, top-down, w*h*3 (the core's layout; Flame delivers RGBA).
void extractRgb(const OFX::Image& img, std::vector<float>& rgb) {
  const OfxRectI b = img.getBounds();
  const int w = b.x2 - b.x1, h = b.y2 - b.y1, c = componentCount(img.getPixelComponents());
  rgb.resize(size_t(w) * h * 3);
  for (int y = b.y1; y < b.y2; ++y) {
    const auto* p = static_cast<const float*>(img.getPixelAddress(b.x1, y));
    float* q = &rgb[size_t(b.y2 - 1 - y) * w * 3];
    for (int x = 0; x < w; ++x, p += c, q += 3) {
      q[0] = p[0];
      q[1] = c >= 3 ? p[1] : p[0];
      q[2] = c >= 3 ? p[2] : p[0];
    }
  }
}

class MatteMergeEffect : public OFX::ImageEffect {
 public:
  explicit MatteMergeEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    detail_ = fetchClip(kDetailClip);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    view_ = fetchChoiceParam(kPView);
    source_ = fetchChoiceParam(kPSource);
    reach_ = fetchDoubleParam(kPReach);
    keepDisc_ = fetchBooleanParam(kPKeepDisc);
    bridge_ = fetchDoubleParam(kPBridge);
    align_ = fetchChoiceParam(kPAlign);
    method_ = fetchChoiceParam(kPMethod);
    detailWidth_ = fetchDoubleParam(kPDetailWidth);
    strength_ = fetchDoubleParam(kPStrength);
    feather_ = fetchDoubleParam(kPFeather);
    evidence_ = fetchChoiceParam(kPEvidence);
    debugLog_ = fetchBooleanParam(kPDebugLog);
    OFX::StringParam* nodeIdParam = fetchStringParam(kPNodeId);
    try {
      nodeIdParam->getValue(nodeId_);
      if (nodeId_.empty()) {
        nodeId_ = randomId();  // saved with the setup; copies of a node share it
        nodeIdParam->setValue(nodeId_);
      }
      logNode_ = "mm:" + nodeId_;
      LogScope scope(logNode_);
      LOGI("createInstance %p context=%d", static_cast<void*>(this), int(getContext()));
      debugLog_->setValue(logLevel() == LogLevel::Debug);  // per user, not stored in setups
    } catch (const std::exception& e) {
      LOGE("createInstance: setting param values FAILED: %s [%s]", e.what(), typeid(e).name());
    }
  }

  ~MatteMergeEffect() override {
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
    int view = mm::kViewResult;
    view_->getValueAtTime(args.time, view);
    view = mm::clampView(view);
    mm::UiValues ui;
    source_->getValueAtTime(args.time, ui.source);
    ui.reachPct = reach_->getValueAtTime(args.time);
    keepDisc_->getValueAtTime(args.time, ui.keepDisconnected);
    ui.bridgePct = bridge_->getValueAtTime(args.time);
    align_->getValueAtTime(args.time, ui.align);
    method_->getValueAtTime(args.time, ui.method);
    ui.detailWidthPct = detailWidth_->getValueAtTime(args.time);
    ui.strength = strength_->getValueAtTime(args.time);
    ui.featherPct = feather_->getValueAtTime(args.time);
    evidence_->getValueAtTime(args.time, ui.evidence);

    // The picture with the base matte in its alpha (mandatory in a filter: Flame always connects it).
    std::string warning;  // a footer that means the merge did not run (the base passes through)
    bool error = false;
    std::unique_ptr<OFX::Image> src(src_->isConnected() ? src_->fetchImage(args.time) : nullptr);
    if (src && src->getPixelDepth() != OFX::eBitDepthFloat) {
      warning = "Only 32-bit float images are supported.";
      error = true;
      src.reset();
    }
    if (!src && warning.empty()) warning = "No picture.";
    const OfxRectI sb = src ? src->getBounds() : out->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    const int sc = src ? componentCount(src->getPixelComponents()) : 0;
    if (src) extractMatte(*src, false, base_);
    else base_.assign(size_t(std::max(w, 0)) * std::max(h, 0), 0.0f);
    if (src && sc == 3) warning = mm::noAlphaFooter();
    if (src && warning.empty() && !base_.empty()) {
      const auto lim = std::minmax_element(base_.begin(), base_.end());
      if (*lim.first == *lim.second) warning = mm::constantBaseFooter(*lim.first);
    }

    // The detail matte (mask clip: Alpha-only in Flame).
    std::string detailLog = "not connected";
    bool haveDetail = false;
    if (warning.empty()) {
      if (!detail_->isConnected()) {
        warning = mm::kNoDetailFooter;
      } else {
        std::unique_ptr<OFX::Image> di;
        try {
          di.reset(detail_->fetchImage(args.time));
        } catch (const std::exception& e) {
          detailLog = std::string("fetch failed: ") + e.what();
        }
        if (!di) {
          if (detailLog == "not connected") detailLog = "no image";
          warning = mm::failedFooter("no detail image from the host");
          error = true;
        } else if (di->getPixelDepth() != OFX::eBitDepthFloat) {
          detailLog = "not float";
          warning = mm::failedFooter("the detail matte is not 32-bit float");
          error = true;
        } else {
          const OfxRectI db = di->getBounds();
          const int dc = componentCount(di->getPixelComponents());
          char b[96];
          std::snprintf(b, sizeof b, "%dch %d,%d-%d,%d", dc, db.x1, db.y1, db.x2, db.y2);
          detailLog = b;
          if (db.x1 != sb.x1 || db.y1 != sb.y1 || db.x2 != sb.x2 || db.y2 != sb.y2) {
            warning = mm::sizeMismatchFooter(db.x2 - db.x1, db.y2 - db.y1, w, h);
            error = true;
          } else {
            extractMatte(*di, true, detailM_);
            haveDetail = true;
          }
        }
      }
    }
    const double readMs = msSince(t0);

    // The merge (or the base passing through).
    const mm::MergeParams p = mm::toMergeParams(ui, h);  // h as delivered: already at render scale
    mm::MergeInfo info;
    double mergeMs = 0;
    bool merged = false;
    const bool needView = view >= mm::kViewAdded;
    viewBuf_.assign(needView ? base_.size() : 0, 0.0f);
    if (haveDetail) {
      const auto tm = std::chrono::steady_clock::now();
      try {
        extractRgb(*src, rgb_);
        result_.resize(base_.size());
        mm::MergeViews mv;
        float* vb = needView ? viewBuf_.data() : nullptr;
        if (view == mm::kViewAdded) mv.added = vb;
        if (view == mm::kViewRejected) mv.rejected = vb;
        if (view == mm::kViewZone) mv.zone = vb;
        if (view == mm::kViewResidual) mv.residual = vb;
        mm::merge(base_.data(), detailM_.data(), rgb_.data(), w, h, p, result_.data(), mv, &info, &ws_);
        merged = true;
      } catch (const std::exception& e) {
        warning = mm::failedFooter(e.what());
        error = true;
        LOGE("render t=%g: merge FAILED: %s [%s]", args.time, e.what(), typeid(e).name());
        std::fill(viewBuf_.begin(), viewBuf_.end(), 0.0f);
      }
      mergeMs = msSince(tm);
    }
    const std::vector<float>& alpha = merged ? result_ : base_;

    // Output (decision 11): alpha = the combined matte, RGB per View, inside the render window.
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
        const bool inside = row >= 0 && row < h && col >= 0 && col < w && !alpha.empty();
        const size_t i = inside ? size_t(row) * w + col : 0;
        const float a = inside ? alpha[i] : 0.0f;
        if (oc == 1) {
          px[0] = a;
          continue;
        }
        if (!inside) {
          px[0] = px[1] = px[2] = 0.0f;
        } else if (view == mm::kViewPicture && src) {
          const auto* sp = static_cast<const float*>(src->getPixelAddress(x, y));
          px[0] = sp[0];
          px[1] = sc >= 3 ? sp[1] : sp[0];
          px[2] = sc >= 3 ? sp[2] : sp[0];
        } else if (view == mm::kViewAdded) {
          const float d = viewBuf_[i];  // > 0 white: added; < 0 green: lowered (Zone Blend)
          px[0] = px[2] = std::max(d, 0.0f);
          px[1] = std::fabs(d);
        } else if (needView) {
          px[0] = px[1] = px[2] = viewBuf_[i];
        } else {
          px[0] = px[1] = px[2] = a;  // Result Matte (and Picture without a picture)
        }
        if (oc == 4) px[3] = a;
      }
    }
    const double writeMs = msSince(tw);

    // Footer: a warning / error every render (as Matte Refine ML); the info line only when it changes.
    std::string footer = warning.empty() ? (merged ? mm::infoFooter(info) : std::string()) : warning;
    try {
      if (footer.empty()) {
        if (!lastFooter_.empty()) clearPersistentMessage();
      } else if (!warning.empty()) {
        setPersistentMessage(error ? OFX::Message::eMessageError : OFX::Message::eMessageWarning, "", footer);
      } else if (footer != lastFooter_) {
        setPersistentMessage(OFX::Message::eMessageMessage, "", footer);
      }
      if (footer != lastFooter_) LOGI("footer: %s", footer.empty() ? "(cleared)" : footer.c_str());
      lastFooter_ = footer;
    } catch (const std::exception& e) {
      LOGW("render: persistent message FAILED: %s", e.what());
    }
    // One DEBUG line per render: settings, alignment, counts, where the time went. No image data.
    const mm::AlignInfo& ai = info.align;
    LOGD("render t=%g %dx%d scale=%g window=%d,%d-%d,%d view=%s | source=%s reach=%.3g%% (%.1f px) keep=%d bridge=%.3g%% (%.2f px) "
         "align=%s method=%s width=%.3g%% (%.2f px) strength=%.3g feather=%.3g%% (%.2f px) evidence=%s | detail %s "
         "| %s: model '%s' uncertainty %.3f px agreement %.2f blocks %d/%d move %.2f,%.2f scale %.5f | zone %ld "
         "kept %ld dropped %ld | read %.1f merge %.1f write %.1f render %.1f ms%s%s",
         args.time, w, h, args.renderScale.x, win.x1, win.y1, win.x2, win.y2, mm::kViewNames[view],
         mm::kSourceNames[p.source == mm::Source::Trimap], ui.reachPct, p.reachPx, int(p.keepDisconnected), ui.bridgePct, p.bridgePx,
         mm::kAlignNames[!p.align], mm::kMethodNames[p.method == mm::Method::ZoneBlend], ui.detailWidthPct,
         p.detailWidthPx, p.strength, ui.featherPct, p.featherPx, mm::kEvidenceNames[p.evidence],
         detailLog.c_str(), merged ? "merged" : "passed through", mm::alignModelName(ai.model), ai.uncertaintyPx,
         ai.agreement, ai.inliers, ai.blocks, info.reg.tx, info.reg.ty, info.reg.scale, info.zonePixels,
         info.keptPixels, info.droppedPixels, readMs, mergeMs, writeMs, msSince(t0), warning.empty() ? "" : " | ",
         warning.c_str());
  }

  // Decision 23 (adapted): the picture passing through keeps its colourspace; a matte on RGB is
  // tagged as a matte. Logged at INFO when the answer changes.
  std::string outputColourspace() {
    LogScope scope(logNode_);
    int view = mm::kViewResult;
    view_->getValue(view);
    const std::string style = getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false);
    const bool viewPicture = mm::clampView(view) == mm::kViewPicture;
    const bool matte = !viewPicture && style == kOfxImageEffectColourManagementOCIO && configHasMatte();
    const std::string cs = mm::outputColourspaceAnswer(viewPicture, src_->isConnected(), style, matte);
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

  // A support bundle (as the other two plugins: the log is shared).
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

  OFX::Clip *src_, *detail_, *dst_;
  OFX::ChoiceParam *view_, *source_, *align_, *method_, *evidence_;
  OFX::BooleanParam *keepDisc_, *debugLog_;
  OFX::DoubleParam *reach_, *bridge_, *detailWidth_, *strength_, *feather_;
  std::string nodeId_, logNode_;
  bool inChangedParam_ = false;
  bool matteChecked_ = false, hasMatte_ = false;
  std::mutex csMutex_;
  std::string lastOutputCs_;
  bool outputLogged_ = false;
  std::string lastFooter_;  // the footer last shown ("" = none)
  std::vector<float> base_, detailM_, rgb_, result_, viewBuf_;  // reused across renders
  mm::MergeWorkspace ws_;                                         // one per instance (matte_merge.h)
};

template <class D>
D* labelled(D* p, const char* label, const char* hint) {
  p->setLabels(label, label, label);
  if (hint && *hint) p->setHint(hint);
  return p;
}

class MatteMergeFactory : public OFX::PluginFactoryHelper<MatteMergeFactory> {
 public:
  // GetOutputColourspace (OpenFX 1.5) is unknown to the 1.4 Support library: answered here (as the
  // other two plugins), everything else goes to the Support library.
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    if (std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0) {
      try {
        auto* fx = dynamic_cast<MatteMergeEffect*>(OFX::Private::retrieveImageEffectPointer((OfxImageEffectHandle)handle));
        const std::string cs = fx ? fx->outputColourspace() : "";
        if (cs.empty()) return kOfxStatReplyDefault;
        OFX::PropertySet(out).propSetString(kOfxImageClipPropColourspace, cs);
        return kOfxStatOK;
      } catch (const std::exception& e) {
        LOGE("colour: GetOutputColourspace FAILED: %s", e.what());
        return kOfxStatFailed;
      }
    }
    return FactoryMainEntryHelper<MatteMergeFactory>::mainEntry(action, handle, in, out);
  }
  OfxPluginEntryPoint* getMainEntry() override { return entry; }

  MatteMergeFactory() : PluginFactoryHelper<MatteMergeFactory>(kPluginId, MM_VERSION_MAJOR, MM_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LogScope scope("mm");
    LOGI("==== Matte Merge %s (%s) load (describe) ====", MM_VERSION, MR_GIT);
    for (const auto& line : environmentSnapshot(hostInfo())) LOGI("env %s", line.c_str());
    desc.setLabels(kPluginLabel, kPluginLabel, kPluginLabel);
    desc.setPluginGrouping("Matte Refine");
    desc.addSupportedContext(OFX::eContextFilter);  // the probe's layout (ofx/merge/CLAUDE.md)
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
    LogScope scope("mm");
    try {
      describeParams(desc);
      LOGI("describeInContext ctx=%d done", int(ctx));
    } catch (const std::exception& e) {
      LOGE("describeInContext ctx=%d FAILED: %s [%s]", int(ctx), e.what(), typeid(e).name());
      throw;
    }
  }

  void describeParams(OFX::ImageEffectDescriptor& desc) {
    // Clips as the probe's "mask" variant (user decision 2026-10-24): Source mandatory (picture on
    // the front, base / trimap in its alpha), Detail an optional mask clip, no colour management.
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);
    OFX::ClipDescriptor* d = desc.defineClip(kDetailClip);
    d->addSupportedComponent(OFX::ePixelComponentAlpha);
    d->addSupportedComponent(OFX::ePixelComponentRGBA);
    d->addSupportedComponent(OFX::ePixelComponentRGB);
    d->setIsMask(true);
    d->setOptional(true);
    d->setSupportsTiles(false);
    d->setLabels("Detail", "Detail", "Detail");
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    // Flame: one group = one column of <= 5 rows (choice 2, others 1; a hidden param takes a slot
    // too), 6 columns per tab. Columns: Matte (3) · Zone (5) · Align (2) · Combine (5) · Support (5):
    // one tab.
    OFX::PageParamDescriptor* page = desc.definePageParam("Matte Merge");
    auto group = [&](const char* name, const char* text) {
      auto* g = desc.defineGroupParam(name);
      g->setLabels(text, text, text);
      g->setOpen(true);
      page->addChild(*g);
      return g;
    };
    auto number = [&](const char* name, const char* text, const char* hint, double def, double lo, double hi,
                      int digits, double increment, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text, hint);
      p->setRange(lo, hi);
      p->setDisplayRange(lo, hi);
      p->setDigits(digits);  // Flame showed doubles without digits as integers
      p->setIncrement(increment);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto choice = [&](const char* name, const char* text, const char* hint, const char* const* opts, int n, int def,
                      OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineChoiceParam(name), text, hint);
      for (int i = 0; i < n; ++i) p->appendOption(opts[i]);  // stored as an index: append only
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    const mm::UiValues def;

    auto* gMatte = group("grpMatte", "Matte");
    auto* view = choice(kPView, "View",
                        "What the image output shows (the matte output is always the combined matte). Result "
                        "Matte: the combined matte. Picture: the front input. Added: what the detail matte added "
                        "(white; green where Zone Blend lowered the base). Rejected: detail that was dropped "
                        "(not connected to the subject, or no support in the picture). Zone: where the detail may "
                        "act. Residual: how far the aligned detail's body still differs from the base.",
                        mm::kViewNames, 6, mm::kViewResult, gMatte);
    view->setAnimates(false);
    auto* version = labelled(desc.defineStringParam("version"), "Version", "");
    version->setStringType(OFX::eStringTypeLabel);
    version->setDefault(MM_VERSION " (" MR_GIT ")");
    version->setIsPersistant(false);
    version->setEvaluateOnChange(false);
    version->setParent(*gMatte);

    auto* gZone = group("grpZone", "Zone");
    choice(kPSource, "Source",
           "Matte Edge: the detail may act around the base matte's edge. Trimap: the source alpha is a trimap "
           "(black, white, grey unknown) and the detail fills its grey zone.",
           mm::kSourceNames, 2, def.source, gZone);
    number(kPReach, "Reach % of height",
           "Matte Edge: how far beyond the base matte's edge the detail may act, as a percentage of the frame "
           "height.",
           def.reachPct, 0.0, 30.0, 2, 0.1, gZone);
    auto* keep = labelled(desc.defineBooleanParam(kPKeepDisc), "Keep Disconnected",
                          "Trimap only: keep detail that does not touch the subject (loose strands, but also blobs "
                          "and other objects). Off: it becomes background.");
    keep->setDefault(def.keepDisconnected);
    keep->setParent(*gZone);
    number(kPBridge, "Bridge % of height",
           "Add Fine Detail with Matte Edge: within this distance of the base matte's edge, take the detail matte as "
           "it is (not only its thin strands), so hair joins the head without a dark gap. As a percentage of the "
           "frame height. 0 = off.",
           def.bridgePct, 0.0, 3.0, 2, 0.05, gZone);

    auto* gAlign = group("grpAlign", "Align");
    choice(kPAlign, "Align",
           "Auto: line the detail matte up with the picture (offset and small scale) before merging. Off: use it "
           "as it is.",
           mm::kAlignNames, 2, def.align, gAlign);

    auto* gCombine = group("grpCombine", "Combine");
    choice(kPMethod, "Method",
           "Add Fine Detail: add the detail's thin structures (hair, fur) on top of the base; never lowers it. "
           "Zone Blend: take the detail matte inside the zone.",
           mm::kMethodNames, 2, def.method, gCombine);
    number(kPDetailWidth, "Detail Width % of height",
           "Structures thinner than this (percentage of the frame height) count as fine detail.", def.detailWidthPct,
           0.05, 5.0, 2, 0.05, gCombine);
    number(kPStrength, "Strength",
           "0 = the base matte, 1 = the full detail. Above 1 = extra gain on the detail that is added (never lowers the "
           "matte, result stays 0-1), to bring up thin strands that are weaker than they should be.",
           def.strength, 0.0, 2.0, 2, 0.05,
           gCombine);
    number(kPFeather, "Feather % of height",
           "Soft transition at the zone's border, as a percentage of the frame height. 0 = hard.", def.featherPct,
           0.0, 2.0, 2, 0.05, gCombine);

    auto* gSupport = group("grpSupport", "Support");
    choice(kPEvidence, "Picture Evidence",
           "On: keep a fine strand only where the picture shows it too (drops invented strands; may also drop "
           "some real ones). View Rejected shows what is dropped.",
           mm::kEvidenceNames, 2, def.evidence, gSupport);
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
    return new MatteMergeEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static MatteMergeFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
