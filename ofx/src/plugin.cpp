// Matte Refine OFX (CLAUDE.md): one node, matte operations as sections (decision 1), all
// rendered inline in render(), no cache, no Process button (decision 14).
// - Matte From picks the input matte (decision 4); enabled ops run in a fixed order (3).
// - Output: RGB = source (or the Show Filled view, 11), alpha = the refined matte.
// - Layout follows Flame's rules: one group = one column of <= 5 rows (CLAUDE.md "Flame host").

#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include "env_info.h"
#include "log.h"
#include "ofxsImageEffect.h"
#include "ops/fill_holes.h"
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

constexpr const char* kPMatteFrom = "matteFrom";
constexpr const char* kPView = "view";
constexpr const char* kPFhEnable = "fhEnable";
constexpr const char* kPFhThreshold = "fhThreshold";
constexpr const char* kPFhMaxHole = "fhMaxHole";
constexpr const char* kPFhBridge = "fhBridge";
constexpr const char* kPFhRim = "fhRim";
constexpr const char* kPFhClosedLeft = "fhClosedLeft";
constexpr const char* kPFhClosedRight = "fhClosedRight";
constexpr const char* kPFhClosedTop = "fhClosedTop";
constexpr const char* kPFhClosedBottom = "fhClosedBottom";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPNodeId = "nodeId";

// Stored as indexes in setups: append only.
enum MatteFrom { kFromAlpha = 0, kFromLuminance = 1, kFromRed = 2 };
const char* const kMatteFromNames[] = {"Alpha", "Luminance", "Red"};
enum View { kViewResult = 0, kViewShowFilled = 1 };
const char* const kViewNames[] = {"Result", "Show Filled"};

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

class MatteRefineEffect : public OFX::ImageEffect {
 public:
  explicit MatteRefineEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    matteFrom_ = fetchChoiceParam(kPMatteFrom);
    view_ = fetchChoiceParam(kPView);
    fhEnable_ = fetchBooleanParam(kPFhEnable);
    fhThreshold_ = fetchDoubleParam(kPFhThreshold);
    fhMaxHole_ = fetchDoubleParam(kPFhMaxHole);
    fhBridge_ = fetchDoubleParam(kPFhBridge);
    fhRim_ = fetchDoubleParam(kPFhRim);
    fhClosedLeft_ = fetchBooleanParam(kPFhClosedLeft);
    fhClosedRight_ = fetchBooleanParam(kPFhClosedRight);
    fhClosedTop_ = fetchBooleanParam(kPFhClosedTop);
    fhClosedBottom_ = fetchBooleanParam(kPFhClosedBottom);
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
    std::unique_ptr<OFX::Image> src(src_->fetchImage(args.time));
    std::string warning;
    if (!src) {
      warning = "No source image (is the input connected?).";
    } else if (src->getPixelDepth() != OFX::eBitDepthFloat) {
      warning = "Only 32-bit float images are supported.";
      src.reset();
    }

    const OfxRectI sb = src ? src->getBounds() : out->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    const int sc = src ? componentCount(src->getPixelComponents()) : 0;
    int from = kFromAlpha, view = kViewResult;
    matteFrom_->getValueAtTime(args.time, from);
    view_->getValueAtTime(args.time, view);
    if (src && from == kFromAlpha && sc == 3) warning = "The source has no alpha: Matte From is Alpha.";

    // Input matte, top-down (OFX rows are bottom-up), clamped to 0-1 (decision 5).
    std::vector<float> matte(size_t(w) * h, 0.0f);
    if (src) {
      for (int y = sb.y1; y < sb.y2; ++y) {
        const auto* p = static_cast<const float*>(src->getPixelAddress(sb.x1, y));
        float* m = &matte[size_t(sb.y2 - 1 - y) * w];
        for (int x = 0; x < w; ++x, p += sc) {
          float v;
          if (sc == 1) v = p[0];
          else if (from == kFromLuminance) v = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
          else if (from == kFromRed) v = p[0];
          else v = sc == 4 ? p[3] : 1.0f;
          m[x] = clamp01(v);
        }
      }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // The operations, in their fixed order (decision 3).
    std::vector<float> result = matte;
    std::string opLog;
    bool fhOn = true;
    fhEnable_->getValueAtTime(args.time, fhOn);
    if (src && fhOn) {
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
      fillHoles(matte.data(), result.data(), w, h, p, &st);
      char b[160];
      std::snprintf(b, sizeof b, " fillHoles %.1f ms (%d filled, %d too large, %lld px)",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count(),
                    st.holesFilled, st.holesTooLarge, st.pixelsChanged);
      opLog += b;
    }

    // Output: RGB = source, or Show Filled (input matte grey, added fill magenta); A = result.
    const OfxRectI db = out->getBounds();
    const OfxRectI& win = args.renderWindow;
    const int dc = componentCount(dst_->getPixelComponents());
    for (int y = std::max(win.y1, db.y1); y < std::min(win.y2, db.y2); ++y) {
      auto* px = static_cast<float*>(out->getPixelAddress(std::max(win.x1, db.x1), y));
      const int row = sb.y2 - 1 - y;
      for (int x = std::max(win.x1, db.x1); x < std::min(win.x2, db.x2); ++x, px += dc) {
        const int col = x - sb.x1;
        const bool inside = src && row >= 0 && row < h && col >= 0 && col < w;
        const size_t i = inside ? size_t(row) * w + col : 0;
        const float a = inside ? result[i] : 0.0f;
        if (dc == 1) {
          px[0] = a;
          continue;
        }
        if (!inside) {
          px[0] = px[1] = px[2] = 0.0f;
        } else if (view == kViewShowFilled) {
          const float m = matte[i], d = a - m;
          px[0] = px[2] = m * (1.0f - d) + d;
          px[1] = m * (1.0f - d);
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
        setPersistentMessage(OFX::Message::eMessageWarning, "", warning);
    } catch (const std::exception& e) {
      LOGW("render: persistent message FAILED: %s", e.what());
    }
    const auto t2 = std::chrono::steady_clock::now();
    LOGD("render t=%g %dx%d scale=%g from=%s view=%s: read %.1f ms,%s total %.1f ms%s%s", args.time,
         w, h, args.renderScale.x, kMatteFromNames[std::min(std::max(from, 0), 2)],
         kViewNames[std::min(std::max(view, 0), 1)],
         std::chrono::duration<double, std::milli>(t1 - t0).count(), opLog.c_str(),
         std::chrono::duration<double, std::milli>(t2 - t0).count(), warning.empty() ? "" : " | ",
         warning.c_str());
  }

 private:
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
  OFX::ChoiceParam* matteFrom_;
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
  OFX::BooleanParam* debugLog_;
  std::string nodeId_;
  bool inChangedParam_ = false;
};

template <class D>
D* labelled(D* p, const char* label, const char* hint) {
  p->setLabels(label, label, label);
  if (hint && *hint) p->setHint(hint);
  return p;
}

class MatteRefineFactory : public OFX::PluginFactoryHelper<MatteRefineFactory> {
 public:
  MatteRefineFactory()
      : PluginFactoryHelper<MatteRefineFactory>(kPluginId, MR_VERSION_MAJOR, MR_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LOGI("==== plugin load (describe) ====");
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
    src->addSupportedComponent(OFX::ePixelComponentAlpha);
    src->setSupportsTiles(false);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    // Flame: params flow into columns of <= 5 rows, 6 columns per tab, each group starts a
    // column, pages are ignored (CLAUDE.md "Planned layout"). Groups in column order.
    OFX::PageParamDescriptor* page = desc.definePageParam("Matte Refine");
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
    auto number = [&](const char* name, const char* text, const char* hint, double def,
                      double maxRange, double maxDisplay, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text, hint);
      p->setRange(0.0, maxRange);
      p->setDisplayRange(0.0, maxDisplay);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };

    // --- Matte (5 rows): input, view, version ---
    auto* gMatte = group("grpMatte", "Matte");
    auto* from = labelled(desc.defineChoiceParam(kPMatteFrom), "Matte From",
                          "Where the input matte is read from: the source alpha (Flame's matte "
                          "input), the luminance of the RGB (Rec.709), or the red channel.");
    for (const char* c : kMatteFromNames) from->appendOption(c);
    from->setDefault(kFromAlpha);
    from->setParent(*gMatte);
    auto* view = labelled(desc.defineChoiceParam(kPView), "View",
                          "Result: the image output is the source. Show Filled: the image "
                          "output is the input matte in grey with the added fill in magenta.");
    for (const char* c : kViewNames) view->appendOption(c);
    view->setDefault(kViewResult);
    view->setParent(*gMatte);
    auto* version = labelled(desc.defineStringParam("version"), "Version", "");
    version->setStringType(OFX::eStringTypeLabel);
    version->setDefault(MR_VERSION " (" MR_GIT ")");
    version->setIsPersistant(false);
    version->setEvaluateOnChange(false);
    version->setParent(*gMatte);

    // --- Fill Holes (5 rows; decisions 8-10) ---
    auto* gFh = group("grpFillHoles", "Fill Holes");
    boolean(kPFhEnable, "Enable", "Fill holes enclosed by solid matte with white.", true, gFh);
    number(kPFhThreshold, "Solid Above", "Matte at or above this value counts as solid.", 0.5, 1.0,
           1.0, gFh);
    number(kPFhMaxHole, "Max Hole %",
           "Holes larger than this percentage of the frame stay open (real gaps, e.g. an arm "
           "on a hip). 0 = fill every enclosed hole.",
           1.0, 100.0, 10.0, gFh);
    number(kPFhBridge, "Bridge Gaps px",
           "Also fill holes whose solid edge has a gap up to this many pixels wide. 0 = only "
           "fully enclosed holes.",
           0.0, 200.0, 20.0, gFh);
    number(kPFhRim, "Rim px",
           "Also lift the soft rim around a filled hole to white, up to this many pixels into "
           "the solid matte.",
           1.0, 50.0, 5.0, gFh);

    // --- Fill Holes Edges (4 rows) ---
    auto* gEdges = group("grpFillHolesEdges", "Fill Holes Edges");
    const char* edgeHint = "Treat this frame edge as solid, e.g. a subject cut by the frame: "
                           "holes reaching it are filled.";
    boolean(kPFhClosedLeft, "Left Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedRight, "Right Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedTop, "Top Closed", edgeHint, false, gEdges);
    boolean(kPFhClosedBottom, "Bottom Closed", edgeHint, false, gEdges);

    // --- Support (2 rows, per user) ---
    auto* gSupport = group("grpSupport", "Support");
    auto* dbg = boolean(kPDebugLog, "Debug Logging",
                        "Verbose logging for troubleshooting (per user).", false, gSupport);
    dbg->setIsPersistant(false);
    dbg->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPCollectLogs), "Collect Logs",
             "Write a support bundle (logs and system details) to your home folder.")
        ->setParent(*gSupport);

    // --- Hidden ---
    auto* nid = desc.defineStringParam(kPNodeId);
    nid->setDefault("");
    nid->setIsSecret(true);
    nid->setEvaluateOnChange(false);
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
