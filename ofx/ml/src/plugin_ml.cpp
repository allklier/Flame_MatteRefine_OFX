// Matte Refine ML, phase 1 (ofx/ml/CLAUDE.md): one node, Temporal (B4, neighbour frames, no model), Smart
// Fill (B3, BiRefNet) then Edge Refine (B1, ViTMatte), chained by render_chain.h. render() stays
// synchronous per frame (decisions 14, 28); the model runs in mr_ml_runner, one process per user
// shared by every node (decision 32: this binary holds no ONNX Runtime and loads no GPU library,
// runner_client.h). Inputs: decision 35 (ofx/ml/CLAUDE.md): a filter, the picture on the front
// (Source, mandatory), the matte in its alpha (Flame's source alpha socket, decision 4). Colour
// tagging as Matte Refine (23, 34); the matte
// outside the band is bit-identical (17). Built to be observed: one INFO log line per render with the stage times.
// Any failure (no runner, no model, no CUDA, a model error, a timeout) is a footer message and the
// input matte; the next render tries again.

#include <dlfcn.h>
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

#include "edge_refine.h"
#include "env_info.h"
#include "gate.h"
#include "log.h"
#include "ocio_encoding.h"
#include "ofxColour.h"
#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"
#include "process.h"
#include "render_chain.h"
#include "runner_client.h"
#include "user_config.h"

#ifndef MRML_VERSION  // from ofx/ml/CMakeLists.txt
#define MRML_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif
#ifndef MRML_VERSION_MAJOR
#define MRML_VERSION_MAJOR 0
#define MRML_VERSION_MINOR 0
#endif

namespace {

using namespace mr;

constexpr const char* kPluginId = "com.janklier.matterefineml";
constexpr const char* kPluginLabel = "Matte Refine ML";

constexpr const char* kPView = "view";
constexpr const char* kPPictureColour = "pictureColour";
constexpr const char* kPTmEnable = "tmEnable";
constexpr const char* kPTmFrames = "tmFrames";
constexpr const char* kPTmStrength = "tmStrength";
constexpr const char* kPTmBand = "tmBand";
constexpr const char* kPSfEnable = "sfEnable";
constexpr const char* kPSfThreshold = "sfThreshold";
constexpr const char* kPSfMaxHole = "sfMaxHole";
constexpr const char* kPSfModel = "sfModel";
constexpr const char* kPEnable = "erEnable";
constexpr const char* kPBand = "erBand";
constexpr const char* kPOuter = "erOuter";
constexpr const char* kPMix = "erMix";
constexpr const char* kPOnlyGrey = "erOnlyGrey";
constexpr const char* kPModel = "erModel";
constexpr const char* kPDebugLog = "debugLog";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr const char* kPNodeId = "nodeId";

// Stored as indexes in setups: append only.
enum View { kViewPicture = 0, kViewMatte = 1, kViewTrimap = 2, kViewChanges = 3, kViewHoles = 4 };
const char* const kViewNames[] = {"Picture", "Matte", "Trimap", "Changes", "Holes"};
constexpr int kViewLast = int(sizeof kViewNames / sizeof kViewNames[0]) - 1;
enum PictureColour { kColourAuto = 0, kColourBypass = 1 };
const char* const kColourNames[] = {"Auto", "Bypass"};
const char* const kModelNames[] = {"Small", "Base"};
const char* const kModelDirs[] = {"vitmatte-small", "vitmatte-base"};
const char* const kSegNames[] = {"Lite", "Full"};  // ml::Segmentation (render_chain.h)
const char* const kTemporalNames[] = {"+-1 frame", "+-2 frames"};  // ml::TemporalFrames (render_chain.h)

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

// "~/x" -> "$HOME/x" for config values.
std::string expandHome(const std::string& p) {
  return p.compare(0, 2, "~/") == 0 ? homeDir() + p.substr(1) : p;
}

// <bundle>/Contents/Resources (this file is <bundle>/Contents/<arch>/MatteRefineML.ofx): the
// runner and, in a release bundle, licences.txt (the setup gate, gate.h).
std::string bundleResourcesDir() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&bundleResourcesDir), &info) || !info.dli_fname) return "";
  std::string p = info.dli_fname;
  for (int i = 0; i < 2; ++i) p = p.substr(0, p.rfind('/'));
  return p + "/Resources";
}

// The runner next to this binary: <bundle>/Contents/Resources/mr_ml_runner. Config key ml_runner
// overrides it.
std::string defaultRunnerPath() {
  const std::string r = bundleResourcesDir();
  return r.empty() ? "" : r + "/mr_ml_runner";
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

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// The input matte, top-down (OFX rows are bottom-up), clamped to 0-1 (as Matte Refine).
void extractMatte(const OFX::Image& src, std::vector<float>& matte) {
  const OfxRectI sb = src.getBounds();
  const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1, sc = componentCount(src.getPixelComponents());
  matte.resize(size_t(w) * h);
  for (int y = sb.y1; y < sb.y2; ++y) {
    const auto* p = static_cast<const float*>(src.getPixelAddress(sb.x1, y));
    float* m = &matte[size_t(sb.y2 - 1 - y) * w];
    for (int x = 0; x < w; ++x, p += sc) m[x] = clamp01(sc == 1 ? p[0] : sc == 4 ? p[3] : 1.0f);
  }
}

// The picture, top-down, w*h*3.
void extractRgb(const OFX::Image& src, std::vector<float>& rgb) {
  const OfxRectI sb = src.getBounds();
  const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1, sc = componentCount(src.getPixelComponents());
  rgb.resize(size_t(w) * h * 3);
  for (int y = sb.y1; y < sb.y2; ++y) {
    const auto* p = static_cast<const float*>(src.getPixelAddress(sb.x1, y));
    float* q = &rgb[size_t(sb.y2 - 1 - y) * w * 3];
    for (int x = 0; x < w; ++x, p += sc, q += 3) {
      q[0] = p[0];
      q[1] = sc >= 3 ? p[1] : p[0];
      q[2] = sc >= 3 ? p[2] : p[0];
    }
  }
}

// Picture Colour Auto: the colourspaces requested for the picture in GetClipPreferences, in
// order (OpenFX 1.5 OfxImageClipPropPreferredColourspaces_Source). Flame 2027.2 honours it (test on
// akflame5, aces2.0 config): the picture arrives converted to the first name, display-encoded
// (pure gamma 2.4, no tone mapping, no clamp: values can exceed 0-1). A list, so a fallback name
// can be added.
const char* const kPicturePref[] = {"Rec.1886 Rec.709 - Display"};

// The GetOutputColourspace answer as a pure function (decisions 23, 34), testable without a host.
// View Picture with a picture connected (VERIFIED in Flame 2027.2, decision 35):
//   - Auto: the first requested name (kPicturePref[0]): the picture arrives converted to it.
//   - Bypass: OfxColourspace_Source (Flame delivers the raw clip with its true tag).
// Other views, or no picture: "Matte" / "Raw" / "ofx_raw" by colour management style, or "" (reply
// default), as before.
std::string outputColourspaceAnswer(bool viewPicture, bool pictureConnected, bool bypass, const std::string& style,
                                    bool configHasMatte) {
  if (viewPicture && pictureConnected) return bypass ? "OfxColourspace_Source" : kPicturePref[0];
  if (style == kOfxImageEffectColourManagementOCIO) return configHasMatte ? "Matte" : "Raw";
  if (style == kOfxImageEffectColourManagementFull || style == kOfxImageEffectColourManagementCore)
    return kOfxColourspaceRaw;
  if (style == kOfxImageEffectColourManagementBasic) return kOfxColourspaceOfxRaw;
  return "";
}

std::string joinNames(const std::vector<std::string>& names) {
  std::string s;
  for (const auto& n : names) s += (s.empty() ? "'" : ", '") + n + "'";
  return "[" + s + "]";
}

// A multi-valued string property through the raw property suite (the 1.4 Support library has no
// colour calls): the set status and what reads back.
std::string setStrings(OfxPropertySetHandle h, const std::string& prop, const std::vector<std::string>& names) {
  std::vector<const char*> v;
  for (const auto& n : names) v.push_back(n.c_str());
  const OfxStatus st = OFX::Private::gPropSuite->propSetStringN(h, prop.c_str(), int(v.size()), v.data());
  int dim = -1;
  const OfxStatus ds = OFX::Private::gPropSuite->propGetDimension(h, prop.c_str(), &dim);
  char b[256];
  std::snprintf(b, sizeof b, "set %s: %s, read back %s %d values", prop.c_str(), OFX::mapStatusToString(st),
                OFX::mapStatusToString(ds), dim);
  return b;
}

// All values of a string property, or its status if it cannot be read.
std::string getStrings(OfxPropertySetHandle h, const char* prop) {
  int dim = 0;
  const OfxStatus ds = OFX::Private::gPropSuite->propGetDimension(h, prop, &dim);
  if (ds != kOfxStatOK) return std::string("<") + OFX::mapStatusToString(ds) + ">";
  std::vector<std::string> v;
  for (int i = 0; i < dim; ++i) {
    char* s = nullptr;
    const OfxStatus st = OFX::Private::gPropSuite->propGetString(h, prop, i, &s);
    v.push_back(st == kOfxStatOK && s ? s : std::string("<") + OFX::mapStatusToString(st) + ">");
  }
  return joinNames(v);
}

class MatteRefineMLEffect : public OFX::ImageEffect {
 public:
  explicit MatteRefineMLEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    view_ = fetchChoiceParam(kPView);
    pictureColour_ = fetchChoiceParam(kPPictureColour);
    tmEnable_ = fetchBooleanParam(kPTmEnable);
    tmFrames_ = fetchChoiceParam(kPTmFrames);
    tmStrength_ = fetchDoubleParam(kPTmStrength);
    tmBand_ = fetchDoubleParam(kPTmBand);
    sfEnable_ = fetchBooleanParam(kPSfEnable);
    sfThreshold_ = fetchDoubleParam(kPSfThreshold);
    sfMaxHole_ = fetchDoubleParam(kPSfMaxHole);
    sfModel_ = fetchChoiceParam(kPSfModel);
    enable_ = fetchBooleanParam(kPEnable);
    band_ = fetchDoubleParam(kPBand);
    outer_ = fetchDoubleParam(kPOuter);
    mix_ = fetchDoubleParam(kPMix);
    onlyGrey_ = fetchBooleanParam(kPOnlyGrey);
    model_ = fetchChoiceParam(kPModel);
    debugLog_ = fetchBooleanParam(kPDebugLog);
    OFX::StringParam* nodeIdParam = fetchStringParam(kPNodeId);
    try {
      nodeIdParam->getValue(nodeId_);
      if (nodeId_.empty()) {
        nodeId_ = randomId();  // saved with the setup; copies of a node share it
        nodeIdParam->setValue(nodeId_);
      }
      logNode_ = "ml:" + nodeId_;  // the log's component is "ofx" for both plugins
      LogScope scope(logNode_);
      LOGI("createInstance %p context=%d", static_cast<void*>(this), int(getContext()));
      debugLog_->setValue(logLevel() == LogLevel::Debug);  // per user, not stored in setups
    } catch (const std::exception& e) {
      LOGE("createInstance: setting param values FAILED: %s [%s]", e.what(), typeid(e).name());
    }
  }

  ~MatteRefineMLEffect() override {
    LogScope scope(logNode_);
    LOGI("destroyInstance %p", static_cast<void*>(this));
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
  }

  // Temporal (B4): the Source frames t - K .. t + K (clamped to the clip's range) when Temporal is on;
  // off: nothing set, the host's default (frame t only). Logged (first call INFO, then DEBUG; the count
  // and the last range go into the render's Temporal line): whether Flame calls it is UNVERIFIED.
  void getFramesNeeded(const OFX::FramesNeededArguments& args, OFX::FramesNeededSetter& frames) override {
    LogScope scope(logNode_);
    bool on = false;
    int fi = ml::kTemporal1;
    tmEnable_->getValueAtTime(args.time, on);
    tmFrames_->getValueAtTime(args.time, fi);
    const ml::FrameRange r = sourceRange();
    double lo = args.time, hi = args.time;
    if (on) {
      ml::framesNeededRange(args.time, ml::temporalK(fi), r, &lo, &hi);
      OfxRangeD range;
      range.min = lo;
      range.max = hi;
      frames.setFramesNeeded(*src_, range);
    }
    char b[200];
    std::snprintf(b, sizeof b, "t=%g %s [%g, %g], clip range %s", args.time,
                  on ? "Temporal on: set Source" : "Temporal off: host default", lo, hi, rangeText(r).c_str());
    std::lock_guard<std::mutex> lock(requestMutex_);
    if (fnTotal_++ == 0) LOGI("getFramesNeeded: first call: %s", b);
    else LOGD("getFramesNeeded: %s", b);
    ++fnCalls_;
    fnLast_ = b;
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
    // Decision 35: the picture is Source, the matte its alpha.
    std::string warning;
    bool error = false;
    std::unique_ptr<OFX::Image> src(src_->isConnected() ? src_->fetchImage(args.time) : nullptr);
    if (src && src->getPixelDepth() != OFX::eBitDepthFloat) {
      warning = "Only 32-bit float images are supported.";
      src.reset();
    }
    const std::string picture = pictureProbe(src.get());
    if (!src && warning.empty()) warning = "No picture.";
    const OfxRectI sb = src ? src->getBounds() : out->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    const int sc = src ? componentCount(src->getPixelComponents()) : 0;
    int view = kViewPicture;
    view_->getValueAtTime(args.time, view);
    if (src && sc == 3)
      warning = "The picture has no alpha, so there is no matte: connect the matte to the node's second "
                "socket (source alpha).";

    if (src) {
      extractMatte(*src, matte_);
    } else {
      matte_.assign(size_t(w) * h, 0.0f);
    }
    if (src && !matte_.empty()) {
      const auto mm = std::minmax_element(matte_.begin(), matte_.end());
      if (*mm.first == *mm.second && warning.empty()) {
        char b[160];
        std::snprintf(b, sizeof b,
                      "The input matte is constant %.2f: no matte on the source alpha. Connect the "
                      "matte to the node's second socket (source alpha).",
                      *mm.first);
        warning = b;
      }
    }
    const double readMs = msSince(t0);

    bool enable = true, onlyGrey = false, sfEnable = false, tmEnable = false;
    enable_->getValueAtTime(args.time, enable);
    onlyGrey_->getValueAtTime(args.time, onlyGrey);
    sfEnable_->getValueAtTime(args.time, sfEnable);
    tmEnable_->getValueAtTime(args.time, tmEnable);
    int model = 0, sfModel = 0;
    model_->getValueAtTime(args.time, model);
    model = std::min(std::max(model, 0), 1);
    sfModel_->getValueAtTime(args.time, sfModel);
    sfModel = std::min(std::max(sfModel, 0), 1);
    ml::ChainParams cp;  // matte -> Temporal -> Smart Fill -> Edge Refine (render_chain.h)
    cp.temporal = tmEnable;
    tmFrames_->getValueAtTime(args.time, cp.tm.frames);
    cp.tm.strength = tmStrength_->getValueAtTime(args.time);
    cp.tm.bandPx = tmBand_->getValueAtTime(args.time);  // full-resolution px: scaled in temporalParams (6)
    cp.smartFill = sfEnable;
    cp.sf.subjectThreshold = sfThreshold_->getValueAtTime(args.time);
    cp.sf.maxHolePercent = sfMaxHole_->getValueAtTime(args.time);
    cp.sf.segmentation = sfModel;
    cp.renderScale = args.renderScale.x;  // decision 6: Temporal's and Smart Fill's pixel params
    cp.edgeRefine = enable;
    ml::RefineParams& p = cp.er;
    p.bandPx = band_->getValueAtTime(args.time) * args.renderScale.x;  // decision 6
    p.outerPx = outer_->getValueAtTime(args.time) * args.renderScale.x;
    p.mix = mix_->getValueAtTime(args.time);
    p.onlyGrey = onlyGrey;
    result_ = matte_;
    mid_ = matte_;
    trimap_.clear();
    ml::ChainResult cr;
    const ml::RefineStats& st = cr.er;
    ml::RunnerClientStats sfCs;  // the runner client's stats of the Smart Fill step
    bool ran = false;
    std::string tmFooter;  // Temporal's footer (merged below)
    bool tmError = false;
    const UserConfig cfg = loadUserConfig();
    auto get = [&](const char* k, const std::string& def) {
      auto it = cfg.find(k);
      return expandHome(it == cfg.end() || it->second.empty() ? def : it->second);
    };
    // The setup gate (gate.h, root decision 38): in a release bundle a missing setup (licences, the
    // standard models, the CUDA libraries) turns every step off, Temporal too, and the footer says
    // what is missing. A dev bundle is not gated.
    const std::string modelDir = get("ml_model_dir", ml::defaultModelDir(homeDir()));
    const ml::GateResult g = ml::checkGate(bundleResourcesDir(), cfg, modelDir, get("ml_cuda_libs", ""));
    if (!g.open()) {
      cp.temporal = cp.smartFill = cp.edgeRefine = false;
      warning = g.footer + (warning.empty() ? std::string() : " " + warning);
      error = true;
    }
    if (g.detail != gateLogged_) {  // one line when the gate's answer changes
      if (!g.open()) LOGI("ml: setup gate closed: %s %s", g.footer.c_str(), g.detail.c_str());
      else if (!gateLogged_.empty()) LOGI("ml: setup gate open");
      gateLogged_ = g.detail;
    }
    // A selected model that is not installed (the full models are optional): that step passes the
    // matte through, the footer says how to install it.
    auto needModel = [&](bool* on, const char* id) {
      if (!*on || ml::modelInstalled(modelDir, id)) return;
      *on = false;
      const std::string f = ml::notInstalledFooter(id);
      warning = (warning.empty() ? f : warning + " " + f);
      error = true;
    };
    needModel(&cp.smartFill, ml::segmentationModel(sfModel));
    needModel(&cp.edgeRefine, kModelDirs[model]);
    if (src && (cp.edgeRefine || cp.smartFill || cp.temporal)) {
      extractRgb(*src, rgb_);
      ml::RunnerClientOptions ro;
      ro.runnerPath = get("ml_runner", defaultRunnerPath());
      ro.tileTimeoutS = std::atof(get("ml_tile_timeout_s", "20").c_str());
      if (!(ro.tileTimeoutS > 0)) ro.tileTimeoutS = 20;
      ro.name = logNode_;
      // One connection per node, kept between renders; a changed config gets a new one.
      if (!client_ || ro.runnerPath != clientPath_ || ro.tileTimeoutS != clientTimeout_) {
        client_.reset(new ml::RunnerClient(ro));
        clientPath_ = ro.runnerPath;
        clientTimeout_ = ro.tileTimeoutS;
      }
      client_->setModel(kModelDirs[model]);  // the model table ids (model_table.h)
      client_->resetStats();
      ml::RunnerSeg seg(*client_, ml::segmentationModel(sfModel));
      ml::Neighbours nb;  // Temporal (B4): the neighbour frames from the host
      if (cp.temporal) {
        // A neighbour: Source at t + k, as the centre (picture RGB, matte = its alpha, decision 35).
        auto fetch = [&](double time, ml::FetchedFrame* f) {
          const auto tf = std::chrono::steady_clock::now();
          std::unique_ptr<OFX::Image> img;
          try {
            img.reset(src_->fetchImage(time));
          } catch (const std::exception& e) {
            f->error = e.what();
          }
          const double hostMs = msSince(tf);
          if (img) {
            f->image = true;
            const OfxRectI b = img->getBounds();
            f->w = b.x2 - b.x1;
            f->h = b.y2 - b.y1;
            f->components = componentCount(img->getPixelComponents());
            f->isFloat = img->getPixelDepth() == OFX::eBitDepthFloat;
            f->note = "uid '" + img->getUniqueIdentifier() + "'";
            if (f->isFloat && f->components == 4) {
              if (!tmPool_.empty()) {  // the buffers of an earlier render: no new pages per frame
                f->rgb.swap(tmPool_.back().rgb);
                f->matte.swap(tmPool_.back().matte);
                tmPool_.pop_back();
              }
              extractRgb(*img, f->rgb);
              extractMatte(*img, f->matte);
              if (samePixels(*f)) f->note += ", SAME PIXELS AS THE CENTRE";
            }
          }
          f->ms = msSince(tf);
          char hb[64];
          std::snprintf(hb, sizeof hb, "%shost %.1f ms, copy %.1f ms", f->note.empty() ? "" : ", ", hostMs, f->ms - hostMs);
          f->note += hb;
        };
        ml::collectNeighbours(args.time, ml::temporalK(cp.tm.frames), sourceRange(), w, h, fetch, &nb,
                              [this]() { return abort(); });
        cp.neighbours = &nb;
      }
      ml::runChain(matte_.data(), rgb_.data(), w, h, cp, seg, *client_, &mid_, &result_, &cr,
                   [this]() { return abort(); }, view == kViewTrimap ? &trimap_ : nullptr, [&]() {
                     sfCs = client_->stats();  // Smart Fill's share; Edge Refine's starts at zero
                     client_->resetStats();
                   }, &tmOut_);
      for (ml::FetchedFrame& f : nb.frames) tmPool_.push_back(std::move(f));
      if (cr.tRan) tmFooter = logTemporal(args, cp, nb, cr, src->getUniqueIdentifier(), &tmError);
      if (cr.sfRan) {
        if (cr.sfFailed) {
          warning = std::string("Smart Fill: ") + cr.sfError + ". The matte passes through this step.";
          error = true;
          LOGE("ml: smart fill FAILED: %s (runner %s)", cr.sfError.c_str(), ro.runnerPath.c_str());
        } else if (cr.sf.segmented && sfCs.ep == "cpu" && !sfCs.why.empty() && warning.empty()) {
          warning = "Smart Fill runs on the CPU (slow): " + sfCs.why;
        }
        if (sfCs.sessionCreated)
          LOGI("ml: the runner (pid %d) created a session for model %s: load %.0f ms, ep %s%s%s",
               client_->runnerPid(), ml::segmentationModel(sfModel), sfCs.loadMs, sfCs.ep.c_str(),
               sfCs.why.empty() ? "" : " because: ", sfCs.why.c_str());
        // One INFO line per frame for Smart Fill (B3): its decisions and where the time went.
        LOGI("render t=%g smart fill: model %s threshold %.3g max hole %.3g%% scale %g | holes found %d filled %d "
             "kept %d too large %d, changed %lld | segmented %s: runner %.1f ms (round trip %.1f, model %.1f) prep "
             "%.1f post %.1f | core %.1f ms (label %.1f score %.1f write %.1f) | total %.1f ms | ep %s%s%s%s",
             args.time, ml::segmentationModel(sfModel), cp.sf.subjectThreshold, cp.sf.maxHolePercent,
             args.renderScale.x, cr.sf.holesFound, cr.sf.holesFilled, cr.sf.holesKept, cr.sf.holesTooLarge,
             cr.sf.pixelsChanged, cr.sf.segmented ? "yes" : "no", seg.runMs, sfCs.roundTripMs, sfCs.runMs, seg.prepMs,
             seg.postMs, cr.sf.totalMs - cr.sf.segMs, cr.sf.labelMs, cr.sf.scoreMs, cr.sf.writeMs, cr.sf.totalMs,
             sfCs.ep.empty() ? "-" : sfCs.ep.c_str(), cr.sf.aborted ? " | aborted by the host" : "",
             cr.sfFailed ? " | FAILED: " : "", cr.sfFailed ? cr.sfError.c_str() : "");
        // Debug level: every hole's size, place and score, to tune Subject Threshold on a shot.
        for (const ml::HoleInfo& h : cr.sf.holes)
          LOGD("smart fill hole: area %lld px, box %d,%d-%d,%d, segmentation %.3f, colour %.3f, score %.3f -> %s",
               h.area, h.x0, h.y0, h.x1, h.y1, h.segMean, h.colour, h.score, ml::decisionName(h.decision));
      }
      if (cr.erRan) {
        ran = !cr.erFailed;
        if (cr.erFailed) {
          warning = (cr.sfFailed ? warning + " " : std::string()) + "Edge Refine: " + cr.erError +
                    ". The matte passes through.";
          error = true;
          LOGE("ml: refine FAILED: %s (runner %s)", cr.erError.c_str(), ro.runnerPath.c_str());
        } else {
          const ml::RunnerClientStats& cs = client_->stats();
          if (cs.ep == "cpu" && !cs.why.empty() && warning.empty())
            warning = "Edge Refine runs on the CPU (slow): " + cs.why;
          if (st.aborted) LOGI("ml: render aborted by the host after %d of %d tiles", st.tilesRun, st.tiles);
        }
        if (client_->stats().sessionCreated)
          LOGI("ml: the runner (pid %d) created a session for model %s: load %.0f ms, ep %s%s%s", client_->runnerPid(),
               kModelNames[model], client_->stats().loadMs, client_->stats().ep.c_str(),
               client_->stats().why.empty() ? "" : " because: ", client_->stats().why.c_str());
      }
    }
    if (view == kViewTrimap && trimap_.empty() && src) {
      std::vector<float> sd;
      ml::buildTrimap(mid_.data(), w, h, std::max(p.bandPx, 0.5), &sd, &trimap_, p.outerPx);  // Edge Refine's input
    }
    marks_.clear();
    if (view == kViewHoles && cr.sfRan && !cr.sfFailed && !cr.sf.aborted)
      ml::markHoles(tmOut_.data(), w, h, ml::smartFillParams(cp.sf, cp.renderScale).solidAbove, cr.sf, &marks_);

    const auto tw = std::chrono::steady_clock::now();
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
        const float a = inside ? result_[i] : 0.0f;
        if (dc == 1) {
          px[0] = a;
          continue;
        }
        if (!inside) {
          px[0] = px[1] = px[2] = 0.0f;
        } else if (view == kViewChanges) {
          ml::changesViewRgb(matte_[i], a, px);  // plugin 1's Show Changes colours
        } else if (view == kViewHoles) {
          ml::holesViewRgb(marks_.empty() ? uint8_t(ml::kMarkNone) : marks_[i], a, px);
        } else if (view == kViewMatte) {
          px[0] = px[1] = px[2] = a;
        } else if (view == kViewTrimap) {
          px[0] = px[1] = px[2] = trimap_.empty() ? a : trimap_[i];
        } else {
          const auto* sp = static_cast<const float*>(src->getPixelAddress(x, y));
          px[0] = sp[0];
          px[1] = sc >= 3 ? sp[1] : sp[0];
          px[2] = sc >= 3 ? sp[2] : sp[0];
        }
        if (dc == 4) px[3] = a;
      }
    }
    const double writeMs = msSince(tw);
    if (!tmFooter.empty()) {  // a Temporal failure leads; its "no neighbours" note only on an empty footer
      if (tmError) {
        warning = tmFooter + (warning.empty() ? std::string() : " " + warning);
        error = true;
      } else if (warning.empty()) {
        warning = tmFooter;
      }
    }
    try {
      if (warning.empty())
        clearPersistentMessage();
      else
        setPersistentMessage(error ? OFX::Message::eMessageError : OFX::Message::eMessageWarning, "", warning);
    } catch (const std::exception& e) {
      LOGW("render: persistent message FAILED: %s", e.what());
    }
    // One INFO line per render for the Flame test (phase 1): what blocks the main thread, and for
    // how long. No image data, no clip or project names.
    const ml::RunnerClientStats cs = client_ ? client_->stats() : ml::RunnerClientStats();
    LOGI("render t=%g %dx%d scale=%g view=%s enable=%d model=%s band=%.3g outer=%.3g mix=%.3g grey=%d | ep=%s session=%s "
         "| tiles %d/%d (T %d) band %.2f%% changed %lld | trimap %.1f prep %.1f run %.1f post %.1f blend %.1f "
         "core %.1f ms | runner pid %d: round trip %.1f model %.1f queue %.1f copy %.1f connect %.1f ms, "
         "reconnects %d starts %d replaced %d | read %.1f write %.1f render %.1f ms | colour %s, %s | picture %s "
         "| aborted=%s%s%s",
         args.time, w, h, args.renderScale.x, kViewNames[std::min(std::max(view, 0), kViewLast)], int(cp.edgeRefine),
         kModelNames[model], p.bandPx, p.outerPx, p.mix, int(onlyGrey), cs.ep.empty() ? "-" : cs.ep.c_str(),
         cs.sessionCreated ? ("new " + std::to_string(int(cs.loadMs)) + " ms").c_str() : ran ? "reused" : "-",
         st.tilesRun, st.tiles, st.tileSize, 100.0 * st.bandFraction, st.pixelsChanged, st.trimapMs, st.prepMs,
         st.runMs, st.postMs, st.blendMs, st.totalMs, client_ ? client_->runnerPid() : -1, cs.roundTripMs, cs.runMs,
         cs.queueMs, cs.copyMs, cs.connectMs, cs.reconnects, cs.runnerStarts, cs.replaced, readMs, writeMs,
         msSince(t0), kColourNames[colourMode()], lastRequest().c_str(), picture.c_str(), st.aborted ? "yes" : "no",
         warning.empty() ? "" : " | ", warning.c_str());
  }

  // Decision 23: the picture passing through is tagged with the colourspace it arrives in (see
  // outputColourspaceAnswer: OBSERVED Flame 2027.2 behaviour, decision 34); a matte on RGB is
  // tagged as a matte ("Matte" if the config has it, else "Raw").
  // The answer is logged at INFO when it changes.
  // When the host re-asks: the spec (ofxColour.h, kOfxImageEffectActionGetOutputColourspace) says
  // a host relying on the output colourspace must call it whenever the instance changes, after
  // GetClipPreferences. Picture Colour is a clip-preferences slave param (describe), which per
  // ofxImageEffect.h triggers GetClipPreferences only; no Support-library call asks the host to
  // re-run GetOutputColourspace. Flame re-asked on a View change (observed); on a Picture Colour
  // change: VERIFIED (decision 34).
  std::string outputColourspace() {
    LogScope scope(logNode_);
    const std::string cs = outputColourspaceFor();
    std::lock_guard<std::mutex> lock(requestMutex_);
    if (!outputLogged_ || cs != lastOutputCs_) {
      LOGI("colour: output colourspace answer '%s'%s", cs.c_str(), cs.empty() ? " (reply default)" : "");
      lastOutputCs_ = cs;
      outputLogged_ = true;
    }
    return cs;
  }

  std::string outputColourspaceFor() {
    int view = kViewPicture;
    view_->getValue(view);
    if (view == kViewPicture && src_->isConnected())
      return outputColourspaceAnswer(true, true, colourMode() == kColourBypass, "", false);
    const std::string style = getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false);
    const bool matte = style == kOfxImageEffectColourManagementOCIO && configHasMatte();
    return outputColourspaceAnswer(false, src_->isConnected(), false, style, matte);
  }

  // Picture Colour (GetClipPreferences, after the Support library has filled out): Auto requests
  // the picture's colourspaces on OfxImageClipPropPreferredColourspaces_Source (OpenFX 1.5
  // ofxColour.h; on the out args, where Flame honours it; on the clip descriptor Flame refuses it);
  // Bypass requests nothing. st = the Support library's status.
  OfxStatus requestPicturePref(OfxPropertySetHandle out, OfxStatus st) {
    LogScope scope(logNode_);
    const int mode = colourMode();
    const std::string style = getPropertySet().propGetString(kOfxImageEffectPropColourManagementStyle, false);
    std::string req = mode == kColourBypass ? "no request" : "no request (no Source)", set;
    if (mode == kColourAuto && src_->isConnected()) {
      const std::vector<std::string> names(std::begin(kPicturePref), std::end(kPicturePref));
      req = "requested " + joinNames(names);
      set = ": " + setStrings(out, std::string(kOfxImageClipPropPreferredColourspaces) + "_" +
                                       kOfxImageEffectSimpleSourceClipName, names);
      if (st == kOfxStatReplyDefault) st = kOfxStatOK;  // out args are read only on OK
    }
    LOGI("colour: clip preferences, Picture Colour %s, style '%s', %s%s", kColourNames[mode], style.c_str(),
         req.c_str(), set.c_str());
    // What the Source clip reports at this point (probe: which colourspace Flame hands us).
    LOGI("colour: clip preferences inputs: Source %s",
         src_->isConnected()
             ? ("connected cs=" + getStrings(src_->getPropertySet().propSetHandle(), kOfxImageClipPropColourspace)).c_str()
             : "not connected");
    std::lock_guard<std::mutex> lock(requestMutex_);
    lastRequest_ = req;
    return st;
  }

 private:
  // A neighbour with the centre's pixels (a host that ignores the time): every 97th pixel compared.
  bool samePixels(const ml::FetchedFrame& f) const {
    if (f.matte.size() != matte_.size() || f.rgb.size() != rgb_.size()) return false;
    for (size_t i = 0; i < matte_.size(); i += 97)
      if (f.matte[i] != matte_[i] || f.rgb[i * 3] != rgb_[i * 3] || f.rgb[i * 3 + 1] != rgb_[i * 3 + 1] ||
          f.rgb[i * 3 + 2] != rgb_[i * 3 + 2])
        return false;
    return true;
  }

  ml::FrameRange sourceRange() {
    ml::FrameRange r;
    try {
      const OfxRangeD fr = src_->getFrameRange();
      r.first = fr.min;
      r.last = fr.max;
      r.valid = fr.max >= fr.min;
    } catch (...) {  // the host does not report it: no clamping (the log says "unknown")
    }
    return r;
  }

  static std::string rangeText(const ml::FrameRange& r) {
    char b[64];
    if (r.valid) std::snprintf(b, sizeof b, "%g-%g", r.first, r.last);
    else std::snprintf(b, sizeof b, "unknown");
    return b;
  }

  // Temporal's INFO line per frame (what the host returned for each neighbour, where the time went,
  // getFramesNeeded since the last render) and its footer: a failure (error), no neighbour frames.
  std::string logTemporal(const OFX::RenderArguments& args, const ml::ChainParams& cp, const ml::Neighbours& nb,
                          const ml::ChainResult& cr, const std::string& centreUid, bool* isError) {
    const ml::StabiliseParams tp = ml::temporalParams(cp.tm, cp.renderScale);
    std::string each;
    for (const ml::NeighbourReport& n : nb.report) {
      char b[200];
      std::snprintf(b, sizeof b, "%s t=%g (%+d) %s %dx%d %dch %.1f ms%s%s", each.empty() ? "" : ";", n.time, n.offset,
                    n.status.c_str(), n.w, n.h, n.components, n.ms, n.note.empty() ? "" : " ", n.note.c_str());
      each += b;
    }
    int fnCalls = 0;
    std::string fnLast;
    {
      std::lock_guard<std::mutex> lock(requestMutex_);
      fnCalls = fnCalls_;
      fnLast = fnLast_.empty() ? "never called" : fnLast_;
      fnCalls_ = 0;
    }
    const ml::StabiliseStats& s = cr.tm;
    LOGI("render t=%g temporal: K %d strength %.3g band %.3g px (scale %g) | clip range %s | neighbours requested %d "
         "fetched %d (used by the core %d): [%s] fetch %.1f ms | flow %.1f region %.1f combine %.1f core %.1f ms (flow "
         "scale %d) | region %lld px, changed %lld, rejected %.1f%% | holes %d filled %d | centre uid '%s' | "
         "getFramesNeeded %d call(s) since the last render, last: %s%s%s%s",
         args.time, tp.frames, tp.strength, tp.bandPx, cp.renderScale, rangeText(sourceRange()).c_str(), nb.requested,
         int(nb.offsets.size()), s.neighbours, each.c_str(), nb.fetchMs, s.flowMs, s.regionMs, s.combineMs, s.totalMs,
         s.flowScale, s.regionPixels, s.pixelsChanged, 100.0 * s.rejectedFraction, s.holes, s.holesFilled,
         centreUid.c_str(), fnCalls, fnLast.c_str(), nb.aborted || cr.tAborted ? " | aborted by the host" : "",
         cr.tFailed ? " | FAILED: " : "", cr.tFailed ? cr.tError.c_str() : "");
    *isError = false;
    if (cr.tFailed) {
      *isError = true;
      LOGE("ml: temporal FAILED: %s", cr.tError.c_str());
      return "Temporal: " + cr.tError + ". The matte passes through this step.";
    }
    if (nb.aborted || cr.tAborted) return "";
    if (nb.requested == 0)
      return "Temporal: no neighbour frames in the clip's frame range (" + rangeText(sourceRange()) + ").";
    if (nb.offsets.empty())
      return "Temporal: the host returned no neighbour frames (" + std::to_string(nb.requested) +
             " requested). The matte passes through this step.";
    return "";
  }

  int colourMode() {
    int mode = kColourAuto;
    pictureColour_->getValue(mode);
    return mode == kColourBypass ? kColourBypass : kColourAuto;
  }

  std::string lastRequest() {
    std::lock_guard<std::mutex> lock(requestMutex_);
    return lastRequest_.empty() ? "no clip preferences yet" : lastRequest_;
  }

  // The render log's picture field: the Source clip's colourspace as the host reports it (only
  // logged, never decided from). At debug level also the host's preferred list on the clip, the
  // picture's statistics (strided over the whole frame, exact 5x5 centre mean), the image bounds
  // and, for 4 channels, the alpha's min/mean/max.
  std::string pictureProbe(const OFX::Image* img) {
    if (!src_->isConnected()) return "no source";
    const OfxPropertySetHandle cp = src_->getPropertySet().propSetHandle();
    std::string s = "cs=" + getStrings(cp, kOfxImageClipPropColourspace);
    if (logLevel() != LogLevel::Debug) return s;
    s += " pref=" + getStrings(cp, kOfxImageClipPropPreferredColourspaces);
    if (!img) return s + " no image";
    const OfxRectI b = img->getBounds();
    const int w = b.x2 - b.x1, h = b.y2 - b.y1, c = componentCount(img->getPixelComponents());
    if (w <= 0 || h <= 0) return s + " empty";
    const int stride = std::max(1, int(std::ceil(std::sqrt(double(w) * h / 250000.0))));
    double mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30}, sum[3] = {0, 0, 0}, ctr[3] = {0, 0, 0};
    double amn = 1e30, amx = -1e30, asum = 0;
    long long n = 0, bad = 0, nc = 0, na = 0;
    auto at = [&](int x, int y, int k) {
      const auto* p = static_cast<const float*>(img->getPixelAddress(x, y));
      return double(p[c >= 3 ? k : 0]);
    };
    for (int y = b.y1; y < b.y2; y += stride)
      for (int x = b.x1; x < b.x2; x += stride) {
        if (c == 4) {
          const double a = static_cast<const float*>(img->getPixelAddress(x, y))[3];
          if (std::isfinite(a)) {
            amn = std::min(amn, a);
            amx = std::max(amx, a);
            asum += a;
            ++na;
          }
        }
        double v[3] = {at(x, y, 0), at(x, y, 1), at(x, y, 2)};
        if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
          ++bad;
          continue;
        }
        for (int k = 0; k < 3; ++k) {
          mn[k] = std::min(mn[k], v[k]);
          mx[k] = std::max(mx[k], v[k]);
          sum[k] += v[k];
        }
        ++n;
      }
    const int cx = b.x1 + w / 2, cy = b.y1 + h / 2;
    for (int y = std::max(b.y1, cy - 2); y <= std::min(b.y2 - 1, cy + 2); ++y)
      for (int x = std::max(b.x1, cx - 2); x <= std::min(b.x2 - 1, cx + 2); ++x, ++nc)
        for (int k = 0; k < 3; ++k) ctr[k] += at(x, y, k);
    char t[600];
    if (n == 0) {
      n = 1;
      for (int k = 0; k < 3; ++k) mn[k] = mx[k] = 0;
    }
    std::snprintf(t, sizeof t,
                  " rgb min %.5g %.5g %.5g mean %.5g %.5g %.5g max %.5g %.5g %.5g centre5x5 %.5g %.5g %.5g "
                  "(stride %d, %d ch, non-finite %lld)",
                  mn[0], mn[1], mn[2], sum[0] / n, sum[1] / n, sum[2] / n, mx[0], mx[1], mx[2], ctr[0] / nc, ctr[1] / nc,
                  ctr[2] / nc, stride, c, bad);
    s += t;
    std::snprintf(t, sizeof t, " bounds %d,%d-%d,%d (%dx%d)", b.x1, b.y1, b.x2, b.y2, w, h);
    s += t;
    if (c == 4) {
      if (na == 0) {
        s += " alpha all non-finite";
      } else {
        std::snprintf(t, sizeof t, " alpha min %.5g mean %.5g max %.5g", amn, asum / na, amx);
        s += t;
      }
    }
    return s;
  }

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

  // A support bundle (as Matte Refine's: the log is shared).
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

  OFX::Clip *src_, *dst_;
  OFX::ChoiceParam *view_, *pictureColour_, *model_, *sfModel_, *tmFrames_;
  OFX::BooleanParam *enable_, *onlyGrey_, *debugLog_, *sfEnable_, *tmEnable_;
  OFX::DoubleParam *band_, *outer_, *mix_, *sfThreshold_, *sfMaxHole_, *tmStrength_, *tmBand_;
  std::string nodeId_, logNode_;
  bool inChangedParam_ = false;
  bool matteChecked_ = false, hasMatte_ = false;
  std::mutex requestMutex_;
  std::string lastRequest_;  // what the last GetClipPreferences requested for the picture
  std::string lastOutputCs_;  // the last GetOutputColourspace answer (logged when it changes)
  bool outputLogged_ = false;
  std::vector<float> matte_, rgb_, mid_, result_, trimap_;  // reused across renders; mid_ = after Smart Fill
  std::vector<float> tmOut_;  // after Temporal (Smart Fill's input)
  std::vector<ml::FetchedFrame> tmPool_;  // neighbour buffers, reused across renders
  int fnCalls_ = 0, fnTotal_ = 0;  // getFramesNeeded calls since the last Temporal render / in all (requestMutex_)
  std::string fnLast_;
  std::vector<uint8_t> marks_;  // view Holes
  std::unique_ptr<ml::RunnerClient> client_;  // this node's connection to mr_ml_runner
  std::string gateLogged_;                     // the setup gate's last logged detail ("" = open)
  std::string clientPath_;
  double clientTimeout_ = 0;
};

template <class D>
D* labelled(D* p, const char* label, const char* hint) {
  p->setLabels(label, label, label);
  if (hint && *hint) p->setHint(hint);
  return p;
}

class MatteRefineMLFactory : public OFX::PluginFactoryHelper<MatteRefineMLFactory> {
 public:
  // GetOutputColourspace (OpenFX 1.5) is unknown to the 1.4 Support library: answered here.
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in,
                         OfxPropertySetHandle out) {
    if (std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0) {
      try {
        auto* fx = dynamic_cast<MatteRefineMLEffect*>(
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
    if (std::strcmp(action, kOfxImageEffectActionGetClipPreferences) == 0) {
      const OfxStatus st = FactoryMainEntryHelper<MatteRefineMLFactory>::mainEntry(action, handle, in, out);
      if (st != kOfxStatOK && st != kOfxStatReplyDefault) return st;
      try {  // Picture Colour: the requested picture colourspaces
        auto* fx = dynamic_cast<MatteRefineMLEffect*>(
            OFX::Private::retrieveImageEffectPointer((OfxImageEffectHandle)handle));
        return fx ? fx->requestPicturePref(out, st) : st;
      } catch (const std::exception& e) {
        LOGE("colour: clip preferences FAILED: %s", e.what());
        return st;
      }
    }
    return FactoryMainEntryHelper<MatteRefineMLFactory>::mainEntry(action, handle, in, out);
  }
  OfxPluginEntryPoint* getMainEntry() override { return entry; }

  MatteRefineMLFactory()
      : PluginFactoryHelper<MatteRefineMLFactory>(kPluginId, MRML_VERSION_MAJOR, MRML_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LogScope scope("ml");
    LOGI("==== Matte Refine ML %s (%s) load (describe), runner protocol %u, default runner %s ====", MRML_VERSION,
         MR_GIT, ml::proto::kVersion, defaultRunnerPath().c_str());
    for (const auto& line : environmentSnapshot(hostInfo())) LOGI("env %s", line.c_str());
    desc.setLabels(kPluginLabel, kPluginLabel, kPluginLabel);
    desc.setPluginGrouping("Matte Refine");
    // Decision 35: a filter, as LTX Alpha Gen (its Source = Flame's front + source alpha sockets).
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    // Temporal (B4): neighbour frames from Source (as LTX Alpha Gen). The host's capability is logged.
    desc.setTemporalClipAccess(true);
    const OFX::ImageEffectHostDescription* hd = OFX::getImageEffectHostDescription();
    LOGI("host: temporalClipAccess %s", hd ? (hd->temporalClipAccess ? "yes" : "NO") : "unknown");
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementStyle,
                                        kOfxImageEffectColourManagementOCIO, false);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementAvailableConfigs,
                                        "ofx-native-v1.5_aces-v1.3_ocio-v2.3", false);
  }

  void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum ctx) override {
    LogScope scope("ml");
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
    src->setSupportsTiles(false);  // decision 35: as LTX Alpha Gen's Source (mandatory)
    src->setTemporalClipAccess(true);  // Temporal (B4) fetches t +- k
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    // Flame: one group = one column of <= 5 rows (choice 2, others 1; a hidden param takes a
    // slot too). Columns, in chain order: Matte (5) · Temporal (5) · Smart Fill (5) · Edge Refine (5) · Model &
    // Support (5).
    OFX::PageParamDescriptor* page = desc.definePageParam("Edge Refine");
    auto group = [&](const char* name, const char* text) {
      auto* g = desc.defineGroupParam(name);
      g->setLabels(text, text, text);
      g->setOpen(true);
      page->addChild(*g);
      return g;
    };
    auto number = [&](const char* name, const char* text, const char* hint, double def, double lo,
                      double hi, int digits, double increment, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text, hint);
      p->setRange(lo, hi);
      p->setDisplayRange(lo, hi);
      p->setDigits(digits);  // Flame showed doubles without digits as integers
      p->setIncrement(increment);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto boolean = [&](const char* name, const char* text, const char* hint, bool def,
                       OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineBooleanParam(name), text, hint);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };

    const UserConfig cfg = loadUserConfig();  // the model folder, for the model choices' labels
    const auto md = cfg.find("ml_model_dir");
    const std::string modelDir =
        md == cfg.end() || md->second.empty() ? ml::defaultModelDir(homeDir()) : expandHome(md->second);
    LOGI("describe: model folder %s: %s %s, %s %s", modelDir.c_str(), kModelDirs[1],
         ml::modelInstalled(modelDir, kModelDirs[1]) ? "installed" : "NOT installed", ml::segmentationModel(1),
         ml::modelInstalled(modelDir, ml::segmentationModel(1)) ? "installed" : "NOT installed");

    auto* gMatte = group("grpMatte", "Matte");
    auto* view = labelled(desc.defineChoiceParam(kPView), "View",
                          "What the image output shows (the matte output is always the refined matte). "
                          "Picture: the Source input. Matte: the refined matte. Trimap: solid / edge band / "
                          "clear. Changes: the input matte in grey, added alpha magenta, removed green. "
                          "Holes: the refined matte in grey, holes Smart Fill filled in green, holes it kept "
                          "as real gaps in red.");
    for (const char* c : kViewNames) view->appendOption(c);
    view->setDefault(kViewPicture);
    view->setAnimates(false);
    view->setParent(*gMatte);
    // Stored as an index in setups: new options at the end. It changes what the picture input is
    // asked for, so the host re-runs GetClipPreferences when it changes.
    auto* colour = labelled(desc.defineChoiceParam(kPPictureColour), "Picture Colour",
                            "Auto: Flame converts the picture to Rec.709 display before Edge Refine (works for "
                            "log, linear and video sources). Bypass: the picture is used as it arrives.");
    for (const char* c : kColourNames) colour->appendOption(c);
    colour->setDefault(kColourAuto);
    colour->setAnimates(false);
    colour->setParent(*gMatte);
    desc.addClipPreferencesSlaveParam(*colour);
    auto* version = labelled(desc.defineStringParam("version"), "Version", "");
    version->setStringType(OFX::eStringTypeLabel);
    version->setDefault(MRML_VERSION " (" MR_GIT ")");
    version->setIsPersistant(false);
    version->setEvaluateOnChange(false);
    version->setParent(*gMatte);

    // Temporal (B4) first, then Smart Fill (B3), then Edge Refine: definition order = chain order (render_chain.h).
    auto* gTm = group("grpTemporal", "Temporal");
    const ml::TemporalUi tmDef;  // the defaults (test_render_chain checks them)
    boolean(kPTmEnable, "Enable",
            "Calm the matte's edge flicker and flickering holes with the neighbour frames, motion-compensated with "
            "the picture. Only the edge band and holes change; the rest of the matte stays as it is.",
            false, gTm);
    auto* tmFrames = labelled(desc.defineChoiceParam(kPTmFrames), "Frames",
                              "Neighbour frames on each side: +-1 (faster) or +-2 (calmer, about 2x the time).");
    for (const char* c : kTemporalNames) tmFrames->appendOption(c);  // stored as an index: append only
    tmFrames->setDefault(tmDef.frames);
    tmFrames->setParent(*gTm);
    number(kPTmStrength, "Strength", "0 = the input matte, 1 = the neighbours' combined edge.", tmDef.strength, 0.0,
           1.0, 2, 0.05, gTm);
    number(kPTmBand, "Band px", "Only pixels this close to the matte's edge (and flickering holes) change.",
           tmDef.bandPx, 1.0, 32.0, 1, 0.5, gTm);

    auto* gSf = group("grpSmartFill", "Smart Fill");
    const ml::SmartFillUi sfDef;  // the defaults (test_render_chain checks them)
    boolean(kPSfEnable, "Enable",
            "Fill holes in the matte that are part of the subject; keep real gaps (between an arm and the body, "
            "between legs) open. A segmentation model (BiRefNet) and the picture decide.",
            false, gSf);
    number(kPSfThreshold, "Subject Threshold",
           "A hole is filled only when the picture and the segmentation both say 'subject' at least this "
           "much (0-1). Higher keeps more holes open: 0 fills every hole up to Max Hole %, 1 keeps them all.",
           sfDef.subjectThreshold, 0.0, 1.0, 2, 0.05, gSf);
    number(kPSfMaxHole, "Max Hole %",
           "Holes larger than this percentage of the frame are never filled. 0 = no limit.", sfDef.maxHolePercent, 0.0,
           100.0, 1,
           0.5, gSf);
    auto* seg = labelled(desc.defineChoiceParam(kPSfModel), "Segmentation",
                         "Lite: faster. Full: about 2x slower, more robust on low-contrast pictures.");
    // Stored as an index: append only. The labels say which models are installed (read at describe:
    // Flame describes again at its next start, so a setup is shown after a restart).
    for (int i = 0; i < 2; ++i)
      seg->appendOption(ml::optionLabel(kSegNames[i], ml::modelInstalled(modelDir, ml::segmentationModel(i))));
    seg->setDefault(sfDef.segmentation);
    seg->setParent(*gSf);

    auto* gEr = group("grpEdgeRefine", "Edge Refine");
    boolean(kPEnable, "Enable", "Re-solve the matte's edge band with the ViTMatte model (needs the picture).",
            true, gEr);
    number(kPBand, "Band px", "The model re-solves this many pixels either side of the matte's edge.", 10.0,
           1.0, 64.0, 1, 0.5, gEr);
    // Added after Band px (append-only ids; 0 = the symmetric band of earlier setups).
    number(kPOuter, "Outer px",
           "The band reaches this many pixels further outside the matte's edge, where hair strands the matte lost "
           "can be found again. For hair: about 100-150. 0 = the same on both sides.",
           0.0, 0.0, 256.0, 1, 1.0, gEr);
    number(kPMix, "Mix", "0 = the input matte, 1 = the model's edge.", 1.0, 0.0, 1.0, 2, 0.05, gEr);
    boolean(kPOnlyGrey, "Only Grey", "Change only pixels that were grey (0.02-0.98) in the input matte.", false,
            gEr);

    auto* gSupport = group("grpSupport", "Model & Support");
    auto* model = labelled(desc.defineChoiceParam(kPModel), "Model",
                           "ViTMatte Small (faster) or Base (about 1.2x slower).");
    for (int i = 0; i < 2; ++i)  // as Segmentation's labels
      model->appendOption(ml::optionLabel(kModelNames[i], ml::modelInstalled(modelDir, kModelDirs[i])));
    model->setDefault(0);
    model->setParent(*gSupport);
    auto* dbg = boolean(kPDebugLog, "Debug Logging", "Verbose logging for troubleshooting (per user).", false,
                        gSupport);
    dbg->setIsPersistant(false);
    dbg->setEvaluateOnChange(false);
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
    return new MatteRefineMLEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static MatteRefineMLFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
