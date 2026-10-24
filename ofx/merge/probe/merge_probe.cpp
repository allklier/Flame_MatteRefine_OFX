// Matte Merge input probe (ofx/merge/CLAUDE.md "Inputs: the main risk"): a logging-only plugin, no
// matte processing. It declares the layout planned for Matte Merge: a filter (Source mandatory, the
// picture on the front, the base matte / trimap in its alpha, decisions 4 and 35) plus one optional
// clip "Detail". How Detail is declared is chosen at load time (describe) by the config key
// probe_detail_kind in ~/.config/matte-refine/config.conf:
//   rgba  (default) Detail is a plain optional clip, RGBA / RGB
//   mask            Detail is an optional mask clip (setIsMask), Alpha / RGBA / RGB (as plugin 1's Matte clip)
//   none            no Detail clip (baseline: does a second clip change what Source delivers?)
// Each render logs (INFO) per clip what the host reports and the pixel statistics, passes Source
// through and puts a short summary in the footer (which socket got which input: alpha min / max).
// Colour: OCIO style declared (to see the 1.5 colour properties), nothing requested, the output tagged
// OfxColourspace_Source (pass-through).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

#include "log.h"
#include "ofxColour.h"
#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"
#include "user_config.h"

#ifndef MP_VERSION
#define MP_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif

namespace {

using namespace mr;

constexpr const char* kPluginId = "com.janklier.mergeprobe";
constexpr const char* kPluginLabel = "Merge Probe";
constexpr const char* kDetailClip = "Detail";
constexpr const char* kConfigKey = "probe_detail_kind";

// The variant read in describe (one per plugin load).
std::string gKind = "rgba";

// ---- raw property access (never throws; a failure reads as <status>) ----

std::string statusText(OfxStatus st) { return std::string("<") + OFX::mapStatusToString(st) + ">"; }

std::string propStrings(OfxPropertySetHandle h, const char* prop) {
  if (!h) return "<no handle>";
  int dim = 0;
  const OfxStatus ds = OFX::Private::gPropSuite->propGetDimension(h, prop, &dim);
  if (ds != kOfxStatOK) return statusText(ds);
  std::string s = "[";
  for (int i = 0; i < dim; ++i) {
    char* v = nullptr;
    const OfxStatus st = OFX::Private::gPropSuite->propGetString(h, prop, i, &v);
    s += (i ? ", '" : "'") + (st == kOfxStatOK && v ? std::string(v) : statusText(st)) + "'";
  }
  return s + "]";
}

std::string propDoubles(OfxPropertySetHandle h, const char* prop) {
  if (!h) return "<no handle>";
  int dim = 0;
  const OfxStatus ds = OFX::Private::gPropSuite->propGetDimension(h, prop, &dim);
  if (ds != kOfxStatOK) return statusText(ds);
  std::string s;
  for (int i = 0; i < dim; ++i) {
    double v = 0;
    const OfxStatus st = OFX::Private::gPropSuite->propGetDouble(h, prop, i, &v);
    char b[48];
    std::snprintf(b, sizeof b, "%g", v);
    s += (i ? "," : "") + (st == kOfxStatOK ? std::string(b) : statusText(st));
  }
  return s;
}

std::string propInts(OfxPropertySetHandle h, const char* prop) {
  if (!h) return "<no handle>";
  int dim = 0;
  const OfxStatus ds = OFX::Private::gPropSuite->propGetDimension(h, prop, &dim);
  if (ds != kOfxStatOK) return statusText(ds);
  std::string s;
  for (int i = 0; i < dim; ++i) {
    int v = 0;
    const OfxStatus st = OFX::Private::gPropSuite->propGetInt(h, prop, i, &v);
    s += (i ? "," : "") + (st == kOfxStatOK ? std::to_string(v) : statusText(st));
  }
  return s;
}

// Strips the OFX prefixes so the log stays readable ("OfxImageComponentRGBA" -> "RGBA").
std::string shortName(std::string s) {
  const char* prefixes[] = {"OfxImageComponent", "OfxBitDepth", "OfxImage", "OfxImageField"};
  for (const char* p : prefixes) {
    size_t at;
    while ((at = s.find(p)) != std::string::npos) s.erase(at, std::strlen(p));
  }
  return s;
}

// What the host reports on a clip instance (no pixels).
std::string clipProps(OFX::Clip* c) {
  OfxPropertySetHandle h = c->getPropertySet().propSetHandle();
  std::string s = "connected=" + propInts(h, kOfxImageClipPropConnected);
  s += " label=" + propStrings(h, kOfxPropLabel);
  s += " optional=" + propInts(h, kOfxImageClipPropOptional) + " isMask=" + propInts(h, kOfxImageClipPropIsMask);
  s += " depth=" + propStrings(h, kOfxImageEffectPropPixelDepth) + " comps=" + propStrings(h, kOfxImageEffectPropComponents);
  s += " unmapped=" + propStrings(h, kOfxImageClipPropUnmappedPixelDepth) + "/" +
       propStrings(h, kOfxImageClipPropUnmappedComponents);
  s += " premult=" + propStrings(h, kOfxImageEffectPropPreMultiplication) + " fieldOrder=" +
       propStrings(h, kOfxImageClipPropFieldOrder);
  s += " PAR=" + propDoubles(h, kOfxImagePropPixelAspectRatio) + " rate=" + propDoubles(h, kOfxImageEffectPropFrameRate);
  s += " range=" + propDoubles(h, kOfxImageEffectPropFrameRange) + " unmappedRange=" +
       propDoubles(h, kOfxImageEffectPropUnmappedFrameRange);
  s += " cs=" + propStrings(h, kOfxImageClipPropColourspace) + " preferredCs=" +
       propStrings(h, kOfxImageClipPropPreferredColourspaces);
  return shortName(s);
}

// What the host reports on a fetched image.
std::string imageProps(OFX::Image* img) {
  OfxPropertySetHandle h = img->getPropertySet().propSetHandle();
  std::string s = "bounds=" + propInts(h, kOfxImagePropBounds) + " rod=" + propInts(h, kOfxImagePropRegionOfDefinition);
  s += " depth=" + propStrings(h, kOfxImageEffectPropPixelDepth) + " comps=" + propStrings(h, kOfxImageEffectPropComponents);
  s += " premult=" + propStrings(h, kOfxImageEffectPropPreMultiplication) + " field=" +
       propStrings(h, kOfxImagePropField);
  s += " PAR=" + propDoubles(h, kOfxImagePropPixelAspectRatio) + " rowBytes=" + std::to_string(img->getRowBytes());
  s += " uid=" + propStrings(h, kOfxImagePropUniqueIdentifier);
  return shortName(s);
}

// ---- pixel statistics ----

int componentCount(const std::string& comps) {
  if (comps == kOfxImageComponentRGBA) return 4;
  if (comps == kOfxImageComponentRGB) return 3;
  if (comps == kOfxImageComponentAlpha) return 1;
  return 0;
}

struct Stats {
  int comps = 0;
  long long n = 0, nonFinite = 0;
  double mn[4], mx[4], sum[4];
  uint64_t rgbHash = 1469598103934665603ull;  // FNV-1a 64 over the RGB float bits, rows bottom-up
  Stats() {
    for (int c = 0; c < 4; ++c) mn[c] = 1e300, mx[c] = -1e300, sum[c] = 0;
  }
  // Channel index for R, G, B, A (-1 if the image has no such channel; Alpha-only: A = 0).
  int index(int ch) const {
    if (comps == 4) return ch;
    if (comps == 3) return ch < 3 ? ch : -1;
    if (comps == 1) return ch == 3 ? 0 : -1;
    return -1;
  }
  std::string channel(int ch) const {
    const int i = index(ch);
    if (i < 0 || n == 0) return "-";
    char b[96];
    std::snprintf(b, sizeof b, "%.4f/%.4f/%.4f", mn[i], sum[i] / n, mx[i]);
    return b;
  }
  std::string alphaRange() const {
    const int i = index(3);
    if (i < 0 || n == 0) return "none";
    char b[64];
    std::snprintf(b, sizeof b, "%.3f-%.3f", mn[i], mx[i]);
    return b;
  }
  std::string hash() const {
    if (comps < 3) return "-";
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)rgbHash);
    return b;
  }
};

// Float images only (the plugin declares float only); others are counted but not read.
Stats measure(OFX::Image* img) {
  Stats s;
  OfxPropertySetHandle h = img->getPropertySet().propSetHandle();
  char* comps = nullptr;
  char* depth = nullptr;
  OFX::Private::gPropSuite->propGetString(h, kOfxImageEffectPropComponents, 0, &comps);
  OFX::Private::gPropSuite->propGetString(h, kOfxImageEffectPropPixelDepth, 0, &depth);
  s.comps = comps ? componentCount(comps) : 0;
  if (!depth || std::strcmp(depth, kOfxBitDepthFloat) != 0 || s.comps == 0) {
    s.comps = 0;
    return s;
  }
  const OfxRectI b = img->getBounds();
  for (int y = b.y1; y < b.y2; ++y) {
    const float* p = static_cast<const float*>(img->getPixelAddress(b.x1, y));
    if (!p) continue;
    for (int x = b.x1; x < b.x2; ++x, p += s.comps) {
      ++s.n;
      for (int c = 0; c < s.comps; ++c) {
        const float v = p[c];
        if (!std::isfinite(v)) {
          ++s.nonFinite;
          continue;
        }
        if (v < s.mn[c]) s.mn[c] = v;
        if (v > s.mx[c]) s.mx[c] = v;
        s.sum[c] += v;
      }
      if (s.comps >= 3) {
        uint32_t bits[3];
        std::memcpy(bits, p, sizeof bits);
        for (uint32_t w : bits)
          for (int k = 0; k < 4; ++k) s.rgbHash = (s.rgbHash ^ ((w >> (8 * k)) & 0xffu)) * 1099511628211ull;
      }
    }
  }
  return s;
}

std::string statsText(const Stats& s) {
  if (s.comps == 0) return "pixels not read (not float or unknown components)";
  return "pixels n=" + std::to_string(s.n) + " nonFinite=" + std::to_string(s.nonFinite) + " R=" + s.channel(0) +
         " G=" + s.channel(1) + " B=" + s.channel(2) + " A=" + s.channel(3) + " (min/mean/max) rgbHash=" + s.hash();
}

std::string randomId() {
  std::random_device rd;
  char b[9];
  std::snprintf(b, sizeof b, "%08x", rd());
  return b;
}

class MergeProbeEffect : public OFX::ImageEffect {
 public:
  explicit MergeProbeEffect(OfxImageEffectHandle handle) : ImageEffect(handle), node_("probe:" + randomId()) {
    LogScope scope(node_);
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    // Fetched whatever the variant says: shows a host that kept an older describe.
    try {
      detail_ = fetchClip(kDetailClip);
    } catch (const std::exception&) {
      detail_ = nullptr;
    }
    LOGI("createInstance %p context=%s, variant (describe) %s, Detail clip %s", static_cast<void*>(this),
         getPropertySet().propGetString(kOfxImageEffectPropContext, false).c_str(), gKind.c_str(),
         detail_ ? "present" : "absent");
  }

  ~MergeProbeEffect() override {
    LogScope scope(node_);
    LOGI("destroyInstance %p", static_cast<void*>(this));
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    LogScope scope(node_);
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
    LOGI("clip preferences: style %s, OCIO config %s", propStrings(effectProps(), kOfxImageEffectPropColourManagementStyle).c_str(),
         propStrings(effectProps(), kOfxImageEffectPropOCIOConfig).c_str());
    for (OFX::Clip* c : clips()) LOGI("clip preferences: %s %s", c->name().c_str(), clipProps(c).c_str());
  }

  void changedClip(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    LogScope scope(node_);
    LOGI("changedClip %s reason=%d time=%g", name.c_str(), int(args.reason), args.time);
  }

  void render(const OFX::RenderArguments& args) override {
    LogScope scope(node_);
    ++renders_;
    LOGI("render #%d t=%g scale=%g,%g window=%d,%d-%d,%d field=%d interactive=%d sequential=%d (variant %s)", renders_,
         args.time, args.renderScale.x, args.renderScale.y, args.renderWindow.x1, args.renderWindow.y1,
         args.renderWindow.x2, args.renderWindow.y2, int(args.fieldToRender), int(args.interactiveRenderStatus),
         int(args.sequentialRenderStatus), gKind.c_str());
    LOGI("render effect: style %s, OCIO config %s, display cs %s",
         propStrings(effectProps(), kOfxImageEffectPropColourManagementStyle).c_str(),
         propStrings(effectProps(), kOfxImageEffectPropOCIOConfig).c_str(),
         propStrings(effectProps(), kOfxImageEffectPropDisplayColourspace).c_str());

    std::unique_ptr<OFX::Image> out(dst_->fetchImage(args.time));
    if (!out) {
      LOGE("render t=%g: no output image", args.time);
      OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    LOGI("render clip Output: %s", clipProps(dst_).c_str());
    LOGI("render image Output: %s", imageProps(out.get()).c_str());

    std::unique_ptr<OFX::Image> src;
    std::string footer = std::string("Probe ") + gKind + " " + MR_GIT + ":";
    for (OFX::Clip* c : clips()) {
      LOGI("render clip %s: %s", c->name().c_str(), clipProps(c).c_str());
      std::string part;
      if (!c->isConnected()) {
        part = "off";
      } else {
        std::unique_ptr<OFX::Image> img(c->fetchImage(args.time));
        if (!img) {
          LOGI("render image %s: fetchImage returned none", c->name().c_str());
          part = "no image";
        } else {
          const Stats s = measure(img.get());
          LOGI("render image %s: %s", c->name().c_str(), imageProps(img.get()).c_str());
          LOGI("render image %s: %s", c->name().c_str(), statsText(s).c_str());
          char b[160];
          std::snprintf(b, sizeof b, "%dch a %s rgb max %s #%.6s", s.comps, s.alphaRange().c_str(),
                        s.comps >= 3 ? fmtMax(s).c_str() : "-", s.hash().c_str());
          part = b;
          if (c == src_) src = std::move(img);
        }
      }
      footer += std::string(" | ") + c->name() + " " + part;
    }
    copySource(src.get(), out.get(), args.renderWindow);
    if (footer != lastFooter_) {
      LOGI("footer: %s", footer.c_str());
      setPersistentMessage(OFX::Message::eMessageWarning, "", footer);
      lastFooter_ = footer;
    }
  }

  // GetOutputColourspace (OpenFX 1.5): pass-through, so the Source's colourspace.
  std::string outputColourspace() {
    LogScope scope(node_);
    LOGI("GetOutputColourspace asked: answer OfxColourspace_Source");
    return "OfxColourspace_Source";
  }

 private:
  OfxPropertySetHandle effectProps() { return getPropertySet().propSetHandle(); }

  std::vector<OFX::Clip*> clips() {
    std::vector<OFX::Clip*> v{src_};
    if (detail_) v.push_back(detail_);
    return v;
  }

  static std::string fmtMax(const Stats& s) {
    char b[64];
    std::snprintf(b, sizeof b, "%.3f", std::max(s.mx[0], std::max(s.mx[1], s.mx[2])));
    return b;
  }

  // Source -> Output in the render window (RGB or RGBA float in, RGBA float out); zero elsewhere.
  static void copySource(OFX::Image* src, OFX::Image* out, const OfxRectI& win) {
    if (out->getPixelDepth() != OFX::eBitDepthFloat || out->getPixelComponents() != OFX::ePixelComponentRGBA) {
      LOGW("output is not float RGBA: not written");
      return;
    }
    int sc = 0;
    OfxRectI sb{0, 0, 0, 0};
    if (src && src->getPixelDepth() == OFX::eBitDepthFloat) {
      sb = src->getBounds();
      sc = src->getPixelComponents() == OFX::ePixelComponentRGBA  ? 4
           : src->getPixelComponents() == OFX::ePixelComponentRGB ? 3
           : src->getPixelComponents() == OFX::ePixelComponentAlpha ? 1
                                                                   : 0;
    }
    const OfxRectI ob = out->getBounds();
    for (int y = std::max(win.y1, ob.y1); y < std::min(win.y2, ob.y2); ++y) {
      for (int x = std::max(win.x1, ob.x1); x < std::min(win.x2, ob.x2); ++x) {
        float* q = static_cast<float*>(out->getPixelAddress(x, y));
        const bool in = sc && x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
        const float* p = in ? static_cast<const float*>(src->getPixelAddress(x, y)) : nullptr;
        if (!p) {
          q[0] = q[1] = q[2] = q[3] = 0.0f;
        } else if (sc == 1) {
          q[0] = q[1] = q[2] = q[3] = p[0];
        } else {
          q[0] = p[0], q[1] = p[1], q[2] = p[2], q[3] = sc == 4 ? p[3] : 1.0f;
        }
      }
    }
  }

  std::string node_;
  OFX::Clip *src_ = nullptr, *dst_ = nullptr, *detail_ = nullptr;
  int renders_ = 0;
  std::string lastFooter_;
};

class MergeProbeFactory : public OFX::PluginFactoryHelper<MergeProbeFactory> {
 public:
  // GetOutputColourspace is unknown to the 1.4 Support library: answered here (as Matte Refine ML).
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    if (std::strcmp(action, kOfxImageEffectActionGetOutputColourspace) == 0) {
      try {
        auto* fx = dynamic_cast<MergeProbeEffect*>(OFX::Private::retrieveImageEffectPointer((OfxImageEffectHandle)handle));
        if (!fx) return kOfxStatReplyDefault;
        OFX::PropertySet(out).propSetString(kOfxImageClipPropColourspace, fx->outputColourspace());
        return kOfxStatOK;
      } catch (const std::exception& e) {
        LOGE("GetOutputColourspace FAILED: %s", e.what());
        return kOfxStatFailed;
      }
    }
    return FactoryMainEntryHelper<MergeProbeFactory>::mainEntry(action, handle, in, out);
  }
  OfxPluginEntryPoint* getMainEntry() override { return entry; }

  MergeProbeFactory() : PluginFactoryHelper<MergeProbeFactory>(kPluginId, 0, 1) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    LogScope scope("probe");
    const UserConfig cfg = loadUserConfig();
    const auto it = cfg.find(kConfigKey);
    const std::string asked = it == cfg.end() ? "" : it->second;
    gKind = asked == "mask" || asked == "none" ? asked : "rgba";
    LOGI("==== Merge Probe %s (%s) load (describe): %s='%s' -> variant %s ====", MP_VERSION, MR_GIT, kConfigKey,
         asked.c_str(), gKind.c_str());
    if (const OFX::ImageEffectHostDescription* d = OFX::getImageEffectHostDescription()) {
      LOGI("host: %s '%s' %s (%d.%d.%d), API %d.%d", d->hostName.c_str(), d->hostLabel.c_str(), d->versionLabel.c_str(),
           d->versionMajor, d->versionMinor, d->versionMicro, d->APIVersionMajor, d->APIVersionMinor);
      LOGI("host: multipleClipDepths %d multipleClipPARs %d multiResolution %d tiles %d temporalClipAccess %d "
           "setableFielding %d", d->supportsMultipleClipDepths, d->supportsMultipleClipPARs, d->supportsMultiResolution,
           d->supportsTiles, d->temporalClipAccess, d->supportsSetableFielding);
      std::string ctx, comps;
      for (auto c : d->_supportedContexts) ctx += std::to_string(int(c)) + " ";
      for (auto c : d->_supportedComponents) comps += std::to_string(int(c)) + " ";
      LOGI("host: contexts (OFX::ContextEnum) %s; components (OFX::PixelComponentEnum) %s", ctx.c_str(), comps.c_str());
    }
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
    // OCIO style as Matte Refine ML (decision 23): only so the host reports the colour properties.
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementStyle, kOfxImageEffectColourManagementOCIO,
                                        false);
    desc.getPropertySet().propSetString(kOfxImageEffectPropColourManagementAvailableConfigs,
                                        "ofx-native-v1.5_aces-v1.3_ocio-v2.3", false);
  }

  void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum ctx) override {
    LogScope scope("probe");
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);
    if (gKind != "none") {
      OFX::ClipDescriptor* d = desc.defineClip(kDetailClip);
      if (gKind == "mask") {
        d->addSupportedComponent(OFX::ePixelComponentAlpha);
        d->addSupportedComponent(OFX::ePixelComponentRGBA);
        d->addSupportedComponent(OFX::ePixelComponentRGB);
        d->setIsMask(true);
      } else {
        d->addSupportedComponent(OFX::ePixelComponentRGBA);
        d->addSupportedComponent(OFX::ePixelComponentRGB);
      }
      d->setOptional(true);
      d->setSupportsTiles(false);
      d->setLabels("Detail", "Detail", "Detail");
    }
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);
    LOGI("describeInContext ctx=%d: Source (mandatory, RGBA/RGB)%s, Output RGBA", int(ctx),
         gKind == "none"   ? ", no Detail"
         : gKind == "mask" ? ", Detail (optional, mask, Alpha/RGBA/RGB)"
                           : ", Detail (optional, RGBA/RGB)");
  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
    return new MergeProbeEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static MergeProbeFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
