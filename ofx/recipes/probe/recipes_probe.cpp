// Edge Recipes Flame probe (docs/edge_recipes.md 4.9, work package 0b): a test tool, not a product.
// One binary exports THREE effects (grouping "Edge Recipes"), each built from its own factory type
// ProbeFactory<Tag> (docs/edge_recipes.md 4.1: the Support library keeps the uid in a static member
// per factory TYPE, ofxsImageEffect.h:206-220, so one class per effect). Each effect is visibly
// different, so a node bound to the wrong descriptor shows at once:
//   Probe Alpha  K=101  RGB tinted red,   matte = source alpha        Source only (2 sockets)
//   Probe Beta   K=202  RGB tinted green, matte = 1 - source alpha    + Detail + Focus Region masks
//   Probe Gamma  K=303  RGB tinted blue,  matte = 0.5 * source alpha  + the two masks, tab 1 full
//                                                                     (6 columns) + an "Advanced" page
// Footer: the effect's name and constant, the frame, which clips are connected with their component
// count and bounds (never pixel values), the hidden node id, the saved param values. The "Plan" label
// string and one boolean's label are rewritten per render (only when the text changes).
// No threads, no heavy work: one pass over the render window.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

#include "log.h"
#include "ofxsImageEffect.h"

#ifndef ERP_VERSION
#define ERP_VERSION "0.0.0"
#endif
#ifndef MR_GIT
#define MR_GIT "unknown"
#endif

namespace {

using namespace mr;

constexpr const char* kDetailClip = "Detail";       // as Matte Merge (plugin_merge.cpp:61)
constexpr const char* kRegionClip = "FocusRegion";  // as Matte Merge (plugin_merge.cpp:62)

// Static definition of one probe effect (the real bundle's RecipeDef, scaled down).
struct ProbeDef {
  const char* id;       // plugin id
  const char* label;    // effect name in Flame's menu
  const char* letter;   // short name for the Plan line
  int constant;         // shown in the footer: identifies the descriptor
  float tint[3];        // RGB tint at Strength 1
  int matte;            // 0 alpha, 1 inverted alpha, 2 half alpha
  bool masks;           // Detail + Focus Region mask clips
  bool advanced;        // tab 1 filled to 6 columns + "Advanced" page
  double strength;      // default Strength (differs per effect: a describe-time check)
};

struct Alpha {
  static const ProbeDef& def() {
    static const ProbeDef d{"com.janklier.edgerecipesprobe.alpha", "Probe Alpha", "A", 101, {1.f, .3f, .3f}, 0,
                            false, false, 0.4};
    return d;
  }
};
struct Beta {
  static const ProbeDef& def() {
    static const ProbeDef d{"com.janklier.edgerecipesprobe.beta", "Probe Beta", "B", 202, {.3f, 1.f, .3f}, 1,
                            true, false, 0.5};
    return d;
  }
};
struct Gamma {
  static const ProbeDef& def() {
    static const ProbeDef d{"com.janklier.edgerecipesprobe.gamma", "Probe Gamma", "G", 303, {.3f, .3f, 1.f}, 2,
                            true, true, 0.6};
    return d;
  }
};

// Param ids (append-only, as in the real plugins).
constexpr const char* kPView = "view";
constexpr const char* kPVersion = "version";
constexpr const char* kPStrength = "strength";
constexpr const char* kPEnable = "enable";
constexpr const char* kPMode = "mode";
constexpr const char* kPPlan = "plan";
constexpr const char* kPPlanStatic = "planStatic";
constexpr const char* kPLabelProbe = "labelProbe";
constexpr const char* kPNodeId = "nodeId";
// Gamma only: tab 1 fillers and the Advanced page.
constexpr const char* kGammaDoubles[] = {"mainA", "mainC", "advA", "advD"};
constexpr const char* kGammaBools[] = {"mainB", "mainD", "advB", "advE"};
constexpr const char* kGammaChoices[] = {"colourA", "colourB", "advC"};

void logOnce() {
  static bool done = false;
  if (done) return;
  done = true;
  logConfigure();
}

std::string randomId() {
  std::random_device rd;
  char b[9];
  std::snprintf(b, sizeof b, "%08x", rd());
  return b;
}

std::string fmt2(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f", v);
  return b;
}

class ProbeEffect : public OFX::ImageEffect {
 public:
  ProbeEffect(OfxImageEffectHandle handle, const ProbeDef& def) : ImageEffect(handle), def_(def) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    if (def_.masks) {
      detail_ = fetchClip(kDetailClip);
      region_ = fetchClip(kRegionClip);
    }
    plan_ = fetchStringParam(kPPlan);
    labelProbe_ = fetchBooleanParam(kPLabelProbe);
    OFX::StringParam* nid = fetchStringParam(kPNodeId);
    nid->getValue(nodeId_);
    const bool fresh = nodeId_.empty();
    if (fresh) {
      nodeId_ = randomId();  // saved with the setup (hidden param set in createInstance)
      nid->setValue(nodeId_);
    }
    logNode_ = std::string("erprobe:") + def_.letter + ":" + nodeId_;
    LogScope scope(logNode_);
    LOGI("createInstance %s id=%s K=%d context=%s nodeId %s (%s)", def_.label, def_.id, def_.constant,
         getPropertySet().propGetString(kOfxImageEffectPropContext, false).c_str(), nodeId_.c_str(),
         fresh ? "new" : "restored");
  }

  ~ProbeEffect() override {
    LogScope scope(logNode_);
    LOGI("destroyInstance %s", def_.label);
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    LogScope scope(logNode_);
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
    LOGI("getClipPreferences %s", def_.label);
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    LogScope scope(logNode_);  // log only: never set a value here (ofx/CLAUDE.md, recursion)
    LOGI("changedParam %s reason=%d", name.c_str(), int(args.reason));
  }

  void changedClip(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    LogScope scope(logNode_);
    LOGI("changedClip %s reason=%d", name.c_str(), int(args.reason));
  }

  void render(const OFX::RenderArguments& args) override {
    LogScope scope(logNode_);
    const int frame = int(args.time);
    double strength = 0;
    bool enable = false;
    int mode = 0, view = 0;
    fetchDoubleParam(kPStrength)->getValue(strength);
    fetchBooleanParam(kPEnable)->getValue(enable);
    fetchChoiceParam(kPMode)->getValue(mode);
    fetchChoiceParam(kPView)->getValue(view);

    std::unique_ptr<OFX::Image> out(dst_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(src_->fetchImage(args.time));
    if (!out) OFX::throwSuiteStatusException(kOfxStatFailed);

    std::string footer = std::string(def_.label) + " K=" + std::to_string(def_.constant) + " | f" +
                         std::to_string(frame) + " | Source " + describe(src_, src.get());
    std::string dTag = "-", fTag = "-";
    if (def_.masks) {
      std::unique_ptr<OFX::Image> d(detail_->isConnected() ? detail_->fetchImage(args.time) : nullptr);
      std::unique_ptr<OFX::Image> r(region_->isConnected() ? region_->fetchImage(args.time) : nullptr);
      footer += " | Detail " + describe(detail_, d.get()) + " | Focus " + describe(region_, r.get());
      if (d) dTag = std::to_string(d->getPixelComponentCount());
      if (r) fTag = std::to_string(r->getPixelComponentCount());
    }
    footer += " | id " + nodeId_ + " | S " + fmt2(strength) + " E" + std::to_string(enable) + " M" +
              std::to_string(mode) + " V" + std::to_string(view);
    if (def_.advanced) {
      footer += " | Adv";
      for (const char* p : kGammaDoubles) {
        double v = 0;
        fetchDoubleParam(p)->getValue(v);
        footer += " " + fmt2(v);
      }
      for (const char* p : kGammaBools) {
        bool v = false;
        fetchBooleanParam(p)->getValue(v);
        footer += " " + std::to_string(v);
      }
      for (const char* p : kGammaChoices) {
        int v = 0;
        fetchChoiceParam(p)->getValue(v);
        footer += " " + std::to_string(v);
      }
    }

    process(src.get(), out.get(), args.renderWindow, enable ? strength : 0.0, view);

    // Runtime UI text: only when it changes (each setValue comes back as changedParam).
    // <= 28 characters: Flame cuts text at about 30 (ofx/CLAUDE.md).
    const std::string plan = std::string(def_.letter) + " f" + std::to_string(frame) + " S" + fmt2(strength) + " D" +
                             dTag + " F" + fTag;
    std::string cur;
    plan_->getValue(cur);
    if (cur != plan) plan_->setValue(plan);
    const std::string label = "Seen at f" + std::to_string(frame);
    if (label != lastLabel_) {
      labelProbe_->setLabel(label);
      lastLabel_ = label;
    }
    if (footer != lastFooter_) {
      LOGI("footer: %s", footer.c_str());
      setPersistentMessage(OFX::Message::eMessageWarning, "", footer);  // no '%' in the text
      lastFooter_ = footer;
    }
    LOGI("render %s t=%g plan '%s'", def_.label, args.time, plan.c_str());
  }

 private:
  // "off", "no image" or "<n>ch x1,y1-x2,y2": what the host delivered, no pixel values.
  static std::string describe(OFX::Clip* c, OFX::Image* img) {
    if (!c->isConnected()) return "off";
    if (!img) return "no image";
    const OfxRectI b = img->getBounds();
    return std::to_string(img->getPixelComponentCount()) + "ch " + std::to_string(b.x1) + "," + std::to_string(b.y1) +
           "-" + std::to_string(b.x2) + "," + std::to_string(b.y2);
  }

  // View 0 Result (tinted picture), 1 Picture, 2 Matte (in RGB). Alpha = the effect's matte always.
  void process(OFX::Image* src, OFX::Image* out, const OfxRectI& win, double s, int view) const {
    if (out->getPixelDepth() != OFX::eBitDepthFloat || out->getPixelComponents() != OFX::ePixelComponentRGBA) return;
    const int sc = src && src->getPixelDepth() == OFX::eBitDepthFloat ? src->getPixelComponentCount() : 0;
    const OfxRectI sb = src ? src->getBounds() : OfxRectI{0, 0, 0, 0};
    const OfxRectI ob = out->getBounds();
    float gain[3];
    for (int c = 0; c < 3; ++c) gain[c] = float(1.0 - s + s * def_.tint[c]);
    for (int y = std::max(win.y1, ob.y1); y < std::min(win.y2, ob.y2); ++y) {
      for (int x = std::max(win.x1, ob.x1); x < std::min(win.x2, ob.x2); ++x) {
        float* q = static_cast<float*>(out->getPixelAddress(x, y));
        const bool in = sc >= 3 && x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
        const float* p = in ? static_cast<const float*>(src->getPixelAddress(x, y)) : nullptr;
        if (!p) {
          q[0] = q[1] = q[2] = q[3] = 0.f;
          continue;
        }
        const float a = sc == 4 ? p[3] : 1.f;
        const float m = def_.matte == 0 ? a : def_.matte == 1 ? 1.f - a : 0.5f * a;
        for (int c = 0; c < 3; ++c) q[c] = view == 1 ? p[c] : view == 2 ? m : p[c] * gain[c];
        q[3] = m;
      }
    }
  }

  const ProbeDef& def_;
  OFX::Clip *src_ = nullptr, *dst_ = nullptr, *detail_ = nullptr, *region_ = nullptr;
  OFX::StringParam* plan_ = nullptr;
  OFX::BooleanParam* labelProbe_ = nullptr;
  std::string nodeId_, logNode_, lastLabel_, lastFooter_;
};

template <class D>
D* labelled(D* p, const std::string& label) {
  p->setLabels(label, label, label);
  return p;
}

// One factory TYPE per effect (docs/edge_recipes.md 4.1).
template <class Tag>
class ProbeFactory : public OFX::PluginFactoryHelper<ProbeFactory<Tag>> {
 public:
  ProbeFactory() : OFX::PluginFactoryHelper<ProbeFactory<Tag>>(Tag::def().id, 0, 1) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logOnce();
    const ProbeDef& d = Tag::def();
    LogScope scope("erprobe");
    LOGI("==== Edge Recipes Probe %s (%s) describe %s id=%s uid=%s K=%d ====", ERP_VERSION, MR_GIT, d.label, d.id,
         this->getUID().c_str(), d.constant);
    desc.setLabels(d.label, d.label, d.label);
    desc.setPluginGrouping("Edge Recipes");
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
    const ProbeDef& d = Tag::def();
    LogScope scope("erprobe");
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setSupportsTiles(false);
    if (d.masks) {  // as Matte Merge's Detail and Focus Region (plugin_merge.cpp:619-638)
      for (const char* name : {kDetailClip, kRegionClip}) {
        OFX::ClipDescriptor* m = desc.defineClip(name);
        m->addSupportedComponent(OFX::ePixelComponentAlpha);
        m->addSupportedComponent(OFX::ePixelComponentRGBA);
        m->addSupportedComponent(OFX::ePixelComponentRGB);
        m->setIsMask(true);
        m->setOptional(true);
        m->setSupportsTiles(false);
        const char* text = name == kDetailClip ? "Detail" : "Focus Region";
        m->setLabels(text, text, text);
      }
    }
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    // Flame layout (ofx/CLAUDE.md "Flame host behaviour"): each group starts a column of <= 5 rows
    // (choice 2, others 1, a hidden param too), 6 columns per tab, a new tab only on overflow, named
    // after the next page.
    OFX::PageParamDescriptor* page = desc.definePageParam(d.label);
    auto group = [&](const char* name, const std::string& text, OFX::PageParamDescriptor* pg) {
      auto* g = labelled(desc.defineGroupParam(name), text);
      g->setOpen(true);
      pg->addChild(*g);
      return g;
    };
    auto number = [&](const char* name, const std::string& text, double def, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineDoubleParam(name), text);
      p->setRange(0, 1);
      p->setDisplayRange(0, 1);
      p->setDigits(2);
      p->setIncrement(0.01);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto boolean = [&](const char* name, const std::string& text, bool def, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineBooleanParam(name), text);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto choice = [&](const char* name, const std::string& text, int def, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineChoiceParam(name), text);
      for (const char* o : {"One", "Two", "Three"}) p->appendOption(o);
      p->setDefault(def);
      p->setParent(*g);
      return p;
    };
    auto labelString = [&](const char* name, const std::string& text, const std::string& value,
                           OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineStringParam(name), text);
      p->setStringType(OFX::eStringTypeLabel);
      p->setDefault(value);
      p->setIsPersistant(false);
      p->setEvaluateOnChange(false);
      p->setParent(*g);
      return p;
    };
    const std::string L = d.letter;

    // Column 1: Matte (4 rows).
    auto* gMatte = group("grpMatte", "Matte", page);
    auto* view = labelled(desc.defineChoiceParam(kPView), "View");
    for (const char* o : {"Result", "Picture", "Matte"}) view->appendOption(o);
    view->setDefault(0);
    view->setAnimates(false);
    view->setParent(*gMatte);
    labelString(kPVersion, "Version", std::string("ERP ") + ERP_VERSION + " " + L + " 0.1 " + MR_GIT, gMatte);
    number(kPStrength, std::string(d.label) + " Strength", d.strength, gMatte);  // label per descriptor

    // Column 2: the effect's own controls (Gamma: 5 rows, full).
    auto* gMain = group("grpMain", std::string(d.label) + " Main", page);
    boolean(kPEnable, L + " Enable Tint", true, gMain);
    choice(kPMode, L + " Mode", 1, gMain);
    OFX::GroupParamDescriptor* gMain2 = nullptr;
    if (d.advanced) {
      number("mainA", "G Main Value A", 0.25, gMain);
      boolean("mainB", "G Main Switch B", false, gMain);
      gMain2 = group("grpMain2", "Gamma Main (cont.)", page);  // column 3
      number("mainC", "G Main Value C", 0.75, gMain2);
      boolean("mainD", "G Main Switch D", true, gMain2);
    }

    // Plan: one label string rewritten per render, one fixed at describe, one boolean relabelled per render.
    auto* gPlan = group("grpPlan", "Plan", page);
    labelString(kPPlan, "Plan", L + " not rendered yet", gPlan);
    labelString(kPPlanStatic, "Described", std::string("described as ") + d.label, gPlan);
    auto* lp = boolean(kPLabelProbe, "Label set at describe", false, gPlan);
    lp->setEvaluateOnChange(false);

    if (d.advanced) {  // column 5: two choices (4 rows)
      auto* gColour = group("grpColour", "Model & Colour", page);
      choice("colourA", "G Colour Choice A", 0, gColour);
      choice("colourB", "G Colour Choice B", 2, gColour);
    }

    // Last tab-1 column: Support with the hidden node id (a hidden param takes a slot: inside a group).
    auto* gSupport = group("grpSupport", "Support", page);
    labelString("supportNote", "Probe", std::string("Edge Recipes probe ") + ERP_VERSION, gSupport);
    auto* nid = desc.defineStringParam(kPNodeId);
    nid->setDefault("");
    nid->setIsSecret(true);
    nid->setEvaluateOnChange(false);
    nid->setParent(*gSupport);

    if (d.advanced) {  // tab 1 has 6 columns now: these overflow to a tab named after this page
      OFX::PageParamDescriptor* adv = desc.definePageParam("Advanced");
      auto* gA = group("grpAdvA", "Advanced A", adv);
      number("advA", "G Adv Value A", 0.5, gA);
      boolean("advB", "G Adv Switch B", true, gA);
      choice("advC", "G Adv Choice C", 2, gA);
      auto* gB = group("grpAdvB", "Advanced B", adv);
      number("advD", "G Adv Value D", 0.1, gB);
      boolean("advE", "G Adv Switch E", false, gB);
    }
    LOGI("describeInContext %s ctx=%d: Source%s, Output RGBA; %s", d.label, int(ctx),
         d.masks ? " + Detail + FocusRegion (optional masks)" : " only",
         d.advanced ? "tab 1 = 6 columns + Advanced page (2 columns)" : "one page");
  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
    return new ProbeEffect(handle, Tag::def());
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static ProbeFactory<Alpha> alpha;
  static ProbeFactory<Beta> beta;
  static ProbeFactory<Gamma> gamma;
  ids.push_back(&alpha);
  ids.push_back(&beta);
  ids.push_back(&gamma);
}
}  // namespace Plugin
}  // namespace OFX
