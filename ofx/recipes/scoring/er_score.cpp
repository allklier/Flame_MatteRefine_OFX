// Edge Recipes WP8: end-to-end scores of the Hair Locks effect (docs/edge_recipes.md 6, 7) on synthetic
// sequences (er_scenes.h, plus the WP6 vote sequence of ofx/tools/synth), driven through a Flame-like host
// (er_host.*: ofx/merge/systest/ofx_host.* with a settable render scale). Classical path only: no model weights
// exist on the scoring machine, so Edge Model is OFF everywhere (the model candidate is NOT scored here).
//
//   er_score sweep  <EdgeRecipes.ofx> <out dir> [first last]   scenes [first, last) x settings -> scores.csv,
//                                                               measures.csv (appends; one row per scene x setting)
//   er_score report <out dir>                                  scores.csv + measures.csv -> summary.md
//   er_score timing <EdgeRecipes.ofx> <out dir>                per-step ms at 424x562, HD, UHD, 4096x2304
//   er_score tracerscenes <EdgeRecipes.ofx> <out dir>          WP3's scenes: tracer module vs the effect
//   er_score smoke  <EdgeRecipes.ofx> <out dir>                one tiny scene, three settings, the contract
//                                                               checks (ctest smoke; exit 1 on a violation)
// The scene list and the settings are fixed in this file (sceneList(), settingList()).
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "er_host.h"
#include "er_scenes.h"
#include "flow.h"
#include "merge_fusion.h"
#include "recipe_def.h"
#include "strand_tracer.h"
#include "synth.h"
#include "tracer_scenes.h"

namespace {

using namespace systest;
using namespace mr::recipes;
using er::ScoreSeq;
using er::SceneSpec;

std::string gHome, gXdg;

bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// ---- settings -----------------------------------------------------------------------------------------------
struct Setting {
  std::string name;
  bool rebuild = true;
  double minConf = 0.45, feather = 0.25, gain = 1.0, tmAmount = 1.0, snapAmount = 1.0;
  bool loose = true, vote = true, temporal = true, snap = true;
  std::string group;  // "default", "grid", "oat:<control>", "matrix"
};

std::vector<Setting> settingList() {
  std::vector<Setting> v;
  // ER_ONLY="default,snap_off": only these settings (a quick A/B; unset = all).
  const char* only = std::getenv("ER_ONLY");
  auto add = [&](Setting s, const std::string& name, const std::string& group) {
    if (only && ("," + std::string(only) + ",").find("," + name + ",") == std::string::npos) return;
    s.name = name;
    s.group = group;
    v.push_back(s);
  };
  const Setting d;  // the node's defaults with Rebuild Detail ON and Edge Model OFF
  add(d, "default", "default");
  Setting off = d;
  off.rebuild = false;
  add(off, "rebuild_off", "grid");
  // The calibration grid: Min Confidence x Region Feather (Rebuild on, the rest default).
  for (double mc : {0.25, 0.35, 0.45, 0.55, 0.7})
    for (double fe : {0.0, 0.5, 1.0, 2.0}) {
      Setting s = d;
      s.minConf = mc;
      s.feather = fe;
      char b[64];
      std::snprintf(b, sizeof b, "mc%.2f_fe%.1f", mc, fe);
      add(s, b, "grid");
    }
  for (double mc : {0.25, 0.35, 0.55, 0.7}) {  // the feather default 0.25 as well
    Setting s = d;
    s.minConf = mc;
    char b[64];
    std::snprintf(b, sizeof b, "mc%.2f_fe0.25", mc);
    add(s, b, "grid");
  }
  for (double g : {0.8, 1.5}) {
    Setting s = d;
    s.gain = g;
    add(s, g < 1 ? "gain0.8" : "gain1.5", "oat:gain");
  }
  {
    Setting s = d;
    s.loose = false;
    add(s, "loose_off", "oat:loose");
  }
  for (double a : {0.0, 0.5}) {
    Setting s = d;
    s.tmAmount = a;
    add(s, a == 0 ? "tm0.0" : "tm0.5", "oat:temporal");
  }
  for (double a : {0.0, 0.5}) {
    Setting s = d;
    s.snapAmount = a;
    add(s, a == 0 ? "snap0.0" : "snap0.5", "oat:snap");
  }
  {
    Setting s = d;
    s.snap = false;
    add(s, "snap_off", "oat:snap");
    s = d;
    s.rebuild = false;
    s.temporal = false;
    add(s, "snap_only", "matrix");  // Rebuild and Temporal off, Snap on: what Snap does to the base on its own
  }
  // The flicker matrix: vote on/off x Temporal on/off (default = on/on).
  Setting s = d;
  s.vote = false;
  add(s, "vote_off", "matrix");
  s = d;
  s.temporal = false;
  add(s, "tm_off", "matrix");
  s = d;
  s.vote = false;
  s.temporal = false;
  add(s, "vote_off_tm_off", "matrix");
  return v;
}

void apply(Instance& inst, const Setting& s) {
  inst.setBool("emEnable", false);
  inst.setBool("jrEnable", false);
  inst.setBool("rbEnable", s.rebuild);
  inst.setDouble("rbMinConf", s.minConf);
  inst.setDouble("rgFeather", s.feather);
  inst.setDouble("emGain", s.gain);
  inst.setBool("rbKeepLoose", s.loose);
  inst.setBool("rbVote", s.vote);
  inst.setBool("tmEnable", s.temporal);
  inst.setDouble("tmStrength", s.tmAmount);
  inst.setBool("rgSnap", s.snap);
  inst.setDouble("rgSnapStrength", s.snapAmount);
}

// ---- scenes -------------------------------------------------------------------------------------------------
struct SceneEntry {
  std::string set;  // "main", "distractor", "vote", "wp6"
  SceneSpec spec;
  bool wp6 = false;
};

std::vector<SceneEntry> sceneList() {
  std::vector<SceneEntry> v;
  const double widths[] = {1.5, 2.5, 4.0, 6.0}, margins[] = {0.0, 10.0, 20.0}, jit[] = {0.0, 0.75, 1.5};
  unsigned seed = 1;
  int idx = 0;
  // main: look x width x margin; distractor and jitter cycle through the scenes.
  for (int look = 0; look < er::kLookCount; ++look)
    for (double w : widths)
      for (double m : margins) {
        SceneEntry e;
        e.set = "main";
        e.spec.look = look;
        e.spec.width = w;
        e.spec.marginPx = m;
        e.spec.distractor = idx % er::kDistractorCount;
        e.spec.jitterPx = jit[(idx / er::kDistractorCount) % 3];
        e.spec.seed = seed++;
        e.spec.frames = 7;
        v.push_back(e);
        ++idx;
      }
  // distractor: every distractor on every look, width 3, drawn loosely by 10 px.
  for (int d = 1; d < er::kDistractorCount; ++d)
    for (int look = 0; look < er::kLookCount; ++look) {
      SceneEntry e;
      e.set = "distractor";
      e.spec.look = look;
      e.spec.width = 3.0;
      e.spec.marginPx = 10.0;
      e.spec.distractor = d;
      e.spec.jitterPx = 0.75;
      e.spec.seed = seed++;
      e.spec.frames = 7;
      v.push_back(e);
    }
  // vote: a dip on frame 3 and a spurious stroke on frame 6, every look, width 3, tight.
  for (int look = 0; look < er::kLookCount; ++look) {
    SceneEntry e;
    e.set = "vote";
    e.spec.look = look;
    e.spec.width = 3.0;
    e.spec.dipFrame = 3;
    e.spec.spuriousFrame = 6;
    e.spec.jitterPx = 0.75;
    e.spec.seed = seed++;
    v.push_back(e);
  }
  // detached (appended after wp6 so the earlier indexes stay): lock 0 starts 18 px outside the head (unanchored,
  // as the real frame's second shape, docs/strand_tracing.md section 5), so Min Confidence applies to it.
  const int dd[3] = {er::kNone, er::kMole, er::kTipBand};
  const double dw[3] = {2.5, 4.0, 3.0};
  std::vector<SceneEntry> det;
  for (int look = 0; look < er::kLookCount; ++look)
    for (int q = 0; q < 3; ++q) {
      SceneEntry e;
      e.set = "detached";
      e.spec.look = look;
      e.spec.width = dw[q];
      e.spec.marginPx = 10.0;
      e.spec.distractor = dd[q];
      e.spec.jitterPx = 0.75;
      e.spec.detachPx = 18.0;
      e.spec.frames = 7;
      e.spec.seed = 100 + unsigned(look * 3 + q);
      det.push_back(e);
    }
  // wp6: the vote system test's sequence (synth::makeLocksSequence, 16 frames, its injected flicker).
  SceneEntry e;
  e.set = "wp6";
  e.wp6 = true;
  e.spec.frames = 16;
  v.push_back(e);
  v.insert(v.end(), det.begin(), det.end());
  return v;
}

ScoreSeq buildScene(const SceneEntry& e) {
  if (!e.wp6) return er::makeScoreSeq(e.spec);
  synth::LockFlicker fl;
  fl.dipFrame = 4;
  fl.vanishFrame = 8;
  fl.spuriousFrame = 11;
  fl.flowFailFrame = 14;
  ScoreSeq s = er::fromLocksSequence(synth::makeLocksSequence(424, 562, 16, 1, fl));
  s.spec.frames = 16;
  return s;
}
std::string sceneName(const SceneEntry& e) { return e.wp6 ? std::string("wp6_locks_seq") : e.spec.name(); }

// ---- host feeds ---------------------------------------------------------------------------------------------
Feeds feedsOf(const ScoreSeq& s) {
  const ScoreSeq* p = &s;
  Feed src{[p](int f, int w, int h, std::vector<float>& o) {
             const float* c = p->rgb[size_t(f)].data();
             const float* m = p->base[size_t(f)].data();
             for (size_t i = 0; i < size_t(w) * h; ++i) {
               o[i * 4] = c[i * 3];
               o[i * 4 + 1] = c[i * 3 + 1];
               o[i * 4 + 2] = c[i * 3 + 2];
               o[i * 4 + 3] = m[i];
             }
           },
           4};
  Feed reg{[p](int f, int w, int h, std::vector<float>& o) {
             const float* r = p->region[size_t(f)].data();
             std::copy(r, r + size_t(w) * h, o.begin());
           },
           1};
  return Feeds{{kOfxImageEffectSimpleSourceClipName, src}, {"FocusRegion", reg}};
}

std::vector<float> alphaOf(const Rendered& r) {
  std::vector<float> a(size_t(r.w) * r.h);
  for (size_t i = 0; i < a.size(); ++i) a[i] = r.rgba[i * 4 + 3];
  return a;
}

// ---- scores -------------------------------------------------------------------------------------------------
struct Scores {
  double coverage = 0, iou = 0, sad = 0, falseIn = 0, distrMass = 0, distrHitRate = 0, crossCov = NAN;
  double spurMass = NAN, dipCov = NAN;
  double flicker = 0, flickerTruth = 0, covStd = 0, covEnds = 0, covMid = 0;
  double edgeErr = 0, edgeErrBase = 0;  // |out - true head| in the head-edge band outside the region
  long long outsideChanged = 0, outsideAddedTruth0 = 0, belowFull = 0, belowMask = 0;
  double outsideMaxAbs = 0;
  uint64_t hash = 1469598103934665603ull;
  double msPerFrame = 0;
};

double motionFlicker(const ScoreSeq& s, const std::vector<std::vector<float>>& a) {
  double sum = 0;
  long long cnt = 0;
  for (int t = 1; t < s.frames; ++t) {
    const synth::Affine back = synth::between(s.motion[size_t(t)], s.motion[size_t(t - 1)]);
    for (int y = 0; y < s.h; ++y)
      for (int x = 0; x < s.w; ++x) {
        const size_t i = size_t(y) * s.w + x;
        if (!(s.region[size_t(t)][i] > 0)) continue;
        double qx, qy;
        back.apply(x, y, &qx, &qy);
        if (qx < 0 || qy < 0 || qx > s.w - 1 || qy > s.h - 1) continue;
        sum += std::fabs(a[size_t(t)][i] - mr::ml::sampleField(a[size_t(t - 1)].data(), s.w, s.h, float(qx), float(qy)));
        ++cnt;
      }
  }
  return cnt ? sum / cnt : 0.0;
}

Scores scoreSeq(const ScoreSeq& s, const std::vector<std::vector<float>>& out, double featherPct) {
  Scores r;
  const size_t n = size_t(s.w) * s.h;
  double M = 0, covN = 0, sad = 0, fin = 0, dm = 0, crossM = 0, crossN = 0, edgeN = 0;
  long long inter = 0, uni = 0, edgeCnt = 0;
  int hits = 0;
  std::vector<double> perFrame;
  std::vector<std::vector<float>> target(size_t(s.frames));
  for (int t = 0; t < s.frames; ++t) {
    const std::vector<float>& o = out[size_t(t)];
    const std::vector<float>& b = s.base[size_t(t)];
    const std::vector<float>& m = s.region[size_t(t)];
    const std::vector<float>& c = s.cov[size_t(t)];
    std::vector<float> wt(n);
    mr::merge::prepareRegion(m.data(), s.w, s.h, 0.0, featherPct / 100.0 * s.h, wt.data());
    target[size_t(t)].resize(n);
    double Mt = 0, covt = 0, dmt = 0;
    for (size_t i = 0; i < n; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &o[i], 4);
      r.hash = (r.hash ^ bits) * 1099511628211ull;
      const double tg = b[i] + (1.0 - b[i]) * c[i];
      target[size_t(t)][i] = float(tg);
      const double miss = tg - b[i], add = double(o[i]) - b[i];
      if (o[i] < b[i]) {
        r.belowFull += wt[i] >= 1.0f;
        r.belowMask += m[i] > 0.0f;
      }
      if (m[i] > 0.0f) {
        Mt += miss;
        covt += std::max(0.0, std::min(add, miss));
        sad += std::fabs(o[i] - tg);
        if (miss < 0.02) fin += std::max(0.0, add);
        if (s.distract[size_t(t)][i]) dmt += std::max(0.0, add);
        const bool P = o[i] >= 0.5f && b[i] < 0.5f, T = tg >= 0.5 && b[i] < 0.5f;
        inter += P && T;
        uni += P || T;
        if (!s.crossCov[size_t(t)].empty()) {
          const double cm = (1.0 - b[i]) * s.crossCov[size_t(t)][i];
          crossM += cm;
          crossN += std::max(0.0, std::min(add, cm));
        }
      } else {
        if (!sameBits(o[i], b[i])) {
          ++r.outsideChanged;
          r.outsideMaxAbs = std::max(r.outsideMaxAbs, std::fabs(add));
          if (s.head[size_t(t)][i] + c[i] < 0.02 && add > 0) ++r.outsideAddedTruth0;
        }
        const float hd = s.head[size_t(t)][i];
        if ((hd > 0.02f && hd < 0.98f) || (b[i] > 0.02f && b[i] < 0.98f)) {
          edgeN += std::fabs(o[i] - hd);
          r.edgeErrBase += std::fabs(b[i] - hd);
          ++edgeCnt;
        }
      }
    }
    M += Mt;
    covN += covt;
    dm += dmt;
    hits += dmt > 1.0;
    perFrame.push_back(Mt > 0 ? covt / Mt : 0.0);
    if (t == s.spec.dipFrame) r.dipCov = Mt > 0 ? covt / Mt : 0.0;
    if (t == s.spec.spuriousFrame && !s.spurious[size_t(t)].empty()) {
      double sm = 0;
      for (size_t i = 0; i < n; ++i)
        if (s.spurious[size_t(t)][i]) sm += std::max(0.0, double(o[i]) - b[i]);
      r.spurMass = sm;
    }
  }
  r.coverage = M > 0 ? covN / M : 0;
  r.sad = M > 0 ? sad / M : 0;
  r.falseIn = M > 0 ? fin / M : 0;
  r.distrMass = M > 0 ? dm / M : 0;
  r.distrHitRate = double(hits) / s.frames;
  r.iou = uni ? double(inter) / uni : 0;
  if (crossM > 0) r.crossCov = crossN / crossM;
  r.edgeErr = edgeCnt ? edgeN / edgeCnt : 0;
  r.edgeErrBase = edgeCnt ? r.edgeErrBase / edgeCnt : 0;
  r.flicker = motionFlicker(s, out);
  r.flickerTruth = motionFlicker(s, target);
  double mean = 0, var = 0;
  for (double v : perFrame) mean += v;
  mean /= perFrame.size();
  for (double v : perFrame) var += (v - mean) * (v - mean);
  r.covStd = std::sqrt(var / perFrame.size());
  r.covEnds = 0.5 * (perFrame.front() + perFrame.back());
  double mid = 0;
  int nm = 0;
  for (size_t t = 2; t + 2 < perFrame.size(); ++t) mid += perFrame[t], ++nm;
  r.covMid = nm ? mid / nm : mean;
  return r;
}

// ---- Analyze measurements on the centre frame (what a WP10 Analyze could read cheaply) ---------------------
struct Measures {
  double regionPct = 0, softness = 0, lumaP1 = 0, lumaP99 = 0, lumaRange = 0, baseCov = 0;
  int pieces = 0, keptDefault = 0, anchored = 0;
  double confMedian = 0, confP25 = 0, confFracAbove45 = 0, widthMedian = 0, jitter = 0;
};

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[size_t(std::min<double>(v.size() - 1, std::max(0.0, q * (v.size() - 1) + 0.5)))];
}

Measures measure(const ScoreSeq& s) {
  Measures r;
  const int t = s.frames / 2;
  const size_t n = size_t(s.w) * s.h;
  const std::vector<float>& m = s.region[size_t(t)];
  const std::vector<float>& b = s.base[size_t(t)];
  std::vector<double> luma;
  long long area = 0, soft = 0;
  double bc = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!(m[i] > 0)) continue;
    ++area;
    soft += m[i] < 1.0f;
    bc += b[i];
    const float* c = &s.rgb[size_t(t)][i * 3];
    luma.push_back(std::pow(std::max(0.0, (double(c[0]) + c[1] + c[2]) / 3.0), 1.0 / 2.4));
  }
  r.regionPct = 100.0 * area / n;
  r.softness = area ? double(soft) / area : 0;
  r.baseCov = area ? bc / area : 0;
  r.lumaP1 = pct(luma, 0.01);
  r.lumaP99 = pct(luma, 0.99);
  r.lumaRange = r.lumaP99 - r.lumaP1;
  // The tracer with its keep bars down: every candidate piece and its confidence.
  std::vector<float> wt(n);
  mr::merge::prepareRegion(m.data(), s.w, s.h, 0.0, 0.25 / 100.0 * s.h, wt.data());
  mr::tracer::TracerInput in;
  in.rgb = s.rgb[size_t(t)].data();
  in.base = b.data();
  in.region = wt.data();
  in.width = s.w;
  in.height = s.h;
  mr::tracer::TracerParams lo;
  lo.keepConf = 0.0;
  lo.minConfUnanchored = 0.0;
  const mr::tracer::TracerResult all = mr::tracer::traceStrands(in, lo);
  std::vector<double> conf, width;
  for (const auto& p : all.pieces) {
    conf.push_back(p.confidence);
    width.push_back(p.width);
    r.anchored += p.anchored;
  }
  r.pieces = int(all.pieces.size());
  r.confMedian = pct(conf, 0.5);
  r.confP25 = pct(conf, 0.25);
  int above = 0;
  for (double c : conf) above += c >= 0.45;
  r.confFracAbove45 = conf.empty() ? 0 : double(above) / conf.size();
  r.widthMedian = pct(width, 0.5);
  r.keptDefault = int(mr::tracer::traceStrands(in, mr::tracer::TracerParams()).pieces.size());
  // Jitter of the base matte: |base_t - mean(base_t-1, base_t+1)| on the edge band (motion cancels to first order).
  if (t >= 1 && t + 1 < s.frames) {
    const std::vector<float>&p0 = s.base[size_t(t - 1)], &p1 = s.base[size_t(t + 1)];
    double sum = 0;
    long long cnt = 0;
    for (size_t i = 0; i < n; ++i) {
      const float mm = 0.5f * (p0[i] + p1[i]);
      if ((b[i] > 0.02f && b[i] < 0.98f) || (mm > 0.02f && mm < 0.98f)) sum += std::fabs(b[i] - mm), ++cnt;
    }
    r.jitter = cnt ? sum / cnt : 0;
  }
  return r;
}

// ---- running ------------------------------------------------------------------------------------------------
Scores runSetting(Instance& inst, const ScoreSeq& s, const Setting& st) {
  apply(inst, st);
  std::vector<std::vector<float>> out(size_t(s.frames));
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < s.frames; ++t) out[size_t(t)] = alphaOf(inst.render(t));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Scores r = scoreSeq(s, out, st.feather);
  r.msPerFrame = ms / s.frames;
  return r;
}

void setupEnv(const std::string& home) {
  signal(SIGPIPE, SIG_IGN);
  gHome = home;
  if (std::system(("mkdir -p '" + gHome + "'").c_str()) != 0) std::exit(2);
  char xdg[] = "/tmp/er-score-XXXXXX";  // a short folder for a runner socket (none is started: Edge Model off)
  if (!mkdtemp(xdg)) std::exit(2);
  gXdg = xdg;
  setenv("HOME", gHome.c_str(), 1);
  setenv("XDG_RUNTIME_DIR", xdg, 1);
}
void cleanupEnv() {
  if (std::system(("rm -rf '" + gXdg + "'").c_str()) != 0) std::fprintf(stderr, "could not remove %s\n", gXdg.c_str());
}

const char* kScoreHeader =
    "set,scene,look,width,margin,distractor,jitter,dip,spur,frames,scale,setting,group,rebuild,minConf,feather,gain,"
    "loose,vote,temporal,tmAmount,snapAmount,coverage,iou,sad,falseIn,distrMass,distrHitRate,crossCov,spurMass,dipCov,"
    "flicker,flickerTruth,covStd,covEnds,covMid,edgeErr,edgeErrBase,outsideChanged,outsideAddedTruth0,outsideMaxAbs,"
    "belowFull,belowMask,hash,msPerFrame\n";
const char* kMeasureHeader =
    "set,scene,look,width,margin,distractor,jitter,regionPct,softness,lumaP1,lumaP99,lumaRange,baseCov,pieces,"
    "keptDefault,anchored,confMedian,confP25,confFracAbove45,widthMedian,jitterMeasured\n";

std::string sceneCols(const SceneEntry& e, const ScoreSeq& s) {
  char b[400];
  std::snprintf(b, sizeof b, "%s,%s,%s,%.1f,%.0f,%s,%.2f,%d,%d,%d", e.set.c_str(), sceneName(e).c_str(),
                e.wp6 ? "dark_skin" : er::lookName(e.spec.look), e.wp6 ? 3.0 : e.spec.width, e.spec.marginPx,
                e.wp6 ? "none" : er::distractorName(e.spec.distractor), e.spec.jitterPx, s.spec.dipFrame >= 0,
                s.spec.spuriousFrame >= 0, s.frames);
  return b;
}
std::string scoreRow(const std::string& sc, double scale, const Setting& st, const Scores& r) {
  char b[1200];
  std::snprintf(b, sizeof b,
                "%s,%.2f,%s,%s,%d,%.2f,%.2f,%.2f,%d,%d,%d,%.2f,%.2f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.5f,%.5f,%.5f,"
                "%.6f,%.6f,%.5f,%.5f,%.5f,%.6f,%.6f,%lld,%lld,%.5f,%lld,%lld,%016llx,%.1f\n",
                sc.c_str(), scale, st.name.c_str(), st.group.c_str(), st.rebuild, st.minConf, st.feather, st.gain,
                st.loose, st.vote, st.temporal, st.tmAmount, st.snapAmount, r.coverage, r.iou, r.sad, r.falseIn,
                r.distrMass, r.distrHitRate, r.crossCov, r.spurMass, r.dipCov, r.flicker, r.flickerTruth, r.covStd,
                r.covEnds, r.covMid, r.edgeErr, r.edgeErrBase, r.outsideChanged, r.outsideAddedTruth0, r.outsideMaxAbs,
                r.belowFull, r.belowMask, (unsigned long long)r.hash, r.msPerFrame);
  return b;
}

void appendFile(const std::string& path, const std::string& header, const std::string& row) {
  struct stat st;
  const bool fresh = stat(path.c_str(), &st) != 0;
  std::ofstream f(path, std::ios::app);
  if (fresh) f << header;
  f << row;
}

// Contract violations of one row (every frame): never below the base where the region weight is 1; with Temporal
// off, nothing outside the region changes and nothing inside goes below the base.
std::string violations(const Setting& st, const Scores& r) {
  std::string v;
  if (r.belowFull) v += " below-base(weight 1)";
  if (!st.temporal && r.outsideChanged) v += " changed-outside";
  if (!st.temporal && r.belowMask) v += " below-base(region)";
  return v;
}

int sweep(const std::string& bundle, const std::string& out, int first, int last) {
  setupEnv(out + "/home");
  Host host(bundle);
  if (!host.loaded() || host.describe() != kOfxStatOK) {
    std::fprintf(stderr, "cannot load %s\n", bundle.c_str());
    return 2;
  }
  const std::vector<SceneEntry> scenes = sceneList();
  const std::vector<Setting> settings = settingList();
  last = std::min(last, int(scenes.size()));
  int bad = 0;
  for (int si = first; si < last; ++si) {
    const SceneEntry& e = scenes[size_t(si)];
    const ScoreSeq s = buildScene(e);
    const std::string sc = sceneCols(e, s);
    {
      const Measures m = measure(s);
      char b[600];
      std::snprintf(b, sizeof b, "%s,%s,%s,%.1f,%.0f,%s,%.2f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.5f\n",
                    e.set.c_str(), sceneName(e).c_str(), e.wp6 ? "dark_skin" : er::lookName(e.spec.look),
                    e.wp6 ? 3.0 : e.spec.width, e.spec.marginPx, e.wp6 ? "none" : er::distractorName(e.spec.distractor),
                    e.spec.jitterPx, m.regionPct, m.softness, m.lumaP1, m.lumaP99, m.lumaRange, m.baseCov, m.pieces,
                    m.keptDefault, m.anchored, m.confMedian, m.confP25, m.confFracAbove45, m.widthMedian, m.jitter);
      appendFile(out + "/measures.csv", kMeasureHeader, b);
    }
    auto inst = host.createInstance(s.w, s.h, s.frames, feedsOf(s));
    inst->setInt(kPPicColour, kColourBypass);  // the scenes are linear
    gRenderScale = 1.0;
    std::string rows;
    for (const Setting& st : settings) {
      const Scores r = runSetting(*inst, s, st);
      rows += scoreRow(sc, 1.0, st, r);
      const std::string v = violations(st, r);
      if (!v.empty()) std::printf("VIOLATION %s %s:%s\n", sceneName(e).c_str(), st.name.c_str(), v.c_str()), ++bad;
    }
    // Proxy: the same scene at half size, render scale 0.5, default setting and vote/Temporal off.
    if (!e.wp6) {
      const ScoreSeq h = er::downsample2(s);
      auto pinst = host.createInstance(h.w, h.h, h.frames, feedsOf(h));
      pinst->setInt(kPPicColour, kColourBypass);
      gRenderScale = 0.5;
      for (const Setting& st : settings)
        if (st.name == "default" || st.name == "vote_off_tm_off") {
          const Scores r = runSetting(*pinst, h, st);
          rows += scoreRow(sc, 0.5, st, r);
          const std::string v = violations(st, r);
          if (!v.empty()) std::printf("VIOLATION %s proxy %s:%s\n", sceneName(e).c_str(), st.name.c_str(), v.c_str()), ++bad;
        }
      gRenderScale = 1.0;
    }
    appendFile(out + "/scores.csv", kScoreHeader, rows);
    std::printf("scene %d/%zu %s done\n", si + 1, scenes.size(), sceneName(e).c_str());
    std::fflush(stdout);
  }
  for (const auto& err : host.errors) std::printf("HOST ERROR: %s\n", err.c_str()), ++bad;
  cleanupEnv();
  return bad ? 1 : 0;
}

// ---- timing -------------------------------------------------------------------------------------------------
int timing(const std::string& bundle, const std::string& out) {
  setupEnv(out + "/timing_home");
  setenv("MR_LOG_LEVEL", "debug", 1);
  Host host(bundle);
  if (!host.loaded() || host.describe() != kOfxStatOK) return 2;
  std::ofstream md(out + "/timing.md");
  char cpu[256] = "unknown";
  {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
      if (line.rfind("model name", 0) == 0) {
        std::snprintf(cpu, sizeof cpu, "%s", line.substr(line.find(':') + 2).c_str());
        break;
      }
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  sched_getaffinity(0, sizeof set, &set);
  md << "# Hair Locks timing (er_score timing)\n\nCPU: " << cpu << "; cores allowed to this process: " << CPU_COUNT(&set)
     << " (affinity); std::thread::hardware_concurrency() = " << std::thread::hardware_concurrency()
     << " (the cores' worker pools size by this, so their threads time-share the allowed cores).\n\n"
     << "Every classical step on (Rebuild Detail + Temporal Vote, Snap, Temporal +-1), Edge Model and Jump Repair off; "
        "5-frame clip (a glove distractor, base jitter 0.75 px), centre frame rendered 3 times; wall ms through the "
        "host (input copy + render + output), then the plugin's per-step ms (DEBUG line of the last render).\n\n"
     << "| size | render 1 | render 2 | render 3 |\n|---|---|---|---|\n";
  struct Sz {
    int w, h;
  };
  std::vector<std::string> lines;
  for (Sz z : {Sz{424, 562}, Sz{1920, 1080}, Sz{3840, 2160}, Sz{4096, 2304}}) {
    SceneSpec sp;
    sp.w = z.w;
    sp.h = z.h;
    sp.frames = 5;
    sp.distractor = er::kGlove;
    sp.jitterPx = 0.75;
    ScoreSeq s = er::makeScoreSeq(sp);
    s.cov.clear();
    s.head.clear();
    auto inst = host.createInstance(s.w, s.h, s.frames, feedsOf(s));
    inst->setInt(kPPicColour, kColourBypass);
    Setting st;
    apply(*inst, st);
    md << "| " << z.w << "x" << z.h;
    for (int i = 0; i < 3; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      inst->render(2);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      char b[64];
      std::snprintf(b, sizeof b, " | %.0f ms", ms);
      md << b;
    }
    md << " |\n";
    md.flush();
    std::printf("timing %dx%d done\n", z.w, z.h);
    std::fflush(stdout);
  }
  md << "\nPer-step ms (plugin DEBUG lines, last render of each size):\n\n```\n";
  std::ifstream log(gHome + "/.cache/matte-refine/logs/matte-refine.log");
  std::string line;
  std::map<std::string, std::string> lastBySize;
  std::vector<std::string> order;
  while (std::getline(log, line)) {
    const size_t a = line.find("render t=");
    if (a == std::string::npos || line.find("hairLocks rev") == std::string::npos) continue;
    const size_t sz = line.find(' ', a + 9);
    const std::string key = line.substr(sz + 1, line.find(' ', sz + 1) - sz - 1);
    if (!lastBySize.count(key)) order.push_back(key);
    const size_t e = line.find(" | inputs");
    lastBySize[key] = line.substr(a, e == std::string::npos ? std::string::npos : e - a);
  }
  for (const std::string& k : order) md << lastBySize[k] << "\n";
  md << "```\n";
  {  // the vote line (neighbours traced, flows)
    std::ifstream lg(gHome + "/.cache/matte-refine/logs/matte-refine.log");
    std::vector<std::string> votes;
    while (std::getline(lg, line))
      if (line.find(" vote: neighbours traced") != std::string::npos) votes.push_back(line.substr(line.find("render t=")));
    md << "\nVote lines (one per render):\n\n```\n";
    for (const std::string& v : votes) md << v << "\n";
    md << "```\n";
  }
  cleanupEnv();
  return 0;
}

// ---- smoke --------------------------------------------------------------------------------------------------
int smoke(const std::string& bundle, const std::string& out) {
  setupEnv(out + "/smoke_home");
  Host host(bundle);
  if (!host.loaded() || host.describe() != kOfxStatOK) {
    std::printf("FAIL load %s\n", bundle.c_str());
    return 1;
  }
  SceneSpec sp;
  sp.w = 212;
  sp.h = 281;
  sp.frames = 5;
  sp.distractor = er::kGlasses;
  sp.jitterPx = 0.75;
  sp.spuriousFrame = 2;
  const ScoreSeq s = er::makeScoreSeq(sp);
  auto inst = host.createInstance(s.w, s.h, s.frames, feedsOf(s));
  inst->setInt(kPPicColour, kColourBypass);
  int bad = 0;
  std::map<std::string, Scores> got;
  for (const Setting& st : settingList()) {
    if (st.name != "default" && st.name != "rebuild_off" && st.name != "vote_off_tm_off") continue;
    const Scores r = runSetting(*inst, s, st);
    got[st.name] = r;
    std::printf("smoke %s: coverage %.3f IoU %.3f SAD %.3f falseIn %.4f distr %.4f flicker %.4f outside %lld below %lld/%lld\n",
                st.name.c_str(), r.coverage, r.iou, r.sad, r.falseIn, r.distrMass, r.flicker, r.outsideChanged,
                r.belowFull, r.belowMask);
    const std::string v = violations(st, r);
    if (!v.empty()) std::printf("FAIL %s:%s\n", st.name.c_str(), v.c_str()), ++bad;
  }
  // Rebuild, Snap and Temporal off: the whole matte bit for bit, every frame (Temporal alone works in the region's
  // feather ring by design, so Rebuild off with Temporal on does change pixels there).
  {
    Setting st;
    st.name = "rebuild_off_tm_off";
    st.rebuild = false;
    st.temporal = false;
    st.snap = false;  // Snap alone changes the base edge in the region (a finding, setting "snap_only")
    apply(*inst, st);
    long long diff = 0;
    for (int t = 0; t < s.frames; ++t) {
      const std::vector<float> a = alphaOf(inst->render(t));
      for (size_t i = 0; i < a.size(); ++i) diff += !sameBits(a[i], s.base[size_t(t)][i]);
    }
    std::printf("smoke rebuild_off_tm_off: %lld px differ from the base\n", diff);
    if (diff) std::printf("FAIL pass-through not bit-identical\n"), ++bad;
  }
  // The tracer finds the strands on this easy scene (a loose floor: the smoke is not a score).
  if (got["vote_off_tm_off"].coverage < 0.3) std::printf("FAIL coverage too low\n"), ++bad;
  for (const auto& err : host.errors) std::printf("FAIL host: %s\n", err.c_str()), ++bad;
  cleanupEnv();
  std::printf("er_score smoke: %s\n", bad ? "FAILED" : "OK");
  return bad ? 1 : 0;
}

// ---- the tracer's own scenes: module vs effect ----------------------------------------------------------------
// WP3's 12 scenes x 6 seeds (docs/strand_tracing.md section 4), each a one-frame clip: the tracer module run
// directly (as tracer_scenes::evaluate, frameHeight 1080) and the effect (Rebuild only: vote, Snap, Temporal off;
// then + Snap). Metrics: tracer_scenes::metrics on the alpha the effect ADDED (rec needs the pieces: module only).
int tracerScenes(const std::string& bundle, const std::string& out) {
  namespace ts = mr::tracer::scenes;
  setupEnv(out + "/tracer_home");
  Host host(bundle);
  if (!host.loaded() || host.describe() != kOfxStatOK) return 2;
  std::ofstream md(out + "/tracer_scenes.md");
  md << "# The tracer's scenes: module vs effect (er_score tracerscenes)\n\n12 scenes x 6 seeds, 112 x 112, one "
        "frame. module = traceStrands directly (frameHeight 1080, as test_strand_tracer); effect = Hair Locks through "
        "the host, Rebuild Detail only (vote, Snap, Temporal off; the region feathered 0.25 %, min piece scaled by the "
        "112-line height), metrics on the alpha it ADDED; effect+snap = the same with Snap 1. rec needs the pieces "
        "(module only).\n\n| scene | module rec | module false | effect false | effect+snap false | module distr | "
        "effect distr | effect+snap distr | module sad | effect sad | effect+snap sad |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
  std::vector<std::vector<ts::Metrics>> all(3);
  for (const std::string& name : ts::sceneNames()) {
    std::vector<std::vector<ts::Metrics>> acc(3);
    for (int k = 0; k < 6; ++k) {
      const ts::Scene S = ts::makeScene(name, k);
      mr::tracer::TracerInput in;
      in.rgb = S.rgb.data();
      in.base = S.alpha.data();
      in.region = S.region.data();
      in.width = S.w;
      in.height = S.h;
      in.frameHeight = 1080;
      const mr::tracer::TracerResult r = mr::tracer::traceStrands(in);
      acc[0].push_back(ts::metrics(S, r.alpha, r.pieces));
      const ts::Scene* sp = &S;
      Feeds f{{kOfxImageEffectSimpleSourceClipName,
               Feed{[sp](int, int w, int h, std::vector<float>& o) {
                      for (size_t i = 0; i < size_t(w) * h; ++i) {
                        for (int q = 0; q < 3; ++q) o[i * 4 + q] = sp->rgb[i * 3 + q];
                        o[i * 4 + 3] = sp->alpha[i];
                      }
                    },
                    4}},
              {"FocusRegion", Feed{[sp](int, int w, int h, std::vector<float>& o) {
                                     std::copy(sp->region.begin(), sp->region.begin() + long(w) * h, o.begin());
                                   },
                                   1}}};
      auto inst = host.createInstance(S.w, S.h, 1, f);
      inst->setInt(kPPicColour, kColourBypass);
      Setting st;
      st.vote = false;
      st.temporal = false;
      st.snap = false;
      for (int v = 1; v <= 2; ++v) {
        st.snap = v == 2;
        apply(*inst, st);
        const std::vector<float> a = alphaOf(inst->render(0));
        std::vector<float> add(a.size());
        for (size_t i = 0; i < a.size(); ++i) add[i] = std::max(0.0f, a[i] - S.alpha[i]);
        acc[size_t(v)].push_back(ts::metrics(S, add, {}));
      }
    }
    ts::Metrics m[3];
    for (int v = 0; v < 3; ++v) m[v] = ts::aggregate(acc[size_t(v)]), all[size_t(v)].push_back(m[v]);
    char b[400];
    std::snprintf(b, sizeof b, "| %s | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f | %.2f |\n",
                  name.c_str(), m[0].rec, m[0].fals, m[1].fals, m[2].fals, m[0].distr, m[1].distr, m[2].distr, m[0].sad,
                  m[1].sad, m[2].sad);
    md << b;
  }
  ts::Metrics m[3];
  for (int v = 0; v < 3; ++v) m[v] = ts::aggregate(all[size_t(v)]);
  char b[400];
  std::snprintf(b, sizeof b, "| **mean** | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f | %.2f |\n", m[0].rec,
                m[0].fals, m[1].fals, m[2].fals, m[0].distr, m[1].distr, m[2].distr, m[0].sad, m[1].sad, m[2].sad);
  md << b << "\nReference (docs/strand_tracing.md section 4, Python prototype, all rules): rec 0.97, false 0.01, distr "
             "0.01, sad 0.30.\n";
  std::printf("%s", b);
  cleanupEnv();
  return 0;
}

}  // namespace

int reportMain(const std::string& out);  // er_report.cpp

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "sweep" && argc >= 4) {
    const int first = argc > 4 ? std::atoi(argv[4]) : 0, last = argc > 5 ? std::atoi(argv[5]) : 1 << 30;
    return sweep(argv[2], argv[3], first, last);
  }
  if (mode == "report" && argc == 3) return reportMain(argv[2]);
  if (mode == "timing" && argc == 4) return timing(argv[2], argv[3]);
  if (mode == "tracerscenes" && argc == 4) return tracerScenes(argv[2], argv[3]);
  if (mode == "smoke" && argc == 4) return smoke(argv[2], argv[3]);
  if (mode == "list") {
    const auto sc = sceneList();
    for (size_t i = 0; i < sc.size(); ++i) std::printf("%zu %s %s\n", i, sc[i].set.c_str(), sceneName(sc[i]).c_str());
    std::printf("%zu settings\n", settingList().size());
    return 0;
  }
  std::fprintf(stderr,
               "usage: er_score sweep <EdgeRecipes.ofx> <out> [first last] | report <out> | timing <EdgeRecipes.ofx> "
               "<out> | smoke <EdgeRecipes.ofx> <out> | tracerscenes <EdgeRecipes.ofx> <out> | list\n");
  return 2;
}
