// hl_probe: runs Hair Locks' cores (the plugin's wiring, plugin_recipes.cpp render) on raw frames from disk and
// writes every stage's plane, so a real Flame case can be reproduced outside Flame (agent hl96, 2026-11-13).
//
//   hl_probe <data dir> <out dir> <t> [key=value ...]
// data dir: info.txt ("w h"), t<N>.<colour>.rgb (w*h*3 float32, top-down), t<N>.matte (w*h), region.f32 (w*h).
// Keys: colour=auto|bypass (which .rgb file; auto = the picture as Flame delivers it under Picture Colour Auto),
//   model=0|1 (Small / Base), edge=0|1, rebuild=0|1, vote=0|1, snap=0|1, temporal=0|1, snapAmt, tmAmt, minConf,
//   feather, grow (the controls' units), first, last (the clip range, default 0..t),
//   runner=<path> (record: the real runner; model outputs saved under cache=<dir>), cache=<dir> (replay when no
//   runner), home=<dir> (HOME for the runner: its config), socket=<path>.
// Outputs (float32 w*h): final, regionBase, regionWeight, cand_model, cand_model_conf, cand_tracer,
//   cand_tracer_conf; text: report.txt (steps), pieces.txt (tracer pieces / joins of the centre frame).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hair_locks_sources.h"
#include "hair_locks_ui.h"
#include "ops/morphology.h"
#include "render_chain.h"
#include "runner_client.h"
#include "runner_crops.h"

using namespace mr;
using namespace mr::recipes;

namespace {

std::vector<float> readF(const std::string& p, size_t n) {
  std::vector<float> v(n);
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + p);
  const size_t got = std::fread(v.data(), sizeof(float), n, f);
  std::fclose(f);
  if (got != n) throw std::runtime_error("short read " + p);
  return v;
}
bool exists(const std::string& p) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}
void writeF(const std::string& p, const std::vector<float>& v) {
  FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + p);
  std::fwrite(v.data(), sizeof(float), v.size(), f);
  std::fclose(f);
}

uint64_t fnv(const void* d, size_t n, uint64_t h = 1469598103934665603ull) {
  const unsigned char* p = static_cast<const unsigned char*>(d);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

// Records the real runner's outputs (keyed by the input's hash) or replays them.
class CachingRunner : public recipe::CropRunner {
 public:
  CachingRunner(recipe::CropRunner* real, std::string dir) : real_(real), dir_(std::move(dir)) {}
  void run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut, int x,
           int y) override {
    uint64_t k = fnv(modelId.data(), modelId.size());
    k = fnv(in, sizeof(float) * size_t(cIn) * h * w, k);
    char name[64];
    std::snprintf(name, sizeof name, "/%016llx_%dx%d.f32", (unsigned long long)k, w, h);
    const std::string p = dir_ + name;
    const size_t n = size_t(cOut) * h * w;
    if (exists(p)) {
      const std::vector<float> v = readF(p, n);
      std::copy(v.begin(), v.end(), out);
      ++hits;
      return;
    }
    if (!real_) throw std::runtime_error("model cache miss " + p + " and no runner");
    real_->run(modelId, in, cIn, h, w, out, cOut, x, y);
    writeF(p, std::vector<float>(out, out + n));
    ++misses;
  }
  int hits = 0, misses = 0;

 private:
  recipe::CropRunner* real_;
  std::string dir_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: hl_probe <data dir> <out dir> <t> [key=value ...]\n");
    return 2;
  }
  const std::string D = argv[1], O = argv[2];
  const int t = std::atoi(argv[3]);
  std::map<std::string, std::string> kv;
  for (int i = 4; i < argc; ++i) {
    const char* e = std::strchr(argv[i], '=');
    if (e) kv[std::string(argv[i], size_t(e - argv[i]))] = e + 1;
  }
  auto num = [&](const char* k, double d) { return kv.count(k) ? std::atof(kv[k].c_str()) : d; };
  auto str = [&](const char* k, const std::string& d) { return kv.count(k) ? kv[k] : d; };
  int w = 0, h = 0;
  {
    std::ifstream f(D + "/info.txt");
    f >> w >> h;
  }
  const size_t n = size_t(w) * h;
  const std::string colour = str("colour", "auto");
  auto rgbOf = [&](int tt) { return readF(D + "/t" + std::to_string(tt) + "." + colour + ".rgb", 3 * n); };
  auto matteOf = [&](int tt) { return readF(D + "/t" + std::to_string(tt) + ".matte", n); };
  const std::string regionPath = exists(D + "/t" + std::to_string(t) + ".region") ? D + "/t" + std::to_string(t) + ".region" : D + "/region.f32";
  std::vector<float> rgb = rgbOf(t), matte = matteOf(t), region = readF(regionPath, n);

  HairLocksUi ui;  // the node's defaults (recipe_def.cpp), with the user's Flame settings
  ui.edgeModel = num("edge", 1) != 0;
  ui.model = int(num("model", 1));
  ui.rebuild = num("rebuild", 1) != 0;
  ui.vote = num("vote", 1) != 0;
  ui.snap = num("snap", 1) != 0;
  ui.temporal = num("temporal", 1) != 0;
  ui.snapAmount = num("snapAmt", 1.0);
  ui.temporalAmount = num("tmAmt", 1.0);
  ui.rebuildMinConf = num("minConf", 0.45);
  ui.featherPct = num("feather", 0.25);
  ui.growPct = num("grow", 0.0);

  // As plugin_recipes.cpp render (lines 400-415).
  ModelSource model;
  TracerSource tracer;
  DetailSource detail;
  WeightedTemporal temporal;
  VoteStep vote{&tracer};
  HairLocks hl{{&model, &tracer, &detail}, &temporal, &vote};
  Workspace ws;
  Workers wk;
  HairLocksSettings s = hairLocksSettings(ui, h, 1.0);
  // Experiment knobs (defaults = the code's own defaults).
  if (kv.count("confFull")) s.fusion.confFull = float(num("confFull", 0));
  if (kv.count("addedMin")) s.snapAddedMin = float(num("addedMin", 1e-3));
  if (kv.count("raiseOnly")) s.snapRaiseOnly = num("raiseOnly", 0) != 0;
  if (kv.count("islandTouch")) s.snapIslandTouch = float(num("islandTouch", 0));
  model.p = modelParams(ui);
  model.region = s.region;
  tracer.p = tracerParams(ui);
  if (kv.count("chain")) tracer.p.chainLength = num("chain", 0) != 0;
  if (kv.count("cone")) tracer.p.coneJoin = num("cone", 0) != 0;
  if (kv.count("soft")) tracer.p.softProfile = num("soft", 0) != 0;
  if (kv.count("sigma")) tracer.p.edgeSigma1080 = num("sigma", 1.0);
  if (kv.count("taper")) tracer.p.taperW = num("taper", 1.5);
  tracer.region = s.region;
  tracer.keepLoose = ui.keepLoose;
  tracer.linearise = colour == "auto";
  tracer.forVote = s.strandVote;
  tracer.vp = voteParams(tracer.p);
  vote.temporalFlows = s.temporal ? 2 * s.tm.frames : 0;
  if (kv.count("bnear")) tracer.vp.blendNear = num("bnear", 0);
  if (kv.count("bfar")) tracer.vp.blendFar = num("bfar", 0);
  if (kv.count("flo")) tracer.vp.fillLo = num("flo", 0.5);
  if (kv.count("fhi")) tracer.vp.fillHi = num("fhi", 0.8);
  if (kv.count("plo")) tracer.vp.presenceLo = num("plo", 0);
  if (kv.count("phi")) tracer.vp.presenceHi = num("phi", 0);
  if (kv.count("moved")) vote.movedRegions = num("moved", 0) != 0;

  std::unique_ptr<ml::RunnerClient> client;
  std::unique_ptr<recipe::RunnerClientCrops> crops;
  const std::string runnerPath = str("runner", "");
  if (!runnerPath.empty()) {
    ml::RunnerClientOptions ro;
    ro.runnerPath = runnerPath;
    ro.buildId = ER_RUNNER_BUILD_ID;
    ro.socketPath = str("socket", "");
    ro.name = "hl_probe";
    ro.tileTimeoutS = 120;
    ro.startTimeoutS = 60;
    if (kv.count("home")) ro.env.push_back("HOME=" + kv["home"]);
    client.reset(new ml::RunnerClient(ro));
    crops.reset(new recipe::RunnerClientCrops(*client));
  }
  CachingRunner cache(crops.get(), str("cache", O + "/cache"));
  model.runner = [&]() { return static_cast<recipe::CropRunner*>(&cache); };

  Recipe recipe = hl.recipe(s);
  const StepNeeds needs = unionNeeds(recipe);
  ml::Neighbours nb;
  ml::FrameRange range;
  range.first = num("first", 0);
  range.last = num("last", t);
  range.valid = true;
  const int tK = std::max(s.temporal ? s.tm.frames : 0, s.strandVote ? vote.K : 0);
  if (needs.neighbourReach > 0)
    ml::collectNeighbours(double(t), needs.neighbourReach, range, w, h,
                          [&](double time, ml::FetchedFrame* f) {
                            const int tt = int(time);
                            f->matteOnly = std::abs(tt - t) > tK;
                            if (!exists(D + "/t" + std::to_string(tt) + ".matte")) return;
                            f->image = true;
                            f->w = w;
                            f->h = h;
                            f->components = 4;
                            f->isFloat = true;
                            if (!f->matteOnly) f->rgb = rgbOf(tt);
                            f->matte = matteOf(tt);
                          },
                          &nb);
  FrameInputs in;
  in.w = w;
  in.h = h;
  in.renderScale = 1.0;
  in.time = t;
  in.rgb = rgb.data();
  in.base = matte.data();
  in.region = region.data();
  in.nb = needs.neighbourReach > 0 ? &nb : nullptr;
  in.modelGateOpen = true;
  std::vector<float> out(n);
  PlanResult res;
  runRecipe(recipe, in, ws, out.data(), &res, &wk, nullptr);

  std::ofstream rep(O + "/report.txt");
  rep << planLine(recipe, res) << "\n" << planDebugLine(recipe, res) << "\n";
  for (const StepReport& r : res.steps)
    rep << r.id << " ran=" << r.ran << " failed=" << r.failed << " changed=" << r.changed << " note=" << r.note
        << " summary=" << r.summary << " ms=" << r.ms << "\n";
  rep << "neighbours:";
  for (int o : nb.offsets) rep << " " << o;
  rep << "\nmodel cache hits " << cache.hits << " misses " << cache.misses << "\n";
  rep << "model crops " << model.stats.crops.size() << " stretched " << model.stats.stretched << " raised "
      << model.stats.pixelsRaised << " piecesRemoved " << model.stats.piecesRemoved << " box " << model.stats.bx0 << ","
      << model.stats.by0 << "-" << model.stats.bx1 << "," << model.stats.by1 << "\n";
  for (const auto& c : model.stats.crops)
    rep << "  crop " << c.x << "," << c.y << " " << c.size_w << "x" << c.size_h << " cell " << c.cx0 << "," << c.cy0
        << "-" << c.cx1 << "," << c.cy1 << "\n";
  rep << "vote ran " << vote.ran << " kept " << vote.last.kept << " dropped " << vote.last.dropped << " reinstated "
      << vote.last.reinstated << " fallback " << vote.last.fallback << "\n";
  writeF(O + "/final.f32", out);
  writeF(O + "/base.f32", matte);
  if (const std::vector<float>* rb = ws.ev.plane("regionBase")) writeF(O + "/regionBase.f32", *rb);
  {
    StepContext c{in, res.stats, ws.ev, res.finalVersion, &wk, nullptr};
    writeF(O + "/regionWeight.f32", regionWeight(c, s.region));
  }
  for (const CandidatePlanes& cp : ws.ev.candidates()) {
    std::vector<float> a(n, 0.0f), cf(n, 0.0f);
    for (int y = cp.box.y0; y < cp.box.y1; ++y)
      for (int x = cp.box.x0; x < cp.box.x1; ++x) {
        const size_t i = size_t(y - cp.box.y0) * cp.box.w() + (x - cp.box.x0);
        a[size_t(y) * w + x] = cp.alpha[i];
        cf[size_t(y) * w + x] = cp.conf[i];
      }
    writeF(O + "/cand_" + cp.id + ".f32", a);
    writeF(O + "/cand_" + cp.id + "_conf.f32", cf);
  }
  std::ofstream pc(O + "/pieces.txt");
  pc << "crop " << tracer.crop.x0 << " " << tracer.crop.y0 << " " << tracer.crop.x1 << " " << tracer.crop.y1 << "\n";
  for (const tracer::Piece& p : tracer.centre.pieces) {
    pc << "piece shape " << p.shape << " anchored " << p.anchored << " edge " << p.touchesRegionEdge << " self "
       << p.selfReferenced << " len " << p.length << " width " << p.width << " conf " << p.confidence << " pts";
    for (const auto& q : p.polyline) pc << " " << q.x + tracer.crop.x0 << "," << q.y + tracer.crop.y0;
    pc << "\n";
  }
  for (const tracer::Join& j : tracer.centre.joins) {
    pc << "join toMatte " << j.toMatte << " a " << j.pieceA << " b " << j.pieceB << " width " << j.width << " pts";
    for (const auto& q : j.polyline) pc << " " << q.x + tracer.crop.x0 << "," << q.y + tracer.crop.y0;
    pc << "\n";
  }
  for (size_t i = 0; i < tracer.centre.rejected.size(); ++i) {
    const tracer::Piece& p = tracer.centre.rejected[i];
    pc << "rejected why " << tracer.centre.rejectReason[i] << " anchored " << p.anchored << " edge " << p.touchesRegionEdge
       << " len " << p.length << " width " << p.width << " conf " << p.confidence << " pts";
    for (const auto& q : p.polyline) pc << " " << q.x + tracer.crop.x0 << "," << q.y + tracer.crop.y0;
    pc << "\n";
  }
  for (const vote::PieceVote& v : vote.last.pieces) {
    pc << "vote index " << v.index << " fate " << int(v.fate) << " score " << v.score << " matched " << v.matched
       << " judged " << v.judged << " conf " << v.confidence << " bar " << v.bar << " width " << v.width << " pts";
    for (const auto& q : v.polyline) pc << " " << q.x + tracer.crop.x0 << "," << q.y + tracer.crop.y0;
    pc << "\n";
  }
  pc << "votenote " << vote.last.note << "\n";
  std::printf("%s\n", planLine(recipe, res).c_str());
  return 0;
}
