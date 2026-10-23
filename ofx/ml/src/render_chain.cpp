#include "render_chain.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

#include "runner_client.h"

namespace mr {
namespace ml {

namespace {
double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

const char* segmentationModel(int segmentation) {
  return segmentation == kSegFull ? "birefnet" : "birefnet-lite";
}

SmartFillParams smartFillParams(const SmartFillUi& ui, double renderScale) {
  SmartFillParams p;
  p.subjectThreshold = ui.subjectThreshold;
  p.maxHoleFraction = std::max(ui.maxHolePercent, 0.0) / 100.0;
  p.rimPx *= renderScale;  // decision 6, as plugin 1's Fill Holes Rim px
  p.ringPx *= renderScale;
  p.bgReachPx *= renderScale;
  return p;
}

int temporalK(int framesIndex) { return framesIndex == kTemporal2 ? 2 : 1; }

StabiliseParams temporalParams(const TemporalUi& ui, double renderScale) {
  StabiliseParams p;
  p.frames = temporalK(ui.frames);
  p.strength = std::min(std::max(ui.strength, 0.0), 1.0);
  p.bandPx = std::max(ui.bandPx, 0.0) * renderScale;  // decision 6
  return p;
}

namespace {
bool inRange(double time, const FrameRange& r) {
  return !r.valid || (time >= r.first - 1e-6 && time <= r.last + 1e-6);
}
}  // namespace

std::vector<int> neighbourOffsets(double t, int K, const FrameRange& r) {
  std::vector<int> o;
  for (int k = 1; k <= K; ++k)
    for (int s : {-k, k})
      if (inRange(t + s, r)) o.push_back(s);
  return o;
}

void framesNeededRange(double t, int K, const FrameRange& r, double* lo, double* hi) {
  *lo = t - std::max(K, 0);
  *hi = t + std::max(K, 0);
  if (r.valid) {
    *lo = std::max(*lo, r.first);
    *hi = std::min(*hi, r.last);
  }
  *lo = std::min(*lo, t);
  *hi = std::max(*hi, t);
}

void collectNeighbours(double t, int K, const FrameRange& r, int w, int h, const FrameFetcher& fetch, Neighbours* nb,
                       const std::function<bool()>& abort) {
  *nb = Neighbours();
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  const std::vector<int> offsets = neighbourOffsets(t, K, r);
  nb->requested = int(offsets.size());
  std::vector<std::pair<int, FetchedFrame>> used;
  for (int k : offsets) {
    if (abort && abort()) {
      nb->aborted = true;
      break;
    }
    FetchedFrame f;
    const auto t0 = std::chrono::steady_clock::now();
    fetch(t + k, &f);
    nb->fetchMs += msSince(t0);
    NeighbourReport rep;
    rep.offset = k;
    rep.time = t + k;
    rep.w = f.w;
    rep.h = f.h;
    rep.components = f.components;
    rep.ms = f.ms;
    rep.note = f.note;
    char b[96];
    if (!f.image) {
      rep.status = f.error.empty() ? "no image" : "threw: " + f.error;
    } else if (!f.isFloat) {
      rep.status = "not float";
    } else if (f.components != 4) {
      std::snprintf(b, sizeof b, "%d channels (no matte)", f.components);
      rep.status = b;
    } else if (f.w != w || f.h != h) {
      std::snprintf(b, sizeof b, "size %dx%d, centre %dx%d", f.w, f.h, w, h);
      rep.status = b;
    } else if (f.matte.size() != n || f.rgb.size() != 3 * n) {
      rep.status = "no pixels";
    } else {
      rep.status = "used";
      rep.used = true;
      used.emplace_back(k, std::move(f));
    }
    nb->report.push_back(rep);
  }
  std::sort(used.begin(), used.end(),
            [](const std::pair<int, FetchedFrame>& a, const std::pair<int, FetchedFrame>& b) { return a.first < b.first; });
  for (auto& u : used) {
    nb->offsets.push_back(u.first);
    nb->frames.push_back(std::move(u.second));
  }
}

void runTemporal(const float* matte, const float* rgb, int w, int h, const StabiliseParams& p, const Neighbours& nb,
                 float* out, StabiliseStats* stats, bool* aborted, const std::function<bool()>& abort) {
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  *aborted = false;
  if (stats) *stats = StabiliseStats();
  if (nb.offsets.size() != nb.frames.size()) throw std::invalid_argument("Temporal: neighbour list mismatch");
  for (size_t i = 0; i < nb.frames.size(); ++i) {
    const FetchedFrame& f = nb.frames[i];
    if (nb.offsets[i] == 0 || f.matte.size() != n || f.rgb.size() != 3 * n) {
      char b[128];
      std::snprintf(b, sizeof b, "Temporal: neighbour %+d has no %dx%d matte and picture", nb.offsets[i], w, h);
      throw std::invalid_argument(b);
    }
  }
  if (out != matte && n) std::copy(matte, matte + n, out);
  if (abort && abort()) {
    *aborted = true;
    return;
  }
  std::vector<TemporalFrame> frames;
  for (size_t i = 0; i < nb.frames.size(); ++i)
    if (nb.offsets[i] < 0) frames.push_back({nb.frames[i].rgb.data(), nb.frames[i].matte.data()});
  const int centre = int(frames.size());
  frames.push_back({rgb, matte});
  for (size_t i = 0; i < nb.frames.size(); ++i)
    if (nb.offsets[i] > 0) frames.push_back({nb.frames[i].rgb.data(), nb.frames[i].matte.data()});
  stabilise(frames.data(), int(frames.size()), centre, w, h, p, out, stats);
  if (abort && abort()) {
    *aborted = true;
    if (out != matte && n) std::copy(matte, matte + n, out);
  }
}

void RunnerSeg::segment(const float* rgb, int w, int h, float* mask) {
  const int S = kBiRefNetSize;
  ++calls;
  in_.resize(size_t(3) * S * S);
  out_.resize(size_t(S) * S);
  auto t = std::chrono::steady_clock::now();
  birefnetPrepare(rgb, w, h, in_.data());
  prepMs = msSince(t);
  t = std::chrono::steady_clock::now();
  client_.runFrame(model_, in_.data(), 3, S, S, out_.data(), 1);
  runMs = msSince(t);
  t = std::chrono::steady_clock::now();
  birefnetPostprocess(out_.data(), w, h, mask);
  postMs = msSince(t);
}

void runChain(const float* matte, const float* rgb, int w, int h, const ChainParams& p, SegRunner& seg,
              TileRunner& tiles, std::vector<float>* mid, std::vector<float>* out, ChainResult* r,
              const std::function<bool()>& abort, std::vector<float>* trimapOut, const std::function<void()>& between,
              std::vector<float>* temporalOut) {
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  *r = ChainResult();
  const float* in = matte;  // Smart Fill's input: Temporal's output when it ran
  std::vector<float> tm;
  if (p.temporal) {
    r->tRan = true;
    tm.resize(n);
    try {
      const Neighbours none;
      runTemporal(matte, rgb, w, h, temporalParams(p.tm, p.renderScale), p.neighbours ? *p.neighbours : none,
                  tm.data(), &r->tm, &r->tAborted, abort);
      in = tm.data();
    } catch (const std::exception& e) {
      r->tFailed = true;  // the step passes the matte through
      r->tError = e.what();
    }
  }
  if (temporalOut) temporalOut->assign(in, in + n);
  mid->assign(in, in + n);
  if (p.smartFill) {
    r->sfRan = true;
    try {
      smartFill(in, rgb, w, h, smartFillParams(p.sf, p.renderScale), seg, mid->data(), &r->sf, abort);
    } catch (const std::exception& e) {
      mid->assign(in, in + n);  // the core leaves out undefined after a throw
      r->sfFailed = true;
      r->sfError = e.what();
    }
  }
  if (between) between();
  out->assign(mid->begin(), mid->end());
  if (p.edgeRefine) {
    r->erRan = true;
    try {
      refineEdge(mid->data(), rgb, out->data(), w, h, p.er, tiles, &r->er, abort, trimapOut);
    } catch (const std::exception& e) {
      out->assign(mid->begin(), mid->end());
      r->erFailed = true;
      r->erError = e.what();
    }
  }
}

void markHoles(const float* matte, int w, int h, float solidAbove, const SmartFillStats& st,
               std::vector<uint8_t>* marks) {
  marks->assign(size_t(std::max(w, 0)) * size_t(std::max(h, 0)), kMarkNone);
  std::vector<size_t> stack;
  for (const HoleInfo& hi : st.holes) {
    const uint8_t m = hi.decision == HoleDecision::Filled        ? kMarkFilled
                      : hi.decision == HoleDecision::KeptRealGap ? kMarkKept
                                                                 : kMarkNone;
    if (m == kMarkNone || hi.sx < 0 || hi.sy < 0 || hi.sx >= w || hi.sy >= h) continue;
    const size_t s = size_t(hi.sy) * w + hi.sx;
    if ((*marks)[s] != kMarkNone || matte[s] >= solidAbove) continue;
    (*marks)[s] = m;
    stack.assign(1, s);
    while (!stack.empty()) {
      const size_t i = stack.back();
      stack.pop_back();
      const int x = int(i % size_t(w)), y = int(i / size_t(w));
      auto visit = [&](size_t j) {
        if ((*marks)[j] == kMarkNone && !(matte[j] >= solidAbove)) {  // NaN is not solid (as the core)
          (*marks)[j] = m;
          stack.push_back(j);
        }
      };
      if (x > 0) visit(i - 1);
      if (x + 1 < w) visit(i + 1);
      if (y > 0) visit(i - size_t(w));
      if (y + 1 < h) visit(i + size_t(w));
    }
  }
}

}  // namespace ml
}  // namespace mr
