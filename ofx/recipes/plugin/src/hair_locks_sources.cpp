#include "hair_locks_sources.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

#include "merge_morph.h"
#include "ops/region_trimap.h"
#include "render_chain.h"

namespace mr {
namespace recipes {

namespace {

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

// The box grown by m px on every side, inside the frame.
Box grown(const Box& b, int m, int w, int h) {
  Box g;
  g.x0 = std::max(0, b.x0 - m);
  g.y0 = std::max(0, b.y0 - m);
  g.x1 = std::min(w, b.x1 + m);
  g.y1 = std::min(h, b.y1 + m);
  return g;
}

// Copies the box's part of a (src-box) crop plane into a box-sized plane.
void copyBox(const float* crop, const Box& cb, const Box& b, float* out) {
  for (int y = b.y0; y < b.y1; ++y)
    std::copy(crop + size_t(y - cb.y0) * cb.w() + (b.x0 - cb.x0), crop + size_t(y - cb.y0) * cb.w() + (b.x1 - cb.x0),
              out + size_t(y - b.y0) * b.w());
}

// Marks a disc of radius r at (x, y) in a w x h byte plane.
void stampDisc(std::vector<uint8_t>& m, int w, int h, double x, double y, double r) {
  const int x0 = std::max(0, int(std::floor(x - r))), x1 = std::min(w - 1, int(std::ceil(x + r)));
  const int y0 = std::max(0, int(std::floor(y - r))), y1 = std::min(h - 1, int(std::ceil(y + r)));
  for (int yy = y0; yy <= y1; ++yy)
    for (int xx = x0; xx <= x1; ++xx)
      if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r) m[size_t(yy) * w + xx] = 1;
}

// A polyline as capsules of radius r (discs at each point and each midpoint: points are 1.5 px apart).
void stampPolyline(std::vector<uint8_t>& m, int w, int h, const std::vector<tracer::Point>& pl, double r) {
  for (size_t i = 0; i < pl.size(); ++i) {
    stampDisc(m, w, h, pl[i].x, pl[i].y, r);
    if (i + 1 < pl.size()) stampDisc(m, w, h, 0.5 * (pl[i].x + pl[i + 1].x), 0.5 * (pl[i].y + pl[i + 1].y), r);
  }
}

}  // namespace

// --- model ------------------------------------------------------------------------------------------
void ModelSource::produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
                          StepReport* r) {
  recipe::CropRunner* run = runner ? runner() : nullptr;
  if (!run) throw std::runtime_error("runner not available");
  const int w = c.in.w, h = c.in.h;
  const size_t n = size_t(w) * h;
  const std::vector<float>& wgt = regionWeight(c, region);
  alpha_.resize(n);
  conf_.resize(n);
  stats = recipe::ModelCandidateStats();
  recipe::modelCandidate(c.in.rgb, matte, wgt.data(), w, h, p, *run, alpha_.data(), conf_.data(), &stats);
  const Box frame{0, 0, w, h};
  copyBox(alpha_.data(), frame, box, alpha);
  copyBox(conf_.data(), frame, box, conf);
  r->changed = stats.pixelsRaised;
  r->summary = std::to_string(stats.crops.size()) + (stats.crops.size() == 1 ? " crop" : " crops") +
               (stats.stretched ? ", stretched" : "");
  if (c.wantsQc(id())) {  // View Trimap: the model's trimap, as model_candidate.cpp builds it per crop
    RegionTrimapParams tp;
    tp.fgThreshold = p.fgThreshold;
    std::vector<float>& tri = c.qc->plane;
    tri.resize(n);
    regionTrimap(matte, wgt.data(), w, h, tp, tri.data());
    for (size_t i = 0; i < n; ++i) tri[i] = matte[i] >= p.fgThreshold ? 1.0f : recipe::quantiseTrimap(tri[i]);
  }
}

// --- tracer -----------------------------------------------------------------------------------------
void TracerSource::fillCrop(const float* rgb, const float* matte, int w, std::vector<float>& rgbOut,
                            std::vector<float>& baseOut) const {
  const int cw = crop.w(), ch = crop.h();
  rgbOut.resize(size_t(cw) * ch * 3);
  baseOut.resize(size_t(cw) * ch);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      const size_t i = size_t(crop.y0 + y) * w + (crop.x0 + x), o = size_t(y) * cw + x;
      for (int k = 0; k < 3; ++k) {
        const float v = rgb[i * 3 + k];
        rgbOut[o * 3 + k] = linearise ? (v > 0.0f ? std::pow(v, 2.4f) : 0.0f) : v;
      }
      baseOut[o] = matte[i];
    }
}

tracer::TracerResult TracerSource::trace(const std::vector<float>& rgb, const std::vector<float>& base,
                                         const float* reg, int, int frameH, double renderScale,
                                         const tracer::TracerParams& tp) const {
  tracer::TracerInput in;
  in.rgb = rgb.data();
  in.channels = 3;
  in.base = base.data();
  in.region = reg;
  in.width = crop.w();
  in.height = crop.h();
  in.renderScale = renderScale > 0 ? renderScale : 1.0;
  in.frameHeight = int(std::lround(frameH / in.renderScale));
  return tracer::traceStrands(in, tp);
}

void TracerSource::keepAnchoredOnly(std::vector<float>& alpha, std::vector<float>& conf) {
  const int cw = crop.w(), ch = crop.h(), np = int(centre.pieces.size());
  keep_.assign(size_t(cw) * ch, 0);
  for (const tracer::Piece& pc : centre.pieces)
    if (pc.anchored) stampPolyline(keep_, cw, ch, pc.polyline, 0.5 * pc.width + 1.5);
  for (const tracer::Join& j : centre.joins) {
    const bool a = j.toMatte || (j.pieceA >= 0 && j.pieceA < np && centre.pieces[size_t(j.pieceA)].anchored) ||
                   (j.pieceB >= 0 && j.pieceB < np && centre.pieces[size_t(j.pieceB)].anchored);
    if (a) stampPolyline(keep_, cw, ch, j.polyline, 0.5 * j.width + 1.5);
  }
  for (size_t i = 0; i < keep_.size() && i < alpha.size(); ++i)
    if (!keep_[i]) alpha[i] = conf[i] = 0.0f;
}

void TracerSource::produce(const StepContext& c, const float* matte, const Box& box, float* alpha, float* conf,
                           StepReport* r) {
  const int w = c.in.w, h = c.in.h;
  crop = grown(box, std::max(p.cropMargin, 0), w, h);  // detectors see past the box edge
  const int cw = crop.w(), ch = crop.h();
  const std::vector<float>& wgt = regionWeight(c, region);
  fillCrop(c.in.rgb, matte, w, rgbCrop, baseCrop);
  weightCrop.resize(size_t(cw) * ch);
  for (int y = 0; y < ch; ++y)
    std::copy(&wgt[size_t(crop.y0 + y) * w + crop.x0], &wgt[size_t(crop.y0 + y) * w + crop.x0] + cw,
              &weightCrop[size_t(y) * cw]);
  centre = trace(rgbCrop, baseCrop, weightCrop.data(), w, h, c.in.renderScale,
                 forVote ? vote::candidateTracerParams(p) : p);
  std::vector<float> a = centre.alpha, cf = centre.confidence;
  pieces = int(centre.pieces.size());
  if (forVote) {  // the real bars on the frame's own pieces: the vote's fallback (no neighbour)
    vote::VoteInput vi;
    vi.width = cw;
    vi.height = ch;
    vi.region = weightCrop.data();
    vi.widenedRegion = weightCrop.data();
    vi.base = baseCrop.data();
    vi.centre = &centre;
    vote::VoteResult own = vote::votePresence(vi, vp);
    a.swap(own.alpha);
    cf.swap(own.confidence);
    pieces = own.kept;
  }
  anchored = 0;
  for (const tracer::Piece& pc : centre.pieces) anchored += pc.anchored;
  if (!keepLoose) {  // Loose Pieces off: alpha only on anchored pieces and their joins
    keepAnchoredOnly(a, cf);
    pieces = std::min(pieces, anchored);
  }
  copyBox(a.data(), crop, box, alpha);
  copyBox(cf.data(), crop, box, conf);
  long long px = 0;
  for (long long i = 0; i < box.pixels(); ++i) px += alpha[i] > 0.0f;
  r->changed = px;
  r->summary = std::to_string(pieces) + (pieces == 1 ? " piece" : " pieces");
}

// --- the strand presence vote -------------------------------------------------------------------------
void VoteStep::run(const StepContext& c, const float* in, float*, StepReport* r) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); };
  ran = false;
  flowMs = traceMs = voteMs = 0;
  traced = 0;
  CandidatePlanes* cand = c.ev.findCandidate("tracer");
  if (!cand) throw std::runtime_error("no Rebuild Detail result");
  const int w = c.in.w, h = c.in.h, cw = t_->crop.w(), ch = t_->crop.h();
  const size_t n = size_t(w) * h;
  // The neighbours within K that came with a picture (render_chain.h Neighbours, frame order).
  std::vector<size_t> used;
  if (c.in.nb)
    for (size_t i = 0; i < c.in.nb->offsets.size(); ++i) {
      const int k = c.in.nb->offsets[i];
      const ml::FetchedFrame& f = c.in.nb->frames[i];
      if (k != 0 && std::abs(k) <= K && f.matte.size() == n && f.rgb.size() == 3 * n) used.push_back(i);
    }
  // 1. The flows: on the flow crop, or the whole frame shared with Temporal (cost rule in the header).
  auto t0 = clock::now();
  const int s = ml::flowScale(h, flowParams.flowLines);
  Box fc;
  fc.x0 = std::max(0, (t_->crop.x0 - s * supportGrid) / s * s);
  fc.y0 = std::max(0, (t_->crop.y0 - s * supportGrid) / s * s);
  fc.x1 = std::min(w, t_->crop.x1 + s * supportGrid);
  fc.y1 = std::min(h, t_->crop.y1 + s * supportGrid);
  const double nv = double(used.size());
  usedCropFlows = double(fc.pixels()) * nv + double(n) * temporalFlows < double(n) * nv;
  ml::StabiliseParams sp = flowParams;
  sp.frames = K;
  int ox = t_->crop.x0, oy = t_->crop.y0;
  const ml::TemporalFlows* flows = nullptr;
  if (!used.empty()) {
    std::vector<const float*> fr(used.size()), fm(used.size());
    for (size_t k = 0; k < used.size(); ++k) fr[k] = c.in.nb->frames[used[k]].rgb.data(), fm[k] = c.in.nb->frames[used[k]].matte.data();
    int fw = w, fh = h;
    if (usedCropFlows) {  // crop every frame's picture and matte
      fw = fc.w();
      fh = fc.h();
      frgb_.resize(used.size() + 1);
      fmatte_.resize(used.size() + 1);
      auto cropOf = [&](const float* rgb, const float* m, size_t slot) {
        frgb_[slot].resize(size_t(fw) * fh * 3);
        fmatte_[slot].resize(size_t(fw) * fh);
        for (int y = 0; y < fh; ++y) {
          const size_t src = size_t(fc.y0 + y) * w + fc.x0;
          std::copy(rgb + src * 3, rgb + (src + fw) * 3, &frgb_[slot][size_t(y) * fw * 3]);
          std::copy(m + src, m + src + fw, &fmatte_[slot][size_t(y) * fw]);
        }
      };
      for (size_t k = 0; k < used.size(); ++k) {
        cropOf(fr[k], fm[k], k);
        fr[k] = frgb_[k].data();
        fm[k] = fmatte_[k].data();
      }
      cropOf(c.in.rgb, in, used.size());
      sp.flowLines = std::max(1, (fh + s - 1) / s);  // the whole frame's flow scale s on the crop
      ox -= fc.x0;
      oy -= fc.y0;
    }
    std::vector<ml::TemporalFrame> frames;
    for (size_t k = 0; k < used.size(); ++k)
      if (c.in.nb->offsets[used[k]] < 0) frames.push_back({fr[k], fm[k]});
    const int centre = int(frames.size());
    frames.push_back({usedCropFlows ? frgb_[used.size()].data() : c.in.rgb,
                      usedCropFlows ? fmatte_[used.size()].data() : in});
    for (size_t k = 0; k < used.size(); ++k)
      if (c.in.nb->offsets[used[k]] > 0) frames.push_back({fr[k], fm[k]});
    if (usedCropFlows) {
      cropFlows_ = ml::computeFlows(frames.data(), int(frames.size()), centre, fw, fh, sp);
      flows = &cropFlows_;
    } else {
      if (!c.ev.flows)
        c.ev.flows = std::make_shared<ml::TemporalFlows>(ml::computeFlows(frames.data(), int(frames.size()), centre, w, h, sp));
      flows = c.ev.flows.get();
    }
  }
  flowMs = ms(t0);
  // 2. Motion per neighbour on the tracer's crop; the region at t widened by it (ruling Q7).
  t0 = clock::now();
  std::vector<vote::DenseMotion> dm;
  for (size_t u : used) dm.push_back(vote::motionFromFlows(*flows, c.in.nb->offsets[u], ox, oy, cw, ch));
  std::vector<vote::Motion> mv;
  for (const vote::DenseMotion& d : dm) mv.push_back(d.view());
  const std::vector<float> wide = vote::widenRegion(t_->weightCrop.data(), cw, ch, mv.data(), int(mv.size()));
  voteMs = ms(t0);
  // 3. Every neighbour traced from its OWN picture and matte, the bars lowered as for the centre.
  t0 = clock::now();
  const tracer::TracerParams cp = vote::candidateTracerParams(t_->p);
  nres_.resize(used.size());
  for (size_t k = 0; k < used.size(); ++k) {
    const ml::FetchedFrame& f = c.in.nb->frames[used[k]];
    t_->fillCrop(f.rgb.data(), f.matte.data(), w, nrgb_, nbase_);
    if (movedRegions) {
      const std::vector<float> mr = vote::movedRegion(t_->weightCrop.data(), cw, ch, mv[k]);
      nres_[k] = t_->trace(nrgb_, nbase_, mr.data(), w, h, c.in.renderScale, cp);
    } else {
      nres_[k] = t_->trace(nrgb_, nbase_, wide.data(), w, h, c.in.renderScale, cp);
    }
  }
  traced = int(used.size());
  traceMs = ms(t0);
  // 4. The vote with the real bars.
  t0 = clock::now();
  std::vector<vote::Neighbour> nb(used.size());
  for (size_t k = 0; k < used.size(); ++k) {
    nb[k].result = &nres_[k];
    nb[k].motion = mv[k];
  }
  vote::VoteInput vi;
  vi.width = cw;
  vi.height = ch;
  vi.region = t_->weightCrop.data();
  vi.widenedRegion = wide.data();
  vi.base = t_->baseCrop.data();
  vi.centre = &t_->centre;
  vi.neighbours = nb.data();
  vi.count = int(nb.size());
  last = vote::votePresence(vi, t_->vp);
  if (!t_->keepLoose) t_->keepAnchoredOnly(last.alpha, last.confidence);
  copyBox(last.alpha.data(), t_->crop, cand->box, cand->alpha.data());
  copyBox(last.confidence.data(), t_->crop, cand->box, cand->conf.data());
  voteMs += ms(t0);
  ran = true;
  r->changed = last.kept + last.reinstated;
  r->summary = "kept " + std::to_string(last.kept) + " dropped " + std::to_string(last.dropped) + " reinstated " +
               std::to_string(last.reinstated) + (last.fallback ? ", own result" : "");
}

void VoteStep::droppedPlane(int w, int h, std::vector<float>& plane) const {
  plane.clear();
  if (!ran) return;
  plane.assign(size_t(w) * h, 0.0f);
  const int cw = t_->crop.w(), ch = t_->crop.h();
  std::vector<uint8_t> m(size_t(cw) * ch, 0);
  for (const vote::PieceVote& v : last.pieces)
    if (v.fate == vote::Fate::Dropped && v.index >= 0 && size_t(v.index) < t_->centre.pieces.size()) {
      const tracer::Piece& pc = t_->centre.pieces[size_t(v.index)];
      stampPolyline(m, cw, ch, pc.polyline, 0.5 * pc.width + 1.5);
    }
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      if (m[size_t(y) * cw + x]) plane[size_t(t_->crop.y0 + y) * w + (t_->crop.x0 + x)] = 1.0f;
}

// --- Detail -----------------------------------------------------------------------------------------
void DetailSource::produce(const StepContext& c, const float*, const Box& box, float* alpha, float* conf,
                           StepReport* r) {
  const int w = c.in.w, h = c.in.h;
  // Matte Merge's openRadius (matte_merge.cpp:32): an opening of 2r + 1 >= the width removes thin structures.
  const int openR = std::max(1, int(std::ceil((detailWidthPx - 1.0) / 2.0)));
  const Box cb = grown(box, openR + ep.radius + ep.smoothRadius + 1, w, h);
  const int cw = cb.w(), ch = cb.h();
  const size_t cn = size_t(cw) * ch;
  d_.resize(cn);
  rgb_.resize(cn * 3);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      const size_t i = size_t(cb.y0 + y) * w + (cb.x0 + x), o = size_t(y) * cw + x;
      d_[o] = clamp01(c.in.detail[i]);
      for (int k = 0; k < 3; ++k) rgb_[o * 3 + k] = c.in.rgb[i * 3 + k];
    }
  op_.resize(cn);
  fine_.resize(cn);
  ev_.resize(cn);
  cand_.resize(cn);
  merge::opening(d_.data(), cw, ch, openR, op_.data(), c.wk, &scratch_);
  for (size_t i = 0; i < cn; ++i) {
    fine_[i] = std::max(0.0f, d_[i] - op_[i]);
    cand_[i] = fine_[i] > 0.01f;
  }
  merge::pictureEvidence(rgb_.data(), fine_.data(), cand_.data(), cw, ch, ep, ev_.data(), nullptr, c.wk);
  for (size_t i = 0; i < cn; ++i) ev_[i] = cand_[i] ? ev_[i] : 1.0f;
  copyBox(d_.data(), cb, box, alpha);
  copyBox(ev_.data(), cb, box, conf);
  long long thin = 0;
  for (int y = box.y0; y < box.y1; ++y)
    for (int x = box.x0; x < box.x1; ++x) thin += cand_[size_t(y - cb.y0) * cw + (x - cb.x0)];
  r->summary = std::to_string(thin) + " thin px";
}

}  // namespace recipes
}  // namespace mr
