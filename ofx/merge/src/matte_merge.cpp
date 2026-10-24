#include "matte_merge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "merge_evidence.h"
#include "merge_morph.h"
#include "merge_picreg.h"
#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

constexpr float kClear = 0.02f, kSolid = 0.98f;  // a base / trimap value is "known" beyond these
constexpr float kPresent = 0.05f;                // a detail pixel above this is part of a component
// Bridge: a kept thin-structure (top-hat) value above this marks a strand the bridge may join to the base.
// Measured 2026-10-24 on hair_flat 1080: the hairless body edge's top-hat is 0 at Detail Width 5.4 px and at
// most 0.155 at 54 px (curvature); strand cores reach 0.3-0.8.
constexpr float kStrandSeed = 0.25f;

float smoothstep01(double t) {
  t = std::min(1.0, std::max(0.0, t));
  return float(t * t * (3 - 2 * t));
}

// Opening half-size for Detail Width px: structures narrower than 2r + 1 >= width vanish.
int openRadius(double widthPx) { return std::max(1, int(std::ceil((widthPx - 1.0) / 2.0))); }

// fn(i) for every pixel, rows split over the threads.
template <class F>
void forPixels(Workers* wk, int w, int h, F fn) {
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) fn(i);
  });
}

// Picture estimate thresholds (measured 2026-10-24 on hair_flat / hair_busy, 2880x1620: a good
// translation + scale fit had uncertainty 0.014 / 0.063 px, RMS 0.06 / 0.12 px, agreement 0.90 /
// 0.32; translation only on the same scenes (true scale 1.01): RMS 0.50-0.66 px, agreement
// 0.07-0.16).
constexpr double kMaxUncertaintyPx = 0.25, kMaxRmsPx = 0.35, kMinAgreement = 0.25;
// Matte estimate: above this sensitivity translation, scale and edge offset trade off (the coarse
// hair base: 9, 1.45 px off); the scale is then not used.
constexpr double kMaxSensitivity = 4.0;

bool acceptable(const PictureFit& f) {
  return f.ok && f.uncertaintyPx <= kMaxUncertaintyPx && f.rmsPx <= kMaxRmsPx &&
         f.inliers >= kMinAgreement * f.blocks;
}

// The move of the detail (decision of the user, 2026-10-24): on the picture (translation + scale,
// else translation only), else on the mattes (translation + scale if well conditioned, else
// translation only), else none.
Registration estimateAlign(const float* base, const float* da, const float* picture, int w, int h, bool trimap,
                           int openR, const uint8_t* zone, Workers* wk, AlignInfo* ai) {
  Registration id;
  id.cx = (w - 1) / 2.0;
  id.cy = (h - 1) / 2.0;
  const float level = trimap ? 0.75f : 0.5f;
  // The start: the mattes' estimate at 1/4 resolution (refined only when it is the answer).
  ai->matte = registerMattes(base, da, w, h, level, openR, wk, false, false);
  Registration mt;  // the matte translation-only estimate, computed when needed
  bool haveMt = false;
  auto matteTranslation = [&]() -> const Registration& {
    if (!haveMt) mt = registerMattes(base, da, w, h, level, openR, wk, true), haveMt = true;
    return mt;
  };
  const double k = h / 1080.0;
  if (picture) {
    const Registration init = ai->matte.ok ? ai->matte : id;
    const int search = ai->matte.ok ? std::max(3, int(std::lround(4 * k))) : std::max(4, int(std::lround(10 * k)));
    PictureFit f = registerToPicture(da, picture, w, h, openR, init, false, search, zone, wk);
    if (!acceptable(f)) {
      Registration t = id;
      if (matteTranslation().ok) t.tx = mt.tx, t.ty = mt.ty;
      const PictureFit ft = registerToPicture(da, picture, w, h, openR, t, true, search, zone, wk);
      if (acceptable(ft)) {
        f = ft;
        ai->model = AlignModel::PictureTranslation;
      }
    } else {
      ai->model = AlignModel::PictureFull;
    }
    ai->blocks = f.blocks;
    ai->inliers = f.inliers;
    if (ai->model != AlignModel::Off) {
      ai->uncertaintyPx = f.uncertaintyPx;
      ai->agreement = f.blocks ? double(f.inliers) / f.blocks : 0.0;
      return f.reg;
    }
  }
  ai->matte = registerMattes(base, da, w, h, level, openR, wk);
  if (ai->matte.ok && ai->matte.sensitivity <= kMaxSensitivity) {
    ai->model = AlignModel::MatteFull;
    ai->uncertaintyPx = ai->matte.sensitivity * ai->matte.rmsPx;
    return ai->matte;
  }
  // No picture model and an ill-conditioned matte model: no alignment (user, 2026-10-24: a predictable
  // error beats a hidden drift; a matte translation-only estimate left up to 13.9 px where the scale is real).
  ai->model = AlignModel::None;
  return id;
}

}  // namespace

const char* alignModelName(AlignModel m) {
  switch (m) {
    case AlignModel::Off: return "off";
    case AlignModel::None: return "none";
    case AlignModel::PictureFull: return "picture: move + scale";
    case AlignModel::PictureTranslation: return "picture: move only";
    case AlignModel::MatteFull: return "mattes: move + scale";
  }
  return "?";
}

void scaleToHeight(MergeParams& p, const HeightSizes& s, int h) {
  p.reachPx = s.reachPct / 100.0 * h;
  p.detailWidthPx = s.detailWidthPct / 100.0 * h;
  p.featherPx = s.featherPct / 100.0 * h;
  p.bridgePx = s.bridgePct / 100.0 * h;
}

void merge(const float* base, const float* detail, const float* picture, int w, int h, const MergeParams& p,
           float* out, const MergeViews& views, MergeInfo* info, MergeWorkspace* wsp) {
  const size_t n = size_t(w) * h;
  Workers wk;
  MergeWorkspace localWs;
  MergeWorkspace& ws = wsp ? *wsp : localWs;
  const bool trimap = p.source == Source::Trimap;
  // Strength 0-1 scales how much of the detail is taken; above 1 it is also a gain on what is ADDED (the detail where it
  // exceeds the reference), clamped to 1: a blurred detail matte has weaker strands than the truth (user, 2026-10-24).
  const double strength = std::min(1.0, std::max(0.0, p.strength));
  const double gain = std::min(2.0, std::max(1.0, p.strength));
  // The gain is a curve, 1 - (1 - x)^gain: it lifts weak (grey) values the most and leaves strong ones, which are already
  // near 1, almost alone (user, 2026-10-24: strong on the grey tips, not on the solid roots). gain 1 = identity.
  auto boost = [gain](double x) { return gain > 1.0 ? 1.0 - std::pow(1.0 - std::min(1.0, std::max(0.0, x)), gain) : x; };
  const double feather = std::max(0.0, p.featherPx);
  MergeInfo inf;
  inf.reg.cx = (w - 1) / 2.0;
  inf.reg.cy = (h - 1) / 2.0;

  // 1. The zone and its feather ramp. (Workspace buffers: resized, every pixel written below.)
  std::vector<uint8_t>& zone = ws.zone;
  std::vector<uint8_t>& tmp = ws.mask;
  std::vector<float>& d2 = ws.d2;
  zone.resize(n);
  tmp.resize(n);
  d2.resize(n);
  if (!trimap) {
    parallelFor(&wk, h, 8, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          tmp[i] = (base[i] > kClear && base[i] < kSolid) || isContour(base, w, h, x, y);
        }
    });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);  // to the grey pixels / the contour
    const double reach2 = p.reachPx * p.reachPx;
    forPixels(&wk, w, h, [&](size_t i) {
      zone[i] = d2[i] <= reach2;
      tmp[i] = base[i] < kSolid;
    });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);  // solid pixels: to the nearest non-solid one
    const double core = std::max(2.0, feather);
    forPixels(&wk, w, h, [&](size_t i) {
      if (base[i] >= kSolid && d2[i] > core * core) zone[i] = 0;
    });
  } else {
    forPixels(&wk, w, h, [&](size_t i) { zone[i] = base[i] > kClear && base[i] < kSolid; });
  }
  std::vector<float>& ramp = ws.ramp;
  ramp.resize(n);
  if (feather > 0) {
    forPixels(&wk, w, h, [&](size_t i) { tmp[i] = !zone[i]; });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) {
      ramp[i] = !zone[i] ? 0.0f : d2[i] >= kFar ? 1.0f : smoothstep01((std::sqrt(double(d2[i])) - 0.5) / feather);
    });
  } else {
    forPixels(&wk, w, h, [&](size_t i) { ramp[i] = zone[i]; });
  }
  inf.zonePixels = long(std::count(zone.begin(), zone.begin() + n, uint8_t(1)));

  // Trimap: the prior ramp from the known foreground (1) to the known background (0).
  std::vector<float>& prior = ws.prior;
  if (trimap) {
    prior.resize(n);
    std::vector<float>& dfg = ws.d2b;
    dfg.resize(n);
    forPixels(&wk, w, h, [&](size_t i) { tmp[i] = base[i] >= kSolid; });
    distanceSquared(tmp.data(), w, h, dfg.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) { tmp[i] = base[i] <= kClear; });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) {
      prior[i] = 0.5f;
      if (zone[i]) {
        const bool fg = dfg[i] < kFar, bg = d2[i] < kFar;
        const double a = std::sqrt(double(dfg[i])), b = std::sqrt(double(d2[i]));
        prior[i] = fg && bg ? float(b / (a + b)) : fg ? 1.0f : bg ? 0.0f : 0.5f;
      }
    });
  }

  // 2. Registration; the detail clamped to 0-1 (non-finite: 0).
  std::vector<float>& da = ws.da;
  da.resize(n);
  forPixels(&wk, w, h, [&](size_t i) {
    const float v = detail[i];
    da[i] = std::isfinite(v) ? std::min(1.0f, std::max(0.0f, v)) : 0.0f;
  });
  const int openR = openRadius(p.detailWidthPx);
  inf.align.model = AlignModel::Off;
  if (p.align && inf.zonePixels > 0) {
    const Registration reg = estimateAlign(base, da.data(), picture, w, h, trimap, openR, zone.data(), &wk, &inf.align);
    if (inf.align.model != AlignModel::None) {
      inf.reg = reg;
      inf.reg.ok = true;
      std::vector<float>& warped = ws.warped;
      warped.resize(n);
      warpDetail(da.data(), w, h, inf.reg, warped.data(), &wk);
      da.swap(warped);
    }
  } else if (p.align) {
    inf.align.model = AlignModel::None;
  }

  // 3. Connected detail: flood over zone pixels above kPresent from the anchors (anchors marked
  // in parallel, the flood itself sequential: it only visits connected detail).
  std::vector<uint8_t>& kept = ws.kept;
  kept.resize(n);
  parallelFor(&wk, h, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        bool anchor = false;
        if (zone[i] && da[i] > kPresent) {
          anchor = !trimap && base[i] >= 0.5f;
          if (trimap)
            for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1) && !anchor; ++yy)
              for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx)
                if (base[size_t(yy) * w + xx] >= kSolid) anchor = true;
        }
        kept[i] = anchor;
      }
  });
  std::vector<int>& stack = ws.stack;
  stack.clear();
  for (size_t i = 0; i < n; ++i)
    if (kept[i]) stack.push_back(int(i));
  while (!stack.empty()) {
    const int i = stack.back(), x = i % w, y = i / w;
    stack.pop_back();
    for (int yy = std::max(0, y - 1); yy <= std::min(h - 1, y + 1); ++yy)
      for (int xx = std::max(0, x - 1); xx <= std::min(w - 1, x + 1); ++xx) {
        const size_t j = size_t(yy) * w + xx;
        if (!kept[j] && zone[j] && da[j] > kPresent) {
          kept[j] = 1;
          stack.push_back(int(j));
        }
      }
  }
  for (size_t i = 0; i < n; ++i)
    if (zone[i] && da[i] > kPresent) (kept[i] ? inf.keptPixels : inf.droppedPixels) += 1;

  // 4. The detail's thin structures (white top-hat): what Add takes and what evidence judges.
  const bool add = !trimap && p.method == Method::AddFineDetail;
  std::vector<float>& opened = ws.opened;
  std::vector<float>& fine = ws.fine;
  opened.resize(n);
  fine.resize(n);
  opening(da.data(), w, h, openR, opened.data(), &wk, &ws.morph);
  forPixels(&wk, w, h, [&](size_t i) { fine[i] = zone[i] ? std::max(0.0f, da[i] - opened[i]) : 0.0f; });

  // 5. Evidence on the kept fine detail: unsupported thin structures are scaled down. A thick
  // area (the detail's body edge) is never judged: a smooth change has no structure to compare.
  const bool useEv = p.evidence && picture;
  std::vector<float>& ev = ws.ev;
  if (useEv) {
    ev.resize(n);
    std::vector<float>& sig = ws.sig;
    sig.resize(n);
    forPixels(&wk, w, h, [&](size_t i) {
      sig[i] = kept[i] ? fine[i] : 0.0f;
      tmp[i] = sig[i] > 0.01f;
    });
    pictureEvidence(picture, sig.data(), tmp.data(), w, h, EvidenceParams(), ev.data(), nullptr, &wk);
  }

  // 5b. Bridge weight (Add, Matte edge): 1 at the base's 0.5 line, 0 from bridgePx away (matte_merge.h). A
  // contour pixel is about half a pixel from the line, so the distance to the line is taken as dc + 0.5
  // (without it the band was half a pixel wider at any size: 18 % more bridged area at 160 than at 320 lines).
  // d2 is free here (Matte edge: only the zone and its ramp used it), tmp too (evidence is done).
  const bool bridge = add && p.bridgePx > 0;
  std::vector<float>& bw = ws.bridge;
  if (bridge) {
    bw.resize(n);
    parallelFor(&wk, h, 8, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) tmp[size_t(y) * w + x] = isContour(base, w, h, x, y);
    });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) {
      bw[i] = d2[i] >= kFar ? 0.0f : 1.0f - smoothstep01((std::sqrt(double(d2[i])) + 0.5) / p.bridgePx);
    });
    // Only where strands attach (user, after the Flame test 2026-10-24: on a hairless edge the bridge took
    // the detail's own soft, fatter body edge, a grey halo): times 1 within bridgePx of a strand (kept fine
    // > kStrandSeed), falling smoothly to 0 at 2 x bridgePx. The roots between the base and a strand's thin
    // part lie within the bridge band, so bridgePx from the strand reaches them; the second bridgePx is the
    // fade along the edge where the hair ends. No strand nearby: weight 0, the output of Bridge 0.
    forPixels(&wk, w, h, [&](size_t i) { tmp[i] = kept[i] && fine[i] > kStrandSeed; });
    distanceSquared(tmp.data(), w, h, d2.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) {
      if (bw[i] > 0)
        bw[i] *= d2[i] >= kFar ? 0.0f : 1.0f - smoothstep01((std::sqrt(double(d2[i])) - p.bridgePx) / p.bridgePx);
    });
  }
  // The bridge part of a pixel (kept or not; the caller gates it): b(d) max(0, detail - base).
  auto bridgeAt = [&](size_t i) -> double { return bridge ? bw[i] * std::max(0.0, double(da[i]) - base[i]) : 0.0; };

  // 6. Combine. Outside the zone, and wherever nothing is taken (Matte edge), the base bit for bit.
  // Blend / Trimap target: the detail with its unsupported fine part removed; dropped (unconnected)
  // detail = no change in a blend; in a trimap fill background, or the detail as it is
  // (keepDisconnected).
  const bool keepLoose = trimap && p.keepDisconnected;
  auto target = [&](size_t i) -> float {
    if (!kept[i] && da[i] > kPresent) return trimap ? (keepLoose ? da[i] : 0.0f) : base[i];
    if (!useEv || !kept[i]) return da[i];
    return opened[i] + ev[i] * fine[i];
  };
  parallelFor(&wk, h, 8, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      out[i] = base[i];
      if (!zone[i]) continue;
      const double wt = ramp[i] * strength;
      if (add) {
        double f = 0.0;
        if (kept[i]) {
          const double fb = bridgeAt(i);
          // fb = 0: the expression before the bridge, unchanged (bridge 0 = bit-identical output).
          f = fb > 0 ? boost(wt * (1.0 - (1.0 - (useEv ? ev[i] : 1.0) * fine[i]) * (1.0 - fb)))
                     : boost(wt * (useEv ? ev[i] : 1.0) * fine[i]);
        }
        if (f > 0) out[i] = float(base[i] + f * (1.0 - base[i]));  // >= base: f (1 - base) >= 0
      } else if (trimap) {
        const double tg = target(i);
        const double d = (tg > prior[i] ? boost(tg) : tg) - prior[i];
        out[i] = std::min(1.0f, std::max(0.0f, float(prior[i] + wt * d)));
      } else {
        const float t = target(i);
        const double d = (t > base[i] ? boost(t) : t) - base[i];
        if (wt > 0 && t != base[i]) out[i] = std::min(1.0f, std::max(0.0f, float(base[i] + wt * d)));
      }
    }
  });

  // Views.
  if (views.added) forPixels(&wk, w, h, [&](size_t i) { views.added[i] = out[i] - base[i]; });
  if (views.zone) forPixels(&wk, w, h, [&](size_t i) { views.zone[i] = ramp[i]; });
  if (views.rejected)
    forPixels(&wk, w, h, [&](size_t i) {
      float all = base[i];
      if (zone[i]) {
        const double wt = ramp[i] * strength;
        if (add) {
          const double fb = bridgeAt(i);  // every detail pixel, connected or not
          all = float(base[i] + boost(fb > 0 ? wt * (1.0 - (1.0 - fine[i]) * (1.0 - fb)) : wt * fine[i]) * (1.0 - base[i]));
        } else {
          const float ref = trimap ? prior[i] : base[i];
          const double d = (da[i] > ref ? boost(da[i]) : da[i]) - ref;
          all = std::min(1.0f, std::max(0.0f, float(ref + wt * d)));
        }
      }
      views.rejected[i] = std::fabs(all - out[i]);
    });
  if (views.residual) {
    std::vector<float>& ob = ws.warped;  // free again after the registration
    ob.resize(n);
    opening(base, w, h, openR, ob.data(), &wk, &ws.morph);
    forPixels(&wk, w, h, [&](size_t i) { views.residual[i] = zone[i] ? std::fabs(opened[i] - ob[i]) : 0.0f; });
  }
  if (info) *info = inf;
}

}  // namespace merge
}  // namespace mr
