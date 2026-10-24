// Matte Merge on the synthetic hair scenes (merge_synth.h) at 2880x1620: strand recovery, body-edge
// damage, blob and fake-strand rejection, registration error, bit-identity outside the zone, the
// trimap fill, evidence decisions on a display versus a log picture, and timing.
// Sizes: the defaults are HeightSizes (Reach 6 %, Detail Width 0.5 %, Feather 0.25 % of the height:
// 97.2, 8.1, 4.05 px here). The synthetic strands are up to 240 lines long at 1080 (360 px here),
// far longer than real hair, so most scores are also given with Reach 380 px ("reach 380").
// Every score is printed; the limits are regression limits set from the first measurement
// (2026-10-24) with a margin, not targets (the report says which scores are weak).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "matte_merge.h"
#include "merge_morph.h"
#include "merge_synth.h"
#include "ops/edge_field.h"

using namespace mr::merge;
using mmsynth::MergeScene;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

struct Run {
  std::vector<float> out, added, rejected, zone, residual;
  MergeInfo info;
  double ms = 0;
};

MergeParams defaults(int h) {
  MergeParams p;
  scaleToHeight(p, HeightSizes(), h);
  return p;
}

Run run(const MergeScene& m, const std::vector<float>& base, const float* picture, const MergeParams& p,
        bool views = true, MergeWorkspace* ws = nullptr) {
  Run r;
  const size_t n = size_t(m.w) * m.h;
  r.out.resize(n);
  MergeViews v;
  if (views) {
    r.added.resize(n);
    r.rejected.resize(n);
    r.zone.resize(n);
    r.residual.resize(n);
    v = MergeViews{r.added.data(), r.rejected.data(), r.zone.data(), r.residual.data()};
  }
  const auto t0 = std::chrono::steady_clock::now();
  merge(base.data(), m.detail.data(), picture, m.w, m.h, p, r.out.data(), v, &r.info, ws);
  r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

struct Scores {
  double recovery = 0;      // 1 - strand error after / before, on real-strand pixels outside the body
  double bodyChange = 0;    // mean |out - base| on the body edge (|sd| < 4 px, no strands nearby)
  double bodyOver = 0;      // mean max(0, out - truth) on the body edge, outside the truth body
  long bodyLowered = 0;     // body-edge pixels below the base
  double blob = 0;          // added / blob alpha on the blob
  double fake = 0;          // added / fake alpha on fake-strand pixels away from real strands
  double offsetPx = 0;      // body-edge position vs the truth, px (+ = fatter), where hair is (a strand within 60 design px)
  double haloMax = 0;       // hairless edge (no strand, fake or blob within 60 design px, base < 0.5, within 25 design
  double haloMean = 0;      // px of the body edge): max / mean of out - base (the bridge's halo seen in Flame)
  double root = 0;          // mean out on real-strand pixels (> 0.3) within 5 design px outside the body
  double rootTruth = 0;     // the same for the truth (the bridge's target, ofx/merge "Bridge")
};

Scores score(const MergeScene& m, const std::vector<float>& base, const std::vector<float>& out) {
  Scores s;
  const size_t n = base.size();
  const double k = m.k;
  double eOut = 0, eBase = 0, bc = 0, bo = 0, ba = 0, bt = 0, fa = 0, ft = 0, so = 0, st = 0, ro = 0, rt = 0;
  long nb = 0, no = 0, len = 0, nr = 0, nh = 0;
  std::vector<uint8_t> hair(n);
  std::vector<float> dh(n);
  for (size_t i = 0; i < n; ++i) hair[i] = m.strands[i] > 0.05f || m.fake[i] > 0.05f || m.blob[i] > 0.05f;
  distanceSquared(hair.data(), m.w, m.h, dh.data());
  const double far2 = 60 * k * 60 * k;
  for (size_t i = 0; i < n; ++i) {
    const double d = out[i] - base[i];
    if (m.strands[i] > 0.05f && m.body[i] < 0.5f) {
      eOut += std::fabs(out[i] - m.truth[i]);
      eBase += std::fabs(base[i] - m.truth[i]);
    }
    const bool clean = m.strands[i] < 0.01f && m.fake[i] < 0.01f && m.blob[i] < 0.01f;
    if (std::fabs(m.bodySd[i]) < 4 * k && clean) {
      bc += std::fabs(d), ++nb;
      s.bodyLowered += out[i] < base[i];
      if (m.bodySd[i] < 0) bo += std::max(0.0, double(out[i]) - m.truth[i]), ++no;
    }
    // Body-edge offset: (sum out - sum truth) over clean pixels within 15 design px of the body edge,
    // per px of edge length (clean pixels within 0.5 px of it): the mean outward shift of the edge.
    if (clean && dh[i] <= far2 && std::fabs(m.bodySd[i]) < 15 * k) so += out[i], st += m.truth[i];
    if (clean && dh[i] <= far2 && std::fabs(m.bodySd[i]) < 0.5) ++len;
    if (dh[i] > far2 && base[i] < 0.5f && std::fabs(m.bodySd[i]) < 25 * k)
      s.haloMax = std::max(s.haloMax, d), s.haloMean += d, ++nh;
    if (m.strands[i] > 0.3f && m.bodySd[i] <= 0 && m.bodySd[i] > -5 * k) ro += out[i], rt += m.truth[i], ++nr;
    if (m.blob[i] > 0.05f) ba += d, bt += m.blob[i];
    if (m.fake[i] > 0.05f && m.strands[i] < 0.02f && m.body[i] < 0.05f && m.blob[i] < 0.01f) fa += d, ft += m.fake[i];
  }
  s.recovery = eBase > 0 ? 1.0 - eOut / eBase : 0;
  s.bodyChange = nb ? bc / nb : 0;
  s.bodyOver = no ? bo / no : 0;
  s.blob = bt > 0 ? ba / bt : 0;
  s.fake = ft > 0 ? fa / ft : 0;
  s.offsetPx = len ? (so - st) / len : 0;
  s.haloMean = nh ? s.haloMean / nh : 0;
  s.root = nr ? ro / nr : 0;
  s.rootTruth = nr ? rt / nr : 0;
  return s;
}

void print(const char* what, const Scores& s) {
  std::printf("  %-34s strand recovery %6.3f | body edge: change %.4f over %.4f lowered %ld offset (hair) %+.2f px | halo "
              "(hairless) max %.4f mean %.5f | roots %.3f (truth %.3f) | blob %.4f | fake %.3f\n",
              what, s.recovery, s.bodyChange, s.bodyOver, s.bodyLowered, s.offsetPx, s.haloMax, s.haloMean, s.root,
              s.rootTruth, s.blob, s.fake);
}

// Max distance between the estimated and the true detail position over the base's 0.5 contour.
double regError(const MergeScene& m, const Registration& r) {
  const double cx = (m.w - 1) / 2.0, cy = (m.h - 1) / 2.0;
  double worst = 0;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      if (!mr::isContour(m.base.data(), m.w, m.h, x, y)) continue;
      double qx, qy;
      r.apply(x, y, &qx, &qy);
      worst = std::max(worst, std::hypot(qx - (cx + m.scale * (x - cx) + m.tx), qy - (cy + m.scale * (y - cy) + m.ty)));
    }
  return worst;
}

long outsideChanged(const Run& r, const std::vector<float>& base) {
  long c = 0;
  for (size_t i = 0; i < base.size(); ++i) c += r.zone[i] == 0.0f && r.out[i] != base[i];
  return c;
}

void testScene(bool busy) {
  const int w = 2880, h = 1620;
  const MergeScene m = mmsynth::makeMergeScene(w, h, busy);
  std::printf("%s %dx%d: true move t (%.2f, %.2f) scale %.3f, blob at (%.0f, %.0f)\n", m.name.c_str(), w, h, m.tx, m.ty,
              m.scale, m.blobX, m.blobY);
  const float* disp = m.rgbDisplay.data();
  const size_t n = size_t(w) * h;

  // Reach covering the strands (up to 240 design px long = 360 px here); the defaults further down.
  MergeParams add = defaults(h);
  add.reachPx = 380;
  print("base alone", score(m, m.base, m.base));
  std::vector<float> detailAsIs(m.detail);
  print("detail alone (as delivered)", score(m, m.base, detailAsIs));

  const Run a = run(m, m.base, disp, add);
  const double regErr = regError(m, a.info.reg);
  std::printf("  registration (%s): t (%.3f, %.3f) scale %.5f -> max error %.3f px; uncertainty %.3f px, agreement %.2f "
              "(%d of %d blocks), block rms %.3f px; matte start: sensitivity %.1f\n", alignModelName(a.info.align.model),
              a.info.reg.tx, a.info.reg.ty, a.info.reg.scale, regErr, a.info.align.uncertaintyPx, a.info.align.agreement,
              a.info.align.inliers, a.info.align.blocks, a.info.reg.rmsPx, a.info.align.matte.sensitivity);
  const Run al = run(m, m.base, m.rgbLog.data(), add, false);
  const double regErrL = regError(m, al.info.reg);
  std::printf("  registration, log picture (%s): max error %.3f px\n", alignModelName(al.info.align.model), regErrL);
  const Run au = run(m, m.baseUniform, disp, add, false);
  const double regErrU = regError(m, au.info.reg);
  std::printf("  registration, base with a uniform edge offset (%s): t (%.3f, %.3f) scale %.5f -> max error %.3f px\n",
              alignModelName(au.info.align.model), au.info.reg.tx, au.info.reg.ty, au.info.reg.scale, regErrU);
  // The mattes alone (no picture): the previous method, for comparison (now only the start and a fallback).
  const Run am = run(m, m.base, nullptr, add, false);
  std::printf("  registration on the mattes alone (%s): max error %.3f px\n", alignModelName(am.info.align.model),
              regError(m, am.info.reg));
  const Scores sa = score(m, m.base, a.out);
  print("Add, align, evidence off, reach 380", sa);
  const long outA = outsideChanged(a, m.base);
  std::printf("  zone %ld px, kept %ld, dropped %ld; changed outside the zone %ld\n", a.info.zonePixels, a.info.keptPixels,
              a.info.droppedPixels, outA);

  // Oracle: the detail moved back with the true move, no estimation.
  {
    Registration truth;
    truth.ok = true;
    truth.cx = (w - 1) / 2.0;
    truth.cy = (h - 1) / 2.0;
    truth.tx = m.tx;
    truth.ty = m.ty;
    truth.scale = m.scale;
    MergeScene o = m;
    warpDetail(m.detail.data(), w, h, truth, o.detail.data());
    MergeParams op = add;
    op.align = false;
    print("Add, oracle alignment", score(m, m.base, run(o, m.base, disp, op, false).out));
    for (double dw : {12.0, 16.0, 24.0}) {
      op.detailWidthPx = dw;
      char what[64];
      std::snprintf(what, sizeof what, "Add, oracle, Detail Width %.0f", dw);
      print(what, score(m, m.base, run(o, m.base, disp, op, false).out));
    }
    op.method = Method::ZoneBlend;
    op.detailWidthPx = 8;
    print("Zone blend, oracle alignment", score(m, m.base, run(o, m.base, disp, op, false).out));
  }
  MergeParams noAlign = add;
  noAlign.align = false;
  const Scores sn = score(m, m.base, run(m, m.base, disp, noAlign, false).out);
  print("Add, align OFF", sn);

  MergeParams ev = add;
  ev.evidence = true;
  const Run e = run(m, m.base, disp, ev);
  const Scores se = score(m, m.base, e.out);
  print("Add, align, evidence ON (display)", se);
  const Run el = run(m, m.base, m.rgbLog.data(), ev);
  const Scores sl = score(m, m.base, el.out);
  print("Add, align, evidence ON (log)", sl);
  // Decision agreement: pixels the detail proposes (added or rejected > 0.05 in either run) where the
  // two runs disagree on taking it (one adds > half of what the other adds, or vice versa).
  long cand = 0, agree = 0;
  double diff = 0;
  for (size_t i = 0; i < n; ++i) {
    const float pd = e.added[i] + e.rejected[i], pl = el.added[i] + el.rejected[i];
    if (std::max(pd, pl) <= 0.05f) continue;
    ++cand;
    const bool td = e.added[i] > 0.5f * pd, tl = el.added[i] > 0.5f * pl;
    agree += td == tl;
    diff += std::fabs(e.out[i] - el.out[i]);
  }
  const double agreement = cand ? double(agree) / cand : 0;
  // Fragmentation: the mean jump of the taken fraction added / (added + rejected) between 4-neighbours
  // that are both proposed (> 0.05). Per-pixel evidence chopped strands into dashes (a high jump).
  double jump = 0;
  long pairs = 0;
  auto taken = [&](size_t i) { return e.added[i] / (e.added[i] + e.rejected[i]); };
  for (int y = 0; y < h - 1; ++y)
    for (int x = 0; x < w - 1; ++x) {
      const size_t i = size_t(y) * w + x;
      if (e.added[i] + e.rejected[i] <= 0.05f) continue;
      for (size_t j : {i + 1, i + size_t(w)})
        if (e.added[j] + e.rejected[j] > 0.05f) jump += std::fabs(taken(i) - taken(j)), ++pairs;
    }
  const double fragmentation = pairs ? jump / pairs : 0;
  std::printf("  evidence fragmentation (mean jump of the taken fraction between neighbours): %.4f\n", fragmentation);
  std::printf("  evidence display vs log: %ld candidate px, decisions agree %.2f %%, mean |out diff| %.4f\n", cand,
              100 * agreement, cand ? diff / cand : 0);

  const Run d = run(m, m.base, disp, defaults(h));
  const Scores sd = score(m, m.base, d.out);
  print("Add, defaults (Reach 6 % = 97 px)", sd);
  MergeParams dOff = defaults(h);
  dOff.bridgePx = 0;
  const Scores sd0 = score(m, m.base, run(m, m.base, disp, dOff, false).out);
  print("Add, defaults, Bridge 0 (off)", sd0);
  MergeParams dEv = defaults(h);
  dEv.evidence = true;
  const Scores sdEv = score(m, m.base, run(m, m.base, disp, dEv, false).out);
  print("Add, defaults, evidence ON", sdEv);
  MergeParams dBlend = defaults(h);
  dBlend.method = Method::ZoneBlend;
  print("Zone blend, defaults", score(m, m.base, run(m, m.base, disp, dBlend, false).out));
  const long outD = outsideChanged(d, m.base);

  MergeParams blend = add;
  blend.method = Method::ZoneBlend;
  blend.reachPx = 380;
  const Run b = run(m, m.base, disp, blend);
  const Scores sb = score(m, m.base, b.out);
  print("Zone blend, align", sb);
  MergeParams blendEv = blend;
  blendEv.evidence = true;
  print("Zone blend, align, evidence ON", score(m, m.base, run(m, m.base, disp, blendEv, false).out));
  const long outB = outsideChanged(b, m.base);

  // Trimap: the unknown band filled; compared with the prior ramp (strength 0) and with no alignment.
  MergeParams tri = defaults(h);
  tri.source = Source::Trimap;
  const Run t = run(m, m.trimap, disp, tri);
  MergeParams tri0 = tri;
  tri0.strength = 0;
  const Run t0 = run(m, m.trimap, disp, tri0, false);
  MergeParams triNA = tri;
  triNA.align = false;
  const Run tn = run(m, m.trimap, disp, triNA, false);
  MergeParams triEv = tri;
  triEv.evidence = true;
  const Run te = run(m, m.trimap, disp, triEv, false);
  double eT = 0, e0 = 0, eN = 0, eE = 0, eS = 0;
  long nu = 0, knownChanged = 0;
  for (size_t i = 0; i < n; ++i) {
    if (m.trimap[i] != 0.5f) {
      knownChanged += t.out[i] != m.trimap[i];
      continue;
    }
    ++nu;
    eT += std::fabs(t.out[i] - m.truth[i]);
    e0 += std::fabs(t0.out[i] - m.truth[i]);
    eN += std::fabs(tn.out[i] - m.truth[i]);
    eE += std::fabs(te.out[i] - m.truth[i]);
    // the error of the unmoved detail (truth + thickened body + fakes + blob), for reference
    eS += std::fabs(std::max(m.truth[i], std::max(m.fake[i], m.blob[i])) - m.truth[i]);
  }
  std::printf("  trimap: registration delta %.2f, max error %.3f px; mean |out - truth| in the unknown band (%ld px): "
              "fill %.4f, evidence on %.4f, no align %.4f, prior ramp %.4f (an unblurred, aligned detail's error from its fakes and blob alone: %.4f); known px changed %ld\n",
              t.info.reg.delta, regError(m, t.info.reg), nu, eT / nu, eE / nu, eN / nu, e0 / nu, eS / nu, knownChanged);
  // In the fill, blob and fake strands are wrong wherever they show (truth is 0 there).
  double tb = 0, tbn = 0, tf = 0, tfn = 0;
  for (size_t i = 0; i < n; ++i) {
    if (m.trimap[i] != 0.5f) continue;
    if (m.blob[i] > 0.05f) tb += t.out[i], tbn += m.blob[i];
    if (m.fake[i] > 0.05f && m.strands[i] < 0.02f && m.body[i] < 0.05f && m.blob[i] < 0.01f)
      tf += t.out[i], tfn += m.fake[i];
  }
  double tfe = 0;
  for (size_t i = 0; i < n; ++i)
    if (m.trimap[i] == 0.5f && m.fake[i] > 0.05f && m.strands[i] < 0.02f && m.body[i] < 0.05f && m.blob[i] < 0.01f)
      tfe += te.out[i];
  std::printf("  trimap fill: blob shows %.4f of its alpha; fake strands show %.3f (evidence off) %.3f (on)\n",
              tbn > 0 ? tb / tbn : 0, tfn > 0 ? tf / tfn : 0, tfn > 0 ? tfe / tfn : 0);

  // ---- Limits. Regression limits from the first measurement (2026-10-24, BFD) with a margin; the
  // measured value is in the comment. Where a target from the brief is not met it says so.
  // Registration on the picture: the 0.5 px target (the mattes alone: 1.45 px, 0.49 trimap, 0.13 uniform).
  CHECK(a.info.align.model == AlignModel::PictureFull && al.info.align.model == AlignModel::PictureFull);
  CHECK(regErr < 0.25);              // 0.034 flat / 0.072 busy
  CHECK(regErrL < 0.25);             // log picture: 0.03 / 0.09
  CHECK(regErrU < 0.25);             // 0.013 / 0.165: base-independent now (was 0.13 on the mattes)
  CHECK(regError(m, t.info.reg) < 0.25);  // trimap 0.053 / 0.041 (was 0.49)
  CHECK(outA == 0 && outB == 0 && outD == 0 && knownChanged == 0);  // bit-identical outside the zone / known trimap
  CHECK(sd.bodyLowered == 0 && sd.blob < 0.01 && sdEv.blob < 0.01);  // defaults
  CHECK(sa.bodyLowered == 0 && se.bodyLowered == 0);   // Add never lowers the body edge
  // Bridge off: the body edge is the base's (0.0002-0.0003); the base sits 4.5 px inside the truth here.
  CHECK(sd0.bodyChange < 0.001 && sd0.bodyLowered == 0);
  CHECK(sd0.offsetPx < -4.0);                          // -4.29 where hair is (the base is 4.5 px inside the truth)
  // Bridge (defaults 0.5 % = 8.1 px here): the strand roots within 5 design px of the body come back
  // (0.26 -> 0.47, truth 0.68) and, where hair is, the body edge moves out towards the truth (-4.29 ->
  // -3.06 px: only within 2 x Bridge px of a strand, the strand mask); it may move out by at most Bridge
  // px and must not pass the truth by more than 1 px. On the hairless edge nothing changes (the halo seen
  // in Flame, 2026-10-24: before the strand mask max 0.98, mean 0.09 at 1080 defaults).
  CHECK(sd.root > sd0.root + 0.15 && sd.root < sd.rootTruth + 0.05);
  CHECK(sd.offsetPx - sd0.offsetPx < defaults(h).bridgePx && sd.offsetPx < 1.0 && sd.offsetPx > sd0.offsetPx + 0.5);
  CHECK(sd.haloMax == 0.0 && sd0.haloMax == 0.0 && sa.haloMax == 0.0 && se.haloMax == 0.0);
  CHECK(sa.offsetPx < 1.0 && se.offsetPx < 1.0);
  CHECK(sd.recovery > sd0.recovery);
  CHECK(sa.blob < 0.01 && se.blob < 0.01);             // 0.0000: the blob is dropped
  CHECK(sb.blob < 0.01);  // 0.0000: in a blend only connectivity drops it (Add's top-hat also removes a thick blob)
  CHECK(sa.recovery > 0.40);                           // 0.446
  CHECK(sa.recovery > sn.recovery + 0.1);              // 0.446 vs 0.280: alignment helps
  if (!busy) {
    // 0.166 vs 0.544 (Bridge off: 0.159 vs 0.536): the bridge takes fake strands' roots too (not judged)
    CHECK(se.fake < 0.32 * sa.fake);
    CHECK(se.recovery > 0.75 * sa.recovery);           // 0.379 vs 0.446
    CHECK(agreement > 0.98);                           // 99.6 %
  } else {
    CHECK(se.fake < 0.6 * sa.fake);                    // 0.276 vs 0.501: KNOWN WEAK on a busy background
    CHECK(se.recovery > 0.6 * sa.recovery);            // 0.322 vs 0.446
    CHECK(agreement > 0.95);                           // 96.7 %
  }
  CHECK(fragmentation < 0.06);  // 0.021 flat, 0.029 busy; 0.072 / 0.154 without the smoothing
  CHECK(eT / nu < 0.07 && eT < 0.7 * eN && eT < 0.25 * e0);  // mean 0.056 vs no align 0.101, prior 0.395
  CHECK(tbn > 0 && tb / tbn < 0.01);                   // 0.0000: the blob is not filled in
  (void)sl;
}

// Timing with the picture (always connected in the plugin) and a reused MergeWorkspace, as the
// plugin will run it; the first call of a fresh workspace and a call without one are given for
// comparison (frame-sized buffers page-faulted on first use).
void testTiming() {
  const int w = 2880, h = 1620;
  const MergeScene m = mmsynth::makeMergeScene(w, h, false);
  struct Cfg {
    const char* name;
    MergeParams p;
  };
  MergeParams add = defaults(h), addEv = add, addNA = add, blend = add, tri = add, far = add;
  addEv.evidence = true;
  addNA.align = false;
  blend.method = Method::ZoneBlend;
  tri.source = Source::Trimap;
  far.reachPx = 380;
  far.evidence = true;
  const Cfg cfgs[] = {{"Add, defaults, align", add},        {"Add, defaults, align, evidence", addEv},
                      {"Add, defaults, no align", addNA},   {"Zone blend, defaults, align", blend},
                      {"Trimap, defaults, align", tri},     {"Add, reach 380, align, evidence", far}};
  for (const Cfg& c : cfgs) {
    const std::vector<float>& base = c.p.source == Source::Trimap ? m.trimap : m.base;
    MergeWorkspace ws;
    const double first = run(m, base, m.rgbDisplay.data(), c.p, false, &ws).ms;
    double best = 1e9, none = 1e9;
    for (int rep = 0; rep < 3; ++rep) best = std::min(best, run(m, base, m.rgbDisplay.data(), c.p, false, &ws).ms);
    for (int rep = 0; rep < 2; ++rep) none = std::min(none, run(m, base, m.rgbDisplay.data(), c.p, false).ms);
    std::printf("timing 2880x1620 %-34s %6.1f ms reused workspace (best of 3; first call %.1f, no workspace %.1f), no views\n",
                c.name, best, first, none);
    CHECK(best < 150.0);  // 40-90 ms on BFD (16 threads); a loose bound against regressions
  }
  MergeWorkspace ws;
  double best = 1e9;
  for (int rep = 0; rep < 3; ++rep) best = std::min(best, run(m, m.base, m.rgbDisplay.data(), far, true, &ws).ms);
  std::printf("timing 2880x1620 %-34s %6.1f ms reused workspace (best of 3), all views\n", "Add, reach 380, align, evidence", best);
}

}  // namespace

int main(int argc, char** argv) {
  const bool timingOnly = argc > 1 && std::string(argv[1]) == "--timing";
  if (!timingOnly) {
    testScene(false);
    testScene(true);
  }
  testTiming();
  std::printf(failures ? "FAILED (%d failure(s))\n" : "OK (0 failure(s))\n", failures);
  return failures ? 1 : 0;
}
