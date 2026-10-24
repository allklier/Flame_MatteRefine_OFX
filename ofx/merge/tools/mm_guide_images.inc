// The Matte Merge images of USERGUIDE.md section 6 (docs/images/mm_*.png), included inside the anonymous
// namespace of ofx/tools/mr_guide_images.cpp when it is built with MR_BUILD_MERGE=ON (CMake links
// mr_merge_core and mr_merge_synth). Scenes: merge_synth.h makeMergeScene at 1920x1080 (the simulated
// detail matte is blurred, shifted and padded with invented strands and a blob), the picture is the
// scene's display-encoded RGB, the core is mr::merge::merge() itself. Settings, unless stated: the
// HeightSizes defaults (Reach 6 %, Detail Width 0.5 %, Feather 0.25 %, Bridge 0.5 % of the height),
// Add Fine Detail, Matte Edge, Align Auto, Evidence Off, Strength 1.
namespace mmimg {

struct MRun {
  Buf out, added, rejected, zone, residual;
  mr::merge::MergeInfo info;
};

MRun mergeRun(const mmsynth::MergeScene& m, const Buf& base, const mr::merge::MergeParams& p) {
  MRun r;
  const size_t n = size_t(m.w) * m.h;
  r.out.resize(n), r.added.resize(n), r.rejected.resize(n), r.zone.resize(n), r.residual.resize(n);
  mr::merge::MergeViews v{r.added.data(), r.rejected.data(), r.zone.data(), r.residual.data()};
  mr::merge::merge(base.data(), m.detail.data(), m.rgbDisplay.data(), m.w, m.h, p, r.out.data(), v, &r.info);
  return r;
}

mr::merge::MergeParams defaults() {
  mr::merge::MergeParams p;
  mr::merge::scaleToHeight(p, mr::merge::HeightSizes(), kH);
  return p;
}

// Mean over the real strands outside the body: share of the truth's strand alpha above the base that
// the result brings back; and the share of the invented strands' alpha (away from real ones) it adds.
struct Share {
  double real = 0, fake = 0;
};
Share shares(const mmsynth::MergeScene& m, const Buf& base, const Buf& out) {
  double got = 0, want = 0, fa = 0, ft = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (m.strands[i] > 0.05f && m.body[i] < 0.5f) {
      want += std::max(0.0f, m.truth[i] - base[i]);
      got += std::min(std::max(0.0f, out[i] - base[i]), std::max(0.0f, m.truth[i] - base[i]));
    }
    if (m.fake[i] > 0.05f && m.strands[i] < 0.01f) ft += m.fake[i], fa += std::max(0.0f, out[i] - base[i]);
  }
  return {want > 0 ? got / want : 0.0, ft > 0 ? fa / ft : 0.0};
}

// The crop: the head's hair line (the strands root on the head's top and sides).
const View kHair{1000, 380, 480, 270, 2, 1};

Img mpanel(const Buf& m, const View& v = kHair) { return panel(v, grey(m)); }

// Hair images carry no red true-edge line: the truth's 0.5 contour of 1-2 px strands is a tangle that
// hides them. The truth is shown as its own panel instead.
const View kRoots{1090, 500, 160, 90, 6, 1};

double maxOf(const Buf& b) { return *std::max_element(b.begin(), b.end()); }

void all() {
  const mmsynth::MergeScene s = mmsynth::makeMergeScene(kW, kH, false);
  const mmsynth::MergeScene busy = mmsynth::makeMergeScene(kW, kH, true);
  const mr::merge::MergeParams d = defaults();
  char b[400];

  // 1. The inputs.
  save("mm_inputs.png", row({mpanel(s.base), mpanel(s.detail), mpanel(s.truth)}),
       "scene hair_flat, crop 480x270 at 2x: base matte | detail matte | truth");

  // 2. The result.
  mr::merge::MergeParams p = d;
  p.strength = 1.3;
  const MRun r = mergeRun(s, s.base, p);
  const Share sh = shares(s, s.base, r.out);
  std::snprintf(b, sizeof b,
                "defaults, Strength 1.3: base | result | truth; strand alpha brought back %.0f%%, invented strand alpha added %.0f%%, "
                "align %s, uncertainty %.2f px",
                100 * sh.real, 100 * sh.fake, mr::merge::alignModelName(r.info.align.model), r.info.align.uncertaintyPx);
  save("mm_result.png", row({mpanel(s.base), mpanel(r.out), mpanel(s.truth)}), b);

  // 3. Bridge.
  {
    mr::merge::MergeParams q = d;
    q.detailWidthPx = 0.05 * kH;  // 5 %
    q.strength = 2.0;
    mr::merge::MergeParams q0 = q;
    q0.bridgePx = 0;
    const MRun a = mergeRun(s, s.base, q0), c = mergeRun(s, s.base, q);
    std::snprintf(b, sizeof b, "crop 160x90 at 6x on the hair roots; Detail Width 5 %%, Strength 2, Reach 6 %%: Bridge 0 | Bridge 0.5 %% | truth");
    save("mm_bridge.png", row({mpanel(a.out, kRoots), mpanel(c.out, kRoots), mpanel(s.truth, kRoots)}), b);
  }

  // 4. Strength.
  {
    std::vector<Img> ps;
    std::string t;
    for (double st : {1.0, 1.5, 2.0}) {
      mr::merge::MergeParams q = d;
      q.strength = st;
      const MRun x = mergeRun(s, s.base, q);
      const Share h = shares(s, s.base, x.out);
      ps.push_back(mpanel(x.out));
      char t1[100];
      std::snprintf(t1, sizeof t1, " | %.1f: real %.0f%%, invented %.0f%%", st, 100 * h.real, 100 * h.fake);
      t += t1;
    }
    save("mm_strength.png", row(ps), "defaults, Strength 1.0 | 1.5 | 2.0" + t);
  }

  // 5. The views.
  {
    Buf res = r.residual;
    const double rm = maxOf(res);
    for (float& v : res) v = float(std::min(1.0, v * 4.0));
    Buf rej = r.rejected;
    const double jm = maxOf(rej);
    for (float& v : rej) v = float(std::min(1.0, v * 4.0));
    std::snprintf(b, sizeof b, "defaults, Strength 1.3: Added | Rejected (x4, max %.3f) / Zone | Residual (x4, max %.3f)", jm, rm);
    save("mm_views.png",
         column({row({mpanel(r.added), mpanel(rej)}), row({mpanel(r.zone), mpanel(res)})}), b);
  }

  // 6. Trimap.
  {
    mr::merge::MergeParams q = d;
    q.source = mr::merge::Source::Trimap;
    const MRun t = mergeRun(s, s.trimap, q);
    const View& v = kHair;
    save("mm_trimap.png", row({panel(v, trimapView(s.trimap)), mpanel(t.out), mpanel(s.truth)}),
         "Source = Trimap, defaults (Keep Disconnected off): trimap | filled | truth");
  }

  // 7. Evidence.
  {
    mr::merge::MergeParams q = d;
    q.strength = 1.3;
    mr::merge::MergeParams e = q;
    e.evidence = true;
    const MRun a = mergeRun(busy, busy.base, q), c = mergeRun(busy, busy.base, e);
    const Share sa = shares(busy, busy.base, a.out), sc = shares(busy, busy.base, c.out);
    std::snprintf(b, sizeof b,
                  "hair_busy, defaults, Strength 1.3: Evidence Off | On | truth (red = invented strand); real %.0f%% -> %.0f%%, invented %.0f%% -> %.0f%%",
                  100 * sa.real, 100 * sc.real, 100 * sa.fake, 100 * sc.fake);
    // The invented strands (known from the scene) are tinted red where the result has them.
    auto marked = [&](const Buf& m) {
      return panel(kHair, Shader([&m, &busy](int x, int y, float* c) {
                     const size_t i = size_t(y) * kW + x;
                     const float f = busy.fake[i] > 0.05f && busy.strands[i] < 0.01f ? 1.0f : 0.0f;
                     c[0] = m[i];
                     c[1] = c[2] = m[i] * (1.0f - f);
                   }));
    };
    save("mm_evidence.png", row({marked(a.out), marked(c.out), mpanel(busy.truth)}), b + std::string("; invented strands tinted red"));
  }
}

}  // namespace mmimg
