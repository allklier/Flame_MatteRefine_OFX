// Temporal Stabilise (B4) evaluation on synth::makeSequence (not a test): flicker and bandError
// before / after, an interior error (the flickering hole), flow endpoint errors against the true
// motion, and the time per frame. Options:
//   --size WxH (1920x1080)  --seeds A-B (1-10)  --frames N (9)  -K k (1)  --strength s (1)
//   --band px (6 at 1080, scaled)  --mean  --plain (unweighted flow)  --matte-gain g  --no-rgb
//   --flow-lines n (600)  --static-bg  --flat-subject  --fast f (motion x f: every f-th frame)
//   --radius r  --lambda l  --iters i  --levels n  --fbmax e  --no-holes  --quiet
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "synth.h"
#include "temporal.h"

using namespace mr::ml;

namespace {
struct Epe {
  double sum = 0;
  long long n = 0;
  void add(double e) { sum += e, ++n; }
  double mean() const { return n ? sum / n : 0; }
};
}  // namespace

int main(int argc, char** argv) {
  int w = 1920, h = 1080, s0 = 1, s1 = 10, frames = 9, fast = 1;
  bool staticBg = false, flat = false, noRgb = false, quiet = false;
  double bandDesign = -1;
  StabiliseParams p;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--size") std::sscanf(next().c_str(), "%dx%d", &w, &h);
    else if (a == "--seeds") std::sscanf(next().c_str(), "%d-%d", &s0, &s1);
    else if (a == "--frames") frames = std::atoi(next().c_str());
    else if (a == "-K") p.frames = std::atoi(next().c_str());
    else if (a == "--strength") p.strength = std::atof(next().c_str());
    else if (a == "--band") bandDesign = std::atof(next().c_str());
    else if (a == "--mean") p.combine = Combine::Mean;
    else if (a == "--damping") p.flow.damping = float(std::atof(next().c_str()));
    else if (a == "--plain") p.subjectWeighted = false;
    else if (a == "--matte-gain") p.flow.matteGain = float(std::atof(next().c_str()));
    else if (a == "--no-rgb") noRgb = true;
    else if (a == "--flow-lines") p.flowLines = std::atoi(next().c_str());
    else if (a == "--static-bg") staticBg = true;
    else if (a == "--flat-subject") flat = true;
    else if (a == "--fast") fast = std::atoi(next().c_str());
    else if (a == "--radius") p.flow.radius = std::atoi(next().c_str());
    else if (a == "--lambda") p.flow.lambda = float(std::atof(next().c_str()));
    else if (a == "--iters") p.flow.iterations = std::atoi(next().c_str());
    else if (a == "--levels") p.flow.levels = std::atoi(next().c_str());
    else if (a == "--fbmax") p.fbMaxPx = std::atof(next().c_str());
    else if (a == "--no-holes") p.fillHoles = false;
    else if (a == "--quiet") quiet = true;
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  const double k = h / 1080.0;
  p.bandPx = (bandDesign > 0 ? bandDesign : 6.0) * k;
  const double scoreBand = 3.0 * k;
  double fl0 = 0, fl1 = 0, flT = 0, be0 = 0, be1 = 0, ie0 = 0, ie1 = 0, ms = 0, flowMs = 0, rej = 0;
  Epe eSub, eBand, eBg;
  int runs = 0, holeOnFrames = 0, holeFixed = 0;
  for (int seed = s0; seed <= s1; ++seed) {
    synth::Sequence full = synth::makeSequence(w, h, (frames - 1) * fast + 1, unsigned(seed), !staticBg);
    synth::Sequence s;  // every fast-th frame: the motion x fast
    s.w = w;
    s.h = h;
    for (int t = 0; t < frames; ++t) {
      const int f = t * fast;
      s.frames.push_back(std::move(full.frames[f]));
      s.edgeDistance.push_back(std::move(full.edgeDistance[f]));
      s.subject.push_back(full.subject[f]);
      s.background.push_back(full.background[f]);
      s.holeOn.push_back(full.holeOn[f]);
    }
    const size_t n = size_t(w) * h;
    if (flat)  // the subject without texture: rgb + truth (flat - fg), exact for the composite
      for (auto& fr : s.frames)
        for (size_t i = 0; i < n; ++i) {
          const float fc[3] = {0.6f, 0.38f, 0.26f};
          for (int c = 0; c < 3; ++c) fr.rgb[i * 3 + c] += fr.truth[i] * (fc[c] - fr.fg[i * 3 + c]);
        }
    std::vector<TemporalFrame> tf(frames);
    std::vector<std::vector<float>> in(frames), out(frames, std::vector<float>(n)), truth(frames);
    for (int t = 0; t < frames; ++t) {
      tf[t].rgb = noRgb ? nullptr : s.frames[t].rgb.data();
      tf[t].matte = s.frames[t].matte.data();
      in[t] = s.frames[t].matte;
      truth[t] = s.frames[t].truth;
    }
    for (int t = 0; t < frames; ++t) {
      const int a = std::max(0, t - p.frames), b = std::min(frames - 1, t + p.frames);
      StabiliseStats st;
      stabilise(&tf[a], b - a + 1, t - a, w, h, p, out[t].data(), &st);
      ms += st.totalMs;
      flowMs += st.flowMs;
      rej += st.rejectedFraction;
      ++runs;
    }
    // Flow endpoint error, centre frame frames / 2 to its +-1 neighbours.
    const int c = frames / 2;
    for (int o : {-1, 1}) {
      const NeighbourFlow nf = flowToNeighbour(tf[c], tf[c + o], w, h, p);
      const synth::Affine sub = synth::between(s.subject[c], s.subject[c + o]);
      const synth::Affine bg = synth::between(s.background[c], s.background[c + o]);
      for (int y = 0; y < h; y += 2)
        for (int x = 0; x < w; x += 2) {
          const size_t i = size_t(y) * w + x;
          const double d = s.edgeDistance[c][i];
          double fx, fy, sx, sy, bx, by;
          flowAt(nf, x, y, &fx, &fy);
          sub.apply(x, y, &sx, &sy);
          bg.apply(x, y, &bx, &by);
          if (std::fabs(d) <= p.bandPx) eBand.add(std::hypot(fx - (sx - x), fy - (sy - y)));
          else if (d > p.bandPx) eSub.add(std::hypot(fx - (sx - x), fy - (sy - y)));
          else if (d < -3 * p.bandPx) eBg.add(std::hypot(fx - (bx - x), fy - (by - y)));
        }
    }
    const double f0 = synth::flicker(s, in, scoreBand), f1 = synth::flicker(s, out, scoreBand);
    const double b0 = synth::bandError(s, in, scoreBand), b1 = synth::bandError(s, out, scoreBand);
    // Interior error: mean |m - truth| deeper inside than the band (the flickering hole).
    double i0 = 0, i1 = 0;
    long long ni = 0;
    for (int t = 0; t < frames; ++t)
      for (size_t i = 0; i < n; ++i)
        if (s.edgeDistance[t][i] > p.bandPx) {
          i0 += std::fabs(in[t][i] - truth[t][i]);
          i1 += std::fabs(out[t][i] - truth[t][i]);
          ++ni;
        }
    i0 /= ni;
    i1 /= ni;
    // The flickering hole: frames where it is on, filled = >= 90 % of its pixels (input < 0.5,
    // truth > 0.99, deeper than the band) now >= 0.5.
    for (int t = 0; t < frames; ++t) {
      if (!s.holeOn[t]) continue;
      long long hp = 0, hf = 0;
      for (size_t i = 0; i < n; ++i)
        if (s.edgeDistance[t][i] > p.bandPx && in[t][i] < 0.5f && truth[t][i] > 0.99f) ++hp, hf += out[t][i] >= 0.5f;
      if (hp) ++holeOnFrames, holeFixed += hf >= 0.9 * hp;
    }
    const double ft = synth::flicker(s, truth, scoreBand);
    if (!quiet)
      std::printf("seed %2d  flicker %.4f -> %.4f (truth %.4f)  bandErr %.4f -> %.4f  interior %.5f -> %.5f\n",
                  seed, f0, f1, ft, b0, b1, i0, i1);
    fl0 += f0, fl1 += f1, flT += ft, be0 += b0, be1 += b1, ie0 += i0, ie1 += i1;
  }
  const int ns = s1 - s0 + 1;
  std::printf("MEAN %dx%d K=%d str=%.2f band=%.1f %s%s%s%s%s%s fast=%d: flicker %.4f -> %.4f (truth %.4f, %+.0f%%)  "
              "bandErr %.4f -> %.4f (%+.0f%%)  interior %.5f -> %.5f  EPE subj %.2f band %.2f bg %.2f px  "
              "rejected %.1f%%  torso hole filled in %d of %d frames  %.1f ms/frame (flow %.1f)\n",
              w, h, p.frames, p.strength, p.bandPx, p.combine == Combine::Mean ? "mean" : "median",
              p.subjectWeighted ? "" : " plain", noRgb ? " no-rgb" : "", p.flow.matteGain > 0 ? " matte-ch" : "",
              staticBg ? " static-bg" : "", flat ? " flat" : "", fast, fl0 / ns, fl1 / ns, flT / ns,
              100 * (fl1 - fl0) / fl0, be0 / ns, be1 / ns, 100 * (be1 - be0) / be0, ie0 / ns, ie1 / ns,
              eSub.mean(), eBand.mean(), eBg.mean(), 100 * rej / runs, holeFixed, holeOnFrames, ms / runs,
              flowMs / runs);
  return 0;
}
