// The synthetic scenes (ofx/tools/synth): sizes, ranges, each defect where it belongs, the ML
// scenes' truth, labels and motion, and the EXR writer's layout.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "synth.h"

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

void testScenes() {
  const int w = 480, h = 270;
  for (const auto& name : synth::sceneNames()) {
    const synth::Scene s = synth::make(name, w, h);
    CHECK(s.matte.size() == size_t(w) * h && s.truth.size() == s.matte.size());
    CHECK(s.rgb.size() == s.matte.size() * 3 && s.fg.size() == s.rgb.size());
    int outOfRange = 0, higher = 0, lower = 0;
    for (size_t i = 0; i < s.matte.size(); ++i) {
      outOfRange += s.matte[i] < 0 || s.matte[i] > 1 || s.truth[i] < 0 || s.truth[i] > 1;
      higher += s.matte[i] > s.truth[i] + 1e-6f;
      lower += s.matte[i] < s.truth[i] - 1e-6f;
    }
    std::printf("%-10s higher %6d lower %6d\n", name.c_str(), higher, lower);
    CHECK(outOfRange == 0);
    if (name == "holes" || name == "eroded") CHECK(higher == 0 && lower > 100);
    if (name == "specks") CHECK(lower == 0 && higher > 1000);
    if (name == "fgcolor") CHECK(higher == 0 && lower == 0);
    if (name != "fgcolor") CHECK(higher + lower > 100);
    // The real gap inside the arm on the hip (design point (-245, -20)) is background.
    const int gx = int(w / 2.0 - 245 * h / 1080.0), gy = int(h / 2.0 - 20 * h / 1080.0);
    CHECK(s.truth[size_t(gy) * w + gx] == 0.0f);
  }
  // Deterministic.
  CHECK(synth::make("all", 64, 36).matte == synth::make("all", 64, 36).matte);
}

// ---- The ML scenes (synth.h "Scenes for Matte Refine ML").

// Fractions of each label, in %.
void labelFractions(const synth::Scene& s, double out[4]) {
  long long c[4] = {0, 0, 0, 0};
  for (uint8_t l : s.label) ++c[l];
  for (int j = 0; j < 4; ++j) out[j] = 100.0 * c[j] / s.label.size();
}

int labelAt(const synth::Scene& s, double x, double y) {  // design units
  const double k = s.h / 1080.0;
  return s.label[size_t(std::lround(s.h / 2.0 + y * k)) * s.w + size_t(std::lround(s.w / 2.0 + x * k))];
}

// Enclosed 4-connected components of matte < 0.5 (as Fill Holes sees them): area and labels.
struct Component {
  long long area = 0;
  int labels = 0;  // bit mask of the labels in it
};
std::vector<Component> enclosedComponents(const synth::Scene& s) {
  const int w = s.w, h = s.h;
  std::vector<int> id(s.matte.size(), -1);
  std::vector<Component> out;
  std::vector<int> stack;
  for (size_t start = 0; start < s.matte.size(); ++start) {
    if (s.matte[start] >= 0.5f || id[start] >= 0) continue;
    Component c;
    bool border = false;
    id[start] = int(out.size());
    stack.push_back(int(start));
    while (!stack.empty()) {
      const int i = stack.back(), x = i % w, y = i / w;
      stack.pop_back();
      ++c.area;
      c.labels |= 1 << s.label[i];
      border = border || x == 0 || y == 0 || x == w - 1 || y == h - 1;
      const int nb[4][2] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
      for (const auto& q : nb) {
        if (q[0] < 0 || q[1] < 0 || q[0] >= w || q[1] >= h) continue;
        const size_t j = size_t(q[1]) * w + q[0];
        if (s.matte[j] < 0.5f && id[j] < 0) {
          id[j] = int(out.size());
          stack.push_back(int(j));
        }
      }
    }
    out.push_back(border ? Component{} : c);  // area 0 = the outside
  }
  return out;
}

void testGaps() {
  // Known answers at HD, seed 1 (measured when written; the bounds leave room for rounding).
  const synth::Scene s = synth::makeGaps(1920, 1080, 1);
  const size_t n = s.matte.size();
  CHECK(s.label.size() == n && s.rgb.size() == 3 * n && s.fg.size() == 3 * n);
  CHECK(s.name == "gaps" && synth::make("gaps", 64, 36).label == synth::makeGaps(64, 36).label);
  double f[4];
  labelFractions(s, f);
  std::printf("gaps 1920x1080 labels %%: bg %.3f subject %.3f defect %.3f real gap %.3f\n", f[0], f[1], f[2], f[3]);
  CHECK(f[0] > 83.5 && f[0] < 85.5);
  CHECK(f[1] > 11.5 && f[1] < 13.5);
  CHECK(f[2] > 1.0 && f[2] < 1.35);
  CHECK(f[3] > 1.55 && f[3] < 1.75);
  // The five designed real gaps (centres in design units) and two subject points.
  CHECK(labelAt(s, -245, -20) == synth::kRealGap);  // arm on hip
  CHECK(labelAt(s, 0, 250) == synth::kRealGap);     // between the legs
  CHECK(labelAt(s, 262, 330) == synth::kRealGap);   // ring in the hand
  CHECK(labelAt(s, 262, 402) == synth::kRealGap);   // small linked ring
  CHECK(labelAt(s, 70, -265) == synth::kRealGap);   // earring
  CHECK(labelAt(s, -900, -500) == synth::kBackground && labelAt(s, 0, 470) == synth::kSubject);

  // Per-pixel consistency of truth, matte, label and picture.
  int bad = 0, touching = 0;
  double red2 = 0, red3 = 0;
  long long n2 = 0, n3 = 0;
  for (size_t i = 0; i < n; ++i) {
    const int l = s.label[i];
    const float m = s.matte[i], t = s.truth[i];
    bad += m < 0 || m > 1 || m > t + 1e-6f;  // the defects only lower the matte
    bad += (l == synth::kBackground || l == synth::kRealGap) != (t < 0.5f);
    bad += (l == synth::kBackground || l == synth::kRealGap) && m != t;
    bad += l == synth::kSubject && m < 0.5f;
    bad += l == synth::kDefectHole && (t != 1.0f || m > 0.71f);
    bad += t == 1.0f && (s.rgb[3 * i] != s.fg[3 * i] || s.rgb[3 * i + 2] != s.fg[3 * i + 2]);
    if (l == synth::kRealGap) {
      const int x = int(i % 1920), y = int(i / 1920);
      touching += s.label[size_t(y) * 1920 + x - 1] == 0 || s.label[size_t(y) * 1920 + x + 1] == 0 ||
                  s.label[i - 1920] == 0 || s.label[i + 1920] == 0;
      red3 += s.rgb[3 * i];
      ++n3;
    }
    if (l == synth::kDefectHole) {
      red2 += s.rgb[3 * i];
      ++n2;
    }
  }
  CHECK(bad == 0 && touching == 0);
  // The picture tells them apart: a defect shows the subject (red), a real gap the background.
  std::printf("gaps mean red: defect %.3f real gap %.3f\n", red2 / n2, red3 / n3);
  CHECK(red2 / n2 > 0.6 && red3 / n3 < 0.25);
  // The wash: subject pixels below the truth but above 0.5.
  long long washed = 0;
  for (size_t i = 0; i < n; ++i) washed += s.label[i] == synth::kSubject && s.matte[i] < 0.9f && s.truth[i] == 1.0f;
  CHECK(washed > 3000);

  // Holes as Fill Holes sees them: each is purely a defect or a real gap; 10 defects, 5 gaps; the
  // two size ranges overlap (no size threshold separates them).
  long long dMin = 1LL << 60, dMax = 0, gMin = 1LL << 60, gMax = 0;
  int defects = 0, gaps = 0, mixed = 0;
  for (const Component& c : enclosedComponents(s)) {
    if (!c.area) continue;
    if (c.labels == 1 << synth::kDefectHole) {
      ++defects;
      dMin = std::min(dMin, c.area);
      dMax = std::max(dMax, c.area);
    } else if (c.labels == 1 << synth::kRealGap) {
      ++gaps;
      gMin = std::min(gMin, c.area);
      gMax = std::max(gMax, c.area);
      std::printf("  real gap %lld px\n", c.area);
    } else {
      ++mixed;
    }
  }
  std::printf("gaps components: %d defects (%lld..%lld px), %d real gaps (%lld..%lld px), %d mixed\n", defects, dMin,
              dMax, gaps, gMin, gMax, mixed);
  CHECK(defects == 10 && gaps == 5 && mixed == 0);
  CHECK(gMin < dMax / 4 && dMax > 2 * gMin && dMin < gMin && gMax > dMax);
  int gapsInside = 0;  // real gaps within the defect size range
  for (const Component& c : enclosedComponents(s))
    gapsInside += c.labels == 1 << synth::kRealGap && c.area >= dMin && c.area <= dMax;
  CHECK(gapsInside >= 3);

  // Size independence: the same fractions at a small size.
  double g[4];
  labelFractions(synth::makeGaps(480, 270, 1), g);
  std::printf("gaps 480x270 labels %%: %.3f %.3f %.3f %.3f\n", g[0], g[1], g[2], g[3]);
  for (int j = 0; j < 4; ++j) CHECK(std::fabs(g[j] - f[j]) < 0.1 * f[j] + 0.05);

  // Determinism; other seeds move the defects (truth and real gaps stay), and always fit.
  const synth::Scene a = synth::makeGaps(480, 270, 3), b = synth::makeGaps(480, 270, 3);
  CHECK(a.matte == b.matte && a.label == b.label && a.rgb == b.rgb);
  for (unsigned seed = 1; seed <= 20; ++seed) {
    const synth::Scene o = synth::makeGaps(480, 270, seed);
    labelFractions(o, g);
    CHECK(o.truth == a.truth && g[2] > 0.9 && g[2] < 1.4 && g[3] > 1.5 && g[3] < 1.8);
    if (seed != 3) CHECK(o.matte != a.matte && o.label != a.label);
  }
}

void testHair() {
  const int w = 1920, h = 1080;
  const synth::Scene flat = synth::makeHair(w, h, false, 1), busy = synth::makeHair(w, h, true, 1);
  const size_t n = size_t(w) * h;
  CHECK(flat.name == "hair_flat" && busy.name == "hair_busy" && flat.label.empty());
  CHECK(synth::make("hair_busy", 96, 54).rgb == synth::makeHair(96, 54, true).rgb);
  // The background changes only the picture.
  CHECK(flat.truth == busy.truth && flat.matte == busy.matte && flat.fg == busy.fg && flat.rgb != busy.rgb);
  int outOfRange = 0;
  for (size_t i = 0; i < n; ++i)
    outOfRange += flat.truth[i] < 0 || flat.truth[i] > 1 || flat.matte[i] < 0 || flat.matte[i] > 1;
  CHECK(outOfRange == 0);
  // Hair the matte misses: truth mass where the matte is 0, partial (thin / transparent) pixels.
  auto lost = [](const synth::Scene& s, long long* partial) {
    double sum = 0;
    *partial = 0;
    for (size_t i = 0; i < s.truth.size(); ++i)
      if (s.matte[i] == 0.0f) {
        sum += s.truth[i];
        *partial += s.truth[i] > 0.02f && s.truth[i] < 0.98f;
      }
    return 100.0 * sum / s.truth.size();
  };
  long long partial = 0;
  const double lostPct = lost(flat, &partial);
  // Strands are thin: horizontal runs of truth > 0.1 where the matte is 0, median width.
  std::vector<int> runs;
  for (int y = 0; y < h; ++y)
    for (int x = 0, run = 0; x <= w; ++x) {
      const bool on = x < w && flat.matte[size_t(y) * w + x] == 0.0f && flat.truth[size_t(y) * w + x] > 0.1f;
      if (on) ++run;
      else if (run) runs.push_back(run), run = 0;
    }
  std::sort(runs.begin(), runs.end());
  const int median = runs.empty() ? 0 : runs[runs.size() / 2];
  // Body: solid in truth and matte at the bottom centre; matte inside the truth's body (eroded).
  const size_t bottom = size_t(h - 5) * w + w / 2;
  long long matteOutside = 0;
  for (size_t i = 0; i < n; ++i) matteOutside += flat.matte[i] > 0.5f && flat.truth[i] < 0.5f;
  std::printf("hair: lost %.3f %% of the frame's alpha, %lld partial px, median run %d px, %zu runs, matte "
              "outside %lld px\n",
              lostPct, partial, median, runs.size(), matteOutside);
  CHECK(lostPct > 0.8 && lostPct < 2.0 && partial > 20000);
  CHECK(median >= 1 && median <= 4 && runs.size() > 10000);
  CHECK(flat.truth[bottom] == 1.0f && flat.matte[bottom] == 1.0f && matteOutside < 2000);
  // Backgrounds: flat is one colour; busy has strong pixel-to-pixel detail.
  double dFlat = 0, dBusy = 0;
  long long cnt = 0;
  for (size_t i = 0; i + 1 < n; ++i)
    if (flat.truth[i] == 0.0f && flat.truth[i + 1] == 0.0f) {
      dFlat += std::fabs(flat.rgb[3 * i] - flat.rgb[3 * i + 3]);
      dBusy += std::fabs(busy.rgb[3 * i] - busy.rgb[3 * i + 3]);
      ++cnt;
    }
  std::printf("hair: background mean |dR/dx| flat %.4f busy %.4f\n", dFlat / cnt, dBusy / cnt);
  CHECK(dFlat == 0.0 && dBusy / cnt > 0.03);
  // Size independence and seeds.
  const synth::Scene small = synth::makeHair(480, 270, false, 1);
  const double lostSmall = lost(small, &partial);
  std::printf("hair 480x270: lost %.3f %%\n", lostSmall);
  CHECK(std::fabs(lostSmall - lostPct) < 0.15 * lostPct);
  CHECK(synth::makeHair(480, 270, false, 1).truth == small.truth);
  const synth::Scene other = synth::makeHair(480, 270, false, 2);
  CHECK(other.truth != small.truth && std::fabs(lost(other, &partial) - lostSmall) < 0.3 * lostSmall);
}

void testSequence() {
  const int w = 960, h = 540;
  const synth::Sequence s = synth::makeSequence(w, h, 9, 1, true);
  CHECK(s.frames.size() == 9 && s.edgeDistance.size() == 9 && s.subject.size() == 9 && s.background.size() == 9);
  const synth::Affine& id = s.subject[0];
  CHECK(id.a == 1 && id.b == 0 && id.c == 0 && id.d == 1 && id.tx == 0 && id.ty == 0);
  // Motion: a rotation + translation, 0.3-0.5 deg and (5-7, -1.5..-0.5) px at 1080 per frame.
  const double k = h / 1080.0;
  int bad = 0, on = 0;
  for (int t = 1; t < 9; ++t) {
    const synth::Affine &m = s.subject[t], &p = s.subject[t - 1];
    bad += std::fabs(m.a * m.d - m.b * m.c - 1) > 1e-12 || m.a != m.d || m.b != -m.c;
    const double step = (std::atan2(m.c, m.a) - std::atan2(p.c, p.a)) * 180 / M_PI;
    double cx, cy, px, py;
    m.apply(w / 2.0, h / 2.0, &cx, &cy);
    p.apply(w / 2.0, h / 2.0, &px, &py);
    bad += step < 0.3 || step > 0.5 || (cx - px) / k < 5 || (cx - px) / k > 7 || (cy - py) / k < -1.5 ||
           (cy - py) / k > -0.5;
    bad += std::fabs(s.background[t].tx + 2.5 * k * t) > 1e-12 || std::fabs(s.background[t].ty - k * t) > 1e-12;
    // between() maps frame t-1 to t: a point and back.
    double x, y, bx, by;
    synth::between(p, m).apply(100, 200, &x, &y);
    synth::between(m, p).apply(x, y, &bx, &by);
    bad += std::fabs(bx - 100) > 1e-9 || std::fabs(by - 200) > 1e-9;
    double ox, oy, qx, qy;
    m.apply(37, 41, &ox, &oy);
    m.inverse().apply(ox, oy, &qx, &qy);
    bad += std::fabs(qx - 37) > 1e-9 || std::fabs(qy - 41) > 1e-9;
  }
  CHECK(bad == 0);
  // Per frame: truth agrees with the edge distance; the hole (design (40, -60), moving with the
  // subject) is in the matte exactly on its frames.
  for (int t = 0; t < 9; ++t) {
    const synth::Scene& f = s.frames[t];
    int wrong = 0;
    for (size_t i = 0; i < f.truth.size(); ++i) wrong += (f.truth[i] > 0.5f) != (s.edgeDistance[t][i] > 0);
    double hx, hy;
    s.subject[t].apply(w / 2.0 + 40 * k, h / 2.0 - 60 * k, &hx, &hy);
    const size_t hc = size_t(std::lround(hy)) * w + size_t(std::lround(hx));
    wrong += f.truth[hc] != 1.0f || (s.holeOn[t] ? f.matte[hc] > 0.4f : f.matte[hc] != 1.0f);
    on += s.holeOn[t];
    CHECK(wrong == 0);
  }
  CHECK(on > 0 && on < 9);
  // The scores: the truth is steady (not 0: bilinear resampling of a 1 px edge at this size) and
  // exact, the input flickers and is off in the band.
  std::vector<std::vector<float>> in, tr;
  for (const auto& f : s.frames) in.push_back(f.matte), tr.push_back(f.truth);
  const double fIn = synth::flicker(s, in, 3), fTr = synth::flicker(s, tr, 3);
  const double eIn = synth::bandError(s, in, 3), eTr = synth::bandError(s, tr, 3);
  std::printf("sequence %dx%d band 3: flicker input %.4f truth %.4f, band error input %.4f truth %.4f\n", w, h, fIn, fTr,
              eIn, eTr);
  CHECK(fTr < 0.04 && fIn > 0.08 && fIn > 3 * fTr && eTr == 0.0 && eIn > 0.06);
  // The noise is new each frame: residuals of neighbouring frames are nearly uncorrelated.
  double sxy = 0, sxx = 0, syy = 0;
  for (size_t i = 0; i < in[4].size(); ++i) {
    const double a = in[4][i] - tr[4][i], b = in[5][i] - tr[5][i];
    sxy += a * b, sxx += a * a, syy += b * b;
  }
  const double corr = sxy / std::sqrt(sxx * syy);
  std::printf("sequence residual correlation frames 4/5: %.3f\n", corr);
  CHECK(std::fabs(corr) < 0.3);
  // Static background, determinism, seeds, size independence (motion in 1080 units).
  const synth::Sequence st = synth::makeSequence(w, h, 3, 1, false);
  CHECK(st.background[2].tx == 0 && st.frames[0].rgb[0] == st.frames[2].rgb[0]);
  CHECK(s.frames[0].rgb[0] != s.frames[2].rgb[0]);
  const synth::Sequence again = synth::makeSequence(w, h, 9, 1, true);
  CHECK(again.frames[5].matte == s.frames[5].matte && again.frames[5].rgb == s.frames[5].rgb);
  const synth::Sequence s2 = synth::makeSequence(w, h, 9, 2, true);
  CHECK(s2.frames[3].matte != s.frames[3].matte && s2.subject[3].tx != s.subject[3].tx);
  const synth::Sequence small = synth::makeSequence(480, 270, 9, 1, true);
  double moveBig = 0, moveSmall = 0, dummy;
  s.subject[8].apply(w / 2.0, h / 2.0, &moveBig, &dummy);
  small.subject[8].apply(240, 135, &moveSmall, &dummy);
  CHECK(std::fabs((moveBig - w / 2.0) / k - (moveSmall - 240) / 0.25) < 1e-9);
  std::vector<std::vector<float>> inSmall;
  for (const auto& f : small.frames) inSmall.push_back(f.matte);
  const double eSmall = synth::bandError(small, inSmall, 3 * 0.5);
  std::printf("sequence 480x270 band 1.5: band error input %.4f\n", eSmall);
  CHECK(std::fabs(eSmall - eIn) < 0.25 * eIn);
}

void testExr(const std::string& dir) {
  const int w = 5, h = 3;
  std::vector<float> rgb(size_t(w) * h * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(i) / 10;
  const std::string path = dir + "/test_synth.exr";
  CHECK(synth::writeExr(path, w, h, 3, rgb));
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  uint32_t magic = 0;
  std::memcpy(&magic, b.data(), 4);
  CHECK(magic == 20000630);
  // The last line holds B, G, R of row 2: R of pixel (4, 2) is the file's last float.
  float last = 0;
  std::memcpy(&last, b.data() + b.size() - 4, 4);
  CHECK(last == rgb[(2 * w + 4) * 3]);
  // Header + offsets (8 per line) + lines (8 + w * 3 channels * 4 bytes).
  const size_t lines = size_t(h) * (8 + size_t(w) * 12);
  const size_t headerEnd = b.size() - lines - 8 * size_t(h);
  CHECK(b[headerEnd - 1] == 0);
  uint64_t first = 0;
  std::memcpy(&first, b.data() + headerEnd, 8);
  CHECK(first == headerEnd + 8 * size_t(h));
  std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  testScenes();
  testGaps();
  testHair();
  testSequence();
  testExr(argc > 1 ? argv[1] : ".");
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
