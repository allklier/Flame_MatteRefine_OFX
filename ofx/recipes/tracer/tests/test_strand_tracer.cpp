// Strand tracer (strand_tracer.h, docs/strand_tracing.md): the prototype's metrics on the C++ re-creation of its 12
// synthetic scenes (tracer_scenes.h), known answers, the user's rulings, determinism, region bit-identity, timing.
//   test_strand_tracer             [smoke] single strand, rulings, determinism, region identity, crop timing
//   test_strand_tracer --scenes    [slow]  known answers on the scene types (6 seeds), all rules vs none
//   test_strand_tracer --ablation  [slow]  every rule switch off in turn (6 seeds x 12 scenes x 12 configs),
//                                          a 4096x2304 frame timing
// Port fidelity: on the prototype's own scene dumps the port reproduces ablation.txt line for line (all 169 rows,
// 2 decimals; tools/tracer_eval scenes). The scenes here use a different random generator, so the asserts carry a
// tolerance against the prototype's table: all rules rec 0.97 / false 0.01 / wander 1 / early 2 / missed 0; none
// rec 0.99 / false 0.31 / wander 37.
// Mutations (2026-11-12, each made, run, reverted): turn-window limit x10 -> the L-corner check fails; running
// confidence replaced by the raw step score -> crossing recall and the all-rules wander / ruled bounds fail; width
// consistency term removed -> region-identity mx check and the frame / brownframe band checks fail; region multiply
// removed -> alpha <= region fails; ruling 1 inverted -> the ruling 1 and 2 piece checks fail; fallback width limit
// x3 -> clearlyDiffers and the two-shape fallback fail; ruling 4 halved -> the minimum-length checks fail;
// the unanchored bar replaced by keepConf (0.45) -> the 0.55 keep-bar check fails; elongation compared with
// the reference width instead of the piece's own -> the elongation check fails.
// The mole acceptance (elongation on: prototype-scene mole false 0.42 -> 0.03, wander 4 -> 0) needs the prototype's
// scene dumps and is checked with tools/tracer_eval, not in ctest; the C++ mole scene passes with or without it.
// SURVIVED: the hard width limit (w > 7 -> score 0) x10: the width term already rejects every band here (redundant guard).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strand_tracer.h"
#include "tracer_scenes.h"

using namespace mr::tracer;
using scenes::Metrics;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

TracerParams proto() {  // the prototype's behaviour (rulings off)
  TracerParams p;
  p.frameHairRef = false;
  p.dropEdgeTouching = false;
  p.minPieceLen1080 = 0.0;
  p.minConfUnanchored = 0.7;  // the prototype's free_keep
  p.rElongation = false;      // added after the prototype
  return p;
}

TracerInput inputOf(const scenes::Scene& S) {
  TracerInput in;
  in.rgb = S.rgb.data();
  in.base = S.alpha.data();
  in.region = S.region.data();
  in.width = S.w;
  in.height = S.h;
  in.frameHeight = 1080;
  return in;
}

Metrics sceneMetrics(const std::string& name, const TracerParams& p, int seeds) {
  std::vector<Metrics> acc;
  for (int k = 0; k < seeds; ++k) {
    const scenes::Scene S = scenes::makeScene(name, k);
    const TracerResult r = traceStrands(inputOf(S), p);
    acc.push_back(scenes::metrics(S, r.alpha, r.pieces));
  }
  return scenes::aggregate(acc);
}

void print(const char* tag, const Metrics& m) {
  std::printf("  %-14s rec %.2f false %.2f distr %.2f sad %.2f wander %d early %d missed %d\n", tag, m.rec, m.fals,
              m.distr, m.sad, m.wander, m.early, m.missed);
}

// ---- a tiny painter for hand-made scenes ----------------------------------------------------------------
struct Canvas {
  int w, h;
  std::vector<float> rgb, base, region;
  Canvas(int w_, int h_) : w(w_), h(h_), rgb((size_t)w_ * h_ * 3), base((size_t)w_ * h_, 0.0f), region((size_t)w_ * h_, 0.0f) {
    for (int y = 0; y < h; ++y)  // skin with a gentle deterministic texture
      for (int x = 0; x < w; ++x) {
        const double t = 1.0 + 0.02 * std::sin(x * 0.37 + y * 0.11) + 0.02 * std::sin(y * 0.29 - x * 0.07);
        float* p = &rgb[((size_t)y * w + x) * 3];
        p[0] = (float)(0.60 * t);
        p[1] = (float)(0.42 * t);
        p[2] = (float)(0.33 * t);
      }
  }
  // a straight dark segment of width wd (coverage = clip(w/2 + 0.5 - d)), colour col
  void segment(double x0, double y0, double x1, double y1, double wd, const double col[3]) {
    const double L = std::hypot(x1 - x0, y1 - y0);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / (L * L);
        t = std::min(std::max(t, 0.0), 1.0);
        const double d = std::hypot(x - (x0 + t * (x1 - x0)), y - (y0 + t * (y1 - y0)));
        const double c = std::min(std::max(wd * 0.5 + 0.5 - d, 0.0), 1.0);
        float* p = &rgb[((size_t)y * w + x) * 3];
        for (int k = 0; k < 3; ++k) p[k] = (float)(p[k] * (1 - c) + col[k] * c);
      }
  }
  void solidLeftOf(int xe) {  // base matte = 1 (and hair colour) left of xe
    const double hair[3] = {0.05, 0.035, 0.025};
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < xe; ++x) {
        base[(size_t)y * w + x] = 1.0f;
        for (int k = 0; k < 3; ++k) rgb[((size_t)y * w + x) * 3 + k] = (float)hair[k];
      }
  }
  void box(int x0, int y0, int x1, int y1, float v = 1.0f) {
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) region[(size_t)y * w + x] = std::max(region[(size_t)y * w + x], v);
  }
  TracerInput input(int frameHeight = 1080) const {
    TracerInput in;
    in.rgb = rgb.data();
    in.base = base.data();
    in.region = region.data();
    in.width = w;
    in.height = h;
    in.frameHeight = frameHeight;
    return in;
  }
};
const double HAIR[3] = {0.05, 0.035, 0.025};

// ---- [smoke] known answers on the prototype's scene types (6 seeds each) -------------------------------
void testKnownScenes() {
  std::printf("known scenes (all rules, rulings on, 6 seeds)\n");
  const TracerParams p;
  const Metrics single = sceneMetrics("single", p, 6);
  print("single", single);
  CHECK(single.rec >= 0.95 && single.fals <= 0.03 && single.wander == 0 && single.missed == 0);
  const Metrics pair = sceneMetrics("pair", p, 6);  // two strands, 2 px gap: both found (merging is fine, ruling 3)
  print("pair", pair);
  CHECK(pair.rec >= 0.95 && pair.missed == 0 && pair.wander == 0);
  const Metrics glove = sceneMetrics("glove", p, 6);
  print("glove", glove);
  CHECK(glove.rec >= 0.9 && glove.distr <= 0.02 && glove.wander == 0);
  const Metrics frame = sceneMetrics("frame", p, 6);  // glasses band through the tip: stop there, never follow it
  print("frame", frame);
  CHECK(frame.rec >= 0.7 && frame.distr <= 0.02 && frame.wander == 0 && frame.missed == 0);
  const Metrics brown = sceneMetrics("brownframe", p, 6);
  print("brownframe", brown);
  CHECK(brown.rec >= 0.7 && brown.distr <= 0.02 && brown.wander == 0);
  const Metrics cross = sceneMetrics("crossing", p, 6);
  print("crossing", cross);
  CHECK(cross.rec >= 0.85 && cross.missed == 0);
  const Metrics mole = sceneMetrics("mole", p, 6);
  print("mole", mole);
  CHECK(mole.rec >= 0.9 && mole.wander <= 1 && mole.distr <= 0.2);
}

// ---- [smoke] one single strand: an anchored piece from the matte, a sane frame reference ---------------
void testSingle() {
  std::printf("single strand\n");
  const TracerParams p;
  const scenes::Scene S = scenes::makeScene("single", 0);
  const TracerResult r = traceStrands(inputOf(S), p);
  int anchored = 0;
  for (const Piece& pc : r.pieces) anchored += pc.anchored;
  CHECK(r.shapes == 1 && anchored >= 1 && r.reference.valid && !r.reference.selfReferenced);
  CHECK(r.reference.width > 2.0 && r.reference.width < 4.5 && r.reference.depth > 1.0);
}

// ---- [smoke] soft strand profile (agent hl_flicker): no hard steps, free ends fade, same strand found --------
// Max 4-neighbour alpha step on the candidate (pixels with alpha > 0 or next to one).
double maxStep(const std::vector<float>& a, int w, int h) {
  double mx = 0.0;
  for (int y = 0; y + 1 < h; ++y)
    for (int x = 0; x + 1 < w; ++x) {
      const size_t i = size_t(y) * w + x;
      mx = std::max(mx, (double)std::fabs(a[i] - a[i + 1]));
      mx = std::max(mx, (double)std::fabs(a[i] - a[i + w]));
    }
  return mx;
}
void testSoftProfile() {
  std::printf("soft profile\n");
  const scenes::Scene S = scenes::makeScene("single", 0);
  TracerParams hard;
  TracerParams soft;
  soft.softProfile = true;
  soft.taperW = 0.75;
  const TracerResult rh = traceStrands(inputOf(S), hard), rs = traceStrands(inputOf(S), soft);
  const double sh = maxStep(rh.alpha, S.w, S.h), ss = maxStep(rs.alpha, S.w, S.h);
  const Metrics mh = scenes::metrics(S, rh.alpha, rh.pieces), ms = scenes::metrics(S, rs.alpha, rs.pieces);
  float peak = 0.0f;
  for (float v : rs.alpha) peak = std::max(peak, v);
  std::printf("  max step hard %.2f soft %.2f, rec hard %.2f soft %.2f, false hard %.3f soft %.3f, sad hard %.2f soft %.2f, "
              "peak %.2f\n", sh, ss, mh.rec, ms.rec, mh.fals, ms.fals, mh.sad, ms.sad, peak);
  CHECK(rs.pieces.size() == rh.pieces.size());  // the alpha changes, not what is kept
  CHECK(sh > 0.6);                              // the capsule's step (the defect)
  CHECK(ss < 0.45);                             // a Gaussian edge of 1 px at 1080 lines: about 0.38 x the peak
  CHECK(peak > 0.7f);                           // still a bright strand
  CHECK(ms.rec >= mh.rec - 0.05 && ms.fals <= 0.05);  // the soft fringe adds a little outside the truth
}

// ---- [smoke] all rules vs none, against the prototype's table -------------------------------------------
void testAllVsNone() {
  std::printf("all vs none (prototype mode, 6 seeds x 12 scenes)\n");
  Metrics all, none, ruled;
  scenes::evaluate(proto(), 6, all);
  TracerParams n = proto();
  n.rTurn = n.rWidth = n.rProfile = n.rColour = n.rContrast = n.rOrient = n.rDecay = n.rConnect = n.rBudget =
      n.rHairCtx = false;
  scenes::evaluate(n, 6, none);
  scenes::evaluate(TracerParams(), 6, ruled);
  print("all", all);
  print("none", none);
  print("all+rulings", ruled);
  // prototype: all 0.97 / 0.01 / 0.01 / wander 1 / early 2 / missed 0; none 0.99 / 0.31 / wander 37
  CHECK(std::fabs(all.rec - 0.97) <= 0.02 && all.fals <= 0.03 && all.distr <= 0.03);
  CHECK(all.wander <= 3 && all.early <= 4 && all.missed == 0);
  CHECK(none.rec >= 0.95 && none.fals >= 0.2 && none.wander >= 25 && none.wander <= 50);
  CHECK(ruled.rec >= 0.95 && ruled.fals <= 0.03 && ruled.wander <= 3 && ruled.missed == 0);
}

// ---- [slow] every rule off in turn --------------------------------------------------------------------
void testAblation() {
  std::printf("ablation (prototype mode, 6 seeds x 12 scenes)\n");
  Metrics all;
  scenes::evaluate(proto(), 6, all);
  print("all", all);
  struct Abl {
    const char* name;
    bool TracerParams::*flag;
    bool mustMatter;  // the prototype's table shows a clear effect (wander up by 3 or more)
  };
  const Abl abl[] = {{"-turn", &TracerParams::rTurn, true},       {"-width", &TracerParams::rWidth, true},
                     {"-profile", &TracerParams::rProfile, true}, {"-colour", &TracerParams::rColour, false},
                     {"-contrast", &TracerParams::rContrast, false}, {"-orient", &TracerParams::rOrient, false},
                     {"-decay", &TracerParams::rDecay, true},     {"-connect", &TracerParams::rConnect, false},
                     {"-budget", &TracerParams::rBudget, false},  {"-hairctx", &TracerParams::rHairCtx, true}};
  for (const Abl& a : abl) {
    TracerParams p = proto();
    p.*(a.flag) = false;
    Metrics m;
    scenes::evaluate(p, 6, m);
    print(a.name, m);
    CHECK(m.rec >= 0.93);  // no rule is needed for recall
    if (a.mustMatter) CHECK(m.wander >= all.wander + 3 && m.fals > all.fals);
  }
  TracerParams cap = proto();
  cap.capsuleAlpha = true;
  Metrics m;
  scenes::evaluate(cap, 6, m);
  print("capsule", m);
  CHECK(m.rec >= 0.95 && m.fals <= 0.03 && m.sad < 0.5);
}

// ---- [smoke] hand-made known answers: a sharp corner, a wide band, a lone dark dot ---------------------
double maxTurnOver(const Piece& pc, int steps) {  // the largest direction change over `steps` steps, degrees
  double mx = 0;
  const std::vector<Point>& P = pc.polyline;
  for (size_t i = 0; i + steps + 1 < P.size(); ++i) {
    const double a = std::atan2(P[i + 1].y - P[i].y, P[i + 1].x - P[i].x);
    const double b = std::atan2(P[i + steps + 1].y - P[i + steps].y, P[i + steps + 1].x - P[i + steps].x);
    mx = std::max(mx, std::fabs(std::atan2(std::sin(b - a), std::cos(b - a))) * 180.0 / 3.14159265358979);
  }
  return mx;
}

void testHandMade() {
  std::printf("hand-made scenes\n");
  const TracerParams p;
  {  // an L-shaped dark line: no piece turns the 90 deg corner within 6 steps (turn window 60 deg + one fan step)
    Canvas c(100, 100);
    c.segment(15, 25, 60, 25, 3.0, HAIR);
    c.segment(60, 25, 60, 85, 3.0, HAIR);
    c.box(8, 15, 72, 92);
    const TracerResult r = traceStrands(c.input(), p);
    CHECK(!r.pieces.empty());
    for (const Piece& pc : r.pieces) CHECK(maxTurnOver(pc, 6) <= 60.0 + 22.5 + 1e-6);
  }
  {  // a 9 px dark band (a glasses frame) alone in the region: wider than any strand, nothing traced
    Canvas c(100, 80);
    const double black[3] = {0.02, 0.02, 0.022};
    c.segment(10, 40, 90, 44, 9.0, black);
    c.box(20, 20, 80, 62);
    const TracerResult r = traceStrands(c.input(), p);
    float mx = 0;
    for (float v : r.alpha) mx = std::max(mx, v);
    CHECK(r.pieces.empty() && mx == 0.0f);
  }
  {  // a lone mole (dark disc, radius 3) in the region: no piece
    Canvas c(60, 60);
    const double mole[3] = {0.08, 0.056, 0.04};
    c.segment(30, 30, 30.01, 30, 6.0, mole);
    c.box(10, 10, 50, 50);
    const TracerResult r = traceStrands(c.input(), p);
    CHECK(r.pieces.empty());
  }
}

// ---- [smoke] the rulings --------------------------------------------------------------------------------
void testRulings() {
  std::printf("rulings\n");
  TracerParams p;
  // ruling 4: 12 px at 1080 lines, scaled with the frame height and the render scale
  CHECK(std::fabs(minPieceLength(p, 1080, 1080, 1.0) - 12.0) < 1e-9);
  CHECK(std::fabs(minPieceLength(p, 2304, 2304, 1.0) - 25.6) < 1e-9);
  CHECK(std::fabs(minPieceLength(p, 2304, 1152, 0.5) - 12.8) < 1e-9);
  CHECK(std::fabs(minPieceLength(p, 0, 1152, 0.5) - 12.8) < 1e-9);  // frame height = buffer / render scale
  {
    const scenes::Scene S = scenes::makeScene("single", 1);
    TracerInput in = inputOf(S);
    const TracerResult a = traceStrands(in, p);
    CHECK(!a.pieces.empty());
    double longest = 0;
    for (const Piece& pc : a.pieces) longest = std::max(longest, pc.length);
    in.frameHeight = (int)std::ceil((longest + 1.0) / 12.0 * 1080.0);  // minimum just above the longest piece
    const TracerResult b = traceStrands(in, p);
    CHECK(b.pieces.empty());
    float mx = 0;
    for (float v : b.alpha) mx = std::max(mx, v);
    CHECK(mx == 0.0f);
  }
  // ruling 1: an unconnected strand (no matte anywhere) is kept when inside the region, dropped when it reaches
  // the region's edge; the prototype (switch off) keeps both
  {
    Canvas inside(120, 80);
    inside.segment(25, 40, 95, 42, 3.0, HAIR);
    inside.box(10, 25, 110, 57);
    const TracerResult r = traceStrands(inside.input(), p);
    CHECK(r.pieces.size() == 1);
    if (r.pieces.size() == 1) {
      CHECK(!r.pieces[0].anchored && !r.pieces[0].touchesRegionEdge && r.pieces[0].selfReferenced);
      CHECK(r.pieces[0].length >= 60.0);
      CHECK(r.reference.valid && r.reference.selfReferenced);
    }
    Canvas edge(120, 80);
    edge.segment(5, 40, 115, 42, 3.0, HAIR);
    edge.box(30, 25, 90, 57);
    const TracerResult e = traceStrands(edge.input(), p);
    CHECK(e.pieces.empty());
    TracerParams off = p;
    off.dropEdgeTouching = false;
    const TracerResult k = traceStrands(edge.input(), off);
    CHECK(k.pieces.size() == 1 && k.pieces[0].touchesRegionEdge);
  }
  // ruling 2: one reference per frame; a shape falls back to its own only when the shared one clearly differs
  {
    HairReference a, b;
    a.valid = b.valid = true;
    a.width = 3.0;
    a.depth = 2.0;
    b = a;
    b.width = 4.5;  // ln 1.5 = 0.41 < 0.5: fits
    CHECK(!clearlyDiffers(b, a, p));
    b.width = 5.2;  // ln 1.73 = 0.55: clearly differs
    CHECK(clearlyDiffers(b, a, p));
    b = a;
    b.depth = 0.6;  // ratio 0.3 < 1/3
    CHECK(clearlyDiffers(b, a, p));
    b = a;
    b.chroma2 = 0.35;  // > 2 colSig
    CHECK(clearlyDiffers(b, a, p));
    // two shapes, A with an anchored 3 px strand from the matte, B with a free 3 px strand: B uses the frame's
    // (anchored) reference; with a 6.5 px strand B falls back to its own (self-referenced)
    for (int wide = 0; wide < 2; ++wide) {
      Canvas c(200, 120);
      c.solidLeftOf(30);
      c.segment(26, 30, 80, 32, 3.0, HAIR);
      c.segment(110, 90, 185, 88, wide ? 6.5 : 3.0, HAIR);
      c.box(30, 20, 70, 42);
      c.box(100, 78, 192, 100);
      const TracerResult r = traceStrands(c.input(), p);
      CHECK(r.shapes == 2);
      CHECK(r.reference.valid && !r.reference.selfReferenced);
      int nb = 0;
      for (const Piece& pc : r.pieces)
        if (pc.shape == 1) {
          ++nb;
          CHECK(!pc.anchored && pc.selfReferenced == (wide == 1));
        }
      CHECK(nb == 1);
    }
  }
  // the unanchored keep bar (minConfUnanchored, "Rebuild Min Confidence"): shape B's free strand, slightly
  // reddish against the shared reference, scores about 0.51: kept at 0.45, dropped at 0.55
  {
    Canvas c(200, 120);
    c.solidLeftOf(30);
    c.segment(26, 30, 80, 32, 3.0, HAIR);
    const double reddish[3] = {0.074, 0.035, 0.025};
    c.segment(110, 90, 185, 88, 4.0, reddish);
    c.box(30, 20, 70, 42);
    c.box(100, 78, 192, 100);
    for (int hi = 0; hi < 2; ++hi) {
      TracerParams q = p;
      q.minConfUnanchored = hi ? 0.55 : 0.45;
      const TracerResult r = traceStrands(c.input(), q);
      int nb = 0;
      for (const Piece& pc : r.pieces)
        if (pc.shape == 1) {
          ++nb;
          CHECK(!pc.anchored && !pc.selfReferenced && pc.confidence > 0.47 && pc.confidence < 0.55);
        }
      CHECK(nb == (hi ? 0 : 1));
    }
  }
  // the elongation rule (rElongation / minElongation): an unanchored 10 px dash traces to about 13.5 px at
  // FWHM 3.1 (4.3 x): kept at minElongation 4.0, dropped at 4.5, kept at 4.5 when the switch is off
  {
    Canvas c(200, 120);
    c.solidLeftOf(30);
    c.segment(26, 30, 80, 32, 3.0, HAIR);
    c.segment(140, 89, 150, 89, 3.5, HAIR);
    c.box(30, 20, 70, 42);
    c.box(110, 74, 180, 104);
    const double k[3] = {4.0, 4.5, 4.5};
    for (int t = 0; t < 3; ++t) {
      TracerParams q = p;
      q.minElongation = k[t];
      q.rElongation = t != 2;
      const TracerResult r = traceStrands(c.input(), q);
      int nb = 0;
      for (const Piece& pc : r.pieces)
        if (pc.shape == 1) {
          ++nb;
          CHECK(!pc.anchored && pc.length / pc.width > 4.0 && pc.length / pc.width < 4.5);
        }
      CHECK(nb == (t == 1 ? 0 : 1));
    }
  }
  // chainLength (agent hl96): a short strand that starts at the matte (a contact piece) is dropped by ruling 4
  // when the minimum is just above its length; with chainLength it is kept (attached to the matte: not a speck),
  // and a loose dash of the same length is still dropped
  {
    Canvas c(160, 100);
    c.solidLeftOf(30);
    c.segment(27, 40, 41, 41, 3.0, HAIR);   // from the matte
    c.segment(30, 50, 80, 52, 3.0, HAIR);   // a long anchored strand (the frame's hair reference)
    c.segment(100, 75, 114, 75, 3.0, HAIR); // a loose dash, same length as the short strand
    c.box(30, 32, 90, 60);
    c.box(92, 66, 124, 84);
    TracerParams q = p;
    q.minPieceLen1080 = 0.0;
    TracerInput in = c.input();
    const TracerResult all = traceStrands(in, q);
    double shortLen = 0, looseLen = 0;
    for (const Piece& pc : all.pieces) {
      if (pc.shape == 0 && pc.anchored && pc.length < 30) shortLen = pc.length;
      if (pc.shape == 1) looseLen = pc.length;
    }
    CHECK(shortLen > 0 && looseLen > 0);
    in.frameHeight = (int)std::ceil((std::max(shortLen, looseLen) + 1.0) / 12.0 * 1080.0);
    for (int chain = 0; chain < 2; ++chain) {
      TracerParams r4 = p;
      r4.chainLength = chain == 1;
      const TracerResult r = traceStrands(in, r4);
      int shortKept = 0, looseKept = 0, longKept = 0;
      for (const Piece& pc : r.pieces) {
        if (pc.shape == 1) ++looseKept;
        else if (pc.length < 30) ++shortKept;
        else ++longKept;
      }
      CHECK(longKept == 1 && looseKept == 0 && shortKept == chain);
    }
  }
  // a reference passed in is used as the frame reference
  {
    const scenes::Scene S = scenes::makeScene("single", 2);
    TracerInput in = inputOf(S);
    HairReference ref;
    ref.valid = true;
    ref.width = 3.1;
    ref.depth = 2.2;
    in.reference = &ref;
    const TracerResult r = traceStrands(in, p);
    CHECK(r.reference.valid && r.reference.width == 3.1 && r.reference.depth == 2.2);
    CHECK(!r.pieces.empty());
  }
}

// ---- [smoke] determinism, bit identity outside the region, region shapes, empty inputs ------------------
void testInvariants() {
  std::printf("invariants\n");
  const TracerParams p;
  for (const char* name : {"combo", "frame", "crossing"}) {
    const scenes::Scene S = scenes::makeScene(name, 3);
    const TracerResult a = traceStrands(inputOf(S), p), b = traceStrands(inputOf(S), p);
    CHECK(a.alpha.size() == b.alpha.size() && !std::memcmp(a.alpha.data(), b.alpha.data(), a.alpha.size() * 4));
    CHECK(!std::memcmp(a.confidence.data(), b.confidence.data(), a.confidence.size() * 4));
    CHECK(a.pieces.size() == b.pieces.size());
    int outside = 0, conf = 0;
    float mx = 0;
    for (size_t i = 0; i < a.alpha.size(); ++i) {
      if (!(S.region[i] > 0.0f)) outside += a.alpha[i] != 0.0f || a.confidence[i] != 0.0f;
      conf += (a.alpha[i] > 0.0f) != (a.confidence[i] > 0.0f);
      mx = std::max(mx, a.alpha[i]);
      CHECK(a.alpha[i] >= 0.0f && a.alpha[i] <= S.region[i] + 1e-6f);
    }
    CHECK(outside == 0 && conf == 0 && mx > 0.5f);
  }
  {  // no region: nothing at all
    scenes::Scene S = scenes::makeScene("single", 0);
    std::fill(S.region.begin(), S.region.end(), 0.0f);
    const TracerResult r = traceStrands(inputOf(S), p);
    float mx = 0;
    for (float v : r.alpha) mx = std::max(mx, v);
    CHECK(r.shapes == 0 && r.pieces.empty() && mx == 0.0f && !r.reference.valid);
  }
  {  // two drawn shapes whose feathers touch stay two shapes; a lone feather blob is a third
    Canvas c(100, 60);
    c.box(10, 10, 40, 40);
    c.box(44, 10, 80, 40);
    c.box(41, 10, 43, 40, 0.3f);
    c.box(90, 50, 92, 52, 0.2f);
    const TracerResult r = traceStrands(c.input(), p);
    CHECK(r.shapes == 3);
  }
}

// ---- timing --------------------------------------------------------------------------------------------
// A 424 x 562 crop with two region shapes around three strands (the real-frame size), and (slow) a 4096 x 2304
// frame with eight such shapes.
Canvas timingCanvas(int w, int h, int groups) {
  Canvas c(w, h);
  c.solidLeftOf(w / 3);
  for (int g = 0; g < groups; ++g) {
    const int y = 40 + g * (h - 80) / std::max(groups, 1);
    c.segment(w / 3 - 4, y, w / 3 + 30, y + 6, 3.0, HAIR);
    c.segment(w / 3 - 4, y + 5, w / 3 + 28, y + 14, 3.5, HAIR);
    c.box(w / 3, y - 8, w / 3 + 30, y + 20);
  }
  return c;
}

double timeIt(const Canvas& c, int reps) {
  traceStrands(c.input(), TracerParams());
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < reps; ++k) traceStrands(c.input(), TracerParams());
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void testTiming(bool slow) {
  const Canvas crop = timingCanvas(424, 562, 2);
  const double ms = timeIt(crop, 5);
  std::printf("timing: 424x562 crop, 2 shapes: %.2f ms\n", ms);
  CHECK(ms < 30.0);  // budget "a few ms"; loose for a loaded machine
  if (slow) {
    const Canvas frame = timingCanvas(4096, 2304, 8);
    const double fm = timeIt(frame, 2);
    std::printf("timing: 4096x2304 frame buffer, 8 shapes: %.2f ms\n", fm);
    CHECK(fm < 100.0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "--ablation") {
    testAblation();
    testTiming(true);
  } else if (mode == "--scenes") {
    testKnownScenes();
    testAllVsNone();
  } else {
    testSingle();
    testSoftProfile();
    testHandMade();
    testRulings();
    testInvariants();
    testTiming(false);
  }
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all passed\n");
  return 0;
}
