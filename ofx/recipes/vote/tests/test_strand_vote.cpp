// Tests for the strand presence vote (strand_vote.h).
//   test_strand_vote             smoke: known answers, rule mutations, region bit-identity, determinism (< 1 s)
//   test_strand_vote --sequence  slow: the synthetic hair-locks sequence (synth::makeLocksSequence) through the
//                                real tracer + Temporal's flows: flicker and coverage before / after, the injected
//                                dip / vanish / spurious / flow-failure frames, the Q3 Temporal A/B, timing
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strand_vote.h"
#include "synth.h"
#include "temporal.h"

using namespace mr;
using tracer::Piece;
using tracer::Point;
using tracer::TracerResult;

static int failures = 0;
static bool quiet = false;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) {                                                          \
      ++failures;                                                        \
      if (!quiet) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
    }                                                                    \
  } while (0)

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------------------
// Known-answer fixtures: a W x H crop, hand-made tracer results (straight pieces), constant motions.

const int W = 80, H = 60;

Piece line(double x0, double y0, double x1, double y1, double width, double conf, bool anchored = false) {
  Piece p;
  const double len = std::hypot(x1 - x0, y1 - y0);
  const int steps = std::max(1, int(len / 1.5));
  for (int i = 0; i <= steps; ++i) p.polyline.push_back(Point{x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps});
  p.width = width;
  p.length = steps * 1.5;
  p.confidence = conf;
  p.anchored = anchored;
  return p;
}

// Geometric capsule alpha x region, confidence = piece confidence where alpha > 0 (as the tracer's maps).
TracerResult result(const std::vector<Piece>& pieces, const std::vector<float>& region) {
  TracerResult r;
  r.alpha.assign(size_t(W) * H, 0.0f);
  r.confidence.assign(size_t(W) * H, 0.0f);
  r.pieces = pieces;
  for (const Piece& p : pieces)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        double best = 1e30;
        for (size_t i = 0; i + 1 < p.polyline.size(); ++i) {
          const Point a = p.polyline[i], b = p.polyline[i + 1];
          const double vx = b.x - a.x, vy = b.y - a.y, l2 = vx * vx + vy * vy;
          double t = l2 > 0 ? ((x - a.x) * vx + (y - a.y) * vy) / l2 : 0;
          t = std::min(std::max(t, 0.0), 1.0);
          best = std::min(best, std::hypot(x - a.x - t * vx, y - a.y - t * vy));
        }
        const size_t i = size_t(y) * W + x;
        const float a = float(std::min(std::max(p.width * 0.5 + 0.5 - best, 0.0), 1.0) * region[i]);
        if (a > r.alpha[i]) r.alpha[i] = a;
        if (a > 0) r.confidence[i] = std::max(r.confidence[i], float(p.confidence));
      }
  return r;
}

std::vector<float> fullRegion() {  // 1 inside, 0 in a 4 px border
  std::vector<float> r(size_t(W) * H, 0.0f);
  for (int y = 4; y < H - 4; ++y)
    for (int x = 4; x < W - 4; ++x) r[size_t(y) * W + x] = 1.0f;
  return r;
}

struct Fixture {
  std::vector<float> region;
  std::vector<TracerResult> results;  // offsets -2..2 (index 2 = centre)
  std::vector<vote::DenseMotion> motion;  // per neighbour offset (-2, -1, 1, 2)
  double shift = 0;  // px per frame in x: the true motion (neighbour piece = centre piece + shift * offset)
  double conf = 1;   // flow confidence
};

Fixture fixture(double shift = 0, double conf = 1) {
  Fixture f;
  f.region = fullRegion();
  f.shift = shift;
  f.conf = conf;
  for (int o : {-2, -1, 1, 2}) {
    vote::DenseMotion m;
    m.offset = o;
    m.valid = true;
    m.dx.assign(size_t(W) * H, float(shift * o));
    m.dy.assign(size_t(W) * H, 0.0f);
    m.conf.assign(size_t(W) * H, float(conf));
    f.motion.push_back(m);
  }
  f.results.resize(5);
  return f;
}

// Pieces per offset: a generator gets the offset and returns the pieces of that frame (in its own coordinates).
template <class G>
void fill(Fixture& f, G gen) {
  for (int o = -2; o <= 2; ++o) f.results[o + 2] = result(gen(o), f.region);
}

vote::VoteResult run(const Fixture& f, const vote::VoteParams& p, std::vector<int> offsets = {-2, -1, 1, 2}) {
  std::vector<vote::Neighbour> nb;
  for (int o : offsets) {
    vote::Neighbour n;
    n.result = &f.results[o + 2];
    n.motion = f.motion[o < 0 ? o + 2 : o + 1].view();
    nb.push_back(n);
  }
  vote::VoteInput in;
  in.width = W;
  in.height = H;
  in.region = f.region.data();
  in.centre = &f.results[2];
  in.neighbours = nb.data();
  in.count = int(nb.size());
  return vote::votePresence(in, p);
}

double alphaSum(const vote::VoteResult& r) {
  double s = 0;
  for (float a : r.alpha) s += a;
  return s;
}

// All known-answer cases; returns the failures added (quiet = no printing, for the mutation runs).
int knownAnswers(const vote::VoteParams& p) {
  const int before = failures;
  // 1. a strand present in all 5 frames stays (own confidence above its bar)
  {
    Fixture f = fixture();
    fill(f, [](int) { return std::vector<Piece>{line(20, 30, 50, 30, 3, 0.7)}; });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate != vote::Fate::Dropped && r.pieces[0].score == 1.0);
    CHECK(r.alpha == f.results[2].alpha);  // a kept piece keeps the tracer's alpha bit for bit
  }
  // 1b. the same with the centre dipping below its bar: kept by the neighbours (rule high)
  {
    Fixture f = fixture();
    fill(f, [](int o) { return std::vector<Piece>{line(20, 30, 50, 30, 3, o == 0 ? 0.5 : 0.7)}; });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptHigh);
  }
  // 2. a one-frame spurious piece (above its bar, below confHigh) is dropped
  {
    Fixture f = fixture();
    fill(f, [](int o) {
      std::vector<Piece> v{line(20, 20, 50, 20, 3, 0.7)};
      if (o == 0) v.push_back(line(20, 40, 45, 42, 2.5, 0.75));
      return v;
    });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 2 && r.pieces[0].fate != vote::Fate::Dropped && r.pieces[1].fate == vote::Fate::Dropped);
    CHECK(r.alpha[size_t(41) * W + 32] == 0.0f && r.alpha[size_t(20) * W + 32] > 0.5f);
  }
  // 3. a one-frame dropout is reinstated from t - 1 and t + 1, with reduced confidence
  {
    Fixture f = fixture(2.0, 1.0);  // moving 2 px a frame
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o != 0) v.push_back(line(20 + 2.0 * o, 30, 50 + 2.0 * o, 30, 3, o == -1 ? 0.8 : 0.7));
      return v;
    });
    const vote::VoteResult r = run(f, p);
    CHECK(r.reinstated == 1 && r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::Reinstated);
    if (!r.pieces.empty()) CHECK(std::fabs(r.pieces[0].confidence - p.reinstateFactor * 0.7) < 1e-12);
    CHECK(r.alpha[size_t(30) * W + 35] > 0.9f && r.alpha[size_t(30) * W + 15] == 0.0f);
    CHECK(r.confidence[size_t(30) * W + 35] > 0.0f);
  }
  // 3b. no reinstatement when t + 1 lacks it (only t - 1 and t - 2 have it)
  {
    Fixture f = fixture();
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o < 0) v.push_back(line(20, 30, 50, 30, 3, 0.8));
      return v;
    });
    CHECK(run(f, p).reinstated == 0);
  }
  // 4. present in 2 of 5 frames (t and t + 1): kept above its bar, dropped below it
  {
    Fixture f = fixture();
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o == 0 || o == 1) v.push_back(line(20, 30, 50, 30, 3, 0.7));
      return v;
    });
    vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptLow && r.pieces[0].score == 0.25);
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o == 0 || o == 1) v.push_back(line(20, 30, 50, 30, 3, 0.5));
      return v;
    });
    r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::Dropped);
    // anchored pieces are held to minConf (0.45), not minConfUnanchored
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o == 0 || o == 1) v.push_back(line(20, 30, 50, 30, 3, 0.5, true));
      return v;
    });
    r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptLow);
  }
  // 5. an isolated but very clear (anchored, step score 0.9 + 0.15) piece survives (confHigh); a free 0.98 does not
  {
    Fixture f = fixture();
    fill(f, [](int o) {
      std::vector<Piece> v;
      if (o == 0) v.push_back(line(20, 30, 50, 30, 3, 1.05, true));
      if (o == 0) v.push_back(line(20, 45, 50, 45, 3, 0.98));
      return v;
    });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 2 && r.pieces[0].fate == vote::Fate::KeptConfident && r.pieces[1].fate == vote::Fate::Dropped);
  }
  // 6. the tolerance grows with the width: a 6 px piece found 3.5 px off in the neighbours still matches
  {
    Fixture f = fixture();
    fill(f, [](int o) { return std::vector<Piece>{line(20, 30 + (o ? 3.5 : 0), 50, 30 + (o ? 3.5 : 0), 6, 0.5)}; });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptHigh);
  }
  // 7. the tolerance grows where the flow is less reliable: conf 0.6, pieces 2.8 px off, width 2
  {
    Fixture f = fixture(0.0, 0.6);
    fill(f, [](int o) { return std::vector<Piece>{line(20, 30 + (o ? 2.8 : 0), 50, 30 + (o ? 2.8 : 0), 2, 0.5)}; });
    const vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptHigh);
  }
  // 8. the flow moves neighbour pieces back to t (motion 3 px a frame): matched
  {
    Fixture f = fixture(3.0, 1.0);
    fill(f, [](int o) { return std::vector<Piece>{line(30 + 3.0 * o, 15, 30 + 3.0 * o, 45, 2.5, 0.5)}; });  // across the motion
    vote::VoteResult r = run(f, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].score == 1.0);
    // and with the motion ignored (zero flow) they are not
    Fixture g = fixture(0.0, 1.0);
    g.results = f.results;
    r = run(g, p);
    CHECK(r.pieces.size() == 1 && r.pieces[0].matched < 4);
  }
  // 9. region gate (ruling Q7): neighbour pieces outside the region at t widened by the flow do not count
  {
    Fixture f = fixture();
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) f.region[size_t(y) * W + x] = x < 36 && x >= 4 && y >= 4 && y < H - 4 ? 1.0f : 0.0f;
    fill(f, [](int o) {  // a piece at x 50-70 (outside the region and its 3 px widening) in t - 1 and t + 1
      std::vector<Piece> v;
      if (o == -1 || o == 1) v.push_back(line(50, 30, 70, 30, 3, 0.8));
      return v;
    });
    const vote::VoteResult r = run(f, p);
    CHECK(r.reinstated == 0);
    CHECK(alphaSum(r) == 0.0);
  }
  // 10. flow check: unreliable flow (conf 0.1) -> nobody judged -> the frame's own result, with a note
  {
    Fixture f = fixture(0.0, 0.1);
    fill(f, [](int o) { return std::vector<Piece>{line(20, 30, 50, 30, 3, o == 0 ? 0.5 : 0.8)}; });
    const vote::VoteResult r = run(f, p);
    CHECK(r.fallback && r.judgedNeighbours == 0 && !r.note.empty());
    CHECK(r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::Dropped);  // 0.5 < its bar 0.62
  }
  // 11. invalid motion (the caller knows the flow failed) and the sequence ends (only t+1, t+2)
  {
    Fixture f = fixture();
    fill(f, [](int) { return std::vector<Piece>{line(20, 30, 50, 30, 3, 0.7)}; });
    f.motion[0].valid = false;  // offset -2
    vote::VoteResult r = run(f, p);
    CHECK(r.judgedNeighbours == 3 && !r.fallback && r.note.find("-2") != std::string::npos);
    r = run(f, p, {1, 2});
    CHECK(r.judgedNeighbours == 2 && r.pieces.size() == 1 && r.pieces[0].score == 1.0);
    r = run(f, p, {});
    CHECK(r.fallback && r.pieces.size() == 1 && r.pieces[0].fate == vote::Fate::KeptFallback);
  }
  return failures - before;
}

// The known answers use their own bars (anchored 0.45, unanchored 0.62), independent of the tracer's defaults.
vote::VoteParams fixtureParams() {
  vote::VoteParams p;
  p.minConf = 0.45;
  p.minConfUnanchored = 0.62;
  return p;
}

void testKnownAnswers() {
  vote::VoteParams p = fixtureParams();
  tracer::TracerParams tp;
  vote::VoteParams q;
  vote::setBars(q, tp);
  CHECK(q.minConf == tp.keepConf && q.minConfUnanchored == tp.minConfUnanchored);
  const tracer::TracerParams cp = vote::candidateTracerParams(tp, 0.15);
  CHECK(std::fabs(cp.keepConf - (tp.keepConf - 0.15)) < 1e-12 && std::fabs(cp.minConfUnanchored - (tp.minConfUnanchored - 0.15)) < 1e-12);
  const int f = knownAnswers(p);
  std::printf("known answers: %s\n", f ? "FAILED" : "ok");
}

// Each rule switched off must fail at least one known answer (the rule is exercised).
void testMutations() {
  struct M {
    const char* name;
    bool vote::VoteParams::*sw;
  } rules[] = {{"high", &vote::VoteParams::rHigh},           {"low", &vote::VoteParams::rLow},
               {"confHigh", &vote::VoteParams::rConfHigh},   {"reinstate", &vote::VoteParams::rReinstate},
               {"widthTol", &vote::VoteParams::rWidthTol},   {"flowTol", &vote::VoteParams::rFlowTol},
               {"regionGate", &vote::VoteParams::rRegionGate}, {"flowCheck", &vote::VoteParams::rFlowCheck}};
  int caught = 0;
  for (const M& m : rules) {
    vote::VoteParams p = fixtureParams();
    p.*(m.sw) = false;
    const int saved = failures;
    quiet = true;
    const int f = knownAnswers(p);
    quiet = false;
    failures = saved;
    std::printf("  mutation %-10s off: %d known answers fail%s\n", m.name, f, f ? "" : "  <- NOT CAUGHT");
    if (f) ++caught;
    CHECK(f > 0);
  }
  std::printf("mutations: %d of %zu caught\n", caught, sizeof rules / sizeof rules[0]);
}

// Exactly 0 where the region is 0 or NaN, even if a (mis-behaving) tracer result has alpha there.
void testRegionZero() {
  Fixture f = fixture();
  fill(f, [](int o) {
    std::vector<Piece> v{line(2, 30, 60, 30, 3, 0.8)};
    if (o == 0) v.push_back(line(10, 2, 10, 50, 3, 0.9));
    return v;
  });
  for (TracerResult& r : f.results)  // alpha also in the border (region 0): simulate a careless producer
    for (size_t i = 0; i < r.alpha.size(); ++i)
      if (f.region[i] == 0.0f) r.alpha[i] = 0.7f, r.confidence[i] = 0.9f;
  f.region[size_t(30) * W + 30] = std::nanf("");
  vote::VoteParams p = fixtureParams();
  const vote::VoteResult r = run(f, p);
  bool zero = true;
  for (size_t i = 0; i < r.alpha.size(); ++i)
    if (!(f.region[i] > 0.0f)) {
      unsigned a, c;
      std::memcpy(&a, &r.alpha[i], 4);
      std::memcpy(&c, &r.confidence[i], 4);
      zero = zero && a == 0u && c == 0u;  // +0.0f bit for bit
    }
  CHECK(zero);
  CHECK(alphaSum(r) > 0.0);
  std::printf("region bit-identical zero: %s\n", zero ? "ok" : "FAILED");
}

// Temporal blend in presence mode (agent hl_flicker): a strand in every frame keeps its alpha, a strand at t only
// (kept by confHigh) fades, a strand in t and t+2 only is partly faded. Fails with blendNear / blendFar = 0.
double sumIn(const vote::VoteResult& r, int x0, int x1, int y0, int y1) {
  double s = 0;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) s += r.alpha[size_t(y) * W + x];
  return s;
}
void testBlend() {
  Fixture f = fixture(1.0, 1.0);
  fill(f, [](int o) {
    std::vector<Piece> v{line(10 + o, 12, 60 + o, 12, 3, 0.8, true)};          // everywhere
    if (o == 0) v.push_back(line(10, 30, 60, 30, 3, 1.1, true));              // t only, confident
    if (o == 0 || o == 2) v.push_back(line(10 + o, 46, 60 + o, 46, 3, 0.8, true));  // t and t+2
    if (o == 1) v.push_back(line(10 + o, 22, 60 + o, 22, 3, 0.8, true));             // t+1 only: a ghost
    return v;
  });
  vote::VoteParams off = fixtureParams(), on = fixtureParams();
  on.blendNear = 1.0;
  on.blendFar = 0.5;
  on.presenceLo = 0.15;
  on.presenceHi = 0.5;
  const vote::VoteResult a = run(f, off), b = run(f, on);
  const double e0 = sumIn(a, 5, 70, 8, 16), e1 = sumIn(b, 5, 70, 8, 16);
  const double c0 = sumIn(a, 5, 70, 26, 34), c1 = sumIn(b, 5, 70, 26, 34);
  const double d0 = sumIn(a, 5, 70, 42, 50), d1 = sumIn(b, 5, 70, 42, 50);
  std::printf("blend: everywhere %.1f -> %.1f, t only %.1f -> %.1f, t and t+2 %.1f -> %.1f\n", e0, e1, c0, c1, d0, d1);
  CHECK(e0 > 50 && e1 >= 0.9 * e0);       // a steady strand keeps its alpha
  CHECK(c0 > 50 && c1 <= 0.35 * c0);      // a one-frame strand fades instead of popping
  CHECK(d0 > 50 && d1 < 0.9 * d0 && d1 > 0.4 * d0);  // partial presence: partly faded
  const double g1 = sumIn(b, 5, 70, 20, 24);
  std::printf("blend: a strand of t+1 alone at t: %.2f\n", g1);
  CHECK(g1 < 1.0);  // a neighbour's strand that t has not got is not painted
  bool zero = true;  // still exactly 0 outside the region
  for (size_t i = 0; i < b.alpha.size(); ++i) zero = zero && (f.region[i] > 0.0f || b.alpha[i] == 0.0f);
  CHECK(zero);
}

// movedRegion: the region at t moved by a constant motion lands shifted in the neighbour, grown by the margin.
void testMovedRegion() {
  std::vector<float> reg(size_t(W) * H, 0.0f);
  for (int y = 20; y < 30; ++y)
    for (int x = 20; x < 30; ++x) reg[size_t(y) * W + x] = 1.0f;
  vote::DenseMotion m;
  m.offset = 1;
  m.valid = true;
  m.dx.assign(size_t(W) * H, 10.0f);
  m.dy.assign(size_t(W) * H, 0.0f);
  const std::vector<float> mr = vote::movedRegion(reg.data(), W, H, m.view(), 3.0);
  auto at = [&](int x, int y) { return mr[size_t(y) * W + x]; };
  CHECK(at(35, 25) == 1.0f && at(30, 25) == 1.0f && at(42, 25) == 1.0f);  // moved 10 px: 30..39, +3 margin
  CHECK(at(25, 25) == 0.0f && at(44, 25) == 0.0f && at(35, 15) == 0.0f);
  std::printf("moved region: %s\n", failures ? "FAILED" : "ok");
}

synth::LockFlicker injected();

// The synthetic sequence generator: deterministic, the locks are not in the base, the region holds them.
void testGenerator() {
  synth::LockFlicker fl = injected();
  fl.flowFailFrame = 3;
  const synth::LocksSequence a = synth::makeLocksSequence(212, 281, 4, 5, fl, 1.0);
  const synth::LocksSequence b = synth::makeLocksSequence(212, 281, 4, 5, fl, 1.0);
  CHECK(a.rgb == b.rgb && a.base == b.base && a.region == b.region && a.strandTruth == b.strandTruth);
  for (int t = 0; t < a.frames; ++t) {
    double truth = 0, inRegion = 0, inBase = 0;
    for (size_t i = 0; i < a.strandTruth[t].size(); ++i) {
      truth += a.strandTruth[t][i];
      inRegion += a.region[t][i] > 0.5f ? a.strandTruth[t][i] : 0.0f;
      inBase += a.base[t][i] >= 0.5f ? a.strandTruth[t][i] : 0.0f;
    }
    CHECK(truth > 100 && inRegion > 0.97 * truth && inBase < 0.05 * truth);
    CHECK(a.strands[t].size() == 3);
  }
  CHECK(a.spurious[0].empty() && a.motion.size() == 4);
  std::printf("generator: %s\n", failures ? "FAILED" : "ok");
}

void testDeterminism() {
  Fixture f = fixture(2.0, 0.8);
  fill(f, [](int o) {
    std::vector<Piece> v;
    if (o != 0) v.push_back(line(20 + 2.0 * o, 30, 50 + 2.0 * o, 33, 3, 0.7));
    v.push_back(line(15, 15 + o, 60, 18 + o, 2, 0.55));
    return v;
  });
  vote::VoteParams p;
  const vote::VoteResult a = run(f, p), b = run(f, p);
  const bool same = a.alpha.size() == b.alpha.size() &&
                    std::memcmp(a.alpha.data(), b.alpha.data(), a.alpha.size() * 4) == 0 &&
                    std::memcmp(a.confidence.data(), b.confidence.data(), a.confidence.size() * 4) == 0 &&
                    a.pieces.size() == b.pieces.size();
  CHECK(same);
  std::printf("determinism: %s\n", same ? "ok" : "FAILED");
}

// ---------------------------------------------------------------------------------------------------------------
// The synthetic hair-locks sequence through the real tracer and Temporal's flows.

struct Crop {
  int x0 = 0, y0 = 0, w = 0, h = 0;
};

std::vector<float> cropPlane(const std::vector<float>& f, int fw, int ch, const Crop& c) {
  std::vector<float> o(size_t(c.w) * c.h * ch);
  for (int y = 0; y < c.h; ++y)
    std::memcpy(&o[size_t(y) * c.w * ch], &f[(size_t(c.y0 + y) * fw + c.x0) * ch], size_t(c.w) * ch * 4);
  return o;
}

void pastePlane(std::vector<float>& f, int fw, const std::vector<float>& o, const Crop& c) {
  for (int y = 0; y < c.h; ++y) std::memcpy(&f[size_t(c.y0 + y) * fw + c.x0], &o[size_t(y) * c.w], size_t(c.w) * 4);
}

Crop regionBox(const synth::LocksSequence& s, int t, int margin) {
  int x0 = s.w, y0 = s.h, x1 = -1, y1 = -1;
  for (int y = 0; y < s.h; ++y)
    for (int x = 0; x < s.w; ++x)
      if (s.region[t][size_t(y) * s.w + x] > 0) x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
  Crop c;
  c.x0 = std::max(x0 - margin, 0);
  c.y0 = std::max(y0 - margin, 0);
  c.w = std::min(x1 + margin + 1, s.w) - c.x0;
  c.h = std::min(y1 + margin + 1, s.h) - c.y0;
  return c;
}

tracer::TracerResult trace(const synth::LocksSequence& s, int t, const Crop& c, const std::vector<float>& region,
                           const tracer::TracerParams& tp) {
  const std::vector<float> rgb = cropPlane(s.rgb[t], s.w, 3, c), base = cropPlane(s.base[t], s.w, 1, c);
  tracer::TracerInput in;
  in.rgb = rgb.data();
  in.channels = 3;
  in.base = base.data();
  in.region = region.data();
  in.width = c.w;
  in.height = c.h;
  in.frameHeight = s.h;
  return tracer::traceStrands(in, tp);
}

struct FrameVote {
  std::vector<float> alpha;  // full frame
  vote::VoteResult r;
  double flowMs = 0, traceMs = 0, voteMs = 0;
};

// The recipe's path for frame t: flows (shared with Temporal), the region widened by the flow, the centre and
// the neighbours traced with the lowered bars, the vote.
FrameVote voteFrame(const synth::LocksSequence& s, int t, int K, const vote::VoteParams& vp,
                    const tracer::TracerParams& tp, int margin = 40) {
  FrameVote fv;
  const int lo = std::max(t - K, 0), hi = std::min(t + K, s.frames - 1);
  std::vector<ml::TemporalFrame> fr;
  for (int i = lo; i <= hi; ++i) {
    ml::TemporalFrame f;
    f.rgb = s.rgb[i].data();
    f.matte = s.base[i].data();
    fr.push_back(f);
  }
  ml::StabiliseParams sp;
  sp.frames = K;
  double t0 = nowMs();
  const ml::TemporalFlows flows = ml::computeFlows(fr.data(), int(fr.size()), t - lo, s.w, s.h, sp);
  fv.flowMs = nowMs() - t0;
  t0 = nowMs();
  const Crop c = regionBox(s, t, margin);
  const std::vector<float> region = cropPlane(s.region[t], s.w, 1, c);
  std::vector<vote::DenseMotion> dm;
  std::vector<vote::Motion> mv;
  for (int i = lo; i <= hi; ++i)
    if (i != t) dm.push_back(vote::motionFromFlows(flows, i - t, c.x0, c.y0, c.w, c.h));
  for (const vote::DenseMotion& d : dm) mv.push_back(d.view());
  const std::vector<float> wide = vote::widenRegion(region.data(), c.w, c.h, mv.data(), int(mv.size()));
  const double prepMs = nowMs() - t0;
  t0 = nowMs();
  const tracer::TracerParams cp = vote::candidateTracerParams(tp);
  const tracer::TracerResult centre = trace(s, t, c, region, cp);
  std::vector<tracer::TracerResult> nres;
  for (int i = lo; i <= hi; ++i)
    if (i != t) nres.push_back(trace(s, i, c, wide, cp));
  fv.traceMs = nowMs() - t0;
  t0 = nowMs();
  std::vector<vote::Neighbour> nb(nres.size());
  for (size_t k = 0; k < nres.size(); ++k) {
    nb[k].result = &nres[k];
    nb[k].motion = mv[k];
  }
  vote::VoteInput in;
  in.width = c.w;
  in.height = c.h;
  in.region = region.data();
  in.widenedRegion = wide.data();
  const std::vector<float> baseC = cropPlane(s.base[t], s.w, 1, c);
  in.base = baseC.data();
  in.centre = &centre;
  in.neighbours = nb.data();
  in.count = int(nb.size());
  fv.r = vote::votePresence(in, vp);
  fv.voteMs = nowMs() - t0 + prepMs;
  fv.alpha.assign(size_t(s.w) * s.h, 0.0f);
  pastePlane(fv.alpha, s.w, fv.r.alpha, c);
  return fv;
}

std::vector<float> rawFrame(const synth::LocksSequence& s, int t, const tracer::TracerParams& tp, int margin = 40) {
  const Crop c = regionBox(s, t, margin);
  const tracer::TracerResult r = trace(s, t, c, cropPlane(s.region[t], s.w, 1, c), tp);
  std::vector<float> a(size_t(s.w) * s.h, 0.0f);
  pastePlane(a, s.w, r.alpha, c);
  return a;
}

// Mean frame-to-frame change over the pixels where mask(t, i) holds, the previous frame read at the true motion.
template <class Mask>
double flickerOf(const synth::LocksSequence& s, const std::vector<std::vector<float>>& a, Mask mask) {
  double sum = 0;
  long long cnt = 0;
  for (int t = 1; t < s.frames; ++t) {
    const synth::Affine back = synth::between(s.motion[t], s.motion[t - 1]);
    for (int y = 0; y < s.h; ++y)
      for (int x = 0; x < s.w; ++x) {
        const size_t i = size_t(y) * s.w + x;
        if (!mask(t, i)) continue;
        double qx, qy;
        back.apply(x, y, &qx, &qy);
        if (qx < 0 || qy < 0 || qx > s.w - 1 || qy > s.h - 1) continue;
        sum += std::fabs(a[t][i] - ml::sampleField(a[t - 1].data(), s.w, s.h, float(qx), float(qy)));
        ++cnt;
      }
  }
  return cnt ? sum / cnt : 0.0;
}

// Strand coverage: sum min(a, truth) / sum truth over the region (all frames, or one).
double coverage(const synth::LocksSequence& s, const std::vector<std::vector<float>>& a, int only = -1) {
  double num = 0, den = 0;
  for (int t = 0; t < s.frames; ++t) {
    if (only >= 0 && t != only) continue;
    for (size_t i = 0; i < a[t].size(); ++i)
      if (s.region[t][i] > 0) {
        num += std::min(a[t][i], s.strandTruth[t][i]);
        den += s.strandTruth[t][i];
      }
  }
  return den > 0 ? num / den : 0.0;
}

double falseAlpha(const synth::LocksSequence& s, const std::vector<std::vector<float>>& a) {
  double num = 0, den = 0;
  for (int t = 0; t < s.frames; ++t)
    for (size_t i = 0; i < a[t].size(); ++i) {
      num += std::max(a[t][i] - s.strandTruth[t][i], 0.0f) * (s.base[t][i] < 0.5f ? 1 : 0);
      den += s.strandTruth[t][i];
    }
  return den > 0 ? num / den : 0.0;
}

synth::LockFlicker injected() {
  synth::LockFlicker f;
  f.dipFrame = 4;
  f.vanishFrame = 8;
  f.spuriousFrame = 11;
  f.flowFailFrame = 14;
  return f;
}

std::vector<float> fuse(const std::vector<float>& base, const std::vector<float>& region, const std::vector<float>& cand) {
  std::vector<float> m(base.size());
  for (size_t i = 0; i < m.size(); ++i) {
    const float r = std::min(std::max(region[i], 0.0f), 1.0f);
    m[i] = base[i] + r * (std::max(base[i], cand[i]) - base[i]);
  }
  return m;
}

void testSequence(int sw, int sh, int frames, bool ab) {
  const int K = 2;
  const synth::LocksSequence s = synth::makeLocksSequence(sw, sh, frames, 1, injected(), 1.0);
  const tracer::TracerParams tp;
  vote::VoteParams vp;
  vote::setBars(vp, tp);
  std::vector<std::vector<float>> raw(frames), voted(frames), truth(frames);
  std::vector<FrameVote> fvs(frames);
  double flowMs = 0, traceMs = 0, voteMs = 0;
  for (int t = 0; t < frames; ++t) {
    raw[t] = rawFrame(s, t, tp);
    fvs[t] = voteFrame(s, t, K, vp, tp);
    voted[t] = fvs[t].alpha;
    truth[t] = s.strandTruth[t];
    flowMs += fvs[t].flowMs;
    traceMs += fvs[t].traceMs;
    voteMs += fvs[t].voteMs;
    const vote::VoteResult& r = fvs[t].r;
    std::printf("  t %2d: judged %d kept %d dropped %d reinstated %d%s%s%s | cov raw %.3f vote %.3f\n", t,
                r.judgedNeighbours, r.kept, r.dropped, r.reinstated, r.fallback ? " FALLBACK" : "",
                r.note.empty() ? "" : " | ", r.note.c_str(), coverage(s, raw, t), coverage(s, voted, t));
    if (std::getenv("VOTE_DEBUG"))
      for (const vote::PieceVote& v : r.pieces)
        std::printf("      piece %d: score %.2f (%d/%d) conf %.2f bar %.2f w %.1f -> %s\n", v.index, v.score, v.matched,
                    v.judged, v.confidence, v.bar, v.width, vote::fateName(v.fate));
  }
  auto inRegion = [&](int t, size_t i) { return s.region[t][i] > 0; };
  const double flT = flickerOf(s, truth, inRegion), flR = flickerOf(s, raw, inRegion), flV = flickerOf(s, voted, inRegion);
  const double cR = coverage(s, raw), cV = coverage(s, voted), fR = falseAlpha(s, raw), fV = falseAlpha(s, voted);
  std::printf("flicker table (%dx%d, %d frames, K=%d; strand alpha in the region):\n", sw, sh, frames, K);
  std::printf("  | candidate | flicker | coverage | false alpha |\n");
  std::printf("  | truth     | %.4f  | 1.000    | 0.000       |\n", flT);
  std::printf("  | raw tracer| %.4f  | %.3f    | %.3f       |\n", flR, cR, fR);
  std::printf("  | vote      | %.4f  | %.3f    | %.3f       |\n", flV, cV, fV);
  CHECK(flV < flR);
  CHECK(cV >= cR - 0.02);  // strands not erased
  // the injected frames
  const synth::LockFlicker fl = s.flicker;
  {  // spurious stroke: in the raw result, gone after the vote
    double a0 = 0, a1 = 0;
    for (size_t i = 0; i < s.spurious[fl.spuriousFrame].size(); ++i)
      if (s.spurious[fl.spuriousFrame][i]) a0 += raw[fl.spuriousFrame][i], a1 += voted[fl.spuriousFrame][i];
    std::printf("  spurious frame %d: alpha on the stroke raw %.1f, vote %.1f\n", fl.spuriousFrame, a0, a1);
    CHECK(a1 <= 0.1 * a0 + 1.0);
  }
  std::printf("  dip frame %d: coverage raw %.3f vote %.3f\n", fl.dipFrame, coverage(s, raw, fl.dipFrame),
              coverage(s, voted, fl.dipFrame));
  for (int f : {fl.dipFrame, fl.vanishFrame}) {  // a lost lock is put back; a lock the tracer kept stays
    const double r0 = coverage(s, raw, f), r1 = coverage(s, voted, f);
    CHECK(r1 >= r0 - 0.02);
    if (r0 < 0.8) CHECK(r1 > r0 + 0.1);
  }
  std::printf("  vanish frame %d: coverage raw %.3f vote %.3f\n", fl.vanishFrame, coverage(s, raw, fl.vanishFrame),
              coverage(s, voted, fl.vanishFrame));
  {  // flow failure: the frame falls back to its own result, its neighbours do not judge it
    const vote::VoteResult& r = fvs[fl.flowFailFrame].r;
    std::printf("  flow-failure frame %d: fallback %d, note \"%s\"; neighbours' notes: \"%s\" / \"%s\"\n",
                fl.flowFailFrame, int(r.fallback), r.note.c_str(), fvs[fl.flowFailFrame - 1].r.note.c_str(),
                fvs[fl.flowFailFrame + 1].r.note.c_str());
    CHECK(r.fallback);
    const double c0 = coverage(s, raw, fl.flowFailFrame), c1 = coverage(s, voted, fl.flowFailFrame);
    std::printf("  flow-failure frame coverage: own result %.3f, vote (fallback) %.3f\n", c0, c1);
    CHECK(std::fabs(c1 - c0) <= 0.02);  // the frame's own result
    CHECK(fvs[fl.flowFailFrame - 1].r.note.find("+1") != std::string::npos);
    CHECK(fvs[fl.flowFailFrame + 1].r.note.find("-1") != std::string::npos);
  }
  std::printf("  per frame (mean): flows %.1f ms, tracer x%d %.1f ms, vote %.2f ms\n", flowMs / frames, 2 * K + 1,
              traceMs / frames, voteMs / frames);

  if (!ab) return;
  // Q3 A/B: the strands fused into the base (never lower, region weight), then Temporal (K = 2) with the
  // neighbours' RAW base mattes, as render_chain does.
  std::vector<std::vector<float>> fusedRaw(frames), fusedVote(frames);
  for (int t = 0; t < frames; ++t) {
    fusedRaw[t] = fuse(s.base[t], s.region[t], raw[t]);
    fusedVote[t] = fuse(s.base[t], s.region[t], voted[t]);
  }
  auto temporal = [&](const std::vector<std::vector<float>>& fused, bool outsideOnly) {
    std::vector<std::vector<float>> out(frames);
    for (int t = 0; t < frames; ++t) {
      const int lo = std::max(t - K, 0), hi = std::min(t + K, frames - 1);
      std::vector<ml::TemporalFrame> fr;
      for (int i = lo; i <= hi; ++i) {
        ml::TemporalFrame f;
        f.rgb = s.rgb[i].data();
        f.matte = i == t ? fused[t].data() : s.base[i].data();
        fr.push_back(f);
      }
      ml::StabiliseParams sp;
      sp.frames = K;
      std::vector<float> wgt;
      if (outsideOnly) {
        wgt.resize(fused[t].size());
        for (size_t i = 0; i < wgt.size(); ++i) wgt[i] = 1.0f - std::min(std::max(s.region[t][i], 0.0f), 1.0f);
      }
      out[t].resize(fused[t].size());
      ml::stabilise(fr.data(), int(fr.size()), t - lo, s.w, s.h, sp, out[t].data(), nullptr,
                    outsideOnly ? wgt.data() : nullptr);
    }
    return out;
  };
  std::vector<std::vector<float>> headTruth(frames);
  for (int t = 0; t < frames; ++t) {
    headTruth[t].resize(s.headDistance[t].size());
    for (size_t i = 0; i < headTruth[t].size(); ++i) headTruth[t][i] = std::min(std::max(s.headDistance[t][i] + 0.5f, 0.0f), 1.0f);
  }
  auto edgeOutside = [&](int t, size_t i) { return std::fabs(s.headDistance[t][i]) < 3.0f && !(s.region[t][i] > 0); };
  struct Row {
    const char* name;
    std::vector<std::vector<float>> m;
  };
  std::vector<Row> rows;
  rows.push_back({"(i) no Temporal, raw tracer", fusedRaw});
  rows.push_back({"(ii) Temporal everywhere, raw tracer", temporal(fusedRaw, false)});
  rows.push_back({"(ii') Temporal everywhere + vote", temporal(fusedVote, false)});
  rows.push_back({"(iii) Temporal outside region + vote", temporal(fusedVote, true)});
  rows.push_back({"(iv) no Temporal, vote", fusedVote});
  std::printf("Q3 A/B (424x562, %d frames, base edge jitter 1 px, K=2; coverage / flicker of the fused matte):\n", frames);
  std::printf("  | variant | strand coverage | strand flicker (region) | edge flicker outside region |\n");
  std::printf("  | truth | 1.000 | %.4f | %.4f |\n", flickerOf(s, truth, inRegion), flickerOf(s, headTruth, edgeOutside));
  std::vector<double> cov;
  for (const Row& r : rows) {
    cov.push_back(coverage(s, r.m));
    std::printf("  | %s | %.3f | %.4f | %.4f |\n", r.name, cov.back(), flickerOf(s, r.m, inRegion),
                flickerOf(s, r.m, edgeOutside));
  }
  CHECK(cov[3] >= cov[4] - 0.02);  // Temporal outside the region does not erase the rebuilt strands
}

// Timing per frame (vote only, and the tracer runs it needs) at 424x562 and at 2048x1152 (crops).
void testTiming() {
  const int K = 2;
  const tracer::TracerParams tp;
  vote::VoteParams vp;
  vote::setBars(vp, tp);
  const int sizes[2][2] = {{424, 562}, {2048, 1152}};
  for (const auto& sz : sizes) {
    const synth::LocksSequence s = synth::makeLocksSequence(sz[0], sz[1], 7, 3);
    double fl = 0, tr = 0, vo = 0;
    int n = 0;
    for (int t = 2; t <= 4; ++t, ++n) {
      const FrameVote fv = voteFrame(s, t, K, vp, tp);
      fl += fv.flowMs;
      tr += fv.traceMs;
      vo += fv.voteMs;
    }
    std::printf("timing %dx%d, 2 shapes, K=%d: vote %.2f ms (incl. motion sampling + widening), tracer x5 %.1f ms, flows %.1f ms\n",
                sz[0], sz[1], K, vo / n, tr / n, fl / n);
    CHECK(vo / n < 100.0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool seq = argc > 1 && std::string(argv[1]) == "--sequence";
  if (seq) {
    testSequence(424, 562, 16, true);
    testSequence(1920, 1080, 16, false);
    testTiming();
  } else {
    testKnownAnswers();
    testMutations();
    testRegionZero();
    testDeterminism();
    testGenerator();
    testBlend();
    testMovedRegion();
  }
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
