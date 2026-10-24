// The plugin's render chain without OFX (render_chain.h): Smart Fill's params from the node's
// controls (defaults, Max Hole %, render scale), RunnerSeg through the fake runner (mode seg: the
// whole transport, prepare / postprocess, one call per frame, none without a hole), the chain order
// (Smart Fill before Edge Refine, against the cores run by hand), every failure passing the matte
// through its step only, both steps off = the input bit for bit, Smart Fill off = Edge Refine alone
// (the build before B3), abort, and the Holes view (marks against a brute-force hole finder, the
// colours). Temporal (B4): params and defaults, the neighbour offsets / frames-needed range at the
// sequence ends, the neighbour collection with a fake fetcher (missing, threw, wrong size, not float,
// no alpha, abort), the chain order (Temporal first, against stabilise() by hand), no neighbours =
// the input, failure and abort pass-through, Temporal off = the previous chain, and the flicker on
// the synthetic sequence through the chain. Usage: test_render_chain <fake runner>.
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_table.h"
#include "ops/edge_field.h"
#include "render_chain.h"
#include "runner_client.h"
#include "smart_fill_ref.h"
#include "synth.h"

using namespace mr::ml;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

std::string gFake, gDir, gSock, gHome;

void writeFile(const std::string& name, const std::string& text) { std::ofstream(gDir + "/" + name) << text; }

RunnerClientOptions opts() {
  RunnerClientOptions o;
  o.runnerPath = gFake;
  o.socketPath = gSock;
  o.name = "chain";
  o.tileTimeoutS = 5;
  o.startTimeoutS = 5;
  o.idleSeconds = 60;
  o.env = {"HOME=" + gHome};
  return o;
}

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// A frame: background (matte 0, blue) with a solid subject (matte 1, red) in a rectangle; inside it
// a defect hole (matte 0 / 0.2, the picture red: subject) and a real gap (matte 0, the picture blue),
// a hole with a soft rim (0.6) around a defect, and a notch from the subject's edge to the
// background (never a hole). The fake segmentation (mode seg) says subject where the picture is red.
struct Frame {
  int w, h;
  std::vector<float> m, rgb;
  Frame(int w_, int h_) : w(w_), h(h_), m(size_t(w) * h, 0.0f), rgb(size_t(w) * h * 3) {
    rect(w / 8, h / 8, w * 6 / 8, h * 6 / 8, 1.0f);
    rect(w / 4, h / 4, w / 16, h / 10, 0.0f);            // defect A (value 0)
    rect(w / 4 - 1, h / 2 - 1, w / 16 + 2, h / 10 + 2, 0.6f);  // defect B: a rim at 0.6 ...
    rect(w / 4, h / 2, w / 16, h / 10, 0.2f);            // ... around a core at 0.2
    rect(w * 5 / 8, h / 3, w / 12, h / 6, 0.0f);         // real gap
    rect(w / 2, h / 8, 3, h / 8, 0.0f);                  // notch from the top edge of the subject
    for (size_t i = 0; i < m.size(); ++i) paint(i, true);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const bool inSubject = x >= w / 8 && x < w / 8 + w * 6 / 8 && y >= h / 8 && y < h / 8 + h * 6 / 8;
        const bool inGap = x >= w * 5 / 8 && x < w * 5 / 8 + w / 12 && y >= h / 3 && y < h / 3 + h / 6;
        const bool inNotch = x >= w / 2 && x < w / 2 + 3 && y >= h / 8 && y < h / 4;
        paint(i, inSubject && !inGap && !inNotch);
      }
  }
  void rect(int x0, int y0, int rw, int rh, float v) {
    for (int y = y0; y < y0 + rh; ++y)
      for (int x = x0; x < x0 + rw; ++x) m[size_t(y) * w + x] = v;
  }
  void paint(size_t i, bool subject) {
    rgb[i * 3] = subject ? 0.8f : 0.1f;
    rgb[i * 3 + 1] = 0.2f;
    rgb[i * 3 + 2] = subject ? 0.1f : 0.7f;
  }
};

// In-process fakes.
struct MaskSeg : SegRunner {
  std::vector<float> mask;
  int calls = 0;
  void segment(const float*, int, int, float* out) override {
    ++calls;
    std::copy(mask.begin(), mask.end(), out);
  }
};
struct ThrowSeg : SegRunner {
  int calls = 0;
  void segment(const float*, int, int, float*) override {
    ++calls;
    throw std::runtime_error("no segmentation today");
  }
};
// ViTMatte-shaped: alpha = 0.1 + 0.8 * trimap (plane 3), a function of the tile's input only.
struct FakeTiles : TileRunner {
  int calls = 0;
  bool fail = false;
  void run(const float* input, int T, float* alpha, int, int) override {
    ++calls;
    if (fail) throw std::runtime_error("tile failed");
    const size_t plane = size_t(T) * T;
    for (size_t i = 0; i < plane; ++i) alpha[i] = 0.1f + 0.8f * input[3 * plane + i];
  }
};

// The fake runner's segmentation, computed here (fake_runner.cpp segLogit, the same expression).
std::vector<float> fakeSegReference(const std::vector<float>& rgb, int w, int h) {
  const int S = kBiRefNetSize;
  std::vector<float> in(size_t(3) * S * S), logits(size_t(S) * S), mask(size_t(w) * h);
  birefnetPrepare(rgb.data(), w, h, in.data());
  for (size_t i = 0; i < logits.size(); ++i) logits[i] = 40.0f * (in[i] * 0.229f + 0.485f - 0.5f);
  birefnetPostprocess(logits.data(), w, h, mask.data());
  return mask;
}

void testParams() {
  const SmartFillUi ui;  // the node's defaults (plugin_ml.cpp describe uses the same values)
  CHECK(ui.subjectThreshold == 0.6 && ui.maxHolePercent == 10.0 && ui.segmentation == kSegLite);
  const SmartFillParams core;
  const SmartFillParams p = smartFillParams(ui, 1.0);
  CHECK(p.subjectThreshold == 0.6 && p.maxHoleFraction == 0.1);
  CHECK(p.mode == ScoreMode::Both && p.segStat == SegStatistic::Mean && p.keepRealGaps);
  CHECK(p.solidAbove == core.solidAbove && p.rimPx == core.rimPx && p.ringPx == core.ringPx &&
        p.bgReachPx == core.bgReachPx);
  CHECK(core.rimPx == 1.0 && core.ringPx == 4.0 && core.bgReachPx == 64.0);
  // Decision 6: pixel params scale with the render scale (as Fill Holes' Rim px in plugin 1).
  const SmartFillParams h = smartFillParams(ui, 0.5);
  CHECK(h.rimPx == 0.5 && h.ringPx == 2.0 && h.bgReachPx == 32.0 && h.maxHoleFraction == 0.1 &&
        h.subjectThreshold == 0.6 && h.solidAbove == core.solidAbove);
  const SmartFillParams q = smartFillParams(ui, 0.25);
  CHECK(q.rimPx == 0.25 && q.ringPx == 1.0 && q.bgReachPx == 16.0);
  SmartFillUi u2;
  u2.maxHolePercent = 0.0;  // no limit
  u2.subjectThreshold = 0.85;
  CHECK(smartFillParams(u2, 1.0).maxHoleFraction == 0.0 && smartFillParams(u2, 1.0).subjectThreshold == 0.85);
  u2.maxHolePercent = 2.5;
  CHECK(std::fabs(smartFillParams(u2, 1.0).maxHoleFraction - 0.025) < 1e-15);
  // Segmentation: Lite / Full = the table's BiRefNet rows (3 -> 1 at 1024 x 1024).
  CHECK(std::string(segmentationModel(kSegLite)) == "birefnet-lite");
  CHECK(std::string(segmentationModel(kSegFull)) == "birefnet");
  for (int s : {kSegLite, kSegFull}) {
    const ModelSpec* m = findModel(segmentationModel(s));
    CHECK(m && m->channelsIn == 3 && m->channelsOut == 1 && m->sizeRule == SizeRule::kFixed &&
          m->fixedW == kBiRefNetSize && m->fixedH == kBiRefNetSize);
  }
  std::printf("params: ok\n");
}

void testRunnerSeg() {
  writeFile("fake.mode", "seg");
  RunnerClient c(opts());
  const Frame f(320, 180);
  RunnerSeg seg(c, "birefnet-lite");
  std::vector<float> mask(size_t(f.w) * f.h, -1.0f);
  seg.segment(f.rgb.data(), f.w, f.h, mask.data());
  const std::vector<float> want = fakeSegReference(f.rgb, f.w, f.h);
  CHECK(sameBits(mask, want));  // the whole 1024 x 1024 frame went through the runner and back
  CHECK(seg.calls == 1 && c.stats().tiles == 1);
  // Away from the colour boundaries the mask is the subject.
  CHECK(mask[size_t(f.h / 2) * f.w + f.w / 8 + 10] > 0.99f && mask[size_t(5) * f.w + 5] < 0.01f);
  CHECK(mask[size_t(f.h / 3 + f.h / 12) * f.w + f.w * 5 / 8 + f.w / 24] < 0.01f);  // the real gap
  // Full: the other row, same shapes.
  RunnerSeg full(c, "birefnet");
  std::vector<float> m2(mask.size());
  full.segment(f.rgb.data(), f.w, f.h, m2.data());
  CHECK(sameBits(m2, want) && c.stats().tiles == 2);
  // A model of another shape is refused by the runner: an exception (the footer text).
  RunnerSeg wrong(c, "vitmatte-small");
  std::string e;
  try {
    wrong.segment(f.rgb.data(), f.w, f.h, m2.data());
  } catch (const std::exception& x) {
    e = x.what();
  }
  CHECK(e.find("4 input channels") != std::string::npos);
  std::printf("runner seg: ok (prep %.1f run %.1f post %.1f ms; refused: '%s')\n", seg.prepMs, seg.runMs, seg.postMs,
              e.c_str());
}

// Smart Fill through the fake runner, Edge Refine off: the defects filled, the real gap kept, all else
// bit-identical; the runner called once; never without a hole.
void testSmartFillThroughRunner() {
  writeFile("fake.mode", "seg");
  RunnerClient c(opts());
  const Frame f(320, 180);
  const size_t n = f.m.size();
  RunnerSeg seg(c, "birefnet-lite");
  FakeTiles tiles;
  ChainParams p;
  p.smartFill = true;
  std::vector<float> mid, out;
  ChainResult r;
  runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg, tiles, &mid, &out, &r);
  CHECK(r.sfRan && !r.sfFailed && !r.erRan && tiles.calls == 0 && seg.calls == 1 && c.stats().tiles == 1);
  CHECK(r.sf.holesFound == 3 && r.sf.holesFilled == 2 && r.sf.holesKept == 1 && r.sf.holesTooLarge == 0);
  CHECK(sameBits(mid, out));
  const auto holes = sfref::findHoles(f.m.data(), f.w, f.h, 0.95f);
  CHECK(holes.size() == 3);
  std::vector<uint8_t> inHole(n, 0);
  for (size_t k = 0; k < holes.size(); ++k)
    for (int i : holes[k].px) inHole[size_t(i)] = uint8_t(k + 1);
  long filled = 0, kept = 0, other = 0, rim = 0;
  for (size_t i = 0; i < n; ++i) {
    const int x = int(i % size_t(f.w)), y = int(i / size_t(f.w));
    const bool gap = x >= f.w * 5 / 8 && x < f.w * 5 / 8 + f.w / 12 && y >= f.h / 3 && y < f.h / 3 + f.h / 6;
    if (inHole[i] && !gap) filled += out[i] == 1.0f;
    else if (gap) kept += out[i] == f.m[i];
    else if (out[i] != f.m[i]) (f.m[i] >= 0.95f ? rim : other)++;  // only solid pixels next to a hole may change
  }
  long defectPx = 0, gapPx = 0;
  for (size_t i = 0; i < n; ++i) {
    const int x = int(i % size_t(f.w)), y = int(i / size_t(f.w));
    const bool gap = x >= f.w * 5 / 8 && x < f.w * 5 / 8 + f.w / 12 && y >= f.h / 3 && y < f.h / 3 + f.h / 6;
    defectPx += inHole[i] && !gap;
    gapPx += gap;
  }
  CHECK(filled == defectPx && kept == gapPx && other == 0 && rim == 0);  // the rim pixels are 1.0 already
  // The matte outside the holes (and their 1 px rim) is the input bit for bit.
  mr::Mask holeMask(n, 0);
  for (size_t i = 0; i < n; ++i) holeMask[i] = inHole[i] != 0;
  const mr::Mask near = mr::dilate(holeMask, f.w, f.h, 1.5);
  long outside = 0;
  for (size_t i = 0; i < n; ++i)
    if (!near[i]) outside += std::memcmp(&out[i], &f.m[i], 4) != 0;
  CHECK(outside == 0);
  // No hole: the runner is not called.
  std::vector<float> plain(n, 0.0f);
  for (int y = 20; y < 160; ++y)
    for (int x = 40; x < 280; ++x) plain[size_t(y) * f.w + x] = 1.0f;
  c.resetStats();
  RunnerSeg seg2(c, "birefnet-lite");
  runChain(plain.data(), f.rgb.data(), f.w, f.h, p, seg2, tiles, &mid, &out, &r);
  CHECK(r.sfRan && !r.sfFailed && r.sf.holesFound == 0 && !r.sf.segmented && seg2.calls == 0 && c.stats().tiles == 0);
  CHECK(sameBits(out, plain));
  // Every hole too large: no runner call either, nothing filled.
  p.sf.maxHolePercent = 0.01;
  RunnerSeg seg3(c, "birefnet-lite");
  runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg3, tiles, &mid, &out, &r);
  CHECK(r.sf.holesTooLarge == 3 && seg3.calls == 0 && sameBits(out, f.m));
  std::printf("smart fill through the runner: ok (%ld defect px filled, %ld gap px kept)\n", filled, kept);
}

// The order: Smart Fill's output is Edge Refine's input; both against the cores run by hand.
void testChainOrder() {
  const Frame f(320, 180);
  const size_t n = f.m.size();
  MaskSeg seg;
  seg.mask = fakeSegReference(f.rgb, f.w, f.h);
  FakeTiles tiles;
  ChainParams p;
  p.smartFill = true;
  p.edgeRefine = true;
  p.er.bandPx = 6;
  std::vector<float> mid, out, trimap;
  ChainResult r;
  runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg, tiles, &mid, &out, &r, std::function<bool()>(), &trimap);
  CHECK(r.sfRan && r.erRan && !r.sfFailed && !r.erFailed && seg.calls == 1 && tiles.calls > 0);
  // By hand: smartFill, then refineEdge on its output.
  std::vector<float> sf(n), want(n), trimapWant;
  FakeTiles t2;
  smartFill(f.m.data(), f.rgb.data(), f.w, f.h, smartFillParams(p.sf, 1.0), seg, sf.data());
  refineEdge(sf.data(), f.rgb.data(), want.data(), f.w, f.h, p.er, t2, nullptr, std::function<bool()>(), &trimapWant);
  CHECK(sameBits(mid, sf) && sameBits(out, want) && sameBits(trimap, trimapWant));
  // The checks above can tell the orders apart: Edge Refine on the input sees another trimap (a band
  // around every defect) and gives another matte. (Edge Refine then Smart Fill happens to give the
  // same matte with this fake model: the fill covers the defects' bands.)
  std::vector<float> er(n), sd, trimapIn;
  refineEdge(f.m.data(), f.rgb.data(), er.data(), f.w, f.h, p.er, t2, nullptr);
  buildTrimap(f.m.data(), f.w, f.h, p.er.bandPx, &sd, &trimapIn);
  CHECK(!sameBits(trimapIn, trimapWant) && !sameBits(er, want));
  // between() runs once, after Smart Fill and before Edge Refine.
  int seen = 0, tilesAtBetween = -1, segAtBetween = -1;
  FakeTiles t3;
  MaskSeg s3;
  s3.mask = seg.mask;
  runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, s3, t3, &mid, &out, &r, std::function<bool()>(), nullptr, [&]() {
    ++seen;
    tilesAtBetween = t3.calls;
    segAtBetween = s3.calls;
  });
  CHECK(seen == 1 && tilesAtBetween == 0 && segAtBetween == 1 && t3.calls > 0);
  std::printf("chain order: ok (%d tiles)\n", tiles.calls);
}

void testOffAndFailures() {
  const Frame f(320, 180);
  const size_t n = f.m.size();
  MaskSeg seg;
  seg.mask = fakeSegReference(f.rgb, f.w, f.h);
  std::vector<float> mid, out;
  ChainResult r;
  // Both off: the input, bit for bit; no model touched.
  {
    FakeTiles tiles;
    ChainParams p;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg, tiles, &mid, &out, &r);
    CHECK(!r.sfRan && !r.erRan && seg.calls == 0 && tiles.calls == 0 && sameBits(out, f.m) && sameBits(mid, f.m));
  }
  // Smart Fill off, Edge Refine on: Edge Refine alone on the input (the build before B3).
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.edgeRefine = true;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg, tiles, &mid, &out, &r);
    std::vector<float> want(n);
    refineEdge(f.m.data(), f.rgb.data(), want.data(), f.w, f.h, p.er, t2, nullptr);
    CHECK(!r.sfRan && r.erRan && seg.calls == 0 && sameBits(out, want) && sameBits(mid, f.m));
  }
  // Smart Fill fails: its step passes the matte through, Edge Refine still runs on the input.
  {
    ThrowSeg bad;
    FakeTiles tiles, t2;
    ChainParams p;
    p.smartFill = true;
    p.edgeRefine = true;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, bad, tiles, &mid, &out, &r);
    std::vector<float> want(n);
    refineEdge(f.m.data(), f.rgb.data(), want.data(), f.w, f.h, p.er, t2, nullptr);
    CHECK(r.sfRan && r.sfFailed && r.sfError == "no segmentation today" && bad.calls == 1);
    CHECK(sameBits(mid, f.m) && r.erRan && !r.erFailed && sameBits(out, want));
  }
  // Through the fake runner: a model error is the footer text, the step passes through.
  {
    writeFile("fake.mode", "error");
    RunnerClient c(opts());
    RunnerSeg rs(c, "birefnet-lite");
    FakeTiles tiles;
    ChainParams p;
    p.smartFill = true;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, rs, tiles, &mid, &out, &r);
    CHECK(r.sfFailed && r.sfError.find("fake model failure") != std::string::npos && sameBits(out, f.m));
    std::printf("runner failure footer text: '%s'\n", r.sfError.c_str());
    writeFile("fake.mode", "seg");
  }
  // No runner at all (not found): the same.
  {
    RunnerClientOptions o = opts();
    o.runnerPath = gDir + "/no-such-runner";
    o.socketPath = gDir + "/other.sock";
    RunnerClient c(o);
    RunnerSeg rs(c, "birefnet-lite");
    FakeTiles tiles;
    ChainParams p;
    p.smartFill = true;
    p.edgeRefine = true;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, rs, tiles, &mid, &out, &r);
    CHECK(r.sfFailed && r.sfError.find("runner not found") != std::string::npos && sameBits(mid, f.m) && r.erRan &&
          !r.erFailed);
  }
  // Edge Refine fails: its step passes Smart Fill's output through; Smart Fill's work stays.
  {
    FakeTiles tiles;
    tiles.fail = true;
    ChainParams p;
    p.smartFill = true;
    p.edgeRefine = true;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, seg, tiles, &mid, &out, &r);
    std::vector<float> sf(n);
    smartFill(f.m.data(), f.rgb.data(), f.w, f.h, smartFillParams(p.sf, 1.0), seg, sf.data());
    CHECK(!r.sfFailed && r.erFailed && r.erError == "tile failed" && sameBits(mid, sf) && sameBits(out, sf));
    CHECK(!sameBits(sf, f.m));
  }
  // Abort: Smart Fill returns its input (the core asks before and after the model).
  {
    FakeTiles tiles;
    ChainParams p;
    p.smartFill = true;
    p.edgeRefine = true;
    int asked = 0;
    MaskSeg s2;
    s2.mask = seg.mask;
    runChain(f.m.data(), f.rgb.data(), f.w, f.h, p, s2, tiles, &mid, &out, &r, [&]() { return ++asked >= 2; });
    CHECK(r.sf.aborted && s2.calls == 1 && sameBits(mid, f.m) && r.er.aborted && sameBits(out, f.m));
  }
  std::printf("off and failures: ok\n");
}

void testHolesView() {
  // Colours.
  float c[3];
  holesViewRgb(kMarkNone, 0.3f, c);
  CHECK(c[0] == 0.3f && c[1] == 0.3f && c[2] == 0.3f);
  holesViewRgb(kMarkFilled, 0.3f, c);
  CHECK(c[0] == 0.0f && c[1] == 1.0f && c[2] == 0.0f);
  holesViewRgb(kMarkKept, 0.3f, c);
  CHECK(c[0] == 1.0f && c[1] == 0.0f && c[2] == 0.0f);
  // Changes view = plugin 1's Show Changes: unchanged grey, raised magenta, lowered green.
  changesViewRgb(0.4f, 0.4f, c);
  CHECK(c[0] == 0.4f && c[1] == 0.4f && c[2] == 0.4f);
  changesViewRgb(0.0f, 1.0f, c);
  CHECK(c[0] == 1.0f && c[1] == 0.0f && c[2] == 1.0f);
  changesViewRgb(1.0f, 0.0f, c);
  CHECK(c[0] == 0.0f && c[1] == 1.0f && c[2] == 0.0f);
  // Marks against the brute-force holes and the core's decisions, on the test frame.
  const Frame f(320, 180);
  MaskSeg seg;
  seg.mask = fakeSegReference(f.rgb, f.w, f.h);
  SmartFillStats st;
  std::vector<float> out(f.m.size());
  smartFill(f.m.data(), f.rgb.data(), f.w, f.h, SmartFillParams(), seg, out.data(), &st);
  std::vector<uint8_t> marks;
  markHoles(f.m.data(), f.w, f.h, 0.95f, st, &marks);
  const auto holes = sfref::findHoles(f.m.data(), f.w, f.h, 0.95f);
  CHECK(holes.size() == st.holes.size() && holes.size() == 3);
  std::vector<uint8_t> want(f.m.size(), kMarkNone);
  for (size_t k = 0; k < holes.size() && k < st.holes.size(); ++k) {
    CHECK(holes[k].px.front() == st.holes[k].sy * f.w + st.holes[k].sx);  // the seed is the first pixel
    const uint8_t m = st.holes[k].decision == HoleDecision::Filled ? kMarkFilled : kMarkKept;
    for (int i : holes[k].px) want[size_t(i)] = m;
  }
  CHECK(marks == want);
  long nf = 0, nk = 0;
  for (uint8_t m : marks) nf += m == kMarkFilled, nk += m == kMarkKept;
  CHECK(nf > 0 && nk > 0);
  // A hole kept as too large is not marked (neither green nor red).
  SmartFillParams small;
  small.maxHoleFraction = 0.001;
  smartFill(f.m.data(), f.rgb.data(), f.w, f.h, small, seg, out.data(), &st);
  markHoles(f.m.data(), f.w, f.h, 0.95f, st, &marks);
  long marked = 0;
  for (size_t k = 0; k < st.holes.size(); ++k)
    if (st.holes[k].decision == HoleDecision::KeptTooLarge)
      for (int i : holes[k].px) marked += marks[size_t(i)] != kMarkNone;
  CHECK(st.holesTooLarge > 0 && marked == 0);
  // No stats (Smart Fill off): no marks.
  markHoles(f.m.data(), f.w, f.h, 0.95f, SmartFillStats(), &marks);
  CHECK(marks.size() == f.m.size() && std::count(marks.begin(), marks.end(), uint8_t(kMarkNone)) == long(marks.size()));
  // Cost at HD: the frame scaled up.
  const Frame hd(1920, 1080);
  MaskSeg sh;
  sh.mask = fakeSegReference(hd.rgb, hd.w, hd.h);
  std::vector<float> o2(hd.m.size());
  smartFill(hd.m.data(), hd.rgb.data(), hd.w, hd.h, SmartFillParams(), sh, o2.data(), &st);
  const auto t0 = std::chrono::steady_clock::now();
  markHoles(hd.m.data(), hd.w, hd.h, 0.95f, st, &marks);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const auto hh = sfref::findHoles(hd.m.data(), hd.w, hd.h, 0.95f);
  long hdMarked = 0, hdHole = 0;
  for (const auto& hole : hh) hdHole += long(hole.px.size());
  for (uint8_t m : marks) hdMarked += m != kMarkNone;
  CHECK(hdMarked == hdHole && st.holesFilled == 2 && st.holesKept == 1);
  std::printf("holes view: ok (HD marks %.1f ms, %ld px)\n", ms, hdMarked);
}

// --- Temporal (B4) ------------------------------------------------------------------------------

void testTemporalParams() {
  const TemporalUi ui;  // the node's defaults (plugin_ml.cpp describe uses the same values)
  CHECK(ui.frames == kTemporal1 && ui.strength == 1.0 && ui.bandPx == 6.0);
  CHECK(temporalK(kTemporal1) == 1 && temporalK(kTemporal2) == 2 && temporalK(7) == 1 && temporalK(-1) == 1);
  const StabiliseParams core;
  const StabiliseParams p = temporalParams(ui, 1.0);
  CHECK(p.frames == 1 && p.strength == 1.0 && p.bandPx == 6.0);
  CHECK(p.fillHoles == core.fillHoles && p.combine == core.combine && p.subjectWeighted == core.subjectWeighted &&
        p.flowLines == core.flowLines && p.fbMaxPx == core.fbMaxPx && p.flow.radius == core.flow.radius);
  CHECK(core.fillHoles && core.combine == Combine::Median && core.bandPx == 6.0);
  TemporalUi u2;
  u2.frames = kTemporal2;
  u2.strength = 0.4;
  u2.bandPx = 10.0;
  const StabiliseParams h = temporalParams(u2, 0.5);  // decision 6: the band scales with the render scale
  CHECK(h.frames == 2 && h.strength == 0.4 && h.bandPx == 5.0);
  u2.strength = 1.5;
  CHECK(temporalParams(u2, 1.0).strength == 1.0);
  u2.strength = -1;
  CHECK(temporalParams(u2, 1.0).strength == 0.0);
  std::printf("temporal params: ok\n");
}

void testNeighbourOffsets() {
  FrameRange r;
  r.first = 0;
  r.last = 9;
  r.valid = true;
  CHECK((neighbourOffsets(5, 1, r) == std::vector<int>{-1, 1}));
  CHECK((neighbourOffsets(5, 2, r) == std::vector<int>{-1, 1, -2, 2}));
  CHECK((neighbourOffsets(0, 2, r) == std::vector<int>{1, 2}));
  CHECK((neighbourOffsets(1, 2, r) == std::vector<int>{-1, 1, 2}));
  CHECK((neighbourOffsets(9, 2, r) == std::vector<int>{-1, -2}));
  CHECK((neighbourOffsets(8, 1, r) == std::vector<int>{-1, 1}));
  CHECK(neighbourOffsets(5, 0, r).empty());
  FrameRange one;
  one.first = one.last = 3;
  one.valid = true;
  CHECK(neighbourOffsets(3, 2, one).empty());
  FrameRange unknown;  // not reported: no clamping
  CHECK((neighbourOffsets(0, 1, unknown) == std::vector<int>{-1, 1}));
  FrameRange flame;  // a clip starting at frame 1001
  flame.first = 1001;
  flame.last = 1100;
  flame.valid = true;
  CHECK((neighbourOffsets(1001, 1, flame) == std::vector<int>{1}));
  double lo = 0, hi = 0;
  framesNeededRange(5, 2, r, &lo, &hi);
  CHECK(lo == 3 && hi == 7);
  framesNeededRange(0, 2, r, &lo, &hi);
  CHECK(lo == 0 && hi == 2);
  framesNeededRange(9, 1, r, &lo, &hi);
  CHECK(lo == 8 && hi == 9);
  framesNeededRange(3, 2, one, &lo, &hi);
  CHECK(lo == 3 && hi == 3);
  framesNeededRange(3, 2, unknown, &lo, &hi);
  CHECK(lo == 1 && hi == 5);
  framesNeededRange(1100, 2, flame, &lo, &hi);
  CHECK(lo == 1098 && hi == 1100);
  std::printf("neighbour offsets / frames needed: ok\n");
}

// A small jittering sequence (synth P4) and its frames as fetched neighbours.
struct Seq {
  int w = 480, h = 270;
  synth::Sequence s;
  Seq() : s(synth::makeSequence(480, 270, 7, 3)) {}
  FetchedFrame fetched(int t) const {
    FetchedFrame f;
    f.image = true;
    f.isFloat = true;
    f.components = 4;
    f.w = w;
    f.h = h;
    f.rgb = s.frames[size_t(t)].rgb;
    f.matte = s.frames[size_t(t)].matte;
    return f;
  }
  const float* m(int t) const { return s.frames[size_t(t)].matte.data(); }
  const float* rgb(int t) const { return s.frames[size_t(t)].rgb.data(); }
  // stabilise() by hand on frames[a..b] (consecutive), centre c.
  std::vector<float> byHand(const std::vector<int>& ts, int centreIndex, const StabiliseParams& p) const {
    std::vector<TemporalFrame> v;
    for (int t : ts) v.push_back({rgb(t), m(t)});
    std::vector<float> out(size_t(w) * h);
    stabilise(v.data(), int(v.size()), centreIndex, w, h, p, out.data());
    return out;
  }
};

void testCollectNeighbours() {
  const Seq q;
  FrameRange r;
  r.first = 0;
  r.last = 6;
  r.valid = true;
  // Per requested time: what the fake host returns.
  enum Kind { kOk, kNull, kThrow, kSmall, kInt, kRgb, kEmpty };
  std::vector<double> asked;
  auto fetcherFor = [&](std::vector<Kind> kinds) {  // kinds by time 0..6
    return [&, kinds](double time, FetchedFrame* f) {
      asked.push_back(time);
      const int t = int(std::lround(time));
      switch (kinds[size_t(t)]) {
        case kOk: *f = q.fetched(t); break;
        case kNull: break;
        case kThrow: f->error = "kOfxStatFailed"; break;
        case kSmall: *f = q.fetched(t), f->w = q.w / 2; break;
        case kInt: f->image = true, f->components = 4, f->w = q.w, f->h = q.h; break;
        case kRgb: f->image = true, f->isFloat = true, f->components = 3, f->w = q.w, f->h = q.h; break;
        case kEmpty: *f = q.fetched(t), f->matte.clear(); break;
      }
      f->ms = 1.5;
      f->note = "uid " + std::to_string(t);
    };
  };
  Neighbours nb;
  // All there, K = 2 at t = 3: nearest first, kept ascending.
  collectNeighbours(3, 2, r, q.w, q.h, fetcherFor({kOk, kOk, kOk, kOk, kOk, kOk, kOk}), &nb);
  CHECK((asked == std::vector<double>{2, 4, 1, 5}));
  CHECK(nb.requested == 4 && (nb.offsets == std::vector<int>{-2, -1, 1, 2}) && nb.frames.size() == 4 && !nb.aborted);
  CHECK(nb.frames[0].matte == q.s.frames[1].matte && nb.frames[3].rgb == q.s.frames[5].rgb);
  CHECK(nb.report.size() == 4 && nb.report[0].offset == -1 && nb.report[0].time == 2 && nb.report[0].used &&
        nb.report[0].status == "used" && nb.report[0].ms == 1.5 && nb.report[0].note == "uid 2");
  // Every failure kind: none used, each with its reason.
  asked.clear();
  collectNeighbours(3, 2, r, q.w, q.h, fetcherFor({kOk, kNull, kThrow, kOk, kSmall, kInt, kOk}), &nb);
  CHECK(nb.requested == 4 && nb.offsets.empty() && nb.frames.empty());
  CHECK(nb.report[0].status == "threw: kOfxStatFailed" && !nb.report[0].used);  // t = 2
  CHECK(nb.report[1].status == "size 240x270, centre 480x270");               // t = 4
  CHECK(nb.report[2].status == "no image");                                    // t = 1
  CHECK(nb.report[3].status == "not float");                                   // t = 5
  collectNeighbours(3, 1, r, q.w, q.h, fetcherFor({kOk, kOk, kRgb, kOk, kEmpty, kOk, kOk}), &nb);
  CHECK(nb.offsets.empty() && nb.report[0].status == "3 channels (no matte)" && nb.report[1].status == "no pixels");
  // Some missing: the rest are used.
  collectNeighbours(3, 2, r, q.w, q.h, fetcherFor({kOk, kOk, kNull, kOk, kOk, kOk, kOk}), &nb);
  CHECK((nb.offsets == std::vector<int>{-2, 1, 2}) && nb.frames[0].matte == q.s.frames[1].matte);
  // The sequence ends.
  asked.clear();
  collectNeighbours(0, 2, r, q.w, q.h, fetcherFor({kOk, kOk, kOk, kOk, kOk, kOk, kOk}), &nb);
  CHECK((asked == std::vector<double>{1, 2}) && nb.requested == 2 && (nb.offsets == std::vector<int>{1, 2}));
  // Abort: asked before each fetch; the rest are not fetched.
  asked.clear();
  int n = 0;
  collectNeighbours(3, 2, r, q.w, q.h, fetcherFor({kOk, kOk, kOk, kOk, kOk, kOk, kOk}), &nb,
                    [&]() { return ++n > 1; });
  CHECK(nb.aborted && asked.size() == 1 && nb.offsets.size() == 1 && nb.report.size() == 1);
  std::printf("collect neighbours: ok\n");
}

Neighbours neighboursOf(const Seq& q, int t, const std::vector<int>& offsets) {
  Neighbours nb;
  for (int k : offsets) {
    nb.offsets.push_back(k);
    nb.frames.push_back(q.fetched(t + k));
  }
  nb.requested = int(offsets.size());
  return nb;
}

void testTemporalChain() {
  const Seq q;
  const size_t n = size_t(q.w) * q.h;
  const int t = 3;
  MaskSeg seg;
  seg.mask.assign(n, 1.0f);
  std::vector<float> mid, out, tmo;
  ChainResult r;
  // Temporal alone, K = 1: = stabilise() on frames 2, 3, 4 by hand; Temporal's output = mid = out.
  {
    FakeTiles tiles;
    ChainParams p;
    p.temporal = true;
    const Neighbours nb = neighboursOf(q, t, {-1, 1});
    p.neighbours = &nb;
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, seg, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
             std::function<void()>(), &tmo);
    const std::vector<float> want = q.byHand({2, 3, 4}, 1, temporalParams(p.tm, 1.0));
    CHECK(r.tRan && !r.tFailed && !r.tAborted && r.tm.neighbours == 2 && !r.sfRan && !r.erRan && seg.calls == 0);
    CHECK(sameBits(tmo, want) && sameBits(mid, want) && sameBits(out, want));
    CHECK(!sameBits(want, q.s.frames[size_t(t)].matte) && r.tm.pixelsChanged > 100);
    // K = 2, render scale 0.5 (the band halves), strength 0.5.
    p.tm.frames = kTemporal2;
    p.tm.strength = 0.5;
    p.renderScale = 0.5;
    const Neighbours nb2 = neighboursOf(q, t, {-2, -1, 1, 2});
    p.neighbours = &nb2;
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, seg, tiles, &mid, &out, &r);
    StabiliseParams sp = temporalParams(p.tm, 0.5);
    CHECK(sp.bandPx == 3.0 && sp.frames == 2 && sp.strength == 0.5);
    CHECK(sameBits(out, q.byHand({1, 2, 3, 4, 5}, 2, sp)) && r.tm.neighbours == 4);
    // A missing -1 with K = 2: the rest close up around the centre ([-2, 0, +1, +2]).
    const Neighbours nb3 = neighboursOf(q, t, {-2, 1, 2});
    p.neighbours = &nb3;
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, seg, tiles, &mid, &out, &r);
    CHECK(sameBits(out, q.byHand({1, 3, 4, 5}, 1, sp)) && r.tm.neighbours == 3);
    // The end of the sequence: one neighbour.
    const Neighbours nb4 = neighboursOf(q, 0, {1});
    p.neighbours = &nb4;
    p.tm = TemporalUi();
    p.renderScale = 1.0;
    runChain(q.m(0), q.rgb(0), q.w, q.h, p, seg, tiles, &mid, &out, &r);
    CHECK(sameBits(out, q.byHand({0, 1}, 0, temporalParams(p.tm, 1.0))) && r.tm.neighbours == 1 &&
          !sameBits(out, q.s.frames[0].matte));
  }
  // The order: Temporal's output is Smart Fill's input, Smart Fill's is Edge Refine's.
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.temporal = p.smartFill = p.edgeRefine = true;
    p.er.bandPx = 6;
    const Neighbours nb = neighboursOf(q, t, {-1, 1});
    p.neighbours = &nb;
    MaskSeg s2;
    s2.mask.assign(n, 1.0f);
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, s2, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
             std::function<void()>(), &tmo);
    const std::vector<float> tm = q.byHand({2, 3, 4}, 1, temporalParams(p.tm, 1.0));
    std::vector<float> sf(n), er(n);
    SmartFillStats sst;
    smartFill(tm.data(), q.rgb(t), q.w, q.h, smartFillParams(p.sf, 1.0), s2, sf.data(), &sst);
    refineEdge(sf.data(), q.rgb(t), er.data(), q.w, q.h, p.er, t2, nullptr);
    CHECK(r.tRan && r.sfRan && r.erRan && !r.tFailed && !r.sfFailed && !r.erFailed);
    CHECK(sameBits(tmo, tm) && sameBits(mid, sf) && sameBits(out, er));
    // Smart Fill on the input instead would differ (the check can tell the orders apart).
    std::vector<float> sfIn(n);
    smartFill(q.m(t), q.rgb(t), q.w, q.h, smartFillParams(p.sf, 1.0), s2, sfIn.data());
    CHECK(!sameBits(sfIn, sf));
    std::printf("temporal chain order: ok (holes in Temporal's output %d, filled by Smart Fill %d)\n", sst.holesFound,
                sst.holesFilled);
  }
  // Enabled, no neighbours (none fetched / none given): the input bit for bit, later steps go on.
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.temporal = p.edgeRefine = true;
    const Neighbours none;
    for (const Neighbours* nbp : {&none, static_cast<const Neighbours*>(nullptr)}) {
      p.neighbours = nbp;
      runChain(q.m(t), q.rgb(t), q.w, q.h, p, seg, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
               std::function<void()>(), &tmo);
      std::vector<float> er(n);
      refineEdge(q.m(t), q.rgb(t), er.data(), q.w, q.h, p.er, t2, nullptr);
      CHECK(r.tRan && !r.tFailed && r.tm.neighbours == 0 && sameBits(tmo, q.s.frames[size_t(t)].matte) &&
            sameBits(mid, q.s.frames[size_t(t)].matte) && r.erRan && sameBits(out, er));
    }
  }
  // Failure (a neighbour of another size reaches the core): passes the matte through, the footer
  // text; Smart Fill and Edge Refine still run on the input.
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.temporal = p.smartFill = p.edgeRefine = true;
    Neighbours bad = neighboursOf(q, t, {-1, 1});
    bad.frames[1].matte.resize(10);
    p.neighbours = &bad;
    MaskSeg s2;
    s2.mask.assign(n, 1.0f);
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, s2, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
             std::function<void()>(), &tmo);
    std::vector<float> sf(n), er(n);
    smartFill(q.m(t), q.rgb(t), q.w, q.h, smartFillParams(p.sf, 1.0), s2, sf.data());
    refineEdge(sf.data(), q.rgb(t), er.data(), q.w, q.h, p.er, t2, nullptr);
    CHECK(r.tRan && r.tFailed && r.tError == "Temporal: neighbour +1 has no 480x270 matte and picture");
    CHECK(sameBits(tmo, q.s.frames[size_t(t)].matte) && sameBits(mid, sf) && sameBits(out, er) && !r.sfFailed &&
          !r.erFailed);
    std::printf("temporal failure footer text: '%s'\n", r.tError.c_str());
  }
  // Smart Fill fails after Temporal: its step passes Temporal's output through (not the input).
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.temporal = p.smartFill = p.edgeRefine = true;
    const Neighbours nb = neighboursOf(q, t, {-1, 1});
    p.neighbours = &nb;
    ThrowSeg bad;
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, bad, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
             std::function<void()>(), &tmo);
    const std::vector<float> tm = q.byHand({2, 3, 4}, 1, temporalParams(p.tm, 1.0));
    std::vector<float> er(n);
    refineEdge(tm.data(), q.rgb(t), er.data(), q.w, q.h, p.er, t2, nullptr);
    CHECK(!r.tFailed && r.sfFailed && bad.calls == 1 && sameBits(tmo, tm) && sameBits(mid, tm) && sameBits(out, er));
  }
  // Abort, before the core and after it: the matte passes through.
  for (int after : {0, 1}) {
    FakeTiles tiles;
    ChainParams p;
    p.temporal = true;
    const Neighbours nb = neighboursOf(q, t, {-1, 1});
    p.neighbours = &nb;
    int asked = 0;
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, seg, tiles, &mid, &out, &r, [&]() { return ++asked > after; }, nullptr,
             std::function<void()>(), &tmo);
    CHECK(r.tAborted && !r.tFailed && asked == after + 1 && sameBits(out, q.s.frames[size_t(t)].matte) &&
          sameBits(tmo, out));
    CHECK(after == 0 ? r.tm.totalMs == 0 : r.tm.neighbours == 2);  // the core ran only in the second case
  }
  // Off: the neighbours are ignored, the chain is the one before B4 (Smart Fill + Edge Refine on the input).
  {
    FakeTiles tiles, t2;
    ChainParams p;
    p.smartFill = p.edgeRefine = true;
    const Neighbours nb = neighboursOf(q, t, {-1, 1});
    p.neighbours = &nb;
    MaskSeg s2;
    s2.mask.assign(n, 1.0f);
    runChain(q.m(t), q.rgb(t), q.w, q.h, p, s2, tiles, &mid, &out, &r, std::function<bool()>(), nullptr,
             std::function<void()>(), &tmo);
    std::vector<float> sf(n), er(n);
    smartFill(q.m(t), q.rgb(t), q.w, q.h, smartFillParams(p.sf, 1.0), s2, sf.data());
    refineEdge(sf.data(), q.rgb(t), er.data(), q.w, q.h, p.er, t2, nullptr);
    CHECK(!r.tRan && r.tm.neighbours == 0 && sameBits(tmo, q.s.frames[size_t(t)].matte) && sameBits(mid, sf) &&
          sameBits(out, er));
  }
  std::printf("temporal chain: ok\n");
}

// The whole sequence through the chain (Temporal alone, the neighbours collected with a fetcher over
// the sequence, clamped at the ends): flicker drops as in the core's evaluation, nothing changes
// outside the band (holes off here so the region is the band alone: the core's rule).
void testTemporalSequence() {
  const int w = 960, h = 540, T = 9;
  const synth::Sequence s = synth::makeSequence(w, h, T, 3);
  const size_t n = size_t(w) * h;
  FrameRange r;
  r.first = 0;
  r.last = T - 1;
  r.valid = true;
  MaskSeg seg;
  FakeTiles tiles;
  for (int K : {1, 2}) {
    ChainParams p;
    p.temporal = true;
    p.tm.frames = K == 2 ? kTemporal2 : kTemporal1;
    p.renderScale = double(h) / 1080;  // a half-resolution render of a 1080 setup: band 3 px
    std::vector<std::vector<float>> outs(T), ins(T);
    long long outside = 0, inside = 0, lowered = 0;
    for (int t = 0; t < T; ++t) {
      Neighbours nb;
      collectNeighbours(t, temporalK(p.tm.frames), r, w, h,
                        [&](double time, FetchedFrame* f) {
                          const int k = int(std::lround(time));
                          f->image = f->isFloat = true;
                          f->components = 4;
                          f->w = w;
                          f->h = h;
                          f->rgb = s.frames[size_t(k)].rgb;
                          f->matte = s.frames[size_t(k)].matte;
                        },
                        &nb);
      CHECK(int(nb.offsets.size()) == std::min(t, K) + std::min(T - 1 - t, K));
      p.neighbours = &nb;
      std::vector<float> mid;
      ChainResult cr;
      runChain(s.frames[size_t(t)].matte.data(), s.frames[size_t(t)].rgb.data(), w, h, p, seg, tiles, &mid, &outs[t],
               &cr);
      CHECK(cr.tRan && !cr.tFailed);
      ins[t] = s.frames[size_t(t)].matte;
      const double band = temporalParams(p.tm, p.renderScale).bandPx;
      const std::vector<float> sd = mr::signedEdgeDistance(ins[t].data(), w, h, 0.5f, band);
      for (size_t i = 0; i < n; ++i) {
        if (outs[t][i] == ins[t][i]) continue;
        if (std::fabs(sd[i]) < band) {
          ++inside;
        } else {
          ++outside;  // only a filled flickering hole may change there: raised, never lowered
          lowered += outs[t][i] < ins[t][i];
        }
      }
    }
    const double band = 3.0 * h / 1080 + 0.5;
    const double f0 = synth::flicker(s, ins, band), f1 = synth::flicker(s, outs, band);
    const double e0 = synth::bandError(s, ins, band), e1 = synth::bandError(s, outs, band);
    std::printf("temporal sequence 960x540 K=%d through the chain: flicker %.4f -> %.4f (%+.0f %%), bandError %.4f -> "
                "%.4f (%+.0f %%), changed in the band %lld, outside %lld (hole fills), lowered outside %lld\n",
                K, f0, f1, 100 * (f1 / f0 - 1), e0, e1, 100 * (e1 / e0 - 1), inside, outside, lowered);
    CHECK(f1 < (K == 1 ? 0.65 : 0.5) * f0);
    CHECK(e1 < e0);
    CHECK(inside > 1000 && outside > 0 && lowered == 0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  gFake = argv[1];
  char tmpl[] = "/tmp/mrml-chain-XXXXXX";  // short: a socket path holds at most 107 bytes
  if (!mkdtemp(tmpl)) return 2;
  gDir = tmpl;
  gSock = gDir + "/runner.sock";
  gHome = gDir + "/home";
  mkdir(gHome.c_str(), 0700);
  setenv("HOME", gHome.c_str(), 1);
  signal(SIGPIPE, SIG_IGN);
  auto test = [](const char* name, void (*fn)()) {
    try {
      fn();
    } catch (const std::exception& e) {
      std::printf("FAIL %s: unexpected exception: %s\n", name, e.what());
      ++failures;
    }
  };
  test("params", testParams);
  test("runner seg", testRunnerSeg);
  test("smart fill through the runner", testSmartFillThroughRunner);
  test("chain order", testChainOrder);
  test("off and failures", testOffAndFailures);
  test("holes view", testHolesView);
  test("temporal params", testTemporalParams);
  test("neighbour offsets", testNeighbourOffsets);
  test("collect neighbours", testCollectNeighbours);
  test("temporal chain", testTemporalChain);
  test("temporal sequence", testTemporalSequence);
  {
    RunnerClientOptions o = opts();
    RunnerClient(o).quit(nullptr);
  }
  if (failures) std::printf("kept %s for inspection\n", gDir.c_str());
  else if (std::system(("rm -rf '" + gDir + "'").c_str()) != 0) std::printf("could not remove %s\n", gDir.c_str());
  std::printf(failures ? "test_render_chain: %d FAILED\n" : "test_render_chain: all passed\n", failures);
  return failures ? 1 : 0;
}
