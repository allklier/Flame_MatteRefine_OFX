// The render chain of Matte Refine ML without OFX (plugin_ml.cpp calls it; the tests run it with
// fakes): matte -> Temporal (B4, temporal.h) -> Smart Fill (B3, smart_fill.h) -> Edge Refine (B1,
// edge_refine.h), each step only when enabled, the order fixed (definition order = chain order, root
// decision 19). C++14.
//
// A step that throws (no runner, no model, a model error, a timeout) passes its input through
// unchanged and records the error (the plugin's footer); the chain goes on with the next step.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "edge_refine.h"
#include "smart_fill.h"
#include "temporal.h"

namespace mr {
namespace ml {

class RunnerClient;

// Smart Fill's controls on the node. Segmentation: an index stored in setups (append only).
enum Segmentation { kSegLite = 0, kSegFull = 1 };
const char* segmentationModel(int segmentation);  // the model table id: birefnet-lite / birefnet
struct SmartFillUi {
  double subjectThreshold = 0.6;  // "Subject Threshold"
  double maxHolePercent = 10.0;   // "Max Hole %", 0 = no limit
  int segmentation = kSegLite;
};
// The core's params for the node's controls at this render scale: the pixel params (rimPx, ringPx,
// bgReachPx) are in full-resolution pixels and scale with it (decision 6); the rest are the core's
// defaults (mode Both, mean, keepRealGaps).
SmartFillParams smartFillParams(const SmartFillUi& ui, double renderScale);

// The segmentation through mr_ml_runner: birefnetPrepare -> RunnerClient::runFrame(model, [3, 1024,
// 1024] -> [1, 1024, 1024]) -> birefnetPostprocess. Throws what the client throws.
class RunnerSeg : public SegRunner {
 public:
  RunnerSeg(RunnerClient& client, const std::string& modelId) : client_(client), model_(modelId) {}
  void segment(const float* rgb, int w, int h, float* mask) override;
  int calls = 0;
  double prepMs = 0, runMs = 0, postMs = 0;  // the last call's

 private:
  RunnerClient& client_;
  std::string model_;
  std::vector<float> in_, out_;
};

// --- Temporal (B4) --------------------------------------------------------------------------------
// Frames: an index stored in setups (append only): "+-1 frame" (K = 1), "+-2 frames" (K = 2).
enum TemporalFrames { kTemporal1 = 0, kTemporal2 = 1 };
int temporalK(int framesIndex);  // 1 or 2; an unknown index: 1
struct TemporalUi {
  int frames = kTemporal1;  // "Frames"
  double strength = 1.0;    // "Strength"
  double bandPx = 6.0;      // "Band px", full-resolution px (the core's default at 1080)
};
// The core's params: K from Frames, the band scaled with the render scale (decision 6, as Edge
// Refine's Band px); the rest the core's defaults (Median, holes on, flow at <= 600 lines).
StabiliseParams temporalParams(const TemporalUi& ui, double renderScale);

// The Source clip's frame range as the host reports it (getFrameRange); valid false = unknown: no
// clamping (a time the host has no image for is then a failed fetch).
struct FrameRange {
  double first = 0, last = 0;
  bool valid = false;
};
// The neighbour offsets fetched for frame t, nearest first (-1, +1, -2, +2 up to K), those whose
// time t + k lies in the range (at the sequence ends fewer; none for a one-frame clip).
std::vector<int> neighbourOffsets(double t, int K, const FrameRange& r);
// getFramesNeeded's range of the Source clip for frame t: [t - K, t + K] clamped to the range (t
// itself always inside).
void framesNeededRange(double t, int K, const FrameRange& r, double* lo, double* hi);

// One neighbour as the host returned it; the plugin's fetcher fills it from an OFX::Image.
struct FetchedFrame {
  bool image = false;   // an image came back (false: null, or the fetch threw: error)
  std::string error;    // what the fetch threw
  int w = 0, h = 0, components = 0;
  bool isFloat = false;
  std::vector<float> rgb, matte;  // w*h*3, w*h, top-down: filled for a float 4-channel image only
  std::string note;     // the fetcher's diagnosis for the log (unique id, same pixels as the centre, ...)
  double ms = 0;        // fetch + copy
};
using FrameFetcher = std::function<void(double time, FetchedFrame* f)>;
struct NeighbourReport {  // one requested neighbour, for the log line
  int offset = 0;
  double time = 0;
  bool used = false;
  std::string status;  // "used", "no image", "threw: ...", "size WxH, centre WxH", "not float", "N channels"
  int w = 0, h = 0, components = 0;
  double ms = 0;
  std::string note;
};
struct Neighbours {
  std::vector<int> offsets;             // the usable neighbours, ascending
  std::vector<FetchedFrame> frames;     // the same order
  std::vector<NeighbourReport> report;  // every requested neighbour, in request order
  int requested = 0;
  double fetchMs = 0;  // all fetches
  bool aborted = false;
};
// Fetches every offset of neighbourOffsets(t, K, r) and keeps the usable ones: an image, float, 4
// channels (the matte is the alpha, decision 35), the centre's size w x h, rgb and matte filled.
// abort() is asked before each fetch (aborted: the rest are not fetched).
void collectNeighbours(double t, int K, const FrameRange& r, int w, int h, const FrameFetcher& fetch, Neighbours* nb,
                       const std::function<bool()>& abort = std::function<bool()>());
// Temporal on the centre (matte, rgb: w*h, w*h*3) with the neighbours: stabilise() on the frames in
// time order (neighbours before, the centre, after; a missing neighbour closes up, so a lone -2 gets
// the -1 weight). Throws std::invalid_argument for a neighbour whose matte or picture is not w*h.
// abort() is asked before and after the core (it has no abort of its own): aborted -> out = matte.
// out: w*h.
void runTemporal(const float* matte, const float* rgb, int w, int h, const StabiliseParams& p, const Neighbours& nb,
                 float* out, StabiliseStats* stats, bool* aborted,
                 const std::function<bool()>& abort = std::function<bool()>());

struct ChainParams {
  bool temporal = false;
  TemporalUi tm;
  const Neighbours* neighbours = nullptr;  // Temporal's neighbour frames (nullptr: none)
  bool smartFill = false;
  SmartFillUi sf;
  double renderScale = 1.0;  // Temporal's and Smart Fill's pixel params (Edge Refine's band: er.bandPx, scaled by the caller)
  bool edgeRefine = false;
  RefineParams er;
};

struct ChainResult {
  bool tRan = false, tFailed = false, tAborted = false;  // tRan: enabled and called (also with no neighbours)
  std::string tError;
  StabiliseStats tm;
  bool sfRan = false, sfFailed = false;  // ran: the step was enabled and called
  std::string sfError;
  SmartFillStats sf;
  bool erRan = false, erFailed = false;
  std::string erError;
  RefineStats er;
};

// temporalOut (if given) = Temporal's output (= Smart Fill's input; the matte when Temporal is off or
// failed), mid = Smart Fill's output (= Edge Refine's input; Smart Fill's input when it is off or
// failed), out = the chain's result; all resized to w*h. abort as the cores define it; between() is
// called after Smart Fill, before Edge Refine (the plugin reads and resets the runner client's stats).
void runChain(const float* matte, const float* rgb, int w, int h, const ChainParams& p, SegRunner& seg,
              TileRunner& tiles, std::vector<float>* mid, std::vector<float>* out, ChainResult* r,
              const std::function<bool()>& abort = std::function<bool()>(), std::vector<float>* trimapOut = nullptr,
              const std::function<void()>& between = std::function<void()>(),
              std::vector<float>* temporalOut = nullptr);

// --- View "Holes" (QC) -------------------------------------------------------------------------
enum HoleMark : uint8_t { kMarkNone = 0, kMarkFilled = 1, kMarkKept = 2 };
// marks (w*h): the pixels of every hole Smart Fill filled / kept as a real gap, by a 4-connected
// flood fill from each hole's seed over matte < solidAbove (matte = Smart Fill's input: the holes
// it found). Cost: the hole pixels plus clearing w*h bytes.
void markHoles(const float* matte, int w, int h, float solidAbove, const SmartFillStats& st,
               std::vector<uint8_t>* marks);
// The view's RGB: the result matte a as grey, filled holes green, kept real gaps red.
inline void holesViewRgb(uint8_t mark, float a, float* rgb) {
  rgb[0] = mark == kMarkKept ? 1.0f : mark == kMarkFilled ? 0.0f : a;
  rgb[1] = mark == kMarkFilled ? 1.0f : mark == kMarkKept ? 0.0f : a;
  rgb[2] = mark == kMarkNone ? a : 0.0f;
}

// View "Changes": the same colours as Matte Refine's Show Changes (plugin.cpp): the input matte m in
// grey, raised pixels tinted magenta, lowered ones green; unchanged pixels stay grey.
inline void changesViewRgb(float m, float a, float* rgb) {
  const float d = a - m;
  if (d >= 0.0f) {
    rgb[0] = rgb[2] = m * (1.0f - d) + d;
    rgb[1] = m * (1.0f - d);
  } else {
    rgb[0] = rgb[2] = a;
    rgb[1] = std::min(1.0f, a - d);
  }
}

}  // namespace ml
}  // namespace mr
