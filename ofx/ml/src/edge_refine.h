// Edge Refine (B1, ofx/ml/CLAUDE.md decision 30): a trimap from the matte itself, the edge tiles
// through a matting model, the result blended back into the band only. Pure C++14: no OFX and no
// ONNX Runtime types; the model is behind TileRunner, so the tests run with a fake one.
//
// Trimap: sd = signed distance to the matte's 0.5 contour (ops/edge_field), > 0 inside; core (1)
// where sd > band, clear (0) where sd < -(band + outer), unknown (0.5) between. outer (Outer px, 0 by
// default) reaches further outside only, where the strands a coarse matte lost lie (hair: 100-150 px
// at 1080 recovers about 5x the strand alpha of the symmetric band; ofx/ml/CLAUDE.md). Tiles of T x T (stride T -
// overlap, clamped inside the frame) that contain an unknown pixel go through the model; the
// tiles are averaged with a linear ramp over `overlap` px at inner tile sides. Model input per
// tile: [1, 4, T, T], RGB clamped to 0-1 (the picture is display-referred already: the plugin's
// Picture Colour Auto has Flame convert it to Rec.709 display, Bypass is the user's choice) and
// normalised (x - 0.5) / 0.5, trimap 0 / 0.5 / 1 as is (transformers VitMatteImageProcessor). Blend back: both sides fade over band / 2 at the end
// of their reach: inside w = 1 up to sd = band / 2, linearly to 0 at sd = band; outside w = 1 up to
// -sd = band + outer - band / 2, linearly to 0 at -sd = band + outer (blendWeight); out = in + mix * w
// * (ml - in). outer = 0: the symmetric rule, bit for bit. Beyond the band out = in, bit-identical
// (decision 17).
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace mr {
class Workers;  // ops/morphology.h
namespace ml {

struct RefineParams {
  double bandPx = 10.0;   // half-width of the unknown band, px at the frame's own resolution
  double outerPx = 0.0;   // the band reaches this much further outside the edge (hair); 0 = symmetric
  double mix = 1.0;       // 0 = the input matte, 1 = the model inside the band
  bool onlyGrey = false;  // pixels of the input at <= 0.02 or >= 0.98 keep their value
  int tile = 512;         // a multiple of 32 (ViTMatte); a smaller frame gets a smaller tile
  int overlap = 32;
};

struct RefineStats {
  double trimapMs = 0, prepMs = 0, runMs = 0, postMs = 0, blendMs = 0, totalMs = 0;
  int tiles = 0;     // selected
  int tilesRun = 0;  // through the model (< tiles after an abort)
  int tileSize = 0;  // the tile size used
  double bandFraction = 0;  // unknown pixels / frame
  long long pixelsChanged = 0;
  bool aborted = false;
};

struct Tile {
  int x, y;
};

// One model call. `input` is [4, size, size] (RGB normalised, trimap), `alpha` size*size floats
// to fill. x, y = the tile's place in the frame (for information; a fake runner uses it). Throws
// std::exception on failure.
class TileRunner {
 public:
  virtual ~TileRunner() {}
  virtual void run(const float* input, int size, float* alpha, int x, int y) = 0;
};

// --- the pieces (shared with mr_ml_bench) -------------------------------------------------------
// outer = the extra outward reach (RefineParams::outerPx): unknown where -(band + outer) <= sd <= band.
std::vector<float> trimapFromDistance(const std::vector<float>& sd, double band, double outer = 0.0);
// sd and the trimap of a w x h matte. sd: signedEdgeDistance to band + 2 px (the same values whatever
// outer is), beyond that, with outer > 0, from a second one out to band + outer + 2 px (where it may
// be the grid distance: within about 1 px).
void buildTrimap(const float* matte, int w, int h, double band, std::vector<float>* sd,
                 std::vector<float>* trimap, double outer = 0.0);
std::vector<Tile> selectTiles(const std::vector<float>& trimap, int w, int h, int tile, int overlap);
// A tile pixel's weight in the overlap average: a ramp towards each side that is not the frame
// edge, never 0, so a selected tile covers all its pixels.
float tileWeight(const Tile& t, int tile, int overlap, int w, int h, int lx, int ly);
// The model input of one tile ([4, T, T]); pixels beyond the frame repeat the edge. Rows in
// parallel on wk (or fresh threads); bit-identical to a single-threaded loop.
void prepareTile(const float* rgb, const float* trimap, int w, int h, const Tile& t, int tile,
                 float* input, Workers* wk = nullptr);
// Adds one tile's alpha (clamped 0-1) to the weighted sums.
void accumulateTile(const float* alpha, const Tile& t, int tile, int overlap, int w, int h,
                    float* acc, float* wsum);
// out = the blend of in and the model's result ml (< 0 = no model value) by sd (see the top).
// out may be in. Returns the number of pixels that changed.
long long blendBack(const float* in, const float* ml, const float* sd, size_t n, double band,
                    double mix, bool onlyGrey, float* out, double outer = 0.0);
// The blend weight (before Mix) at signed distance sd (see the top); 0 outside the unknown band.
double blendWeight(double sd, double band, double outer);
// A picture value as the model sees it before the normalisation: clamped to 0-1 (NaN -> 0).
float clampPicture(float v);

// --- the whole operation ----------------------------------------------------------------------
// matte w*h (0-1, row 0 = top), rgb w*h*3 display-referred (any range: clamped for the model); out w*h (may be matte). abort() is asked
// between tiles: when it says yes, out = the input matte and stats->aborted. trimapOut, if
// given, receives the trimap. Exceptions from the runner propagate (out is then undefined).
void refineEdge(const float* matte, const float* rgb, float* out, int w, int h,
                const RefineParams& p, TileRunner& runner, RefineStats* stats,
                const std::function<bool()>& abort = std::function<bool()>(),
                std::vector<float>* trimapOut = nullptr);

}  // namespace ml
}  // namespace mr
