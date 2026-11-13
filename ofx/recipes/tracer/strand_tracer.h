// Strand tracer (Edge Recipes WP3, decision 39 candidate (b), docs/strand_tracing.md section 3): finds thin
// dark hair strands inside a user-drawn soft region and turns them into candidate alpha. A C++14 port of the
// Python prototype (tracer.py), pure float arrays, no OFX types (decision 3). The tracer never changes the base
// matte: it returns candidate alpha + confidence (0 outside the region) and the caller combines (never lower).
//
// Order (section 3): per region shape (a 4-connected component of the drawn region, weight > 0.5, plus the feather
// pixels > 0.05 nearest to it, so two shapes with touching feathers stay two; cropped with a 30 px margin)
//   detectors (black top-hat, Hessian dark-line measure + direction) -> contact seeds (traced out of the solid
//   matte and back) -> free seeds -> joins (end -> matte, end <-> end, unambiguous only) -> hair reference
//   (per FRAME, shared by all shapes, per-shape fallback; ruling 2) -> rescore + keep -> rulings 1 and 4
//   -> matting-equation alpha in a capsule x the soft region.
//
// Rulings (user, 2026-11-12) on top of the prototype, each a switch so the prototype can be reproduced:
//   1 dropEdgeTouching: an UNANCHORED piece that reaches the region's edge (any point with region < 0.5) is
//     dropped (it most likely continues outside: a band, a glove edge); unconnected pieces are kept otherwise.
//   2 frameHairRef: one hair reference per frame, pooled over all shapes' anchored pieces (else the frame's longest
//     self-qualifying piece); a shape uses its own reference only when the shared one CLEARLY does not fit
//     (clearlyDiffers below). Free seeds of a shape without contact pieces still self-reference (as the
//     prototype); the shared reference decides at the rescore / keep step.
//   4 minPieceLen1080: pieces shorter than 12 px at 1080 lines are dropped, scaled with the frame height and the
//     render scale (minimum = 12 * frameHeight / 1080 * renderScale px in the delivered image).
//   5 the base matte is never lowered: the tracer only outputs candidate alpha (the caller does max()).
// Ruling 3: separating the gap between two strands is NOT a target; a lock may come out as one piece.
//
// All pixel constants below (sigma, step, widths, gaps) describe what a strand looks like in the delivered
// image and are NOT scaled with the render scale (at proxy scale thin strands simply fall under the detector);
// only the length rule (ruling 4) scales.
#pragma once

#include <vector>

namespace mr {
namespace tracer {

// The hair reference: median profile of confident steps (depth in stops, FWHM in px, chroma shift per stop).
// Per frame; WP6 (presence vote) and the recipe can carry it between frames (TracerInput::reference).
struct HairReference {
  bool valid = false;
  double depth = 0.0;    // d: darker than the background, in stops (log2)
  double width = 0.0;    // w: FWHM in px
  double chroma1 = 0.0;  // v1: log2(R/G) shift per stop of depth
  double chroma2 = 0.0;  // v2: log2(B/G) shift per stop of depth
  bool selfReferenced = false;  // from the longest qualifying unanchored piece (no anchored piece anywhere)
  int samples = 0;              // profiles the medians came from
};

struct TracerInput {
  const float* rgb = nullptr;     // linear scene-referred, interleaved, `channels` floats per pixel (RGB first)
  int channels = 3;               // 3 or 4
  const float* base = nullptr;    // base matte 0..1 (w*h)
  const float* region = nullptr;  // soft region weight 0..1 (w*h); the tracer changes nothing where it is 0
  int width = 0, height = 0;      // of the buffers (a crop is fine)
  int frameHeight = 0;            // full-resolution frame height in lines (ruling 4); 0 = height / renderScale
  double renderScale = 1.0;
  const HairReference* reference = nullptr;  // optional: use this frame reference instead of estimating it
};

struct TracerParams {
  // detector
  double hessSigma = 1.2, seedBth = 0.35;
  int closeSize = 7;
  // tracing
  double step = 1.5, maxTurnDeg = 22.5, turnSigmaDeg = 12.0, winTurnDeg = 60.0, refine = 1.0, profHalf = 7.0;
  int winSteps = 6;
  // rule switches (the ablation of docs/strand_tracing.md section 4)
  bool rTurn = true, rWidth = true, rProfile = true, rColour = true, rContrast = true, rOrient = true,
       rDecay = true, rConnect = true, rBudget = true, rHairCtx = true;
  // Elongation (added after the prototype): an UNANCHORED piece must be at least minElongation x its own median
  // FWHM long (length = points x step). Stops the mole's pieces (3.1-3.3 x on the prototype's scenes) that pass
  // the lower keep bar; the real lock is 3.75 x (22.5 px / 6.0 px), crossing strands 8.5 x and more. Thin margin.
  bool rElongation = true;
  double minElongation = 3.5;
  // scores
  double d0 = 0.25, dSoft = 0.06, wMax = 7.0, wSig = 0.35, symSig = 0.6, colSig = 0.15, oriSigDeg = 20.0,
         wSigCtx = 0.5, contrastFloor = 1.0 / 3.0;
  // stopping
  double ema = 0.4, cStop = 0.30, hLow = 0.20, trimH = 0.40, budgetFactor = 1.5;
  int gapSteps = 3, searchDilate = 4;
  // pieces / connection
  double minLen = 6.0, contactPx = 6.0, freeLenW = 4.0, keepConf = 0.45, anchorBonus = 0.15,
         freeSeedConf = 0.6, gapMax = 12.0, endAngleDeg = 35.0, joinMax = 1.0, ambig = 0.8;
  // Keep bar for UNANCHORED pieces scored against a shared (not self) hair reference = the Hair Locks control
  // "Rebuild Min Confidence". Anchored and self-referenced pieces only need keepConf. The prototype used 0.7
  // (free_keep). With 0.45 the mole's pieces pass the bar; the elongation rule (rElongation) stops them.
  double minConfUnanchored = 0.45;
  bool capsuleAlpha = false;  // false = matting equation (section 3.9), true = geometric coverage
  int cropMargin = 30;        // px around each shape's bounding box
  // user rulings 2026-11-12 (false / 0 = the prototype's behaviour)
  bool frameHairRef = true;        // ruling 2
  bool dropEdgeTouching = true;    // ruling 1
  double minPieceLen1080 = 12.0;   // ruling 4; 0 = off
  // Ruling 4 measures a STRAND, not a fragment: true = a piece's length for the rule is the length of its chain
  // (pieces linked by end-to-end joins, plus the joins), and a chain ATTACHED to the matte (it holds a contact
  // piece or a matte join) is exempt: it continues the matte, it cannot be a speck. false = each piece on its
  // own (WP3). Found on real frames (agent hl96): a 10.5 px anchored piece that bridges
  // a lock to the matte was dropped by ruling 4, leaving the lock it anchored floating (frame 1096); a lock's
  // 10.5 px contact piece was dropped though it continues the matte (frame 1063, the right lock left short).
  bool chainLength = false;
  // End -> matte join: false = only to the NEAREST solid pixel (WP3); true = when that one fails (angle or cost),
  // the cheapest solid edge pixel within gapMax that passes the same angle and cost rules (the strand points past
  // the nearest pixel: frame 1096's right lock, agent hl96).
  bool coneJoin = false;
  // Soft strand profile (agent hl_flicker, Flame retest 2026-11-13: rebuilt strands came out as crisp capsules
  // whose per-pixel matting alpha changed from frame to frame). true = a strand's alpha is its peak x the
  // coverage of a bar of the strand's width blurred by a Gaussian of edgeSigma1080 px at 1080 lines (scaled with
  // the frame height and the render scale, as ruling 4), and a free end (not joined to the matte or another piece)
  // fades out over taperW x the width; the peak is the median matting alpha on the piece's centreline (1 when the
  // matting equation does not apply), the same for the whole piece, so it does not follow the pixel noise.
  // A join gets the peak of the piece it starts from. false = the matting alpha in a hard-edged capsule (WP3).
  bool softProfile = false;
  double edgeSigma1080 = 1.0;
  double taperW = 1.5;
};

struct Point {
  double x, y;
};

struct Piece {
  std::vector<Point> polyline;  // centreline, step 1.5 px, frame (buffer) coordinates
  double width = 0.0;           // median FWHM, px
  double length = 0.0;          // points x step, px (the prototype's measure)
  double confidence = 0.0;      // median step score (+0.15 when anchored; x0.8 when self-referenced)
  bool anchored = false;        // reaches the solid matte (contact seed, matte join or a join to an anchored piece)
  bool touchesRegionEdge = false;  // some point has region < 0.5
  bool selfReferenced = false;  // scored against a self-derived hair reference
  int shape = 0;                // index of the region shape
};

struct Join {
  std::vector<Point> polyline;  // cubic Hermite, frame coordinates
  double width = 0.0, cost = 0.0;
  bool toMatte = false;
  int pieceA = -1, pieceB = -1;  // indices into TracerResult::pieces (equal for a matte join)
};

struct TracerResult {
  std::vector<float> alpha;       // w*h candidate strand alpha (0 where region is 0)
  std::vector<float> confidence;  // w*h, the piece confidence where alpha > 0 (joins 0.5), else 0
  std::vector<Piece> pieces;      // kept pieces
  std::vector<Join> joins;        // joins between kept pieces / to the matte
  HairReference reference;        // the frame reference (invalid when nothing was found)
  int shapes = 0;
  // Diagnostics: pieces the keep step dropped and why (1 conf < keepConf, 2 unanchored conf < minConfUnanchored,
  // 3 unanchored shorter than freeLenW x hair width, 4 ruling 1 (edge), 5 elongation, 6 ruling 4 (length)).
  std::vector<Piece> rejected;
  std::vector<int> rejectReason;
};

// Minimum piece length in px of the delivered image (ruling 4).
double minPieceLength(const TracerParams& p, int frameHeight, int height, double renderScale);

// Ruling 2's test: the shared reference clearly does not fit a shape whose own reference is `own` when
// |ln(own.w / shared.w)| > wSigCtx (one context sigma, the width tolerance of the rescoring), or the depth ratio
// own.d / shared.d is outside [contrastFloor, 1 / contrastFloor], or the chroma shift differs by more than
// 2 colSig. Inside those limits every score factor stays above about 0.6 for the shape's typical step.
bool clearlyDiffers(const HairReference& own, const HairReference& shared, const TracerParams& p);

TracerResult traceStrands(const TracerInput& in, const TracerParams& p = TracerParams());

}  // namespace tracer
}  // namespace mr
