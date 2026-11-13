// Synthetic test mattes (CLAUDE.md "Synthetic mattes"): a figure drawn from distance-defined
// shapes with known defects, its clean truth, and a picture to go with it. Used by the unit
// tests and, written as EXR by mr_synth, for evaluation and demos in Flame. Deterministic.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace synth {

struct Scene {
  std::string name, about;
  int w = 0, h = 0;
  std::vector<float> matte;  // w*h, 0-1, row 0 = top: the input matte, with the scene's defects
  std::vector<float> truth;  // w*h: the clean matte the defects were added to
  std::vector<float> rgb;    // w*h*3: the picture, truth-alpha composite of fg over a background
  std::vector<float> fg;     // w*h*3: the true foreground colour (for Foreground Colour)
  std::vector<uint8_t> label;  // "gaps" only (GapLabel per pixel, w*h); empty for the others
};

// Scene names, in demo order: holes, specks, chatter, mixed_edge, eroded, coarse, fgcolor, all.
std::vector<std::string> sceneNames();

// Geometry scales with h (designed at 1080); throws std::invalid_argument for an unknown name.
Scene make(const std::string& name, int w, int h, unsigned seed = 1);

// ---- Scenes for Matte Refine ML (ofx/ml/PLAN_FEATURES.md, P4). Same conventions: design units
// at 1080 lines, origin = frame centre, deterministic per (w, h, seed); sceneNames() is unchanged.

// "gaps", "hair_flat", "hair_busy": make() builds these too. The sequence: makeSequence().
std::vector<std::string> mlSceneNames();

// B3 Smart Fill Holes, scene "gaps". A warm, striped subject over a cool textured background.
// truth: the subject, 2 px edge. Real gaps (truth 0, enclosed by the subject): the arm on the
// hip (~20500 design units^2), between the legs (feet touching, ~9000), a ring held in the right
// hand (~4100), a small ring linked below it (~700), an earring (~200). Defect holes in the
// matte (truth 1): 10 at fixed sizes, areas ~50 .. ~9900 (radius 4..44 circles and one 75x42
// ellipse), positions and angle from the seed, value 0 (even ones) or 0.1-0.4 (odd ones), so
// every hole is below 0.5 and the two size ranges overlap: no size threshold separates them.
// A grey wash (0.58-1, never below 0.5) on the torso's right side: label 1, matte < truth.
// label: 0 background (truth < 0.5, 4-connected to the frame border), 1 subject (truth >= 0.5,
// not in a defect hole), 2 defect hole (truth >= 0.5 and inside a hole's geometry: fill it),
// 3 real gap (truth < 0.5, not connected to the border, 4-connected as Fill Holes: keep it).
// At 1920x1080, seed 1: 84.7 % / 12.5 % / 1.18 % / 1.66 % (test_synth holds the bounds).
enum GapLabel : uint8_t { kBackground = 0, kSubject = 1, kDefectHole = 2, kRealGap = 3 };
Scene makeGaps(int w, int h, unsigned seed = 1);

// B1/B2 hair, scenes "hair_flat" / "hair_busy". A head and shoulders rising from the bottom of
// the frame (top of the head at the frame centre) with 260 strands from the top of the head: 1-3
// px wide at 1080 (scaled with h, tapering), curved, opacity 0.35-1 (fading to the tip), 60-240
// px long. truth: body (2 px edge) and strands composited "over" (1 - prod(1 - a)), coverage
// resolved per strand (sub-pixel width = partial coverage). matte: the coarse input a user
// would have: truth blurred (Gaussian, sigma 6 px), thresholded (0.46-0.54 ramp), eroded 3 px
// (square min filter): the strands are gone, the body edge is 3 px inside. fg: the hair / skin
// colour (strands lighter than the head). rgb: fg over a flat grey-blue (busy = false) or a busy
// multi-scale, saturated texture (busy = true); truth, matte and fg do not depend on busy.
Scene makeHair(int w, int h, bool busyBackground, unsigned seed = 1);

// B4 temporal. x' = a x + b y + tx, y' = c x + d y + ty, in pixels: pixel (px, py) is the point
// (px, py) (its centre), row 0 = top.
struct Affine {
  double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
  void apply(double x, double y, double* ox, double* oy) const;
  Affine inverse() const;
};
// Maps a point of frame t0 to frame t1, given each frame's motion from frame 0: to * from^-1.
Affine between(const Affine& from, const Affine& to);

// A moving figure (the "Synthetic mattes" figure without hair) over a textured background.
struct Sequence {
  int w = 0, h = 0;
  std::vector<Scene> frames;   // per frame: matte (noisy input), truth (clean), rgb, fg
  std::vector<std::vector<float>> edgeDistance;  // per frame, w*h: signed distance to the
                                                 // truth edge in px, > 0 inside the subject
  std::vector<Affine> subject;     // frame 0 -> frame t, subject points (subject[0] = identity)
  std::vector<Affine> background;  // frame 0 -> frame t, background points (translation only)
  std::vector<bool> holeOn;        // the flickering hole is in frame t's matte
};
// Per frame the subject turns 0.3-0.5 deg about the frame centre and moves (5-7, -1.5 to -0.5)
// px at 1080 (scaled with h; drawn from the seed). The background moves (-2.5, 1) px a frame
// when movingBackground, else stands still. The input matte has noise that is independent
// between frames: edge position jitter (+-2.5 px, ~18 px features), speckle in the edge band
// (2 % of the pixels within 6 px of the edge get a random value), and a hole (radius 18, value
// 0.1-0.4) inside the torso, moving with the subject, present on about half the frames (on at
// least one, off at least one). truth: the noise-free matte, 2 px edge.
Sequence makeSequence(int w, int h, int frames = 9, unsigned seed = 1, bool movingBackground = true);

// Scores for a sequence of mattes (one w*h matte per frame, e.g. a stabiliser's output; pass the
// frames' matte or truth to see the input's or the ideal score). The band: pixels whose
// |edgeDistance| <= bandPx in frame t.
// flicker: mean |m_t(p) - m_{t-1}(q)| over the band of frames t >= 1, q = p moved back with the
// true subject motion (bilinear; q outside the frame: skipped). The truth scores near 0 (0.014
// at 1920x1080, band 3: resampling a 2 px edge), the input 0.21.
double flicker(const Sequence& s, const std::vector<std::vector<float>>& mattes, double bandPx);
// bandError: mean |m_t - truth_t| over the band of all frames.
double bandError(const Sequence& s, const std::vector<std::vector<float>>& mattes, double bandPx);

// Jump Repair (ofx/ml/src/jump_repair.h): 20 frames of clean mattes with legitimate motion and
// injected short jumps. Design units at 1080 lines, origin = frame centre, 2 px edges. Legitimate
// (in truth and matte): the figure moving right 6 px a frame (from x -250); a static rounded box
// (centre (600, 250), half size 150); a fast disc (r 16, y -480, x = -900 + 80 t: 5 diameters a
// frame); a swinging bar (r 18, from (850, 520) upwards, length 200 + 90 cos(2 pi t / 8): longest at
// t = 0, 8, 16, shortest at 4, 12). Jumps (matte only, discs): (0) added r 20 at (300, -300), frame
// 3; (1) removed r 22 at (600, 250) (in the box), frame 5; (2) added r 18 straddling the moving
// figure's outer edge (right arm, y -40), frame 7; (3) added r 20 at (-600, -420), frames 10-11; (4) removed r
// 24 at (600, 250), frames 14-16. jitterPx > 0 adds per-frame edge jitter (+- jitterPx at 1080, 18 px
// features) to the matte. rimPx > 0 gives every jump a soft rim (a hallucination's falloff): beyond the
// disc's radius r its alpha is exp(-4.5 ((d - r) / rimPx)^2) (0.5 at 0.39 rimPx, 0.011 at rimPx; rimPx
// at 1080, scaled with h); 0 = the 2 px edge (the default, unchanged). legit[t]: the fast disc's and the
// bar's pixels at frame t, grown by 3 px.
struct JumpDefect {
  int first = 0, last = 0;        // frames
  double cx = 0, cy = 0, r = 0;   // pixels, row 0 = top
  bool added = true;
};
struct JumpSequence {
  int w = 0, h = 0, frames = 0;
  std::vector<std::vector<float>> matte, truth;  // per frame, w*h
  std::vector<JumpDefect> defects;
  std::vector<std::vector<uint8_t>> legit;       // per frame, w*h
};
JumpSequence makeJumpSequence(int w, int h, double jitterPx = 0.0, unsigned seed = 1, double rimPx = 0.0);

// Uncompressed scanline OpenEXR, 32-bit float channels: 1 channel = "Y", 3 = "R G B",
// 4 = "R G B A"; data w*h*channels, interleaved, row 0 = top. Returns false on an I/O error.
bool writeExr(const std::string& path, int w, int h, int channels, const std::vector<float>& data);

// Edge Recipes WP6 (strand presence vote, docs/edge_recipes.md section 6): a hair-locks SEQUENCE in the
// style of the strand tracer's scenes (docs/strand_tracing.md section 4): a dark hair-mass head
// silhouette over a static skin-like background, moving, turning and skewing, with two locks leaving its
// right edge over the background: lock 0 one 3 px strand, lock 1 a pair of 2.5 px strands with a 2 px
// gap (centres 4.5 px apart). Design units at 562 lines (k = h / 562 scales the head, motion and lock
// lengths; strand widths stay in frame px). Linear RGB. Per frame t, motion[t] maps a frame-0 point
// (frame px) to frame t: about the head centre, rotation 0.8 deg, x skew 0.01 and translation (1.8,
// -0.5) k px per frame. base: the head silhouette (soft 1 px edge), without the locks; with
// baseJitterPx > 0 its edge moves by up to +-baseJitterPx per frame (independent between frames, the
// matte noise Temporal is for). region: one soft roto shape per lock that follows the motion (14 k px
// around the lock's centreline, root inside the head, 5 k px feather). strandTruth: the locks' coverage
// outside the head (the alpha a rebuild should add). Sensor noise (0.6 %) is independent per frame.
// Injected flicker (all off by default; a frame index < 0 = off):
//   dipFrame       lock 0 drawn at dipContrast of its darkness on that frame (its tracer confidence dips
//                  below the keep bar; truth unchanged). Measured with the WP3 tracer: contrast 0.30 -> conf
//                  1.04, 0.25 -> 0.87, 0.22 -> 0.73, <= 0.20 -> not seeded at all: the tracer's confidence
//                  falls off a cliff (seed bar) rather than gradually, so the default 0.20 makes it a dropout
//   vanishFrame    lock 1 drawn at vanishContrast (about invisible: the tracer loses it; truth unchanged)
//   spuriousFrame  a 22 px, 2.5 px wide hair-coloured stroke next to lock 1 (9.5 px from its middle),
//                  on that frame only, not anchored; not hair: not in strandTruth, marked in `spurious`
//   flowFailFrame  that frame's picture gets a flash (x flashGain) and strong per-pixel noise, so an
//                  optical flow to or from it fails (truth unchanged)
struct LockFlicker {
  int dipFrame = -1, vanishFrame = -1, spuriousFrame = -1, flowFailFrame = -1;
  double dipContrast = 0.20, vanishContrast = 0.08, flashGain = 3.0, flashNoise = 0.35;
};
struct LockStrand {
  std::vector<double> x, y;  // centreline in frame px (row 0 = top), the part outside the head
  double width = 0;
  int lock = 0;
};
struct LocksSequence {
  int w = 0, h = 0, frames = 0;
  std::vector<std::vector<float>> rgb;          // per frame, w*h*3
  std::vector<std::vector<float>> base;         // per frame, w*h
  std::vector<std::vector<float>> region;       // per frame, w*h
  std::vector<std::vector<float>> strandTruth;  // per frame, w*h
  std::vector<std::vector<float>> headDistance;  // per frame, w*h: signed distance to the true head edge, > 0 inside
  std::vector<std::vector<uint8_t>> spurious;   // per frame, w*h (empty when the frame has no spurious stroke)
  std::vector<std::vector<LockStrand>> strands;  // per frame: the truth centrelines
  std::vector<Affine> motion;                   // frame 0 -> frame t
  LockFlicker flicker;
};
LocksSequence makeLocksSequence(int w, int h, int frames = 16, unsigned seed = 1,
                                const LockFlicker& flicker = LockFlicker(), double baseJitterPx = 0.0);

}  // namespace synth
