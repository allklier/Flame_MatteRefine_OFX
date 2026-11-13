// Edge Recipes WP8: synthetic hair-locks SEQUENCES for the end-to-end scores of Hair Locks. A parameterised
// version of synth::makeLocksSequence (ofx/tools/synth.h; same head, motion and two locks) with the knobs the
// scores sweep: picture look (docs/model_candidate.md: the research's four looks), strand width, how loosely
// the Focus Region is drawn, one distractor of the tracer research (docs/strand_tracing.md section 4), base
// matte jitter (what Temporal is for) and the vote's injected flicker (a dip and a spurious stroke).
//
// Geometry (frame px, k = h / 562): head ellipse rx 120k, ry 160k left of centre; lock 0 = one strand of
// `width`, lock 1 = a pair of `width` with a 2 px gap. Region per lock: radius 9 + the lock's half width +
// `marginPx` around its centreline (marginPx 0 = tight, 10 / 20 = drawn loosely), 5k px feather. Motion: about
// the head centre, 0.8 deg + 0.01 x skew + (1.8, -0.5) k px per frame. Distractors move with the head.
// Linear RGB. Truth: `cov` = the strands' coverage (lock strands + a crossing strand); the target matte of a
// frame is base + (1 - base) cov; the effect should add (1 - base) cov.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "synth.h"

namespace er {

enum Look { kDarkSkin = 0, kDarkTexture, kDarkShadow, kLightSkin, kLookCount };
enum Distractor { kNone = 0, kGlove, kGlasses, kMole, kCrossing, kTipBand, kDistractorCount };
const char* lookName(int look);
const char* distractorName(int d);

struct SceneSpec {
  int w = 424, h = 562, frames = 9;
  int look = kDarkSkin;
  double width = 3.0;     // strand width px
  double marginPx = 0.0;  // region drawn this much looser than tight
  int distractor = kNone;
  double jitterPx = 0.0;  // base matte edge jitter per frame (independent between frames)
  int dipFrame = -1;       // lock 0 at 20 % contrast on that frame (truth unchanged)
  int spuriousFrame = -1;  // a 22 px hair-coloured stroke next to lock 1 on that frame, not hair
  double detachPx = 0.0;   // > 0: lock 0 starts this far OUTSIDE the head (unanchored: no matte end near it)
  unsigned seed = 1;
  std::string name() const;
};

struct ScoreSeq {
  SceneSpec spec;
  int w = 0, h = 0, frames = 0;
  std::vector<std::vector<float>> rgb;      // w*h*3
  std::vector<std::vector<float>> base;     // w*h
  std::vector<std::vector<float>> region;   // w*h, the drawn soft roto (what the user connects)
  std::vector<std::vector<float>> cov;      // w*h, truth strand coverage (all strands, crossing included)
  std::vector<std::vector<float>> crossCov;  // w*h, the crossing strand alone (empty unless kCrossing)
  std::vector<std::vector<float>> head;     // w*h, the true head coverage (no jitter)
  std::vector<std::vector<uint8_t>> distract;  // w*h, distractor pixels (cover > 0.2, no strand > 0.05)
  std::vector<std::vector<uint8_t>> spurious;  // w*h, the spurious stroke (empty on other frames)
  std::vector<synth::Affine> motion;        // frame 0 -> frame t
};

ScoreSeq makeScoreSeq(const SceneSpec& s);
// 2x box downsample of everything (a proxy at render scale 0.5); masks: any of the four.
ScoreSeq downsample2(const ScoreSeq& s);
// synth::makeLocksSequence as the vote's system test uses it (WP6 reference), converted (cov = strandTruth
// over the head as well: the head there has no jitter, so (1 - base) cov = strandTruth).
ScoreSeq fromLocksSequence(const synth::LocksSequence& L);

}  // namespace er
