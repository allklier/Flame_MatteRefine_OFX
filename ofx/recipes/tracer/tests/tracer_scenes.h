// Synthetic scenes for the strand tracer (a C++ re-creation of the prototype's synth.py, docs/strand_tracing.md
// section 4) and the prototype's metrics (evaluate.py). Not part of the library; tests and tools only.
//
// Every scene is 112 x 112, linear RGB: a solid "head" matte on the left (edge x ~ 30 with a wobble), skin-like
// noisy background, strands leaving the head (cubic Bezier, width w px, hair colour), a soft region over the gap
// (strand pixels within 30 px of the head, dilated 8 px, feathered). The 12 scenes:
//   single     one 3 px strand              pair      two 3.5 px strands, 2 px gap
//   thin       1.5 + 2 px strands, 8 px apart  thick   one 6 px strand
//   light      a low-contrast pair (hair x6) crossing  a 2.5 px strand crossing the main one
//   glove      wide dark area, edge 3 px from the strand, continuing past its tip (strand cut at x 52)
//   frame      8 px black band through the tip      brownframe  6.5 px band in a hair colour
//   mole       two dark dots by the strand and its tip   texture  five skin creases near the strand
//   combo      a pair + glove + creases
// The random draws use a fixed portable generator (splitmix64 + Box-Muller), so the noise differs from the
// prototype's numpy draws; the geometry and colours are the same.
#pragma once

#include <string>
#include <vector>

#include "strand_tracer.h"

namespace mr {
namespace tracer {
namespace scenes {

struct Strand {
  std::vector<Point> poly;
  double width;
};

struct Scene {
  int w = 112, h = 112;
  std::vector<float> rgb;     // w*h*3
  std::vector<float> alpha;   // base matte
  std::vector<float> region;  // soft region
  std::vector<double> gt;     // truth strand alpha
  std::vector<Strand> polys;  // truth centrelines (clipped as in the prototype)
  std::vector<unsigned char> dmask;  // distractor pixels (cover > 0.2, no truth)
  std::vector<std::vector<double>> creases;  // crease cover maps
};

const std::vector<std::string>& sceneNames();
Scene makeScene(const std::string& name, int seed);

struct Metrics {
  double rec = 0, fals = 0, distr = 0, sad = 0;  // rec is NaN when no strand was inside the region
  int wander = 0, early = 0, missed = 0;
};
Metrics metrics(const Scene& S, const std::vector<float>& A, const std::vector<Piece>& pieces);

// Mean over a set of runs: rec / false / distr / sad averaged (rec ignoring NaN), counts summed.
Metrics aggregate(const std::vector<Metrics>& m);

// Runs every scene x seeds with params p; returns per-scene aggregates (scene order) and the MEAN row in `mean`
// (as evaluate.py: the mean of the per-scene means, counts summed).
std::vector<Metrics> evaluate(const TracerParams& p, int seeds, Metrics& mean);

}  // namespace scenes
}  // namespace tracer
}  // namespace mr
