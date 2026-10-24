// Synthetic test data for Matte Merge (ofx/merge/CLAUDE.md "Reuse"): plugin 1's hair scenes
// (ofx/tools/synth.h makeHair: head and shoulders with 260 strands, a coarse base that loses them)
// plus a simulated detail matte with the faults a real one has, a matching picture in two
// encodings and a trimap. Deterministic per (w, h, busy, seed); design units at 1080 lines (k = h / 1080).
#pragma once

#include <string>
#include <vector>

namespace mmsynth {

struct MergeScene {
  std::string name;  // "hair_flat" / "hair_busy"
  int w = 0, h = 0;
  double k = 1;      // px per design unit
  std::vector<float> base;    // the coarse base matte (synth's matte: strands lost, body edge 3 px inside)
  // A base whose edge sits a uniform 3 px (at 1080) inside the truth body, 2 px soft, no hair: separates
  // registration accuracy from the coarse base's direction-dependent edge offset (synth's square
  // erosion: 3 px on the axes, 4.2 px on diagonals, plus the hair mass its blur swallowed).
  std::vector<float> baseUniform;
  std::vector<float> truth;   // the clean matte (body + real strands)
  std::vector<float> body;    // truth body alone (2 px edge, no strands)
  std::vector<float> bodySd;  // signed distance to the truth body edge, px, > 0 inside
  std::vector<float> strands; // real strands alone (alpha), truth = 1 - (1 - body)(1 - strands)
  // The detail matte as delivered: (body thickened by 2 px, real strands, fake strands, a blob),
  // Gaussian-blurred (sigma 0.8 px at 1080), then moved: detail(A(p)) = the unmoved one at p.
  std::vector<float> detail;
  std::vector<float> fake;    // the hallucinated strands alone, in base (truth) coordinates
  std::vector<float> blob;    // the floating blob alone, in base coordinates
  double blobX = 0, blobY = 0, blobR = 0;  // px
  // A(p) = c + scale (p - c) + (tx, ty), c = ((w - 1) / 2, (h - 1) / 2): where base pixel p is in the detail.
  double tx = 0, ty = 0, scale = 1;
  std::vector<float> trimap;  // 1 deeper than 6 px (at 1080) inside the body, 0 beyond 120 px outside, 0.5 between
  // The picture (w*h*3), synth's rgb read as scene-linear: display = sRGB curve, log = a Log3G10-like
  // curve. Real strands are in it, fake strands and the blob are not.
  std::vector<float> rgbLinear, rgbDisplay, rgbLog;
};

// busy = synth's hair_busy background. fakeStrands: hallucinated strands rooted on the head.
MergeScene makeMergeScene(int w, int h, bool busy, unsigned seed = 1, int fakeStrands = 40);

// The encodings, per channel.
float srgbEncode(float linear);
float log3g10Encode(float linear);

}  // namespace mmsynth
