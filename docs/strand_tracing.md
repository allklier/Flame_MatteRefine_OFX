# Strand tracing: candidate 2 for Focus Region strand rebuild (classical)

Status: PROPOSED (research + Python prototype, 2026-11-12). Nothing here is tested in a host.
Prototype: `tracer.py` (algorithm), `synth.py` (scenes with ground truth), `evaluate.py` (ablation),
`realcheck.py` (numbers-only check on one real frame), `diag.py` (per-piece debug numbers).

## 1. Problem

Inside a user-drawn soft region, rebuild missing thin dark hair strands as alpha. The strands leave a
solid matte, cross a 15-25 px gap where the base matte is ~0, and must not continue onto
non-hair dark structures next to them (thick dark edges such as a glove, a glasses-frame band, moles,
skin creases). Output: strand alpha, a confidence, and the list of open curve pieces.

## 2. Prior art (what was taken from where)

| Topic | Source | Used for |
|---|---|---|
| Hair orientation fields, oriented filters | Paris, Briceño, Sillion, "Capture of Hair Geometry from Multiple Images", ACM TOG 23(3) (SIGGRAPH 2004), https://artis.inrialpes.fr/Publications/2004/PBS04/ ; Luo et al., "Multi-View Hair Capture Using Orientation Fields", CVPR 2012 | per-pixel line direction (here from the Hessian instead of a Gabor bank) |
| Strand tracing in an orientation field | Chai et al., "Single-View Hair Modeling for Portrait Manipulation", ACM TOG 31(4) 116 (SIGGRAPH 2012), dblp journals/tog/ChaiWWYGZ12 | step-wise tracing that follows the orientation and stops on low confidence or large turn |
| Hessian line / vessel filters | Frangi et al., "Multiscale vessel enhancement filtering", MICCAI 1998 (from memory); Jerman et al., "Enhancement of Vascular Structures in 3D and 2D Angiographic Images", IEEE TMI 35(9):2107-2118, 2016, doi 10.1109/TMI.2016.2550102; Law & Chung, Optimally Oriented Flux, ECCV 2008 (from memory) | dark-line measure + direction at sigma 1.2 |
| Line position + width with sub-pixel accuracy | Steger, "An Unbiased Detector of Curvilinear Structures", IEEE PAMI 20(2):113-125, 1998, doi 10.1109/34.659930 | cross-profile: centre, width from edge points on both sides, asymmetry |
| Minimal paths / live-wire | Cohen & Kimmel, "Global Minimum for Active Contour Models: A Minimal Path Approach", IJCV 24(1):57-78, 1997; Mortensen & Barrett, "Intelligent Scissors", SIGGRAPH 1995 (from memory); Sethian fast marching (from memory) | NOT used for open tracing (needs both ends, and a cheap path along a glove edge wins); fits gap bridging between two known ends |
| Path openings | Luengo Hendriks, "Constrained and Dimensionality-Independent Path Openings" (IEEE TIP), DIPlib `PathOpening` docs (constrained mode keeps lines within 22.5 deg) https://ftp.imphys.tudelft.nl/diplib/2.8/docs/reference/PathOpening.html | alternative detector for long thin structures; not needed at 25 px scale |
| Curve completion | Kimia, Frankel, Popescu, "Euler Spiral for Shape Completion", IJCV 54, 2003; Mumford elastica (1994), Williams & Jacobs stochastic completion fields (1997), tensor voting (Medioni et al. 2000), association field (Field, Hayes, Hess 1993) - all from memory | end-to-end join cost: bending energy of a cubic Hermite between the two end tangents (a cheap stand-in for elastica / Euler spiral at small angles) |
| Steered tracing in vessels / neurites | Can et al. 1999 exploratory vessel tracing (from memory) | candidate-direction fan at each step |

"From memory" = citation not re-checked in this session (UNVERIFIED details such as pages).

## 3. Recommended algorithm (as prototyped)

All on the picture's log2 luminance `l` (Gaussian 0.5 px) and log-chroma `c1=log2(R/G)`, `c2=log2(B/G)`
(Gaussian 0.7 px). Linear input; no colour-space conversion (luminance weights are a nuisance
parameter: Rec.709 weights were used on an AP1 picture).

1. **Detectors**: black top-hat `bth = closing7x7(l) - l` (the existing detector); Hessian at
   sigma 1.2 (scale-normalised): line measure `max(lmax,0) - 0.5|min(lmin,0)| - max(lmin,0)` (penalises
   blobs, where both eigenvalues are positive) and the along-line unit vector.
2. **Cross-profile** at a point with normal n: 29 samples of `l` at t = -7..7 step 0.5 (bilinear).
   Centre = argmin within |t| <= 1 (sub-pixel re-centring, also stops jumping to a neighbouring strand).
   Side backgrounds bl, br = 80th percentile of each side beyond 1.5 px (robust to a second strand).
   depth d = min(bl,br) - lc (stops); width w = FWHM from the two half-depth crossings;
   asym = |bl - br|; chroma shift per stop v = (c_centre - c_sides)/d; B = 2^((bl+br)/2) (linear background);
   double-dip flag (two minima 3-8 px apart, bump >= 25 % of depth).
3. **Step score** H in 0..1 for a candidate point given a reference (d, w, v):
   `H = sigmoid((d - 0.25)/0.06) * exp(-0.5 (ln(w/w_ref)/0.35)^2) [0 if w > 7 or w < 0.5]`
   `  * exp(-0.5 (asym/(d+0.05)/0.6)^2) * exp(-0.5 |v - v_ref|^2/0.15^2)`
   `  * min(1, (d/d_ref / (1/3))^2) * exp(-0.5 (angle(step, hessian dir)/20deg)^2)`.
4. **Tracing**: step 1.5 px; candidate directions = current +-22.5 deg in 7.5 deg steps, each scored
   by `H * exp(-0.5 (turn/12deg)^2)`; best wins; the realised direction is taken from the re-centred
   point. Running confidence `C = 0.6 C + 0.4 H`. Stop when C < 0.30, or H < 0.20 for 3 steps
   (gap tolerance 4.5 px), or turn over the last 6 steps > 60 deg, or leaving the search mask (region
   dilated 4 px), or entering base alpha > 0.5 (anchored), or touching an earlier piece (junction),
   or length > 1.5 x the search box diagonal. Trim trailing steps with H < 0.40. The width reference
   adapts slowly (0.7 old + 0.3 median of last 4).
5. **Seeds**: (a) contact seeds = Hessian local maxima with bth > 0.35 within 6 px of the solid matte,
   direction pointing away from the matte, traced out and back to the matte (root); (b) free seeds =
   remaining maxima inside the region, at least 1 px from used pixels, score >= 0.6, traced both ways.
   Used pixels are marked with radius w/2, so a parallel strand >= w/2+1 px away can still seed:
   two strands with a gap stay two pieces when the picture resolves them.
6. **Connect** (only low-cost, unambiguous joins):
   - piece end -> solid matte: nearest solid pixel within 12 px, end tangent within 35 deg of the chord,
     cost `J = 0.5*bend*arc + 0.3*L/12 + 0.6*(1-support) <= 1.0` (bend = integral of curvature^2 along
     a cubic Hermite between the tangents; support = fraction of the bridge with bth > 0.175)
     -> the piece is anchored;
   - end <-> end: L <= 12 px, both end tangents within 35 deg of the chord, J (above + 0.5|ln(w1/w2)|)
     <= 1.0, mutual best, and best < 0.8 x second best (otherwise abstain).
7. **Hair context**: median (d, w, v) of anchored pieces' confident steps. Unanchored pieces are
   re-scored against it (width tolerance 0.5 in log) and kept only with conf >= 0.7 and length >= 4 w.
   No anchored piece: self-reference from the longest piece that passes absolute tests (FWHM <= 7,
   median asym/d <= 0.3, d >= 0.5 stops), conf x 0.8, flagged.
8. **Confidence** per piece = median H (+0.15 if anchored); keep >= 0.45.
9. **Alpha**: matting equation along the piece, `alpha = clamp((B - I)/(B - F))` inside a capsule of
   radius w'/2 + 1 (w' = sqrt(max(w^2 - 1.5, 0.5)), PSF removed); F = 10th percentile of the core
   luminance where FWHM >= 3 px, else the 10th percentile of the solid matte's luminance within 12 px of
   its edge in the search area. Joins render as capsules. Multiply by the soft region.

Port notes: everything is float arrays + bilinear sampling + a few loops; detectors are separable
Gaussians and a 7x7 min/max. Per step: 7 candidates x 29 samples. A 50x50 region traces in a few
ms in C++ (UNVERIFIED: estimate, the Python prototype takes ~0.3 s per 112x112 scene).

## 4. Synthetic evaluation (6 seeds x 12 scenes, 112x112, strands 1.5-6 px, skin-like noisy background)

Scenes: single, pair (3.5 px, 2 px gap), thin (1.5 + 2 px), thick (6 px), light (low-contrast pair),
crossing, glove (wide dark area, edge 3 px from the strand, continuing past the tip), frame (8 px black
band through the tip), brownframe (6.5 px band in hair colour), mole (two dark dots next to the
strand and its tip), texture (skin creases near the strand), combo (pair + glove + creases).
Metrics (inside the region): rec = fraction of truth centreline within 1.5 px of a kept trace;
false = alpha > 1 px from any strand / truth alpha; distr = alpha on distractors / truth alpha;
sad = |alpha - truth| / truth; wander = pieces with >= 4 consecutive steps > 2.5 px off truth;
early = strands with rec 0.1-0.8; missed = rec < 0.1. wander/early/missed are counts over 72 runs.

| rules | rec | false | distr | sad | wander | early | missed |
|---|---|---|---|---|---|---|---|
| all | 0.97 | 0.01 | 0.01 | 0.30 | 1 | 2 | 0 |
| - turn limit | 0.98 | 0.10 | 0.09 | 0.39 | 11 | 1 | 0 |
| - width consistency | 0.98 | 0.17 | 0.17 | 0.46 | 14 | 2 | 0 |
| - profile symmetry | 0.98 | 0.04 | 0.03 | 0.33 | 5 | 1 | 0 |
| - colour shift | 0.97 | 0.01 | 0.01 | 0.30 | 1 | 2 | 0 |
| - contrast floor | 0.97 | 0.02 | 0.02 | 0.31 | 3 | 2 | 0 |
| - orientation agreement | 0.97 | 0.03 | 0.02 | 0.31 | 2 | 2 | 0 |
| - confidence decay (raw depth stop) | 0.97 | 0.04 | 0.04 | 0.33 | 5 | 2 | 0 |
| - connect | 0.96 | 0.01 | 0.01 | 0.31 | 1 | 2 | 1 |
| - length budget | 0.97 | 0.01 | 0.01 | 0.30 | 1 | 2 | 0 |
| - hair context | 0.97 | 0.10 | 0.09 | 0.39 | 9 | 2 | 0 |
| none | 0.99 | 0.31 | 0.32 | 0.58 | 37 | 1 | 0 |
| all, capsule alpha | 0.97 | 0.01 | 0.01 | 0.33 | 1 | 2 | 0 |

Per scene with all rules: wandering is gone on frame / brownframe (none: 12 + 12 events, distr 1.69 /
1.43) and mole drops to 1 event (none: 11). The cost is recall at a strand tip that touches a band
(frame 0.82, brownframe 0.91: the trace stops 2-4 px early, i.e. it abstains at the contact).
Width, turn and hair context carry most of the protection; profile symmetry and decay add some.
Colour and budget show no effect here: the synthetic palette makes the colour shift weak, and the
region already bounds length. They are kept as cheap guards; tune on real data before relying on them.
The synthetic glove is not adversarial (none: 1 event): its one-sided profile is rejected by depth alone.

## 5. Real frame (numbers only; one frame, two drawn shapes, no ground truth)

Shapes: 327 px and 260 px (about 29x23 and 24x24), touching the soft matte edge (base alpha in the
shapes mean 0.08 / 0.07, max 0.67 / 0.57). Bright-closing ridge pixels in the shapes: 114 / 94.
- Shape 1: 1 piece, 27 px, 100 % inside the drawn shape, anchored by a 6.4 px matte bridge (J 0.54),
  conf 0.79, FWHM median 5.3 px, depth 1.05 stops, asym 0.24, curvature median 0.087 rad/px (max 0.30),
  net turn 78 deg. Double-dip profiles 0/18.
- Shape 2: 1 piece, 22.5 px, 100 % inside, not anchored (no matte end within 12 px in a compatible
  direction), self-referenced hair model, conf 0.64, FWHM 6.0, depth 0.96, net turn 74 deg, 0/15 double dips.
- The two strands per lock are not separated: FWHM 5.3-6.0 px with no double-dip profile means the
  picture at this scale does not resolve the gap; the tracer follows the lock as one piece.
- Misleading structures: with the shapes grown by 10 / 20 px, the no-rules tracer adds 1-2 extra pieces
  per shape (depth 0.43-0.64 stops, asym/depth 0.6-1.6 = one-sided, FWHM 4-7) and puts 106-401 alpha
  units outside the drawn shapes; with all rules those pieces are rejected. With the 10 px growth, the
  lock of shape 1 enters the grown shape 2 search and is traced there (the shapes are close); the
  final output is clipped by the drawn region, so this only matters if a region is drawn loosely.
- The lock of shape 2 is borderline (conf 0.64 self-referenced, 0.72 against shape 1's model); at 20 px
  growth it fell below the 0.7 unanchored threshold and was dropped (stopped-too-early / abstain).

## 6. Limits

- Two strands with a 1-2 px gap are only separated when the picture resolves the gap (synthetic yes;
  the real frame no). Then the output is one lock-wide alpha, not two strands.
- Alpha amplitude depends on the hair luminance F; for light hair with no thick core the matte's F is
  too dark (light scene: sad 0.57 with all rules vs 0.21 with none; cause not isolated, F choice suspected).
- Real curvature (0.09 rad/px median, 74-78 deg over ~25 px) is within the limits, but the turn limit is
  the main brake against bands; very curly hair would need a looser window (30 deg/step, 90 deg/9 px
  changed nothing in either test).
- Hair over a dark band (crossing it) is not traced across: abstain by design.
- Thresholds were tuned on synthetic scenes only (risk of fitting them); one real frame is no validation.

## 7. Open questions

1. Should unanchored pieces inside the region be kept at all (region = user's assertion of hair) or
   only when anchored (safer)? Prototype: kept with conf >= 0.7.
2. Hair context per drawn shape or per frame (shape 2 borrowed a width 30 % narrower than its own)?
3. Combine with candidate 1 (model): the tracer's pieces could gate the model's alpha (accept model
   alpha only within w/2+2 of a confident piece) rather than produce alpha itself.
4. Temporal: trace per frame, or propagate pieces between frames (flicker risk at the 0.7 threshold).
