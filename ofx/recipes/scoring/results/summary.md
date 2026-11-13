# Hair Locks end-to-end scores (er_score, Edge Recipes WP8)

Synthetic sequences only (er_scenes.h; `wp6` = the vote system test's sequence), driven through the Flame-like host. CLASSICAL PATH ONLY: Edge Model is off in every row (no model weights on the scoring machine), so the model candidate, Strand Gain and Edge Model are NOT scored. Picture Colour = Bypass (linear scenes). 85 scenes, 3398 rows.

Metrics (inside the drawn region, mask > 0; target = base + (1 - base) strand coverage): coverage = added alpha that is truth / missing truth; IoU at 0.5 on pixels the base had below 0.5; SAD = |out - target| / missing truth; false in region = alpha added where the truth adds < 0.02, / missing truth; distractor alpha = alpha added on distractor pixels / missing truth; hit rate = share of frames with more than 1.0 alpha unit on a distractor; flicker excess = motion-compensated mean |a_t - a_t-1| in the region minus the same for the target; edge error = mean |out - true head| on the head's edge band outside the region.

## 1. Default setting (Rebuild on, Min Confidence 0.45, Feather 0.25 %, Loose on, Vote on, Snap 1, Temporal 1), per scene set

| set | coverage | IoU@0.5 | SAD/truth | false in region | distractor alpha | distractor hit rate | flicker excess | n |
|---|---|---|---|---|---|---|---|
| main | 0.520 | 0.475 | 0.566 | 0.0793 | 0.0095 | 0.327 | 0.0002 | 48 |
| distractor | 0.559 | 0.526 | 0.522 | 0.0749 | 0.0094 | 0.450 | -0.0005 | 20 |
| detached | 0.543 | 0.494 | 0.528 | 0.0649 | 0.0017 | 0.167 | 0.0003 | 12 |
| vote | 0.591 | 0.554 | 0.466 | 0.0503 | 0.0000 | 0.000 | 0.0034 | 4 |
| wp6 | 0.941 | 0.926 | 0.111 | 0.0419 | 0.0000 | 0.000 | 0.0114 | 1 |

### Per look (main + distractor sets)

| look | coverage | IoU@0.5 | SAD/truth | false in region | distractor alpha | distractor hit rate | flicker excess | n |
|---|---|---|---|---|---|---|---|
| dark_skin | 0.927 | 0.902 | 0.131 | 0.0507 | 0.0201 | 0.513 | 0.0014 | 17 |
| dark_texture | 0.908 | 0.871 | 0.232 | 0.1289 | 0.0128 | 0.487 | 0.0038 | 17 |
| dark_shadow | 0.260 | 0.180 | 0.830 | 0.0854 | 0.0042 | 0.303 | 0.0015 | 17 |
| light_skin | 0.030 | 0.006 | 1.019 | 0.0472 | 0.0009 | 0.151 | -0.0068 | 17 |

### Per strand width (main set)

| width px | coverage | IoU@0.5 | SAD/truth | false in region | distractor alpha | distractor hit rate | flicker excess | n |
|---|---|---|---|---|---|---|---|
| 1.5 | 0.487 | 0.411 | 0.661 | 0.1365 | 0.0145 | 0.429 | -0.0027 | 12 |
| 2.5 | 0.571 | 0.508 | 0.518 | 0.0815 | 0.0019 | 0.202 | -0.0006 | 12 |
| 4.0 | 0.546 | 0.524 | 0.518 | 0.0593 | 0.0102 | 0.488 | 0.0006 | 12 |
| 6.0 | 0.474 | 0.457 | 0.568 | 0.0400 | 0.0114 | 0.190 | 0.0034 | 12 |

### Per region looseness (main set)

| margin px | coverage | IoU@0.5 | SAD/truth | false in region | distractor alpha | distractor hit rate | flicker excess | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.560 | 0.524 | 0.495 | 0.0481 | 0.0000 | 0.000 | 0.0008 | 16 |
| 10 | 0.467 | 0.416 | 0.618 | 0.0789 | 0.0170 | 0.473 | -0.0008 | 16 |
| 20 | 0.532 | 0.484 | 0.585 | 0.1110 | 0.0116 | 0.509 | 0.0005 | 16 |

### Per look x width, coverage / false in region (main set, default)

| look | 1.5 | 2.5 | 4.0 | 6.0 |
|---|---|---|---|---|
| dark_skin | 0.902 / 0.060 | 0.976 / 0.028 | 0.955 / 0.045 | 0.831 / 0.059 |
| dark_texture | 0.863 / 0.273 | 0.956 / 0.157 | 0.942 / 0.075 | 0.821 / 0.036 |
| dark_shadow | 0.147 / 0.137 | 0.323 / 0.091 | 0.255 / 0.079 | 0.223 / 0.041 |
| light_skin | 0.037 / 0.075 | 0.030 / 0.050 | 0.030 / 0.038 | 0.023 / 0.024 |

## 2. Distractors (distractor set: width 3, region 10 px loose, every look; default and variants)

| distractor | setting | distractor alpha | hit rate | coverage | false in region | crossing strand coverage |
|---|---|---|---|---|---|---|
| glove | default | 0.0352 | 1.000 | 0.506 | 0.0975 | - |
| glove | snap_off | 0.0006 | 0.036 | 0.529 | 0.0104 | - |
| glove | loose_off | 0.0351 | 1.000 | 0.504 | 0.0971 | - |
| glove | vote_off | 0.0351 | 1.000 | 0.507 | 0.0974 | - |
| glove | mc0.70_fe0.25 | 0.0352 | 1.000 | 0.506 | 0.0974 | - |
| glasses | default | 0.0038 | 0.500 | 0.566 | 0.0751 | - |
| glasses | snap_off | 0.0003 | 0.036 | 0.554 | 0.0136 | - |
| glasses | loose_off | 0.0038 | 0.500 | 0.564 | 0.0750 | - |
| glasses | vote_off | 0.0038 | 0.500 | 0.567 | 0.0755 | - |
| glasses | mc0.70_fe0.25 | 0.0038 | 0.500 | 0.565 | 0.0750 | - |
| mole | default | 0.0000 | 0.000 | 0.582 | 0.0679 | - |
| mole | snap_off | 0.0000 | 0.000 | 0.547 | 0.0140 | - |
| mole | loose_off | 0.0000 | 0.000 | 0.580 | 0.0678 | - |
| mole | vote_off | 0.0000 | 0.000 | 0.583 | 0.0682 | - |
| mole | mc0.70_fe0.25 | 0.0000 | 0.000 | 0.580 | 0.0678 | - |
| crossing | default | 0.0000 | 0.000 | 0.563 | 0.0609 | 0.561 |
| crossing | snap_off | 0.0000 | 0.000 | 0.546 | 0.0140 | 0.565 |
| crossing | loose_off | 0.0000 | 0.000 | 0.506 | 0.0567 | 0.412 |
| crossing | vote_off | 0.0000 | 0.000 | 0.566 | 0.0611 | 0.557 |
| crossing | mc0.70_fe0.25 | 0.0000 | 0.000 | 0.560 | 0.0606 | 0.557 |
| tipband | default | 0.0080 | 0.750 | 0.579 | 0.0730 | - |
| tipband | snap_off | 0.0013 | 0.179 | 0.572 | 0.0139 | - |
| tipband | loose_off | 0.0077 | 0.714 | 0.571 | 0.0722 | - |
| tipband | vote_off | 0.0080 | 0.750 | 0.581 | 0.0735 | - |
| tipband | mc0.70_fe0.25 | 0.0080 | 0.750 | 0.579 | 0.0730 | - |

Distractor hit rate per look (distractor set, default):

| distractor | dark_skin | dark_texture | dark_shadow | light_skin |
|---|---|---|---|---|
| glove | 1.000 | 1.000 | 1.000 | 1.000 |
| glasses | 1.000 | 1.000 | 0.000 | 0.000 |
| mole | 0.000 | 0.000 | 0.000 | 0.000 |
| tipband | 1.000 | 1.000 | 1.000 | 0.000 |

### Detached lock 0 (starts 18 px off the head: unanchored), per Min Confidence and Loose Pieces

| setting | coverage | SAD | false in region | distractor alpha | distractor hit rate |
|---|---|---|---|---|---|
| mc0.25_fe0.25 | 0.716 | 0.362 | 0.0715 | 0.0022 | 0.222 |
| mc0.35_fe0.25 | 0.716 | 0.362 | 0.0715 | 0.0022 | 0.222 |
| default | 0.716 | 0.362 | 0.0715 | 0.0022 | 0.222 |
| mc0.55_fe0.25 | 0.715 | 0.363 | 0.0714 | 0.0022 | 0.222 |
| mc0.70_fe0.25 | 0.699 | 0.378 | 0.0703 | 0.0021 | 0.206 |
| loose_off | 0.444 | 0.613 | 0.0529 | 0.0000 | 0.000 |

(light_skin excluded here: the tracer finds no pieces on that look.)

## 3. Sensitivity (main + distractor + detached sets; one control moved from the default)

| control | value | coverage | SAD | false in region | distractor alpha | flicker excess | edge error outside | identical to default |
|---|---|---|---|---|---|---|---|---|
| Min Confidence | 0.25 | 0.533 | 0.549 | 0.0761 | 0.0083 | 0.0000 | 0.1589 | 74/80 |
| Min Confidence | 0.35 | 0.533 | 0.549 | 0.0761 | 0.0083 | 0.0000 | 0.1589 | 76/80 |
| Min Confidence | 0.45 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Min Confidence | 0.55 | 0.533 | 0.550 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 72/80 |
| Min Confidence | 0.70 | 0.530 | 0.553 | 0.0758 | 0.0083 | -0.0001 | 0.1589 | 65/80 |
| Region Feather % | 0.0 | 0.533 | 0.549 | 0.0761 | 0.0083 | 0.0000 | 0.1589 | 0/80 |
| Region Feather % | 0.25 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Region Feather % | 0.5 | 0.533 | 0.549 | 0.0758 | 0.0083 | 0.0000 | 0.1589 | 0/80 |
| Region Feather % | 1.0 | 0.533 | 0.548 | 0.0743 | 0.0083 | -0.0000 | 0.1589 | 0/80 |
| Region Feather % | 2.0 | 0.532 | 0.543 | 0.0671 | 0.0075 | -0.0003 | 0.1589 | 0/80 |
| Strand Gain | 0.8 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Strand Gain | 1.0 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Strand Gain | 1.5 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Loose Pieces | on | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Loose Pieces | off | 0.494 | 0.584 | 0.0726 | 0.0073 | 0.0000 | 0.1589 | 46/80 |
| Temporal Amount | 0 | 0.533 | 0.547 | 0.0751 | 0.0083 | 0.0002 | 0.1904 | 0/80 |
| Temporal Amount | 0.5 | 0.533 | 0.548 | 0.0756 | 0.0083 | 0.0001 | 0.1675 | 0/80 |
| Temporal Amount | 1 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Temporal | off | 0.533 | 0.547 | 0.0751 | 0.0083 | 0.0002 | 0.1904 | 0/80 |
| Snap Amount | 0 | 0.520 | 0.515 | 0.0177 | 0.0018 | -0.0004 | 0.1589 | 0/80 |
| Snap Amount | 0.5 | 0.529 | 0.528 | 0.0469 | 0.0051 | -0.0005 | 0.1589 | 0/80 |
| Snap Amount | 1 | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Snap to Picture | off | 0.520 | 0.515 | 0.0177 | 0.0018 | -0.0004 | 0.1589 | 0/80 |
| Temporal Vote | on | 0.533 | 0.549 | 0.0760 | 0.0083 | 0.0000 | 0.1589 | 80/80 |
| Temporal Vote | off | 0.534 | 0.550 | 0.0773 | 0.0093 | 0.0007 | 0.1583 | 0/80 |
| Rebuild Detail | off | 0.043 | 1.004 | 0.0452 | 0.0019 | -0.0070 | 0.1588 | 0/80 |
| Snap only (Rebuild, Temporal off) | - | 0.043 | 1.001 | 0.0442 | 0.0018 | -0.0068 | 0.1904 | 0/80 |

## 4. Temporal Vote x Temporal (flicker of the strand alpha; ends)

| set | vote | Temporal | coverage | flicker | truth flicker | coverage std over frames | coverage end frames | coverage middle frames | dip-frame coverage | spurious stroke alpha | edge error outside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | on | on | 0.520 | 0.0120 | 0.0118 | 0.016 | 0.518 | 0.520 | - | - | 0.1430 |
| main | off | on | 0.521 | 0.0130 | 0.0118 | 0.021 | 0.521 | 0.519 | - | - | 0.1426 |
| main | on | off | 0.520 | 0.0123 | 0.0118 | 0.016 | 0.518 | 0.520 | - | - | 0.1699 |
| main | off | off | 0.521 | 0.0132 | 0.0118 | 0.021 | 0.521 | 0.519 | - | - | 0.1699 |
| distractor | on | on | 0.559 | 0.0105 | 0.0110 | 0.017 | 0.558 | 0.559 | - | - | 0.1826 |
| distractor | off | on | 0.561 | 0.0108 | 0.0110 | 0.020 | 0.561 | 0.558 | - | - | 0.1818 |
| distractor | on | off | 0.559 | 0.0107 | 0.0110 | 0.017 | 0.558 | 0.559 | - | - | 0.2215 |
| distractor | off | off | 0.561 | 0.0110 | 0.0110 | 0.020 | 0.561 | 0.558 | - | - | 0.2215 |
| detached | on | on | 0.543 | 0.0105 | 0.0102 | 0.042 | 0.531 | 0.547 | - | - | 0.1826 |
| detached | off | on | 0.540 | 0.0109 | 0.0102 | 0.044 | 0.527 | 0.543 | - | - | 0.1820 |
| detached | on | off | 0.543 | 0.0108 | 0.0102 | 0.042 | 0.531 | 0.547 | - | - | 0.2205 |
| detached | off | off | 0.540 | 0.0111 | 0.0102 | 0.044 | 0.527 | 0.543 | - | - | 0.2205 |
| vote | on | on | 0.591 | 0.0216 | 0.0183 | 0.060 | 0.590 | 0.580 | 0.438 | 0.000 | 0.1801 |
| vote | off | on | 0.588 | 0.0221 | 0.0183 | 0.065 | 0.590 | 0.575 | 0.418 | 0.000 | 0.1792 |
| vote | on | off | 0.591 | 0.0220 | 0.0183 | 0.060 | 0.590 | 0.580 | 0.438 | 0.000 | 0.2228 |
| vote | off | off | 0.588 | 0.0226 | 0.0183 | 0.065 | 0.590 | 0.575 | 0.418 | 0.000 | 0.2228 |
| wp6 | on | on | 0.941 | 0.0236 | 0.0122 | 0.125 | 0.996 | 0.931 | 0.668 | 3.435 | 0.0067 |
| wp6 | off | on | 0.928 | 0.0264 | 0.0122 | 0.167 | 0.996 | 0.914 | 0.601 | 69.298 | 0.0069 |
| wp6 | on | off | 0.941 | 0.0236 | 0.0122 | 0.125 | 0.996 | 0.931 | 0.668 | 3.435 | 0.0000 |
| wp6 | off | off | 0.928 | 0.0264 | 0.0122 | 0.167 | 0.996 | 0.914 | 0.601 | 69.298 | 0.0000 |

## 5. Proxy (render scale 0.5: the scene box-downsampled 2x) vs full scale, default setting

| set | width px (full res) | coverage 1.0 | coverage 0.5 | SAD 1.0 | SAD 0.5 | false in region 1.0 | 0.5 |
|---|---|---|---|---|---|---|---|
| main | 1.5 | 0.487 | 0.319 | 0.661 | 0.830 | 0.1365 | 0.0897 |
| main | 2.5 | 0.571 | 0.441 | 0.518 | 0.625 | 0.0815 | 0.0557 |
| main | 4.0 | 0.546 | 0.556 | 0.518 | 0.508 | 0.0593 | 0.0415 |
| main | 6.0 | 0.474 | 0.637 | 0.568 | 0.412 | 0.0400 | 0.0415 |
| distractor | 3.0 | 0.559 | 0.485 | 0.522 | 0.582 | 0.0749 | 0.0499 |
| detached | 2.5 | 0.557 | 0.410 | 0.530 | 0.678 | 0.0794 | 0.0667 |
| detached | 3.0 | 0.495 | 0.543 | 0.575 | 0.548 | 0.0650 | 0.0727 |
| detached | 4.0 | 0.575 | 0.637 | 0.479 | 0.452 | 0.0502 | 0.0610 |

## 6. Contract checks (every frame of every row)

- below the base where the region weight is 1: 0 px over all 3398 rows
- rows with Temporal off (339): changed outside the region 0 px; alpha added outside where the truth is 0: 0 px; below the base inside the region 0 px
- rows with Temporal on: Temporal changes the matte outside the region by design (largest change 1.000); see the edge error columns for whether it helps

## 7. Draft Analyze rules (WP10 calibration)

Per scene (main + distractor + detached sets, scale 1) the best setting of each control Analyze would set, from the grid rows; a value is ACCEPTABLE when its score is within the tolerance of the scene's best. Inside controls (Rebuild on/off, Min Confidence at Feather 0.25, Feather at Min Confidence 0.45) are scored by SAD (tolerance max(0.02, 5 % of the best)); Temporal Amount by the edge error outside the region (tolerance 5 % of the best). A rule = one measurement of the centre frame and one threshold (a decision stump), fitted by exhaustive search; hit rate = share of scenes where the rule's value is acceptable; 2-fold = fitted on even, tested on odd scenes and vice versa. Strand Gain, Edge Model: not calibrated (model not scored). Jump Repair Min Size: not calibrated (no jump scenes here; decision 42).

| control | best value counts (over 80 scenes) | constant rule: value, hit rate | stump rule | stump hit rate | 2-fold hit rate |
|---|---|---|---|---|---|
| Rebuild Detail (0 off / 1 on) | 1: 80 | 1, 1.00 | (constant) < 0 ? 1 : 1 | 1.00 | 1.00 |
| Min Confidence | 0.25: 79, 0.7: 1 | 0.25, 1.00 | (constant) < 0 ? 0.25 : 0.25 | 1.00 | 1.00 |
| Region Feather % | 0: 2, 1: 6, 2: 72 | 2, 0.97 | regionPct < 3.324 ? 2 : 1 | 0.99 | 0.95 |
| Temporal Amount | 0: 18, 1: 62 | 1, 0.78 | jitterMeasured < 0.3095 ? 0 : 1 | 0.99 | 0.99 |

The default setting's SAD regret against the scene's best grid row: mean 0.007; scenes with regret > 0.05: 1 of 80.

### Per scene: measurements (centre frame) and the best values

| scene | region % | soft | luma p1 / p99 | base cov | pieces (kept) | conf median | width med | jitter | best rebuild | best mc | best feather | best tm | SAD default / best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dark_skin_w1.5_m0_none_j0.00_s1 | 1.47 | 0.45 | 0.195 / 0.569 | 0.211 | 3 (3) | 0.94 | 1.6 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.063 / 0.057 |
| dark_skin_w1.5_m10_glove_j0.00_s2 | 3.21 | 0.29 | 0.155 / 0.564 | 0.262 | 3 (3) | 0.95 | 1.6 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.321 / 0.318 |
| dark_skin_w1.5_m20_glasses_j0.00_s3 | 5.47 | 0.22 | 0.158 / 0.566 | 0.295 | 3 (3) | 0.98 | 1.7 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.142 / 0.140 |
| dark_skin_w2.5_m0_mole_j0.00_s4 | 1.59 | 0.43 | 0.195 / 0.568 | 0.217 | 3 (3) | 1.07 | 2.2 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.036 / 0.032 |
| dark_skin_w2.5_m10_crossing_j0.00_s5 | 3.36 | 0.28 | 0.196 / 0.568 | 0.266 | 5 (5) | 1.06 | 2.2 | 0.3093 | 1 | 0.25 | 1 | 0 | 0.045 / 0.044 |
| dark_skin_w2.5_m20_tipband_j0.00_s6 | 5.67 | 0.21 | 0.193 / 0.570 | 0.297 | 3 (3) | 1.06 | 2.3 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.106 / 0.104 |
| dark_skin_w4.0_m0_none_j0.75_s7 | 1.76 | 0.41 | 0.192 / 0.569 | 0.226 | 3 (3) | 1.01 | 3.4 | 0.3181 | 1 | 0.25 | 2 | 1 | 0.028 / 0.024 |
| dark_skin_w4.0_m10_glove_j0.75_s8 | 3.60 | 0.27 | 0.156 / 0.565 | 0.272 | 3 (3) | 0.97 | 3.4 | 0.3084 | 1 | 0.25 | 2 | 1 | 0.143 / 0.140 |
| dark_skin_w4.0_m20_glasses_j0.75_s9 | 5.96 | 0.21 | 0.159 / 0.565 | 0.302 | 3 (3) | 0.99 | 3.4 | 0.3147 | 1 | 0.25 | 2 | 1 | 0.111 / 0.107 |
| dark_skin_w6.0_m0_mole_j0.75_s10 | 2.01 | 0.38 | 0.196 / 0.562 | 0.237 | 6 (6) | 0.95 | 5.5 | 0.3105 | 1 | 0.25 | 2 | 1 | 0.062 / 0.059 |
| dark_skin_w6.0_m10_crossing_j0.75_s11 | 3.93 | 0.26 | 0.194 / 0.567 | 0.277 | 3 (3) | 0.93 | 5.5 | 0.3178 | 1 | 0.25 | 2 | 1 | 0.428 / 0.426 |
| dark_skin_w6.0_m20_tipband_j0.75_s12 | 6.37 | 0.20 | 0.194 / 0.568 | 0.304 | 5 (5) | 0.89 | 5.6 | 0.3134 | 1 | 0.25 | 2 | 1 | 0.203 / 0.150 |
| dark_texture_w1.5_m0_none_j1.50_s13 | 1.47 | 0.45 | 0.194 / 0.584 | 0.205 | 3 (3) | 0.97 | 1.7 | 0.3413 | 1 | 0.25 | 2 | 1 | 0.266 / 0.224 |
| dark_texture_w1.5_m10_glove_j1.50_s14 | 3.21 | 0.29 | 0.155 / 0.584 | 0.260 | 3 (3) | 0.91 | 1.6 | 0.3566 | 1 | 0.25 | 2 | 1 | 0.536 / 0.520 |
| dark_texture_w1.5_m20_glasses_j1.50_s15 | 5.47 | 0.22 | 0.158 / 0.589 | 0.293 | 3 (3) | 0.91 | 1.6 | 0.3551 | 1 | 0.25 | 2 | 1 | 0.492 / 0.464 |
| dark_texture_w2.5_m0_mole_j1.50_s16 | 1.59 | 0.43 | 0.196 / 0.588 | 0.213 | 3 (3) | 1.06 | 2.3 | 0.3231 | 1 | 0.25 | 2 | 1 | 0.168 / 0.138 |
| dark_texture_w2.5_m10_crossing_j1.50_s17 | 3.36 | 0.28 | 0.194 / 0.591 | 0.262 | 5 (5) | 1.04 | 2.2 | 0.3527 | 1 | 0.25 | 1 | 1 | 0.196 / 0.192 |
| dark_texture_w2.5_m20_tipband_j1.50_s18 | 5.67 | 0.21 | 0.194 / 0.590 | 0.302 | 3 (3) | 1.08 | 2.2 | 0.3549 | 1 | 0.25 | 2 | 1 | 0.285 / 0.269 |
| dark_texture_w4.0_m0_none_j0.00_s19 | 1.76 | 0.41 | 0.195 / 0.583 | 0.224 | 3 (3) | 0.98 | 3.4 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.082 / 0.072 |
| dark_texture_w4.0_m10_glove_j0.00_s20 | 3.60 | 0.27 | 0.155 / 0.586 | 0.270 | 3 (3) | 0.91 | 3.4 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.185 / 0.182 |
| dark_texture_w4.0_m20_glasses_j0.00_s21 | 5.96 | 0.21 | 0.158 / 0.591 | 0.300 | 3 (3) | 1.00 | 3.4 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.152 / 0.149 |
| dark_texture_w6.0_m0_mole_j0.00_s22 | 2.01 | 0.38 | 0.193 / 0.585 | 0.233 | 6 (6) | 0.88 | 5.6 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.067 / 0.060 |
| dark_texture_w6.0_m10_crossing_j0.00_s23 | 3.93 | 0.26 | 0.193 / 0.589 | 0.276 | 5 (5) | 0.97 | 5.5 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.441 / 0.439 |
| dark_texture_w6.0_m20_tipband_j0.00_s24 | 6.37 | 0.20 | 0.192 / 0.587 | 0.304 | 5 (5) | 0.96 | 5.5 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.144 / 0.142 |
| dark_shadow_w1.5_m0_none_j0.75_s25 | 1.47 | 0.45 | 0.194 / 0.259 | 0.213 | 3 (2) | 0.57 | 1.8 | 0.3182 | 1 | 0.25 | 2 | 1 | 0.914 / 0.906 |
| dark_shadow_w1.5_m10_glove_j0.75_s26 | 3.21 | 0.29 | 0.155 / 0.257 | 0.260 | 2 (2) | 0.60 | 2.0 | 0.3098 | 1 | 0.25 | 2 | 1 | 1.005 / 0.993 |
| dark_shadow_w1.5_m20_glasses_j0.75_s27 | 5.47 | 0.22 | 0.158 / 0.258 | 0.294 | 6 (6) | 0.65 | 1.9 | 0.3115 | 1 | 0.25 | 2 | 1 | 1.070 / 1.067 |
| dark_shadow_w2.5_m0_mole_j0.75_s28 | 1.59 | 0.43 | 0.195 / 0.259 | 0.212 | 4 (4) | 0.69 | 2.5 | 0.3203 | 1 | 0.25 | 2 | 1 | 0.732 / 0.726 |
| dark_shadow_w2.5_m10_crossing_j0.75_s29 | 3.36 | 0.28 | 0.194 / 0.259 | 0.263 | 6 (6) | 0.74 | 2.5 | 0.3205 | 1 | 0.25 | 1 | 1 | 0.750 / 0.749 |
| dark_shadow_w2.5_m20_tipband_j0.75_s30 | 5.67 | 0.21 | 0.193 / 0.258 | 0.297 | 2 (2) | 0.70 | 2.7 | 0.3163 | 1 | 0.7 | 2 | 1 | 0.837 / 0.827 |
| dark_shadow_w4.0_m0_none_j1.50_s31 | 1.76 | 0.41 | 0.199 / 0.257 | 0.226 | 3 (3) | 0.64 | 3.8 | 0.3373 | 1 | 0.25 | 0 | 1 | 0.726 / 0.726 |
| dark_shadow_w4.0_m10_glove_j1.50_s32 | 3.60 | 0.27 | 0.156 / 0.258 | 0.266 | 2 (2) | 0.68 | 4.0 | 0.3521 | 1 | 0.25 | 2 | 1 | 0.910 / 0.909 |
| dark_shadow_w4.0_m20_glasses_j1.50_s33 | 5.96 | 0.21 | 0.159 / 0.260 | 0.300 | 3 (3) | 0.59 | 3.9 | 0.3602 | 1 | 0.25 | 2 | 1 | 0.851 / 0.846 |
| dark_shadow_w6.0_m0_mole_j1.50_s34 | 2.01 | 0.38 | 0.194 / 0.259 | 0.226 | 3 (3) | 0.72 | 5.6 | 0.3377 | 1 | 0.25 | 2 | 1 | 0.805 / 0.803 |
| dark_shadow_w6.0_m10_crossing_j1.50_s35 | 3.93 | 0.26 | 0.194 / 0.259 | 0.270 | 2 (2) | 0.86 | 5.8 | 0.3599 | 1 | 0.25 | 2 | 1 | 0.877 / 0.875 |
| dark_shadow_w6.0_m20_tipband_j1.50_s36 | 6.37 | 0.20 | 0.193 / 0.260 | 0.306 | 3 (3) | 0.82 | 6.0 | 0.3267 | 1 | 0.25 | 2 | 1 | 0.780 / 0.777 |
| light_skin_w1.5_m0_none_j0.00_s37 | 1.47 | 0.45 | 0.466 / 0.571 | 0.211 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 1.004 / 0.995 |
| light_skin_w1.5_m10_glove_j0.00_s38 | 3.21 | 0.29 | 0.155 / 0.567 | 0.262 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 1.030 / 1.022 |
| light_skin_w1.5_m20_glasses_j0.00_s39 | 5.47 | 0.22 | 0.158 / 0.569 | 0.295 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 1.087 / 1.079 |
| light_skin_w2.5_m0_mole_j0.00_s40 | 1.59 | 0.43 | 0.473 / 0.567 | 0.217 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 0.996 / 0.989 |
| light_skin_w2.5_m10_crossing_j0.00_s41 | 3.36 | 0.28 | 0.471 / 0.571 | 0.266 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 1.014 / 1.009 |
| light_skin_w2.5_m20_tipband_j0.00_s42 | 5.67 | 0.21 | 0.469 / 0.573 | 0.297 | 0 (0) | 0.00 | 0.0 | 0.3093 | 1 | 0.25 | 2 | 0 | 1.056 / 1.049 |
| light_skin_w4.0_m0_none_j0.75_s43 | 1.76 | 0.41 | 0.465 / 0.569 | 0.221 | 0 (0) | 0.00 | 0.0 | 0.3102 | 1 | 0.25 | 2 | 1 | 0.988 / 0.985 |
| light_skin_w4.0_m10_glove_j0.75_s44 | 3.60 | 0.27 | 0.155 / 0.563 | 0.271 | 0 (0) | 0.00 | 0.0 | 0.3178 | 1 | 0.25 | 0 | 1 | 1.006 / 1.006 |
| light_skin_w4.0_m20_glasses_j0.75_s45 | 5.96 | 0.21 | 0.159 / 0.567 | 0.301 | 0 (0) | 0.00 | 0.0 | 0.3107 | 1 | 0.25 | 2 | 1 | 1.035 / 1.030 |
| light_skin_w6.0_m0_mole_j0.75_s46 | 2.01 | 0.38 | 0.468 / 0.571 | 0.231 | 0 (0) | 0.00 | 0.0 | 0.3099 | 1 | 0.25 | 2 | 1 | 0.992 / 0.989 |
| light_skin_w6.0_m10_crossing_j0.75_s47 | 3.93 | 0.26 | 0.467 / 0.570 | 0.276 | 0 (0) | 0.00 | 0.0 | 0.3183 | 1 | 0.25 | 2 | 1 | 1.000 / 0.998 |
| light_skin_w6.0_m20_tipband_j0.75_s48 | 6.37 | 0.20 | 0.470 / 0.572 | 0.305 | 0 (0) | 0.00 | 0.0 | 0.3106 | 1 | 0.25 | 2 | 1 | 1.014 / 1.012 |
| dark_skin_w3.0_m10_glove_j0.75_s49 | 3.44 | 0.28 | 0.156 / 0.562 | 0.270 | 3 (3) | 0.99 | 2.5 | 0.3190 | 1 | 0.25 | 2 | 1 | 0.196 / 0.193 |
| dark_texture_w3.0_m10_glove_j0.75_s50 | 3.44 | 0.28 | 0.155 / 0.587 | 0.265 | 3 (3) | 0.98 | 2.5 | 0.3252 | 1 | 0.25 | 2 | 1 | 0.265 / 0.260 |
| dark_shadow_w3.0_m10_glove_j0.75_s51 | 3.44 | 0.28 | 0.155 / 0.257 | 0.265 | 3 (1) | 0.12 | 2.9 | 0.3184 | 1 | 0.25 | 2 | 1 | 0.904 / 0.902 |
| light_skin_w3.0_m10_glove_j0.75_s52 | 3.44 | 0.28 | 0.155 / 0.565 | 0.269 | 0 (0) | 0.00 | 0.0 | 0.3076 | 1 | 0.25 | 2 | 1 | 1.019 / 1.015 |
| dark_skin_w3.0_m10_glasses_j0.75_s53 | 3.44 | 0.28 | 0.158 / 0.570 | 0.267 | 3 (3) | 0.98 | 2.5 | 0.3133 | 1 | 0.25 | 2 | 1 | 0.118 / 0.113 |
| dark_texture_w3.0_m10_glasses_j0.75_s54 | 3.44 | 0.28 | 0.158 / 0.591 | 0.267 | 3 (3) | 0.97 | 2.5 | 0.3232 | 1 | 0.25 | 2 | 1 | 0.200 / 0.190 |
| dark_shadow_w3.0_m10_glasses_j0.75_s55 | 3.44 | 0.28 | 0.158 / 0.259 | 0.266 | 2 (2) | 0.76 | 3.1 | 0.3176 | 1 | 0.25 | 2 | 1 | 0.721 / 0.718 |
| light_skin_w3.0_m10_glasses_j0.75_s56 | 3.44 | 0.28 | 0.158 / 0.568 | 0.265 | 0 (0) | 0.00 | 0.0 | 0.3160 | 1 | 0.25 | 2 | 1 | 1.023 / 1.020 |
| dark_skin_w3.0_m10_mole_j0.75_s57 | 3.44 | 0.28 | 0.192 / 0.569 | 0.268 | 3 (3) | 1.04 | 2.5 | 0.3152 | 1 | 0.25 | 2 | 1 | 0.059 / 0.054 |
| dark_texture_w3.0_m10_mole_j0.75_s58 | 3.44 | 0.28 | 0.192 / 0.587 | 0.266 | 3 (3) | 1.00 | 2.6 | 0.3125 | 1 | 0.25 | 2 | 1 | 0.136 / 0.128 |
| dark_shadow_w3.0_m10_mole_j0.75_s59 | 3.44 | 0.28 | 0.197 / 0.260 | 0.265 | 4 (4) | 0.75 | 3.0 | 0.3242 | 1 | 0.25 | 2 | 1 | 0.749 / 0.744 |
| light_skin_w3.0_m10_mole_j0.75_s60 | 3.44 | 0.28 | 0.470 / 0.566 | 0.269 | 0 (0) | 0.00 | 0.0 | 0.3197 | 1 | 0.25 | 2 | 1 | 1.024 / 1.019 |
| dark_skin_w3.0_m10_crossing_j0.75_s61 | 3.44 | 0.28 | 0.193 / 0.571 | 0.268 | 5 (5) | 1.02 | 2.5 | 0.3130 | 1 | 0.25 | 1 | 1 | 0.054 / 0.053 |
| dark_texture_w3.0_m10_crossing_j0.75_s62 | 3.44 | 0.28 | 0.192 / 0.590 | 0.266 | 5 (5) | 1.00 | 2.5 | 0.3128 | 1 | 0.25 | 1 | 1 | 0.136 / 0.134 |
| dark_shadow_w3.0_m10_crossing_j0.75_s63 | 3.44 | 0.28 | 0.193 / 0.259 | 0.268 | 3 (3) | 0.70 | 2.8 | 0.3186 | 1 | 0.25 | 1 | 1 | 0.808 / 0.807 |
| light_skin_w3.0_m10_crossing_j0.75_s64 | 3.44 | 0.28 | 0.469 / 0.567 | 0.268 | 0 (0) | 0.00 | 0.0 | 0.3119 | 1 | 0.25 | 2 | 1 | 1.021 / 1.018 |
| dark_skin_w3.0_m10_tipband_j0.75_s65 | 3.44 | 0.28 | 0.192 / 0.567 | 0.271 | 3 (3) | 1.00 | 2.5 | 0.3157 | 1 | 0.25 | 2 | 1 | 0.116 / 0.111 |
| dark_texture_w3.0_m10_tipband_j0.75_s66 | 3.44 | 0.28 | 0.194 / 0.591 | 0.267 | 3 (3) | 1.02 | 2.6 | 0.3063 | 1 | 0.25 | 2 | 1 | 0.194 / 0.186 |
| dark_shadow_w3.0_m10_tipband_j0.75_s67 | 3.44 | 0.28 | 0.195 / 0.259 | 0.263 | 3 (3) | 0.72 | 2.9 | 0.3227 | 1 | 0.25 | 2 | 1 | 0.676 / 0.671 |
| light_skin_w3.0_m10_tipband_j0.75_s68 | 3.44 | 0.28 | 0.472 / 0.572 | 0.270 | 0 (0) | 0.00 | 0.0 | 0.3108 | 1 | 0.25 | 2 | 1 | 1.012 / 1.010 |
| dark_skin_w2.5_m10_none_j0.75_detached_s100 | 3.20 | 0.29 | 0.195 / 0.569 | 0.160 | 3 (3) | 1.01 | 2.2 | 0.3165 | 1 | 0.25 | 2 | 1 | 0.073 / 0.063 |
| dark_skin_w4.0_m10_mole_j0.75_detached_s101 | 3.43 | 0.28 | 0.197 / 0.569 | 0.170 | 3 (3) | 0.96 | 3.4 | 0.3222 | 1 | 0.25 | 2 | 1 | 0.039 / 0.034 |
| dark_skin_w3.0_m10_tipband_j0.75_detached_s102 | 3.28 | 0.28 | 0.196 / 0.568 | 0.164 | 3 (3) | 0.97 | 2.5 | 0.3124 | 1 | 0.25 | 2 | 1 | 0.159 / 0.151 |
| dark_texture_w2.5_m10_none_j0.75_detached_s103 | 3.20 | 0.29 | 0.196 / 0.594 | 0.163 | 3 (3) | 0.99 | 2.3 | 0.3158 | 1 | 0.25 | 2 | 1 | 0.170 / 0.153 |
| dark_texture_w4.0_m10_mole_j0.75_detached_s104 | 3.43 | 0.28 | 0.195 / 0.590 | 0.171 | 3 (3) | 0.95 | 3.4 | 0.3117 | 1 | 0.25 | 2 | 1 | 0.109 / 0.099 |
| dark_texture_w3.0_m10_tipband_j0.75_detached_s105 | 3.28 | 0.28 | 0.197 / 0.593 | 0.165 | 3 (2) | 0.97 | 2.5 | 0.3085 | 1 | 0.25 | 2 | 1 | 0.235 / 0.190 |
| dark_shadow_w2.5_m10_none_j0.75_detached_s106 | 3.20 | 0.29 | 0.195 / 0.260 | 0.163 | 4 (4) | 0.70 | 2.4 | 0.3119 | 1 | 0.25 | 2 | 1 | 0.842 / 0.831 |
| dark_shadow_w4.0_m10_mole_j0.75_detached_s107 | 3.43 | 0.28 | 0.196 / 0.260 | 0.173 | 4 (4) | 0.62 | 4.0 | 0.3141 | 1 | 0.25 | 2 | 1 | 0.751 / 0.740 |
| dark_shadow_w3.0_m10_tipband_j0.75_detached_s108 | 3.28 | 0.28 | 0.197 / 0.261 | 0.163 | 4 (2) | 0.61 | 2.9 | 0.3192 | 1 | 0.25 | 2 | 1 | 0.880 / 0.873 |
| light_skin_w2.5_m10_none_j0.75_detached_s109 | 3.20 | 0.29 | 0.474 / 0.569 | 0.164 | 0 (0) | 0.00 | 0.0 | 0.3148 | 1 | 0.25 | 2 | 1 | 1.037 / 1.032 |
| light_skin_w4.0_m10_mole_j0.75_detached_s110 | 3.43 | 0.28 | 0.475 / 0.569 | 0.172 | 0 (0) | 0.00 | 0.0 | 0.3191 | 1 | 0.25 | 2 | 1 | 1.017 / 1.016 |
| light_skin_w3.0_m10_tipband_j0.75_detached_s111 | 3.28 | 0.28 | 0.477 / 0.573 | 0.166 | 0 (0) | 0.00 | 0.0 | 0.3146 | 1 | 0.25 | 2 | 1 | 1.026 / 1.022 |

## 8. Speed during the sweep (ms per frame through the host, 424x562, 2 cores)

- default: 202.621 ms
- vote_off: 107.715 ms
- tm_off: 198.861 ms
- vote_off_tm_off: 21.014 ms
- rebuild_off: 100.831 ms
