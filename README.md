# Matte Refine OFX for Flame

OFX plugins for Autodesk Flame (Rocky Linux 9; Matte Refine also on macOS) that refine an existing
matte: they keep a good matte and fix its defects. Two nodes, both under *Matte Refine* in Batch:

- **Matte Refine**: eight classical operations, real time. Each has its own column with an
  Enable switch; they run in a fixed order. No processing step, no cache.
- **Matte Refine ML**: operations driven by machine-learning models (NVIDIA GPU, Rocky Linux
  only), also rendered live, frame by frame.

**How to use every operation, with before / after pictures: [USERGUIDE.md](USERGUIDE.md).**

## Operations

Matte Refine, in chain order:

- **Clean & Crush**: removes small islands outside the subject and crushes background haze to 0.
- **Fill Holes** (with **Fill Holes Edges**): fills holes enclosed by solid matte; real gaps
  larger than Max Hole % stay open. On by default.
- **Solid Core**: makes everything deeper than Depth px inside the matte solid.
- **Contour Smooth**: takes wiggle and chatter off the edge line, keeping its softness.
- **Shrink / Grow**: moves the edge in or out, keeping each part's softness.
- **Edge Width**: gives the whole edge one consistent softness.
- **Snap to Image**: pulls the edge onto the picture's edges (needs the picture).
- **Edge Colour Unmix**: removes the old background's colour from the soft edges of the picture
  (needs a scene-linear picture; changes the RGB, not the matte).
- **Analyze Frame** (a button): sets Clean & Crush, Fill Holes and Solid Core from the current
  frame and says why.

Matte Refine ML, in chain order:

- **Smart Fill**: fills holes that are part of the subject and keeps real gaps, decided per hole
  by a segmentation model and the picture's colour.
- **Edge Refine**: re-solves a band around the matte's edge with a matting model (ViTMatte);
  outside the band the matte is unchanged.
- **Temporal**: coming soon.

## Inputs and outputs

- **Matte Refine**: the matte goes to the node's **Matte** input, or, if that is empty, it is
  read from the front input's alpha (Flame's source alpha socket). The picture on the front is
  optional; only Snap to Image and Edge Colour Unmix need it.
- **Matte Refine ML**: the picture on the front is **required**; the matte goes to the second
  socket (source alpha), as with LTX Alpha Gen.
- The node's **matte output** is always the refined matte. Its **image output** follows
  **View**: Matte Refine has Picture (default), Show Changes, Matte, Trimap and Grey Inside;
  Matte Refine ML has Picture (default), Matte, Trimap, Changes and Holes (USERGUIDE.md).

## Install

From a release package, with Flame closed:

```
sudo ./install.sh              # installs /usr/OFX/Plugins/MatteRefine.ofx.bundle
sudo ./install.sh --uninstall  # removes it
```

## Support

The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system
details, no images). Turn on **Debug Logging** first to include per-frame timings and
settings (Matte Refine ML shares the same log). **Save Matte** (Matte Refine only) writes the current frame's input matte to
`~/.cache/matte-refine/dumps/` for analysis; it stays on your machine unless you send it. The log lives
in `~/.cache/matte-refine/logs/`.

## Licence

Mozilla Public License 2.0 (`LICENSE`). The vendored OpenFX headers and support library
(`third_party/openfx`) are BSD licensed.
