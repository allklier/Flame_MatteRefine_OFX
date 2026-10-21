# Matte Refine OFX for Flame

An OFX plugin for Autodesk Flame (Rocky Linux 9) that cleans up mattes. One node, **Matte
Refine** (under *Matte Refine* in Batch), holds a list of matte operations; each has its own
section with an Enable switch. Everything renders live, frame by frame: no processing step, no
cache.

## Operations

### Fill Holes

Fills holes inside a matte, the speckles and gaps a keyer or a model leaves inside a
subject, without touching the matte's outer edge (unlike a dilate or close). A hole is any area
below **Solid Above** that is fully surrounded by solid matte.

| Control | Default | What it does |
|---|---|---|
| Enable | on | Fill Holes on or off |
| Solid Above | 0.5 | Matte at or above this value counts as solid |
| Max Hole % | 1.0 | Holes larger than this share of the frame stay open, so real gaps (an arm on a hip, between the legs) survive. 0 = fill every enclosed hole |
| Bridge Gaps px | 0 | Also fill holes whose solid edge has a gap up to this wide |
| Rim px | 1 | Also lift the soft rim around a filled hole to white |
| Left / Right / Top / Bottom Closed | off | Treat that frame edge as solid, for a subject cut by the frame |

Filled holes become fully white (1.0). Pixels are never made darker.

### Edge Refine (planned)

Edge refinement with a trimap-based matting model. Not available yet.

## Inputs and outputs

- **Matte From**: where the matte is read from: the source alpha (Flame's matte input), the
  luminance of the RGB, or the red channel.
- The node's **matte output** is the refined matte. Its **image output** is the source
  unchanged, or, with **View = Show Filled**, the input matte in grey with everything that was
  filled shown in magenta.

## Install

From a release package, with Flame closed:

```
sudo ./install.sh              # installs /usr/OFX/Plugins/MatteRefine.ofx.bundle
sudo ./install.sh --uninstall  # removes it
```

## Support

The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system
details, no images). Turn on **Debug Logging** first to include per-frame timings. The log lives
in `~/.cache/matte-refine/logs/`.

## Licence

Mozilla Public License 2.0 (`LICENSE`). The vendored OpenFX headers and support library
(`third_party/openfx`) are BSD licensed.
