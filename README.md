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
| Solid Above | 0.95 | Matte at or above this value counts as solid |
| Max Hole % | 1.0 | Holes larger than this share of the frame stay open, so real gaps (an arm on a hip, between the legs) survive. 0 = fill every enclosed hole |
| Bridge Gaps px | 0 | Also fill holes whose solid edge has a gap up to this wide |
| Rim px | 1 | Also lift the soft rim around a filled hole to white |
| Left / Right / Top / Bottom Closed | off | Treat that frame edge as solid, for a subject cut by the frame |

Filled holes become fully white (1.0). Pixels are never made darker. For keyed mattes whose
"solid" areas are slightly grey, raise Solid Above (0.95–0.98 worked on a product shot).

### Solid Core

Makes everything deeper than **Depth px** inside the matte solid white, for areas a key left
semi-transparent that are joined to the edge (so they aren't holes). The outer Depth pixels of
the edge are not touched, and real gaps keep the same margin. Off by default: it also makes
deliberately semi-transparent areas deeper than Depth solid.

| Control | Default | What it does |
|---|---|---|
| Enable | off | Solid Core on or off |
| Depth px | 6 | The core starts this far inside the edge |
| Edge At | 0.5 | The matte value the edge is measured from |

Runs after Fill Holes.

### Edge Refine (planned)

Edge refinement with a trimap-based matting model. Not available yet.

## Inputs and outputs

- **Matte From**: where the matte is read from: the source alpha (Flame's matte input), the
  luminance of the RGB, or the red channel.
- The node's **matte output** is always the refined matte. Its **image output** follows
  **View**: **Matte** (default) shows the refined matte too; **Show Filled** shows the input
  matte in grey with everything that was filled in magenta; **Source** passes the source image
  through, for a front + matte chain.

## Install

From a release package, with Flame closed:

```
sudo ./install.sh              # installs /usr/OFX/Plugins/MatteRefine.ofx.bundle
sudo ./install.sh --uninstall  # removes it
```

## Support

The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system
details, no images). Turn on **Debug Logging** first to include per-frame timings and
settings. **Save Matte** writes the current frame's input matte to
`~/.cache/matte-refine/dumps/` for analysis; it stays on your machine unless you send it. The log lives
in `~/.cache/matte-refine/logs/`.

## Licence

Mozilla Public License 2.0 (`LICENSE`). The vendored OpenFX headers and support library
(`third_party/openfx`) are BSD licensed.
