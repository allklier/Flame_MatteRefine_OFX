# Matte Refine OFX for Flame

OFX plugins for Autodesk Flame (Rocky Linux 9; Matte Refine and Matte Merge also on macOS, Apple Silicon) that refine an existing
matte: they keep a good matte and fix its defects. Three nodes, all under *Matte Refine* in Batch:

- **Matte Refine**: eight classical operations, real time. Each has its own column with an
  Enable switch; they run in a fixed order. No processing step, no cache.
- **Matte Refine ML**: operations driven by machine-learning models (NVIDIA GPU, Rocky Linux
  only), also rendered live, frame by frame.
- **Matte Merge**: takes a matte with poor edge detail and a second matte with better detail (hair, fur) and
  merges the detail into the first, aligned to the picture. CPU only, real time (Rocky Linux and macOS).

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
- **Temporal**: steadies the matte's edge and holes from frame to frame, using the neighbouring frames.

## Inputs and outputs

- **Matte Refine** and **Matte Refine ML**: the picture on the front is **required**; the matte
  goes to the node's second socket (source alpha), as with LTX Alpha Gen.
- **Matte Merge**: the picture on the front, the base matte (or a trimap) on the second socket (source alpha), and
  the detail matte on the third (Detail).
- The node's **matte output** is always the refined matte. Its **image output** follows
  **View**: Matte Refine has Picture (default), Show Changes, Matte, Trimap and Grey Inside;
  Matte Refine ML has Picture (default), Matte, Trimap, Changes and Holes (USERGUIDE.md).

## Install

Five packages; each installs on its own:

| Package | Platform | Contents |
|---|---|---|
| `MatteRefine-<version>-linux-x86_64.tar.gz` | Rocky Linux 9 | Matte Refine (classical clean-up and edge operations) |
| `MatteRefineML-<version>-linux-x86_64.tar.gz` | Rocky Linux 9, NVIDIA GPU | Matte Refine ML (machine-learning operations) |
| `MatteMerge-<version>-linux-x86_64.tar.gz` | Rocky Linux 9 | Matte Merge (base matte + detail matte -> one matte) |
| `MatteRefine-<version>-macos-arm64.tar.gz` | macOS 15 or later, Apple Silicon | Matte Refine |
| `MatteMerge-<version>-macos-arm64.tar.gz` | macOS 15 or later, Apple Silicon | Matte Merge |

Matte Refine ML is Linux only. `SHA256SUMS` lists the checksums (`sha256sum -c SHA256SUMS`; on macOS
`shasum -a 256 -c SHA256SUMS`). Unpack a package, then, with Flame closed:

```
sudo ./install.sh              # copies the bundle to /usr/OFX/Plugins (macOS: /Library/OFX/Plugins)
sudo ./install.sh --uninstall  # removes all three plugins
```

All nodes appear under **Matte Refine** in Flame. Installing a newer package replaces the older version.

**macOS:** the bundles are not notarised by Apple. A downloaded tarball is marked as quarantined; `install.sh` removes
that flag from the installed bundle, so Gatekeeper does not block it. Install with `install.sh` rather than copying the
bundle by hand. The Mac builds have been tested on Apple Silicon outside Flame; they have not been tried in Flame yet.

**Matte Refine ML needs a second step, as your normal user (not root):**

```
./setup.sh
```

The models and the NVIDIA libraries are not part of the plugin: `setup.sh` shows their licences (ViTMatte and
BiRefNet are MIT; ONNX Runtime is MIT; NVIDIA CUDA and cuDNN are under NVIDIA's own terms), asks you to accept them, asks
whether to also install the full-size models (default: no), and downloads the files to `~/.local/share/matte-refine-ml/`
(about 1.9 GB; 3.2 GB with the full models), checking each against a fixed checksum. It writes your choices to
`~/.config/matte-refine/config.conf`. Until the licences are accepted the node passes the matte through and says so in
its footer. Run `./setup.sh --full` later to add the full models and `./setup.sh --remove` to take everything out. Then
restart Flame. Options: `--accept-licences`, `--full` / `--no-full`, `--prefix DIR`, `--from DIR` (offline install from a
folder of the downloaded files).

## Support

The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system
details, no images). Turn on **Debug Logging** first to include per-frame timings and
settings (Matte Refine ML shares the same log). **Save Matte** (Matte Refine only) writes the current frame's input matte to
`~/.cache/matte-refine/dumps/` for analysis; it stays on your machine unless you send it. The log lives
in `~/.cache/matte-refine/logs/`.

## Licence

Mozilla Public License 2.0 (`LICENSE`). The vendored OpenFX headers and support library
(`third_party/openfx`) are BSD licensed.
