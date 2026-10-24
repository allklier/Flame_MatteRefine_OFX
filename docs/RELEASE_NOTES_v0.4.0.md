# Matte Refine family, early release (v0.4.0)

Three OFX plugins for Autodesk Flame that refine an existing matte. They keep a good matte and fix its defects; none of
them pulls a new matte from the picture. All three are under *Matte Refine* in Batch, with the same inputs: **the picture
on the front (required), the matte on the second socket (source alpha)**. This is an early release: it has been tested in
Flame 2027.1 / 2027.2 on Rocky Linux 9 by the author; the Mac builds have been tested outside Flame only.

| Plugin | Version | What it does | Platforms |
|---|---|---|---|
| **Matte Refine** | 0.4.0 | Eight classical operations in real time: Clean & Crush, Fill Holes, Solid Core, Contour Smooth, Shrink / Grow, Edge Width, Snap to Image, Edge Colour Unmix, plus Analyze Frame | Rocky Linux 9, macOS 15+ (Apple Silicon) |
| **Matte Refine ML** | 0.1.0 | Smart Fill (segmentation-guided hole fill), Edge Refine (ViTMatte) and Temporal, all on an NVIDIA GPU | Rocky Linux 9 |
| **Matte Merge** | 0.1.0 | Merges the better edge detail of a second matte (hair, fur; for example from LTX Alpha Gen) into a base matte, aligned to the picture | Rocky Linux 9, macOS 15+ (Apple Silicon) |

## What changed since 0.3.0 (Matte Refine)
- **The Matte input is gone.** Matte Refine now takes the picture on the front and the matte on its alpha, like the other two
  plugins. Reason: with the extra input connected, Flame converted the picture and tagged it as a matte. Saved settings load;
  nodes saved in older batches need their matte moved to the second socket.

## Install
Download the package for your platform (see `README.md`), check it against `SHA256SUMS`, unpack, and with Flame closed run
`sudo ./install.sh`. **Matte Refine ML needs one more step as your normal user, `./setup.sh`**: it shows the licences of the
models and of NVIDIA's CUDA libraries, asks you to accept them and downloads the models and libraries (about 1.9 GB;
3.2 GB with the full-size models) to `~/.local/share/matte-refine-ml/`. Until then the node passes the matte through.

## Known limits
- Matte Refine ML needs an NVIDIA GPU with a driver that supports CUDA 13 libraries; macOS is not supported for it.
- The macOS packages are not notarised; `install.sh` removes the download quarantine flag. They have not been tried in
  Flame on a Mac yet.
- Matte Merge is new. It can add only what the detail matte contains; Strength above 1 brightens weak strands, including
  invented ones. The user guide (section 6) explains each control with pictures.
- The test scenes in the guide are synthetic; results on real footage vary, hair in particular.

## Support
The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system details, no images). Please attach
it to an issue together with the Flame version and a description of the shot. The user guide is `USERGUIDE.md`.

Licence: Mozilla Public License 2.0. Models and libraries downloaded by `setup.sh` keep their own licences, shown there.
