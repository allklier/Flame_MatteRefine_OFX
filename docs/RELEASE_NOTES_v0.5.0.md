# Matte Refine ML 0.2.0, tag v0.5.0

This release adds one feature to **Matte Refine ML**. The source also contains a fourth plugin, **Edge Recipes** 0.1.0,
tagged `edge-recipes-v0.1.0` as **experimental source only: there is no package for it and nobody should use it yet**. Matte Refine (0.4.0) and Matte Merge (0.1.0) are unchanged in behaviour and are not
re-released here; their packages from v0.4.0 stay current.

| Plugin | Version | Status |
|---|---|---|
| **Matte Refine ML** | 0.2.0 | Release. New: Jump Repair. Rocky Linux 9, NVIDIA GPU for the model operations (Jump Repair needs none). |
| **Edge Recipes** | 0.1.0 | **Experimental, source only, no package.** First recipe: Hair Locks. |

## Matte Refine ML 0.2.0: Jump Repair
A matting model sometimes makes a patch of matte that appears (or a hole that opens) for one to three frames while the
frames around it agree. **Jump Repair** finds those patches by comparing each frame with its neighbours' mattes and puts
the neighbours' value back, only there. No model, no picture, deterministic; everything outside a repaired patch is
unchanged bit for bit. It runs first in the chain (Jump Repair, Temporal, Smart Fill, Edge Refine).
- Controls: Enable (off by default), Jump Above, Min Size %, Span (1-3 frames) and Motion Guards. A new **Jumps** view
  shows what was repaired (magenta, green) and what was kept because it looks like motion (red).
- Motion Guards keep a patch that moves like a real object instead of repairing it; switch them off if you know the red
  patches are hallucinations.
- A repaired patch's soft rim is removed with it, so no faint outline stays behind.
- Limits: it cannot tell a one-frame real object (a thrown object, a flash) from a hallucination; the first and last
  frame of a clip cannot be judged. Check the Jumps view and keyframe Enable off for such frames.
- Also in this version: Temporal can take a weight map (used by Edge Recipes; no change when unused); the runner has its own
  build id per plugin so Matte Refine ML and Edge Recipes never replace each other's runner process.
- Tried on one real 97-frame matte (two clear one-frame blobs repaired, twelve motion-like blobs kept, the rest unchanged)
  and on synthetic sequences. Not yet tried by other users.

User guide: `USERGUIDE.md`, section 3 (Jump Repair).

## Edge Recipes 0.1.0 (experimental, source only): Hair Locks
Edge Recipes runs several of the existing methods in a fixed order for one common problem. Nodes appear under **Edge
Recipes** in Batch. **Hair Locks** rebuilds thin hair locks that a good, tight matte has lost, inside a soft roto you draw
(the **Focus Region**, the fourth socket): a model candidate (ViTMatte) and a traced strand (Rebuild Detail, which creates
matte from the picture) are weighed together, Snap to Picture runs on the new pixels, and Temporal steadies the matte
outside the region.

**Read this before using it:**
- It has been tried on one difficult real shot and on synthetic scenes. On that shot the locks are recovered and connected
  to the matte, but the result still shows temporal instability and halos; **it was not good enough for that production
  shot.** Treat it as a preview of a technique, not a finished tool.
- Two switches are temporary test controls and will go: **Snap Mode** (Limited / Off, Hair Locks 2 column) and **Strand
  Blend** (Advanced, Rebuild Detail column).
- Nothing in Edge Recipes changes the matte outside the Focus Region except Temporal.
- It has its own bundle and model process; it can be built from this source (`MR_BUILD_RECIPES=ON`, see `docs/edge_recipes.md`). The Analyze
  button is not built yet.

User guide: `USERGUIDE.md`, section 7.

## Install
Download the Matte Refine ML package (see `README.md`), check it against `SHA256SUMS`, unpack, and with Flame closed run
`sudo ./install.sh` (it installs and removes only the plugin(s) in its own package). Matte Refine ML and Edge Recipes' model steps
need one more step as your normal user, `./setup.sh` (shows the licences, asks you to accept them, downloads the models and
NVIDIA libraries, about 1.9 GB; once covers both).

## Known limits
- Linux only for both plugins; NVIDIA GPU with a driver that supports CUDA 13 libraries for the model steps.
- The matting model cannot add what the picture does not show; thin dark hair on dark skin is the hardest case.
- The test scenes in the guide are synthetic; results on real footage vary, hair in particular.

## Support
The **Collect Logs** button writes `~/matte-refine-support-<date>.tar.gz` (logs and system details, no images). Please attach it
to an issue together with the Flame version and a description of the shot.

Licence: Mozilla Public License 2.0. Models and libraries downloaded by `setup.sh` keep their own licences, shown there.
