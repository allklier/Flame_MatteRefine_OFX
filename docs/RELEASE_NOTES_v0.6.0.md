# Plate Tools 0.1.0, tag v0.6.0

This release adds a new bundle, **Plate Tools**, whose first effect is **Plate Merge**. Matte Refine (0.4.0), Matte Merge
(0.1.0) and Matte Refine ML (0.2.0) are unchanged and are not re-released here; their packages from v0.4.0 and v0.5.0 stay
current. Edge Recipes is still experimental source only.

| Plugin | Version | Status |
|---|---|---|
| **Plate Tools** (Plate Merge) | 0.1.0 | Release. Rocky Linux 9 and macOS 15+ (Apple Silicon). CPU only. |

## Plate Merge
Two plates of the same camera, aligned to within about a pixel, from different takes; a person moves in each. With one tight
matte per person (for example from a matting model) Plate Merge builds one plate:
- **B Over A**: plate B's person on top, plate A's person removed wherever plate B has clean background, so A's person never
  peeks out from under B's.
- **Clean Background**: both persons removed wherever the other take is clean.
- **Level Match** (scene-linear plates): compares the two plates on a ring outside both mattes and matches B's level to A's.
- Grow and Feather widen the mattes (soft shadows and defocus halos reach past a tight matte); an optional **Keep A** matte
  protects what plate A must keep (a reflection). Views show the weights, the ring, the swap matte and the residual.
- Plate A is returned bit for bit where neither matte is on; float, no clamping.

Limits: the plates must be aligned (stabilised or locked off); where both persons overlap no clean pixel exists and the
upper person wins; Level Match assumes scene-linear plates. User guide: `USERGUIDE.md`, section 8; design: `docs/plate_merge.md`.

## Tested
Synthetic scenes (unit and host system tests) and one real Flame job on Linux, where the result was correct. **Not yet tried
on macOS** (built and checked for arm64, minos 15.0, system libraries only; not signed or notarised, the installer removes the
quarantine flag). Not yet tried by other users.

## Packages
`PlateTools-0.1.0-linux-x86_64.tar.gz`, `PlateTools-0.1.0-macos-arm64.tar.gz`, `SHA256SUMS`. Install as the other packages
(README, "Install"): quit Flame, unpack, `sudo ./install.sh`, restart Flame; the node is **Plate Merge** under *Matte Refine*.
