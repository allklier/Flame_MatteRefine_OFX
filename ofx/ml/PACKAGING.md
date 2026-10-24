# ofx/ml/PACKAGING.md — packaging of Matte Refine and Matte Refine ML (decision 38, built 2026-10-24)

Design reasons are in the root `CLAUDE.md` (decision 38). This file is how it works. Tested on BFD with a local mirror and
the scratch host, then built on akflame5 (Rocky 9); the real Flame trial install is the next step.

## Pieces
| File | Role |
|---|---|
| `scripts/package_release_mac.sh` | run from the Linux dev box (`HOST=lbm REMOTE=... CMAKE=... CMAKE_ARGS=... DIST=...`): syncs, builds clean in `REMOTE/release_build`, ctest on the Mac, checks arm64 / minos 15.0 / system libraries only / no build paths, stages on the Mac's local disk (the SMB share ignores chmod) and writes `MatteRefine-<ver>-macos-arm64.tar.gz` and `MatteMerge-<ver>-macos-arm64.tar.gz`; merges its lines into `DIST/SHA256SUMS`. **Run the Linux script first, then the Mac one.** Ad-hoc linker signature only (no notarisation; `codesign -v` on the bundle reports missing resources, `dlopen` is fine; Flame on the Mac untested) |
| `scripts/package_release.sh <build dir>` | on the Rocky 9 host: configure with `MRML_BUNDLE_ORT=ON`, build, ctest, check the runner's RPATH / RUNPATH is `$ORIGIN/lib` only (Rocky's linker writes DT_RPATH), stage, write the tarballs and `SHA256SUMS` into `dist/`. Env: `ORT_DIR` (extracted ORT 1.31.0 CUDA 13 package, required), `MR_GIT`, `DIST` |
| `packaging/install.sh` | root; copies the bundle(s) found next to it; stops a running runner first; refuses while Flame runs; `--uninstall`; `--plugins-dir` (tests; also set `MRML_INSTALL_NO_ROOT_CHECK=1` to skip the root check); prints the `setup.sh` instruction |
| `packaging/setup.sh` | per user, refuses root; bash + curl + sha256sum + unzip. Licence screen (`licences.txt`), accept, full-models question, downloads with resume and sha256, writes config. Options `--accept-licences`, `--full` / `--no-full`, `--prefix DIR`, `--from DIR` (offline), `--remove [-y]`. Test-only: `MRML_SETUP_BASE_HF` / `MRML_SETUP_BASE_PYPI` point at a mirror |
| `packaging/licences.txt` | the licence screen; its sha256 is the accepted-licence key (a changed text re-asks) |
| `ofx/ml/src/gate.{h,cpp}` | SHA-256, `checkGate()`, `defaultModelDir()`, the CUDA library list (11), `modelInstalled()`, `optionLabel()`, `notInstalledFooter()` |
| `ofx/ml/tests/test_gate.cpp`, `check_bundle_deps.sh` | gate tests (mutation-checked); the `.ofx` must import no `dlopen` / `dlsym` and contain no ORT name |

## Tarballs (no models, no CUDA libraries)
Three Linux tarballs and two macOS ones (Matte Refine and Matte Merge). `MatteMerge-<MM_VERSION>-linux-x86_64.tar.gz`: bundle, `install.sh`, README, LICENSE, USERGUIDE and its images (no models).
`MatteRefine-<MR_VERSION>-linux-x86_64.tar.gz` (5 MB): bundle, `install.sh`, README, LICENSE, USERGUIDE, referenced images.
`MatteRefineML-<MRML_VERSION>-linux-x86_64.tar.gz` (249 MB): `MatteRefineML.ofx.bundle` (the `.ofx`; `Resources/mr_ml_runner`,
`Resources/licences.txt`, `Resources/lib/` = ORT + CUDA provider, 317 MB), `install.sh`, `setup.sh`, `licences.txt`,
`THIRD_PARTY/` (ORT licence and notices), README, LICENSE, USERGUIDE, images. No TensorRT provider.

## Data written by setup.sh (per user)
`~/.local/share/matte-refine-ml/` (or `--prefix`): `models/<id>/model.onnx` (vitmatte-small, birefnet-lite; with `--full`
also vitmatte-base, birefnet), `cuda_libs/` (the 11 libraries from NVIDIA's PyPI wheels, 1.53 GB, plus `licences/`).
Default 1.9 GB on disk, about 1.4 GB downloaded; `--full` adds 1.36 GB. Sources and checksums: Hugging Face and PyPI, the
pins are in `ofx/ml/package/fetch.sh`'s values. Config keys written to `~/.config/matte-refine/config.conf` (other lines
kept; `--remove` deletes all four): `ml_model_dir`, `ml_cuda_libs`, `ml_licence_accepted` (sha256 of the shipped
`licences.txt`), `ml_full_declined=1` (only after a "no": the question is not asked again; the summary of every run says
how to add them: `setup.sh --full`). The licence key is written right after acceptance so a failed download does not re-ask.
ONNX Runtime keeps its own cache in `~/.cache/Microsoft/DeveloperTools/.onnxruntime`.

## The gate (in the node, `plugin_ml.cpp` render)
A release bundle is one with `Contents/Resources/licences.txt`; dev builds are not gated. While the licence hash does not
match, a standard model (vitmatte-small, birefnet-lite) or a CUDA library is missing, EVERY step is off, Temporal too (user,
2026-10-24: avoids confusion), the matte passes through and the footer says what to do; no runner starts. A selected
optional model that is missing turns off only its own step ("ViTMatte Base is not installed: run setup.sh --full."). The
Model and Segmentation dropdown labels are set at describe from the model folder ("Base (not installed)", "Full (not
installed)"); indexes and param ids are unchanged. One INFO log line when the gate answer changes. Default model folder
changed to `~/.local/share/matte-refine-ml/models`.

## Choices and facts
- CUDA trimming saves only 127 MB (cublasLt 546 MB, cudnn engines 489 MB are all needed): not worth it; one list of 11
  libraries serves the runner, the gate and setup.sh's checksums.
- `MRML_BUNDLE_ORT` (CMake, default OFF): ORT goes into `Contents/Resources/lib`, `BUILD_WITH_INSTALL_RPATH`; without it CMake
  adds the build path of ORT to the runner's path.
- Not yet verified: real Hugging Face / PyPI downloads from Rocky 9 (trial running 2026-10-24), the install under `sudo`
  (the `sudo -H -u $SUDO_USER` runner stop), the macOS path of `install.sh`, and the dropdown labels in Flame.
- `ofx/ml/package/build_rocky9.sh` builds only the benchmark tool; it is not the release build (`package_release.sh` is).
