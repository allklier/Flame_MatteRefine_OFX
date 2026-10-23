The pictures of USERGUIDE.md, made from the synthetic test scenes (ofx/tools/synth.h) with the
plugins' own operation code by ofx/tools/mr_guide_images.cpp. To make them again (from the repo root):

  # Matte Refine images only (no ML runtime needed; needs zlib):
  cmake -S . -B build && cmake --build build --target mr_guide_images
  build/mr_guide_images docs/images

  # All images, including Matte Refine ML (Edge Refine, Smart Fill, the Temporal preview):
  cmake -S . -B build_ml -DMR_BUILD_ML=ON -DORT_DIR=<onnxruntime-linux-x64-gpu dir>
  cmake --build build_ml --target mr_guide_images
  build_ml/mr_guide_images docs/images <model dir> <cuda lib dir>

<model dir> holds vitmatte-small/model.onnx and birefnet-lite/model.onnx (the real models, run
in-process; CUDA if available, else the CPU). A missing model skips its images (the tool prints
SKIPPED); nothing is faked. The tool prints each image's settings and measurements; the same
settings are given under each image in USERGUIDE.md. Scenes: 1920x1080, seed 1. Takes about 40 s.
