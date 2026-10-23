#!/bin/bash
# fetch.sh <dest dir>: downloads ONNX Runtime 1.31.0, (Linux) the CUDA 13 / cuDNN 9 libraries from
# PyPI wheels, and the models from Hugging Face, and checks every sha256. ~3.3 GB on Linux.
set -eu
D=$(mkdir -p "$1" && cd "$1" && pwd); cd "$D"
get() { [ -f "$2" ] || curl -fL --create-dirs -o "$2" "$1"; echo "$3  $2" | sha256sum -c - 2>/dev/null || echo "$3  $2" | shasum -a 256 -c -; }
GH=https://github.com/microsoft/onnxruntime/releases/download/v1.31.0
if [ "$(uname)" = Darwin ]; then
  get $GH/onnxruntime-osx-arm64-1.31.0.tgz ort.tgz 1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc
else
  get $GH/onnxruntime-linux-x64-gpu_cuda13-1.31.0.tgz ort.tgz 2867dec99c6a4926e8142361b9d4d1518c16c85b854b4d41d28d65d3b2c81135
  PY=https://files.pythonhosted.org/packages
  mkdir -p wheels cuda_libs
  get $PY/eb/1b/adc70acf3fde509b1f0ca29d1c0e8d9b00aaecd871f8cf877257c8b68d9c/nvidia_cublas-13.8.1.7-py3-none-manylinux_2_27_x86_64.whl wheels/cublas.whl c11a27fd4379510e5b1f84b367a2514d1e52fe5cc13442117a0e0a1addee3cf2
  get $PY/98/8a/3431271f6344874b8f1ac03f16b3d679c91493f8da63f716160403e6d0a0/nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl wheels/cudart.whl 9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5
  get $PY/07/73/3ee8e5b4cb891401e603ffd3a59b35c6afe785fd2de123afe7c7029603dc/nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl wheels/curand.whl 25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5
  get $PY/af/75/96ea5c5368eb595c39d629cde08a66227a864e71ccb0e593add2612bb952/nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl wheels/cudnn.whl 9677e76f21862eb5da7ee5ed69d544738b2d8b5c3ce7e5ec125c5592e6cdbdc8
  for w in wheels/*.whl; do mkdir -p x && unzip -qo "$w" 'nvidia/*' -d x; done
  find x -name '*.so*' -exec cp -a {} cuda_libs/ \; ; rm -rf x
fi
mkdir -p ort && tar -xzf ort.tgz -C ort
HF=https://huggingface.co
get $HF/Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model.onnx models/vitmatte-small/model.onnx bf28d2e0be2c073286e88d60ad649d7123da2749a2d99133fd1098d5887e0225
get $HF/Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model.onnx models/vitmatte-base/model.onnx f6978437f5068849bcbf49b1f4e37b90aaca5155744e9fde4e6c689f70c2b9ee
get $HF/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx models/birefnet-lite/model.onnx 5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333
get $HF/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model_fp16.onnx models/birefnet-lite/model_fp16.onnx d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402
get $HF/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model.onnx models/birefnet/model.onnx 58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c
get $HF/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model_fp16.onnx models/birefnet/model_fp16.onnx 3654c741eb80bd926ada8fed1713b506ccf8d30eb1f6487e87eb9f234f33df09
echo "ORT_DIR=$D/ort/$(ls ort)  MODELS=$D/models  CUDA_LIBS=$D/cuda_libs"
