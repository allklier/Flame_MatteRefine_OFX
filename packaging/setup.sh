#!/usr/bin/env bash
# Matte Refine ML for Flame: per-user setup. Run it as your normal user (not root, not sudo), after
# the admin ran install.sh. It shows the licences of the models and libraries (licences.txt), and
# when you accept them it downloads the models and the NVIDIA CUDA libraries into your home folder,
# checks every file's SHA-256 and writes three lines into ~/.config/matte-refine/config.conf
# (ml_model_dir, ml_cuda_libs, ml_licence_accepted; ml_full_declined=1 when you said no to the full
# models, so you are not asked again; other lines are kept). Then restart Flame.
#
#   ./setup.sh                      interactive: licences, then "full models?", then download
#   ./setup.sh --accept-licences    accept licences.txt without the prompt (scripts)
#   ./setup.sh --full | --no-full   also install / do not install the full models (no question);
#                                   --full later adds them to an existing setup
#   ./setup.sh --prefix DIR         data folder (default ~/.local/share/matte-refine-ml)
#   ./setup.sh --from DIR           offline: take the files from DIR (laid out as ofx/ml/package/fetch.sh
#                                   writes it: models/<id>/model.onnx, cuda_libs/*.so*, wheels/*.whl);
#                                   every file is verified, nothing is downloaded
#   ./setup.sh --remove [-y]        delete the data folder's models and libraries and the ml_* keys
#
# Running it again is safe: files already there are verified and kept, a broken download resumes,
# only what is missing is fetched, and the licences are asked again only if licences.txt changed.
# Needs: bash, curl, sha256sum, unzip, df (all on Rocky Linux 9). ONNX Runtime is inside the plugin;
# when it runs it keeps a small cache of its own in ~/.cache/Microsoft/DeveloperTools/.onnxruntime.
#
# Test-only (not for users): MRML_SETUP_BASE_HF and MRML_SETUP_BASE_PYPI replace the download
# hosts (https://huggingface.co, https://files.pythonhosted.org/packages) with a local mirror.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LICENCES="$HERE/licences.txt"
CONFIG="$HOME/.config/matte-refine/config.conf"
HF="${MRML_SETUP_BASE_HF:-https://huggingface.co}"
PYPI="${MRML_SETUP_BASE_PYPI:-https://files.pythonhosted.org/packages}"

# Models: id | path on Hugging Face | sha256 | bytes | core (always) or full (asked).
# The ids and files are ofx/ml/src/model_table.cpp's (fp32 only).
MODELS=(
  "vitmatte-small|Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model.onnx|bf28d2e0be2c073286e88d60ad649d7123da2749a2d99133fd1098d5887e0225|103885865|core"
  "birefnet-lite|onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx|5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333|224005088|core"
  "vitmatte-base|Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model.onnx|f6978437f5068849bcbf49b1f4e37b90aaca5155744e9fde4e6c689f70c2b9ee|387371620|full"
  "birefnet|onnx-community/BiRefNet-ONNX/resolve/main/onnx/model.onnx|58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c|972666916|full"
)
# NVIDIA's PyPI wheels (the versions ofx/ml/package/fetch.sh pins): name | path on PyPI | sha256 | bytes.
WHEELS=(
  "cublas|eb/1b/adc70acf3fde509b1f0ca29d1c0e8d9b00aaecd871f8cf877257c8b68d9c/nvidia_cublas-13.8.1.7-py3-none-manylinux_2_27_x86_64.whl|c11a27fd4379510e5b1f84b367a2514d1e52fe5cc13442117a0e0a1addee3cf2|439315821"
  "cudart|98/8a/3431271f6344874b8f1ac03f16b3d679c91493f8da63f716160403e6d0a0/nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl|9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5|2494438"
  "curand|07/73/3ee8e5b4cb891401e603ffd3a59b35c6afe785fd2de123afe7c7029603dc/nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl|25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5|61498332"
  "cudnn|af/75/96ea5c5368eb595c39d629cde08a66227a864e71ccb0e593add2612bb952/nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl|9677e76f21862eb5da7ee5ed69d544738b2d8b5c3ce7e5ec125c5592e6cdbdc8|536771498"
)
# The CUDA libraries the runner opens (ofx/ml/src/gate.cpp, the same 11; nothing else is needed:
# cuDNN's cnn / adv / ext are never loaded): file | wheel | sha256 | bytes.
LIBS=(
  "libcudart.so.13|cudart|7fb00519a24512bf677b59e87921e5c05aac8246ceb6a57435c0d9dfeef364f6|798496"
  "libcublasLt.so.13|cublas|5c4b539a1df9ce975e042c852bfc80f7a7e439873d9298d4f699a28f6db112c4|545528240"
  "libcublas.so.13|cublas|0d381da85431cfe42c9741452d92233b085b08cf0f237d0db2f4e03adff8842f|57594520"
  "libcurand.so.10|curand|21bb4e5731e8bc3f1656b9c51f4a56ebcd27c3173e6ee80b82a2b3c0c8bd2473|126468312"
  "libcudnn_graph.so.9|cudnn|67df1f319154780e3ac7ef606d838c142b96f9c1b77e0ec95d7a5139d16e2800|134437112"
  "libcudnn_ops.so.9|cudnn|d301a6ffb44b5d8cf04f67f3b49037e1a31801a5365892b03b74f171deba19e2|39855912"
  "libcudnn_heuristic.so.9|cudnn|a24474ae86db461aa5ab7cc2470fb9b88e09449d8bd6a8bba5d26eff20db2e29|96473904"
  "libcudnn_engines_precompiled.so.9|cudnn|0edc16383897f60dd4466776d2421d5351e023fd2950ac62bf5cf68ee399f42b|272654912"
  "libcudnn_engines_runtime_compiled.so.9|cudnn|334022cd6b47115ced9e3a289963d204705abfccefec052fd85234cb1d0e6355|44075680"
  "libcudnn_engines_tensor_ir.so.9|cudnn|4b47408d368863e5f37aeb019d8a05224213cf74335ce3ca6a8ba11d39ce70b3|216026256"
  "libcudnn.so.9|cudnn|4c3ccb552ee0c43aacca7498e12914fbda31be4fcb52cbdc1aaa87e78fb96e19|133336"
)

die() { echo "error: $*" >&2; exit 1; }
mb() { echo "$(( ($1 + 500000) / 1000000 )) MB"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
ask() {  # ask "question": true on y / yes; an empty answer or end of input = no
  local a=""
  printf '%s' "$1"
  read -r a || true
  [ -t 0 ] || echo "$a"  # answers from a pipe: shown, so the output reads as a dialogue
  case "$a" in y | Y | yes | Yes | YES) return 0 ;; *) return 1 ;; esac
}
config_get() { [ -f "$CONFIG" ] && grep "^$1=" "$CONFIG" | tail -n 1 | cut -d= -f2- || true; }
config_write() {  # config_write KEY [VALUE]: drop every KEY= line, then add KEY=VALUE if a value is given
  local tmp="$CONFIG.tmp.$$"
  mkdir -p "$(dirname "$CONFIG")"
  { if [ -f "$CONFIG" ]; then grep -v "^$1=" "$CONFIG" || true; fi
    if [ $# -ge 2 ]; then printf '%s=%s\n' "$1" "$2"; fi; } >"$tmp"
  mv -f "$tmp" "$CONFIG"
}

[ "$(id -u)" -ne 0 ] || die "do not run setup.sh as root or with sudo: run it as the user who runs Flame (it installs into that user's home folder)."

ACCEPT=0 FULL="" DATA="" FROM="" REMOVE=0 YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --accept-licences) ACCEPT=1 ;;
    --full) FULL=1 ;;
    --no-full) FULL=0 ;;
    --prefix) [ $# -ge 2 ] || die "--prefix needs a folder"; DATA="$2"; shift ;;
    --from) [ $# -ge 2 ] || die "--from needs a folder"; FROM="$(cd "$2" && pwd)" || die "no folder $2"; shift ;;
    --remove) REMOVE=1 ;;
    -y) YES=1 ;;
    -h | --help) sed -n '2,25p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done

# --remove: the data folder setup.sh wrote (from the config when no --prefix is given).
if [ "$REMOVE" = 1 ]; then
  if [ -z "$DATA" ]; then
    md="$(config_get ml_model_dir)" cl="$(config_get ml_cuda_libs)"
    if [ -n "$md" ] && [ "${md%/models}/cuda_libs" = "$cl" ]; then DATA="${md%/models}"; fi
  fi
  DATA="${DATA:-$HOME/.local/share/matte-refine-ml}"
  echo "This deletes $DATA/models, $DATA/cuda_libs and $DATA/downloads,"
  echo "and the lines ml_model_dir, ml_cuda_libs, ml_licence_accepted and ml_full_declined in $CONFIG."
  [ "$YES" = 1 ] || ask "Remove Matte Refine ML's models and libraries? [y/N] " || { echo "Nothing was changed."; exit 1; }
  rm -rf "${DATA:?}/models" "${DATA:?}/cuda_libs" "${DATA:?}/downloads"
  rmdir "$DATA" 2>/dev/null || true
  if [ -f "$CONFIG" ]; then for k in ml_model_dir ml_cuda_libs ml_licence_accepted ml_full_declined; do config_write "$k"; done; fi
  echo "Removed. Matte Refine ML's nodes pass the matte through until setup.sh is run again; restart Flame."
  exit 0
fi

DATA="${DATA:-$HOME/.local/share/matte-refine-ml}"
for t in curl sha256sum unzip df; do command -v "$t" >/dev/null || die "$t is missing (install it, then run setup.sh again)"; done
[ -f "$LICENCES" ] || die "licences.txt not found next to setup.sh"

# 1-2. The licences.
HASH="$(sha "$LICENCES")"
if [ "$(config_get ml_licence_accepted)" = "$HASH" ]; then
  echo "Licences: already accepted ($LICENCES)."
elif [ "$ACCEPT" = 1 ]; then
  echo "Licences: accepted with --accept-licences ($LICENCES)."
  config_write ml_licence_accepted "$HASH"
else
  if [ -t 0 ] && [ -t 1 ]; then more "$LICENCES"; else cat "$LICENCES"; fi
  echo
  if ! ask "Accept these licences? [y/N] "; then
    echo "Not accepted: nothing was downloaded and nothing was changed."
    exit 1
  fi
  config_write ml_licence_accepted "$HASH"
fi

mkdir -p "$DATA" || die "cannot create $DATA"
DATA="$(cd "$DATA" && pwd)"

# 3. The full models: asked once. Kept without asking when both are already installed; a "no" is
# remembered (ml_full_declined=1) and not asked again: --full adds them later.
want_full() {  # both full models are there (verified in step 4 like every file)
  local m
  for m in "${MODELS[@]}"; do
    IFS='|' read -r id _ _ _ set <<<"$m"
    if [ "$set" = full ] && [ ! -f "$DATA/models/$id/model.onnx" ]; then return 1; fi
  done
}
if [ -z "$FULL" ]; then
  if want_full; then
    FULL=1
  elif [ "$(config_get ml_full_declined)" = 1 ]; then
    FULL=0
  else
    full_bytes=0
    for m in "${MODELS[@]}"; do IFS='|' read -r _ _ _ b set <<<"$m"; [ "$set" = full ] && full_bytes=$((full_bytes + b)); done
    FULL=0
    ask "Also install the full models (ViTMatte Base, BiRefNet full: about $(mb $full_bytes) more)? [y/N] " && FULL=1
  fi
fi

# 4. What is missing (every file present is verified now, once).
echo "Checking $DATA ..."
declare -A OK=()
todo_models=() todo_libs=() need=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r id _ sum b set <<<"$m"
  [ "$set" = core ] || [ "$FULL" = 1 ] || continue
  f="$DATA/models/$id/model.onnx"
  if [ -f "$f" ] && [ "$(sha "$f")" = "$sum" ]; then OK[$id]=1; else todo_models+=("$m"); need=$((need + b)); fi
done
for l in "${LIBS[@]}"; do
  IFS='|' read -r name _ sum b <<<"$l"
  f="$DATA/cuda_libs/$name"
  if [ -f "$f" ] && [ "$(sha "$f")" = "$sum" ]; then OK[$name]=1; else todo_libs+=("$l"); need=$((need + b)); fi
done
# The wheels to fetch: those holding a missing library that --from cannot give as a plain file.
todo_wheels=()
for w in "${WHEELS[@]}"; do
  IFS='|' read -r wn _ _ wb <<<"$w"
  for l in "${todo_libs[@]+"${todo_libs[@]}"}"; do
    IFS='|' read -r name lw _ _ <<<"$l"
    [ "$lw" = "$wn" ] || continue
    [ -n "$FROM" ] && [ -f "$FROM/cuda_libs/$name" ] && continue
    todo_wheels+=("$w")
    [ -n "$FROM" ] || need=$((need + wb))
    break
  done
done
if [ $((${#todo_models[@]} + ${#todo_libs[@]})) -eq 0 ]; then
  echo "All files are installed and verified."
else
  avail_kb="$(df -Pk "$DATA" | awk 'NR == 2 { print $4 }')"
  if [ $((avail_kb * 1024)) -lt $((need + 100000000)) ]; then
    die "not enough disk space in $DATA: $(mb "$need") needed (plus 100 MB), $(mb $((avail_kb * 1024))) free. Free some space or use --prefix DIR on a larger disk."
  fi
  echo "To install: ${#todo_models[@]} model(s), ${#todo_libs[@]} CUDA libraries; $(mb "$need") of disk space needed, $(mb $((avail_kb * 1024))) free."
fi

PROGRESS=-sS
[ -t 2 ] && PROGRESS=-#
fetch() {  # fetch URL DEST SHA BYTES LABEL: download to DEST.part (resuming), verify, rename
  local url="$1" dest="$2" sum="$3" bytes="$4" label="$5" part="$2.part" have=0
  mkdir -p "$(dirname "$dest")"
  [ -f "$part" ] && have="$(stat -c %s "$part")"
  if [ "$have" -gt "$bytes" ]; then rm -f "$part"; have=0; fi
  if [ "$have" -lt "$bytes" ]; then
    if [ "$have" -gt 0 ]; then echo "  $label: resuming at $(mb "$have") of $(mb "$bytes")"; else echo "  $label: downloading $(mb "$bytes")"; fi
    curl -fL --retry 3 --connect-timeout 30 "$PROGRESS" -C - -o "$part" "$url" ||
      die "$label: the download failed ($url). Run setup.sh again to resume."
  fi
  if [ "$(sha "$part")" != "$sum" ]; then
    rm -f "$part"
    die "$label: the downloaded file does not match its SHA-256 (deleted). Run setup.sh again."
  fi
  mv -f "$part" "$dest"
}
copy_verified() {  # copy_verified SRC DEST SHA LABEL (offline: --from)
  local src="$1" dest="$2" sum="$3" label="$4"
  [ -f "$src" ] || die "$label: $src is missing (--from)"
  mkdir -p "$(dirname "$dest")"
  echo "  $label: copying from $src"
  cp -f "$src" "$dest.part"
  if [ "$(sha "$dest.part")" != "$sum" ]; then rm -f "$dest.part"; die "$label: $src does not match its SHA-256"; fi
  mv -f "$dest.part" "$dest"
}

n=0 total=$((${#todo_models[@]} + ${#todo_wheels[@]}))
for m in "${todo_models[@]+"${todo_models[@]}"}"; do
  IFS='|' read -r id path sum b _ <<<"$m"
  n=$((n + 1))
  dest="$DATA/models/$id/model.onnx"
  if [ -n "$FROM" ]; then copy_verified "$FROM/models/$id/model.onnx" "$dest" "$sum" "[$n/$total] model $id"
  else fetch "$HF/$path" "$dest" "$sum" "$b" "[$n/$total] model $id"; fi
  OK[$id]=1
done
mkdir -p "$DATA/cuda_libs"
if [ -n "$FROM" ]; then  # plain library files first (fetch.sh's cuda_libs/), with NVIDIA's licence texts if there
  if [ -d "$FROM/cuda_libs/licences" ]; then mkdir -p "$DATA/cuda_libs/licences" && cp -f "$FROM/cuda_libs/licences/"* "$DATA/cuda_libs/licences/"; fi
  for l in "${todo_libs[@]+"${todo_libs[@]}"}"; do
    IFS='|' read -r name _ sum _ <<<"$l"
    [ -f "$FROM/cuda_libs/$name" ] && copy_verified "$FROM/cuda_libs/$name" "$DATA/cuda_libs/$name" "$sum" "$name" && OK[$name]=1
  done
fi
for w in "${todo_wheels[@]+"${todo_wheels[@]}"}"; do
  IFS='|' read -r wn path wsum wb <<<"$w"
  n=$((n + 1))
  whl="$DATA/downloads/$wn.whl"
  if [ -n "$FROM" ]; then copy_verified "$FROM/wheels/$wn.whl" "$whl" "$wsum" "[$n/$total] NVIDIA $wn package"
  else fetch "$PYPI/$path" "$whl" "$wsum" "$wb" "[$n/$total] NVIDIA $wn package"; fi
  mkdir -p "$DATA/cuda_libs/licences"
  unzip -p "$whl" '*.dist-info/licenses/License.txt' >"$DATA/cuda_libs/licences/$wn-License.txt" || true
  for l in "${todo_libs[@]}"; do
    IFS='|' read -r name lw sum _ <<<"$l"
    [ "$lw" = "$wn" ] && [ -z "${OK[$name]:-}" ] || continue
    rm -rf "$DATA/cuda_libs/.unzip" && mkdir -p "$DATA/cuda_libs/.unzip"
    unzip -q -o -j "$whl" "nvidia/*/lib/$name" -d "$DATA/cuda_libs/.unzip" || die "$name not found in the $wn package"
    [ "$(sha "$DATA/cuda_libs/.unzip/$name")" = "$sum" ] || die "$name (from the $wn package) does not match its SHA-256"
    mv -f "$DATA/cuda_libs/.unzip/$name" "$DATA/cuda_libs/$name"
    OK[$name]=1
  done
  rm -rf "$DATA/cuda_libs/.unzip" "$whl"
done
rmdir "$DATA/downloads" 2>/dev/null || true
for l in "${LIBS[@]}"; do IFS='|' read -r name _ _ _ <<<"$l"; [ -n "${OK[$name]:-}" ] || die "$name was not installed"; done

# 5. The config: only these keys change.
config_write ml_model_dir "$DATA/models"
config_write ml_cuda_libs "$DATA/cuda_libs"
config_write ml_licence_accepted "$HASH"
if [ "$FULL" = 1 ] || want_full; then config_write ml_full_declined; else config_write ml_full_declined 1; fi

# 6. Done.
echo
echo "Matte Refine ML is set up for $(id -un):"
for m in "${MODELS[@]}"; do
  IFS='|' read -r id _ _ _ _ <<<"$m"
  [ -n "${OK[$id]:-}" ] && echo "  model  $DATA/models/$id/model.onnx"
done
if [ -d "$DATA/cuda_libs/licences" ]; then lt="NVIDIA's licence texts in cuda_libs/licences"; else lt="NVIDIA's licences: see licences.txt"; fi
echo "  CUDA   $DATA/cuda_libs (${#LIBS[@]} libraries; $lt)"
echo "  config $CONFIG (ml_model_dir, ml_cuda_libs, ml_licence_accepted)"
want_full || echo "  The full models (ViTMatte Base, BiRefNet Full) are not installed: add them with $0 --full"
echo "  size   $(du -sh "$DATA" | cut -f1) in $DATA"
echo "Restart Flame (quit it completely) so the Matte Refine ML nodes pick this up."
