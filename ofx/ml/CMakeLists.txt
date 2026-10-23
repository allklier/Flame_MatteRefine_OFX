# Matte Refine ML (ofx/ml/CLAUDE.md): only with -DMR_BUILD_ML=ON. Phase 0: the benchmark tool.
# ONNX Runtime from a prebuilt package: -DORT_DIR=<dir with include/ and lib/> or the ORT_DIR
# environment variable. No network access during the build.
if(NOT ORT_DIR)
  set(ORT_DIR $ENV{ORT_DIR})
endif()
if(NOT ORT_DIR OR NOT EXISTS ${ORT_DIR}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "MR_BUILD_ML: set ORT_DIR to an extracted ONNX Runtime package (include/, lib/)")
endif()
find_library(ORT_LIB onnxruntime PATHS ${ORT_DIR}/lib NO_DEFAULT_PATH REQUIRED)

# Edge Refine (B1) without a model or OFX: trimap, tiles, blend back (C++14; tests use a fake model).
add_library(mr_ml_core STATIC src/edge_refine.cpp src/model_table.cpp)
target_include_directories(mr_ml_core PUBLIC src)
target_link_libraries(mr_ml_core PUBLIC mr_ops)
target_compile_options(mr_ml_core PRIVATE -Wall -Wextra -fvisibility=hidden)

add_executable(mr_ml_bench tools/mr_ml_bench.cpp)
target_include_directories(mr_ml_bench PRIVATE ${ORT_DIR}/include)
target_link_libraries(mr_ml_bench PRIVATE mr_ml_core mr_ops mr_synth_lib ${ORT_LIB})
target_compile_options(mr_ml_bench PRIVATE -Wall -Wextra)
# ORT 1.31 headers (C and C++ API) need C++17 (noexcept function typedefs, structured
# bindings); the plugin code stays C++14, so ORT must live in its own C++17 translation units.
set_target_properties(mr_ml_bench PROPERTIES CXX_STANDARD 17 BUILD_RPATH ${ORT_DIR}/lib)

# The log and the per-user config, shared with Matte Refine (one log file; ofx/src, unchanged).
add_library(mr_ml_support STATIC ${CMAKE_SOURCE_DIR}/ofx/src/log.cpp ${CMAKE_SOURCE_DIR}/ofx/src/user_config.cpp)
target_include_directories(mr_ml_support PUBLIC ${CMAKE_SOURCE_DIR}/ofx/src)
target_compile_options(mr_ml_support PRIVATE -fvisibility=hidden)

# ViTMatte through ONNX Runtime: the only C++17 code, behind a C++14 header (ort_runner.h).
add_library(mr_ml_ort STATIC src/ort_runner.cpp)
target_include_directories(mr_ml_ort PRIVATE ${ORT_DIR}/include)
target_link_libraries(mr_ml_ort PUBLIC mr_ml_core mr_ml_support ${ORT_LIB} ${CMAKE_DL_LIBS})
target_compile_options(mr_ml_ort PRIVATE -Wall -Wextra -fvisibility=hidden)
set_target_properties(mr_ml_ort PROPERTIES CXX_STANDARD 17)

set(MRML_VERSION "0.1.0")
string(REPLACE "." ";" MRML_VERSION_PARTS "${MRML_VERSION}")
list(GET MRML_VERSION_PARTS 0 MRML_VERSION_MAJOR)
list(GET MRML_VERSION_PARTS 1 MRML_VERSION_MINOR)
set(MRML_BUNDLE_DIR ${CMAKE_BINARY_DIR}/MatteRefineML.ofx.bundle)

# The runner protocol (decision 32). The client is C++14 and links nothing GPU: it goes into the
# plugin. Both sides compare MRML_BUILD_ID at the handshake (a different build is replaced).
set(MRML_BUILD_ID "${MRML_VERSION} ${MR_GIT}")
add_library(mr_ml_client STATIC src/runner_ipc.cpp src/runner_client.cpp)
target_include_directories(mr_ml_client PUBLIC src)
target_link_libraries(mr_ml_client PUBLIC mr_ml_core mr_ml_support)
target_compile_definitions(mr_ml_client PRIVATE MRML_BUILD_ID="${MRML_BUILD_ID}")
target_compile_options(mr_ml_client PRIVATE -Wall -Wextra -fvisibility=hidden)
# The server side without a model (C++14): the real runner and the fake test runner.
add_library(mr_ml_server STATIC src/runner_server.cpp)
target_link_libraries(mr_ml_server PUBLIC mr_ml_client Threads::Threads)
target_compile_options(mr_ml_server PRIVATE -Wall -Wextra -fvisibility=hidden)

# mr_ml_runner: owns ORT, CUDA and the models; one per user and machine, started by the plugin.
# Dev builds: inside the bundle, Contents/Resources/mr_ml_runner (where the plugin looks by
# default), libonnxruntime through the RUNPATH = the ORT_DIR lib folder (absolute). A release
# package will put ORT in Contents/Resources/lib ($ORIGIN/lib, also in the RUNPATH); the CUDA
# libraries come from the config's ml_cuda_libs (runner_main.cpp).
add_executable(mr_ml_runner src/runner_main.cpp)
target_include_directories(mr_ml_runner PRIVATE ${ORT_DIR}/include)
target_link_libraries(mr_ml_runner PRIVATE mr_ml_server mr_ml_ort ${CMAKE_DL_LIBS})
target_compile_options(mr_ml_runner PRIVATE -Wall -Wextra)
set_target_properties(mr_ml_runner PROPERTIES CXX_STANDARD 17
  RUNTIME_OUTPUT_DIRECTORY ${MRML_BUNDLE_DIR}/Contents/Resources
  BUILD_RPATH "${ORT_DIR}/lib;\$ORIGIN/lib" INSTALL_RPATH "\$ORIGIN/lib")

# The plugin (decision 27): bundle MatteRefineML.ofx.bundle, its own version. No ONNX Runtime and
# no CUDA in it (decision 32; test ml_bundle_deps): tiles go to mr_ml_runner.
add_library(MatteRefineML MODULE src/plugin_ml.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/process.cpp ${CMAKE_SOURCE_DIR}/ofx/src/env_info.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/ocio_encoding.cpp ${CMAKE_SOURCE_DIR}/${MR_PLATFORM_SOURCES})
target_link_libraries(MatteRefineML PRIVATE ofx_support mr_ml_chain mr_ml_smart_fill mr_ml_client mr_ml_core mr_ops
  mr_ml_support ${CMAKE_DL_LIBS})
add_dependencies(MatteRefineML mr_ml_runner)
target_include_directories(MatteRefineML PRIVATE ${OFX_DIR}/Support/Library)
# MR_VERSION: env_info's "plugin: Matte Refine <MR_VERSION>" line reads "Matte Refine ML 0.1.0".
target_compile_definitions(MatteRefineML PRIVATE MRML_VERSION="${MRML_VERSION}" MR_GIT="${MR_GIT}"
  MR_VERSION="ML ${MRML_VERSION}"
  MRML_VERSION_MAJOR=${MRML_VERSION_MAJOR} MRML_VERSION_MINOR=${MRML_VERSION_MINOR})
target_compile_options(MatteRefineML PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
if(APPLE)  # untested (decision 13): the layout only
  target_link_options(MatteRefineML PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
else()
  target_link_options(MatteRefineML PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(MatteRefineML PROPERTIES PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${MRML_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

# Tests. The core with a fake model; the bundle loads without ORT or CUDA (decision 32) and links
# neither; a model smoke test through the in-process wrapper (skipped, exit 77, without
# MR_ML_MODELS; CUDA part skipped when no GPU is usable); the runner client against a fake runner;
# the real runner (skipped without the model).
add_executable(test_edge_refine tests/test_edge_refine.cpp)
target_link_libraries(test_edge_refine PRIVATE mr_ml_core mr_ops mr_synth_lib)
add_test(NAME ml_edge_refine COMMAND test_edge_refine)
add_executable(test_load_bundle_ml tests/test_load_bundle_ml.cpp)
target_include_directories(test_load_bundle_ml PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle_ml PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_load_bundle_ml MatteRefineML)
add_test(NAME ml_load_bundle COMMAND test_load_bundle_ml
  ${MRML_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MatteRefineML.ofx)
add_test(NAME ml_bundle_deps COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_bundle_deps.sh
  ${MRML_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MatteRefineML.ofx)
add_executable(test_fake_runner tests/fake_runner.cpp)
target_link_libraries(test_fake_runner PRIVATE mr_ml_server)
add_executable(test_runner_client tests/test_runner_client.cpp)
target_link_libraries(test_runner_client PRIVATE mr_ml_server Threads::Threads)
add_dependencies(test_runner_client test_fake_runner)
add_test(NAME ml_runner_client COMMAND test_runner_client $<TARGET_FILE:test_fake_runner>)
set_tests_properties(ml_runner_client PROPERTIES TIMEOUT 300)
set(MR_ML_CUDA_LIBS "" CACHE PATH "Folder with the CUDA libraries for the ml_model smoke test")
add_executable(test_ml_model tests/test_ml_model.cpp)
target_link_libraries(test_ml_model PRIVATE mr_ml_ort mr_ml_core mr_ops mr_synth_lib)
set_target_properties(test_ml_model PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_test(NAME ml_model COMMAND test_ml_model "${MR_ML_MODELS}" "${MR_ML_CUDA_LIBS}")
set_tests_properties(ml_model PROPERTIES SKIP_RETURN_CODE 77)
add_executable(test_runner_real tests/test_runner_real.cpp)
target_link_libraries(test_runner_real PRIVATE mr_ml_client mr_ml_ort mr_ml_core mr_ops mr_synth_lib Threads::Threads)
set_target_properties(test_runner_real PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_dependencies(test_runner_real mr_ml_runner)
add_test(NAME ml_runner_real COMMAND test_runner_real $<TARGET_FILE:mr_ml_runner> "${MR_ML_MODELS}" "${MR_ML_CUDA_LIBS}")
set_tests_properties(ml_runner_real PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# The tiling covers every unknown pixel (no model needed); a smoke run when MR_ML_MODELS names a
# folder with vitmatte-small/model.onnx, else skipped (exit code 77).
add_test(NAME ml_tiling COMMAND mr_ml_bench --selfcheck)
set(MR_ML_MODELS "" CACHE PATH "Folder with the downloaded models (ofx/ml smoke test)")
add_test(NAME ml_smoke COMMAND sh -c
  "m='${MR_ML_MODELS}/vitmatte-small/model.onnx'; [ -f \"$m\" ] || { echo 'no model: skipped'; exit 77; }; exec \"$0\" --model \"$m\" --ep cpu --scene coarse --size 640x360 --tile 256 --repeat 1"
  $<TARGET_FILE:mr_ml_bench>)
set_tests_properties(ml_smoke PROPERTIES SKIP_RETURN_CODE 77)

# Smart Fill Holes (B3) core: holes classified by a segmentation (SegRunner) and / or the picture
# colour; BiRefNet pre / post. C++14, no OFX, no ORT. Tests with fake segmentations; the evaluation
# on the "gaps" scene (fakes + the real BiRefNet in-process; the real part skipped, exit 77, without
# MR_ML_MODELS/birefnet-lite/model.onnx).
add_library(mr_ml_smart_fill STATIC src/smart_fill.cpp)
target_include_directories(mr_ml_smart_fill PUBLIC src)
target_link_libraries(mr_ml_smart_fill PUBLIC mr_ml_core mr_ops)
target_compile_options(mr_ml_smart_fill PRIVATE -Wall -Wextra -fvisibility=hidden)
add_executable(test_smart_fill tests/test_smart_fill.cpp)
target_link_libraries(test_smart_fill PRIVATE mr_ml_smart_fill mr_ops mr_synth_lib)
add_test(NAME ml_smart_fill COMMAND test_smart_fill)
add_executable(eval_smart_fill tests/eval_smart_fill.cpp)
target_link_libraries(eval_smart_fill PRIVATE mr_ml_smart_fill mr_ml_ort mr_ops mr_synth_lib)
set_target_properties(eval_smart_fill PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_test(NAME ml_smart_fill_real COMMAND eval_smart_fill --quick --real-only "${MR_ML_MODELS}" "${MR_ML_CUDA_LIBS}")
set_tests_properties(ml_smart_fill_real PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

# Temporal Stabilise (B4) without OFX: dense flow + the band / hole combine (C++14, no ORT).
add_library(mr_ml_temporal STATIC src/flow.cpp src/temporal.cpp)
target_include_directories(mr_ml_temporal PUBLIC src)
target_link_libraries(mr_ml_temporal PUBLIC mr_ops)
target_compile_options(mr_ml_temporal PRIVATE -Wall -Wextra -fvisibility=hidden)
add_executable(test_temporal tests/test_temporal.cpp)
target_link_libraries(test_temporal PRIVATE mr_ml_temporal mr_ops mr_synth_lib)
add_test(NAME ml_temporal COMMAND test_temporal)
# The evaluation on synth::makeSequence (not a test; ofx/ml/tests/temporal_eval.cpp for options).
add_executable(mr_temporal_eval tests/temporal_eval.cpp)
target_link_libraries(mr_temporal_eval PRIVATE mr_ml_temporal mr_ops mr_synth_lib)

# Render chain (matte -> Temporal -> Smart Fill -> Edge Refine) without OFX; the plugin links it.
add_library(mr_ml_chain STATIC src/render_chain.cpp)
target_include_directories(mr_ml_chain PUBLIC src)
target_link_libraries(mr_ml_chain PUBLIC mr_ml_smart_fill mr_ml_temporal mr_ml_client mr_ml_core mr_ops)
target_compile_options(mr_ml_chain PRIVATE -Wall -Wextra -fvisibility=hidden)
add_executable(test_render_chain tests/test_render_chain.cpp)
target_link_libraries(test_render_chain PRIVATE mr_ml_chain mr_ml_server mr_synth_lib Threads::Threads)
add_dependencies(test_render_chain test_fake_runner)
add_test(NAME ml_render_chain COMMAND test_render_chain $<TARGET_FILE:test_fake_runner>)
set_tests_properties(ml_render_chain PROPERTIES TIMEOUT 300)

# Edge Refine's trimap rule against the synthetic truth with the real ViTMatte (B1 polish, hair
# trimap; not a test: ofx/ml/tests/trimap_eval.cpp for options).
add_executable(mr_trimap_eval tests/trimap_eval.cpp)
target_link_libraries(mr_trimap_eval PRIVATE mr_ml_ort mr_ml_core mr_ops mr_synth_lib)
set_target_properties(mr_trimap_eval PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
