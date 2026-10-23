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

add_executable(mr_ml_bench tools/mr_ml_bench.cpp)
target_include_directories(mr_ml_bench PRIVATE ${ORT_DIR}/include)
target_link_libraries(mr_ml_bench PRIVATE mr_ops mr_synth_lib ${ORT_LIB})
target_compile_options(mr_ml_bench PRIVATE -Wall -Wextra)
# ORT 1.31 headers (C and C++ API) need C++17 (noexcept function typedefs, structured
# bindings); the plugin code stays C++14, so ORT must live in its own C++17 translation units.
set_target_properties(mr_ml_bench PROPERTIES CXX_STANDARD 17 BUILD_RPATH ${ORT_DIR}/lib)

# The tiling covers every unknown pixel (no model needed); a smoke run when MR_ML_MODELS names a
# folder with vitmatte-small/model.onnx, else skipped (exit code 77).
add_test(NAME ml_tiling COMMAND mr_ml_bench --selfcheck)
set(MR_ML_MODELS "" CACHE PATH "Folder with the downloaded models (ofx/ml smoke test)")
add_test(NAME ml_smoke COMMAND sh -c
  "m='${MR_ML_MODELS}/vitmatte-small/model.onnx'; [ -f \"$m\" ] || { echo 'no model: skipped'; exit 77; }; exec \"$0\" --model \"$m\" --ep cpu --scene coarse --size 640x360 --tile 256 --repeat 1"
  $<TARGET_FILE:mr_ml_bench>)
set_tests_properties(ml_smoke PROPERTIES SKIP_RETURN_CODE 77)
