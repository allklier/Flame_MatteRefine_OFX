# Edge Recipes WP8: end-to-end scores of Hair Locks on synthetic sequences (er_score; not a ctest test apart from the
# smoke below). Usage in er_score.cpp; results and how to read them: the summary.md it writes.
add_executable(er_score er_score.cpp er_report.cpp er_scenes.cpp er_host.cpp)
target_include_directories(er_score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OFX_DIR}/include)
target_link_libraries(er_score PRIVATE er_hair_locks mr_tracer_scenes mr_synth_lib ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(er_score PRIVATE -Wall -Wextra)
add_dependencies(er_score EdgeRecipes)
add_test(NAME recipes_score_smoke COMMAND er_score smoke ${CMAKE_BINARY_DIR}/EdgeRecipes.ofx.bundle/Contents/${MR_ARCH_DIR}/EdgeRecipes.ofx
  ${CMAKE_CURRENT_BINARY_DIR}/smoke_out)
set_tests_properties(recipes_score_smoke PROPERTIES LABELS smoke)
# hl_probe: Hair Locks' cores on raw frames from disk, every stage's plane written (real-footage reproduction; agent hl96).
add_executable(hl_probe hl_probe.cpp)
target_link_libraries(hl_probe PRIVATE er_hair_locks mr_ml_client mr_ml_support Threads::Threads)
target_compile_definitions(hl_probe PRIVATE ER_RUNNER_BUILD_ID="ER 0.1.0 ${MR_GIT}")  # = plugin/CMakeLists.txt ER_RUNNER_BUILD_ID
