# Edge Recipes bundle (decision 41, docs/edge_recipes.md 1.1, 4): EdgeRecipes.ofx.bundle, one OFX effect per
# recipe. WP1: the multi-effect skeleton; WP7: Hair Locks runs its planner (ofx/recipes/core). Linked like Matte Refine ML (C++14, only the
# two OFX entry points exported, libstdc++ dynamic, no ONNX Runtime / CUDA in the .ofx, decision 32).
set(ER_VERSION "0.1.0")
set(ER_BUNDLE_DIR ${CMAKE_BINARY_DIR}/EdgeRecipes.ofx.bundle)
# The runner build id (plugin 2's is "<MRML_VERSION> <git>", ofx/ml/CMakeLists.txt MRML_BUILD_ID): own runner process.
set(ER_RUNNER_BUILD_ID "ER ${ER_VERSION} ${MR_GIT}")

# The recipe table without OFX types (layout unit test).
add_library(er_recipe_defs STATIC src/recipe_def.cpp)
target_include_directories(er_recipe_defs PUBLIC src)
target_compile_options(er_recipe_defs PRIVATE -Wall -Wextra -fvisibility=hidden)

# Hair Locks without OFX types: controls -> core params, Plan texts, view colours (hair_locks_ui), and the three
# candidate sources (hair_locks_sources: model, tracer, Detail).
add_library(er_hair_locks STATIC src/hair_locks_ui.cpp src/hair_locks_sources.cpp)
target_include_directories(er_hair_locks PUBLIC src)
target_link_libraries(er_hair_locks PUBLIC er_recipe_defs mr_recipe_core mr_recipe_model mr_tracer mr_strand_vote mr_merge_core
  mr_ml_chain mr_ops)
target_compile_options(er_hair_locks PRIVATE -Wall -Wextra -fvisibility=hidden)

# Shared support sources of the plugin binaries (built once for the three variants below).
add_library(er_plugin_support STATIC ${CMAKE_SOURCE_DIR}/ofx/src/process.cpp ${CMAKE_SOURCE_DIR}/ofx/src/env_info.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/ocio_encoding.cpp ${CMAKE_SOURCE_DIR}/${MR_PLATFORM_SOURCES})
target_include_directories(er_plugin_support PUBLIC ${CMAKE_SOURCE_DIR}/ofx/src)
target_compile_options(er_plugin_support PRIVATE -fvisibility=hidden)
target_compile_definitions(er_plugin_support PRIVATE MR_VERSION="Edge Recipes ${ER_VERSION}" MR_GIT="${MR_GIT}")

# name: target; outdir: where the .ofx goes; defs: extra compile definitions (test variants).
function(er_plugin name outdir)
  add_library(${name} MODULE src/plugin_recipes.cpp)
  target_link_libraries(${name} PRIVATE ofx_support er_hair_locks er_recipe_defs er_plugin_support mr_ml_client
    mr_ml_core mr_ml_support Threads::Threads ${CMAKE_DL_LIBS})
  target_include_directories(${name} PRIVATE ${OFX_DIR}/Support/Library)
  target_compile_definitions(${name} PRIVATE ER_VERSION="${ER_VERSION}" MR_GIT="${MR_GIT}"
    ER_RUNNER_BUILD_ID="${ER_RUNNER_BUILD_ID}" ${ARGN})
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
    -fvisibility=hidden)
  target_link_options(${name} PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
  set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX ".ofx" OUTPUT_NAME ${name}
    LIBRARY_OUTPUT_DIRECTORY ${outdir})
endfunction()

# The product: Hair Locks only.
er_plugin(EdgeRecipes ${ER_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})
# The bundle's own runner (docs/edge_recipes.md 1.1, 1.3): plugin 2's runner source built with Edge Recipes' own
# build id, so the two plugins never share or replace each other's runner process (mrml_runner, ofx/ml/CMakeLists.txt;
# the .ofx sends the same id, plugin_recipes.cpp runnerFor). Release layout (ORT in Contents/Resources/lib) as plugin 2.
mrml_runner(er_ml_runner ${ER_BUNDLE_DIR} "${ER_RUNNER_BUILD_ID}")
add_dependencies(EdgeRecipes er_ml_runner)
# Test builds (never packaged): + the Test Dummy recipe with its own factory type; and the mutant where the
# dummy shares Hair Locks' factory class, which the system test must reject.
set(ER_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_bundles)
er_plugin(EdgeRecipesTest ${ER_TEST_DIR} ER_TEST_RECIPES)
er_plugin(EdgeRecipesMutant ${ER_TEST_DIR} ER_TEST_RECIPES ER_MUTANT_SHARED_FACTORY)

# Tests (smoke: each well under a few seconds).
add_executable(test_hair_locks_ui tests/test_hair_locks_ui.cpp)
target_link_libraries(test_hair_locks_ui PRIVATE er_hair_locks)
add_test(NAME recipes_hair_locks_ui COMMAND test_hair_locks_ui)

add_executable(test_recipe_def tests/test_recipe_def.cpp)
target_link_libraries(test_recipe_def PRIVATE er_recipe_defs)
add_test(NAME recipes_def COMMAND test_recipe_def)

add_executable(test_load_bundle_recipes tests/test_load_bundle_recipes.cpp)
target_include_directories(test_load_bundle_recipes PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle_recipes PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_load_bundle_recipes EdgeRecipes EdgeRecipesTest)
add_test(NAME recipes_load_bundle COMMAND test_load_bundle_recipes
  ${ER_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/EdgeRecipes.ofx ${ER_TEST_DIR}/EdgeRecipesTest.ofx)

add_executable(recipes_systest tests/recipes_systest.cpp ${CMAKE_SOURCE_DIR}/ofx/merge/systest/ofx_host.cpp)
target_include_directories(recipes_systest PRIVATE ${CMAKE_SOURCE_DIR}/ofx/merge/systest ${OFX_DIR}/include)
target_link_libraries(recipes_systest PRIVATE er_hair_locks mr_tracer_scenes mr_synth_lib ${CMAKE_DL_LIBS}
  Threads::Threads)
target_compile_definitions(recipes_systest PRIVATE ER_RUNNER_BUILD_ID="${ER_RUNNER_BUILD_ID}")
add_dependencies(recipes_systest EdgeRecipes EdgeRecipesTest EdgeRecipesMutant test_fake_runner)
add_test(NAME recipes_systest COMMAND recipes_systest ${ER_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/EdgeRecipes.ofx
  ${ER_TEST_DIR}/EdgeRecipesTest.ofx ${CMAKE_CURRENT_BINARY_DIR}/systest_home $<TARGET_FILE:test_fake_runner>)
# Mutation check: the shared-factory build must FAIL the same descriptor checks.
add_test(NAME recipes_mutant COMMAND recipes_systest --mutant ${ER_TEST_DIR}/EdgeRecipesMutant.ofx
  ${CMAKE_CURRENT_BINARY_DIR}/mutant_home)
set_tests_properties(recipes_hair_locks_ui recipes_def recipes_load_bundle recipes_systest recipes_mutant PROPERTIES
  LABELS smoke)
# Timing through the host (not a test): synthetic HD and UHD frames, every step on, per-step ms from the log.
add_test(NAME recipes_timing COMMAND recipes_systest --timing ${ER_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/EdgeRecipes.ofx
  ${CMAKE_CURRENT_BINARY_DIR}/timing_home $<TARGET_FILE:test_fake_runner>)
set_tests_properties(recipes_timing PROPERTIES LABELS slow TIMEOUT 600)
set_tests_properties(recipes_mutant PROPERTIES WILL_FAIL TRUE)
