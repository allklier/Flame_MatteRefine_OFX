# Edge Recipes core (WP2, docs/edge_recipes.md sections 2-3): step interface, per-render evidence,
# planner, the step adapters over the existing cores, the Hair Locks recipe. Pure C++14, no OFX.
add_library(mr_recipe_core STATIC recipe_core.cpp planner.cpp steps.cpp hair_locks.cpp)
target_include_directories(mr_recipe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mr_recipe_core PUBLIC mr_ml_chain mr_ml_jump mr_ml_temporal mr_merge_core mr_ops)
target_compile_options(mr_recipe_core PRIVATE -Wall -Wextra -fvisibility=hidden)

add_executable(test_recipe_core tests/test_recipe_core.cpp)
target_link_libraries(test_recipe_core PRIVATE mr_recipe_core)
add_test(NAME recipe_core COMMAND test_recipe_core)
add_executable(test_hair_locks tests/test_hair_locks.cpp)
target_link_libraries(test_hair_locks PRIVATE mr_recipe_core mr_synth_lib)
add_test(NAME recipe_hair_locks COMMAND test_hair_locks)
set_tests_properties(recipe_core recipe_hair_locks PROPERTIES LABELS smoke)
