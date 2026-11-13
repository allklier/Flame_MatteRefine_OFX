# Edge Recipes WP4: the model candidate for the Focus Region (model_candidate.h, docs/model_candidate.md).
# C++14, no ONNX Runtime: the crops go to mr_ml_runner through RunnerClient (runner_crops.h).
add_library(mr_recipe_model STATIC model_candidate.cpp)
target_include_directories(mr_recipe_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mr_recipe_model PUBLIC mr_ops mr_ml_client)
target_compile_options(mr_recipe_model PRIVATE -Wall -Wextra -fvisibility=hidden)

# Fake runners (in process, and the fake mr_ml_runner process for the RunnerClient adapter). Smoke.
add_executable(test_model_candidate tests/test_model_candidate.cpp)
target_link_libraries(test_model_candidate PRIVATE mr_recipe_model mr_ml_client Threads::Threads)
add_dependencies(test_model_candidate test_fake_runner)
add_test(NAME recipe_model_candidate COMMAND test_model_candidate $<TARGET_FILE:test_fake_runner>)
set_tests_properties(recipe_model_candidate PROPERTIES LABELS smoke TIMEOUT 120)

# The evaluation with the real mr_ml_runner on the research's synthetic scenes (not a test; see the file).
add_executable(mr_model_eval tests/model_eval.cpp)
target_link_libraries(mr_model_eval PRIVATE mr_recipe_model mr_ml_client mr_ml_core Threads::Threads)
add_dependencies(mr_model_eval mr_ml_runner)
