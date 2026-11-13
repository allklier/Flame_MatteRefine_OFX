# Strand tracer (Edge Recipes WP3, docs/strand_tracing.md): library mr_tracer, pure C++14 on float arrays.
add_library(mr_tracer STATIC strand_tracer.cpp)
target_include_directories(mr_tracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(mr_tracer PRIVATE -Wall -Wextra -fvisibility=hidden)

# Tests: synthetic scenes of the prototype (tests/tracer_scenes.*), metrics, rule ablation.
add_library(mr_tracer_scenes STATIC tests/tracer_scenes.cpp)
target_include_directories(mr_tracer_scenes PUBLIC tests)
target_link_libraries(mr_tracer_scenes PUBLIC mr_tracer)
target_compile_options(mr_tracer_scenes PRIVATE -Wall -Wextra)

add_executable(test_strand_tracer tests/test_strand_tracer.cpp)
target_link_libraries(test_strand_tracer PRIVATE mr_tracer_scenes)
add_test(NAME strand_tracer COMMAND test_strand_tracer)
set_tests_properties(strand_tracer PROPERTIES LABELS smoke)
add_test(NAME strand_tracer_scenes COMMAND test_strand_tracer --scenes)
set_tests_properties(strand_tracer_scenes PROPERTIES LABELS slow)
add_test(NAME strand_tracer_ablation COMMAND test_strand_tracer --ablation)
set_tests_properties(strand_tracer_ablation PROPERTIES LABELS slow)

# Not a test: evaluates raw float dumps (prototype scenes, one real crop; numbers only), see tools/tracer_eval.cpp.
add_executable(tracer_eval tools/tracer_eval.cpp)
target_link_libraries(tracer_eval PRIVATE mr_tracer_scenes)
