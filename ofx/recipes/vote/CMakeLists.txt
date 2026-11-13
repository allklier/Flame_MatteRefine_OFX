# Strand presence vote (Edge Recipes WP6, ofx/recipes/vote/strand_vote.h): library mr_strand_vote, pure C++14.
# Consumes the tracer's results (mr_tracer) and Temporal's flows (mr_ml_temporal, for motionFromFlows).
add_library(mr_strand_vote STATIC strand_vote.cpp)
target_include_directories(mr_strand_vote PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mr_strand_vote PUBLIC mr_tracer mr_ml_temporal)
target_compile_options(mr_strand_vote PRIVATE -Wall -Wextra -fvisibility=hidden)

# Tests: known answers, region bit-identity, determinism, rule mutations (smoke); the synthetic hair-locks
# sequence: flicker before / after, coverage, flow failure, the Q3 Temporal A/B, timing (slow).
add_executable(test_strand_vote tests/test_strand_vote.cpp)
target_link_libraries(test_strand_vote PRIVATE mr_strand_vote mr_ml_temporal mr_synth_lib)
add_test(NAME strand_vote COMMAND test_strand_vote)
set_tests_properties(strand_vote PROPERTIES LABELS smoke)
add_test(NAME strand_vote_sequence COMMAND test_strand_vote --sequence)
set_tests_properties(strand_vote_sequence PROPERTIES LABELS slow TIMEOUT 600)
