# Merge Probe (ofx/merge/probe): a logging-only plugin that shows how Flame lays out and delivers the
# inputs planned for Matte Merge (decision 37, ofx/merge/CLAUDE.md "Inputs"). Not a product: own bundle,
# off by default (MR_BUILD_MERGE_PROBE), linked like Matte Refine (C++14, two exports, libstdc++ dynamic).
set(MP_VERSION "0.1.0")
set(MP_BUNDLE_DIR ${CMAKE_BINARY_DIR}/MergeProbe.ofx.bundle)
add_library(MergeProbe MODULE merge_probe.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/log.cpp ${CMAKE_SOURCE_DIR}/ofx/src/user_config.cpp)
target_include_directories(MergeProbe PRIVATE ${CMAKE_SOURCE_DIR}/ofx/src ${OFX_DIR}/Support/Library)
target_link_libraries(MergeProbe PRIVATE ofx_support Threads::Threads)
target_compile_definitions(MergeProbe PRIVATE MP_VERSION="${MP_VERSION}" MR_GIT="${MR_GIT}")
target_compile_options(MergeProbe PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
if(APPLE)  # untested: the layout only
  target_link_options(MergeProbe PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
else()
  target_link_options(MergeProbe PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(MergeProbe PROPERTIES PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${MP_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

add_executable(test_load_probe test_load_probe.cpp)
target_include_directories(test_load_probe PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_probe PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_load_probe MergeProbe)
add_test(NAME merge_probe_load COMMAND test_load_probe ${MP_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MergeProbe.ofx)
