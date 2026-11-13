# Edge Recipes Flame probe (docs/edge_recipes.md 4.9, work package 0b): three effects from one binary,
# grouping "Edge Recipes", one factory type per effect. Not a product: own bundle, off by default
# (MR_BUILD_RECIPES_PROBE), linked like Matte Refine (C++14, two exports, libstdc++ dynamic).
set(ERP_VERSION "0.1.0")
set(ERP_BUNDLE_DIR ${CMAKE_BINARY_DIR}/EdgeRecipesProbe.ofx.bundle)
add_library(EdgeRecipesProbe MODULE recipes_probe.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/log.cpp ${CMAKE_SOURCE_DIR}/ofx/src/user_config.cpp)
target_include_directories(EdgeRecipesProbe PRIVATE ${CMAKE_SOURCE_DIR}/ofx/src ${OFX_DIR}/Support/Library)
target_link_libraries(EdgeRecipesProbe PRIVATE ofx_support Threads::Threads)
target_compile_definitions(EdgeRecipesProbe PRIVATE ERP_VERSION="${ERP_VERSION}" MR_GIT="${MR_GIT}")
target_compile_options(EdgeRecipesProbe PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
if(APPLE)  # untested: the layout only
  target_link_options(EdgeRecipesProbe PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
else()
  target_link_options(EdgeRecipesProbe PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(EdgeRecipesProbe PROPERTIES PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${ERP_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

# All three effects through the Flame-like host, by plugin index (ofx_host.h Host(path, index)).
add_executable(test_recipes_probe test_recipes_probe.cpp ${CMAKE_SOURCE_DIR}/ofx/merge/systest/ofx_host.cpp)
target_include_directories(test_recipes_probe PRIVATE ${CMAKE_SOURCE_DIR}/ofx/merge/systest ${OFX_DIR}/include)
target_link_libraries(test_recipes_probe PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
add_dependencies(test_recipes_probe EdgeRecipesProbe)
add_test(NAME recipes_probe COMMAND test_recipes_probe
  ${ERP_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/EdgeRecipesProbe.ofx ${CMAKE_CURRENT_BINARY_DIR}/probe_home)
set_tests_properties(recipes_probe PROPERTIES LABELS smoke)
