# Matte Merge (ofx/merge/CLAUDE.md, decision 37): only with -DMR_BUILD_MERGE=ON. The classical core
# (pure C++14, no OFX), its synthetic scenes and tests, then the plugin (MatteMerge.ofx.bundle).
add_library(mr_merge_core STATIC src/matte_merge.cpp src/merge_register.cpp src/merge_picreg.cpp src/merge_evidence.cpp
  src/merge_morph.cpp)
target_include_directories(mr_merge_core PUBLIC src)
target_link_libraries(mr_merge_core PUBLIC mr_ops)
target_compile_options(mr_merge_core PRIVATE -Wall -Wextra -fvisibility=hidden)

add_library(mr_merge_synth STATIC tools/merge_synth.cpp)
target_include_directories(mr_merge_synth PUBLIC tools)
target_link_libraries(mr_merge_synth PUBLIC mr_synth_lib)
add_executable(mm_synth tools/mm_synth.cpp)
target_link_libraries(mm_synth PRIVATE mr_merge_synth)

add_executable(test_merge_core tests/test_merge_core.cpp)
target_link_libraries(test_merge_core PRIVATE mr_merge_core)
add_test(NAME merge_core COMMAND test_merge_core)
add_executable(test_merge_scenes tests/test_merge_scenes.cpp)
target_link_libraries(test_merge_scenes PRIVATE mr_merge_core mr_merge_synth)
add_test(NAME merge_scenes COMMAND test_merge_scenes)
add_executable(test_merge_align tests/test_merge_align.cpp)
target_link_libraries(test_merge_align PRIVATE mr_merge_core mr_merge_synth)
add_test(NAME merge_align COMMAND test_merge_align)

# The plugin (decision 37, as MatteRefineML, decision 27): bundle MatteMerge.ofx.bundle, its own
# version, linked like Matte Refine (C++14, only the two OFX entry points, libstdc++ dynamic).
set(MM_VERSION "0.1.0")
string(REPLACE "." ";" MM_VERSION_PARTS "${MM_VERSION}")
list(GET MM_VERSION_PARTS 0 MM_VERSION_MAJOR)
list(GET MM_VERSION_PARTS 1 MM_VERSION_MINOR)
set(MM_BUNDLE_DIR ${CMAKE_BINARY_DIR}/MatteMerge.ofx.bundle)

# Plugin logic without OFX types (param mapping, footers, colourspace answer): unit-tested.
add_library(mr_merge_ui STATIC src/merge_ui.cpp)
target_include_directories(mr_merge_ui PUBLIC src PRIVATE ${OFX_DIR}/include)
target_link_libraries(mr_merge_ui PUBLIC mr_merge_core)
target_compile_options(mr_merge_ui PRIVATE -Wall -Wextra -fvisibility=hidden)

add_library(MatteMerge MODULE src/plugin_merge.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/log.cpp ${CMAKE_SOURCE_DIR}/ofx/src/user_config.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/process.cpp ${CMAKE_SOURCE_DIR}/ofx/src/env_info.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/ocio_encoding.cpp ${CMAKE_SOURCE_DIR}/${MR_PLATFORM_SOURCES})
target_link_libraries(MatteMerge PRIVATE ofx_support mr_merge_ui mr_merge_core mr_ops Threads::Threads)
# ofxsSupportPrivate.h: retrieveImageEffectPointer for the colour entry point (decision 23).
target_include_directories(MatteMerge PRIVATE ${CMAKE_SOURCE_DIR}/ofx/src ${OFX_DIR}/Support/Library)
# MR_VERSION: env_info's "plugin: Matte Refine <MR_VERSION>" line reads "Matte Refine Merge 0.1.0".
target_compile_definitions(MatteMerge PRIVATE MM_VERSION="${MM_VERSION}" MR_GIT="${MR_GIT}"
  MR_VERSION="Merge ${MM_VERSION}" MM_VERSION_MAJOR=${MM_VERSION_MAJOR} MM_VERSION_MINOR=${MM_VERSION_MINOR})
target_compile_options(MatteMerge PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
if(APPLE)  # decision 13: as Matte Refine (root CMakeLists.txt)
  target_link_options(MatteMerge PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
  file(WRITE ${MM_BUNDLE_DIR}/Contents/Info.plist
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key><string>MatteMerge.ofx</string>
  <key>CFBundleIdentifier</key><string>com.janklier.mattemerge</string>
  <key>CFBundleName</key><string>Matte Merge</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>${MM_VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${MM_VERSION}</string>
</dict>
</plist>
")
else()
  target_link_options(MatteMerge PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(MatteMerge PROPERTIES PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${MM_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

add_executable(test_merge_ui tests/test_merge_ui.cpp)
target_link_libraries(test_merge_ui PRIVATE mr_merge_ui)
add_test(NAME merge_ui COMMAND test_merge_ui)
add_executable(test_load_bundle_merge tests/test_load_bundle_merge.cpp)
target_include_directories(test_load_bundle_merge PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle_merge PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_load_bundle_merge MatteMerge)
add_test(NAME merge_load_bundle COMMAND test_load_bundle_merge
  ${MM_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MatteMerge.ofx)
if(NOT APPLE)
  add_test(NAME merge_bundle_exports COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_exports.sh
    ${MM_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MatteMerge.ofx)
endif()

# System test: MatteMerge.ofx in a Flame-like host (systest/ofx_host.*, from LTX Alpha Gen) on the
# synthetic scenes; the render equals merge() called directly. Its own HOME (log, config, bundle).
add_executable(merge_systest systest/merge_systest.cpp systest/ofx_host.cpp)
target_include_directories(merge_systest PRIVATE systest ${OFX_DIR}/include)
target_link_libraries(merge_systest PRIVATE mr_merge_ui mr_merge_synth ${CMAKE_DL_LIBS} Threads::Threads)
add_dependencies(merge_systest MatteMerge)
add_test(NAME merge_systest COMMAND merge_systest ${MM_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/MatteMerge.ofx
  ${CMAKE_CURRENT_BINARY_DIR}/systest_home)
set_tests_properties(merge_systest PROPERTIES TIMEOUT 300)
