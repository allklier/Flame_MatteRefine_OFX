cmake_minimum_required(VERSION 3.20)
# macOS: the oldest macOS the bundle loads on (decision 13); without it, the build Mac's own
# version (minos 26 on macOS 26). 15.0 = what LTX Alpha Gen's Mac bundles carry. Before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Oldest macOS the bundle loads on")
project(MatteRefineOFX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)  # OFX 1.4 Support uses throw() specs and auto_ptr
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Release version (0.1.0 = Fill Holes; 0.1.1 = Solid Core, View Matte default, Solid Above
# 0.95, decimals, Save Matte, diagnostics; 0.1.2 = soft Solid Core (Feather px), macOS; 0.2.0 = Clean & Crush, Contour Smooth, Shrink/Grow, Edge Width, Snap to Image, Foreground Colour, QC views, matte on source alpha). Also the OFX plugin version (major.minor) and the
# release package name.
set(MR_VERSION "0.3.0")
string(REPLACE "." ";" MR_VERSION_PARTS "${MR_VERSION}")
list(GET MR_VERSION_PARTS 0 MR_VERSION_MAJOR)
list(GET MR_VERSION_PARTS 1 MR_VERSION_MINOR)
# Build identity for the version label and the log. -DMR_GIT=<hash> overrides it
# (scripts/build_remote.sh builds from a copy without .git).
if(NOT MR_GIT)
  execute_process(COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse --short HEAD
    OUTPUT_VARIABLE MR_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT MR_GIT)
  set(MR_GIT "nogit")
endif()
# Matte Refine ML tools (ofx/ml, decision 27): off by default, a plain build needs no ML runtime.
option(MR_BUILD_ML "Build the Matte Refine ML tools" OFF)
set(OFX_DIR ${CMAKE_SOURCE_DIR}/third_party/openfx)
set(BUNDLE_NAME MatteRefine)
set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/${BUNDLE_NAME}.ofx.bundle)
find_package(Threads REQUIRED)

# OpenFX 1.4 Support library (vendored, BSD). It still uses std::auto_ptr.
add_library(ofx_support STATIC
  ${OFX_DIR}/Support/Library/ofxsCore.cpp
  ${OFX_DIR}/Support/Library/ofxsImageEffect.cpp
  ${OFX_DIR}/Support/Library/ofxsInteract.cpp
  ${OFX_DIR}/Support/Library/ofxsLog.cpp
  ${OFX_DIR}/Support/Library/ofxsMultiThread.cpp
  ${OFX_DIR}/Support/Library/ofxsParams.cpp
  ${OFX_DIR}/Support/Library/ofxsProperty.cpp
  ${OFX_DIR}/Support/Library/ofxsPropertyValidation.cpp)
target_include_directories(ofx_support PUBLIC ${OFX_DIR}/include ${OFX_DIR}/Support/include)
target_compile_options(ofx_support PRIVATE -w -fvisibility=hidden)

# OS-specific sources behind ofx/src/platform.h and the bundle's binary folder (OFX:
# Linux-x86-64; MacOS, as the installed Mac plugins and LTX Alpha Gen use). macOS: decision 13.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MR_PLATFORM_SOURCES ofx/src/platform_linux.cpp)
  set(MR_ARCH_DIR Linux-x86-64)
elseif(APPLE)
  set(MR_PLATFORM_SOURCES ofx/src/platform_macos.cpp)
  set(MR_ARCH_DIR MacOS)
else()
  message(FATAL_ERROR "No platform layer for ${CMAKE_SYSTEM_NAME} yet (ofx/src/platform.h)")
endif()

# The operations: pure logic, shared with the unit tests (decision 3).
add_library(mr_ops STATIC ofx/src/ops/morphology.cpp ofx/src/ops/edge_field.cpp
  ofx/src/ops/fill_holes.cpp ofx/src/ops/solid_core.cpp ofx/src/ops/snap_to_image.cpp
  ofx/src/ops/foreground_colour.cpp ofx/src/ops/clean_background.cpp
  ofx/src/ops/contour_smooth.cpp ofx/src/ops/analyze_frame.cpp ofx/src/ops/shrink_grow.cpp ofx/src/ops/edge_width.cpp)
target_include_directories(mr_ops PUBLIC ofx/src)
target_compile_options(mr_ops PRIVATE -Wall -Wextra -fvisibility=hidden)
target_link_libraries(mr_ops PUBLIC Threads::Threads)

add_library(${BUNDLE_NAME} MODULE
  ofx/src/plugin.cpp
  ofx/src/log.cpp
  ofx/src/user_config.cpp
  ofx/src/process.cpp
  ofx/src/env_info.cpp
  ofx/src/ocio_encoding.cpp
  ${MR_PLATFORM_SOURCES})
target_link_libraries(${BUNDLE_NAME} PRIVATE ofx_support mr_ops)
# ofxsSupportPrivate.h: retrieveImageEffectPointer for the colour entry point (decision 23).
target_include_directories(${BUNDLE_NAME} PRIVATE ${OFX_DIR}/Support/Library)
target_compile_definitions(${BUNDLE_NAME} PRIVATE MR_VERSION="${MR_VERSION}" MR_GIT="${MR_GIT}"
  MR_VERSION_MAJOR=${MR_VERSION_MAJOR} MR_VERSION_MINOR=${MR_VERSION_MINOR})
target_compile_options(${BUNDLE_NAME} PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
# Export only the two OFX entry points. libstdc++ is linked dynamically: Flame uses the system
# /lib64/libstdc++.so.6 (GCC 11), so building with Rocky 9's GCC 11 matches the host exactly.
# macOS: the vendored osxSymbols list; ld's default for undefined symbols is already "error".
if(APPLE)
  target_link_options(${BUNDLE_NAME} PRIVATE
    -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
  # Minimal bundle description, as most installed Mac OFX bundles carry one.
  file(WRITE ${BUNDLE_DIR}/Contents/Info.plist
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key><string>${BUNDLE_NAME}.ofx</string>
  <key>CFBundleIdentifier</key><string>com.janklier.matterefine</string>
  <key>CFBundleName</key><string>Matte Refine</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>${MR_VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${MR_VERSION}</string>
</dict>
</plist>
")
else()
  target_link_options(${BUNDLE_NAME} PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(${BUNDLE_NAME} PROPERTIES
  PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

# Synthetic test mattes (CLAUDE.md "Synthetic mattes"): the unit tests use them in memory,
# mr_synth writes them as EXR for evaluation and demos in Flame.
add_library(mr_synth_lib STATIC ofx/tools/synth.cpp)
target_include_directories(mr_synth_lib PUBLIC ofx/tools)
add_executable(mr_synth ofx/tools/mr_synth.cpp)
target_link_libraries(mr_synth PRIVATE mr_synth_lib)

enable_testing()
add_executable(test_fill_holes ofx/tests/test_fill_holes.cpp)
target_link_libraries(test_fill_holes PRIVATE mr_ops mr_synth_lib)
add_test(NAME fill_holes COMMAND test_fill_holes)
add_executable(test_solid_core ofx/tests/test_solid_core.cpp)
target_link_libraries(test_solid_core PRIVATE mr_ops mr_synth_lib)
add_test(NAME solid_core COMMAND test_solid_core)
add_executable(test_edge_field ofx/tests/test_edge_field.cpp)
target_link_libraries(test_edge_field PRIVATE mr_ops mr_synth_lib)
add_test(NAME edge_field COMMAND test_edge_field)
add_executable(test_ocio_encoding ofx/tests/test_ocio_encoding.cpp ofx/src/ocio_encoding.cpp)
target_include_directories(test_ocio_encoding PRIVATE ofx/src)
add_test(NAME ocio_encoding COMMAND test_ocio_encoding)
add_executable(test_synth ofx/tests/test_synth.cpp)
target_link_libraries(test_synth PRIVATE mr_synth_lib)
add_test(NAME synth COMMAND test_synth ${CMAKE_BINARY_DIR})
add_executable(test_snap_to_image ofx/tests/test_snap_to_image.cpp)
target_link_libraries(test_snap_to_image PRIVATE mr_ops mr_synth_lib)
add_test(NAME snap_to_image COMMAND test_snap_to_image)
add_executable(test_foreground_colour ofx/tests/test_foreground_colour.cpp)
target_link_libraries(test_foreground_colour PRIVATE mr_ops mr_synth_lib)
add_test(NAME foreground_colour COMMAND test_foreground_colour)
add_executable(test_clean_background ofx/tests/test_clean_background.cpp)
target_link_libraries(test_clean_background PRIVATE mr_ops mr_synth_lib)
add_test(NAME clean_background COMMAND test_clean_background)
add_executable(test_contour_smooth ofx/tests/test_contour_smooth.cpp)
target_link_libraries(test_contour_smooth PRIVATE mr_ops mr_synth_lib)
add_test(NAME contour_smooth COMMAND test_contour_smooth)
add_executable(test_analyze_frame ofx/tests/test_analyze_frame.cpp)
target_link_libraries(test_analyze_frame PRIVATE mr_ops mr_synth_lib)
add_test(NAME analyze_frame COMMAND test_analyze_frame)
add_executable(test_shrink_grow ofx/tests/test_shrink_grow.cpp)
target_link_libraries(test_shrink_grow PRIVATE mr_ops mr_synth_lib)
add_test(NAME shrink_grow COMMAND test_shrink_grow)
add_executable(test_edge_width ofx/tests/test_edge_width.cpp)
target_link_libraries(test_edge_width PRIVATE mr_ops mr_synth_lib)
add_test(NAME edge_width COMMAND test_edge_width)

add_executable(test_load_bundle ofx/tests/test_load_bundle.cpp)
target_include_directories(test_load_bundle PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle PRIVATE dl)
add_dependencies(test_load_bundle ${BUNDLE_NAME})
add_test(NAME load_bundle
  COMMAND test_load_bundle ${BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/${BUNDLE_NAME}.ofx)

if(MR_BUILD_ML)
  add_subdirectory(ofx/ml)
endif()
